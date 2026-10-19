"""
Experiment 1: isolated disruptions + spectral recovery analysis.

Purpose
-------
Evaluate the frozen NRPC controllers under isolated drift, pulse, and
dynamics-cancellation disruptions, while adding the theory-driven spectral
recovery margin

    gamma(t) = lambda_min(Q^T H(t) Q),

with

    H(t) = c_delta * (K/degree) * L + beta * g(t) * M(t).

The original performance metrics are preserved. The new spectral quantities
are computed post hoc from the same trajectories and masks, so they do not
alter the controller, training, disturbances, seeds, or physical simulation.

Outputs are written to:
    results/experiment1/<run_id>/
        config.json
        raw.csv
        summary.csv
        results.pkl
"""

import argparse
import csv
import pickle
from pathlib import Path

import networkx as nx
import numpy as np
import torch
import torch.nn as nn

from kuramoto.simulate import simulate_trajectory_topology
from kuramoto.perturbations import (
    FreqDriftNoise,
    GaussianPulse,
    CancelDynamics,
)


import json
from scipy.linalg import null_space, eigh


# ============================================================
# 1. ARGUMENTS
# ============================================================

parser = argparse.ArgumentParser()

parser.add_argument(
    "--trials",
    type=int,
    default=100,
)

parser.add_argument(
    "--betas",
    type=float,
    nargs="+",
    default=[0.5, 1.0, 5.0, 10.0, 20.0],
)

parser.add_argument(
    "--base-seed",
    type=int,
    default=71000,
)

parser.add_argument(
    "--tag",
    type=str,
    default="dev",
)

args = parser.parse_args()

N_TRIALS = args.trials
BETAS = args.betas
BASE_SEED = args.base_seed
TAG = args.tag


# ============================================================
# 2. DEVICE / REPRODUCIBILITY
# ============================================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

torch.manual_seed(BASE_SEED)
np.random.seed(BASE_SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(BASE_SEED)


# ============================================================
# 3. SYSTEM CONFIGURATION
# ============================================================

N_COMMUNITIES = 5
COMMUNITY_SIZE = 20
N_AGENTS = (
    N_COMMUNITIES
    *
    COMMUNITY_SIZE
)

# Frozen Experiment 1 protocol
DT = 0.5
T_MAX = 100.0
STEPS = int(T_MAX / DT)

K_COORD = 0.40

# The simulator uses (K_COORD / degree) * interaction.
# The modular graph is degree-4 regular, so the spectral operator must
# use the same normalization to match the evaluated dynamics.
NETWORK_DEGREE = 4.0

# Fixed post-disruption window used for the spectral recovery metric.
# It is identical for every method, avoiding method-dependent windows.
RECOVERY_WINDOW = 10.0

# ------------------------------------------------------------------
# Protocol lock.
# Experiment 1 protocol is intentionally fixed to the values below. Fail loudly if a stale
# copy or an accidental edit changes them.
# ------------------------------------------------------------------
PROTOCOL_ID = "EXP1_ISOLATED_V2_DT05_T100_K04"

assert np.isclose(DT, 0.5), (
    f"Experiment 1 protocol mismatch: DT={DT}, expected 0.5"
)
assert np.isclose(T_MAX, 100.0), (
    f"Experiment 1 protocol mismatch: T_MAX={T_MAX}, expected 100.0"
)
assert np.isclose(K_COORD, 0.4), (
    f"Experiment 1 protocol mismatch: K_COORD={K_COORD}, expected 0.4"
)

K_SELECT = 10
TARGET_MASK = (
    K_SELECT
    /
    N_AGENTS
)

ALPHA = 10.0
U_CLIP = 2.0

ENERGY_SCALE = (
    TARGET_MASK
    *
    U_CLIP ** 2
)

SCENARIOS = [
    "drift",
    "pulse",
    "cancel",
]

METHODS = [
    "nrpc_gl",
    "nrpc_local",
    "full",
    "random",
    "betweenness",
    "uncontrolled",
]

SPARSE_METHODS = {
    "nrpc_gl",
    "nrpc_local",
    "random",
    "betweenness",
}

# Figure 2 uses one representative gain fixed a priori.
THEORY_BETA = 5.0
THEORY_METHODS = {
    "nrpc_gl",
    "nrpc_local",
    "random",
    "betweenness",
}


# ============================================================
# 4. PATHS
# ============================================================

RESULTS_ROOT = Path("results")
RESULTS_ROOT.mkdir(parents=True, exist_ok=True)

# Training checkpoints stay where they already live.
GL_PATH = RESULTS_ROOT / "data_train_v4_modular_global_local.pt"
LOCAL_PATH = RESULTS_ROOT / "data_train_v4_modular_local.pt"

def _token(x):
    return str(x).replace(".", "p")

RUN_ID = (
    f"{TAG}_seed{BASE_SEED}"
    f"_T{_token(T_MAX)}"
    f"_dt{_token(DT)}"
    f"_K{_token(K_COORD)}"
    f"_rec{_token(RECOVERY_WINDOW)}"
)

EXPERIMENT_DIR = RESULTS_ROOT / "experiment1" / RUN_ID
EXPERIMENT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_PKL = EXPERIMENT_DIR / "results.pkl"
OUTPUT_RAW_CSV = EXPERIMENT_DIR / "raw.csv"
OUTPUT_SUMMARY_CSV = EXPERIMENT_DIR / "summary.csv"
OUTPUT_CONFIG_JSON = EXPERIMENT_DIR / "config.json"


# ============================================================
# 5. MODULAR DEGREE-4 TOPOLOGY
# ============================================================

def build_modular_degree4_graph():

    A = np.zeros(
        (
            N_AGENTS,
            N_AGENTS
        ),
        dtype=np.float32
    )

    # --------------------------------------------------------
    # Five degree-4 ring-lattice communities
    # --------------------------------------------------------

    for c in range(
        N_COMMUNITIES
    ):

        base = (
            c
            *
            COMMUNITY_SIZE
        )

        for i in range(
            COMMUNITY_SIZE
        ):

            for offset in [
                1,
                2,
            ]:

                j = (
                    i
                    +
                    offset
                ) % COMMUNITY_SIZE

                u = base + i
                v = base + j

                A[u, v] = 1.0
                A[v, u] = 1.0

    # --------------------------------------------------------
    # Degree-preserving inter-community rewiring
    # --------------------------------------------------------

    bridge_nodes = set()

    for c in range(
        N_COMMUNITIES
    ):

        d = (
            c + 1
        ) % N_COMMUNITIES

        u1 = (
            c
            *
            COMMUNITY_SIZE
            +
            4
        )

        u2 = (
            c
            *
            COMMUNITY_SIZE
            +
            5
        )

        v1 = (
            d
            *
            COMMUNITY_SIZE
            +
            12
        )

        v2 = (
            d
            *
            COMMUNITY_SIZE
            +
            13
        )

        assert A[u1, u2] == 1.0
        assert A[v1, v2] == 1.0

        A[u1, u2] = 0.0
        A[u2, u1] = 0.0

        A[v1, v2] = 0.0
        A[v2, v1] = 0.0

        A[u1, v1] = 1.0
        A[v1, u1] = 1.0

        A[u2, v2] = 1.0
        A[v2, u2] = 1.0

        bridge_nodes.update(
            [
                u1,
                u2,
                v1,
                v2,
            ]
        )

    return (
        A,
        sorted(
            bridge_nodes
        )
    )


A_np, BRIDGE_NODES = (
    build_modular_degree4_graph()
)

A = torch.tensor(
    A_np,
    dtype=torch.float32,
    device=device
)

G = nx.from_numpy_array(
    A_np
)

degrees_np = np.array(
    [
        G.degree(i)
        for i in range(
            N_AGENTS
        )
    ],
    dtype=np.float32
)

assert np.all(
    degrees_np == 4
)

assert nx.is_connected(
    G
)

degree = (
    A.sum(dim=1)
    .clamp(min=1.0)
)

# ------------------------------------------------------------
# Local structural features used by NRPC
# ------------------------------------------------------------

clustering_dict = (
    nx.clustering(
        G
    )
)

clustering_np = np.array(
    [
        clustering_dict[i]
        for i in range(
            N_AGENTS
        )
    ],
    dtype=np.float32
)

clustering = torch.tensor(
    clustering_np,
    dtype=torch.float32,
    device=device
)

neighbor_clustering = (
    A @ clustering
    /
    degree
)

# ------------------------------------------------------------
# Betweenness baseline
# ------------------------------------------------------------

betweenness_dict = (
    nx.betweenness_centrality(
        G,
        normalized=True
    )
)

BETWEENNESS_TOP10 = sorted(
    betweenness_dict,
    key=betweenness_dict.get,
    reverse=True
)[:K_SELECT]

# ------------------------------------------------------------
# lambda_2 diagnostic
# ------------------------------------------------------------

L = (
    np.diag(
        degrees_np
    )
    -
    A_np
)

lambda2 = float(
    np.sort(
        np.linalg.eigvalsh(
            L
        )
    )[1]
)


# ============================================================
# 5B. SPECTRAL RECOVERY OPERATOR
# ============================================================
#
# Theory evaluated in Experiment 1:
#
#   H(t) = c_delta * (K_COORD / NETWORK_DEGREE) * L
#          + beta * g(t) * M(t)
#
#   gamma(t) = lambda_min( Q^T H(t) Q )
#
# where Q is an orthonormal basis of the disagreement space 1^perp.
#
# NOTE:
# The K/degree normalization is intentional: simulate_trajectory_topology
# evolves the physical system using (K_COORD / 4.0) * interaction.
# Matching that normalization is essential for gamma to characterize the
# same dynamics that are actually evaluated.
# ============================================================

DELTA_MAX = np.pi / 2.0
C_DELTA = np.sin(DELTA_MAX) / DELTA_MAX  # 2/pi

Q = null_space(
    np.ones((1, N_AGENTS), dtype=np.float64)
)

NETWORK_COUPLING_SCALE = (
    C_DELTA
    *
    K_COORD
    /
    NETWORK_DEGREE
)

H_NETWORK = (
    NETWORK_COUPLING_SCALE
    *
    L.astype(np.float64)
)

H_NETWORK_RED = (
    Q.T
    @ H_NETWORK
    @ Q
)

GAMMA_NETWORK = (
    NETWORK_COUPLING_SCALE
    *
    lambda2
)

EDGE_I, EDGE_J = np.where(
    np.triu(A_np, k=1) > 0.0
)


# Orthogonal projector onto the disagreement subspace.
P_DIS = (
    np.eye(N_AGENTS, dtype=np.float64)
    -
    np.ones((N_AGENTS, N_AGENTS), dtype=np.float64)
    /
    N_AGENTS
)

# Cache Q^T M Q for masks that recur during evaluation.
_PINNING_RED_CACHE = {}


def reduced_pinning_matrix(mask):
    mask = np.asarray(
        mask,
        dtype=np.float64
    ).reshape(-1)

    key = np.packbits(
        (mask > 0.5).astype(np.uint8)
    ).tobytes()

    cached = _PINNING_RED_CACHE.get(key)

    if cached is not None:
        return cached

    reduced = (
        Q.T
        @ (mask[:, None] * Q)
    )

    _PINNING_RED_CACHE[key] = reduced
    return reduced


def spectral_recovery_margin(mask, r, beta):
    """Compute gamma(t) for one controller decision state."""

    g = np.tanh(
        ALPHA
        *
        (1.0 - float(r))
    )

    H_red = (
        H_NETWORK_RED
        +
        float(beta)
        *
        g
        *
        reduced_pinning_matrix(mask)
    )

    gamma = eigh(
        H_red,
        subset_by_index=[0, 0],
        eigvals_only=True,
        check_finite=False,
    )[0]

    return float(gamma), float(g)


# ============================================================
# 6. V4 FEATURES
# ============================================================

def build_node_features_v4(
    theta,
    mode,
):

    if theta.dim() == 1:

        theta = theta.unsqueeze(
            0
        )

    B, N = theta.shape

    # --------------------------------------------------------
    # Global information
    # --------------------------------------------------------

    zbar = torch.mean(
        torch.exp(
            1j * theta
        ),
        dim=1,
        keepdim=True
    )

    psi = torch.angle(
        zbar
    )

    r = torch.abs(
        zbar
    ).real

    global_sin = torch.sin(
        theta - psi
    )

    global_cos = torch.cos(
        theta - psi
    )

    global_r = r.expand(
        -1,
        N
    )

    # --------------------------------------------------------
    # Local first and second circular moments
    # --------------------------------------------------------

    theta_i = theta.unsqueeze(
        2
    )

    theta_j = theta.unsqueeze(
        1
    )

    delta = (
        theta_j
        -
        theta_i
    )

    A_batch = A.unsqueeze(
        0
    )

    deg = degree.view(
        1,
        N
    )

    local_sin_1 = (
        (
            torch.sin(
                delta
            )
            *
            A_batch
        ).sum(dim=2)
        /
        deg
    )

    local_cos_1 = (
        (
            torch.cos(
                delta
            )
            *
            A_batch
        ).sum(dim=2)
        /
        deg
    )

    local_r = torch.sqrt(
        local_sin_1 ** 2
        +
        local_cos_1 ** 2
        +
        1e-12
    )

    local_sin_2 = (
        (
            torch.sin(
                2.0 * delta
            )
            *
            A_batch
        ).sum(dim=2)
        /
        deg
    )

    local_cos_2 = (
        (
            torch.cos(
                2.0 * delta
            )
            *
            A_batch
        ).sum(dim=2)
        /
        deg
    )

    # --------------------------------------------------------
    # Local structural descriptors
    # --------------------------------------------------------

    c_i = (
        clustering
        .view(
            1,
            N
        )
        .expand(
            B,
            -1
        )
    )

    c_neigh = (
        neighbor_clustering
        .view(
            1,
            N
        )
        .expand(
            B,
            -1
        )
    )

    if mode == "local":

        global_sin = (
            torch.zeros_like(
                global_sin
            )
        )

        global_cos = (
            torch.zeros_like(
                global_cos
            )
        )

        global_r = (
            torch.zeros_like(
                global_r
            )
        )

    elif mode != "global_local":

        raise ValueError(
            mode
        )

    return torch.stack(
        [
            global_sin,
            global_cos,
            global_r,

            local_sin_1,
            local_cos_1,
            local_r,

            local_sin_2,
            local_cos_2,

            c_i,
            c_neigh,
        ],
        dim=-1
    )


# ============================================================
# 7. NETWORK
# ============================================================

class NodeScoreNetwork(
    nn.Module
):

    def __init__(
        self,
        in_features=10,
        hidden_dim=64,
    ):

        super().__init__()

        self.net = nn.Sequential(

            nn.Linear(
                in_features,
                hidden_dim
            ),

            nn.Tanh(),

            nn.Linear(
                hidden_dim,
                hidden_dim
            ),

            nn.Tanh(),

            nn.Linear(
                hidden_dim,
                hidden_dim
            ),

            nn.Tanh(),

            nn.Linear(
                hidden_dim,
                1
            ),
        )

        self.output_scale = (
            nn.Parameter(
                torch.tensor(
                    1.0
                )
            )
        )

    def forward(
        self,
        features,
    ):

        B, N, F = (
            features.shape
        )

        x = features.reshape(
            B * N,
            F
        )

        return (
            self.net(
                x
            )
            .reshape(
                B,
                N
            )
            *
            self.output_scale
        )


# ============================================================
# 8. SHARED CONSENSUS FEEDBACK
# ============================================================

def full_feedback(
    theta,
    beta,
):

    zbar = torch.mean(
        torch.exp(
            1j * theta
        ),
        dim=1,
        keepdim=True
    )

    r = torch.abs(
        zbar
    ).real

    psi = torch.angle(
        zbar
    )

    u = (
        -beta
        *
        torch.sin(
            theta - psi
        )
    )

    gate = torch.tanh(
        ALPHA
        *
        (
            1.0 - r
        )
    )

    u = (
        u
        *
        gate
    )

    return torch.clamp(
        u,
        -U_CLIP,
        U_CLIP
    )


# ============================================================
# 9. CONTROLLERS
# ============================================================

class NRPCV4Controller:

    def __init__(
        self,
        model,
        mode,
        beta,
    ):

        self.model = model
        self.mode = mode
        self.beta = beta

    def __call__(
        self,
        theta,
    ):

        if theta.dim() == 1:

            theta = theta.view(
                1,
                -1
            )

        features = (
            build_node_features_v4(
                theta,
                self.mode
            )
        )

        logits = self.model(
            features
        )

        idx = torch.topk(
            logits,
            k=K_SELECT,
            dim=1
        ).indices

        m = torch.zeros_like(
            logits
        )

        m.scatter_(
            1,
            idx,
            1.0
        )

        u = (
            m
            *
            full_feedback(
                theta,
                self.beta
            )
        )

        return u, m


class FullConsensus:

    def __init__(
        self,
        beta,
    ):

        self.beta = beta

    def __call__(
        self,
        theta,
    ):

        u = full_feedback(
            theta,
            self.beta
        )

        return (
            u,
            torch.ones_like(
                theta
            )
        )


class NoControl:

    def __call__(
        self,
        theta,
    ):

        return (
            torch.zeros_like(
                theta
            ),
            torch.zeros_like(
                theta
            ),
        )


class FixedMaskConsensus:

    def __init__(
        self,
        beta,
        selected_indices,
    ):

        self.beta = beta

        self.selected_indices = (
            list(
                selected_indices
            )
        )

    def __call__(
        self,
        theta,
    ):

        m = torch.zeros_like(
            theta
        )

        m[
            :,
            self.selected_indices
        ] = 1.0

        u = (
            m
            *
            full_feedback(
                theta,
                self.beta
            )
        )

        return u, m


class PhaseErrorTopK:

    def __init__(
        self,
        beta,
    ):

        self.beta = beta

    def __call__(
        self,
        theta,
    ):

        zbar = torch.mean(
            torch.exp(
                1j * theta
            ),
            dim=1,
            keepdim=True
        )

        psi = torch.angle(
            zbar
        )

        scores = torch.abs(
            torch.sin(
                theta - psi
            )
        )

        idx = torch.topk(
            scores,
            k=K_SELECT,
            dim=1
        ).indices

        m = torch.zeros_like(
            theta
        )

        m.scatter_(
            1,
            idx,
            1.0
        )

        u = (
            m
            *
            full_feedback(
                theta,
                self.beta
            )
        )

        return u, m


# ============================================================
# 10. LOAD CHECKPOINTS
# ============================================================

if not GL_PATH.exists():

    raise FileNotFoundError(
        GL_PATH
    )

if not LOCAL_PATH.exists():

    raise FileNotFoundError(
        LOCAL_PATH
    )


ckpt_gl = torch.load(
    GL_PATH,
    map_location=device,
    weights_only=False
)

ckpt_local = torch.load(
    LOCAL_PATH,
    map_location=device,
    weights_only=False
)


# ------------------------------------------------------------
# Sanity checks against current experiment
# ------------------------------------------------------------

for name, ckpt in [
    ("GL", ckpt_gl),
    ("Local", ckpt_local),
]:

    assert (
        ckpt["n_agents"]
        ==
        N_AGENTS
    )

    assert (
        ckpt["k_select"]
        ==
        K_SELECT
    )

    assert (
        ckpt["in_features"]
        ==
        10
    )

    assert np.allclose(
        ckpt["adjacency"],
        A_np
    ), (
        f"{name} checkpoint "
        "uses a different topology."
    )


model_gl = NodeScoreNetwork(
    in_features=10,
    hidden_dim=64
).to(device)

model_gl.load_state_dict(
    ckpt_gl[
        "state_dict"
    ]
)

model_gl.eval()


model_local = NodeScoreNetwork(
    in_features=10,
    hidden_dim=64
).to(device)

model_local.load_state_dict(
    ckpt_local[
        "state_dict"
    ]
)

model_local.eval()


# ============================================================
# 11. HELPERS
# ============================================================

def community_nodes(
    community,
):

    start = (
        community
        *
        COMMUNITY_SIZE
    )

    return list(
        range(
            start,
            start
            +
            COMMUNITY_SIZE
        )
    )


def build_initial_state(
    seed,
):

    rng = np.random.default_rng(
        seed
    )

    common_phase = rng.uniform(
        0.0,
        2.0 * np.pi
    )

    theta0 = torch.tensor(
        common_phase
        +
        rng.normal(
            0.0,
            0.12,
            size=N_AGENTS
        ),
        dtype=torch.float32,
        device=device
    )

    omegas = torch.tensor(
        rng.normal(
            1.0,
            0.05,
            size=N_AGENTS
        ),
        dtype=torch.float32,
        device=device
    )

    return (
        theta0,
        omegas
    )


def build_perturbation(
    scenario,
    community,
):

    sources = community_nodes(
        community
    )

    if scenario == "drift":

        return FreqDriftNoise(
            t0=25.0,
            tf=35.0,
            source_idxs=sources,
            drift=1.6,
            sigma=1.0,
        )

    if scenario == "pulse":

        return GaussianPulse(
            t0=25.0,
            tf=31.0,
            source_idxs=sources,
            amp=20.0,
            width=2.0,
        )

    if scenario == "cancel":

        return CancelDynamics(
            t0=25.0,
            tf=35.0,
            source_idxs=sources,
        )

    raise ValueError(
        scenario
    )


# ============================================================
# 12. RESILIENCE
# ============================================================

def resilience_metric(
    r,
    t0,
    tf,
):

    impact_idx = int(
        t0 / DT
    )

    recovery_idx = int(
        tf / DT
    )

    r_during = r[
        impact_idx:
        recovery_idx
    ]

    r_post = r[
        recovery_idx:
    ]

    absorption = (
        0.5
        *
        torch.min(
            r_during
        )
        +
        0.5
        *
        torch.mean(
            r_during
        )
    )

    recovery_error = (
        torch.mean(
            (
                1.0
                -
                r_post
            ) ** 2
        )
    )

    return (
        absorption
        *
        (
            1.0
            -
            recovery_error
        )
    )


# ============================================================
# 13. HISTORY HELPERS
# ============================================================

def stack_history(
    history,
):

    return torch.stack(
        [
            x.detach().reshape(
                -1
            )
            for x in history
        ],
        dim=0
    )


def phase_error_mask_from_states(
    theta_states,
):

    zbar = torch.mean(
        torch.exp(
            1j
            *
            theta_states
        ),
        dim=1,
        keepdim=True
    )

    psi = torch.angle(
        zbar
    )

    scores = torch.abs(
        torch.sin(
            theta_states
            -
            psi
        )
    )

    idx = torch.topk(
        scores,
        k=K_SELECT,
        dim=1
    ).indices

    m = torch.zeros_like(
        theta_states
    )

    m.scatter_(
        1,
        idx,
        1.0
    )

    return m


def window_mean(
    values,
    mask,
):

    if not np.any(
        mask
    ):

        return np.nan

    return float(
        values[
            mask
        ].mean()
    )



def spectral_recovery_metrics(
    m_hist,
    theta_hist,
    theta0,
    omegas,
    u_hist,
    beta,
    perturbation,
    collect_theory_samples=False,
):
    """
    Compute theory-driven recovery metrics over the fixed window

        (tf, tf + RECOVERY_WINDOW].

    Time alignment is exact for the Euler simulator:
      - theta_pre[k] is the state observed by the controller at t_k,
      - m[k] is selected from theta_pre[k],
      - theta_post[k] is the state after one Euler step,
      - gamma(t_k) is evaluated from theta_pre[k] and m[k],
      - Vdot_fd[k] = (V_post[k] - V_pre[k]) / DT.

    For the local Lyapunov mechanism, define

        lambda_rest(t)
          = -( Vdot(t) - e(t)^T P omega ) / ||e(t)||^2,

    after the external perturbation has ended.

    Instantaneous samples are only collected for THEORY_BETA and are stored
    inside results.pkl (not in a separate CSV) to avoid proliferating files.
    """

    m = (
        stack_history(m_hist)
        .detach()
        .cpu()
        .numpy()
        .astype(np.float64)
    )

    u = (
        stack_history(u_hist)
        .detach()
        .cpu()
        .numpy()
        .astype(np.float64)
    )

    theta_post_t = stack_history(
        theta_hist
    )

    theta_pre_t = torch.cat(
        [
            theta0.reshape(1, -1),
            theta_post_t[:-1],
        ],
        dim=0
    )

    theta_pre = (
        theta_pre_t
        .detach()
        .cpu()
        .numpy()
        .astype(np.float64)
    )

    theta_post = (
        theta_post_t
        .detach()
        .cpu()
        .numpy()
        .astype(np.float64)
    )

    omega_np = (
        omegas.detach()
        .reshape(-1)
        .cpu()
        .numpy()
        .astype(np.float64)
    )

    assert theta_pre.shape == m.shape
    assert theta_post.shape == m.shape
    assert u.shape == m.shape

    omega_proj = (
        P_DIS
        @ omega_np
    )

    t_axis = (
        np.arange(STEPS)
        *
        DT
    )

    recovery = (
        (t_axis > perturbation.tf)
        &
        (
            t_axis
            <=
            perturbation.tf
            +
            RECOVERY_WINDOW
        )
    )

    idx = np.flatnonzero(
        recovery
    )

    empty_metrics = {
        "gamma_rec_mean": np.nan,
        "gamma_rec_min": np.nan,
        "gamma_rec_integral": np.nan,
        "gamma_coh_mean": np.nan,
        "gamma_directional_coh_mean": np.nan,
        "lambda_rest_rhs_coh_mean": np.nan,
        "lambda_rest_coh_mean": np.nan,
        "theory_gap_directional_coh_mean": np.nan,
        "theory_gap_coh_mean": np.nan,
        "cohesive_fraction_rec": np.nan,
        "max_edge_phase_diff_rec": np.nan,
        "clip_fraction_rec": np.nan,
        "gate_rec_mean": np.nan,
    }

    if idx.size == 0:
        return empty_metrics, []

    gamma_vals = []
    gate_vals = []
    cohesive_vals = []
    max_edge_diffs = []
    lambda_rest_vals = []
    lambda_rest_rhs_vals = []
    gamma_directional_vals = []
    gamma_network_directional_vals = []
    gamma_pinning_directional_vals = []
    theory_gap_directional_vals = []
    theory_gap_vals = []

    n_active = 0
    n_clipped = 0

    theory_samples = []

    eps = 1e-12

    for k in idx:

        theta_k = theta_pre[k]
        theta_next = theta_post[k]
        mask_k = m[k]

        # ----------------------------------------------------
        # Circular mean / gate / spectral margin
        # ----------------------------------------------------
        zbar = np.mean(
            np.exp(
                1j
                *
                theta_k
            )
        )

        r_k = float(
            np.abs(
                zbar
            )
        )

        psi_k = float(
            np.angle(
                zbar
            )
        )

        gamma_k, gate_k = (
            spectral_recovery_margin(
                mask=mask_k,
                r=r_k,
                beta=beta,
            )
        )

        # ----------------------------------------------------
        # Cohesiveness condition on graph edges
        # ----------------------------------------------------
        edge_diff = (
            theta_k[EDGE_I]
            -
            theta_k[EDGE_J]
        )

        edge_diff = np.angle(
            np.exp(
                1j
                *
                edge_diff
            )
        )

        max_diff = float(
            np.max(
                np.abs(
                    edge_diff
                )
            )
        )

        cohesive_k = bool(
            max_diff
            <=
            DELTA_MAX
        )

        # ----------------------------------------------------
        # Disagreement state and Lyapunov decrement
        # ----------------------------------------------------
        zbar = np.mean(np.exp(1j * theta_k))
        psi = np.angle(zbar)

        e_circ = np.angle(
            np.exp(1j * (theta_k - psi))
        )

        e_k = P_DIS @ e_circ

        zbar_next = np.mean(np.exp(1j * theta_next))
        psi_next = np.angle(zbar_next)

        e_circ_next = np.angle(
            np.exp(1j * (theta_next - psi_next))
        )

        e_next = P_DIS @ e_circ_next

        e_norm2 = float(
            e_k
            @
            e_k
        )

        V_k = 0.5 * e_norm2

        V_next = 0.5 * float(
            e_next
            @
            e_next
        )

        Vdot_fd = (
            V_next
            -
            V_k
        ) / DT

        forcing_omega = float(
            e_k
            @
            omega_proj
        )

        lambda_rest_k = (
            -(
                Vdot_fd
                -
                forcing_omega
            )
            /
            max(
                e_norm2,
                eps
            )
        )

        theory_gap_k = (
            lambda_rest_k
            -
            gamma_k
        )

        # ----------------------------------------------------
        # Directional theoretical margin.
        #
        # gamma(t) is the worst-case margin over all disagreement
        # directions. The quantity below evaluates the SAME theoretical
        # quadratic form along the disagreement direction actually visited
        # by the trajectory:
        #
        #   gamma_e(t) = e^T H(t) e / ||e||^2.
        #
        # By the Rayleigh quotient definition, gamma_e(t) >= gamma(t)
        # whenever e != 0.
        # ----------------------------------------------------
        network_quad = float(
            e_k
            @
            (
                H_NETWORK
                @
                e_k
            )
        )

        pinning_quad = float(
            float(beta)
            *
            gate_k
            *
            np.sum(
                mask_k
                *
                e_k ** 2
            )
        )

        gamma_network_directional_k = (
            network_quad
            /
            max(
                e_norm2,
                eps
            )
        )

        gamma_pinning_directional_k = (
            pinning_quad
            /
            max(
                e_norm2,
                eps
            )
        )

        gamma_directional_k = (
            gamma_network_directional_k
            +
            gamma_pinning_directional_k
        )

        # ----------------------------------------------------
        # Exact continuous-time restoring rate evaluated from the SAME
        # pre-integration state used by the simulator.
        #
        # This avoids using the finite-difference approximation of Vdot
        # when validating the continuous-time Lyapunov inequality.
        #
        # After tf the external perturbation is inactive, so
        #
        #   theta_dot = omega + coupling + u.
        #
        # Since e^T P = e^T, subtracting the omega contribution leaves
        # the restoring contribution from coupling + control:
        #
        #   lambda_rest_rhs
        #       = - e^T (coupling + u) / ||e||^2.
        # ----------------------------------------------------
        delta_theta = (
            theta_k[None, :]
            -
            theta_k[:, None]
        )

        interaction_k = np.sum(
            np.sin(
                delta_theta
            )
            *
            A_np,
            axis=1,
        )

        coupling_k = (
            K_COORD
            /
            NETWORK_DEGREE
        ) * interaction_k

        u_k = u[k]

        lambda_rest_rhs_k = (
            -float(
                e_k
                @
                (
                    coupling_k
                    +
                    u_k
                )
            )
            /
            max(
                e_norm2,
                eps
            )
        )

        theory_gap_directional_k = (
            lambda_rest_rhs_k
            -
            gamma_directional_k
        )

        # ----------------------------------------------------
        # Saturation diagnostic
        # ----------------------------------------------------
        active = (
            mask_k
            >
            0.5
        )

        nominal_control = (
            float(beta)
            *
            gate_k
            *
            np.abs(
                np.sin(
                    theta_k
                    -
                    psi_k
                )
            )
        )

        clipped_nodes = (
            active
            &
            (
                nominal_control
                >=
                U_CLIP
                -
                1e-12
            )
        )

        any_clip_k = bool(
            np.any(
                clipped_nodes
            )
        )

        if np.any(active):
            n_active += int(
                np.sum(
                    active
                )
            )

            n_clipped += int(
                np.sum(
                    clipped_nodes
                )
            )

        gamma_vals.append(gamma_k)
        gate_vals.append(gate_k)
        cohesive_vals.append(cohesive_k)
        max_edge_diffs.append(max_diff)
        lambda_rest_vals.append(lambda_rest_k)
        lambda_rest_rhs_vals.append(lambda_rest_rhs_k)
        gamma_directional_vals.append(gamma_directional_k)
        gamma_network_directional_vals.append(gamma_network_directional_k)
        gamma_pinning_directional_vals.append(gamma_pinning_directional_k)
        theory_gap_directional_vals.append(theory_gap_directional_k)
        theory_gap_vals.append(theory_gap_k)

        if collect_theory_samples:
            theory_samples.append(
                {
                    "t": float(t_axis[k]),
                    "gamma_t": float(gamma_k),
                    "gamma_directional": float(gamma_directional_k),
                    "gamma_network_directional": float(gamma_network_directional_k),
                    "gamma_pinning_directional": float(gamma_pinning_directional_k),
                    "lambda_rest_rhs": float(lambda_rest_rhs_k),
                    "lambda_rest": float(lambda_rest_k),
                    "theory_gap_directional": float(theory_gap_directional_k),
                    "theory_gap": float(theory_gap_k),
                    "V": float(V_k),
                    "Vdot_fd": float(Vdot_fd),
                    "e_norm2": float(e_norm2),
                    "forcing_omega": float(forcing_omega),
                    "gate": float(gate_k),
                    "r": float(r_k),
                    "max_edge_phase_diff": float(max_diff),
                    "cohesive": int(cohesive_k),
                    "any_clip": int(any_clip_k),
                }
            )

    gamma_vals = np.asarray(
        gamma_vals,
        dtype=float
    )

    gate_vals = np.asarray(
        gate_vals,
        dtype=float
    )

    cohesive_vals = np.asarray(
        cohesive_vals,
        dtype=bool
    )

    lambda_rest_vals = np.asarray(
        lambda_rest_vals,
        dtype=float
    )

    lambda_rest_rhs_vals = np.asarray(
        lambda_rest_rhs_vals,
        dtype=float
    )

    gamma_directional_vals = np.asarray(
        gamma_directional_vals,
        dtype=float
    )

    gamma_network_directional_vals = np.asarray(
        gamma_network_directional_vals,
        dtype=float
    )

    gamma_pinning_directional_vals = np.asarray(
        gamma_pinning_directional_vals,
        dtype=float
    )

    theory_gap_directional_vals = np.asarray(
        theory_gap_directional_vals,
        dtype=float
    )

    theory_gap_vals = np.asarray(
        theory_gap_vals,
        dtype=float
    )

    # Theory-supported aggregates: only where the phase-cohesion
    # assumption is satisfied.
    if np.any(cohesive_vals):
        gamma_coh_mean = float(
            np.mean(
                gamma_vals[
                    cohesive_vals
                ]
            )
        )

        gamma_directional_coh_mean = float(
            np.mean(
                gamma_directional_vals[
                    cohesive_vals
                ]
            )
        )

        lambda_rest_rhs_coh_mean = float(
            np.mean(
                lambda_rest_rhs_vals[
                    cohesive_vals
                ]
            )
        )

        lambda_rest_coh_mean = float(
            np.mean(
                lambda_rest_vals[
                    cohesive_vals
                ]
            )
        )

        theory_gap_directional_coh_mean = float(
            np.mean(
                theory_gap_directional_vals[
                    cohesive_vals
                ]
            )
        )

        theory_gap_coh_mean = float(
            np.mean(
                theory_gap_vals[
                    cohesive_vals
                ]
            )
        )
    else:
        gamma_coh_mean = np.nan
        gamma_directional_coh_mean = np.nan
        lambda_rest_rhs_coh_mean = np.nan
        lambda_rest_coh_mean = np.nan
        theory_gap_directional_coh_mean = np.nan
        theory_gap_coh_mean = np.nan

    metrics = {
        # Existing spectral metrics retained for backward compatibility.
        "gamma_rec_mean":
            float(
                np.mean(
                    gamma_vals
                )
            ),

        "gamma_rec_min":
            float(
                np.min(
                    gamma_vals
                )
            ),

        "gamma_rec_integral":
            float(
                np.sum(
                    gamma_vals
                )
                *
                DT
            ),

        "gate_rec_mean":
            float(
                np.mean(
                    gate_vals
                )
            ),

        "cohesive_fraction_rec":
            float(
                np.mean(
                    cohesive_vals
                )
            ),

        "max_edge_phase_diff_rec":
            float(
                np.max(
                    max_edge_diffs
                )
            ),

        "clip_fraction_rec":
            (
                float(
                    n_clipped
                    /
                    n_active
                )
                if n_active > 0
                else 0.0
            ),

        # New theory-aligned aggregate metrics.
        "gamma_coh_mean":
            gamma_coh_mean,

        "gamma_directional_coh_mean":
            gamma_directional_coh_mean,

        "lambda_rest_rhs_coh_mean":
            lambda_rest_rhs_coh_mean,

        "lambda_rest_coh_mean":
            lambda_rest_coh_mean,

        "theory_gap_directional_coh_mean":
            theory_gap_directional_coh_mean,

        "theory_gap_coh_mean":
            theory_gap_coh_mean,
    }

    return metrics, theory_samples


# ============================================================
# 14. SELECTION ANALYSIS
# ============================================================

def selection_metrics(
    m_hist,
    theta_hist,
    theta0,
    perturbation,
    disturbed_nodes,
):

    m = stack_history(
        m_hist
    )

    theta_post = stack_history(
        theta_hist
    )

    # --------------------------------------------------------
    # m[t] was chosen from the state BEFORE Euler integration.
    #
    # theta_hist[t] stores the state AFTER integration.
    #
    # Reconstruct the actual controller-observation states.
    # --------------------------------------------------------

    theta_pre = torch.cat(
        [
            theta0.reshape(
                1,
                -1
            ),
            theta_post[:-1],
        ],
        dim=0
    )

    assert (
        theta_pre.shape
        ==
        m.shape
    )

    phase_oracle = (
        phase_error_mask_from_states(
            theta_pre
        )
    )

    bridge_vec = torch.zeros(
        N_AGENTS,
        device=device
    )

    bridge_vec[
        BRIDGE_NODES
    ] = 1.0

    disturbed_vec = torch.zeros(
        N_AGENTS,
        device=device
    )

    disturbed_vec[
        disturbed_nodes
    ] = 1.0

    # Fraction of the 10-node budget assigned to each set.
    bridge_rate = (
        (
            m
            *
            bridge_vec.view(
                1,
                -1
            )
        ).sum(dim=1)
        /
        K_SELECT
    )

    disturbed_rate = (
        (
            m
            *
            disturbed_vec.view(
                1,
                -1
            )
        ).sum(dim=1)
        /
        K_SELECT
    )

    # --------------------------------------------------------
    # Similarity to Phase-error ON THE SAME STATE.
    #
    # 1.0 = exactly same Top-10
    # 0.0 = disjoint masks
    # --------------------------------------------------------

    phase_overlap = (
        (
            m
            *
            phase_oracle
        ).sum(dim=1)
        /
        K_SELECT
    )

    # --------------------------------------------------------
    # Fraction of selected nodes replaced between two
    # consecutive decisions.
    #
    # 0 = same mask
    # 1 = completely new Top-10
    # --------------------------------------------------------

    switch_fraction = (
        0.5
        *
        torch.sum(
            torch.abs(
                m[1:]
                -
                m[:-1]
            ),
            dim=1
        )
        /
        K_SELECT
    )

    t_axis = (
        np.arange(
            STEPS
        )
        *
        DT
    )

    during = (
        (t_axis >= perturbation.t0)
        &
        (t_axis <= perturbation.tf)
    )

    post = (
        t_axis
        >
        perturbation.tf
    )

    # switch_fraction corresponds to transitions ending at t[1:]
    t_switch = t_axis[1:]

    during_switch = (
        (t_switch >= perturbation.t0)
        &
        (t_switch <= perturbation.tf)
    )

    bridge_np = (
        bridge_rate
        .detach()
        .cpu()
        .numpy()
    )

    disturbed_np = (
        disturbed_rate
        .detach()
        .cpu()
        .numpy()
    )

    phase_np = (
        phase_overlap
        .detach()
        .cpu()
        .numpy()
    )

    switch_np = (
        switch_fraction
        .detach()
        .cpu()
        .numpy()
    )

    return {

        "bridge_during":
            window_mean(
                bridge_np,
                during
            ),

        "bridge_post":
            window_mean(
                bridge_np,
                post
            ),

        "disturbed_during":
            window_mean(
                disturbed_np,
                during
            ),

        "disturbed_post":
            window_mean(
                disturbed_np,
                post
            ),

        "phase_overlap_during":
            window_mean(
                phase_np,
                during
            ),

        "phase_overlap_post":
            window_mean(
                phase_np,
                post
            ),

        "switch_fraction":
            float(
                np.mean(
                    switch_np
                )
            ),

        "switch_during":
            window_mean(
                switch_np,
                during_switch
            ),
    }


# ============================================================
# 15. SINGLE TRAJECTORY EVALUATION
# ============================================================

@torch.no_grad()
def evaluate_controller(
    controller,
    theta0,
    omegas,
    perturbation,
    noise_seed,
    sparse,
    beta,
):

    # --------------------------------------------------------
    # Critical:
    # every method receives the SAME stochastic disturbance
    # realization.
    # --------------------------------------------------------

    torch.manual_seed(
        noise_seed
    )

    np.random.seed(
        noise_seed
    )

    if torch.cuda.is_available():

        torch.cuda.manual_seed_all(
            noise_seed
        )

    (
        r_hist,
        u_hist,
        theta_hist,
        m_hist,
    ) = simulate_trajectory_topology(

        controller,

        theta0,
        omegas,

        [
            perturbation
        ],

        False,

        A,

        STEPS,
        DT,

        10,
        10,

        N_AGENTS,

        K_COORD,
    )

    r = stack_history(
        r_hist
    ).reshape(
        -1
    )

    u = stack_history(
        u_hist
    )

    resilience = (
        resilience_metric(
            r,
            perturbation.t0,
            perturbation.tf
        )
    )

    energy = torch.mean(
        u ** 2
    )

    out = {

        "resilience":
            float(
                resilience.item()
            ),

        "energy":
            float(
                energy.item()
            ),

        "energy_normalized":
            float(
                (
                    energy
                    /
                    ENERGY_SCALE
                ).item()
            ),

        "r_mean":
            float(
                r.mean().item()
            ),

        "r_min":
            float(
                r.min().item()
            ),

        "r_final":
            float(
                r[-1].item()
            ),
    }


    # --------------------------------------------------------
    # Theory-driven spectral recovery metrics.
    # Computed for EVERY method (sparse, full, and uncontrolled).
    # --------------------------------------------------------
    spectral, theory_samples = spectral_recovery_metrics(
        m_hist=m_hist,
        theta_hist=theta_hist,
        theta0=theta0,
        omegas=omegas,
        u_hist=u_hist,
        beta=beta,
        perturbation=perturbation,
        collect_theory_samples=(
            np.isclose(beta, THEORY_BETA)
        ),
    )

    out.update(
        spectral
    )

    if sparse:

        sel = selection_metrics(
            m_hist,
            theta_hist,
            theta0,
            perturbation,
            perturbation.source_idxs
        )

        out.update(
            sel
        )

    else:

        for key in [

            "bridge_during",
            "bridge_post",

            "disturbed_during",
            "disturbed_post",

            "phase_overlap_during",
            "phase_overlap_post",

            "switch_fraction",
            "switch_during",
        ]:

            out[key] = np.nan

    return out, theory_samples


# ============================================================
# 16. PRINT SETUP
# ============================================================

print("=" * 82)
print("EXPERIMENT 1 | THEORY VALIDATION + DIRECTIONAL LYAPUNOV MARGIN")
print("=" * 82)

print(
    f"Protocol                  : "
    f"{PROTOCOL_ID}"
)

print(
    f"Device                    : "
    f"{device}"
)

print(
    f"N                         : "
    f"{N_AGENTS}"
)

print(
    f"Communities               : "
    f"{N_COMMUNITIES} x "
    f"{COMMUNITY_SIZE}"
)

print(
    f"Degree                    : "
    f"4"
)

print(
    f"lambda_2                  : "
    f"{lambda2:.6f}"
)

print(
    f"K                         : "
    f"{K_COORD}"
)

print(
    f"Sparse budget             : "
    f"{K_SELECT}/{N_AGENTS}"
)

print(
    f"Betas                     : "
    f"{BETAS}"
)

print(
    f"Trials/scenario/beta      : "
    f"{N_TRIALS}"
)

print(
    f"DT / T_MAX                : "
    f"{DT} / {T_MAX}"
)

print(
    f"Recovery window           : "
    f"({RECOVERY_WINDOW:g} s after tf)"
)

print(
    f"c_delta                    : "
    f"{C_DELTA:.6f}"
)

print(
    f"Network gamma baseline    : "
    f"{GAMMA_NETWORK:.8f}"
)

print(
    f"Output directory          : "
    f"{EXPERIMENT_DIR}"
)

print(
    f"GL best update            : "
    f"{ckpt_gl['best_update']}"
)

print(
    f"Local best update         : "
    f"{ckpt_local['best_update']}"
)

print(
    f"Bridge nodes              : "
    f"{BRIDGE_NODES}"
)

print(
    f"Betweenness Top-10        : "
    f"{BETWEENNESS_TOP10}"
)

print("=" * 82)


# ============================================================
# 17. EVALUATION
# ============================================================

raw_rows = []

# Instantaneous samples used only for the theory-validation Figure 2.
# Stored inside results.pkl to avoid creating another standalone file.
theory_samples_rows = []

results = {}


for scenario_idx, scenario in enumerate(
    SCENARIOS
):

    results[
        scenario
    ] = {}

    print()
    print("=" * 82)
    print(
        f"SCENARIO = "
        f"{scenario.upper()}"
    )
    print("=" * 82)

    for beta in BETAS:

        storage = {
            method: []
            for method in METHODS
        }

        print()
        print(
            f"--- beta = "
            f"{beta:.1f} ---"
        )

        for trial in range(
            N_TRIALS
        ):

            # ------------------------------------------------
            # IMPORTANT:
            # State seed is independent of:
            #   - method
            #   - beta
            #   - scenario
            #
            # Therefore the same trial has the same theta0,
            # omegas and random baseline mask everywhere.
            # ------------------------------------------------

            state_seed = (
                BASE_SEED
                +
                trial
            )

            theta0, omegas = (
                build_initial_state(
                    state_seed
                )
            )

            # Balanced community coverage:
            # with 20 trials -> exactly 4 trials/community.
            community = (
                trial
                %
                N_COMMUNITIES
            )

            perturbation = (
                build_perturbation(
                    scenario,
                    community
                )
            )

            # ------------------------------------------------
            # Random sparse baseline:
            # SAME fixed Top-10 for the entire trajectory,
            # and SAME mask across beta/scenario for this trial.
            # ------------------------------------------------

            random_rng = (
                np.random.default_rng(
                    state_seed
                    +
                    100000
                )
            )

            random_nodes = sorted(
                random_rng.choice(
                    N_AGENTS,
                    size=K_SELECT,
                    replace=False
                ).tolist()
            )

            # ------------------------------------------------
            # Same disturbance-noise realization for every
            # method AND every beta for this scenario/trial.
            # ------------------------------------------------

            noise_seed = (
                200000
                +
                1000
                *
                scenario_idx
                +
                trial
            )

            controllers = {

                "nrpc_gl":
                    NRPCV4Controller(
                        model_gl,
                        "global_local",
                        beta
                    ),

                "nrpc_local":
                    NRPCV4Controller(
                        model_local,
                        "local",
                        beta
                    ),

                "full":
                    FullConsensus(
                        beta
                    ),

                "random":
                    FixedMaskConsensus(
                        beta,
                        random_nodes
                    ),

                "betweenness":
                    FixedMaskConsensus(
                        beta,
                        BETWEENNESS_TOP10
                    ),

                "uncontrolled":
                    NoControl(),
            }

            for (
                method,
                controller
            ) in controllers.items():

                out, theory_samples = (
                    evaluate_controller(
                        controller,
                        theta0,
                        omegas,
                        perturbation,
                        noise_seed,
                        sparse=(
                            method
                            in
                            SPARSE_METHODS
                        ),
                        beta=beta,
                    )
                )

                storage[
                    method
                ].append(
                    out
                )

                row = {
                    "scenario":
                        scenario,

                    "beta":
                        beta,

                    "trial":
                        trial,

                    "state_seed":
                        state_seed,

                    "noise_seed":
                        noise_seed,

                    "community":
                        community,

                    "method":
                        method,

                    "random_nodes":
                        ",".join(
                            map(
                                str,
                                random_nodes
                            )
                        ),
                }

                row.update(
                    out
                )

                raw_rows.append(
                    row
                )

                # Store detailed theory samples only for the representative
                # beta and only for the sparse methods used in Figure 2.
                if (
                    np.isclose(beta, THEORY_BETA)
                    and
                    method in THEORY_METHODS
                ):
                    for sample in theory_samples:
                        theory_samples_rows.append(
                            {
                                "scenario": scenario,
                                "beta": float(beta),
                                "trial": int(trial),
                                "state_seed": int(state_seed),
                                "noise_seed": int(noise_seed),
                                "community": int(community),
                                "method": method,
                                **sample,
                            }
                        )

            print(
                f" trial "
                f"{trial + 1:02d}/"
                f"{N_TRIALS}",
                end="\r"
            )

        print()

        # ====================================================
        # SUMMARY
        # ====================================================

        summary = {}

        metric_names = [
            "resilience",
            "energy",
            "energy_normalized",
            "r_mean",
            "r_min",
            "r_final",

            # Theory-driven recovery metrics
            "gamma_rec_mean",
            "gamma_rec_min",
            "gamma_rec_integral",
            "gate_rec_mean",
            "cohesive_fraction_rec",
            "max_edge_phase_diff_rec",
            "clip_fraction_rec",
            "gamma_coh_mean",
            "gamma_directional_coh_mean",
            "lambda_rest_rhs_coh_mean",
            "lambda_rest_coh_mean",
            "theory_gap_directional_coh_mean",
            "theory_gap_coh_mean",

            "bridge_during",
            "bridge_post",

            "disturbed_during",
            "disturbed_post",

            "phase_overlap_during",
            "phase_overlap_post",

            "switch_fraction",
            "switch_during",
        ]

        for method in METHODS:

            summary[
                method
            ] = {}

            for metric in metric_names:

                values = np.array(
                    [
                        x[
                            metric
                        ]
                        for x
                        in storage[
                            method
                        ]
                    ],
                    dtype=float
                )

                valid = values[
                    ~np.isnan(
                        values
                    )
                ]

                if len(valid) == 0:

                    mean = np.nan
                    std = np.nan
                    sem = np.nan

                else:

                    mean = float(
                        np.mean(
                            valid
                        )
                    )

                    std = float(
                        np.std(
                            valid,
                            ddof=1
                        )
                    ) if len(valid) > 1 else 0.0

                    sem = float(
                        std
                        /
                        np.sqrt(
                            len(valid)
                        )
                    )

                summary[
                    method
                ][metric] = {
                    "mean": mean,
                    "std": std,
                    "sem": sem,
                }

        results[
            scenario
        ][beta] = {
            "summary":
                summary,

            "raw":
                storage,
        }

        # ====================================================
        # PERFORMANCE TABLE
        # ====================================================

        print()
        print("Performance")
        print(
            f"{'Method':<15}"
            f"{'Res':>9}"
            f"{'SD':>8}"
            f"{'Energy':>11}"
            f"{'Enorm':>9}"
            f"{'r_min':>9}"
            f"{'r_final':>9}"
        )

        print("-" * 70)

        for method in METHODS:

            s = summary[
                method
            ]

            print(
                f"{method:<15}"
                f"{s['resilience']['mean']:>9.4f}"
                f"{s['resilience']['std']:>8.4f}"
                f"{s['energy']['mean']:>11.5f}"
                f"{s['energy_normalized']['mean']:>9.4f}"
                f"{s['r_min']['mean']:>9.4f}"
                f"{s['r_final']['mean']:>9.4f}"
            )

        # ====================================================
        # SPECTRAL RECOVERY TABLE
        # ====================================================

        print()
        print(
            "Spectral recovery "
            f"(fixed {RECOVERY_WINDOW:g}-s post-event window)"
        )

        print(
            f"{'Method':<15}"
            f"{'GamMean':>10}"
            f"{'GamMin':>10}"
            f"{'CohFrac':>10}"
            f"{'MaxEdge':>10}"
            f"{'ClipFrac':>10}"
        )

        print("-" * 65)

        for method in METHODS:

            s = summary[
                method
            ]

            print(
                f"{method:<15}"
                f"{s['gamma_rec_mean']['mean']:>10.5f}"
                f"{s['gamma_rec_min']['mean']:>10.5f}"
                f"{s['cohesive_fraction_rec']['mean']:>10.3f}"
                f"{s['max_edge_phase_diff_rec']['mean']:>10.3f}"
                f"{s['clip_fraction_rec']['mean']:>10.3f}"
            )

        # ====================================================
        # SELECTION TABLE
        # ====================================================

        print()
        print(
            "Selection behavior "
            "(fraction of 10-node budget)"
        )

        print(
            f"{'Method':<15}"
            f"{'BrHit':>8}"
            f"{'BrPost':>8}"
            f"{'DstHit':>8}"
            f"{'DstPost':>8}"
            f"{'PhHit':>8}"
            f"{'PhPost':>8}"
            f"{'Switch':>8}"
        )

        print("-" * 71)

        for method in [
            "nrpc_gl",
            "nrpc_local",
            "random",
            "betweenness",
        ]:

            s = summary[
                method
            ]

            print(
                f"{method:<15}"
                f"{s['bridge_during']['mean']:>8.3f}"
                f"{s['bridge_post']['mean']:>8.3f}"
                f"{s['disturbed_during']['mean']:>8.3f}"
                f"{s['disturbed_post']['mean']:>8.3f}"
                f"{s['phase_overlap_during']['mean']:>8.3f}"
                f"{s['phase_overlap_post']['mean']:>8.3f}"
                f"{s['switch_fraction']['mean']:>8.3f}"
            )


# ============================================================
# 18. SANITY CHECK:
# uncontrolled must be beta-independent
# ============================================================

print()
print("=" * 82)
print("SANITY CHECKS")
print("=" * 82)

for scenario in SCENARIOS:

    uncontrolled_means = []

    for beta in BETAS:

        uncontrolled_means.append(
            results[
                scenario
            ][beta][
                "summary"
            ][
                "uncontrolled"
            ][
                "resilience"
            ][
                "mean"
            ]
        )

    spread = (
        max(
            uncontrolled_means
        )
        -
        min(
            uncontrolled_means
        )
    )

    print(
        f"{scenario:<10} "
        f"uncontrolled beta spread = "
        f"{spread:.10f}"
    )


# ------------------------------------------------------------
# Spectral sanity checks
# ------------------------------------------------------------

uncontrolled_gamma_errors = []
full_gamma_errors = []

for row in raw_rows:

    if row["method"] == "uncontrolled":

        uncontrolled_gamma_errors.append(
            abs(
                row["gamma_rec_mean"]
                -
                GAMMA_NETWORK
            )
        )

    if row["method"] == "full":

        expected_full_gamma = (
            GAMMA_NETWORK
            +
            float(row["beta"])
            *
            row["gate_rec_mean"]
        )

        full_gamma_errors.append(
            abs(
                row["gamma_rec_mean"]
                -
                expected_full_gamma
            )
        )

print(
    "spectral uncontrolled max error = "
    f"{max(uncontrolled_gamma_errors):.3e}"
)

print(
    "spectral full closed-form max error = "
    f"{max(full_gamma_errors):.3e}"
)


# ============================================================
# 19. SAVE PICKLE
# ============================================================

payload = {

    "configuration": {

        "protocol_id":
            PROTOCOL_ID,

        "n_agents":
            N_AGENTS,

        "n_communities":
            N_COMMUNITIES,

        "community_size":
            COMMUNITY_SIZE,

        "dt":
            DT,

        "t_max":
            T_MAX,

        "k_coord":
            K_COORD,

        "k_select":
            K_SELECT,

        "target_mask":
            TARGET_MASK,

        "u_clip":
            U_CLIP,

        "energy_scale":
            ENERGY_SCALE,

        "betas":
            BETAS,

        "n_trials":
            N_TRIALS,

        "base_seed":
            BASE_SEED,

        "tag":
            TAG,

        "scenarios":
            SCENARIOS,

        "bridge_nodes":
            BRIDGE_NODES,

        "betweenness_top10":
            BETWEENNESS_TOP10,

        "lambda2":
            lambda2,

        "network_degree":
            NETWORK_DEGREE,

        "delta_max":
            DELTA_MAX,

        "c_delta":
            C_DELTA,

        "recovery_window":
            RECOVERY_WINDOW,

        "gamma_network":
            GAMMA_NETWORK,

        "spectral_operator":
            "H = c_delta*(K/degree)*L + beta*g*M",

        "theory_beta":
            THEORY_BETA,

        "lambda_rest_definition":
            "-(Vdot_fd - e^T P omega) / ||e||^2",

        "gamma_directional_definition":
            "e^T H e / ||e||^2",

        "lambda_rest_rhs_definition":
            "-e^T(coupling + u_actual) / ||e||^2",

        "gl_checkpoint":
            str(
                GL_PATH
            ),

        "local_checkpoint":
            str(
                LOCAL_PATH
            ),

        "gl_best_update":
            ckpt_gl[
                "best_update"
            ],

        "local_best_update":
            ckpt_local[
                "best_update"
            ],
    },

    "adjacency":
        A_np,

    "results":
        results,

    "raw_rows":
        raw_rows,

    "theory_samples":
        theory_samples_rows,
}


with open(
    OUTPUT_PKL,
    "wb"
) as f:

    pickle.dump(
        payload,
        f
    )


# Human-readable run manifest.
config_json = {
    "experiment": "Experiment 1 - isolated disruptions",
    "protocol_id": PROTOCOL_ID,
    "run_id": RUN_ID,
    "tag": TAG,
    "base_seed": BASE_SEED,
    "n_trials": N_TRIALS,
    "betas": [float(x) for x in BETAS],
    "dt": DT,
    "t_max": T_MAX,
    "steps": STEPS,
    "k_coord": K_COORD,
    "network_degree": NETWORK_DEGREE,
    "k_select": K_SELECT,
    "alpha": ALPHA,
    "u_clip": U_CLIP,
    "scenarios": SCENARIOS,
    "recovery_window": RECOVERY_WINDOW,
    "delta_max": DELTA_MAX,
    "c_delta": C_DELTA,
    "lambda2": lambda2,
    "gamma_network": GAMMA_NETWORK,
    "spectral_operator":
        "H(t) = c_delta*(K/degree)*L + beta*g(t)*M(t)",
    "gamma_definition":
        "lambda_min(Q^T H(t) Q)",
    "primary_spectral_metric":
        "gamma_rec_mean",
    "theory_beta": THEORY_BETA,
    "theory_methods": sorted(THEORY_METHODS),
    "lambda_rest_definition":
        "-(Vdot_fd - e^T P omega) / ||e||^2",
    "gamma_directional_definition":
        "e^T H e / ||e||^2",
    "lambda_rest_rhs_definition":
        "-e^T(coupling + u_actual) / ||e||^2",
    "theory_samples_location":
        "results.pkl['theory_samples']",
    "gl_checkpoint": str(GL_PATH),
    "local_checkpoint": str(LOCAL_PATH),
}

with open(
    OUTPUT_CONFIG_JSON,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        config_json,
        f,
        indent=2
    )


# ============================================================
# 20. SAVE RAW CSV
# ============================================================

raw_fieldnames = list(
    raw_rows[0].keys()
)

with open(
    OUTPUT_RAW_CSV,
    "w",
    newline=""
) as f:

    writer = csv.DictWriter(
        f,
        fieldnames=raw_fieldnames
    )

    writer.writeheader()
    writer.writerows(
        raw_rows
    )


# ============================================================
# 21. SAVE SUMMARY CSV
# ============================================================

summary_rows = []

for scenario in SCENARIOS:

    for beta in BETAS:

        summary = (
            results[
                scenario
            ][beta][
                "summary"
            ]
        )

        for method in METHODS:

            row = {
                "scenario":
                    scenario,

                "beta":
                    beta,

                "method":
                    method,
            }

            for (
                metric,
                stats
            ) in summary[
                method
            ].items():

                row[
                    f"{metric}_mean"
                ] = stats[
                    "mean"
                ]

                row[
                    f"{metric}_std"
                ] = stats[
                    "std"
                ]

                row[
                    f"{metric}_sem"
                ] = stats[
                    "sem"
                ]

            summary_rows.append(
                row
            )


summary_fieldnames = list(
    summary_rows[0].keys()
)

with open(
    OUTPUT_SUMMARY_CSV,
    "w",
    newline=""
) as f:

    writer = csv.DictWriter(
        f,
        fieldnames=summary_fieldnames
    )

    writer.writeheader()
    writer.writerows(
        summary_rows
    )


print()
print("=" * 82)
print("[done] Experiment 1 complete")
print(f"[done] Folder  : {EXPERIMENT_DIR}")
print(f"[done] Config  : {OUTPUT_CONFIG_JSON}")
print(f"[done] Pickle  : {OUTPUT_PKL}")
print(f"[done] Raw CSV : {OUTPUT_RAW_CSV}")
print(f"[done] Summary : {OUTPUT_SUMMARY_CSV}")
print(f"[done] Theory beta : {THEORY_BETA}")
print(f"[done] Theory samples in PKL : {len(theory_samples_rows)}")
print("=" * 82)
