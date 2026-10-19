import argparse
import csv
import json
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

# exp2

# ============================================================
# 1. ARGUMENTS
# ============================================================

parser = argparse.ArgumentParser()

parser.add_argument(
    "--trials",
    type=int,
    default=20,
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
    default=51000,
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

DT = 0.5
T_MAX = 100.0
STEPS = int(T_MAX / DT)

K_COORD = 0.40

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


# ============================================================
# 4. PATHS
# ============================================================

RESULTS_DIR = Path(
    "results"
)

RESULTS_DIR.mkdir(
    parents=True,
    exist_ok=True
)

GL_PATH = (
    RESULTS_DIR
    /
    "data_train_v4_modular_global_local.pt"
)

LOCAL_PATH = (
    RESULTS_DIR
    /
    "data_train_v4_modular_local.pt"
)


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

    return out


# ============================================================
# 16. PRINT SETUP
# ============================================================

# ============================================================
# 16. MULTI-EVENT PROTOCOL
# ============================================================

import itertools


# Three sequential disruptions in one trajectory.
EVENT_STARTS = [
    10.0,
    27.0,
    44.0,
]

# IMPORTANT:
# Every event is evaluated with the SAME post-event recovery horizon.
# Seven seconds is the longest common recovery window guaranteed before
# the next event for all three event types.
RECOVERY_WINDOW = 7.0

# Mean coherence immediately before each event. This helps quantify
# whether the system enters later disruptions already degraded.
PRE_EVENT_WINDOW = 2.0

# Mean coherence at the end of each fixed recovery window.
RECOVERY_END_WINDOW = 1.0

# Main control-effort window. It is fixed across all trials so the
# multi-event energy comparison is not diluted by the arbitrary tail
# after the final event.
PROTOCOL_START = EVENT_STARTS[0]
PROTOCOL_END = 61.0


# All six temporal orders. Every trajectory contains exactly one
# Drift, one Pulse, and one Cancel event.
EVENT_ORDERS = list(
    itertools.permutations(
        [
            "drift",
            "pulse",
            "cancel",
        ]
    )
)


# Balanced community patterns:
# - every event in a trajectory affects a different community;
# - across blocks of five trials, every temporal position sees all
#   five communities exactly once.
COMMUNITY_PATTERNS = [
    (0, 2, 4),
    (1, 3, 0),
    (2, 4, 1),
    (3, 0, 2),
    (4, 1, 3),
]


def build_event(
    event_type,
    community,
    t0,
):

    sources = community_nodes(
        community
    )

    if event_type == "drift":

        return FreqDriftNoise(
            t0=t0,
            tf=t0 + 10.0,
            source_idxs=sources,
            drift=1.6,
            sigma=1.0,
        )

    if event_type == "pulse":

        return GaussianPulse(
            t0=t0,
            tf=t0 + 6.0,
            source_idxs=sources,
            amp=20.0,
            width=2.0,
        )

    if event_type == "cancel":

        return CancelDynamics(
            t0=t0,
            tf=t0 + 10.0,
            source_idxs=sources,
        )

    raise ValueError(
        event_type
    )


def build_multievent_protocol(
    trial,
):
    """
    Deterministic balanced protocol.

    The joint order/community pattern repeats every lcm(6, 5)=30 trials.
    """

    event_order = (
        EVENT_ORDERS[
            trial
            %
            len(EVENT_ORDERS)
        ]
    )

    communities = (
        COMMUNITY_PATTERNS[
            trial
            %
            len(COMMUNITY_PATTERNS)
        ]
    )

    assert len(
        set(
            communities
        )
    ) == 3

    events = []

    for (
        event_type,
        community,
        t0
    ) in zip(
        event_order,
        communities,
        EVENT_STARTS,
    ):

        events.append(
            build_event(
                event_type,
                community,
                t0,
            )
        )

    # The fixed 7-s evaluation window must fit before the next event.
    for j in range(
        len(events) - 1
    ):

        assert (
            events[j].tf
            +
            RECOVERY_WINDOW
            <=
            events[j + 1].t0
            +
            1e-12
        )

    return (
        events,
        event_order,
        communities,
    )


# ============================================================
# 17. MULTI-EVENT PERFORMANCE METRICS
# ============================================================

def _time_slice(
    t0,
    tf,
    n_steps,
):
    """Half-open [t0, tf) slice clipped to the simulated trajectory."""

    start = max(
        0,
        int(
            np.floor(
                t0
                /
                DT
            )
        )
    )

    end = min(
        n_steps,
        int(
            np.floor(
                tf
                /
                DT
            )
        )
    )

    return slice(
        start,
        max(
            start + 1,
            end
        ),
    )


def _mean_squared_effort(
    u,
    sl,
):
    return torch.mean(
        u[
            sl
        ] ** 2
    )


def multievent_metrics(
    r,
    u,
    events,
    event_order,
):
    """
    Metrics for sequential disruption robustness.

    Main trajectory-level metrics
    -----------------------------
    resilience_mean:
        Arithmetic mean over the three event scores.

    resilience_worst:
        Minimum event score. This is the PRIMARY robustness metric:
        a controller should not fail badly on one event while looking
        acceptable on average.

    resilience_harmonic:
        Harmonic mean over event scores, which penalizes low events more
        strongly than the arithmetic mean.

    Event-level resilience
    ----------------------
    Each event is evaluated using:
        impact   = [t0, tf)
        recovery = [tf, tf + RECOVERY_WINDOW)

    The same 7-s recovery horizon is used for all events and all positions,
    avoiding the previous confound in which the final event was evaluated
    over the entire remaining trajectory.

    Accumulation metrics
    --------------------
    event1/2/3_resilience:
        Resilience by temporal position.

    resilience_drop_1_to_3:
        event1 - event3. Positive values indicate degradation across the
        sequence.

    pre_event{1,2,3}_r:
        Mean coherence during the 2 s before each event.

    pre_event_r_drop_1_to_3:
        pre_event1_r - pre_event3_r. Positive values indicate that later
        events start from a more degraded state.

    Event-type metrics
    ------------------
    drift/pulse/cancel_resilience map each score back to perturbation type,
    allowing us to distinguish temporal-position effects from event severity.
    """

    event_scores = []
    pre_event_r = []
    recovery_end_r = []
    event_efforts = []

    for event in events:

        hit_slice = _time_slice(
            event.t0,
            event.tf,
            len(r),
        )

        recovery_slice = _time_slice(
            event.tf,
            event.tf
            +
            RECOVERY_WINDOW,
            len(r),
        )

        pre_slice = _time_slice(
            event.t0
            -
            PRE_EVENT_WINDOW,
            event.t0,
            len(r),
        )

        recovery_end_slice = _time_slice(
            event.tf
            +
            RECOVERY_WINDOW
            -
            RECOVERY_END_WINDOW,
            event.tf
            +
            RECOVERY_WINDOW,
            len(r),
        )

        event_window_slice = _time_slice(
            event.t0,
            event.tf
            +
            RECOVERY_WINDOW,
            len(r),
        )

        r_hit = r[
            hit_slice
        ]

        r_recovery = r[
            recovery_slice
        ]

        absorption = (
            0.5
            *
            torch.min(
                r_hit
            )
            +
            0.5
            *
            torch.mean(
                r_hit
            )
        )

        recovery_error = torch.mean(
            (
                1.0
                -
                r_recovery
            ) ** 2
        )

        score = (
            absorption
            *
            (
                1.0
                -
                recovery_error
            )
        )

        event_scores.append(
            score
        )

        pre_event_r.append(
            torch.mean(
                r[
                    pre_slice
                ]
            )
        )

        recovery_end_r.append(
            torch.mean(
                r[
                    recovery_end_slice
                ]
            )
        )

        event_efforts.append(
            _mean_squared_effort(
                u,
                event_window_slice,
            )
        )

    event_scores = torch.stack(
        event_scores
    )

    pre_event_r = torch.stack(
        pre_event_r
    )

    recovery_end_r = torch.stack(
        recovery_end_r
    )

    event_efforts = torch.stack(
        event_efforts
    )

    mean_score = torch.mean(
        event_scores
    )

    worst_score = torch.min(
        event_scores
    )

    eps = 1e-8

    harmonic_score = (
        len(
            event_scores
        )
        /
        torch.sum(
            1.0
            /
            (
                event_scores
                +
                eps
            )
        )
    )

    protocol_slice = _time_slice(
        PROTOCOL_START,
        PROTOCOL_END,
        len(r),
    )

    effort_protocol = _mean_squared_effort(
        u,
        protocol_slice,
    )

    effort_full = torch.mean(
        u ** 2
    )

    r_protocol = r[
        protocol_slice
    ]

    protocol_end_idx = min(
        len(r) - 1,
        int(
            PROTOCOL_END
            /
            DT
        )
        -
        1,
    )

    # Map score by perturbation type.
    by_type = {
        event_type:
            event_scores[j]
        for j, event_type
        in enumerate(
            event_order
        )
    }

    worst_idx = int(
        torch.argmin(
            event_scores
        ).item()
    )

    return {
        "resilience_mean":
            float(
                mean_score.item()
            ),

        "resilience_worst":
            float(
                worst_score.item()
            ),

        "resilience_harmonic":
            float(
                harmonic_score.item()
            ),

        # Temporal position
        "event1_resilience":
            float(
                event_scores[0].item()
            ),

        "event2_resilience":
            float(
                event_scores[1].item()
            ),

        "event3_resilience":
            float(
                event_scores[2].item()
            ),

        "resilience_drop_1_to_3":
            float(
                (
                    event_scores[0]
                    -
                    event_scores[2]
                ).item()
            ),

        # Event type
        "drift_resilience":
            float(
                by_type[
                    "drift"
                ].item()
            ),

        "pulse_resilience":
            float(
                by_type[
                    "pulse"
                ].item()
            ),

        "cancel_resilience":
            float(
                by_type[
                    "cancel"
                ].item()
            ),

        # State before each disruption
        "pre_event1_r":
            float(
                pre_event_r[0].item()
            ),

        "pre_event2_r":
            float(
                pre_event_r[1].item()
            ),

        "pre_event3_r":
            float(
                pre_event_r[2].item()
            ),

        "pre_event_r_drop_1_to_3":
            float(
                (
                    pre_event_r[0]
                    -
                    pre_event_r[2]
                ).item()
            ),

        # End of standardized recovery window
        "recovery1_end_r":
            float(
                recovery_end_r[0].item()
            ),

        "recovery2_end_r":
            float(
                recovery_end_r[1].item()
            ),

        "recovery3_end_r":
            float(
                recovery_end_r[2].item()
            ),

        # Control effort
        "energy":
            float(
                effort_protocol.item()
            ),

        "energy_normalized":
            float(
                (
                    effort_protocol
                    /
                    ENERGY_SCALE
                ).item()
            ),

        "energy_full_trajectory":
            float(
                effort_full.item()
            ),

        "event1_energy":
            float(
                event_efforts[0].item()
            ),

        "event2_energy":
            float(
                event_efforts[1].item()
            ),

        "event3_energy":
            float(
                event_efforts[2].item()
            ),

        # Protocol-level coherence
        "r_mean_protocol":
            float(
                r_protocol.mean().item()
            ),

        "r_min_protocol":
            float(
                r_protocol.min().item()
            ),

        "r_protocol_end":
            float(
                r[
                    protocol_end_idx
                ].item()
            ),

        # Final simulator state kept only as a diagnostic.
        "r_final":
            float(
                r[-1].item()
            ),

        "worst_event_position":
            worst_idx
            +
            1,

        "worst_event_type":
            event_order[
                worst_idx
            ],
    }


# ============================================================
# 18. MULTI-EVENT SELECTION DIAGNOSTICS
# ============================================================

def multievent_selection_metrics(
    m_hist,
    events,
):
    """
    Secondary diagnostics for sparse controllers.

    These are NOT headline performance metrics. They are retained only to
    explain how adaptive masks respond as the disturbed community changes.
    """

    m = stack_history(
        m_hist
    )

    bridge_vec = torch.zeros(
        N_AGENTS,
        device=device
    )

    bridge_vec[
        BRIDGE_NODES
    ] = 1.0

    bridge_rate = (
        (
            m
            *
            bridge_vec.view(
                1,
                -1
            )
        ).sum(
            dim=1
        )
        /
        K_SELECT
    )

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

    switch_times = t_axis[
        1:
    ]

    bridge_hit_values = []
    bridge_recovery_values = []

    disturbed_hit_values = []
    disturbed_recovery_values = []

    event_switch_values = []

    event_disturbed_hit = []

    bridge_np = (
        bridge_rate
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

    for event in events:

        hit_mask = (
            (t_axis >= event.t0)
            &
            (t_axis < event.tf)
        )

        recovery_mask = (
            (t_axis >= event.tf)
            &
            (
                t_axis
                <
                event.tf
                +
                RECOVERY_WINDOW
            )
        )

        disturbed_vec = torch.zeros(
            N_AGENTS,
            device=device
        )

        disturbed_vec[
            event.source_idxs
        ] = 1.0

        disturbed_rate = (
            (
                m
                *
                disturbed_vec.view(
                    1,
                    -1
                )
            ).sum(
                dim=1
            )
            /
            K_SELECT
        )

        disturbed_np = (
            disturbed_rate
            .detach()
            .cpu()
            .numpy()
        )

        bridge_hit_values.append(
            float(
                bridge_np[
                    hit_mask
                ].mean()
            )
        )

        bridge_recovery_values.append(
            float(
                bridge_np[
                    recovery_mask
                ].mean()
            )
        )

        current_disturbed_hit = float(
            disturbed_np[
                hit_mask
            ].mean()
        )

        event_disturbed_hit.append(
            current_disturbed_hit
        )

        disturbed_hit_values.append(
            current_disturbed_hit
        )

        disturbed_recovery_values.append(
            float(
                disturbed_np[
                    recovery_mask
                ].mean()
            )
        )

        switch_mask = (
            (switch_times >= event.t0)
            &
            (
                switch_times
                <
                event.tf
                +
                RECOVERY_WINDOW
            )
        )

        event_switch_values.append(
            float(
                switch_np[
                    switch_mask
                ].mean()
            )
        )

    return {
        "bridge_hit":
            float(
                np.mean(
                    bridge_hit_values
                )
            ),

        "bridge_recovery":
            float(
                np.mean(
                    bridge_recovery_values
                )
            ),

        "disturbed_hit":
            float(
                np.mean(
                    disturbed_hit_values
                )
            ),

        "disturbed_recovery":
            float(
                np.mean(
                    disturbed_recovery_values
                )
            ),

        "event1_disturbed_hit":
            event_disturbed_hit[0],

        "event2_disturbed_hit":
            event_disturbed_hit[1],

        "event3_disturbed_hit":
            event_disturbed_hit[2],

        "switch_fraction":
            float(
                switch_fraction
                .mean()
                .item()
            ),

        "switch_event_windows":
            float(
                np.mean(
                    event_switch_values
                )
            ),
    }


# ============================================================
# 19. MULTI-EVENT TRAJECTORY EVALUATION
# ============================================================

@torch.no_grad()
def evaluate_multievent_controller(
    controller,
    theta0,
    omegas,
    events,
    event_order,
    noise_seed,
    sparse,
):

    # Every method at a given trial receives the same stochastic realization.
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

        events,

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

    out = multievent_metrics(
        r,
        u,
        events,
        event_order,
    )

    if sparse:

        out.update(
            multievent_selection_metrics(
                m_hist,
                events,
            )
        )

    else:

        for key in [
            "bridge_hit",
            "bridge_recovery",
            "disturbed_hit",
            "disturbed_recovery",
            "event1_disturbed_hit",
            "event2_disturbed_hit",
            "event3_disturbed_hit",
            "switch_fraction",
            "switch_event_windows",
        ]:

            out[
                key
            ] = np.nan

    return out


# ============================================================
# 20. OUTPUT PATHS
# ============================================================

def _fmt(
    x,
):
    return str(
        x
    ).replace(
        ".",
        "p"
    )


RUN_ID = (
    f"{TAG}"
    f"_seed{BASE_SEED}"
    f"_T{_fmt(T_MAX)}"
    f"_dt{_fmt(DT)}"
    f"_K{_fmt(K_COORD)}"
)

EXPERIMENT_DIR = (
    RESULTS_DIR
    /
    "experiment2"
    /
    RUN_ID
)

EXPERIMENT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

OUTPUT_PKL = (
    EXPERIMENT_DIR
    /
    "results.pkl"
)

OUTPUT_RAW_CSV = (
    EXPERIMENT_DIR
    /
    "raw.csv"
)

OUTPUT_SUMMARY_CSV = (
    EXPERIMENT_DIR
    /
    "summary.csv"
)

OUTPUT_CONFIG_JSON = (
    EXPERIMENT_DIR
    /
    "config.json"
)


# ============================================================
# 21. PRINT SETUP
# ============================================================

print("=" * 86)
print("EXPERIMENT 2 | SEQUENTIAL MULTI-EVENT ROBUSTNESS")
print("=" * 86)

print(f"Device                    : {device}")
print(f"N                         : {N_AGENTS}")
print(f"Communities               : {N_COMMUNITIES} x {COMMUNITY_SIZE}")
print(f"Degree                    : 4")
print(f"lambda_2                  : {lambda2:.6f}")
print(f"K                         : {K_COORD}")
print(f"Sparse budget             : {K_SELECT}/{N_AGENTS}")
print(f"Betas                     : {BETAS}")
print(f"Trials/beta               : {N_TRIALS}")
print(f"Events/trajectory         : 3")
print(f"Event starts              : {EVENT_STARTS}")
print(f"Fixed recovery window     : {RECOVERY_WINDOW} s")
print(f"Protocol effort window    : [{PROTOCOL_START}, {PROTOCOL_END}) s")
print("Types/trajectory          : Drift + Pulse + Cancel")
print("Type order                : balanced over all 6 permutations")
print("Community                 : changes at every event")
print(f"Methods                   : {METHODS}")
print(f"Output directory          : {EXPERIMENT_DIR}")
print(f"Base seed                 : {BASE_SEED}")
print(f"GL best update            : {ckpt_gl['best_update']}")
print(f"Local best update         : {ckpt_local['best_update']}")

print("=" * 86)


# ============================================================
# 22. EVALUATION
# ============================================================

raw_rows = []

results = {}


for beta in BETAS:

    storage = {
        method: []
        for method in METHODS
    }

    print()
    print("=" * 86)
    print(
        f"BETA = {beta:.1f}"
    )
    print("=" * 86)

    for trial in range(
        N_TRIALS
    ):

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

        (
            events,
            event_order,
            communities,
        ) = build_multievent_protocol(
            trial
        )

        # Fixed Random-k mask for the entire trajectory.
        random_rng = np.random.default_rng(
            state_seed
            +
            100000
        )

        random_nodes = sorted(
            random_rng.choice(
                N_AGENTS,
                size=K_SELECT,
                replace=False
            ).tolist()
        )

        # Same event-noise realization across every method and beta.
        noise_seed = (
            BASE_SEED
            +
            200000
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

            out = evaluate_multievent_controller(
                controller,
                theta0,
                omegas,
                events,
                event_order,
                noise_seed,
                sparse=(
                    method
                    in
                    SPARSE_METHODS
                ),
            )

            storage[
                method
            ].append(
                out
            )

            row = {

                "beta":
                    beta,

                "trial":
                    trial,

                "state_seed":
                    state_seed,

                "noise_seed":
                    noise_seed,

                "method":
                    method,

                "event1_type":
                    event_order[0],

                "event2_type":
                    event_order[1],

                "event3_type":
                    event_order[2],

                "event1_community":
                    communities[0],

                "event2_community":
                    communities[1],

                "event3_community":
                    communities[2],

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

        print(
            f" trial "
            f"{trial + 1:03d}/"
            f"{N_TRIALS}",
            end="\r"
        )

    print()

    # ========================================================
    # SUMMARY
    # ========================================================

    metric_names = [

        # Headline multi-event robustness
        "resilience_mean",
        "resilience_worst",
        "resilience_harmonic",

        # Temporal position / accumulation
        "event1_resilience",
        "event2_resilience",
        "event3_resilience",
        "resilience_drop_1_to_3",

        # Perturbation type
        "drift_resilience",
        "pulse_resilience",
        "cancel_resilience",

        # State entering each event
        "pre_event1_r",
        "pre_event2_r",
        "pre_event3_r",
        "pre_event_r_drop_1_to_3",

        # Standardized end-of-recovery state
        "recovery1_end_r",
        "recovery2_end_r",
        "recovery3_end_r",

        # Control effort
        "energy",
        "energy_normalized",
        "energy_full_trajectory",
        "event1_energy",
        "event2_energy",
        "event3_energy",

        # Protocol coherence
        "r_mean_protocol",
        "r_min_protocol",
        "r_protocol_end",
        "r_final",

        # Secondary selection diagnostics
        "bridge_hit",
        "bridge_recovery",
        "disturbed_hit",
        "disturbed_recovery",
        "event1_disturbed_hit",
        "event2_disturbed_hit",
        "event3_disturbed_hit",
        "switch_fraction",
        "switch_event_windows",
    ]

    summary = {}

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

            if len(
                valid
            ) == 0:

                mean = np.nan
                std = np.nan
                sem = np.nan

            else:

                mean = float(
                    np.mean(
                        valid
                    )
                )

                std = (
                    float(
                        np.std(
                            valid,
                            ddof=1
                        )
                    )
                    if len(
                        valid
                    ) > 1
                    else 0.0
                )

                sem = float(
                    std
                    /
                    np.sqrt(
                        len(
                            valid
                        )
                    )
                )

            summary[
                method
            ][
                metric
            ] = {
                "mean":
                    mean,

                "std":
                    std,

                "sem":
                    sem,
            }

    results[
        beta
    ] = {
        "summary":
            summary,

        "raw":
            storage,
    }

    # ========================================================
    # MAIN PERFORMANCE TABLE
    # ========================================================

    print()
    print("Sequential multi-event performance")
    print(
        f"{'Method':<15}"
        f"{'Rmean':>9}"
        f"{'Rworst':>9}"
        f"{'Rharm':>9}"
        f"{'R1':>8}"
        f"{'R2':>8}"
        f"{'R3':>8}"
        f"{'Drop':>8}"
        f"{'Effort':>10}"
    )

    print("-" * 92)

    for method in METHODS:

        s = summary[
            method
        ]

        print(
            f"{method:<15}"
            f"{s['resilience_mean']['mean']:>9.4f}"
            f"{s['resilience_worst']['mean']:>9.4f}"
            f"{s['resilience_harmonic']['mean']:>9.4f}"
            f"{s['event1_resilience']['mean']:>8.4f}"
            f"{s['event2_resilience']['mean']:>8.4f}"
            f"{s['event3_resilience']['mean']:>8.4f}"
            f"{s['resilience_drop_1_to_3']['mean']:>8.4f}"
            f"{s['energy']['mean']:>10.5f}"
        )

    # ========================================================
    # EVENT-TYPE TABLE
    # ========================================================

    print()
    print("Resilience by disruption type")
    print(
        f"{'Method':<15}"
        f"{'Drift':>10}"
        f"{'Pulse':>10}"
        f"{'Cancel':>10}"
        f"{'Pre1':>10}"
        f"{'Pre2':>10}"
        f"{'Pre3':>10}"
    )

    print("-" * 75)

    for method in METHODS:

        s = summary[
            method
        ]

        print(
            f"{method:<15}"
            f"{s['drift_resilience']['mean']:>10.4f}"
            f"{s['pulse_resilience']['mean']:>10.4f}"
            f"{s['cancel_resilience']['mean']:>10.4f}"
            f"{s['pre_event1_r']['mean']:>10.4f}"
            f"{s['pre_event2_r']['mean']:>10.4f}"
            f"{s['pre_event3_r']['mean']:>10.4f}"
        )

    # ========================================================
    # SECONDARY SELECTION TABLE
    # ========================================================

    print()
    print(
        "Sparse-selection diagnostics "
        "(secondary)"
    )

    print(
        f"{'Method':<15}"
        f"{'DstHit':>9}"
        f"{'DstRec':>9}"
        f"{'D1':>8}"
        f"{'D2':>8}"
        f"{'D3':>8}"
        f"{'Switch':>9}"
    )

    print("-" * 66)

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
            f"{s['disturbed_hit']['mean']:>9.3f}"
            f"{s['disturbed_recovery']['mean']:>9.3f}"
            f"{s['event1_disturbed_hit']['mean']:>8.3f}"
            f"{s['event2_disturbed_hit']['mean']:>8.3f}"
            f"{s['event3_disturbed_hit']['mean']:>8.3f}"
            f"{s['switch_event_windows']['mean']:>9.3f}"
        )


# ============================================================
# 23. SANITY CHECKS
# ============================================================

print()
print("=" * 86)
print("SANITY CHECKS")
print("=" * 86)

for metric in [
    "resilience_mean",
    "resilience_worst",
    "resilience_harmonic",
    "energy",
]:

    uncontrolled_values = [
        results[
            beta
        ][
            "summary"
        ][
            "uncontrolled"
        ][
            metric
        ][
            "mean"
        ]
        for beta in BETAS
    ]

    spread = (
        max(
            uncontrolled_values
        )
        -
        min(
            uncontrolled_values
        )
    )

    print(
        f"uncontrolled {metric:<20} beta spread = "
        f"{spread:.10f}"
    )


# Verify that the protocol generation really cycles through every order.
orders_seen = {
    tuple(
        build_multievent_protocol(
            trial
        )[1]
    )
    for trial in range(
        max(
            30,
            N_TRIALS
        )
    )
}

print(
    f"event orders represented        = "
    f"{len(orders_seen)}/6"
)

assert len(
    orders_seen
) == 6


# ============================================================
# 24. SAVE PICKLE / CONFIG
# ============================================================

payload = {

    "configuration": {

        "experiment":
            "sequential_multi_event_robustness",

        "protocol_version":
            "EXP2_MULTI_EVENT_FIXED_RECOVERY_V2",

        "base_seed":
            BASE_SEED,

        "tag":
            TAG,

        "n_trials":
            N_TRIALS,

        "betas":
            BETAS,

        "dt":
            DT,

        "t_max":
            T_MAX,

        "k_coord":
            K_COORD,

        "k_select":
            K_SELECT,

        "event_starts":
            EVENT_STARTS,

        "event_orders":
            EVENT_ORDERS,

        "community_patterns":
            COMMUNITY_PATTERNS,

        "events_per_trajectory":
            3,

        "recovery_window":
            RECOVERY_WINDOW,

        "pre_event_window":
            PRE_EVENT_WINDOW,

        "protocol_effort_window":
            [
                PROTOCOL_START,
                PROTOCOL_END,
            ],

        "primary_metric":
            "resilience_worst",

        "secondary_metrics":
            [
                "resilience_harmonic",
                "resilience_mean",
                "resilience_drop_1_to_3",
            ],

        "effort_metric":
            "mean squared control effort over fixed multi-event protocol window",

        "methods":
            METHODS,

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
}


with open(
    OUTPUT_PKL,
    "wb"
) as f:

    pickle.dump(
        payload,
        f
    )


config_json = {
    "experiment":
        "Experiment 2 - sequential multi-event robustness",

    "protocol_version":
        "EXP2_MULTI_EVENT_FIXED_RECOVERY_V2",

    "run_id":
        RUN_ID,

    "base_seed":
        BASE_SEED,

    "n_trials":
        N_TRIALS,

    "betas":
        [
            float(
                x
            )
            for x in BETAS
        ],

    "dt":
        DT,

    "t_max":
        T_MAX,

    "k_coord":
        K_COORD,

    "n_agents":
        N_AGENTS,

    "network_degree":
        4,

    "k_select":
        K_SELECT,

    "event_starts":
        EVENT_STARTS,

    "recovery_window":
        RECOVERY_WINDOW,

    "pre_event_window":
        PRE_EVENT_WINDOW,

    "protocol_effort_window":
        [
            PROTOCOL_START,
            PROTOCOL_END,
        ],

    "primary_metric":
        "resilience_worst",

    "secondary_metrics":
        [
            "resilience_harmonic",
            "resilience_mean",
            "resilience_drop_1_to_3",
        ],

    "methods":
        METHODS,

    "gl_checkpoint":
        str(
            GL_PATH
        ),

    "local_checkpoint":
        str(
            LOCAL_PATH
        ),
}


with open(
    OUTPUT_CONFIG_JSON,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        config_json,
        f,
        indent=2,
    )


# ============================================================
# 25. SAVE RAW CSV
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
# 26. SAVE SUMMARY CSV
# ============================================================

summary_rows = []

for beta in BETAS:

    summary = results[
        beta
    ][
        "summary"
    ]

    for method in METHODS:

        row = {
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
print("=" * 86)
print("[done] Experiment 2 complete")
print(f"[done] Folder  : {EXPERIMENT_DIR}")
print(f"[done] Config  : {OUTPUT_CONFIG_JSON}")
print(f"[done] Pickle  : {OUTPUT_PKL}")
print(f"[done] Raw CSV : {OUTPUT_RAW_CSV}")
print(f"[done] Summary : {OUTPUT_SUMMARY_CSV}")
print("=" * 86)
