import argparse
import copy
import math
import random
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


# ============================================================
# 1. ARGUMENTS
# ============================================================

parser = argparse.ArgumentParser()

parser.add_argument(
    "--mode",
    type=str,
    choices=["global_local", "local"],
    required=True,
)

parser.add_argument(
    "--updates",
    type=int,
    default=800,
)

parser.add_argument(
    "--rollouts",
    type=int,
    default=2,
)

args = parser.parse_args()

MODE = args.mode
N_UPDATES = args.updates
TRAJECTORIES_PER_UPDATE = args.rollouts


# ============================================================
# 2. DEVICE / REPRODUCIBILITY
# ============================================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

BASE_SEED = 42

torch.manual_seed(BASE_SEED)
np.random.seed(BASE_SEED)
random.seed(BASE_SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(BASE_SEED)


# ============================================================
# 3. SYSTEM CONFIGURATION
# ============================================================

N_COMMUNITIES = 5
COMMUNITY_SIZE = 20
N_AGENTS = N_COMMUNITIES * COMMUNITY_SIZE

T_MAX = 60.0
DT = 0.1
STEPS = int(T_MAX / DT)

# Modular network operating regime
K_MIN = 0.45
K_MAX = 0.55

# ------------------------------------------------------------
# Sparse actuation
# ------------------------------------------------------------

K_SELECT = 10
TARGET_MASK = K_SELECT / N_AGENTS

# ------------------------------------------------------------
# Controller gain
# ------------------------------------------------------------

BETA_MIN = 1.0
BETA_MAX = 10.0

ALPHA = 10.0
U_CLIP = 2.0

# ------------------------------------------------------------
# Optimizer
# ------------------------------------------------------------

LEARNING_RATE = 1e-3
MIN_LEARNING_RATE = 2e-5

GRAD_CLIP = 1.0

TAU_START = 1.0
TAU_END = 0.30

# ------------------------------------------------------------
# V4 LOSS
#
# All main terms now live on interpretable scales.
# ------------------------------------------------------------

LAMBDA_RESILIENCE = 1.0

LAMBDA_ENERGY = 0.50

LAMBDA_SPARSITY = 10.0

LAMBDA_SMOOTHNESS = 0.05

LAMBDA_ENTROPY = 0.005

# Maximum mean-squared effort possible under the
# sparse budget if every selected actuator saturates.
#
# rho * u_max^2 = 0.10 * 4 = 0.4
ENERGY_SCALE = (
    TARGET_MASK
    *
    U_CLIP ** 2
)


# ============================================================
# 4. OUTPUT
# ============================================================

RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(
    parents=True,
    exist_ok=True
)

OUTPUT_MODEL = (
    RESULTS_DIR
    /
    f"data_train_v4_modular_{MODE}.pt"
)


# ============================================================
# 5. MODULAR DEGREE-4 TOPOLOGY
# ============================================================

def build_modular_degree4_graph():

    A = np.zeros(
        (N_AGENTS, N_AGENTS),
        dtype=np.float32
    )

    # --------------------------------------------------------
    # Five degree-4 ring-lattice communities
    # --------------------------------------------------------

    for c in range(N_COMMUNITIES):

        base = c * COMMUNITY_SIZE

        for i in range(COMMUNITY_SIZE):

            for offset in [1, 2]:

                j = (
                    i + offset
                ) % COMMUNITY_SIZE

                u = base + i
                v = base + j

                A[u, v] = 1.0
                A[v, u] = 1.0

    # --------------------------------------------------------
    # Degree-preserving inter-community rewiring
    # --------------------------------------------------------

    bridge_nodes = set()

    for c in range(N_COMMUNITIES):

        d = (
            c + 1
        ) % N_COMMUNITIES

        u1 = c * COMMUNITY_SIZE + 4
        u2 = c * COMMUNITY_SIZE + 5

        v1 = d * COMMUNITY_SIZE + 12
        v2 = d * COMMUNITY_SIZE + 13

        assert A[u1, u2] == 1.0
        assert A[v1, v2] == 1.0

        # Remove two internal edges
        A[u1, u2] = 0.0
        A[u2, u1] = 0.0

        A[v1, v2] = 0.0
        A[v2, v1] = 0.0

        # Add inter-community edges
        A[u1, v1] = 1.0
        A[v1, u1] = 1.0

        A[u2, v2] = 1.0
        A[v2, u2] = 1.0

        bridge_nodes.update(
            [u1, u2, v1, v2]
        )

    return A, sorted(bridge_nodes)


A_np, BRIDGE_NODES = (
    build_modular_degree4_graph()
)

G = nx.from_numpy_array(
    A_np
)

degrees_np = np.array(
    [
        G.degree(i)
        for i in range(N_AGENTS)
    ],
    dtype=np.float32
)

assert np.all(
    degrees_np == 4
)

assert nx.is_connected(G)

A = torch.tensor(
    A_np,
    dtype=torch.float32,
    device=device
)

degree = A.sum(
    dim=1
).clamp(min=1.0)

# ------------------------------------------------------------
# Local structural information
#
# IMPORTANT:
# We DO NOT give the controller:
#   - betweenness
#   - community identity
#   - bridge label
#   - node ID
#
# Only locally computable structural descriptors.
# ------------------------------------------------------------

clustering_dict = nx.clustering(
    G
)

clustering_np = np.array(
    [
        clustering_dict[i]
        for i in range(N_AGENTS)
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

# Diagnostics only -- NOT controller features
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

L = (
    np.diag(degrees_np)
    -
    A_np
)

lambda2 = float(
    np.sort(
        np.linalg.eigvalsh(L)
    )[1]
)


# ============================================================
# 6. PRINT CONFIGURATION
# ============================================================

print("=" * 76)
print(
    f"NRPC-V4 MODULAR TRAINING | "
    f"MODE = {MODE}"
)
print("=" * 76)

print(f"Device                : {device}")
print(f"Agents                : {N_AGENTS}")

print(
    f"Communities           : "
    f"{N_COMMUNITIES} x "
    f"{COMMUNITY_SIZE}"
)

print(
    f"Degree min/max        : "
    f"{degrees_np.min():.0f}/"
    f"{degrees_np.max():.0f}"
)

print(f"lambda_2              : {lambda2:.6f}")

print(f"DT                    : {DT}")
print(f"Steps                 : {STEPS}")

print(
    f"Actuation budget      : "
    f"{K_SELECT}/{N_AGENTS}"
)

print(
    f"Target mask           : "
    f"{TARGET_MASK:.2f}"
)

print(
    f"Beta range            : "
    f"[{BETA_MIN}, {BETA_MAX}] "
    f"(log-uniform)"
)

print(
    f"K range               : "
    f"[{K_MIN}, {K_MAX}]"
)

print(f"Updates               : {N_UPDATES}")

print(
    f"Trajectories/update   : "
    f"{TRAJECTORIES_PER_UPDATE}"
)

print(
    f"Energy scale          : "
    f"{ENERGY_SCALE:.4f}"
)

print(
    f"Bridge nodes          : "
    f"{BRIDGE_NODES}"
)

print(
    f"Betweenness Top-10    : "
    f"{BETWEENNESS_TOP10}"
)

print("=" * 76)


# ============================================================
# 7. V4 NODE FEATURES
# ============================================================

def build_node_features_v4(
    theta,
    mode,
):
    """
    Rotation-invariant node-wise features.

    GLOBAL features: 3
        0 sin(theta_i - psi)
        1 cos(theta_i - psi)
        2 global r

    LOCAL DYNAMIC features: 5
        3 avg_j sin(theta_j - theta_i)
        4 avg_j cos(theta_j - theta_i)
        5 local first-moment magnitude
        6 avg_j sin(2(theta_j - theta_i))
        7 avg_j cos(2(theta_j - theta_i))

    LOCAL STRUCTURAL features: 2
        8 local clustering coefficient C_i
        9 mean clustering coefficient of neighbors

    Total = 10.

    local mode zeros only the first 3 global channels.
    """

    if theta.dim() == 1:
        theta = theta.unsqueeze(0)

    B, N = theta.shape

    # --------------------------------------------------------
    # Global circular information
    # --------------------------------------------------------

    zbar = torch.mean(
        torch.exp(1j * theta),
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
    # Local circular moments
    # --------------------------------------------------------

    theta_i = theta.unsqueeze(2)
    theta_j = theta.unsqueeze(1)

    delta = (
        theta_j
        -
        theta_i
    )

    A_batch = A.unsqueeze(0)

    deg = degree.view(
        1,
        N
    )

    local_sin_1 = (
        (
            torch.sin(delta)
            *
            A_batch
        ).sum(dim=2)
        /
        deg
    )

    local_cos_1 = (
        (
            torch.cos(delta)
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
    # Local structural information
    # --------------------------------------------------------

    c_i = (
        clustering
        .view(1, N)
        .expand(B, -1)
    )

    c_neigh = (
        neighbor_clustering
        .view(1, N)
        .expand(B, -1)
    )

    # --------------------------------------------------------
    # Local-only selector:
    # remove global channels, leave local channels unchanged.
    # --------------------------------------------------------

    if mode == "local":

        global_sin = torch.zeros_like(
            global_sin
        )

        global_cos = torch.zeros_like(
            global_cos
        )

        global_r = torch.zeros_like(
            global_r
        )

    elif mode != "global_local":

        raise ValueError(
            f"Unknown mode: {mode}"
        )

    features = torch.stack(
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

    return features


# ============================================================
# 8. PER-NODE NEURAL SCORER
# ============================================================

class NodeScoreNetwork(nn.Module):

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

        self.output_scale = nn.Parameter(
            torch.tensor(1.0)
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

        z = self.net(
            x
        ).reshape(
            B,
            N
        )

        return (
            z
            *
            self.output_scale
        )


# ============================================================
# 9. V4 CONTROLLER
# ============================================================

class NRPCV4Controller:

    def __init__(
        self,
        model,
        mode,
        beta,
        k_select,
        tau,
        soft_selection,
    ):

        self.model = model
        self.mode = mode

        self.beta = beta

        self.k_select = k_select
        self.tau = tau

        self.soft_selection = (
            soft_selection
        )

    def __call__(
        self,
        theta,
    ):

        squeeze_back = False

        if theta.dim() == 1:

            theta = theta.view(
                1,
                -1
            )

            squeeze_back = True

        # ----------------------------------------------------
        # SAME global feedback law for GL and L.
        #
        # This is deliberate:
        # the two models differ ONLY in information available
        # to the node-selection rule.
        # ----------------------------------------------------

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

        u_cons = (
            -self.beta
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

        u_cons = (
            u_cons
            *
            gate
        )

        # ----------------------------------------------------
        # Node-selection score
        # ----------------------------------------------------

        features = (
            build_node_features_v4(
                theta,
                self.mode
            )
        )

        logits = self.model(
            features
        )

        # ----------------------------------------------------
        # Differentiable training / hard deployment
        # ----------------------------------------------------

        if self.soft_selection:

            # ------------------------------------------------
            # Soft surrogate used only for gradients
            # ------------------------------------------------

            probs = torch.softmax(
                logits / self.tau,
                dim=1
            )

            m_soft = (
                probs
                *
                self.k_select
            )

            # ------------------------------------------------
            # HARD Top-k mask used in the forward trajectory
            # ------------------------------------------------

            m_hard = torch.zeros_like(
                logits
            )

            idx = torch.topk(
                logits,
                k=self.k_select,
                dim=1
            ).indices

            m_hard.scatter_(
                1,
                idx,
                1.0
            )

            # ------------------------------------------------
            # Straight-through estimator:
            #
            # forward  -> m_hard
            # backward -> gradients through m_soft
            # ------------------------------------------------

            m = (
                m_hard
                -
                m_soft.detach()
                +
                m_soft
            )

        else:

            m = torch.zeros_like(
                logits
            )

            idx = torch.topk(
                logits,
                k=self.k_select,
                dim=1
            ).indices

            m.scatter_(
                1,
                idx,
                1.0
            )

        u = (
            m
            *
            u_cons
        )

        u = torch.clamp(
            u,
            -U_CLIP,
            U_CLIP
        )

        if squeeze_back:

            return (
                u.squeeze(0),
                m.squeeze(0)
            )

        return u, m


# ============================================================
# 10. COMMUNITY HELPERS
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


# ============================================================
# 11. RANDOM TRAINING DISRUPTION
# ============================================================

def build_random_perturbation():
    """
    Every trajectory:
      - choose one community
      - perturb ALL 20 nodes
      - choose one disruption family

    Training does NOT directly encode bridge information.
    """

    community = random.randrange(
        N_COMMUNITIES
    )

    sources = community_nodes(
        community
    )

    scenario = random.choice(
        [
            "drift",
            "cancel",
            "pulse",
        ]
    )

    t0 = random.uniform(
        23.0,
        29.0
    )

    if scenario == "drift":

        duration = random.uniform(
            8.0,
            12.0
        )

        tf = min(
            t0 + duration,
            43.0
        )

        pert = FreqDriftNoise(
            t0=t0,
            tf=tf,
            source_idxs=sources,
            drift=random.uniform(
                1.2,
                2.0
            ),
            sigma=random.uniform(
                0.5,
                1.5
            ),
        )

    elif scenario == "cancel":

        duration = random.uniform(
            8.0,
            12.0
        )

        tf = min(
            t0 + duration,
            43.0
        )

        pert = CancelDynamics(
            t0=t0,
            tf=tf,
            source_idxs=sources,
        )

    elif scenario == "pulse":

        duration = random.uniform(
            4.0,
            7.0
        )

        tf = min(
            t0 + duration,
            40.0
        )

        pert = GaussianPulse(
            t0=t0,
            tf=tf,
            source_idxs=sources,
            amp=random.uniform(
                15.0,
                25.0
            ),
            width=random.uniform(
                1.5,
                2.5
            ),
        )

    return (
        [pert],
        scenario,
        community,
    )


# ============================================================
# 12. RESILIENCE METRIC
# ============================================================

def differentiable_resilience(
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

    impact_idx = max(
        0,
        min(
            impact_idx,
            len(r) - 1
        )
    )

    recovery_idx = max(
        impact_idx + 1,
        min(
            recovery_idx,
            len(r) - 1
        )
    )

    r_during = r[
        impact_idx:recovery_idx
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

    recovery_error = torch.mean(
        (
            1.0 - r_post
        ) ** 2
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
# 13. MASK REGULARIZATION
# ============================================================

def temporal_smoothness(
    m,
):

    if m.shape[0] < 2:

        return torch.tensor(
            0.0,
            device=m.device
        )

    return torch.mean(
        (
            m[1:]
            -
            m[:-1]
        ) ** 2
    )


def spatial_entropy(
    m,
    eps=1e-8,
):

    p = (
        m
        /
        (
            m.sum(
                dim=-1,
                keepdim=True
            )
            +
            eps
        )
    )

    entropy = -(
        p
        *
        torch.log(
            p + eps
        )
    ).sum(
        dim=-1
    ).mean()

    return entropy


# ============================================================
# 14. V4 LOSS
# ============================================================

def calculate_loss(
    r_history,
    u_history,
    m_history,
    perturbations,
):

    r = torch.stack(
        r_history
    ).squeeze()

    u = torch.stack(
        u_history
    ).squeeze()

    m = torch.stack(
        m_history
    ).squeeze()

    resilience_values = []

    for pert in perturbations:

        resilience_values.append(
            differentiable_resilience(
                r,
                pert.t0,
                pert.tf
            )
        )

    resilience = torch.stack(
        resilience_values
    ).mean()

    # --------------------------------------------------------
    # Actual mean-squared control effort
    # --------------------------------------------------------

    energy = torch.mean(
        u ** 2
    )

    # --------------------------------------------------------
    # Normalize by maximum mean effort compatible with
    # rho=0.10 and saturation |u_i| <= U_CLIP.
    # --------------------------------------------------------

    energy_normalized = (
        energy
        /
        ENERGY_SCALE
    )

    mask_usage = torch.mean(
        m
    )

    sparsity = (
        mask_usage
        -
        TARGET_MASK
    ) ** 2

    smoothness = (
        temporal_smoothness(
            m
        )
    )

    entropy = (
        spatial_entropy(
            m
        )
    )

    # --------------------------------------------------------
    # V4 objective
    # --------------------------------------------------------

    loss = (
        LAMBDA_RESILIENCE
        *
        (
            1.0
            -
            resilience
        )

        +
        LAMBDA_ENERGY
        *
        energy_normalized

        +
        LAMBDA_SPARSITY
        *
        sparsity

        +
        LAMBDA_SMOOTHNESS
        *
        smoothness

        +
        LAMBDA_ENTROPY
        *
        entropy
    )

    metrics = {
        "resilience":
            resilience,

        "energy":
            energy,

        "energy_normalized":
            energy_normalized,

        "mask_usage":
            mask_usage,

        "sparsity":
            sparsity,

        "smoothness":
            smoothness,

        "entropy":
            entropy,
    }

    return (
        loss,
        metrics
    )


# ============================================================
# 15. MODEL
# ============================================================

model = NodeScoreNetwork(
    in_features=10,
    hidden_dim=64
).to(device)

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=LEARNING_RATE,
    weight_decay=1e-5
)

scheduler = (
    torch.optim.lr_scheduler
    .CosineAnnealingLR(
        optimizer,
        T_max=N_UPDATES,
        eta_min=MIN_LEARNING_RATE
    )
)

train_ctrl = NRPCV4Controller(
    model=model,
    mode=MODE,
    beta=1.0,
    k_select=K_SELECT,
    tau=TAU_START,
    soft_selection=True,
)

validation_ctrl = NRPCV4Controller(
    model=model,
    mode=MODE,
    beta=1.0,
    k_select=K_SELECT,
    tau=TAU_END,
    soft_selection=False,
)


# ============================================================
# 16. TEMPERATURE SCHEDULE
# ============================================================

def temperature_schedule(
    update,
):

    progress = (
        update
        /
        max(
            N_UPDATES - 1,
            1
        )
    )

    return (
        TAU_START
        *
        (
            TAU_END
            /
            TAU_START
        ) ** progress
    )


# ============================================================
# 17. INITIAL CONDITION
# ============================================================

def build_initial_state(
    rng,
):

    # System starts close to synchronization:
    # resilience experiment, not initial acquisition.
    common_phase = rng.uniform(
        0.0,
        2.0 * np.pi
    )

    theta_std = rng.uniform(
        0.08,
        0.18
    )

    theta0 = torch.tensor(
        common_phase
        +
        rng.normal(
            0.0,
            theta_std,
            size=N_AGENTS
        ),
        dtype=torch.float32,
        device=device
    )

    omega_std = rng.uniform(
        0.03,
        0.06
    )

    omegas = torch.tensor(
        rng.normal(
            1.0,
            omega_std,
            size=N_AGENTS
        ),
        dtype=torch.float32,
        device=device
    )

    return (
        theta0,
        omegas
    )


# ============================================================
# 18. FIXED VALIDATION CASES
# ============================================================

VALIDATION_CASES = [
    # seed, scenario, community, beta, K
    (30001, "drift",  0, 1.0,  0.50),
    (30002, "drift",  2, 5.0,  0.50),
    (30003, "cancel", 1, 5.0,  0.50),
    (30004, "cancel", 3, 10.0, 0.50),
    (30005, "pulse",  4, 1.0,  0.50),
    (30006, "pulse",  2, 10.0, 0.50),
]


def fixed_validation_perturbation(
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

    if scenario == "cancel":

        return CancelDynamics(
            t0=25.0,
            tf=35.0,
            source_idxs=sources,
        )

    if scenario == "pulse":

        return GaussianPulse(
            t0=25.0,
            tf=31.0,
            source_idxs=sources,
            amp=20.0,
            width=2.0,
        )

    raise ValueError(
        scenario
    )


# ============================================================
# 19. HARD TOP-K VALIDATION
# ============================================================

@torch.no_grad()
def validate_model():

    model.eval()

    res_values = []
    energy_values = []
    energy_norm_values = []

    for (
        seed,
        scenario,
        community,
        beta,
        k_coord
    ) in VALIDATION_CASES:

        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(
                seed
            )

        rng = np.random.default_rng(
            seed
        )

        theta0, omegas = (
            build_initial_state(
                rng
            )
        )

        pert = (
            fixed_validation_perturbation(
                scenario,
                community
            )
        )

        validation_ctrl.beta = beta

        r_hist, u_hist, _, _ = (
            simulate_trajectory_topology(
                validation_ctrl,
                theta0,
                omegas,
                [pert],
                False,
                A,
                STEPS,
                DT,
                10,
                10,
                N_AGENTS,
                k_coord,
            )
        )

        r = torch.stack(
            r_hist
        ).squeeze()

        u = torch.stack(
            u_hist
        ).squeeze()

        res = (
            differentiable_resilience(
                r,
                pert.t0,
                pert.tf
            )
        )

        energy = torch.mean(
            u ** 2
        )

        energy_norm = (
            energy
            /
            ENERGY_SCALE
        )

        res_values.append(
            res.item()
        )

        energy_values.append(
            energy.item()
        )

        energy_norm_values.append(
            energy_norm.item()
        )

    mean_res = float(
        np.mean(
            res_values
        )
    )

    mean_energy = float(
        np.mean(
            energy_values
        )
    )

    mean_energy_norm = float(
        np.mean(
            energy_norm_values
        )
    )

    # Same resilience-energy trade-off
    # used by the main training loss.
    score = (
        mean_res
        -
        LAMBDA_ENERGY
        *
        mean_energy_norm
    )

    model.train()

    return (
        score,
        mean_res,
        mean_energy,
        mean_energy_norm,
    )


# ============================================================
# 20. LOGS
# ============================================================

loss_log = []
resilience_log = []

energy_log = []
energy_norm_log = []

mask_log = []
sparsity_log = []
smoothness_log = []
entropy_log = []

beta_log = []
kcoord_log = []
tau_log = []

scenario_log = []
community_log = []

validation_log = []

best_validation_score = (
    -np.inf
)

best_state_dict = None
best_update = -1


# ============================================================
# 21. TRAINING
# ============================================================

print()
print(
    "[info] Starting "
    "NRPC-V4 modular training..."
)
print()


for update in range(
    N_UPDATES
):

    optimizer.zero_grad()

    tau = temperature_schedule(
        update
    )

    train_ctrl.tau = tau

    update_loss = []
    update_res = []

    update_energy = []
    update_energy_norm = []

    update_mask = []
    update_sparsity = []
    update_smooth = []
    update_entropy = []

    update_beta = []
    update_k = []

    update_scenarios = []
    update_communities = []

    last_m = None

    last_scenario = None
    last_community = None

    # ========================================================
    # Multiple trajectories per optimizer update
    # ========================================================

    for rollout in range(
        TRAJECTORIES_PER_UPDATE
    ):

        episode_seed = (
            1000
            +
            update
            *
            TRAJECTORIES_PER_UPDATE
            +
            rollout
        )

        torch.manual_seed(
            episode_seed
        )

        np.random.seed(
            episode_seed
        )

        random.seed(
            episode_seed
        )

        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(
                episode_seed
            )

        rng = np.random.default_rng(
            episode_seed
        )

        # ----------------------------------------------------
        # Log-uniform beta:
        # equal representation across low/high gains.
        # ----------------------------------------------------

        beta_epoch = math.exp(
            random.uniform(
                math.log(
                    BETA_MIN
                ),
                math.log(
                    BETA_MAX
                )
            )
        )

        train_ctrl.beta = (
            beta_epoch
        )

        k_coord_epoch = random.uniform(
            K_MIN,
            K_MAX
        )

        theta0, omegas = (
            build_initial_state(
                rng
            )
        )

        (
            perturbations,
            scenario,
            community,
        ) = build_random_perturbation()

        # ----------------------------------------------------
        # Differentiable trajectory
        # ----------------------------------------------------

        (
            r_hist,
            u_hist,
            _,
            m_hist
        ) = simulate_trajectory_topology(
            train_ctrl,
            theta0,
            omegas,
            perturbations,
            False,
            A,
            STEPS,
            DT,
            10,
            10,
            N_AGENTS,
            k_coord_epoch,
        )

        # ----------------------------------------------------
        # GLOBAL trajectory-level loss
        # ----------------------------------------------------

        loss, metrics = (
            calculate_loss(
                r_hist,
                u_hist,
                m_hist,
                perturbations,
            )
        )

        (
            loss
            /
            TRAJECTORIES_PER_UPDATE
        ).backward()

        update_loss.append(
            loss.item()
        )

        update_res.append(
            metrics[
                "resilience"
            ].item()
        )

        update_energy.append(
            metrics[
                "energy"
            ].item()
        )

        update_energy_norm.append(
            metrics[
                "energy_normalized"
            ].item()
        )

        update_mask.append(
            metrics[
                "mask_usage"
            ].item()
        )

        update_sparsity.append(
            metrics[
                "sparsity"
            ].item()
        )

        update_smooth.append(
            metrics[
                "smoothness"
            ].item()
        )

        update_entropy.append(
            metrics[
                "entropy"
            ].item()
        )

        update_beta.append(
            beta_epoch
        )

        update_k.append(
            k_coord_epoch
        )

        update_scenarios.append(
            scenario
        )

        update_communities.append(
            community
        )

        last_m = torch.stack(
            m_hist
        ).squeeze()

        last_scenario = scenario
        last_community = community

    # ========================================================
    # Gradient step
    # ========================================================

    grad_norm = (
        torch.nn.utils
        .clip_grad_norm_(
            model.parameters(),
            GRAD_CLIP
        )
    )

    optimizer.step()
    scheduler.step()

    # ========================================================
    # Means
    # ========================================================

    mean_loss = float(
        np.mean(
            update_loss
        )
    )

    mean_res = float(
        np.mean(
            update_res
        )
    )

    mean_energy = float(
        np.mean(
            update_energy
        )
    )

    mean_energy_norm = float(
        np.mean(
            update_energy_norm
        )
    )

    mean_mask = float(
        np.mean(
            update_mask
        )
    )

    mean_sparsity = float(
        np.mean(
            update_sparsity
        )
    )

    mean_smooth = float(
        np.mean(
            update_smooth
        )
    )

    mean_entropy = float(
        np.mean(
            update_entropy
        )
    )

    mean_beta = float(
        np.mean(
            update_beta
        )
    )

    mean_k = float(
        np.mean(
            update_k
        )
    )

    # ========================================================
    # Save logs
    # ========================================================

    loss_log.append(
        mean_loss
    )

    resilience_log.append(
        mean_res
    )

    energy_log.append(
        mean_energy
    )

    energy_norm_log.append(
        mean_energy_norm
    )

    mask_log.append(
        mean_mask
    )

    sparsity_log.append(
        mean_sparsity
    )

    smoothness_log.append(
        mean_smooth
    )

    entropy_log.append(
        mean_entropy
    )

    beta_log.append(
        mean_beta
    )

    kcoord_log.append(
        mean_k
    )

    tau_log.append(
        tau
    )

    scenario_log.append(
        update_scenarios
    )

    community_log.append(
        update_communities
    )

    # ========================================================
    # Diagnostics
    # ========================================================

    if (
        update % 10 == 0
        or update
        ==
        N_UPDATES - 1
    ):

        lr = optimizer.param_groups[
            0
        ]["lr"]

        if last_m is not None:

            mean_mask_per_node = (
                last_m
                .mean(dim=0)
            )

            top_nodes = (
                torch.topk(
                    mean_mask_per_node,
                    k=K_SELECT
                )
                .indices
                .detach()
                .cpu()
                .numpy()
                .tolist()
            )

            bridge_overlap = (
                len(
                    set(top_nodes)
                    &
                    set(BRIDGE_NODES)
                )
                /
                K_SELECT
            )

            disturbed_nodes = set(
                community_nodes(
                    last_community
                )
            )

            disturbed_overlap = (
                len(
                    set(top_nodes)
                    &
                    disturbed_nodes
                )
                /
                K_SELECT
            )

            mask_var = (
                last_m
                .var(dim=1)
                .mean()
                .item()
            )

            if (
                last_m.shape[0]
                >
                1
            ):

                delta_mask = (
                    torch.mean(
                        torch.abs(
                            last_m[1:]
                            -
                            last_m[:-1]
                        )
                    )
                    .item()
                )

            else:
                delta_mask = 0.0

        else:

            top_nodes = []
            bridge_overlap = np.nan
            disturbed_overlap = np.nan
            mask_var = np.nan
            delta_mask = np.nan

        print(
            f"[train] "
            f"Update={update:04d} | "
            f"Loss={mean_loss:.4f} | "
            f"Res={mean_res:.4f} | "
            f"E={mean_energy:.5f} | "
            f"En={mean_energy_norm:.4f} | "
            f"Mask={mean_mask:.4f} | "
            f"Beta={mean_beta:.2f} | "
            f"K={mean_k:.3f} | "
            f"Tau={tau:.3f} | "
            f"Grad={float(grad_norm):.3f} | "
            f"LR={lr:.2e}"
        )

        print(
            f"        "
            f"scenario={last_scenario} | "
            f"community={last_community} | "
            f"MaskVar={mask_var:.4f} | "
            f"DeltaMask={delta_mask:.4f}"
        )

        print(
            f"        "
            f"TopNodes={sorted(top_nodes)}"
        )

        print(
            f"        "
            f"BridgeOverlap="
            f"{bridge_overlap:.2f} | "
            f"DisturbedOverlap="
            f"{disturbed_overlap:.2f}"
        )

    # ========================================================
    # HARD TOP-K VALIDATION
    # ========================================================

    if (
        update % 25 == 0
        or update
        ==
        N_UPDATES - 1
    ):

        (
            val_score,
            val_res,
            val_energy,
            val_energy_norm,
        ) = validate_model()

        validation_log.append(
            {
                "update":
                    update,

                "score":
                    val_score,

                "resilience":
                    val_res,

                "energy":
                    val_energy,

                "energy_normalized":
                    val_energy_norm,
            }
        )

        print(
            f"[valid] "
            f"Update={update:04d} | "
            f"Score={val_score:.5f} | "
            f"Res={val_res:.4f} | "
            f"E={val_energy:.5f} | "
            f"En={val_energy_norm:.4f}"
        )

        if (
            val_score
            >
            best_validation_score
        ):

            best_validation_score = (
                val_score
            )

            best_update = update

            best_state_dict = (
                copy.deepcopy(
                    model.state_dict()
                )
            )

            print(
                f"[best] "
                f"New best model at "
                f"update {update}"
            )


# ============================================================
# 22. SAVE
# ============================================================

if best_state_dict is None:

    best_state_dict = (
        copy.deepcopy(
            model.state_dict()
        )
    )

    best_update = (
        N_UPDATES - 1
    )


checkpoint = {

    "version":
        "NRPC-V4-modular",

    "mode":
        MODE,

    "state_dict":
        best_state_dict,

    "final_state_dict":
        model.state_dict(),

    "seed":
        BASE_SEED,

    # --------------------------------------------------------
    # Topology
    # --------------------------------------------------------

    "adjacency":
        A_np,

    "n_agents":
        N_AGENTS,

    "n_communities":
        N_COMMUNITIES,

    "community_size":
        COMMUNITY_SIZE,

    "bridge_nodes":
        BRIDGE_NODES,

    "lambda2":
        lambda2,

    "clustering":
        clustering_np,

    # Diagnostic only
    "betweenness_top10":
        BETWEENNESS_TOP10,

    # --------------------------------------------------------
    # Dynamics
    # --------------------------------------------------------

    "dt":
        DT,

    "t_max":
        T_MAX,

    "k_coord_range":
        (
            K_MIN,
            K_MAX
        ),

    "beta_range":
        (
            BETA_MIN,
            BETA_MAX
        ),

    # --------------------------------------------------------
    # Actuation
    # --------------------------------------------------------

    "k_select":
        K_SELECT,

    "target_mask":
        TARGET_MASK,

    "u_clip":
        U_CLIP,

    "energy_scale":
        ENERGY_SCALE,

    # --------------------------------------------------------
    # Features
    # --------------------------------------------------------

    "in_features":
        10,

    "feature_definition": [
        "sin_global_error",
        "cos_global_error",
        "global_r",
        "local_sin_first_moment",
        "local_cos_first_moment",
        "local_r",
        "local_sin_second_moment",
        "local_cos_second_moment",
        "clustering_coefficient",
        "neighbor_mean_clustering",
    ],

    # --------------------------------------------------------
    # Training
    # --------------------------------------------------------

    "n_updates":
        N_UPDATES,

    "trajectories_per_update":
        TRAJECTORIES_PER_UPDATE,

    "tau_start":
        TAU_START,

    "tau_end":
        TAU_END,

    "loss_weights": {

        "resilience":
            LAMBDA_RESILIENCE,

        "energy":
            LAMBDA_ENERGY,

        "sparsity":
            LAMBDA_SPARSITY,

        "smoothness":
            LAMBDA_SMOOTHNESS,

        "entropy":
            LAMBDA_ENTROPY,
    },

    "best_update":
        best_update,

    "best_validation_score":
        best_validation_score,

    # --------------------------------------------------------
    # Logs
    # --------------------------------------------------------

    "loss":
        loss_log,

    "resilience":
        resilience_log,

    "energy":
        energy_log,

    "energy_normalized":
        energy_norm_log,

    "mask_usage":
        mask_log,

    "sparsity":
        sparsity_log,

    "smoothness":
        smoothness_log,

    "entropy":
        entropy_log,

    "beta_history":
        beta_log,

    "k_coord_history":
        kcoord_log,

    "tau_history":
        tau_log,

    "scenario_history":
        scenario_log,

    "community_history":
        community_log,

    "validation":
        validation_log,
}


torch.save(
    checkpoint,
    OUTPUT_MODEL
)


print()
print("=" * 76)
print("[done] NRPC-V4 modular training complete")
print(f"[done] Mode              : {MODE}")
print(f"[done] Best update       : {best_update}")
print(
    f"[done] Best val score   : "
    f"{best_validation_score:.5f}"
)
print(f"[done] Saved            : {OUTPUT_MODEL}")
print("=" * 76)
