#!/usr/bin/env python3
"""
NRPC-V4 structural node-failure activation experiment.

Purpose
-------
Evaluate the trained NRPC-GL and NRPC-L selectors under temporary
STRUCTURAL node failures on the 5-community modular topology.

The script is designed for the activation-frequency experiment:
    (a) localized failure around the C1-C2 interface,
    (b) the same failure pattern shifted to the C3-C4 interface,
    (c) severe multi-community failure of all inter-community bridge nodes.

During the failure interval:
    - failed nodes are disconnected from the adjacency matrix,
    - failed nodes are frozen (dtheta_i = 0),
    - failed nodes are removed from the EFFECTIVE Top-k candidate set,
    - an unconstrained raw Top-k preference is also stored diagnostically,
      allowing us to measure whether the network would have preferred
      unavailable nodes without an availability constraint.

The controller observes the CURRENT adjacency matrix. Thus local dynamic
features and local structural features reflect the temporary network failure.

Outputs
-------
1) Raw trial-level metrics CSV
2) Node-level activation-frequency CSV
3) Pickle with all aggregated arrays/metadata
4) NPZ with activation arrays for easy plotting

No figures are produced here. The plotting script can be built separately.
"""

from pathlib import Path
import argparse
import pickle
import random

import numpy as np
import pandas as pd
import torch

from kuramoto.controllers import RNMaskControllerNode

#exp3

# ============================================================
# 1. CLI
# ============================================================

parser = argparse.ArgumentParser()

parser.add_argument(
    "--trials",
    type=int,
    default=100,
)

parser.add_argument(
    "--beta",
    type=float,
    default=10.0,
)

parser.add_argument(
    "--base-seed",
    type=int,
    default=131000,
)

parser.add_argument(
    "--tag",
    type=str,
    default="final",
)

parser.add_argument(
    "--t0",
    type=float,
    default=20.0,
)

parser.add_argument(
    "--tf",
    type=float,
    default=40.0,
)

parser.add_argument(
    "--local-failures",
    type=int,
    default=10,
    help="Number of failed nodes in scenarios a and b.",
)

parser.add_argument(
    "--gl-checkpoint",
    type=str,
    default="results/data_train_v4_modular_global_local.pt",
)

parser.add_argument(
    "--local-checkpoint",
    type=str,
    default="results/data_train_v4_modular_local.pt",
)

args = parser.parse_args()


# ============================================================
# 2. SYSTEM CONFIGURATION
# ============================================================

device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

N_COMMUNITIES = 5
COMMUNITY_SIZE = 20
N_AGENTS = N_COMMUNITIES * COMMUNITY_SIZE

ROWS = 10
COLS = 10

T_MAX = 100.0
DT = 0.5
STEPS = int(T_MAX / DT)

K_COORD = 0.4

K_SELECT = 10
ALPHA = 10.0
U_CLIP = 2.0

BETA = args.beta

FAIL_T0 = args.t0
FAIL_TF = args.tf

RESULTS_DIR = Path("results") / "experiment3"
RESULTS_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# 3. REPRODUCIBILITY
# ============================================================

torch.manual_seed(
    args.base_seed
)

np.random.seed(
    args.base_seed
)

random.seed(
    args.base_seed
)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(
        args.base_seed
    )


# ============================================================
# 4. MODULAR DEGREE-PRESERVING TOPOLOGY
# ============================================================

def add_edge(
    A,
    i,
    j,
):
    A[i, j] = 1.0
    A[j, i] = 1.0


def remove_edge(
    A,
    i,
    j,
):
    A[i, j] = 0.0
    A[j, i] = 0.0


def build_modular_topology():
    """
    Five communities x 20 nodes.

    Each community starts as a degree-4 ring lattice:
        +/-1 and +/-2 neighbors.

    Then degree-preserving inter-community edge swaps connect
    neighboring communities in a periodic community ring.
    """

    A = np.zeros(
        (
            N_AGENTS,
            N_AGENTS,
        ),
        dtype=np.float32,
    )

    # --------------------------------------------------------
    # Internal degree-4 ring lattice per community
    # --------------------------------------------------------

    for c in range(
        N_COMMUNITIES
    ):

        offset = (
            c
            *
            COMMUNITY_SIZE
        )

        for local_i in range(
            COMMUNITY_SIZE
        ):

            i = (
                offset
                +
                local_i
            )

            for shift in [
                1,
                2,
            ]:

                j = (
                    offset
                    +
                    (
                        local_i
                        +
                        shift
                    )
                    %
                    COMMUNITY_SIZE
                )

                add_edge(
                    A,
                    i,
                    j,
                )

    # --------------------------------------------------------
    # Degree-preserving swaps
    #
    # interface c -> d=(c+1)%5:
    # remove:
    #   (c*20+4, c*20+5)
    #   (d*20+12, d*20+13)
    #
    # add:
    #   (c*20+4, d*20+12)
    #   (c*20+5, d*20+13)
    # --------------------------------------------------------

    for c in range(
        N_COMMUNITIES
    ):

        d = (
            c
            +
            1
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

        remove_edge(
            A,
            u1,
            u2,
        )

        remove_edge(
            A,
            v1,
            v2,
        )

        add_edge(
            A,
            u1,
            v1,
        )

        add_edge(
            A,
            u2,
            v2,
        )

    np.fill_diagonal(
        A,
        0.0,
    )

    return A


A_NOMINAL_NP = (
    build_modular_topology()
)

A_NOMINAL = torch.tensor(
    A_NOMINAL_NP,
    dtype=torch.float32,
    device=device,
)

degree_nominal = (
    A_NOMINAL_NP.sum(
        axis=1
    )
)

if not np.allclose(
    degree_nominal,
    4.0,
):
    raise RuntimeError(
        "Nominal topology is not degree 4."
    )


# ============================================================
# 5. BRIDGE / COMMUNITY HELPERS
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
            COMMUNITY_SIZE,
        )
    )


def interface_bridge_nodes(
    c,
):
    """
    Four bridge nodes associated with interface
    c -> (c+1) mod 5.
    """

    d = (
        c
        +
        1
    ) % N_COMMUNITIES

    return [
        c * COMMUNITY_SIZE + 4,
        c * COMMUNITY_SIZE + 5,
        d * COMMUNITY_SIZE + 12,
        d * COMMUNITY_SIZE + 13,
    ]


BRIDGE_NODES = sorted(
    {
        node
        for c
        in range(
            N_COMMUNITIES
        )
        for node
        in interface_bridge_nodes(
            c
        )
    }
)


def expand_local_failure(
    interface,
    n_fail,
):
    """
    Start from the four bridge nodes at one interface and expand
    deterministically through nominal one-hop neighbors within
    the two communities touching that interface.
    """

    c = interface
    d = (
        c
        +
        1
    ) % N_COMMUNITIES

    allowed = set(
        community_nodes(
            c
        )
        +
        community_nodes(
            d
        )
    )

    selected = []

    frontier = list(
        interface_bridge_nodes(
            interface
        )
    )

    while (
        frontier
        and
        len(
            selected
        )
        <
        n_fail
    ):

        node = frontier.pop(
            0
        )

        if (
            node
            not in selected
            and
            node in allowed
        ):
            selected.append(
                node
            )

        neigh = np.flatnonzero(
            A_NOMINAL_NP[
                node
            ]
            >
            0
        ).tolist()

        neigh = [
            j
            for j in sorted(
                neigh
            )
            if j in allowed
        ]

        for j in neigh:
            if (
                j
                not in selected
                and
                j
                not in frontier
            ):
                frontier.append(
                    j
                )

    if len(
        selected
    ) < n_fail:

        remaining = sorted(
            allowed
            -
            set(
                selected
            )
        )

        selected.extend(
            remaining[
                :
                n_fail
                -
                len(
                    selected
                )
            ]
        )

    return sorted(
        selected[
            :
            n_fail
        ]
    )


SCENARIOS = {
    # a) localized C1-C2 interface failure
    "a_interface_C1_C2":
        expand_local_failure(
            interface=0,
            n_fail=args.local_failures,
        ),

    # b) identical idea shifted to C3-C4
    "b_interface_C3_C4":
        expand_local_failure(
            interface=2,
            n_fail=args.local_failures,
        ),

    # c) severe structural disruption:
    # all 20 inter-community bridge nodes fail
    "c_all_bridge_nodes":
        BRIDGE_NODES.copy(),
}


# ============================================================
# 6. FAILURE ADJACENCY
# ============================================================

def adjacency_with_failed_nodes(
    failed_nodes,
):
    A = A_NOMINAL_NP.copy()

    failed = np.asarray(
        failed_nodes,
        dtype=int,
    )

    A[
        failed,
        :
    ] = 0.0

    A[
        :,
        failed
    ] = 0.0

    return torch.tensor(
        A,
        dtype=torch.float32,
        device=device,
    )


FAILED_ADJACENCY = {
    scenario:
        adjacency_with_failed_nodes(
            nodes
        )
    for (
        scenario,
        nodes
    )
    in SCENARIOS.items()
}


# ============================================================
# 7. STRUCTURAL FEATURES
# ============================================================

def clustering_coefficients(
    adjacency,
):
    """
    Undirected local clustering coefficient.

    Isolated and degree-1 nodes receive C_i = 0.
    """

    A = (
        adjacency
        .detach()
        .cpu()
        .numpy()
        >
        0
    ).astype(
        np.float32
    )

    N = A.shape[0]

    C = np.zeros(
        N,
        dtype=np.float32,
    )

    for i in range(
        N
    ):

        neigh = np.flatnonzero(
            A[
                i
            ]
        )

        k = len(
            neigh
        )

        if k < 2:
            C[i] = 0.0
            continue

        sub = A[
            np.ix_(
                neigh,
                neigh,
            )
        ]

        # each undirected edge appears twice
        e = (
            sub.sum()
            /
            2.0
        )

        C[i] = (
            2.0
            *
            e
            /
            (
                k
                *
                (
                    k
                    -
                    1
                )
            )
        )

    return torch.tensor(
        C,
        dtype=torch.float32,
        device=device,
    )


def mean_neighbor_clustering(
    adjacency,
    clustering,
):
    deg = adjacency.sum(
        dim=1
    )

    num = (
        adjacency
        @
        clustering.view(
            -1,
            1
        )
    ).squeeze(
        1
    )

    return (
        num
        /
        deg.clamp(
            min=1.0
        )
    )


def build_structural_cache(
    adjacency,
):
    C = clustering_coefficients(
        adjacency
    )

    C_neigh = (
        mean_neighbor_clustering(
            adjacency,
            C,
        )
    )

    return {
        "clustering":
            C,

        "neighbor_clustering":
            C_neigh,
    }


STRUCTURAL_CACHE = {
    "nominal":
        build_structural_cache(
            A_NOMINAL
        )
}

for (
    scenario,
    A_fail
) in FAILED_ADJACENCY.items():

    STRUCTURAL_CACHE[
        scenario
    ] = build_structural_cache(
        A_fail
    )


# ============================================================
# 8. V4 ROTATION-INVARIANT FEATURES
# ============================================================

def build_v4_features(
    theta,
    adjacency,
    structural_cache,
    mode,
):
    """
    10 features used by V4:

    Global dynamic (3)
      0 sin(theta_i - psi)
      1 cos(theta_i - psi)
      2 global r

    Local dynamic (5)
      3 mean_j sin(theta_j - theta_i)
      4 mean_j cos(theta_j - theta_i)
      5 magnitude of local first moment
      6 mean_j sin(2(theta_j - theta_i))
      7 mean_j cos(2(theta_j - theta_i))

    Local structural (2)
      8 local clustering coefficient
      9 mean-neighbor clustering

    In local mode, global channels 0:3 are zeroed.
    """

    if theta.dim() == 1:
        theta = theta.view(
            1,
            -1
        )

    B, N = theta.shape

    # --------------------------------------------------------
    # Global dynamic features
    # --------------------------------------------------------

    zbar = torch.mean(
        torch.exp(
            1j
            *
            theta
        ),
        dim=1,
        keepdim=True,
    )

    psi = torch.angle(
        zbar
    )

    r = torch.abs(
        zbar
    ).real

    global_sin = torch.sin(
        theta
        -
        psi
    )

    global_cos = torch.cos(
        theta
        -
        psi
    )

    global_r = r.repeat(
        1,
        N,
    )

    # --------------------------------------------------------
    # Relative local phase moments
    # --------------------------------------------------------

    deg = adjacency.sum(
        dim=1
    ).clamp(
        min=1.0
    )

    delta = (
        theta.unsqueeze(
            2
        )
        -
        theta.unsqueeze(
            1
        )
    )
    # delta[b, i, j] = theta_i - theta_j

    # We want theta_j - theta_i
    delta_ji = -delta

    A = adjacency.view(
        1,
        N,
        N,
    )

    local_sin_1 = (
        (
            torch.sin(
                delta_ji
            )
            *
            A
        ).sum(
            dim=2
        )
        /
        deg.view(
            1,
            N
        )
    )

    local_cos_1 = (
        (
            torch.cos(
                delta_ji
            )
            *
            A
        ).sum(
            dim=2
        )
        /
        deg.view(
            1,
            N
        )
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
                2.0
                *
                delta_ji
            )
            *
            A
        ).sum(
            dim=2
        )
        /
        deg.view(
            1,
            N
        )
    )

    local_cos_2 = (
        (
            torch.cos(
                2.0
                *
                delta_ji
            )
            *
            A
        ).sum(
            dim=2
        )
        /
        deg.view(
            1,
            N
        )
    )

    C = (
        structural_cache[
            "clustering"
        ]
        .view(
            1,
            N
        )
        .repeat(
            B,
            1
        )
    )

    C_neigh = (
        structural_cache[
            "neighbor_clustering"
        ]
        .view(
            1,
            N
        )
        .repeat(
            B,
            1
        )
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
            C,
            C_neigh,
        ],
        dim=-1,
    )

    if mode == "local":
        features = features.clone()
        features[
            ...,
            0:3
        ] = 0.0

    return features


# ============================================================
# 9. DEPLOYMENT CONTROLLER
# ============================================================

class NRPCV4FailureController:
    def __init__(
        self,
        neural_model,
        mode,
        beta,
    ):
        self.model = neural_model
        self.mode = mode
        self.beta = beta

    @torch.no_grad()
    def __call__(
        self,
        theta,
        adjacency,
        structural_cache,
        available_mask=None,
    ):
        """
        Returns:
            u              : physically applied control
            mask_raw       : unconstrained Top-k preference over all nodes
            mask_effective : Top-k restricted to currently available nodes

        This distinction is important under structural failures:
        failed nodes are physically unavailable and therefore must not
        consume the effective actuation budget. The raw mask is retained
        only as a diagnostic of where the network would have preferred
        to actuate without availability information.
        """

        if theta.dim() == 1:
            theta = theta.view(
                1,
                -1
            )

        zbar = torch.mean(
            torch.exp(
                1j
                *
                theta
            ),
            dim=1,
            keepdim=True,
        )

        r = torch.abs(
            zbar
        ).real

        psi = torch.angle(
            zbar
        )

        gate = torch.tanh(
            ALPHA
            *
            (
                1.0
                -
                r
            )
        )

        u_full = (
            -
            self.beta
            *
            gate
            *
            torch.sin(
                theta
                -
                psi
            )
        )

        features = build_v4_features(
            theta,
            adjacency,
            structural_cache,
            self.mode,
        )

        logits = self.model(
            features
        )

        # ----------------------------------------------------
        # 1) RAW preference: Top-k over all nodes
        # ----------------------------------------------------
        idx_raw = torch.topk(
            logits,
            k=K_SELECT,
            dim=1,
        ).indices

        mask_raw = torch.zeros_like(
            logits
        )

        mask_raw.scatter_(
            1,
            idx_raw,
            1.0,
        )

        # ----------------------------------------------------
        # 2) EFFECTIVE preference: Top-k over AVAILABLE nodes
        # ----------------------------------------------------
        if available_mask is None:
            available = torch.ones_like(
                logits,
                dtype=torch.bool,
            )
        else:
            if available_mask.dim() == 1:
                available = available_mask.view(
                    1,
                    -1
                ).bool()
            else:
                available = available_mask.bool()

            available = available.expand_as(
                logits
            )

        n_available = int(
            available[0].sum().item()
        )

        if n_available < K_SELECT:
            raise RuntimeError(
                f"Only {n_available} nodes are available, "
                f"but K_SELECT={K_SELECT}."
            )

        constrained_logits = logits.masked_fill(
            ~available,
            -torch.inf,
        )

        idx_effective = torch.topk(
            constrained_logits,
            k=K_SELECT,
            dim=1,
        ).indices

        mask_effective = torch.zeros_like(
            logits
        )

        mask_effective.scatter_(
            1,
            idx_effective,
            1.0,
        )

        # Physical control uses ONLY the effective available-node mask.
        u = (
            mask_effective
            *
            u_full
        )

        u = torch.clamp(
            u,
            -U_CLIP,
            U_CLIP,
        )

        return (
            u.squeeze(0),
            mask_raw.squeeze(0),
            mask_effective.squeeze(0),
        )


# ============================================================
# 10. LOAD TRAINED MODELS
# ============================================================

def load_model(
    checkpoint_path,
):
    checkpoint_path = Path(
        checkpoint_path
    )

    if not checkpoint_path.exists():
        raise FileNotFoundError(
            checkpoint_path
        )

    ckpt = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False,
    )

    model = RNMaskControllerNode(
        in_features=10,
        hidden_dim=64,
    ).to(
        device
    )

    state = ckpt[
        "state_dict"
    ]

    model.load_state_dict(
        state
    )

    model.eval()

    return (
        model,
        ckpt,
    )


model_gl, ckpt_gl = load_model(
    args.gl_checkpoint
)

model_local, ckpt_local = load_model(
    args.local_checkpoint
)

controllers = {
    "global_local":
        NRPCV4FailureController(
            model_gl,
            "global_local",
            BETA,
        ),

    "local":
        NRPCV4FailureController(
            model_local,
            "local",
            BETA,
        ),
}


# ============================================================
# 11. INITIAL CONDITION
# ============================================================

def build_initial_state(
    seed,
):
    rng = np.random.default_rng(
        seed
    )

    common_phase = rng.uniform(
        0.0,
        2.0
        *
        np.pi,
    )

    theta0 = (
        common_phase
        +
        rng.normal(
            0.0,
            0.12,
            size=N_AGENTS,
        )
    )

    omegas = rng.normal(
        1.0,
        0.05,
        size=N_AGENTS,
    )

    return (
        torch.tensor(
            theta0,
            dtype=torch.float32,
            device=device,
        ),
        torch.tensor(
            omegas,
            dtype=torch.float32,
            device=device,
        ),
    )


# ============================================================
# 12. RESILIENCE METRIC
# ============================================================

def resilience_score(
    r,
):
    impact_idx = int(
        FAIL_T0
        /
        DT
    )

    recovery_idx = int(
        FAIL_TF
        /
        DT
    )

    r_hit = r[
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
            r_post
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
# 13. STRUCTURAL FAILURE SIMULATOR
# ============================================================

@torch.no_grad()
def simulate_structural_failure(
    controller,
    mode,
    theta0,
    omegas,
    scenario,
):
    """
    Structural failure semantics:

    During [FAIL_T0, FAIL_TF):
      1. failed nodes are disconnected (rows/cols of A = 0),
      2. failed nodes are frozen (dtheta_i = 0),
      3. failed nodes cannot receive effective control,
      4. the selector still emits its raw top-k mask.

    The order parameter used for the main resilience score is
    computed over all N=100 nodes, so structural failure is not
    hidden by changing the denominator.
    """

    failed_nodes = SCENARIOS[
        scenario
    ]

    failed_idx = torch.tensor(
        failed_nodes,
        dtype=torch.long,
        device=device,
    )

    active_vector = torch.ones(
        N_AGENTS,
        dtype=torch.float32,
        device=device,
    )

    active_vector[
        failed_idx
    ] = 0.0

    curr_theta = theta0.clone().view(
        1,
        -1,
    )

    history_r = []
    history_r_active = []
    history_u = []
    history_mask_raw = []
    history_mask_effective = []
    history_theta = []

    for step in range(
        STEPS
    ):

        t = (
            step
            *
            DT
        )

        failure_active = (
            t
            >=
            FAIL_T0
            and
            t
            <
            FAIL_TF
        )

        if failure_active:
            adjacency = FAILED_ADJACENCY[
                scenario
            ]

            structural = STRUCTURAL_CACHE[
                scenario
            ]
        else:
            adjacency = A_NOMINAL

            structural = STRUCTURAL_CACHE[
                "nominal"
            ]

        # During the structural failure, failed nodes are not valid
        # actuation candidates. Outside the failure, all nodes are available.
        available_mask = (
            active_vector.bool()
            if failure_active
            else torch.ones(
                N_AGENTS,
                dtype=torch.bool,
                device=device,
            )
        )

        u_control, mask_raw, effective_mask = controller(
            curr_theta,
            adjacency,
            structural,
            available_mask=available_mask,
        )

        if u_control.dim() == 1:
            u_control = u_control.view(
                1,
                -1,
            )

        if mask_raw.dim() == 1:
            mask_raw = mask_raw.view(
                1,
                -1,
            )

        if effective_mask.dim() == 1:
            effective_mask = effective_mask.view(
                1,
                -1,
            )

        # ----------------------------------------------------
        # Kuramoto interaction under current topology
        # ----------------------------------------------------

        theta_i = curr_theta.unsqueeze(
            2
        )

        theta_j = curr_theta.unsqueeze(
            1
        )

        delta_theta = (
            theta_j
            -
            theta_i
        )

        interaction = (
            torch.sin(
                delta_theta
            )
            *
            adjacency.view(
                1,
                N_AGENTS,
                N_AGENTS,
            )
        ).sum(
            dim=2
        )

        dtheta = (
            omegas.view(
                1,
                -1
            )
            +
            (
                K_COORD
                /
                4.0
            )
            *
            interaction
            +
            u_control
        )

        # True temporary node deactivation:
        # failed oscillator states do not evolve.
        if failure_active:
            dtheta[
                :,
                failed_idx
            ] = 0.0

        curr_theta = (
            curr_theta
            +
            dtheta
            *
            DT
        )

        # ----------------------------------------------------
        # Synchronization metrics
        # ----------------------------------------------------

        r_all = torch.abs(
            torch.mean(
                torch.exp(
                    1j
                    *
                    curr_theta
                ),
                dim=1,
            )
        ).real

        if failure_active:
            theta_active = curr_theta[
                :,
                active_vector.bool()
            ]

            r_active = torch.abs(
                torch.mean(
                    torch.exp(
                        1j
                        *
                        theta_active
                    ),
                    dim=1,
                )
            ).real
        else:
            r_active = r_all

        history_r.append(
            r_all.squeeze(
                0
            )
        )

        history_r_active.append(
            r_active.squeeze(
                0
            )
        )

        history_u.append(
            u_control.squeeze(
                0
            )
        )

        history_mask_raw.append(
            mask_raw.squeeze(
                0
            )
        )

        history_mask_effective.append(
            effective_mask.squeeze(
                0
            )
        )

        history_theta.append(
            curr_theta.squeeze(
                0
            )
        )

    return {
        "r":
            torch.stack(
                history_r
            ),

        "r_active":
            torch.stack(
                history_r_active
            ),

        "u":
            torch.stack(
                history_u
            ),

        "mask_raw":
            torch.stack(
                history_mask_raw
            ),

        "mask_effective":
            torch.stack(
                history_mask_effective
            ),

        "theta":
            torch.stack(
                history_theta
            ),
    }


# ============================================================
# 14. ACCUMULATORS
# ============================================================

MODES = [
    "global_local",
    "local",
]

WINDOWS = [
    "all",
    "pre",
    "during",
    "post",
]

activation_sum_raw = {}

activation_sum_effective = {}

activation_count = {}

for mode in MODES:

    activation_sum_raw[
        mode
    ] = {}

    activation_sum_effective[
        mode
    ] = {}

    activation_count[
        mode
    ] = {}

    for scenario in SCENARIOS:

        activation_sum_raw[
            mode
        ][
            scenario
        ] = {
            window:
                np.zeros(
                    N_AGENTS,
                    dtype=np.float64,
                )
            for window
            in WINDOWS
        }

        activation_sum_effective[
            mode
        ][
            scenario
        ] = {
            window:
                np.zeros(
                    N_AGENTS,
                    dtype=np.float64,
                )
            for window
            in WINDOWS
        }

        activation_count[
            mode
        ][
            scenario
        ] = {
            window:
                0
            for window
            in WINDOWS
        }


raw_rows = []


# ============================================================
# 15. WINDOW HELPERS
# ============================================================

time_axis = (
    np.arange(
        STEPS
    )
    *
    DT
)

WINDOW_MASKS = {
    "all":
        np.ones(
            STEPS,
            dtype=bool,
        ),

    "pre":
        time_axis
        <
        FAIL_T0,

    "during":
        (
            time_axis
            >=
            FAIL_T0
        )
        &
        (
            time_axis
            <
            FAIL_TF
        ),

    "post":
        time_axis
        >=
        FAIL_TF,
}


# ============================================================
# 16. EVALUATION LOOP
# ============================================================

print("=" * 88)
print("NRPC-V4 | STRUCTURAL NODE-FAILURE ACTIVATION EVALUATION")
print("=" * 88)
print(f"Device                    : {device}")
print(f"N                         : {N_AGENTS}")
print(f"Communities               : {N_COMMUNITIES} x {COMMUNITY_SIZE}")
print(f"Nominal degree            : 4")
print(f"K                         : {K_COORD}")
print(f"beta                      : {BETA}")
print(f"Sparse budget             : {K_SELECT}/{N_AGENTS}")
print(f"Trials/scenario/mode      : {args.trials}")
print(f"Failure interval          : [{FAIL_T0}, {FAIL_TF})")
print(f"Base seed                 : {args.base_seed}")
print(f"Bridge nodes              : {BRIDGE_NODES}")
print("=" * 88)

for scenario, failed_nodes in SCENARIOS.items():

    print()
    print("-" * 88)
    print(
        f"{scenario} | "
        f"failed={len(failed_nodes)} nodes | "
        f"{failed_nodes}"
    )
    print("-" * 88)

    failed_set = set(
        failed_nodes
    )

    for trial in range(
        args.trials
    ):

        state_seed = (
            args.base_seed
            +
            trial
        )

        theta0, omegas = (
            build_initial_state(
                state_seed
            )
        )

        for mode in MODES:

            # Same state for GL and Local.
            torch.manual_seed(
                state_seed
            )

            np.random.seed(
                state_seed
            )

            out = simulate_structural_failure(
                controllers[
                    mode
                ],
                mode,
                theta0,
                omegas,
                scenario,
            )

            r = (
                out[
                    "r"
                ]
                .detach()
                .cpu()
                .numpy()
            )

            r_active = (
                out[
                    "r_active"
                ]
                .detach()
                .cpu()
                .numpy()
            )

            u = (
                out[
                    "u"
                ]
                .detach()
                .cpu()
                .numpy()
            )

            mask_raw = (
                out[
                    "mask_raw"
                ]
                .detach()
                .cpu()
                .numpy()
            )

            mask_effective = (
                out[
                    "mask_effective"
                ]
                .detach()
                .cpu()
                .numpy()
            )

            # ------------------------------------------------
            # Activation-frequency accumulation
            # ------------------------------------------------

            for window in WINDOWS:

                idx = WINDOW_MASKS[
                    window
                ]

                activation_sum_raw[
                    mode
                ][
                    scenario
                ][
                    window
                ] += mask_raw[
                    idx
                ].sum(
                    axis=0
                )

                activation_sum_effective[
                    mode
                ][
                    scenario
                ][
                    window
                ] += mask_effective[
                    idx
                ].sum(
                    axis=0
                )

                activation_count[
                    mode
                ][
                    scenario
                ][
                    window
                ] += int(
                    idx.sum()
                )

            # ------------------------------------------------
            # Trial metrics
            # ------------------------------------------------

            r_tensor = out[
                "r"
            ]

            resilience = float(
                resilience_score(
                    r_tensor
                ).item()
            )

            energy = float(
                np.mean(
                    u ** 2
                )
            )

            during_idx = WINDOW_MASKS[
                "during"
            ]

            post_idx = WINDOW_MASKS[
                "post"
            ]

            wasted_budget = float(
                mask_raw[
                    during_idx
                ][
                    :,
                    failed_nodes
                ].sum(
                    axis=1
                ).mean()
                /
                K_SELECT
            )

            effective_budget = float(
                mask_effective[
                    during_idx
                ].sum(
                    axis=1
                ).mean()
                /
                K_SELECT
            )

            bridge_hit = float(
                mask_effective[
                    during_idx
                ][
                    :,
                    BRIDGE_NODES
                ].sum(
                    axis=1
                ).mean()
                /
                K_SELECT
            )

            bridge_post = float(
                mask_effective[
                    post_idx
                ][
                    :,
                    BRIDGE_NODES
                ].sum(
                    axis=1
                ).mean()
                /
                K_SELECT
            )

            raw_rows.append(
                {
                    "scenario":
                        scenario,

                    "mode":
                        mode,

                    "trial":
                        trial,

                    "state_seed":
                        state_seed,

                    "beta":
                        BETA,

                    "failure_t0":
                        FAIL_T0,

                    "failure_tf":
                        FAIL_TF,

                    "n_failed":
                        len(
                            failed_nodes
                        ),

                    "failed_nodes":
                        ",".join(
                            map(
                                str,
                                failed_nodes
                            )
                        ),

                    "resilience":
                        resilience,

                    "energy":
                        energy,

                    "r_min":
                        float(
                            r.min()
                        ),

                    "r_final":
                        float(
                            r[
                                -1
                            ]
                        ),

                    "r_active_during_mean":
                        float(
                            r_active[
                                during_idx
                            ].mean()
                        ),

                    "wasted_budget_during":
                        wasted_budget,

                    "effective_budget_during":
                        effective_budget,

                    "bridge_fraction_during":
                        bridge_hit,

                    "bridge_fraction_post":
                        bridge_post,
                }
            )

        print(
            f" trial "
            f"{trial + 1:03d}/"
            f"{args.trials}",
            end="\r",
        )

    print()


# ============================================================
# 17. FINAL ACTIVATION FREQUENCIES
# ============================================================

activation_frequency_raw = {}

activation_frequency_effective = {}

for mode in MODES:

    activation_frequency_raw[
        mode
    ] = {}

    activation_frequency_effective[
        mode
    ] = {}

    for scenario in SCENARIOS:

        activation_frequency_raw[
            mode
        ][
            scenario
        ] = {}

        activation_frequency_effective[
            mode
        ][
            scenario
        ] = {}

        for window in WINDOWS:

            count = activation_count[
                mode
            ][
                scenario
            ][
                window
            ]

            activation_frequency_raw[
                mode
            ][
                scenario
            ][
                window
            ] = (
                activation_sum_raw[
                    mode
                ][
                    scenario
                ][
                    window
                ]
                /
                count
            )

            activation_frequency_effective[
                mode
            ][
                scenario
            ][
                window
            ] = (
                activation_sum_effective[
                    mode
                ][
                    scenario
                ][
                    window
                ]
                /
                count
            )


# ============================================================
# 18. NODE-LEVEL TABLE
# ============================================================

node_rows = []

bridge_set = set(
    BRIDGE_NODES
)

for mode in MODES:

    for (
        scenario,
        failed_nodes
    ) in SCENARIOS.items():

        failed_set = set(
            failed_nodes
        )

        for node in range(
            N_AGENTS
        ):

            node_rows.append(
                {
                    "mode":
                        mode,

                    "scenario":
                        scenario,

                    "node":
                        node,

                    "community":
                        node
                        //
                        COMMUNITY_SIZE,

                    "is_bridge":
                        int(
                            node
                            in
                            bridge_set
                        ),

                    "is_failed":
                        int(
                            node
                            in
                            failed_set
                        ),

                    "activation_raw_all":
                        activation_frequency_raw[
                            mode
                        ][
                            scenario
                        ][
                            "all"
                        ][
                            node
                        ],

                    "activation_raw_pre":
                        activation_frequency_raw[
                            mode
                        ][
                            scenario
                        ][
                            "pre"
                        ][
                            node
                        ],

                    "activation_raw_during":
                        activation_frequency_raw[
                            mode
                        ][
                            scenario
                        ][
                            "during"
                        ][
                            node
                        ],

                    "activation_raw_post":
                        activation_frequency_raw[
                            mode
                        ][
                            scenario
                        ][
                            "post"
                        ][
                            node
                        ],

                    "activation_effective_all":
                        activation_frequency_effective[
                            mode
                        ][
                            scenario
                        ][
                            "all"
                        ][
                            node
                        ],

                    "activation_effective_pre":
                        activation_frequency_effective[
                            mode
                        ][
                            scenario
                        ][
                            "pre"
                        ][
                            node
                        ],

                    "activation_effective_during":
                        activation_frequency_effective[
                            mode
                        ][
                            scenario
                        ][
                            "during"
                        ][
                            node
                        ],

                    "activation_effective_post":
                        activation_frequency_effective[
                            mode
                        ][
                            scenario
                        ][
                            "post"
                        ][
                            node
                        ],
                }
            )


# ============================================================
# 19. SAVE OUTPUTS
# ============================================================

beta_tag = str(
    BETA
).replace(
    ".",
    "p",
)

stem = (
    f"structural_activation_"
    f"{args.tag}_"
    f"beta{beta_tag}_"
    f"seed{args.base_seed}"
)

raw_csv = (
    RESULTS_DIR
    /
    f"{stem}_raw.csv"
)

nodes_csv = (
    RESULTS_DIR
    /
    f"{stem}_nodes.csv"
)

pkl_path = (
    RESULTS_DIR
    /
    f"{stem}.pkl"
)

npz_path = (
    RESULTS_DIR
    /
    f"{stem}.npz"
)

pd.DataFrame(
    raw_rows
).to_csv(
    raw_csv,
    index=False,
)

pd.DataFrame(
    node_rows
).to_csv(
    nodes_csv,
    index=False,
)


payload = {
    "configuration": {
        "beta":
            BETA,

        "trials":
            args.trials,

        "base_seed":
            args.base_seed,

        "T_MAX":
            T_MAX,

        "DT":
            DT,

        "K_COORD":
            K_COORD,

        "K_SELECT":
            K_SELECT,

        "failure_t0":
            FAIL_T0,

        "failure_tf":
            FAIL_TF,

        "scenarios":
            SCENARIOS,

        "bridge_nodes":
            BRIDGE_NODES,

        "gl_checkpoint":
            args.gl_checkpoint,

        "local_checkpoint":
            args.local_checkpoint,

        "gl_best_update":
            ckpt_gl.get(
                "best_update",
                None,
            ),

        "local_best_update":
            ckpt_local.get(
                "best_update",
                None,
            ),
    },

    "adjacency_nominal":
        A_NOMINAL_NP,

    "activation_frequency_raw":
        activation_frequency_raw,

    "activation_frequency_effective":
        activation_frequency_effective,

    "raw_rows":
        raw_rows,

    "node_rows":
        node_rows,
}

with open(
    pkl_path,
    "wb",
) as f:
    pickle.dump(
        payload,
        f,
    )


# Flat NPZ keys: convenient for plotting later.
npz_payload = {
    "adjacency_nominal":
        A_NOMINAL_NP,

    "bridge_nodes":
        np.asarray(
            BRIDGE_NODES,
            dtype=int,
        ),
}

for (
    scenario,
    failed_nodes
) in SCENARIOS.items():

    npz_payload[
        f"{scenario}__failed_nodes"
    ] = np.asarray(
        failed_nodes,
        dtype=int,
    )

    for mode in MODES:

        for window in WINDOWS:

            npz_payload[
                f"{mode}__{scenario}__raw__{window}"
            ] = (
                activation_frequency_raw[
                    mode
                ][
                    scenario
                ][
                    window
                ]
            )

            npz_payload[
                f"{mode}__{scenario}__effective__{window}"
            ] = (
                activation_frequency_effective[
                    mode
                ][
                    scenario
                ][
                    window
                ]
            )

np.savez(
    npz_path,
    **npz_payload,
)


# ============================================================
# 20. PRINT SUMMARY
# ============================================================

raw_df = pd.DataFrame(
    raw_rows
)

summary = (
    raw_df
    .groupby(
        [
            "scenario",
            "mode",
        ]
    )
    .agg(
        resilience_mean=(
            "resilience",
            "mean",
        ),
        resilience_std=(
            "resilience",
            "std",
        ),
        energy_mean=(
            "energy",
            "mean",
        ),
        r_min_mean=(
            "r_min",
            "mean",
        ),
        r_final_mean=(
            "r_final",
            "mean",
        ),
        wasted_budget_mean=(
            "wasted_budget_during",
            "mean",
        ),
        effective_budget_mean=(
            "effective_budget_during",
            "mean",
        ),
        bridge_during_mean=(
            "bridge_fraction_during",
            "mean",
        ),
        bridge_post_mean=(
            "bridge_fraction_post",
            "mean",
        ),
    )
    .reset_index()
)

summary_csv = (
    RESULTS_DIR
    /
    f"{stem}_summary.csv"
)

summary.to_csv(
    summary_csv,
    index=False,
)

print()
print("=" * 88)
print("STRUCTURAL FAILURE ACTIVATION SUMMARY")
print("=" * 88)

print(
    summary.to_string(
        index=False
    )
)

print()
print("[done] Structural activation evaluation complete")
print(f"[done] Pickle    : {pkl_path}")
print(f"[done] NPZ       : {npz_path}")
print(f"[done] Raw CSV   : {raw_csv}")
print(f"[done] Nodes CSV : {nodes_csv}")
print(f"[done] Summary   : {summary_csv}")
print("=" * 88)
