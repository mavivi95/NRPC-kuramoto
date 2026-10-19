#!/usr/bin/env python3
"""
Experiment IV: zero-shot generalization to unseen modular topologies.

The pretrained NRPC-GL and NRPC-L selectors are evaluated WITHOUT retraining
on a frozen bank of rewired degree-4 modular graphs.

Protocol
--------
- N = 100, five communities x 20 nodes
- K = 0.5, dt = 0.1, T = 60
- sparse actuation budget k = 10
- beta in {0.5, 1, 5, 10, 20}
- three sequential events per trajectory
- event starts: 10, 27, 44
- six permutations of Drift / Pulse / Cancel
- five balanced community patterns
- 20 unseen graphs per rewiring level
- 6 episodes per graph = 120 episodes per level
- the exact same state/noise/random-mask seeds are reused across topology levels
- nominal topology q=0 is evaluated with the same conditions for a matched reference

Methods
-------
nrpc_gl, nrpc_local, full, random, betweenness, uncontrolled

Primary metrics
---------------
- resilience_harmonic: harmonic mean of the three event-specific scores
- resilience_worst
- mean squared control effort E = mean(u^2)
- normalized_resilience:
      (rho_method - rho_uncontrolled) / (rho_full - rho_uncontrolled)
  computed within the SAME topology/episode/beta
- generalization_gap:
      rho_unseen - rho_nominal
  under matched state/noise/event conditions

Robustness / reproducibility
----------------------------
The raw CSV is checkpointed after every completed topology x episode x beta
block. Re-running the same command resumes automatically unless --no-resume
is supplied.

Expected topology bank
----------------------
results/experiment4/unseen_topology_bank.npz
"""

from __future__ import annotations

from pathlib import Path
import argparse
import itertools
import math
import pickle
import random
import time

import numpy as np
import pandas as pd
import torch

from kuramoto.controllers import RNMaskControllerNode

#exp4

# ============================================================
# 1. CLI
# ============================================================

parser = argparse.ArgumentParser()

parser.add_argument(
    "--bank",
    type=str,
    default="results/experiment4/unseen_topology_bank.npz",
)

parser.add_argument(
    "--betas",
    type=float,
    nargs="+",
    default=[0.5, 1.0, 5.0, 10.0, 20.0],
)

parser.add_argument(
    "--graphs-per-level",
    type=int,
    default=20,
    help="Use the first G graphs from each rewiring level.",
)

parser.add_argument(
    "--orders-per-graph",
    type=int,
    default=6,
    choices=[1, 2, 3, 4, 5, 6],
)

parser.add_argument(
    "--base-seed",
    type=int,
    default=91000,
    help="91000 reproduces the seed schedule used in Experiment II.",
)

parser.add_argument(
    "--tag",
    type=str,
    default="final",
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

parser.add_argument(
    "--outdir",
    type=str,
    default="results/experiment4",
)

parser.add_argument(
    "--no-nominal",
    action="store_true",
    help="Skip q=0 nominal reference. Not recommended for the final experiment.",
)

parser.add_argument(
    "--no-resume",
    action="store_true",
    help="Ignore an existing raw checkpoint and start over.",
)

parser.add_argument(
    "--smoke",
    action="store_true",
    help="Quick validation: 1 graph/level, 1 event order, beta=10 only.",
)

args = parser.parse_args()


# ============================================================
# 2. CONFIGURATION
# ============================================================

device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

N_COMMUNITIES = 5
COMMUNITY_SIZE = 20
N = N_COMMUNITIES * COMMUNITY_SIZE
DEGREE = 4

T_MAX = 100.0
DT = 0.1
STEPS = int(T_MAX / DT)

K_COORD = 0.5
K_SELECT = 10
ALPHA = 10.0
U_CLIP = 2.0
ENERGY_REFERENCE = 0.4

PHASE_STD = 0.12
OMEGA_MEAN = 1.0
OMEGA_STD = 0.05

EVENT_STARTS = [10.0, 27.0, 44.0]

EVENT_ORDERS = list(
    itertools.permutations(
        ["drift", "pulse", "cancel"]
    )
)

# Same five patterns used in Experiment II.
COMMUNITY_PATTERNS = [
    (0, 2, 4),
    (1, 3, 0),
    (2, 4, 1),
    (3, 0, 2),
    (4, 1, 3),
]

EVENT_DURATION = {
    "drift": 10.0,
    "pulse": 6.0,
    "cancel": 10.0,
}

DRIFT_MAGNITUDE = 1.6
DRIFT_SIGMA = 1.0

PULSE_AMPLITUDE = 20.0
PULSE_WIDTH = 2.0

METHODS = [
    "nrpc_gl",
    "nrpc_local",
    "full",
    "random",
    "betweenness",
    "uncontrolled",
]


OUTDIR = Path(args.outdir)
OUTDIR.mkdir(
    parents=True,
    exist_ok=True,
)

# Use the explicit tag even in smoke mode.
# If the user did not provide a custom smoke tag, keep the convenient "smoke" name.
if args.smoke and args.tag == "final":
    suffix = "smoke"
else:
    suffix = args.tag

# Encode the simulation horizon in every output name so runs with different
# T_MAX values can never silently resume from one another.
t_token = f"T{T_MAX:g}".replace(".", "p")
run_id = f"{suffix}_{t_token}"

RAW_PATH = (
    OUTDIR
    /
    f"eval_v4_unseen_topologies_{run_id}_seed{args.base_seed}_raw.csv"
)

ENRICHED_PATH = (
    OUTDIR
    /
    f"eval_v4_unseen_topologies_{run_id}_seed{args.base_seed}_enriched.csv"
)

GRAPH_PATH = (
    OUTDIR
    /
    f"eval_v4_unseen_topologies_{run_id}_seed{args.base_seed}_graphlevel.csv"
)

SUMMARY_PATH = (
    OUTDIR
    /
    f"eval_v4_unseen_topologies_{run_id}_seed{args.base_seed}_summary.csv"
)

PKL_PATH = (
    OUTDIR
    /
    f"eval_v4_unseen_topologies_{run_id}_seed{args.base_seed}.pkl"
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
# 4. LOAD TOPOLOGY BANK
# ============================================================

bank_path = Path(args.bank)

if not bank_path.exists():
    raise FileNotFoundError(
        bank_path
    )

bank = np.load(
    bank_path,
    allow_pickle=True,
)

A_NOMINAL_NP = np.asarray(
    bank["adjacency_nominal"],
    dtype=np.float32,
)

A_BANK_NP = np.asarray(
    bank["adjacency_bank"],
    dtype=np.float32,
)

TARGET_LEVELS = np.asarray(
    bank["target_rewiring"],
    dtype=float,
)

ACTUAL_REWIRING = np.asarray(
    bank["actual_rewiring"],
    dtype=float,
)

LAMBDA2_BANK = np.asarray(
    bank["lambda2"],
    dtype=float,
)

CLUSTERING_BANK = np.asarray(
    bank["mean_clustering"],
    dtype=float,
)

MODULARITY_BANK = np.asarray(
    bank["modularity"],
    dtype=float,
)

BRIDGE_COUNT_BANK = np.asarray(
    bank["bridge_node_count"],
    dtype=int,
)

if A_NOMINAL_NP.shape != (N, N):
    raise RuntimeError(
        f"Unexpected nominal adjacency shape: {A_NOMINAL_NP.shape}"
    )

if A_BANK_NP.ndim != 4:
    raise RuntimeError(
        f"Unexpected adjacency bank shape: {A_BANK_NP.shape}"
    )

AVAILABLE_GRAPHS = A_BANK_NP.shape[1]

if args.smoke:
    GRAPH_COUNT = 1
    ORDERS_PER_GRAPH = 1
    BETAS = [10.0]
else:
    GRAPH_COUNT = min(
        int(args.graphs_per_level),
        AVAILABLE_GRAPHS,
    )
    ORDERS_PER_GRAPH = int(
        args.orders_per_graph
    )
    BETAS = [
        float(x)
        for x in args.betas
    ]


# ============================================================
# 5. TOPOLOGY HELPERS
# ============================================================

def adjacency_edges(
    A,
):
    ii, jj = np.where(
        np.triu(
            A,
            k=1,
        )
        >
        0.5
    )

    return [
        (
            int(i),
            int(j),
        )
        for i, j in zip(
            ii,
            jj,
        )
    ]


def graph_lambda2(
    A,
):
    L = (
        np.diag(
            A.sum(
                axis=1
            )
        )
        -
        A
    )

    values = np.linalg.eigvalsh(
        L
    )

    values = np.sort(
        np.real(
            values
        )
    )

    return float(
        values[1]
    )


def mean_clustering_np(
    A,
):
    values = []

    for i in range(N):
        nbr = np.where(
            A[i]
            >
            0.5
        )[0]

        k = len(
            nbr
        )

        if k < 2:
            values.append(
                0.0
            )
            continue

        sub = A[
            np.ix_(
                nbr,
                nbr,
            )
        ]

        values.append(
            float(
                sub.sum()
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
        )

    return float(
        np.mean(
            values
        )
    )


def fixed_partition_modularity(
    A,
):
    community = np.repeat(
        np.arange(
            N_COMMUNITIES
        ),
        COMMUNITY_SIZE,
    )

    k = A.sum(
        axis=1
    )

    m2 = float(
        A.sum()
    )

    Q = 0.0

    for c in range(
        N_COMMUNITIES
    ):
        idx = np.where(
            community
            ==
            c
        )[0]

        Q += (
            A[
                np.ix_(
                    idx,
                    idx,
                )
            ].sum()
            -
            np.outer(
                k[idx],
                k[idx],
            ).sum()
            /
            m2
        )

    return float(
        Q
        /
        m2
    )


def bridge_nodes_from_adjacency(
    A,
):
    bridge = set()

    for i, j in adjacency_edges(
        A
    ):
        ci = (
            i
            //
            COMMUNITY_SIZE
        )

        cj = (
            j
            //
            COMMUNITY_SIZE
        )

        if ci != cj:
            bridge.add(
                i
            )
            bridge.add(
                j
            )

    return sorted(
        bridge
    )


def brandes_betweenness(
    A,
):
    """
    Unnormalized Brandes betweenness centrality for an undirected graph.
    Only the ranking is used.
    """
    neighbors = [
        np.where(
            A[i]
            >
            0.5
        )[0].tolist()
        for i in range(N)
    ]

    C = np.zeros(
        N,
        dtype=float,
    )

    for s in range(N):
        stack = []

        predecessors = [
            []
            for _ in range(N)
        ]

        sigma = np.zeros(
            N,
            dtype=float,
        )

        sigma[s] = 1.0

        distance = np.full(
            N,
            -1,
            dtype=int,
        )

        distance[s] = 0

        queue = [
            s
        ]

        head = 0

        while head < len(
            queue
        ):
            v = queue[
                head
            ]

            head += 1

            stack.append(
                v
            )

            for w in neighbors[
                v
            ]:
                if distance[w] < 0:
                    queue.append(
                        w
                    )
                    distance[w] = (
                        distance[v]
                        +
                        1
                    )

                if (
                    distance[w]
                    ==
                    distance[v]
                    +
                    1
                ):
                    sigma[w] += sigma[v]
                    predecessors[w].append(
                        v
                    )

        dependency = np.zeros(
            N,
            dtype=float,
        )

        while stack:
            w = stack.pop()

            if sigma[w] > 0:
                coeff = (
                    1.0
                    +
                    dependency[w]
                ) / sigma[w]

                for v in predecessors[
                    w
                ]:
                    dependency[v] += (
                        sigma[v]
                        *
                        coeff
                    )

            if w != s:
                C[w] += dependency[w]

    # Undirected graph: paths are counted twice.
    C *= 0.5

    return C


# ============================================================
# 6. STRUCTURAL FEATURES
# ============================================================

def clustering_coefficients(
    adjacency,
):
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

    C = np.zeros(
        N,
        dtype=np.float32,
    )

    for i in range(N):
        nbr = np.flatnonzero(
            A[i]
        )

        k = len(
            nbr
        )

        if k < 2:
            continue

        sub = A[
            np.ix_(
                nbr,
                nbr,
            )
        ]

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


def build_structural_cache(
    adjacency,
):
    C = clustering_coefficients(
        adjacency
    )

    degree = adjacency.sum(
        dim=1
    ).clamp(
        min=1.0
    )

    C_neigh = (
        adjacency
        @
        C.view(
            -1,
            1
        )
    ).squeeze(
        1
    ) / degree

    return {
        "clustering":
            C,

        "neighbor_clustering":
            C_neigh,
    }


def build_v4_features(
    theta,
    adjacency,
    structural_cache,
    mode,
):
    """
    V4 feature vector:
      Global dynamic:
        0 sin(theta_i - psi)
        1 cos(theta_i - psi)
        2 global r
      Local dynamic:
        3 local mean sin(theta_j-theta_i)
        4 local mean cos(theta_j-theta_i)
        5 local first-moment magnitude
        6 local mean sin(2(theta_j-theta_i))
        7 local mean cos(2(theta_j-theta_i))
      Local structural:
        8 local clustering coefficient
        9 mean-neighbor clustering

    NRPC-L zeros channels 0:3.
    """

    if theta.dim() == 1:
        theta = theta.view(
            1,
            -1
        )

    B = theta.shape[0]

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

    degree = adjacency.sum(
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
        degree.view(
            1,
            N,
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
        degree.view(
            1,
            N,
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
        degree.view(
            1,
            N,
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
        degree.view(
            1,
            N,
        )
    )

    C = (
        structural_cache[
            "clustering"
        ]
        .view(
            1,
            N,
        )
        .repeat(
            B,
            1,
        )
    )

    C_neigh = (
        structural_cache[
            "neighbor_clustering"
        ]
        .view(
            1,
            N,
        )
        .repeat(
            B,
            1,
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
# 7. CHECKPOINTS / LEARNED SELECTORS
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

    model.load_state_dict(
        ckpt["state_dict"]
    )

    model.eval()

    return (
        model,
        ckpt,
    )


MODEL_GL, CKPT_GL = load_model(
    args.gl_checkpoint
)

MODEL_LOCAL, CKPT_LOCAL = load_model(
    args.local_checkpoint
)


@torch.no_grad()
def base_control(
    theta,
    beta,
):
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

    return (
        -
        beta
        *
        gate
        *
        torch.sin(
            theta
            -
            psi
        )
    )


@torch.no_grad()
def learned_mask(
    theta,
    adjacency,
    structural_cache,
    model,
    mode,
):
    features = build_v4_features(
        theta,
        adjacency,
        structural_cache,
        mode,
    )

    logits = model(
        features
    )

    idx = torch.topk(
        logits,
        k=K_SELECT,
        dim=1,
    ).indices

    mask = torch.zeros_like(
        logits
    )

    mask.scatter_(
        1,
        idx,
        1.0,
    )

    return mask


# ============================================================
# 8. INITIAL CONDITIONS / EVENT SCHEDULE
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
            PHASE_STD,
            size=N,
        )
    )

    omegas = rng.normal(
        OMEGA_MEAN,
        OMEGA_STD,
        size=N,
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


def community_nodes(
    c,
):
    start = (
        c
        *
        COMMUNITY_SIZE
    )

    return np.arange(
        start,
        start
        +
        COMMUNITY_SIZE,
        dtype=int,
    )


def episode_design(
    graph_index,
    order_index,
):
    """
    Reproduces Experiment II's 120-trial schedule when:
        graph_index = trial // 6
        order_index = trial % 6

    trial_id = 0..119
    order     = trial_id % 6
    pattern   = trial_id % 5
    """

    trial_id = (
        graph_index
        *
        6
        +
        order_index
    )

    order = EVENT_ORDERS[
        trial_id
        %
        len(
            EVENT_ORDERS
        )
    ]

    communities = COMMUNITY_PATTERNS[
        trial_id
        %
        len(
            COMMUNITY_PATTERNS
        )
    ]

    state_seed = (
        args.base_seed
        +
        trial_id
    )

    random_seed = (
        args.base_seed
        +
        100000
        +
        trial_id
    )

    noise_seed = (
        args.base_seed
        +
        200000
        +
        trial_id
    )

    rng_random = np.random.default_rng(
        random_seed
    )

    random_nodes = np.sort(
        rng_random.choice(
            N,
            size=K_SELECT,
            replace=False,
        )
    )

    return {
        "trial":
            trial_id,

        "order":
            tuple(
                order
            ),

        "communities":
            tuple(
                communities
            ),

        "state_seed":
            state_seed,

        "random_seed":
            random_seed,

        "noise_seed":
            noise_seed,

        "random_nodes":
            random_nodes,
    }


def build_noise_schedule(
    seed,
):
    rng = np.random.default_rng(
        seed
    )

    return torch.tensor(
        rng.normal(
            0.0,
            1.0,
            size=(
                STEPS,
                N,
            ),
        ),
        dtype=torch.float32,
        device=device,
    )


def event_records(
    order,
    communities,
):
    records = []

    for event_index in range(
        3
    ):
        event_type = order[
            event_index
        ]

        start = EVENT_STARTS[
            event_index
        ]

        duration = EVENT_DURATION[
            event_type
        ]

        end = (
            start
            +
            duration
        )

        community = communities[
            event_index
        ]

        records.append(
            {
                "type":
                    event_type,

                "start":
                    start,

                "end":
                    end,

                "community":
                    int(
                        community
                    ),

                "nodes":
                    community_nodes(
                        int(
                            community
                        )
                    ),
            }
        )

    return records


# ============================================================
# 9. MULTI-EVENT RESILIENCE
# ============================================================

def event_resilience_score(
    r,
    start,
    end,
    recovery_end,
):
    """
    Event-local score used in Experiment II:
      absorption = 0.5 * min(r_hit) + 0.5 * mean(r_hit)
      recovery   = 1 - mean((1-r_post)^2)
      score      = absorption * recovery

    Recovery is truncated before the next event.
    """

    start_idx = int(
        round(
            start
            /
            DT
        )
    )

    end_idx = int(
        round(
            end
            /
            DT
        )
    )

    recovery_end_idx = int(
        round(
            recovery_end
            /
            DT
        )
    )

    hit = r[
        start_idx:
        end_idx
    ]

    post = r[
        end_idx:
        recovery_end_idx
    ]

    if hit.numel() == 0:
        raise RuntimeError(
            "Empty disruption window."
        )

    absorption = (
        0.5
        *
        torch.min(
            hit
        )
        +
        0.5
        *
        torch.mean(
            hit
        )
    )

    if post.numel() > 0:
        recovery_error = torch.mean(
            (
                1.0
                -
                post
            ) ** 2
        )
    else:
        recovery_error = torch.tensor(
            0.0,
            device=r.device,
            dtype=r.dtype,
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

    return float(
        torch.clamp(
            score,
            0.0,
            1.0,
        ).item()
    )


def multi_event_metrics(
    r,
    events,
):
    scores = []

    for event_index, event in enumerate(
        events
    ):
        if event_index < 2:
            recovery_end = events[
                event_index
                +
                1
            ][
                "start"
            ]
        else:
            recovery_end = T_MAX

        scores.append(
            event_resilience_score(
                r,
                event["start"],
                event["end"],
                recovery_end,
            )
        )

    scores_np = np.asarray(
        scores,
        dtype=float,
    )

    harmonic = float(
        len(
            scores_np
        )
        /
        np.sum(
            1.0
            /
            np.clip(
                scores_np,
                1e-12,
                None,
            )
        )
    )

    return {
        "resilience_mean":
            float(
                scores_np.mean()
            ),

        "resilience_worst":
            float(
                scores_np.min()
            ),

        "resilience_harmonic":
            harmonic,

        "event1_resilience":
            float(
                scores_np[0]
            ),

        "event2_resilience":
            float(
                scores_np[1]
            ),

        "event3_resilience":
            float(
                scores_np[2]
            ),
    }


# ============================================================
# 10. SIMULATOR
# ============================================================

@torch.no_grad()
def simulate(
    method,
    beta,
    theta0,
    omegas,
    adjacency,
    structural_cache,
    events,
    noise_schedule,
    random_nodes,
    betweenness_nodes,
):
    theta = theta0.clone().view(
        1,
        -1,
    )

    fixed_random_mask = torch.zeros(
        (
            1,
            N,
        ),
        dtype=torch.float32,
        device=device,
    )

    fixed_random_mask[
        0,
        torch.tensor(
            random_nodes,
            dtype=torch.long,
            device=device,
        )
    ] = 1.0

    fixed_betweenness_mask = torch.zeros(
        (
            1,
            N,
        ),
        dtype=torch.float32,
        device=device,
    )

    fixed_betweenness_mask[
        0,
        torch.tensor(
            betweenness_nodes,
            dtype=torch.long,
            device=device,
        )
    ] = 1.0

    r_history = []
    u_sq_history = []

    for step in range(
        STEPS
    ):
        t = (
            step
            *
            DT
        )

        # ----------------------------------------------------
        # Control
        # ----------------------------------------------------

        if method == "uncontrolled":
            u = torch.zeros(
                (
                    1,
                    N,
                ),
                dtype=torch.float32,
                device=device,
            )

        else:
            u_base = base_control(
                theta,
                beta,
            )

            if method == "full":
                mask = torch.ones_like(
                    u_base
                )

            elif method == "random":
                mask = fixed_random_mask

            elif method == "betweenness":
                mask = fixed_betweenness_mask

            elif method == "nrpc_gl":
                mask = learned_mask(
                    theta,
                    adjacency,
                    structural_cache,
                    MODEL_GL,
                    "global_local",
                )

            elif method == "nrpc_local":
                mask = learned_mask(
                    theta,
                    adjacency,
                    structural_cache,
                    MODEL_LOCAL,
                    "local",
                )

            else:
                raise ValueError(
                    method
                )

            u = (
                mask
                *
                u_base
            )

            u = torch.clamp(
                u,
                -U_CLIP,
                U_CLIP,
            )

        # ----------------------------------------------------
        # Kuramoto interaction
        # ----------------------------------------------------

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

        interaction = (
            torch.sin(
                delta
            )
            *
            adjacency.view(
                1,
                N,
                N,
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
                DEGREE
            )
            *
            interaction
            +
            u
        )

        # ----------------------------------------------------
        # Scheduled disruption
        # ----------------------------------------------------

        for event in events:
            if (
                t
                >=
                event["start"]
                and
                t
                <
                event["end"]
            ):
                nodes = torch.tensor(
                    event["nodes"],
                    dtype=torch.long,
                    device=device,
                )

                if event["type"] == "drift":
                    dtheta[
                        :,
                        nodes
                    ] += (
                        DRIFT_MAGNITUDE
                        +
                        DRIFT_SIGMA
                        *
                        noise_schedule[
                            step,
                            nodes
                        ].view(
                            1,
                            -1,
                        )
                    )

                elif event["type"] == "pulse":
                    pulse = (
                        PULSE_AMPLITUDE
                        *
                        math.exp(
                            -
                            (
                                t
                                -
                                event["start"]
                            ) ** 2
                            /
                            (
                                2.0
                                *
                                PULSE_WIDTH ** 2
                            )
                        )
                    )

                    dtheta[
                        :,
                        nodes
                    ] += pulse

                elif event["type"] == "cancel":
                    dtheta[
                        :,
                        nodes
                    ] += (
                        -
                        omegas[
                            nodes
                        ].view(
                            1,
                            -1,
                        )
                        -
                        (
                            K_COORD
                            /
                            DEGREE
                        )
                        *
                        interaction[
                            :,
                            nodes
                        ]
                    )

                else:
                    raise ValueError(
                        event["type"]
                    )

        theta = (
            theta
            +
            DT
            *
            dtheta
        )

        r = torch.abs(
            torch.mean(
                torch.exp(
                    1j
                    *
                    theta
                ),
                dim=1,
            )
        ).real

        r_history.append(
            r.squeeze(
                0
            )
        )

        u_sq_history.append(
            torch.mean(
                u ** 2
            )
        )

    r = torch.stack(
        r_history
    )

    mean_squared_effort = float(
        torch.stack(
            u_sq_history
        ).mean().item()
    )

    metrics = multi_event_metrics(
        r,
        events,
    )

    metrics.update(
        {
            "energy":
                mean_squared_effort,

            "energy_normalized":
                mean_squared_effort
                /
                ENERGY_REFERENCE,

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
    )

    return metrics


# ============================================================
# 11. TOPOLOGY RECORDS
# ============================================================

def topology_records():
    records = []

    include_nominal = (
        not args.no_nominal
    )

    if include_nominal:
        nominal_lambda2 = graph_lambda2(
            A_NOMINAL_NP
        )

        nominal_clustering = mean_clustering_np(
            A_NOMINAL_NP
        )

        nominal_modularity = fixed_partition_modularity(
            A_NOMINAL_NP
        )

        nominal_bridges = len(
            bridge_nodes_from_adjacency(
                A_NOMINAL_NP
            )
        )

        # Repeat the same nominal graph over graph_index=0..G-1 so the
        # exact matched conditions can be compared against each unseen graph.
        for graph_index in range(
            GRAPH_COUNT
        ):
            records.append(
                {
                    "topology_label":
                        "q0_nominal",

                    "level_index":
                        -1,

                    "graph_index":
                        graph_index,

                    "target_rewiring":
                        0.0,

                    "actual_rewiring":
                        0.0,

                    "lambda2":
                        nominal_lambda2,

                    "mean_clustering":
                        nominal_clustering,

                    "modularity":
                        nominal_modularity,

                    "bridge_node_count":
                        nominal_bridges,

                    "adjacency_np":
                        A_NOMINAL_NP,
                }
            )

    for level_index, target in enumerate(
        TARGET_LEVELS
    ):
        for graph_index in range(
            GRAPH_COUNT
        ):
            records.append(
                {
                    "topology_label":
                        f"q{int(round(100*target)):02d}",

                    "level_index":
                        int(
                            level_index
                        ),

                    "graph_index":
                        int(
                            graph_index
                        ),

                    "target_rewiring":
                        float(
                            target
                        ),

                    "actual_rewiring":
                        float(
                            ACTUAL_REWIRING[
                                level_index,
                                graph_index,
                            ]
                        ),

                    "lambda2":
                        float(
                            LAMBDA2_BANK[
                                level_index,
                                graph_index,
                            ]
                        ),

                    "mean_clustering":
                        float(
                            CLUSTERING_BANK[
                                level_index,
                                graph_index,
                            ]
                        ),

                    "modularity":
                        float(
                            MODULARITY_BANK[
                                level_index,
                                graph_index,
                            ]
                        ),

                    "bridge_node_count":
                        int(
                            BRIDGE_COUNT_BANK[
                                level_index,
                                graph_index,
                            ]
                        ),

                    "adjacency_np":
                        A_BANK_NP[
                            level_index,
                            graph_index,
                        ],
                }
            )

    return records


# ============================================================
# 12. RESUME / CHECKPOINT HELPERS
# ============================================================

KEY_COLUMNS = [
    "topology_label",
    "graph_index",
    "order_index",
    "beta",
    "method",
]


def normalize_key_value(
    column,
    value,
):
    if column in {
        "graph_index",
        "order_index",
    }:
        return int(
            value
        )

    if column == "beta":
        return float(
            value
        )

    return str(
        value
    )


def row_key(
    row,
):
    return tuple(
        normalize_key_value(
            col,
            row[col],
        )
        for col in KEY_COLUMNS
    )


if (
    RAW_PATH.exists()
    and
    not args.no_resume
):
    existing = pd.read_csv(
        RAW_PATH
    )

    raw_rows = existing.to_dict(
        orient="records"
    )

    done = {
        row_key(
            row
        )
        for row
        in raw_rows
    }

    print(
        f"[resume] Loaded {len(raw_rows)} existing rows from {RAW_PATH}"
    )

else:
    raw_rows = []
    done = set()

    if (
        RAW_PATH.exists()
        and
        args.no_resume
    ):
        RAW_PATH.unlink()


def save_raw_checkpoint():
    pd.DataFrame(
        raw_rows
    ).to_csv(
        RAW_PATH,
        index=False,
    )


# ============================================================
# 13. EVALUATION
# ============================================================

print("=" * 96)
print("EXPERIMENT IV | ZERO-SHOT TOPOLOGICAL GENERALIZATION")
print("=" * 96)
print(f"Device                    : {device}")
print(f"Topology bank             : {bank_path}")
print(f"Graphs/level              : {GRAPH_COUNT}")
print(f"Orders/graph              : {ORDERS_PER_GRAPH}")
print(f"T_MAX / DT / steps        : {T_MAX} / {DT} / {STEPS}")
print(f"Betas                     : {BETAS}")
print(f"Methods                   : {METHODS}")
print(f"Nominal reference         : {not args.no_nominal}")
print(f"Base seed                 : {args.base_seed}")
print(f"Checkpoint/resume CSV     : {RAW_PATH}")
print("=" * 96)


start_wall = time.time()

records = topology_records()

for topology_counter, topo in enumerate(
    records,
    start=1,
):
    topology_label = topo[
        "topology_label"
    ]

    graph_index = topo[
        "graph_index"
    ]

    A_np = np.asarray(
        topo[
            "adjacency_np"
        ],
        dtype=np.float32,
    )

    degree = A_np.sum(
        axis=1
    )

    if not np.allclose(
        degree,
        DEGREE,
    ):
        raise RuntimeError(
            f"{topology_label}, graph={graph_index}: graph is not degree {DEGREE}."
        )

    adjacency = torch.tensor(
        A_np,
        dtype=torch.float32,
        device=device,
    )

    structural_cache = build_structural_cache(
        adjacency
    )

    betweenness = brandes_betweenness(
        A_np
    )

    # Stable tie break: larger centrality first, then smaller node id.
    betweenness_nodes = np.lexsort(
        (
            np.arange(
                N
            ),
            -
            betweenness,
        )
    )[
        :
        K_SELECT
    ]

    bridge_nodes = bridge_nodes_from_adjacency(
        A_np
    )

    print()
    print(
        f"[{topology_counter:03d}/{len(records):03d}] "
        f"{topology_label} | graph={graph_index:02d} "
        f"| q={topo['actual_rewiring']:.3f} "
        f"| λ2={topo['lambda2']:.5f} "
        f"| C={topo['mean_clustering']:.4f}"
    )

    for order_index in range(
        ORDERS_PER_GRAPH
    ):
        design = episode_design(
            graph_index,
            order_index,
        )

        theta0, omegas = build_initial_state(
            design[
                "state_seed"
            ]
        )

        noise_schedule = build_noise_schedule(
            design[
                "noise_seed"
            ]
        )

        events = event_records(
            design[
                "order"
            ],
            design[
                "communities"
            ],
        )

        # Uncontrolled is beta-independent. Simulate once and reuse.
        uncontrolled_metrics = None

        uncontrolled_key = (
            topology_label,
            int(
                graph_index
            ),
            int(
                order_index
            ),
            float(
                BETAS[0]
            ),
            "uncontrolled",
        )

        # Reuse a prior uncontrolled row if it is already on disk.
        existing_unc = [
            row
            for row
            in raw_rows
            if (
                str(
                    row[
                        "topology_label"
                    ]
                )
                ==
                topology_label
                and
                int(
                    row[
                        "graph_index"
                    ]
                )
                ==
                int(
                    graph_index
                )
                and
                int(
                    row[
                        "order_index"
                    ]
                )
                ==
                int(
                    order_index
                )
                and
                str(
                    row[
                        "method"
                    ]
                )
                ==
                "uncontrolled"
            )
        ]

        if existing_unc:
            source = existing_unc[
                0
            ]

            uncontrolled_metrics = {
                key:
                    float(
                        source[
                            key
                        ]
                    )
                for key
                in [
                    "resilience_mean",
                    "resilience_worst",
                    "resilience_harmonic",
                    "event1_resilience",
                    "event2_resilience",
                    "event3_resilience",
                    "energy",
                    "energy_normalized",
                    "r_mean",
                    "r_min",
                    "r_final",
                ]
            }

        else:
            uncontrolled_metrics = simulate(
                method="uncontrolled",
                beta=0.0,
                theta0=theta0,
                omegas=omegas,
                adjacency=adjacency,
                structural_cache=structural_cache,
                events=events,
                noise_schedule=noise_schedule,
                random_nodes=design[
                    "random_nodes"
                ],
                betweenness_nodes=betweenness_nodes,
            )

        for beta in BETAS:
            # Add / reconstruct uncontrolled row for every beta so the raw
            # table can be joined directly by beta.
            key_unc = (
                topology_label,
                int(
                    graph_index
                ),
                int(
                    order_index
                ),
                float(
                    beta
                ),
                "uncontrolled",
            )

            if key_unc not in done:
                row = {
                    "topology_label":
                        topology_label,

                    "level_index":
                        topo[
                            "level_index"
                        ],

                    "graph_index":
                        int(
                            graph_index
                        ),

                    "target_rewiring":
                        topo[
                            "target_rewiring"
                        ],

                    "actual_rewiring":
                        topo[
                            "actual_rewiring"
                        ],

                    "lambda2":
                        topo[
                            "lambda2"
                        ],

                    "mean_clustering":
                        topo[
                            "mean_clustering"
                        ],

                    "modularity":
                        topo[
                            "modularity"
                        ],

                    "bridge_node_count":
                        topo[
                            "bridge_node_count"
                        ],

                    "betweenness_nodes":
                        ",".join(
                            map(
                                str,
                                sorted(
                                    betweenness_nodes.tolist()
                                ),
                            )
                        ),

                    "bridge_nodes":
                        ",".join(
                            map(
                                str,
                                bridge_nodes,
                            )
                        ),

                    "order_index":
                        int(
                            order_index
                        ),

                    "trial":
                        design[
                            "trial"
                        ],

                    "state_seed":
                        design[
                            "state_seed"
                        ],

                    "noise_seed":
                        design[
                            "noise_seed"
                        ],

                    "random_seed":
                        design[
                            "random_seed"
                        ],

                    "random_nodes":
                        ",".join(
                            map(
                                str,
                                design[
                                    "random_nodes"
                                ].tolist(),
                            )
                        ),

                    "beta":
                        float(
                            beta
                        ),

                    "method":
                        "uncontrolled",

                    "event1_type":
                        design[
                            "order"
                        ][
                            0
                        ],

                    "event2_type":
                        design[
                            "order"
                        ][
                            1
                        ],

                    "event3_type":
                        design[
                            "order"
                        ][
                            2
                        ],

                    "event1_community":
                        design[
                            "communities"
                        ][
                            0
                        ],

                    "event2_community":
                        design[
                            "communities"
                        ][
                            1
                        ],

                    "event3_community":
                        design[
                            "communities"
                        ][
                            2
                        ],
                }

                row.update(
                    uncontrolled_metrics
                )

                raw_rows.append(
                    row
                )

                done.add(
                    key_unc
                )

            for method in [
                "nrpc_gl",
                "nrpc_local",
                "full",
                "random",
                "betweenness",
            ]:
                key = (
                    topology_label,
                    int(
                        graph_index
                    ),
                    int(
                        order_index
                    ),
                    float(
                        beta
                    ),
                    method,
                )

                if key in done:
                    continue

                metrics = simulate(
                    method=method,
                    beta=float(
                        beta
                    ),
                    theta0=theta0,
                    omegas=omegas,
                    adjacency=adjacency,
                    structural_cache=structural_cache,
                    events=events,
                    noise_schedule=noise_schedule,
                    random_nodes=design[
                        "random_nodes"
                    ],
                    betweenness_nodes=betweenness_nodes,
                )

                row = {
                    "topology_label":
                        topology_label,

                    "level_index":
                        topo[
                            "level_index"
                        ],

                    "graph_index":
                        int(
                            graph_index
                        ),

                    "target_rewiring":
                        topo[
                            "target_rewiring"
                        ],

                    "actual_rewiring":
                        topo[
                            "actual_rewiring"
                        ],

                    "lambda2":
                        topo[
                            "lambda2"
                        ],

                    "mean_clustering":
                        topo[
                            "mean_clustering"
                        ],

                    "modularity":
                        topo[
                            "modularity"
                        ],

                    "bridge_node_count":
                        topo[
                            "bridge_node_count"
                        ],

                    "betweenness_nodes":
                        ",".join(
                            map(
                                str,
                                sorted(
                                    betweenness_nodes.tolist()
                                ),
                            )
                        ),

                    "bridge_nodes":
                        ",".join(
                            map(
                                str,
                                bridge_nodes,
                            )
                        ),

                    "order_index":
                        int(
                            order_index
                        ),

                    "trial":
                        design[
                            "trial"
                        ],

                    "state_seed":
                        design[
                            "state_seed"
                        ],

                    "noise_seed":
                        design[
                            "noise_seed"
                        ],

                    "random_seed":
                        design[
                            "random_seed"
                        ],

                    "random_nodes":
                        ",".join(
                            map(
                                str,
                                design[
                                    "random_nodes"
                                ].tolist(),
                            )
                        ),

                    "beta":
                        float(
                            beta
                        ),

                    "method":
                        method,

                    "event1_type":
                        design[
                            "order"
                        ][
                            0
                        ],

                    "event2_type":
                        design[
                            "order"
                        ][
                            1
                        ],

                    "event3_type":
                        design[
                            "order"
                        ][
                            2
                        ],

                    "event1_community":
                        design[
                            "communities"
                        ][
                            0
                        ],

                    "event2_community":
                        design[
                            "communities"
                        ][
                            1
                        ],

                    "event3_community":
                        design[
                            "communities"
                        ][
                            2
                        ],
                }

                row.update(
                    metrics
                )

                raw_rows.append(
                    row
                )

                done.add(
                    key
                )

            # Crash-safe checkpoint after a complete beta block.
            save_raw_checkpoint()

        print(
            f"  order {order_index + 1}/{ORDERS_PER_GRAPH}"
            f" | trial={design['trial']:03d}"
            f" | {'-'.join(design['order'])}"
            f" | C={design['communities']}"
        )


# ============================================================
# 14. ENRICH WITH NORMALIZED SCORE + GENERALIZATION GAP
# ============================================================

raw = pd.DataFrame(
    raw_rows
)

raw = raw.sort_values(
    [
        "target_rewiring",
        "graph_index",
        "order_index",
        "beta",
        "method",
    ]
).reset_index(
    drop=True
)

JOIN_KEYS = [
    "topology_label",
    "graph_index",
    "order_index",
    "trial",
    "beta",
]

full = (
    raw[
        raw[
            "method"
        ]
        ==
        "full"
    ][
        JOIN_KEYS
        +
        [
            "resilience_harmonic"
        ]
    ]
    .rename(
        columns={
            "resilience_harmonic":
                "rho_full"
        }
    )
)

unc = (
    raw[
        raw[
            "method"
        ]
        ==
        "uncontrolled"
    ][
        JOIN_KEYS
        +
        [
            "resilience_harmonic"
        ]
    ]
    .rename(
        columns={
            "resilience_harmonic":
                "rho_uncontrolled"
        }
    )
)

enriched = (
    raw
    .merge(
        full,
        on=JOIN_KEYS,
        how="left",
        validate="many_to_one",
    )
    .merge(
        unc,
        on=JOIN_KEYS,
        how="left",
        validate="many_to_one",
    )
)

denom = (
    enriched[
        "rho_full"
    ]
    -
    enriched[
        "rho_uncontrolled"
    ]
)

enriched[
    "normalized_resilience"
] = np.where(
    np.abs(
        denom
    )
    >
    1e-10,
    (
        enriched[
            "resilience_harmonic"
        ]
        -
        enriched[
            "rho_uncontrolled"
        ]
    )
    /
    denom,
    np.nan,
)


# Matched nominal reference by graph slot/order/beta/method.
if (
    "q0_nominal"
    in enriched[
        "topology_label"
    ].unique()
):
    nominal = (
        enriched[
            enriched[
                "topology_label"
            ]
            ==
            "q0_nominal"
        ][
            [
                "graph_index",
                "order_index",
                "trial",
                "beta",
                "method",
                "resilience_harmonic",
                "energy",
                "normalized_resilience",
            ]
        ]
        .rename(
            columns={
                "resilience_harmonic":
                    "rho_nominal",

                "energy":
                    "energy_nominal",

                "normalized_resilience":
                    "normalized_resilience_nominal",
            }
        )
    )

    enriched = enriched.merge(
        nominal,
        on=[
            "graph_index",
            "order_index",
            "trial",
            "beta",
            "method",
        ],
        how="left",
        validate="many_to_one",
    )

    enriched[
        "generalization_gap"
    ] = (
        enriched[
            "resilience_harmonic"
        ]
        -
        enriched[
            "rho_nominal"
        ]
    )

    enriched[
        "effort_shift_vs_nominal"
    ] = (
        enriched[
            "energy"
        ]
        -
        enriched[
            "energy_nominal"
        ]
    )

else:
    enriched[
        "rho_nominal"
    ] = np.nan

    enriched[
        "energy_nominal"
    ] = np.nan

    enriched[
        "normalized_resilience_nominal"
    ] = np.nan

    enriched[
        "generalization_gap"
    ] = np.nan

    enriched[
        "effort_shift_vs_nominal"
    ] = np.nan


enriched.to_csv(
    ENRICHED_PATH,
    index=False,
)


# ============================================================
# 15. GRAPH-LEVEL INDEPENDENT UNITS
# ============================================================

GRAPH_GROUP = [
    "topology_label",
    "level_index",
    "target_rewiring",
    "graph_index",
    "beta",
    "method",
]

graph_level = (
    enriched
    .groupby(
        GRAPH_GROUP,
        as_index=False,
    )
    .agg(
        actual_rewiring=(
            "actual_rewiring",
            "first",
        ),

        lambda2=(
            "lambda2",
            "first",
        ),

        mean_clustering=(
            "mean_clustering",
            "first",
        ),

        modularity=(
            "modularity",
            "first",
        ),

        bridge_node_count=(
            "bridge_node_count",
            "first",
        ),

        n_episodes=(
            "order_index",
            "nunique",
        ),

        resilience_harmonic=(
            "resilience_harmonic",
            "mean",
        ),

        resilience_worst=(
            "resilience_worst",
            "mean",
        ),

        resilience_mean=(
            "resilience_mean",
            "mean",
        ),

        energy=(
            "energy",
            "mean",
        ),

        normalized_resilience=(
            "normalized_resilience",
            "mean",
        ),

        generalization_gap=(
            "generalization_gap",
            "mean",
        ),

        effort_shift_vs_nominal=(
            "effort_shift_vs_nominal",
            "mean",
        ),
    )
)

graph_level.to_csv(
    GRAPH_PATH,
    index=False,
)


# ============================================================
# 16. SUMMARY ACROSS GRAPHS
# ============================================================

def sem(
    x,
):
    x = np.asarray(
        x,
        dtype=float,
    )

    x = x[
        np.isfinite(
            x
        )
    ]

    if len(
        x
    ) <= 1:
        return np.nan

    return float(
        np.std(
            x,
            ddof=1,
        )
        /
        np.sqrt(
            len(
                x
            )
        )
    )


summary_rows = []

for keys, df in graph_level.groupby(
    [
        "topology_label",
        "target_rewiring",
        "beta",
        "method",
    ]
):
    topology_label, target_q, beta, method = keys

    row = {
        "topology_label":
            topology_label,

        "target_rewiring":
            target_q,

        "beta":
            beta,

        "method":
            method,

        "n_graphs":
            int(
                df[
                    "graph_index"
                ].nunique()
            ),

        "actual_rewiring_mean":
            float(
                df[
                    "actual_rewiring"
                ].mean()
            ),

        "lambda2_mean":
            float(
                df[
                    "lambda2"
                ].mean()
            ),

        "mean_clustering_mean":
            float(
                df[
                    "mean_clustering"
                ].mean()
            ),
    }

    for metric in [
        "resilience_harmonic",
        "resilience_worst",
        "resilience_mean",
        "energy",
        "normalized_resilience",
        "generalization_gap",
        "effort_shift_vs_nominal",
    ]:
        values = df[
            metric
        ].to_numpy(
            dtype=float
        )

        row[
            f"{metric}_mean"
        ] = float(
            np.nanmean(
                values
            )
        )

        row[
            f"{metric}_sem"
        ] = sem(
            values
        )

    summary_rows.append(
        row
    )


summary = pd.DataFrame(
    summary_rows
)

summary = summary.sort_values(
    [
        "target_rewiring",
        "beta",
        "method",
    ]
)

summary.to_csv(
    SUMMARY_PATH,
    index=False,
)


# ============================================================
# 17. PICKLE / REPORT
# ============================================================

payload = {
    "configuration":
        {
            "experiment":
                "unseen_topology_generalization",

            "base_seed":
                args.base_seed,

            "tag":
                suffix,

            "run_id":
                run_id,

            "betas":
                BETAS,

            "graphs_per_level":
                GRAPH_COUNT,

            "orders_per_graph":
                ORDERS_PER_GRAPH,

            "event_starts":
                EVENT_STARTS,

            "event_orders":
                EVENT_ORDERS,

            "community_patterns":
                COMMUNITY_PATTERNS,

            "k_coord":
                K_COORD,

            "k_select":
                K_SELECT,

            "dt":
                DT,

            "t_max":
                T_MAX,

            "methods":
                METHODS,

            "checkpoint_gl":
                args.gl_checkpoint,

            "checkpoint_local":
                args.local_checkpoint,
        },

    "raw":
        enriched,

    "graph_level":
        graph_level,

    "summary":
        summary,
}

with PKL_PATH.open(
    "wb"
) as f:
    pickle.dump(
        payload,
        f,
    )


elapsed = (
    time.time()
    -
    start_wall
)

print()
print("=" * 96)
print("EXPERIMENT IV COMPLETE")
print("=" * 96)
print(f"Elapsed                   : {elapsed/60.0:.1f} min")
print(f"Raw checkpoint            : {RAW_PATH}")
print(f"Enriched episode CSV      : {ENRICHED_PATH}")
print(f"Graph-level CSV           : {GRAPH_PATH}")
print(f"Summary CSV               : {SUMMARY_PATH}")
print(f"Pickle                    : {PKL_PATH}")
print("=" * 96)

print()
print("Primary quick-look: mean normalized resilience S by topology / beta / method")
quick = summary[
    summary[
        "method"
    ].isin(
        [
            "nrpc_gl",
            "nrpc_local",
            "random",
            "betweenness",
        ]
    )
][
    [
        "topology_label",
        "beta",
        "method",
        "normalized_resilience_mean",
        "generalization_gap_mean",
        "energy_mean",
    ]
]

print(
    quick.to_string(
        index=False
    )
)
