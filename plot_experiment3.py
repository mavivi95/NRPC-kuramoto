#!/usr/bin/env python3
"""
Network-topology activation maps for NRPC-V4 structural node failures.

This script visualizes the structural-failure experiment as the ACTUAL
modular graph instead of a 10x10 index heatmap.

Expected input
--------------
NPZ produced by:
    eval_v4_structural_activation_v2.py

Default visualization
---------------------
- node color  : effective activation frequency DURING the failure
- black X     : failed node
- gray edges  : surviving edges during the failure
- faint dashed red edges: nominal edges disabled by the failure
- white rings : bridge nodes
- fixed node positions across every panel

Outputs
-------
For scenarios a, b, c:
    structural_a_comparison.pdf/png   # NRPC-GL | NRPC-L
    structural_b_comparison.pdf/png
    structural_c_comparison.pdf/png

Individual panels:
    structural_a_global_local.pdf/png
    structural_a_local.pdf/png
    ...

Overview:
    structural_overview_2x3.pdf/png

No NetworkX dependency is required.
"""

from pathlib import Path
import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.collections import LineCollection

mpl.rcParams["pdf.fonttype"] = 42
mpl.rcParams["ps.fonttype"] = 42

plt.rcParams["font.family"] = "DejaVu Serif"
plt.rcParams["font.serif"] = ["Times New Roman", "DejaVu Serif"]


# ============================================================
# 1. CLI
# ============================================================

parser = argparse.ArgumentParser()

parser.add_argument(
    "--npz",
    type=str,
    default=None,
    help=(
        "NPZ produced by eval_v4_structural_activation_v2.py. "
        "If omitted, the newest final_v2 NPZ in "
        "results/structural_activation/ is used."
    ),
)

parser.add_argument(
    "--outdir",
    type=str,
    default="figures/experiment3",
)

parser.add_argument(
    "--kind",
    type=str,
    default="effective",
    choices=["effective", "raw"],
)

parser.add_argument(
    "--window",
    type=str,
    default="during",
    choices=["all", "pre", "during", "post"],
)

parser.add_argument(
    "--node-size",
    type=float,
    default=230.0,
)

parser.add_argument(
    "--dpi",
    type=int,
    default=300,
)

parser.add_argument(
    "--hide-disabled-edges",
    action="store_true",
    help="Do not draw nominal edges removed by the structural failure.",
)

parser.add_argument(
    "--hide-bridge-rings",
    action="store_true",
    help="Do not draw white rings around structural bridge nodes.",
)

args = parser.parse_args()


# ============================================================
# 2. PATHS
# ============================================================

ROOT = Path(__file__).resolve().parent

# If the script is stored in a scripts/ or figures/ folder,
# walk one level up to find results/.
if not (ROOT / "results").exists():
    if (ROOT.parent / "results").exists():
        ROOT = ROOT.parent

RESULTS_DIR = ROOT / "results" / "experiment3"

if args.npz is None:

    candidates = sorted(
        RESULTS_DIR.glob(
            "structural_activation_final_*.npz"
        ),
        key=lambda p: p.stat().st_mtime,
    )

    if not candidates:
        candidates = sorted(
            RESULTS_DIR.glob(
                "structural_activation_final_*.npz"
            ),
            key=lambda p: p.stat().st_mtime,
        )

    if not candidates:
        raise FileNotFoundError(
            "No final structural activation NPZ was found. "
            "Pass the file explicitly with --npz."
        )

    NPZ_PATH = candidates[-1]

else:

    NPZ_PATH = Path(
        args.npz
    )

    if not NPZ_PATH.is_absolute():
        NPZ_PATH = ROOT / NPZ_PATH


OUTDIR = Path(
    args.outdir
)

if not OUTDIR.is_absolute():
    OUTDIR = ROOT / OUTDIR

OUTDIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# 3. LOAD RESULTS
# ============================================================

if not NPZ_PATH.exists():
    raise FileNotFoundError(
        NPZ_PATH
    )

data = np.load(
    NPZ_PATH,
    allow_pickle=True,
)

A_NOMINAL = np.asarray(
    data[
        "adjacency_nominal"
    ],
    dtype=float,
)

N_AGENTS = A_NOMINAL.shape[0]

if N_AGENTS != 100:
    raise RuntimeError(
        f"Expected 100 nodes, found {N_AGENTS}."
    )

N_COMMUNITIES = 5
COMMUNITY_SIZE = 20

BRIDGE_NODES = sorted(
    np.asarray(
        data[
            "bridge_nodes"
        ],
        dtype=int,
    ).tolist()
)

SCENARIOS = [
    "a_interface_C1_C2",
    "b_interface_C3_C4",
    "c_all_bridge_nodes",
]

SCENARIO_LETTER = {
    "a_interface_C1_C2":
        "a",

    "b_interface_C3_C4":
        "b",

    "c_all_bridge_nodes":
        "c",
}

SCENARIO_TITLES = {
    "a_interface_C1_C2":
        "Localized failure near C1-C2",

    "b_interface_C3_C4":
        "Shifted failure near C3-C4",

    "c_all_bridge_nodes":
        "Failure of all bridge nodes",
}

MODES = [
    "global_local",
    "local",
]

MODE_TITLES = {
    "global_local":
        "NRPC-GL",

    "local":
        "NRPC-L",
}


# ============================================================
# 4. FIXED ANALYTICAL MODULAR LAYOUT
# ============================================================

def build_fixed_layout():
    """
    Five community centers are arranged on a large pentagon.

    Ordinary nodes are placed on a small ring around each community
    center. Bridge nodes are placed slightly farther outward and oriented
    toward the neighboring community they connect to.

    Positions are fixed ONCE and reused across all panels.
    """

    positions = {}

    outer_radius = 6.25
    inner_radius = 2.82
    bridge_radius = 1.58

    # Start with C1 at the top and proceed clockwise.
    community_angles = (
        np.pi / 2
        -
        np.arange(
            N_COMMUNITIES
        )
        *
        2.0
        *
        np.pi
        /
        N_COMMUNITIES
    )

    centers = {}

    for c in range(
        N_COMMUNITIES
    ):

        angle = community_angles[
            c
        ]

        centers[c] = np.array(
            [
                outer_radius
                *
                np.cos(
                    angle
                ),

                outer_radius
                *
                np.sin(
                    angle
                ),
            ],
            dtype=float,
        )

    # --------------------------------------------------------
    # Ordinary nodes
    # --------------------------------------------------------

    for c in range(
        N_COMMUNITIES
    ):

        center = centers[
            c
        ]

        ordinary = [
            i
            for i in range(
                COMMUNITY_SIZE
            )
            if i
            not in {
                4,
                5,
                12,
                13,
            }
        ]

        # Put the 16 ordinary nodes around the module.
        ordinary_angles = (
            np.linspace(
                0.0,
                2.0
                *
                np.pi,
                len(
                    ordinary
                ),
                endpoint=False,
            )
            +
            0.08
        )

        for local_i, angle in zip(
            ordinary,
            ordinary_angles,
        ):

            node = (
                c
                *
                COMMUNITY_SIZE
                +
                local_i
            )

            positions[
                node
            ] = (
                center
                +
                inner_radius
                *
                np.array(
                    [
                        np.cos(
                            angle
                        ),
                        np.sin(
                            angle
                        ),
                    ]
                )
            )

        # ----------------------------------------------------
        # Bridge nodes:
        #
        # local 4,5  point toward next community
        # local 12,13 point toward previous community
        # ----------------------------------------------------

        next_c = (
            c
            +
            1
        ) % N_COMMUNITIES

        prev_c = (
            c
            -
            1
        ) % N_COMMUNITIES

        direction_next = (
            centers[
                next_c
            ]
            -
            center
        )

        direction_prev = (
            centers[
                prev_c
            ]
            -
            center
        )

        angle_next = np.arctan2(
            direction_next[
                1
            ],
            direction_next[
                0
            ],
        )

        angle_prev = np.arctan2(
            direction_prev[
                1
            ],
            direction_prev[
                0
            ],
        )

        offsets = [
            -0.15,
            +0.15,
        ]

        for local_i, offset in zip(
            [
                4,
                5,
            ],
            offsets,
        ):

            node = (
                c
                *
                COMMUNITY_SIZE
                +
                local_i
            )

            angle = (
                angle_next
                +
                offset
            )

            positions[
                node
            ] = (
                center
                +
                bridge_radius
                *
                np.array(
                    [
                        np.cos(
                            angle
                        ),
                        np.sin(
                            angle
                        ),
                    ]
                )
            )

        for local_i, offset in zip(
            [
                12,
                13,
            ],
            offsets,
        ):

            node = (
                c
                *
                COMMUNITY_SIZE
                +
                local_i
            )

            angle = (
                angle_prev
                +
                offset
            )

            positions[
                node
            ] = (
                center
                +
                bridge_radius
                *
                np.array(
                    [
                        np.cos(
                            angle
                        ),
                        np.sin(
                            angle
                        ),
                    ]
                )
            )

    return (
        positions,
        centers,
    )


POSITIONS, COMMUNITY_CENTERS = (
    build_fixed_layout()
)

XY = np.asarray(
    [
        POSITIONS[
            i
        ]
        for i in range(
            N_AGENTS
        )
    ],
    dtype=float,
)


# ============================================================
# 5. GRAPH HELPERS
# ============================================================

def edge_list_from_adjacency(
    A,
):
    edges = []

    for i in range(
        A.shape[
            0
        ]
    ):

        for j in range(
            i
            +
            1,
            A.shape[
                1
            ]
        ):

            if A[
                i,
                j
            ] > 0:

                edges.append(
                    (
                        i,
                        j,
                    )
                )

    return edges


NOMINAL_EDGES = (
    edge_list_from_adjacency(
        A_NOMINAL
    )
)


def failed_nodes(
    scenario,
):
    return sorted(
        np.asarray(
            data[
                f"{scenario}"
                "__failed_nodes"
            ],
            dtype=int,
        ).tolist()
    )


def surviving_and_disabled_edges(
    scenario,
):
    failed = set(
        failed_nodes(
            scenario
        )
    )

    surviving = []

    disabled = []

    for i, j in NOMINAL_EDGES:

        if (
            i in failed
            or
            j in failed
        ):
            disabled.append(
                (
                    i,
                    j,
                )
            )

        else:
            surviving.append(
                (
                    i,
                    j,
                )
            )

    return (
        surviving,
        disabled,
    )


def edge_segments(
    edges,
):
    return [
        [
            POSITIONS[
                i
            ],
            POSITIONS[
                j
            ],
        ]
        for i, j in edges
    ]


def activation_vector(
    mode,
    scenario,
):
    key = (
        f"{mode}"
        f"__{scenario}"
        f"__{args.kind}"
        f"__{args.window}"
    )

    if key not in data:
        raise KeyError(
            f"Missing array in NPZ: {key}"
        )

    return np.asarray(
        data[
            key
        ],
        dtype=float,
    )


# ============================================================
# 6. COMMON COLOR SCALE
# ============================================================

all_values = []

for mode in MODES:

    for scenario in SCENARIOS:

        all_values.append(
            activation_vector(
                mode,
                scenario,
            )
        )

all_values = np.concatenate(
    all_values
)

VMIN = 0.0

VMAX = float(
    np.max(
        all_values
    )
)

if VMAX <= 0.0:
    VMAX = 1.0

NORM = mpl.colors.Normalize(
    vmin=VMIN,
    vmax=VMAX,
)

CMAP = plt.get_cmap(
    "viridis"
)


# ============================================================
# 7. DRAW ONE NETWORK PANEL
# ============================================================

def draw_panel(
    ax,
    mode,
    scenario,
    title=None,
    show_community_labels=True,
):
    activation = activation_vector(
        mode,
        scenario,
    )

    failed = set(
        failed_nodes(
            scenario
        )
    )

    surviving_edges, disabled_edges = (
        surviving_and_disabled_edges(
            scenario
        )
    )

    # --------------------------------------------------------
    # Community halos
    # --------------------------------------------------------

    for c in range(
        N_COMMUNITIES
    ):

        center = COMMUNITY_CENTERS[
            c
        ]

        halo = plt.Circle(
            center,
            radius=2.3,
            facecolor="#f7f7f7",
            edgecolor="#c8c8c8",
            linewidth=0.9,
            alpha=0.75,
            zorder=0,
        )

        ax.add_patch(
            halo
        )

    # --------------------------------------------------------
    # Disabled nominal edges
    # --------------------------------------------------------

    if (
        not args.hide_disabled_edges
        and
        len(
            disabled_edges
        )
        >
        0
    ):

        disabled_collection = (
            LineCollection(
                edge_segments(
                    disabled_edges
                ),
                colors="red",
                linewidths=0.7,
                linestyles="dashed",
                alpha=0.5,
                zorder=1,
            )
        )

        ax.add_collection(
            disabled_collection
        )

    # --------------------------------------------------------
    # Surviving edges
    # --------------------------------------------------------

    internal_segments = []

    bridge_segments = []

    for i, j in surviving_edges:

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

        segment = [
            POSITIONS[
                i
            ],
            POSITIONS[
                j
            ],
        ]

        if ci == cj:
            internal_segments.append(
                segment
            )
        else:
            bridge_segments.append(
                segment
            )

    if internal_segments:

        ax.add_collection(
            LineCollection(
                internal_segments,
                colors="#9b9b9b",
                linewidths=0.55,
                alpha=0.24,
                zorder=1,
            )
        )

    if bridge_segments:

        ax.add_collection(
            LineCollection(
                bridge_segments,
                colors="#4f4f4f",
                linewidths=1.35,
                alpha=0.55,
                zorder=2,
            )
        )

    # --------------------------------------------------------
    # Surviving nodes
    # --------------------------------------------------------

    surviving_nodes = [
        i
        for i in range(
            N_AGENTS
        )
        if i
        not in failed
    ]

    surviving_xy = XY[
        surviving_nodes
    ]

    surviving_values = activation[
        surviving_nodes
    ]

    ax.scatter(
        surviving_xy[
            :,
            0
        ],
        surviving_xy[
            :,
            1
        ],
        c=surviving_values,
        cmap=CMAP,
        norm=NORM,
        s=args.node_size,
        edgecolors="black",
        linewidths=0.55,
        zorder=4,
    )

    # --------------------------------------------------------
    # Bridge-node rings
    # --------------------------------------------------------

    if not args.hide_bridge_rings:

        visible_bridges = [
            node
            for node in BRIDGE_NODES
            if node
            not in failed
        ]

        # if visible_bridges:

        #     bridge_xy = XY[
        #         visible_bridges
        #     ]

        #     ax.scatter(
        #         bridge_xy[
        #             :,
        #             0
        #         ],
        #         bridge_xy[
        #             :,
        #             1
        #         ],
        #         facecolors="none",
        #         edgecolors="blue",
        #         s=(
        #             args.node_size
        #             *
        #             1.25
        #         ),
        #         linewidths=1.2,
        #         zorder=5,
        #     )

    # --------------------------------------------------------
    # Failed nodes
    # --------------------------------------------------------

    failed_list = sorted(
        failed
    )

    if failed_list:

        failed_xy = XY[
            failed_list
        ]

        # pale circle gives the failed location a physical footprint
        ax.scatter(
            failed_xy[
                :,
                0
            ],
            failed_xy[
                :,
                1
            ],
            c="#eeeeee",
            s=args.node_size,
            edgecolors="#555555",
            linewidths=0.7,
            zorder=5,
        )

        ax.scatter(
            failed_xy[
                :,
                0
            ],
            failed_xy[
                :,
                1
            ],
            marker="x",
            c="black",
            s=(
                args.node_size
                *
                0.55
            ),
            linewidths=1.8,
            zorder=6,
        )

    # --------------------------------------------------------
    # Community labels
    # --------------------------------------------------------

    if show_community_labels:

        for c in range(
            N_COMMUNITIES
        ):

            center = COMMUNITY_CENTERS[
                c
            ]

            radial = (
                center
                /
                np.linalg.norm(
                    center
                )
            )

            label_xy = (
                center
                +
                1.57
                *
                radial
            )

            ax.text(
                label_xy[
                    0
                ],
                label_xy[
                    1
                ],
                f"C{c + 1}",
                ha="center",
                va="center",
                fontsize=12,
                fontweight="bold",
                zorder=8,
            )

    if title is not None:

        ax.set_title(
            title,
            fontsize=15,
            fontweight="bold",
            pad=8,
        )

    ax.set_aspect(
        "equal"
    )

    margin = 2.3

    ax.set_xlim(
        XY[
            :,
            0
        ].min()
        -
        margin,
        XY[
            :,
            0
        ].max()
        +
        margin,
    )

    ax.set_ylim(
        XY[
            :,
            1
        ].min()
        -
        margin,
        XY[
            :,
            1
        ].max()
        +
        margin,
    )

    ax.axis(
        "off"
    )


# ============================================================
# 8. SAVE HELPER
# ============================================================

def save_figure(
    fig,
    stem,
):
    pdf = (
        OUTDIR
        /
        f"{stem}.pdf"
    )

    png = (
        OUTDIR
        /
        f"{stem}.png"
    )

    fig.savefig(
        pdf,
        dpi=args.dpi,
        bbox_inches="tight",
    )

    fig.savefig(
        png,
        dpi=args.dpi,
        bbox_inches="tight",
    )

    print(
        f"[saved] {pdf}"
    )

    print(
        f"[saved] {png}"
    )


def shared_colorbar(
    fig,
    axes,
):
    scalar = mpl.cm.ScalarMappable(
        norm=NORM,
        cmap=CMAP,
    )

    scalar.set_array(
        []
    )

    cbar = fig.colorbar(
        scalar,
        ax=np.ravel(
            axes
        ).tolist(),
        fraction=0.027,
        pad=0.02,
    )

    cbar.set_label(
        "Activation frequency",
        fontsize=12,
        fontweight="bold",
    )

    cbar.ax.tick_params(
        labelsize=10
    )

    return cbar


# ============================================================
# 9. SCENARIO COMPARISONS: GL | LOCAL
# ============================================================

for scenario in SCENARIOS:

    letter = SCENARIO_LETTER[
        scenario
    ]

    fig, axes = plt.subplots(
    1,
    2,
    figsize=(15.5, 6.0),
    )

    draw_panel(
        axes[
            0
        ],
        "global_local",
        scenario,
        title="NRPC-GL",
    )

    draw_panel(
        axes[
            1
        ],
        "local",
        scenario,
        title="NRPC-L",
    )

    shared_colorbar(
        fig,
        axes,
    )

    fig.suptitle(
        f"{letter}) "
        f"{SCENARIO_TITLES[scenario]}",
        fontsize=18,
        fontweight="bold",
        y=0.98,
    )

    fig.text(
        0.5,
        0.025,
        (
            "Black crosses: failed nodes   |   "
            "White rings: surviving bridge nodes"
        ),
        ha="center",
        va="bottom",
        fontsize=10.5,
    )

    plt.tight_layout(
        rect=[
            0.0,
            0.05,
            0.94,
            0.94,
        ]
    )

    save_figure(
        fig,
        f"structural_{letter}_comparison",
    )

    plt.close(
        fig
    )


# ============================================================
# 10. INDIVIDUAL PANELS
# ============================================================

for scenario in SCENARIOS:

    letter = SCENARIO_LETTER[
        scenario
    ]

    for mode in MODES:

        fig, ax = plt.subplots(
            1,
            1,
            figsize=(
                6.2,
                5.7,
            ),
        )

        draw_panel(
            ax,
            mode,
            scenario,
            title=MODE_TITLES[
                mode
            ],
        )

        scalar = mpl.cm.ScalarMappable(
            norm=NORM,
            cmap=CMAP,
        )

        scalar.set_array(
            []
        )

        cbar = fig.colorbar(
            scalar,
            ax=ax,
            fraction=0.045,
            pad=0.02,
        )

        cbar.set_label(
            "Activation frequency",
            fontsize=11,
            fontweight="bold",
        )

        cbar.ax.tick_params(
            labelsize=9
        )

        fig.suptitle(
            f"{letter}) "
            f"{SCENARIO_TITLES[scenario]}",
            fontsize=15,
            fontweight="bold",
            y=0.98,
        )

        plt.tight_layout(
            rect=[
                0.0,
                0.0,
                0.95,
                0.94,
            ]
        )

        save_figure(
            fig,
            (
                f"structural_"
                f"{letter}_"
                f"{mode}"
            ),
        )

        plt.close(
            fig
        )


# ============================================================
# 11. OVERVIEW 2x3
# ============================================================

# Cambia la paleta aquí si lo deseas (ej. "magma", "plasma", "inferno")
CMAP_OVERVIEW = plt.get_cmap("inferno")

fig, axes = plt.subplots(
    2,
    3,
    figsize=(
        17.5,
        10.0,
    ),
    gridspec_kw={"wspace": 0.05, "hspace": 0.08},
)

for i, mode in enumerate(MODES):
    # Escala de color independiente por fila
    row_values = np.concatenate([
        activation_vector(mode, sc) for sc in SCENARIOS
    ])
    row_vmax = float(np.max(row_values)) if float(np.max(row_values)) > 0 else 1.0
    row_norm = mpl.colors.Normalize(vmin=0.0, vmax=row_vmax)

    for j, scenario in enumerate(SCENARIOS):
        title = (
            f"{SCENARIO_LETTER[scenario]}) "
            f"{SCENARIO_TITLES[scenario]}"
            if i == 0
            else None
        )

        draw_panel(
            axes[i, j],
            mode,
            scenario,
            title=title,
        )

        # Aplicar la norma y la nueva paleta a los nodos de este panel
        for coll in axes[i, j].collections:
            if isinstance(coll, mpl.collections.PathCollection) and coll.get_array() is not None:
                coll.set_cmap(CMAP_OVERVIEW)
                coll.set_norm(row_norm)

        if j == 0:
            axes[i, j].text(
                -0.08,
                0.50,
                MODE_TITLES[mode],
                transform=axes[i, j].transAxes,
                rotation=90,
                ha="center",
                va="center",
                fontsize=16,
                fontweight="bold",
            )

    # Barra de color anclada al último subplot de la fila para evitar solapamientos
    row_scalar = mpl.cm.ScalarMappable(
        norm=row_norm,
        cmap=CMAP_OVERVIEW,
    )
    row_scalar.set_array([])

    cbar = fig.colorbar(
        row_scalar,
        ax=axes[i, 2],
        fraction=0.046,
        pad=0.05,
        shrink=0.82,
    )
    cbar.set_label(
        f"Activation frequency ({MODE_TITLES[mode]})",
        fontsize=11,
        fontweight="bold",
    )
    cbar.ax.tick_params(labelsize=9.5)

# fig.text(
#     0.5,
#     0.018,
#     (
#         "Node color: activation frequency   |   "
#         "Black crosses: failed nodes   |   "
#         "Orange rings: surviving bridge nodes   |   "
#         "Dashed red edges: links removed by the failure"
#     ),
#     ha="center",
#     fontsize=10.5,
# )

plt.tight_layout(
    rect=[
        0.02,
        0.045,
        0.96,
        0.95,
    ]
)

save_figure(
    fig,
    "structural_overview_2x3",
)

plt.close(fig)


# ============================================================
# 12. INFO
# ============================================================

print()
print("=" * 78)
print("STRUCTURAL NETWORK FIGURES COMPLETE")
print("=" * 78)
print(f"Input NPZ              : {NPZ_PATH}")
print(f"Activation kind        : {args.kind}")
print(f"Time window            : {args.window}")
print(f"Common color scale     : [{VMIN:.6f}, {VMAX:.6f}]")
print(f"Output directory       : {OUTDIR}")
print("=" * 78)
