#!/usr/bin/env python3
"""
Figure 4B — Transfer advantage on unseen topologies.

For each unseen topology, compute the resilience advantage of the learned
selectors relative to a static sparse baseline:

    Delta rho = rho_method - rho_baseline

Default baseline:
    Betweenness-k

The plot shows NRPC-GL and NRPC-L only. A horizontal line at Delta rho = 0
marks parity with the baseline.

Expected input:
    Experiment 4 enriched CSV, e.g.
    results/experiment4/eval_v4_unseen_topologies_final_T100_seed91000_enriched.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# ---------------------------------------------------------------------
# Paper style
# ---------------------------------------------------------------------

mpl.rcParams["pdf.fonttype"] = 42
mpl.rcParams["ps.fonttype"] = 42
mpl.rcParams["font.family"] = "DejaVu Serif"
mpl.rcParams["font.serif"] = ["Times New Roman", "DejaVu Serif"]
mpl.rcParams["axes.linewidth"] = 0.9
mpl.rcParams["xtick.major.width"] = 0.8
mpl.rcParams["ytick.major.width"] = 0.8
mpl.rcParams["legend.frameon"] = True


METHODS = [
    "nrpc_gl",
    "nrpc_local",
]

METHOD_LABELS = {
    "nrpc_gl": "NRPC-GL",
    "nrpc_local": "NRPC-L",
}

MARKERS = {
    "nrpc_gl": "o",
    "nrpc_local": "^",
}


def latest_enriched(root: Path) -> Path:
    candidates = sorted(
        root.glob("results/experiment4/*enriched.csv"),
        key=lambda p: p.stat().st_mtime,
    )

    if not candidates:
        candidates = sorted(
            root.glob("results/experiment4/**/*enriched.csv"),
            key=lambda p: p.stat().st_mtime,
        )

    if not candidates:
        raise FileNotFoundError(
            "No enriched CSV found under results/experiment4/. "
            "Pass --csv explicitly."
        )

    return candidates[-1]


def normalize_method_name(value):
    raw = str(value).strip()
    key = raw.lower().replace("-", "_").replace(" ", "_")

    aliases = {
        "nrpc_gl": "nrpc_gl",
        "global_local": "nrpc_gl",
        "nrpc_global_local": "nrpc_gl",
        "gl": "nrpc_gl",

        "nrpc_l": "nrpc_local",
        "nrpc_local": "nrpc_local",
        "local": "nrpc_local",

        "random": "random",
        "random_k": "random",
        "random_topk": "random",

        "betweenness": "betweenness",
        "betweenness_k": "betweenness",

        "full": "full",
        "full_consensus": "full",
        "consensus": "full",

        "uncontrolled": "uncontrolled",
        "no_control": "uncontrolled",
        "nocontrol": "uncontrolled",
    }

    return aliases.get(key, key)


def find_column(
    df,
    candidates,
    explicit=None,
    required=True,
):
    if explicit is not None:
        if explicit not in df.columns:
            raise ValueError(
                f"Requested column '{explicit}' not found.\n"
                f"Available columns:\n{list(df.columns)}"
            )
        return explicit

    lower_map = {
        c.lower(): c
        for c in df.columns
    }

    for candidate in candidates:
        if candidate.lower() in lower_map:
            return lower_map[
                candidate.lower()
            ]

    for candidate in candidates:
        key = candidate.lower()
        for col in df.columns:
            if key in col.lower():
                return col

    if required:
        raise ValueError(
            "Could not detect a required column.\n"
            f"Tried: {candidates}\n"
            f"Available columns:\n{list(df.columns)}"
        )

    return None


def aggregate_topology_level(
    df,
    topology_col,
    lambda2_col,
    resilience_col,
    method_col,
    beta_col=None,
    beta=None,
):
    work = df.copy()

    work["_method_norm"] = (
        work[
            method_col
        ].map(
            normalize_method_name
        )
    )

    if beta_col is not None and beta is not None:
        numeric_beta = pd.to_numeric(
            work[
                beta_col
            ],
            errors="coerce",
        )

        work = work[
            np.isclose(
                numeric_beta,
                beta,
            )
        ].copy()

        if work.empty:
            available = sorted(
                pd.to_numeric(
                    df[
                        beta_col
                    ],
                    errors="coerce",
                )
                .dropna()
                .unique()
                .tolist()
            )

            raise ValueError(
                f"No rows found for beta={beta:g}. "
                f"Available beta values: {available}"
            )

    work[
        lambda2_col
    ] = pd.to_numeric(
        work[
            lambda2_col
        ],
        errors="coerce",
    )

    work[
        resilience_col
    ] = pd.to_numeric(
        work[
            resilience_col
        ],
        errors="coerce",
    )

    work = work.dropna(
        subset=[
            topology_col,
            lambda2_col,
            resilience_col,
            "_method_norm",
        ]
    )

    agg = (
        work.groupby(
            [
                topology_col,
                "_method_norm",
            ],
            as_index=False,
        )
        .agg(
            lambda2=(
                lambda2_col,
                "mean",
            ),
            resilience=(
                resilience_col,
                "mean",
            ),
        )
    )

    agg = agg.rename(
        columns={
            "_method_norm":
                "method",
        }
    )

    return agg


def build_advantage_table(
    agg,
    topology_col,
    baseline,
):
    baseline_rows = (
        agg[
            agg[
                "method"
            ]
            ==
            baseline
        ][
            [
                topology_col,
                "lambda2",
                "resilience",
            ]
        ]
        .rename(
            columns={
                "resilience":
                    "baseline_resilience",
            }
        )
    )

    if baseline_rows.empty:
        raise ValueError(
            f"Baseline '{baseline}' was not found. "
            f"Methods available: {sorted(set(agg['method']))}"
        )

    rows = []

    for method in METHODS:
        method_rows = (
            agg[
                agg[
                    "method"
                ]
                ==
                method
            ][
                [
                    topology_col,
                    "lambda2",
                    "resilience",
                ]
            ]
            .rename(
                columns={
                    "resilience":
                        "method_resilience",
                    "lambda2":
                        "lambda2_method",
                }
            )
        )

        merged = method_rows.merge(
            baseline_rows,
            on=topology_col,
            how="inner",
            suffixes=(
                "_method",
                "_baseline",
            ),
        )

        if merged.empty:
            continue

        # Prefer baseline lambda2; topology structure is shared by method.
        merged["lambda2"] = (
            merged[
                "lambda2"
            ]
            if "lambda2" in merged.columns
            else merged[
                "lambda2_baseline"
            ]
        )

        merged["delta_rho"] = (
            merged[
                "method_resilience"
            ]
            -
            merged[
                "baseline_resilience"
            ]
        )

        merged["method"] = method

        rows.append(
            merged[
                [
                    topology_col,
                    "lambda2",
                    "method",
                    "method_resilience",
                    "baseline_resilience",
                    "delta_rho",
                ]
            ]
        )

    if not rows:
        raise ValueError(
            "No overlapping topology-level rows were found "
            "between learned methods and the baseline."
        )

    return pd.concat(
        rows,
        ignore_index=True,
    )


def make_figure(
    advantage,
    topology_col,
    outdir,
    beta,
    baseline,
):
    """
    Grouped bar chart:
        x = unseen topology (ordered by lambda_2)
        y = Delta rho against the selected sparse baseline
    """

    # One ordered topology list shared by both learned methods.
    topology_order = (
        advantage[
            [
                topology_col,
                "lambda2",
            ]
        ]
        .drop_duplicates()
        .sort_values(
            "lambda2"
        )
        .reset_index(
            drop=True
        )
    )

    topology_order[
        "topology_label"
    ] = [
        rf"$\lambda_2={x:.3f}$"
        for x
        in topology_order[
            "lambda2"
        ].to_numpy(
            dtype=float
        )
    ]

    x = np.arange(
        len(
            topology_order
        )
    )

    width = 0.34

    fig, ax = plt.subplots(
        figsize=(
            8.4,
            4.9,
        )
    )

    offsets = {
        "nrpc_gl":
            -width / 2,

        "nrpc_local":
            width / 2,
    }

    for method in METHODS:

        # Match by lambda2 instead of topology ID. The enriched table may
        # encode topology identifiers with different dtypes after aggregation.
        # lambda2 is the structural key shared by the plotted topology rows.
        d = (
            advantage[
                advantage["method"] == method
            ][["lambda2", "delta_rho"]]
            .groupby("lambda2", as_index=False)["delta_rho"]
            .mean()
            .sort_values("lambda2")
        )

        merged = topology_order[["lambda2", "topology_label"]].merge(
            d,
            on="lambda2",
            how="left",
            validate="one_to_one",
        )

        values = merged["delta_rho"].to_numpy(dtype=float)

        if np.isnan(values).any():
            raise ValueError(
                f"Missing bar values for {method}. "
                f"Merged data:\n{merged.to_string(index=False)}"
            )

        print(f"[bars] {method}: {np.round(values, 4).tolist()}")

        ax.bar(
            x + offsets[method],
            values,
            width=width,
            label=METHOD_LABELS[
                method
            ],
            alpha=0.94,
            edgecolor="black",
            linewidth=0.65,
            zorder=3,
        )

    ax.axhline(
        0.0,
        linestyle="--",
        linewidth=1.15,
        alpha=0.80,
        zorder=1,
    )

    ax.set_xticks(
        x,
        topology_order[
            "topology_label"
        ],
    )

    ax.set_xlabel(
        r"Unseen Topology "
        r"(ordered by $\lambda_2(L)$)",
        fontsize=14,
        fontweight="bold",
    )

    ax.set_ylabel(
        r"Resilience Advantage $\Delta\rho$",
        fontsize=16,
        fontweight="bold",
    )

    baseline_label = {
        "betweenness":
            r"Betweenness-$k$",

        "random":
            r"Random-$k$",
    }.get(
        baseline,
        baseline
    )

    # title = (
    #     "Transfer Advantage over "
    #     f"{baseline_label}"
    # )

    # if beta is not None:
    #     title += (
    #         rf" ($\beta={beta:g}$)"
    #     )

    # ax.set_title(
    #     title,
    #     fontsize=14,
    #     fontweight="bold",
    #     pad=10,
    # )

    ax.set_ylim(0, 0.3)

    ax.grid(
        axis="y",
        alpha=0.13,
        linewidth=0.75,
        zorder=0,
    )

    ax.tick_params(
        axis="both",
        labelsize=11.0,
    )

    leg = ax.legend(
        ncol=2,
        fontsize=10.8,
        frameon=True,
        fancybox=True,
        framealpha=0.95,
        edgecolor="0.45",
        facecolor="white",
        borderpad=0.65,
        loc="upper center",
    )

    leg.get_frame().set_linewidth(
        0.9
    )

    fig.tight_layout()

    beta_suffix = (
        ""
        if beta is None
        else f"_beta{str(beta).replace('.', 'p')}"
    )

    baseline_suffix = baseline.replace(
        "_",
        "-"
    )

    pdf_path = (
        outdir
        /
        (
            "fig4B_transfer_advantage_bars_"
            f"vs_{baseline_suffix}"
            f"{beta_suffix}.pdf"
        )
    )

    png_path = (
        outdir
        /
        (
            "fig4B_transfer_advantage_bars_"
            f"vs_{baseline_suffix}"
            f"{beta_suffix}.png"
        )
    )

    fig.savefig(
        pdf_path,
        bbox_inches="tight",
    )

    fig.savefig(
        png_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(
        fig
    )

    print(
        f"[saved] {pdf_path}"
    )

    print(
        f"[saved] {png_path}"
    )

def print_summary(
    advantage,
    baseline,
):
    print()
    print(
        "Transfer advantage summary"
    )
    print(
        "-" * 62
    )

    for method in METHODS:
        d = advantage[
            advantage[
                "method"
            ]
            ==
            method
        ]

        if d.empty:
            continue

        values = d[
            "delta_rho"
        ].to_numpy(
            dtype=float
        )

        print(
            f"{METHOD_LABELS[method]:<12} | "
            f"mean Δρ={np.mean(values):+.4f} | "
            f"min Δρ={np.min(values):+.4f} | "
            f"wins={np.mean(values > 0.0):.2f}"
        )

    print(
        "-" * 62
    )

    print(
        f"Baseline: {baseline}"
    )


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--csv",
        type=Path,
        default=None,
        help="Path to Experiment 4 enriched CSV.",
    )

    parser.add_argument(
        "--outdir",
        type=Path,
        default=Path(
            "figures/experiment4"
        ),
    )

    parser.add_argument(
        "--beta",
        type=float,
        default=5.0,
    )

    parser.add_argument(
        "--baseline",
        type=str,
        default="betweenness",
        choices=[
            "betweenness",
            "random",
        ],
        help=(
            "Sparse baseline used to define "
            "Delta rho."
        ),
    )

    parser.add_argument(
        "--topology-col",
        type=str,
        default=None,
    )

    parser.add_argument(
        "--lambda2-col",
        type=str,
        default=None,
    )

    parser.add_argument(
        "--resilience-col",
        type=str,
        default=None,
    )

    parser.add_argument(
        "--method-col",
        type=str,
        default=None,
    )

    parser.add_argument(
        "--beta-col",
        type=str,
        default=None,
    )

    args = parser.parse_args()

    csv_path = (
        args.csv
        if args.csv is not None
        else latest_enriched(
            Path.cwd()
        )
    )

    df = pd.read_csv(
        csv_path
    )

    topology_col = find_column(
        df,
        candidates=[
            "topology_id",
            "graph_id",
            "topology",
            "graph",
            "network_id",
            "network",
            "topology_name",
        ],
        explicit=args.topology_col,
    )

    lambda2_col = find_column(
        df,
        candidates=[
            "lambda2",
            "lambda_2",
            "algebraic_connectivity",
            "spectral_gap",
            "laplacian_lambda2",
        ],
        explicit=args.lambda2_col,
    )

    resilience_col = find_column(
        df,
        candidates=[
            "resilience",
            "rho",
            "resilience_mean",
            "mean_resilience",
        ],
        explicit=args.resilience_col,
    )

    method_col = find_column(
        df,
        candidates=[
            "method",
            "controller",
            "policy",
            "model",
        ],
        explicit=args.method_col,
    )

    beta_col = find_column(
        df,
        candidates=[
            "beta",
            "control_gain",
            "gain",
        ],
        explicit=args.beta_col,
        required=False,
    )

    use_beta = (
        args.beta
        if beta_col is not None
        else None
    )

    print(
        f"[input] {csv_path}"
    )
    print(
        f"[detected] topology   = {topology_col}"
    )
    print(
        f"[detected] lambda2    = {lambda2_col}"
    )
    print(
        f"[detected] resilience = {resilience_col}"
    )
    print(
        f"[detected] method     = {method_col}"
    )
    print(
        f"[detected] beta       = {beta_col}"
    )
    print(
        f"[baseline] {args.baseline}"
    )

    agg = aggregate_topology_level(
        df=df,
        topology_col=topology_col,
        lambda2_col=lambda2_col,
        resilience_col=resilience_col,
        method_col=method_col,
        beta_col=beta_col,
        beta=use_beta,
    )

    advantage = build_advantage_table(
        agg=agg,
        topology_col=topology_col,
        baseline=args.baseline,
    )

    args.outdir.mkdir(
        parents=True,
        exist_ok=True,
    )

    beta_suffix = (
        ""
        if use_beta is None
        else f"_beta{str(use_beta).replace('.', 'p')}"
    )

    table_path = (
        args.outdir
        /
        (
            "fig4B_transfer_advantage_data"
            f"_vs_{args.baseline}"
            f"{beta_suffix}.csv"
        )
    )

    advantage.to_csv(
        table_path,
        index=False,
    )

    print(
        f"[saved] {table_path}"
    )

    print_summary(
        advantage,
        args.baseline,
    )

    make_figure(
        advantage=advantage,
        topology_col=topology_col,
        outdir=args.outdir,
        beta=use_beta,
        baseline=args.baseline,
    )


if __name__ == "__main__":
    main()
