#!/usr/bin/env python3
"""
Standalone Figure 2 for Experiment I.

Direct validation of the theoretical recovery mechanism using the dedicated
beta=5 evaluation produced by experiment1_theory_v2.py (or the corrected
circular-error version of that evaluator).

The figure uses only:
    - NRPC-GL
    - NRPC-L
    - cohesive recovery samples
    - unsaturated samples

and compares

    gamma_e(t) = e(t)^T H(t)e(t) / ||e(t)||^2

against

    lambda_rest^RHS(t)
        = -e(t)^T(coupling(t) + u(t)) / ||e(t)||^2.

Outputs
-------
fig2_theory_validation.pdf
fig2_theory_validation.png
fig2_theory_validation_stats.csv
"""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr


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
mpl.rcParams["legend.frameon"] = False


SCENARIOS = ["drift", "pulse", "cancel"]

SCENARIO_LABELS = {
    "drift": "Frequency drift",
    "pulse": "Transient pulse",
    "cancel": "Dynamics cancellation",
}

METHODS = [
    "nrpc_gl",
    "nrpc_local",
]

METHOD_LABELS = {
    "nrpc_gl": "NRPC-GL",
    "nrpc_local": "NRPC-L",
}

METHOD_STYLE = {
    "nrpc_gl": {
        "marker": "o",
        "zorder": 5,
    },
    "nrpc_local": {
        "marker": "^",
        "zorder": 4,
    },
}


def latest_theory_pkl(root: Path) -> Path:
    """Prefer a run tagged theory_final; otherwise use latest Experiment-I PKL."""

    candidates = sorted(
        root.glob("results/experiment1/theory_final*/results.pkl"),
        key=lambda p: p.stat().st_mtime,
    )

    if not candidates:
        candidates = sorted(
            root.glob("results/experiment1/theory*/results.pkl"),
            key=lambda p: p.stat().st_mtime,
        )

    if not candidates:
        candidates = sorted(
            root.glob("results/experiment1/*/results.pkl"),
            key=lambda p: p.stat().st_mtime,
        )

    if not candidates:
        raise FileNotFoundError(
            "No Experiment-I results.pkl found. Pass --pkl explicitly."
        )

    return candidates[-1]


def save_both(fig, outbase: Path):
    fig.savefig(
        outbase.with_suffix(".pdf"),
        bbox_inches="tight",
    )

    fig.savefig(
        outbase.with_suffix(".png"),
        dpi=300,
        bbox_inches="tight",
    )

    print(f"[saved] {outbase.with_suffix('.pdf')}")
    print(f"[saved] {outbase.with_suffix('.png')}")


def format_axis(ax, xlabel=None, title=None):
    if xlabel is not None:
        ax.set_xlabel(
            xlabel,
            fontsize=16,
            fontweight="bold",
        )

    if title is not None:
        ax.set_title(
            title,
            fontsize=14,
            fontweight="bold",
            pad=10,
        )

    ax.tick_params(
        axis="both",
        labelsize=11.5,
    )

    ax.grid(
        alpha=0.18,
        linewidth=0.8,
    )


def load_samples(pkl_path: Path):
    with open(
        pkl_path,
        "rb",
    ) as f:
        payload = pickle.load(f)

    samples = payload.get(
        "theory_samples",
        [],
    )

    if not samples:
        raise ValueError(
            "The PKL does not contain theory_samples."
        )

    df = pd.DataFrame(
        samples
    )

    required = {
        "scenario",
        "beta",
        "trial",
        "method",
        "gamma_t",
        "gamma_directional",
        "lambda_rest_rhs",
        "cohesive",
        "any_clip",
    }

    missing = sorted(
        required
        -
        set(
            df.columns
        )
    )

    if missing:
        raise ValueError(
            "This PKL was produced by an older evaluator. "
            f"Missing columns: {missing}"
        )

    return df, payload


def filter_theory_samples(
    df: pd.DataFrame,
    theory_beta: float,
):
    """
    Main-paper theoretical subset:
        beta = 5,
        learned controllers only,
        cohesive,
        no clipping.
    """

    out = df[
        df["method"].isin(
            METHODS
        )
        &
        np.isclose(
            df["beta"].astype(float),
            theory_beta,
        )
        &
        (
            df["cohesive"].astype(int)
            ==
            1
        )
        &
        (
            df["any_clip"].astype(int)
            ==
            0
        )
        &
        np.isfinite(
            df["gamma_directional"]
        )
        &
        np.isfinite(
            df["lambda_rest_rhs"]
        )
    ].copy()

    if out.empty:
        raise ValueError(
            "No cohesive, unsaturated theory samples remain after filtering."
        )

    return out


def make_stats(
    theory: pd.DataFrame,
):
    rows = []

    for scenario in SCENARIOS:
        for method in METHODS:

            d = theory[
                (
                    theory["scenario"]
                    ==
                    scenario
                )
                &
                (
                    theory["method"]
                    ==
                    method
                )
            ].copy()

            if d.empty:
                continue

            x = d[
                "gamma_directional"
            ].to_numpy(
                dtype=float
            )

            y = d[
                "lambda_rest_rhs"
            ].to_numpy(
                dtype=float
            )

            # Also retain the worst-case spectral margin for the first
            # theoretical sanity check gamma <= gamma_e.
            gamma_min = d[
                "gamma_t"
            ].to_numpy(
                dtype=float
            )

            directional_gap = (
                x
                -
                gamma_min
            )

            dynamic_gap = (
                y
                -
                x
            )

            rho, pvalue = spearmanr(
                x,
                y,
                nan_policy="omit",
            )

            rows.append(
                {
                    "scenario": scenario,
                    "method": method,
                    "n": len(d),
                    "gamma_directional_mean":
                        float(np.mean(x)),
                    "lambda_rest_rhs_mean":
                        float(np.mean(y)),
                    "spearman_rho":
                        float(rho),
                    "spearman_p":
                        float(pvalue),
                    "fraction_gamma_directional_ge_gamma":
                        float(
                            np.mean(
                                directional_gap
                                >=
                                -1e-10
                            )
                        ),
                    "fraction_lambda_ge_gamma_directional":
                        float(
                            np.mean(
                                dynamic_gap
                                >=
                                -1e-10
                            )
                        ),
                    "mean_dynamic_gap":
                        float(
                            np.mean(
                                dynamic_gap
                            )
                        ),
                }
            )

    return pd.DataFrame(
        rows
    )


def figure2_theory(
    theory: pd.DataFrame,
    outdir: Path,
    theory_beta: float,
):
    """
    Figure 2:
        x = directional theoretical margin gamma_e(t)
        y = empirical RHS restoring rate lambda_rest^RHS(t)
    """

    fig, axes = plt.subplots(
        1,
        3,
        figsize=(16.5, 5.4),
        sharex=True,
        sharey=True,
    )

    all_values = np.concatenate(
        [
            theory[
                "gamma_directional"
            ].to_numpy(
                dtype=float
            ),
            theory[
                "lambda_rest_rhs"
            ].to_numpy(
                dtype=float
            ),
        ]
    )

    all_values = all_values[
        np.isfinite(
            all_values
        )
    ]

    lo = float(
        np.min(
            all_values
        )
    )

    hi = float(
        np.max(
            all_values
        )
    )

    span = max(
        hi - lo,
        1e-6,
    )

    lo -= 0.06 * span
    hi += 0.06 * span

    # Linear scale is preferred here because after the theoretical filters
    # both quantities should live in a compact, physically meaningful range.
    for ax, scenario in zip(
        axes,
        SCENARIOS,
    ):

        sub = theory[
            theory["scenario"]
            ==
            scenario
        ].copy()

        for method in METHODS:

            d = sub[
                sub["method"]
                ==
                method
            ]

            if d.empty:
                continue

            style = METHOD_STYLE[
                method
            ]

            ax.scatter(
                d[
                    "gamma_directional"
                ],
                d[
                    "lambda_rest_rhs"
                ],
                marker=style[
                    "marker"
                ],
                s=45,
                alpha=0.40,
                label=METHOD_LABELS[
                    method
                ],
                zorder=style[
                    "zorder"
                ],
                edgecolors="none",
            )

            # One large marker per method gives the centroid without hiding
            # the instantaneous point cloud.
            # ax.scatter(
            #     [
            #         d[
            #             "gamma_directional"
            #         ].mean()
            #     ],
            #     [
            #         d[
            #             "lambda_rest_rhs"
            #         ].mean()
            #     ],
            #     marker=style[
            #         "marker"
            #     ],
            #     s=125,
            #     linewidths=0.9,
            #     edgecolors="black",
            #     zorder=style[
            #         "zorder"
            #     ]
            #     +
            #     2,
            # )

        reference = np.linspace(
            lo,
            hi,
            400,
        )

        ax.plot(
            reference,
            reference,
            linestyle="--",
            linewidth=1.3,
            alpha=0.8,
            zorder=1,
        )

        x = sub[
            "gamma_directional"
        ].to_numpy(
            dtype=float
        )

        y = sub[
            "lambda_rest_rhs"
        ].to_numpy(
            dtype=float
        )

        rho, pvalue = spearmanr(
            x,
            y,
            nan_policy="omit",
        )

        support_fraction = float(
            np.mean(
                (
                    y
                    -
                    x
                )
                >=
                -1e-10
            )
        )

        format_axis(
            ax,
            xlabel=r"Directional Margin $\gamma_e(t)$",
            title=SCENARIO_LABELS[
                scenario
            ],
        )

        ax.set_xlim(
            lo,
            hi,
        )

        ax.set_ylim(
            lo,
            hi,
        )

        ax.text(
            0.03,
            0.97,
            (
                rf"$\rho_s={rho:.2f}$"
                "\n"
                rf"$\Pr[\lambda_{{\rm rest}}^{{\rm RHS}}\geq"
                rf"\gamma_e]={support_fraction:.2f}$"
            ),
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=10.5,
            bbox=dict(
                boxstyle="round,pad=0.25",
                facecolor="white",
                alpha=0.82,
                edgecolor="gray",
            ),
        )

    axes[0].set_ylabel(
        "Empirical \nRestoring Rate "
        r"$\lambda_{\rm rest}^{\rm RHS}(t)$",
        fontsize=18,
        fontweight="bold",
    )

    handles, labels = axes[-1].get_legend_handles_labels()

    unique = dict(
        zip(
            labels,
            handles,
        )
    )

    fig.legend(
        unique.values(),
        unique.keys(),
        fontsize=11.5,
        loc="lower center",
        bbox_to_anchor=(
            0.5,
            -0.035,
        ),
        ncol=2,
        framealpha=0.9,
    )

    # fig.text(
    #     0.50,
    #     0.955,
    #     (
    #         r"Directional Lyapunov recovery validation "
    #         rf"($\beta={theory_beta:g}$)"
    #     ),
    #     ha="center",
    #     va="top",
    #     fontsize=14,
    #     fontweight="bold",
    # )

    # fig.text(
    #     0.50,
    #     0.905,
    #     (
    #         r"Cohesive and unsaturated recovery samples only; "
    #         r"dashed line: "
    #         r"$\lambda_{\rm rest}^{\rm RHS}(t)=\gamma_e(t)$."
    #     ),
    #     ha="center",
    #     va="top",
    #     fontsize=10.5,
    #     bbox=dict(
    #         boxstyle="round,pad=0.25",
    #         facecolor="cyan",
    #         alpha=0.08,
    #         edgecolor="gray",
    #     ),
    # )

    fig.subplots_adjust(
        left=0.075,
        right=0.99,
        top=0.82,
        bottom=0.22,
        wspace=0.16,
    )

    save_both(
        fig,
        outdir
        /
        "fig2_theory_validation",
    )

    plt.close(
        fig
    )


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--pkl",
        type=Path,
        default=None,
        help=(
            "Path to results.pkl from the dedicated beta=5 theory run. "
            "If omitted, the latest theory run is used."
        ),
    )

    parser.add_argument(
        "--outdir",
        type=Path,
        default=Path(
            "figures/experiment1"
        ),
    )

    parser.add_argument(
        "--beta",
        type=float,
        default=5.0,
        help="Representative theoretical gain.",
    )

    args = parser.parse_args()

    root = Path.cwd()

    pkl_path = (
        args.pkl
        if args.pkl is not None
        else latest_theory_pkl(
            root
        )
    )

    df, payload = load_samples(
        pkl_path
    )

    theory = filter_theory_samples(
        df,
        theory_beta=args.beta,
    )

    args.outdir.mkdir(
        parents=True,
        exist_ok=True,
    )

    print(
        f"[input] {pkl_path}"
    )

    print(
        f"[beta] {args.beta:g}"
    )

    print(
        "[filter] learned methods + cohesive + no clipping"
    )

    print(
        f"[samples] {len(theory)}"
    )

    for scenario in SCENARIOS:
        for method in METHODS:
            n = len(
                theory[
                    (
                        theory["scenario"]
                        ==
                        scenario
                    )
                    &
                    (
                        theory["method"]
                        ==
                        method
                    )
                ]
            )

            print(
                f"  {scenario:<8} "
                f"{method:<12}: "
                f"{n}"
            )

    stats = make_stats(
        theory
    )

    stats_path = (
        args.outdir
        /
        "fig2_theory_validation_stats.csv"
    )

    stats.to_csv(
        stats_path,
        index=False,
    )

    print(
        f"[saved] {stats_path}"
    )

    figure2_theory(
        theory,
        outdir=args.outdir,
        theory_beta=args.beta,
    )


if __name__ == "__main__":
    main()
