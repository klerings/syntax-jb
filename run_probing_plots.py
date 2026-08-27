import argparse
import os
import pickle

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import seaborn as sns
from helper import get_variants

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

VARIANTS = get_variants()

# Short labels for axis ticks
SHORT_LABELS = {
    "imperative": "IMP",
    "present_active": "INT-PRES-ACT",
    "past_active": "INT-PAST-ACT",
    "future_active": "INT-FUT-ACT",
    "present_passive": "INT-PRES-PASS",
    "declarative": "DECL",
    "nominalized": "NOM",
    "conditional": "COND",
}

# One distinct color per in-domain variant (for the layerwise plot)
VARIANT_COLORS = {
    "imperative": "#CC6677",
    "present_active": "#DDCC77",
    "past_active": "#117733",
    "future_active": "#44AA99",
    "present_passive": "#88CCEE",
    "declarative": "#332288",
    "nominalized": "#AA4499",
    "conditional": "#882255",
}


# ---------------------------------------------------------------------------
# Plot 1: Cluster accuracy heatmap
# ---------------------------------------------------------------------------


def plot_heatmap(acc_cl_results: dict, model_name: str, pos: str, out_path: str):
    """
    acc_cl_results: dict[train_variant][test_variant] = float
    """
    # Build DataFrame: rows = train, cols = test
    data = pd.DataFrame(
        {
            test_v: {train_v: acc_cl_results[train_v][test_v] for train_v in VARIANTS}
            for test_v in VARIANTS
        }
    )
    data.index = [SHORT_LABELS[v] for v in VARIANTS]
    data.columns = [SHORT_LABELS[v] for v in VARIANTS]

    fig, ax = plt.subplots(figsize=(7, 5.5))

    sns.heatmap(
        data,
        ax=ax,
        annot=True,
        fmt=".2f",
        cmap="Blues",
        vmin=0.5,
        vmax=1.0,
        linewidths=0.5,
        linecolor="#cccccc",
        annot_kws={"size": 10, "weight": "normal"},
        cbar_kws={"label": "Cluster Accuracy", "shrink": 0.85},
    )

    ax.set_title(
        f"{model_name}  —  Cluster Probe Accuracy ({pos})", fontsize=13, pad=14
    )
    ax.set_xlabel("Test variant", fontsize=11, labelpad=8)
    ax.set_ylabel("Train variant", fontsize=11, labelpad=8)
    ax.tick_params(axis="both", labelsize=10)
    ax.xaxis.set_label_position("bottom")
    ax.xaxis.tick_bottom()

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved heatmap  -> {out_path}")


# ---------------------------------------------------------------------------
# Plot 2: Layerwise in-domain accuracy
# ---------------------------------------------------------------------------


def plot_layerwise(
    acc_lw_results: dict,
    model_name: str,
    pos: str,
    out_path: str,
    acc_lw_std: dict = None,
):
    """
    acc_lw_results: dict[train_variant][test_variant][layer] = float
    Only plots the in-domain diagonal: train_variant == test_variant.
    """
    fig, ax = plt.subplots(figsize=(9, 5))

    for variant in VARIANTS:
        layer_acc = acc_lw_results[variant][variant]  # in-domain only
        layers = sorted(layer_acc.keys())
        accs = [layer_acc[l] for l in layers]

        ax.plot(
            layers,
            accs,
            label=SHORT_LABELS[variant],
            color=VARIANT_COLORS[variant],
            linewidth=2,
            marker="o",
            markersize=3.5,
            alpha=0.9,
        )

        if acc_lw_std is not None:
            stds = [acc_lw_std[variant][variant][l] for l in layers]
            lo = [a - s for a, s in zip(accs, stds)]
            hi = [a + s for a, s in zip(accs, stds)]
            ax.fill_between(
                layers, lo, hi, color=VARIANT_COLORS[variant], alpha=0.15, linewidth=0
            )

    ax.set_title(
        f"{model_name}  —  Layerwise Diff-in-Means (in-domain) - {pos}",
        fontsize=13,
        pad=12,
    )
    ax.set_xlabel("Layer", fontsize=11)
    ax.set_ylabel("Accuracy", fontsize=11)
    ax.set_ylim(0.45, 1.02)
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))
    ax.axhline(
        0.5, color="gray", linewidth=0.8, linestyle="--", alpha=0.6, label="Chance"
    )
    ax.legend(
        title="Variant",
        fontsize=9,
        title_fontsize=9,
        loc="lower right",
        framealpha=0.85,
    )
    ax.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.5)
    ax.grid(axis="x", linestyle=":", linewidth=0.4, alpha=0.4)
    ax.tick_params(axis="both", labelsize=10)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved layerwise -> {out_path}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--posistion", type=str, default="mean")
    parser.add_argument("--pred", type=str, default="hard", choices=["hard", "soft"])
    parser.add_argument("--models", type=str, nargs="+", default=None)
    parser.add_argument(
        "--cv",
        action="store_true",
        help="Look for CV results (must match flag used in probing.py).",
    )
    parser.add_argument("--n_folds", type=int, default=5)
    parser.add_argument(
        "--out_dir", type=str, default=None, help="Override output directory for plots."
    )
    args = parser.parse_args()

    model_names = args.models
    suffix = f"cv{args.n_folds}" if args.cv else "split"
    probe_dir = f"probing/variant_probes_{args.position}_{args.pred}_{suffix}"
    plot_dir = args.out_dir or f"plots/plots_{args.position}_{args.pred}_{suffix}"
    os.makedirs(plot_dir, exist_ok=True)

    for model_name in model_names:
        lw_path = os.path.join(probe_dir, f"{model_name}_layerwise.pkl")
        cl_path = os.path.join(probe_dir, f"{model_name}_cluster_heatmap.csv")

        if not os.path.exists(lw_path):
            print(f"Skipping {model_name}: {lw_path} not found.")
            continue
        if not os.path.exists(cl_path):
            print(f"Skipping {model_name}: {cl_path} not found.")
            continue

        print(f"\n=== {model_name} ===")

        # Load layerwise pickle.
        with open(lw_path, "rb") as f:
            lw_payload = pickle.load(f)
        if isinstance(lw_payload, dict) and "mean" in lw_payload:
            acc_lw_results = lw_payload["mean"]
            acc_lw_std = lw_payload.get("std", None)
        else:
            acc_lw_results = lw_payload  # backwards-compatible
            acc_lw_std = None

        # Load cluster CSV and reconstruct nested dict[train_v][test_v] = float
        cl_df = pd.read_csv(cl_path, index_col=0)
        acc_cl_results = {
            train_v: {test_v: float(cl_df.loc[train_v, test_v]) for test_v in VARIANTS}
            for train_v in VARIANTS
        }

        plot_heatmap(
            acc_cl_results,
            model_name,
            args.position,
            os.path.join(plot_dir, f"{model_name}_cluster_heatmap.png"),
        )
        plot_layerwise(
            acc_lw_results,
            model_name,
            args.position,
            os.path.join(plot_dir, f"{model_name}_layerwise.png"),
            acc_lw_std=acc_lw_std,
        )
