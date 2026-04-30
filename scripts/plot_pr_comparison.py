"""
Plot overlaid PR curves for all models.

Scans a results directory for y_true.npy / y_prob.npy pairs and plots them
on a single figure. Uses the 2-layer run for GNNs if both 1L and 2L exist.
Produces two variants: one with dashed XGBoost line, one with all solid.

Usage:
    PYTHONPATH=. python scripts/plot_pr_comparison.py results/defaults
    PYTHONPATH=. python scripts/plot_pr_comparison.py results/defaults --out figures/pr_comparison.png
"""

import argparse
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, average_precision_score

COLORS = {
    "xgboost":    "#1b1b1b",
    "hgt":        "#4e79a7",
    "hmpnn":      "#f28e2b",
    "hetero_gat": "#76b7b2",
}

LABELS = {
    "xgboost":    "XGBoost",
    "hgt":        "HGT",
    "hmpnn":      "HMPNN",
    "hetero_gat": "HeteroGAT",
}

MARKERS = {
    "xgboost":    "^",
    "hgt":        "o",
    "hmpnn":      "D",
    "hetero_gat": "s",
}


def find_runs(base: Path) -> dict:
    """Find the most recent 2-layer (or default) run per model."""
    runs = {}
    for npy in sorted(base.rglob("y_true.npy")):
        run_dir = npy.parent
        model_key = _infer_model(run_dir)
        if model_key is None:
            continue

        n_layers = _get_num_layers(run_dir)
        existing = runs.get(model_key)
        if existing is None:
            runs[model_key] = run_dir
        elif model_key != "xgboost" and n_layers == 2:
            runs[model_key] = run_dir

    return runs


def _infer_model(run_dir: Path) -> str | None:
    name = run_dir.name.lower()
    for key in ("hgt", "hmpnn", "hetero_gat", "xgboost"):
        if key in name:
            return key
    parent = run_dir.parent.name.lower()
    for key in ("hgt", "hmpnn", "hetero_gat", "xgboost"):
        if key in parent:
            return key
    if parent in ("tab", "tabular"):
        return "xgboost"
    return None


def _get_num_layers(run_dir: Path) -> int | None:
    metrics_file = run_dir / "metrics.json"
    if not metrics_file.exists():
        return None
    with open(metrics_file) as f:
        saved = json.load(f)
    return saved.get("meta", {}).get("hyperparams", {}).get("num_layers")


def plot(runs: dict, out_path: Path, dashed_baseline: bool = False):
    fig, ax = plt.subplots(figsize=(6, 5.5))

    ax.set_facecolor("white")
    fig.patch.set_facecolor("white")

    order = ["xgboost", "hgt", "hmpnn", "hetero_gat"]

    for model_key in order:
        if model_key not in runs:
            continue
        run_dir = runs[model_key]
        y_true = np.load(run_dir / "y_true.npy")
        y_prob = np.load(run_dir / "y_prob.npy")

        precision, recall, _ = precision_recall_curve(y_true, y_prob)
        auprc = average_precision_score(y_true, y_prob)

        label = f"{LABELS[model_key]}  (PR-AUC = {auprc:.4f})"
        color = COLORS[model_key]
        ls = "--" if (dashed_baseline and model_key == "xgboost") else "-"

        ax.plot(recall, precision, linewidth=1.3, color=color,
                linestyle=ls, label=label, alpha=0.9)

    ax.set_xlabel("Recall", fontsize=12, labelpad=8)
    ax.set_ylabel("Precision", fontsize=12, labelpad=8)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.05])
    ax.tick_params(labelsize=11)
    ax.legend(fontsize=10, loc="upper right", framealpha=0.95,
              edgecolor="#cccccc", fancybox=False)
    ax.grid(True, alpha=0.2, linewidth=0.5)

    for spine in ax.spines.values():
        spine.set_color("#cccccc")
        spine.set_linewidth(0.6)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved: {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("results_dir", type=str,
                        help="Base results directory to scan for y_true.npy files")
    parser.add_argument("--out", type=str, default=None,
                        help="Output path (default: <results_dir>/pr_comparison.png)")
    args = parser.parse_args()

    base = Path(args.results_dir)
    runs = find_runs(base)

    if not runs:
        print(f"No y_true.npy files found under {base}")
        return

    print(f"Found runs: {', '.join(f'{k} ({v.name})' for k, v in runs.items())}")

    out_stem = Path(args.out) if args.out else base / "pr_comparison.png"
    out_dashed = out_stem.with_name(out_stem.stem + "_dashed" + out_stem.suffix)

    plot(runs, out_stem, dashed_baseline=False)
    plot(runs, out_dashed, dashed_baseline=True)


if __name__ == "__main__":
    main()
