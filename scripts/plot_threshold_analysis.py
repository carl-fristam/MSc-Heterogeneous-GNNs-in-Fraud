"""
Threshold analysis plot: precision and recall vs decision threshold for each model.
Shows how each model's operating characteristics change across thresholds.

Usage:
    PYTHONPATH=. python scripts/plot_threshold_analysis.py results/defaults
    PYTHONPATH=. python scripts/plot_threshold_analysis.py results/defaults --out figures/threshold_analysis.png
"""

import argparse
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

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


def find_runs(base: Path) -> dict:
    """Find the most recent 2-layer (or default) run per model."""
    runs = {}
    for npy in sorted(base.rglob("y_true.npy")):
        run_dir = npy.parent
        model_key = _infer_model(run_dir)
        if model_key is None:
            continue

        # Prefer 2-layer runs for GNNs
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


def plot(runs: dict, out_path: Path):
    n_models = len(runs)
    fig, axes = plt.subplots(1, n_models, figsize=(5 * n_models, 4.5),
                              squeeze=False)
    fig.patch.set_facecolor("white")

    thresholds = np.arange(0.01, 0.99, 0.01)

    order = ["xgboost", "hgt", "hmpnn", "hetero_gat"]

    col = 0
    for model_key in order:
        if model_key not in runs:
            continue
        ax = axes[0, col]
        ax.set_facecolor("white")

        run_dir = runs[model_key]
        y_true = np.load(run_dir / "y_true.npy")
        y_prob = np.load(run_dir / "y_prob.npy")

        precisions = []
        recalls = []
        for t in thresholds:
            preds = (y_prob >= t).astype(int)
            tp = ((preds == 1) & (y_true == 1)).sum()
            fp = ((preds == 1) & (y_true == 0)).sum()
            fn = ((preds == 0) & (y_true == 1)).sum()
            p = tp / (tp + fp) if (tp + fp) > 0 else 0
            r = tp / (tp + fn) if (tp + fn) > 0 else 0
            precisions.append(p)
            recalls.append(r)

        color = COLORS[model_key]
        ax.plot(thresholds, precisions, linewidth=1.3, color=color,
                linestyle="-", label="Precision", alpha=0.9)
        ax.plot(thresholds, recalls, linewidth=1.3, color=color,
                linestyle="--", label="Recall", alpha=0.9)

        # Find optimal F1 threshold
        f1s = [2 * p * r / (p + r) if (p + r) > 0 else 0
               for p, r in zip(precisions, recalls)]
        best_idx = np.argmax(f1s)
        best_t = thresholds[best_idx]
        ax.axvline(x=best_t, color=color, linewidth=0.8,
                   linestyle=":", alpha=0.5)
        ax.text(best_t + 0.02, 0.95, f"t={best_t:.2f}",
                fontsize=8, color=color, alpha=0.7,
                transform=ax.get_xaxis_transform())

        ax.set_xlabel("Threshold", fontsize=11, labelpad=6)
        if col == 0:
            ax.set_ylabel("Score", fontsize=11, labelpad=6)
        ax.set_title(LABELS[model_key], fontsize=12, fontweight="bold", pad=10)
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1.05])
        ax.tick_params(labelsize=10)
        ax.legend(fontsize=9, loc="upper right", framealpha=0.95,
                  edgecolor="#cccccc", fancybox=False)
        ax.grid(True, alpha=0.2, linewidth=0.5)

        for spine in ax.spines.values():
            spine.set_color("#cccccc")
            spine.set_linewidth(0.6)

        col += 1

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved: {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("results_dir", type=str)
    parser.add_argument("--out", type=str, default=None)
    args = parser.parse_args()

    base = Path(args.results_dir)
    runs = find_runs(base)

    if not runs:
        print(f"No runs found under {base}")
        return

    print(f"Found: {', '.join(f'{k} ({v.name})' for k, v in runs.items())}")

    out = Path(args.out) if args.out else base / "threshold_analysis.png"
    plot(runs, out)


if __name__ == "__main__":
    main()
