"""
Plot overlaid PR curves for all models.

Scans a results directory for y_true.npy / y_prob.npy pairs and plots them
on a single figure.

Usage:
    python scripts/plot_pr_comparison.py results/defaults
    python scripts/plot_pr_comparison.py results/defaults --out figures/pr_comparison.png
"""

import argparse
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, average_precision_score

COLORS = {
    "xgboost": "#1e293b",
    "hgt":     "#2563eb",
    "hmpnn":   "#dc2626",
    "hetero_gat": "#16a34a",
}

LABELS = {
    "xgboost": "XGBoost",
    "hgt": "HGT",
    "hmpnn": "HMPNN",
    "hetero_gat": "HeteroGAT",
}


def find_runs(base: Path) -> dict:
    """Find the most recent run with y_true.npy for each model."""
    runs = {}
    for npy in sorted(base.rglob("y_true.npy")):
        run_dir = npy.parent
        model_key = _infer_model(run_dir)
        if model_key:
            runs[model_key] = run_dir
    return runs


def _infer_model(run_dir: Path) -> str | None:
    name = run_dir.name.lower()
    for key in ("hgt", "hmpnn", "hetero_gat", "xgboost"):
        if key in name:
            return key
    parent = run_dir.parent.name.lower()
    for key in ("hgt", "hmpnn", "hetero_gat", "xgboost", "tabular"):
        if key in parent:
            return "xgboost" if key == "tabular" else key
    return None


def plot(runs: dict, out_path: Path):
    fig, ax = plt.subplots(figsize=(8, 6))

    for model_key in ["xgboost", "hgt", "hmpnn", "hetero_gat"]:
        if model_key not in runs:
            continue
        run_dir = runs[model_key]
        y_true = np.load(run_dir / "y_true.npy")
        y_prob = np.load(run_dir / "y_prob.npy")

        precision, recall, _ = precision_recall_curve(y_true, y_prob)
        auprc = average_precision_score(y_true, y_prob)

        label = f"{LABELS.get(model_key, model_key)}  (PR-AUC = {auprc:.4f})"
        color = COLORS.get(model_key, "#6b7280")
        ax.plot(recall, precision, linewidth=2, color=color, label=label)

    ax.set_xlabel("Recall", fontsize=13)
    ax.set_ylabel("Precision", fontsize=13)
    ax.set_title("Precision–Recall Curves", fontsize=14, pad=12)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    ax.legend(fontsize=11, loc="upper right")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200)
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

    out = Path(args.out) if args.out else base / "pr_comparison.png"
    plot(runs, out)


if __name__ == "__main__":
    main()
