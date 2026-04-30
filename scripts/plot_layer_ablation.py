"""
Layer ablation plot: PR-AUC vs number of layers for each GNN model.
Styled after Johannessen & Jullum (2023) — thin dashed lines, distinct markers.

Usage:
    PYTHONPATH=. python scripts/plot_layer_ablation.py results/defaults
    PYTHONPATH=. python scripts/plot_layer_ablation.py results/defaults --out figures/layer_ablation.png
"""

import argparse
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import average_precision_score

MODELS = {
    "hgt":        {"label": "HGT",       "color": "#4e79a7", "marker": "o"},
    "hmpnn":      {"label": "HMPNN",      "color": "#f28e2b", "marker": "D"},
    "hetero_gat": {"label": "HeteroGAT",  "color": "#76b7b2", "marker": "s"},
}


def find_runs(base: Path) -> dict:
    """Find all GNN runs, grouped by (model, num_layers)."""
    runs = {}
    for metrics_file in sorted(base.rglob("metrics.json")):
        run_dir = metrics_file.parent
        if not (run_dir / "y_true.npy").exists():
            continue

        with open(metrics_file) as f:
            saved = json.load(f)
        meta = saved.get("meta", {})
        model = meta.get("model")
        hp = meta.get("hyperparams", {})
        n_layers = hp.get("num_layers")

        if model in MODELS and n_layers is not None:
            key = (model, int(n_layers))
            runs[key] = run_dir

    return runs


def plot(runs: dict, out_path: Path):
    fig, ax = plt.subplots(figsize=(5.5, 5))

    ax.set_facecolor("white")
    fig.patch.set_facecolor("white")

    all_layers = sorted(set(k[1] for k in runs.keys()))

    for model_key, style in MODELS.items():
        layers = []
        auprcs = []
        for n_layers in all_layers:
            key = (model_key, n_layers)
            if key not in runs:
                continue
            run_dir = runs[key]
            y_true = np.load(run_dir / "y_true.npy")
            y_prob = np.load(run_dir / "y_prob.npy")
            auprc = average_precision_score(y_true, y_prob)
            layers.append(n_layers)
            auprcs.append(auprc)

        if not layers:
            continue

        ax.plot(layers, auprcs,
                color=style["color"],
                marker=style["marker"],
                markersize=9,
                linewidth=1.2,
                linestyle="--",
                label=style["label"],
                alpha=0.9,
                markeredgecolor="white",
                markeredgewidth=0.8)

    ax.set_xlabel("Layers", fontsize=12, labelpad=8)
    ax.set_ylabel("PR AUC", fontsize=12, labelpad=8)
    ax.set_xticks(all_layers)
    ax.tick_params(labelsize=11)
    ax.legend(fontsize=10, loc="upper left", framealpha=0.95,
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
    parser.add_argument("results_dir", type=str)
    parser.add_argument("--out", type=str, default=None)
    args = parser.parse_args()

    base = Path(args.results_dir)
    runs = find_runs(base)

    if not runs:
        print(f"No layer ablation runs found under {base}")
        return

    print(f"Found: {', '.join(f'{m} {n}L' for m, n in sorted(runs.keys()))}")

    out = Path(args.out) if args.out else base / "layer_ablation.png"
    plot(runs, out)


if __name__ == "__main__":
    main()
