"""
Print precision at fixed recall points for all models.
Output is copy-pasteable into a LaTeX table.

Usage:
    PYTHONPATH=. python scripts/print_precision_at_recall.py results/defaults
"""

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import precision_recall_curve, average_precision_score, roc_auc_score

RECALL_POINTS = [1, 5, 10, 50]

ORDER = ["xgboost", "hgt", "hmpnn", "hetero_gat"]
LABELS = {
    "xgboost":    "XGBoost",
    "hgt":        "HGT",
    "hmpnn":      "HMPNN",
    "hetero_gat": "HeteroGAT",
}


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


def find_runs(base: Path) -> dict:
    """Find all runs, keyed by (model, num_layers)."""
    runs = {}
    for npy in sorted(base.rglob("y_true.npy")):
        run_dir = npy.parent
        model_key = _infer_model(run_dir)
        if model_key is None:
            continue
        n_layers = _get_num_layers(run_dir)
        key = (model_key, n_layers)
        runs[key] = run_dir
    return runs


def precision_at_recall(y_true, y_prob, target_recall_pct):
    """Find precision when recall first reaches target_recall_pct%."""
    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    target = target_recall_pct / 100.0
    # precision_recall_curve returns recall in decreasing order
    for i in range(len(recall)):
        if recall[i] <= target:
            return precision[i] * 100
    return 0.0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("results_dir", type=str)
    args = parser.parse_args()

    base = Path(args.results_dir)
    runs = find_runs(base)

    if not runs:
        print(f"No runs found under {base}")
        return

    # Header
    recall_cols = " & ".join(f"{r}" for r in RECALL_POINTS)
    print(f"{'Model':<20} {'Layers':>6}  | " +
          " | ".join(f"P@R={r}%" for r in RECALL_POINTS) +
          " | PR AUC  | ROC AUC")
    print("-" * 100)

    # LaTeX version
    print("\n% LaTeX table rows:")
    print(f"Model & Layers & " +
          " & ".join(f"R={r}\\%" for r in RECALL_POINTS) +
          " & PR AUC & ROC AUC \\\\")
    print("\\midrule")

    for model_key in ORDER:
        model_runs = sorted(
            [(k, v) for k, v in runs.items() if k[0] == model_key],
            key=lambda x: x[0][1] if x[0][1] is not None else 0
        )
        for (mk, n_layers), run_dir in model_runs:
            y_true = np.load(run_dir / "y_true.npy")
            y_prob = np.load(run_dir / "y_prob.npy")

            prec_at_recalls = [precision_at_recall(y_true, y_prob, r)
                               for r in RECALL_POINTS]
            auprc = average_precision_score(y_true, y_prob)
            auroc = roc_auc_score(y_true, y_prob)

            layers_str = str(n_layers) if n_layers is not None else "--"
            label = LABELS[model_key]

            # Console
            prec_str = " | ".join(f"{p:6.2f}" for p in prec_at_recalls)
            print(f"{label:<20} {layers_str:>6}  | {prec_str} | {auprc:.4f}  | {auroc:.4f}")

            # LaTeX
            prec_latex = " & ".join(f"{p:.2f}" for p in prec_at_recalls)
            print(f"% {label} & {layers_str} & {prec_latex} & {auprc:.4f} & {auroc:.4f} \\\\")


if __name__ == "__main__":
    main()
