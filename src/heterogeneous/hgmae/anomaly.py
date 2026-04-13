"""
Anomaly scoring and visualisation for HGMAE.

Edge-level anomaly score = mean reconstruction error of the source and
destination account nodes. Higher score = more anomalous transaction.

Visualisations produced:
  1. Score distribution — fraud vs legitimate overlaid histogram
  2. Precision-recall curve with AUPRC
  3. ROC curve with AUROC
  4. Threshold table — same operational format as supervised models

These answer the evaluation question: even without a supervised classifier,
reconstruction error can serve as an unsupervised fraud signal. AUPRC and
AUROC measure how well the signal separates fraud from legitimate.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
    precision_recall_curve,
    roc_curve,
)

from src.utils.threshold_table import print_threshold_table


def score_edges(model, data) -> dict:
    """
    Compute per-edge anomaly scores from node reconstruction errors.

    Score for edge (u, v) = (error[u] + error[v]) / 2

    Returns:
        dict {edge_type: (E,) numpy array of anomaly scores}
    """
    node_errors = model.reconstruction_error(data)
    node_errors_np = {nt: err.cpu().numpy() for nt, err in node_errors.items()}

    scores = {}
    for et in data.edge_types:
        src_type, _, dst_type = et
        ei = data[et].edge_index.cpu().numpy()
        src_err = node_errors_np[src_type][ei[0]]
        dst_err = node_errors_np[dst_type][ei[1]]
        scores[et] = (src_err + dst_err) / 2

    return scores


def evaluate_anomaly(model, data, save_dir: str = "outputs/hgmae") -> dict:
    """
    Score all edges, evaluate against fraud labels, print metrics,
    and save visualisation plots.

    Args:
        model:    trained HGMAEModel
        data:     PyG HeteroData with .y and .test_mask on edge types
        save_dir: directory to save plots

    Returns:
        dict of metrics (auroc, auprc per edge type + combined)
    """
    Path(save_dir).mkdir(parents=True, exist_ok=True)

    edge_scores = score_edges(model, data)

    all_scores = []
    all_labels = []
    all_amounts = []

    for et in data.edge_types:
        if not hasattr(data[et], "y") or data[et].y is None:
            continue

        mask   = data[et].test_mask.cpu().numpy().astype(bool)
        labels = data[et].y.cpu().numpy()[mask]
        scores = edge_scores[et][mask]

        all_scores.append(scores)
        all_labels.append(labels)

        if hasattr(data[et], "amounts") and data[et].amounts is not None:
            all_amounts.append(data[et].amounts.cpu().numpy()[mask])

    if not all_scores:
        print("No labelled edges found — cannot evaluate.")
        return {}

    scores_combined = np.concatenate(all_scores)
    labels_combined = np.concatenate(all_labels)
    amounts_combined = np.concatenate(all_amounts) if all_amounts else None

    auroc = roc_auc_score(labels_combined, scores_combined) if labels_combined.sum() > 0 else 0.0
    auprc = average_precision_score(labels_combined, scores_combined) if labels_combined.sum() > 0 else 0.0

    print(f"\n{'='*50}")
    print("HGMAE Anomaly Detection — TEST RESULTS")
    print(f"{'='*50}")
    print(f"  AUROC:  {auroc:.4f}")
    print(f"  AUPRC:  {auprc:.4f}")
    print(f"  Fraud:  {int(labels_combined.sum())} / {len(labels_combined)} "
          f"({100 * labels_combined.mean():.2f}%)")

    print_threshold_table(
        labels_combined, scores_combined,
        amounts=amounts_combined,
        model_name="HGMAE (anomaly)",
    )

    _plot_score_distribution(scores_combined, labels_combined, save_dir)
    _plot_pr_curve(labels_combined, scores_combined, auprc, save_dir)
    _plot_roc_curve(labels_combined, scores_combined, auroc, save_dir)

    return {"auroc": auroc, "auprc": auprc}


# ── Plots ─────────────────────────────────────────────────────────────────────

def _plot_score_distribution(scores, labels, save_dir):
    """Histogram of anomaly scores separated by fraud/legitimate."""
    fig, ax = plt.subplots(figsize=(8, 4))

    legit_scores = scores[labels == 0]
    fraud_scores = scores[labels == 1]

    ax.hist(legit_scores, bins=80, alpha=0.6, color="steelblue",
            label=f"Legitimate (n={len(legit_scores):,})", density=True)
    ax.hist(fraud_scores, bins=80, alpha=0.7, color="crimson",
            label=f"Fraud (n={len(fraud_scores):,})", density=True)

    ax.set_xlabel("Reconstruction error (anomaly score)")
    ax.set_ylabel("Density")
    ax.set_title("HGMAE anomaly score distribution")
    ax.legend()
    fig.tight_layout()
    path = Path(save_dir) / "score_distribution.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {path}")


def _plot_pr_curve(labels, scores, auprc, save_dir):
    """Precision-recall curve."""
    precision, recall, _ = precision_recall_curve(labels, scores)

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(recall, precision, color="steelblue", lw=2,
            label=f"AUPRC = {auprc:.4f}")
    ax.axhline(labels.mean(), color="gray", linestyle="--", lw=1,
               label=f"Baseline (fraud rate = {labels.mean():.3%})")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("HGMAE precision-recall curve")
    ax.legend()
    fig.tight_layout()
    path = Path(save_dir) / "pr_curve.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {path}")


def _plot_roc_curve(labels, scores, auroc, save_dir):
    """ROC curve."""
    fpr, tpr, _ = roc_curve(labels, scores)

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, color="steelblue", lw=2,
            label=f"AUROC = {auroc:.4f}")
    ax.plot([0, 1], [0, 1], color="gray", linestyle="--", lw=1, label="Random")
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title("HGMAE ROC curve")
    ax.legend()
    fig.tight_layout()
    path = Path(save_dir) / "roc_curve.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {path}")
