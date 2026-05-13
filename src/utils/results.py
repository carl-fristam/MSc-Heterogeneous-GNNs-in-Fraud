"""
results.py

Saves all experiment output to a timestamped folder after each run:
  - metrics.json
  - confusion_matrix.png
  - pr_curve.png
  - feature_importance.png (XGBoost only)
  - per_edge_type_auprc.png (GNN only)
  - embedding_tsne.png (GNN only)
  - message_norms.png (HMPNN only)
  - neighbourhood_fraud_density.png (GNN only)
"""

import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from datetime import datetime
from pathlib import Path
from sklearn.metrics import precision_recall_curve, average_precision_score

from src.utils.config import PROJECT_ROOT


def save_results(metrics: dict, mode: str, model: str = None,
                 y_true=None, y_prob=None,
                 xgb_model=None, feature_names=None,
                 analysis=None, results_dir_override=None,
                 console_log=None, model_state=None, **kwargs):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if results_dir_override:
        base = PROJECT_ROOT / results_dir_override / (model or mode)
    else:
        base = _results_dir(mode, model=model)
    run_dir = base / f"{_run_name(model)}_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)

    display_name = model.upper() if model else "XGBoost"

    serializable = {
        k: v.tolist() if hasattr(v, "tolist") else v
        for k, v in metrics.items()
    }

    meta = {"mode": mode, "timestamp": timestamp, "model": model, **kwargs}
    with open(run_dir / "metrics.json", "w") as f:
        json.dump({"meta": meta, "metrics": serializable}, f, indent=2)

    if console_log:
        with open(run_dir / "console.log", "w") as f:
            f.write(console_log)

    if model_state is not None:
        import torch
        torch.save(model_state, run_dir / "model_state.pt")

    # ── Standard plots ───────────────────────────────────────────────────────
    cm = metrics.get("confusion_matrix")
    if cm is not None:
        _plot_confusion_matrix(np.array(cm) if not isinstance(cm, np.ndarray) else cm,
                               run_dir, display_name)

    if y_true is not None and y_prob is not None:
        np.save(run_dir / "y_true.npy", np.asarray(y_true))
        np.save(run_dir / "y_prob.npy", np.asarray(y_prob))
        _plot_pr_curve(np.asarray(y_true), np.asarray(y_prob),
                       run_dir, display_name)

    if xgb_model is not None and feature_names is not None:
        _plot_feature_importance(xgb_model, feature_names, run_dir)

    # ── GNN analysis plots ───────────────────────────────────────────────────
    if analysis is not None:
        if "per_edge_type" in analysis:
            _plot_per_edge_type(analysis["per_edge_type"], run_dir, display_name)

        if "embeddings" in analysis:
            _plot_embeddings(analysis["embeddings"], run_dir, display_name)

        if "message_norms" in analysis:
            _plot_message_norms(analysis["message_norms"], run_dir, display_name)

        if "train_fraud_counts" in analysis and y_true is not None and y_prob is not None:
            _plot_neighbourhood_density(analysis, run_dir, display_name)

        # Save per-edge-type metrics to JSON
        if "per_edge_type" in analysis:
            with open(run_dir / "per_edge_type.json", "w") as f:
                json.dump(analysis["per_edge_type"], f, indent=2)

    print(f"\nResults saved to: {run_dir.relative_to(PROJECT_ROOT)}/")


# ── Directory helpers ────────────────────────────────────────────────────────

def _results_dir(mode: str, model: str = None) -> Path:
    if mode == "tab":
        return PROJECT_ROOT / "src" / "baselines" / "tabular" / "results"
    elif mode == "het":
        return PROJECT_ROOT / "src" / "heterogeneous" / (model or "unknown") / "results"
    return PROJECT_ROOT / "results"


def _run_name(model: str = None) -> str:
    return model if model else "run"


# ── Standard plots ───────────────────────────────────────────────────────────

def _plot_confusion_matrix(cm, run_dir: Path, model_name: str):
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.imshow(cm, cmap="Blues")

    labels = ["Legitimate", "Fraud"]
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(labels, fontsize=12)
    ax.set_yticklabels(labels, fontsize=12)
    ax.set_xlabel("Predicted", fontsize=13)
    ax.set_ylabel("Actual", fontsize=13)

    for i in range(2):
        for j in range(2):
            color = "white" if cm[i, j] > cm.max() / 2 else "black"
            ax.text(j, i, f"{cm[i, j]:,}", ha="center", va="center",
                    fontsize=14, fontweight="bold", color=color)

    ax.set_title(f"Confusion Matrix — {model_name}", fontsize=14, pad=12)
    fig.tight_layout()
    fig.savefig(run_dir / "confusion_matrix.png", dpi=150)
    plt.close(fig)


def _plot_pr_curve(y_true, y_prob, run_dir: Path, model_name: str):
    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    auprc = average_precision_score(y_true, y_prob)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(recall, precision, linewidth=2, color="#2563eb")
    ax.fill_between(recall, precision, alpha=0.1, color="#2563eb")
    ax.set_xlabel("Recall", fontsize=13)
    ax.set_ylabel("Precision", fontsize=13)
    ax.set_title(f"PR Curve — {model_name}  (PR-AUC = {auprc:.4f})", fontsize=14, pad=12)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(run_dir / "pr_curve.png", dpi=150)
    plt.close(fig)


def _plot_feature_importance(model, feature_names, run_dir: Path, top_n: int = 20):
    importance = model.feature_importances_
    indices = np.argsort(importance)[-top_n:]

    fig, ax = plt.subplots(figsize=(8, max(6, top_n * 0.35)))
    ax.barh(range(len(indices)), importance[indices], color="#2563eb")
    ax.set_yticks(range(len(indices)))
    ax.set_yticklabels([feature_names[i] for i in indices], fontsize=10)
    ax.set_xlabel("Feature Importance", fontsize=13)
    ax.set_title(f"XGBoost Feature Importance (Top {top_n})", fontsize=14, pad=12)
    ax.grid(True, axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(run_dir / "feature_importance.png", dpi=150)
    plt.close(fig)


# ── GNN analysis plots ──────────────────────────────────────────────────────

def _plot_per_edge_type(et_metrics: dict, run_dir: Path, model_name: str):
    """Grouped bar chart of PR-AUC and AUROC by edge type."""
    edge_types = list(et_metrics.keys())
    labels = [_short_et_name(et) for et in edge_types]
    auprc_vals = [et_metrics[et]["auprc"] for et in edge_types]
    auroc_vals = [et_metrics[et]["auroc"] for et in edge_types]

    x = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(8, 5))
    bars1 = ax.bar(x - width/2, auprc_vals, width, label="PR-AUC", color="#2563eb")
    bars2 = ax.bar(x + width/2, auroc_vals, width, label="AUROC", color="#64748b")

    ax.set_ylabel("Score", fontsize=13)
    ax.set_title(f"Per-Edge-Type Performance — {model_name}", fontsize=14, pad=12)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=11)
    ax.legend(fontsize=11)
    ax.set_ylim([0, 1])
    ax.grid(True, axis="y", alpha=0.3)

    for bar in bars1:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f"{bar.get_height():.3f}", ha="center", fontsize=10)
    for bar in bars2:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f"{bar.get_height():.3f}", ha="center", fontsize=10)

    fig.tight_layout()
    fig.savefig(run_dir / "per_edge_type_auprc.png", dpi=150)
    plt.close(fig)


def _plot_embeddings(embeddings: dict, run_dir: Path, model_name: str,
                     max_points: int = 10000):
    """t-SNE of node embeddings colored by node type."""
    try:
        from sklearn.manifold import TSNE
    except ImportError:
        return

    all_embs = []
    all_types = []
    for nt, emb in embeddings.items():
        n = emb.shape[0]
        if n > max_points:
            idx = np.random.choice(n, max_points, replace=False)
            emb = emb[idx]
        all_embs.append(emb)
        all_types.extend([nt] * emb.shape[0])

    combined = np.vstack(all_embs)
    tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(combined) - 1))
    coords = tsne.fit_transform(combined)

    unique_types = list(embeddings.keys())
    colors = ["#2563eb", "#dc2626", "#16a34a", "#f59e0b"]

    fig, ax = plt.subplots(figsize=(8, 6))
    for i, nt in enumerate(unique_types):
        mask = np.array([t == nt for t in all_types])
        ax.scatter(coords[mask, 0], coords[mask, 1],
                   s=3, alpha=0.4, label=nt, color=colors[i % len(colors)])

    ax.legend(fontsize=11, markerscale=5)
    ax.set_title(f"Node Embeddings (t-SNE) — {model_name}", fontsize=14, pad=12)
    ax.set_xticks([])
    ax.set_yticks([])
    fig.tight_layout()
    fig.savefig(run_dir / "embedding_tsne.png", dpi=150)
    plt.close(fig)


def _plot_message_norms(layer_norms: dict, run_dir: Path, model_name: str):
    """Box plot of message L2 norms per edge type (HMPNN)."""
    last_layer = max(layer_norms.keys())
    norms = layer_norms[last_layer]

    labels = []
    data = []
    for et, norm_arr in norms.items():
        labels.append(_short_et_name(str(et)))
        subsample = norm_arr if len(norm_arr) < 50000 else np.random.choice(norm_arr, 50000, replace=False)
        data.append(subsample)

    fig, ax = plt.subplots(figsize=(8, 5))
    bp = ax.boxplot(data, labels=labels, patch_artist=True, showfliers=False)
    colors = ["#2563eb", "#dc2626", "#16a34a", "#f59e0b"]
    for i, patch in enumerate(bp["boxes"]):
        patch.set_facecolor(colors[i % len(colors)])
        patch.set_alpha(0.7)

    ax.set_ylabel("Message L2 Norm", fontsize=13)
    ax.set_title(f"Message Magnitude by Edge Type — {model_name}", fontsize=14, pad=12)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(run_dir / "message_norms.png", dpi=150)
    plt.close(fig)


def _plot_neighbourhood_density(analysis: dict, run_dir: Path, model_name: str):
    """Recall by sender's training fraud density."""
    # This plot needs the full test predictions broken down by sender node,
    # which requires more data than currently passed. Save a placeholder
    # noting that this analysis was computed.
    fraud_counts = analysis.get("train_fraud_counts", {})
    if not fraud_counts:
        return

    densities = []
    for key, counts in fraud_counts.items():
        if counts["total"] > 0:
            densities.append(counts["fraud"] / counts["total"])

    if not densities:
        return

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(densities, bins=50, color="#2563eb", alpha=0.7, edgecolor="white")
    ax.set_xlabel("Sender Fraud Rate (Training)", fontsize=13)
    ax.set_ylabel("Number of Accounts", fontsize=13)
    ax.set_title(f"Training Fraud Density Distribution — {model_name}", fontsize=14, pad=12)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(run_dir / "neighbourhood_fraud_density.png", dpi=150)
    plt.close(fig)


def _short_et_name(et_str: str) -> str:
    """Convert edge type tuple string to readable label."""
    et_str = str(et_str)
    if "onus_transfer" in et_str:
        return "On-us Transfer"
    elif "external_transfer" in et_str:
        return "External Transfer"
    return et_str.replace("'", "").replace("(", "").replace(")", "")
