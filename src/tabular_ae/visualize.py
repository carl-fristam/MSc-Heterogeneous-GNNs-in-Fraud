"""
visualize.py

Visualisation for the tabular autoencoder pipeline.

1. UMAP of bottleneck latent vectors, coloured by Is_laundering
2. Reconstruction error histogram split by label
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import umap


def plot_umap_tabular(
    z: np.ndarray,
    labels: np.ndarray,
    save_path: str = "results/tabular_ae_umap.png",
    max_sample_genuine: int = 50_000,
    n_neighbors: int = 30,
    min_dist: float = 0.1,
    title: str = "Tabular AE — UMAP of Latent Space",
):
    """
    UMAP scatter of bottleneck latent vectors.

    Subsamples genuine transactions while keeping ALL fraud visible.
    """
    fraud_idx = np.where(labels == 1)[0]
    genuine_idx = np.where(labels == 0)[0]

    if len(genuine_idx) > max_sample_genuine:
        rng = np.random.RandomState(42)
        genuine_idx = rng.choice(genuine_idx, max_sample_genuine, replace=False)

    keep = np.concatenate([genuine_idx, fraud_idx])
    z_sub = z[keep]
    labels_sub = labels[keep]

    print(f"UMAP: {len(keep)} points ({len(genuine_idx)} genuine, {len(fraud_idx)} fraud)")

    reducer = umap.UMAP(
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        n_jobs=-1,
        low_memory=True,
    )
    emb = reducer.fit_transform(z_sub)

    genuine_mask = labels_sub == 0
    fraud_mask = labels_sub == 1

    fig, ax = plt.subplots(figsize=(10, 8))

    ax.scatter(
        emb[genuine_mask, 0], emb[genuine_mask, 1],
        c="#9ecae1", s=2, alpha=0.3, rasterized=True, label="Genuine",
    )
    ax.scatter(
        emb[fraud_mask, 0], emb[fraud_mask, 1],
        c="#e34a33", s=30, alpha=0.9, zorder=5, label="Laundering",
    )

    ax.set_title(title, fontsize=14)
    ax.legend(fontsize=11, markerscale=3)
    ax.set_xticks([])
    ax.set_yticks([])

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"UMAP saved: {save_path}")


def plot_score_histogram(
    scores: np.ndarray,
    labels: np.ndarray,
    save_path: str = "results/tabular_ae_score_hist.png",
    percentile_cap: float = 99.5,
    title: str = "Tabular AE — Reconstruction Error Distribution",
):
    """
    Overlapping histograms + KDE of anomaly scores for genuine vs fraud.
    """
    from scipy.stats import gaussian_kde

    genuine_scores = scores[labels == 0]
    fraud_scores = scores[labels == 1]

    x_max = np.percentile(genuine_scores, percentile_cap)

    fig, ax = plt.subplots(figsize=(10, 6))

    bins = np.linspace(0, x_max, 80)

    ax.hist(genuine_scores, bins=bins, density=True, alpha=0.4,
            color="#9ecae1", label="Genuine", edgecolor="white", linewidth=0.3)
    ax.hist(fraud_scores[fraud_scores <= x_max], bins=bins, density=True, alpha=0.5,
            color="#e34a33", label="Laundering", edgecolor="white", linewidth=0.3)

    x_grid = np.linspace(0, x_max, 300)
    kde_genuine = gaussian_kde(genuine_scores[genuine_scores <= x_max])
    ax.plot(x_grid, kde_genuine(x_grid), color="#2171b5", linewidth=2)

    if len(fraud_scores[fraud_scores <= x_max]) > 1:
        kde_fraud = gaussian_kde(fraud_scores[fraud_scores <= x_max])
        ax.plot(x_grid, kde_fraud(x_grid), color="#cb181d", linewidth=2)

    ax.axvline(genuine_scores.mean(), color="#2171b5", linestyle="--", linewidth=1.2,
               label=f"Genuine mean ({genuine_scores.mean():.4f})")
    ax.axvline(fraud_scores.mean(), color="#cb181d", linestyle="--", linewidth=1.2,
               label=f"Fraud mean ({fraud_scores.mean():.4f})")

    ax.set_xlim(0, x_max)
    ax.set_xlabel("Reconstruction Error (MSE)", fontsize=12)
    ax.set_ylabel("Density", fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.legend(fontsize=10)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Histogram saved: {save_path}")
