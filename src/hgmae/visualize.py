"""
visualize.py

UMAP visualisation of HGMAE node embeddings.

Produces a 2-D scatter plot coloured by laundering label.
Laundering nodes are a tiny minority (~0.14%), so they are
overplotted on top of the background with a distinct colour and
larger marker so they don't disappear into the crowd.

Usage (called from train.py when --visualize is passed):
    from src.hgmae.visualize import plot_umap
    plot_umap(embeds, labels_np, save_path="results/hgmae_umap.png")
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import umap


def plot_umap(
    embeds: np.ndarray,
    labels: np.ndarray,
    save_path: str = "results/hgmae_umap.png",
    n_neighbors: int = 30,
    min_dist: float = 0.1,
    max_sample: int = 50_000,
    title: str = "HGMAE Node Embeddings — UMAP Projection",
):
    """
    Project embeddings to 2-D with UMAP and plot coloured by label.

    Args:
        embeds:       np.ndarray of shape [N, D] — frozen encoder output
        labels:       np.ndarray of shape [N]    — binary (0=clean, 1=laundering)
        save_path:    where to save the PNG
        n_neighbors:  UMAP neighbourhood size (larger = more global structure)
        min_dist:     UMAP minimum distance (smaller = tighter clusters)
        max_sample:   max clean nodes to sample (all laundering nodes always kept)
        title:        plot title
    """
    # Subsample clean nodes — UMAP doesn't scale well past ~100K points.
    # All laundering nodes are always kept so they appear in the plot.
    launder_idx = np.where(labels == 1)[0]
    clean_idx   = np.where(labels == 0)[0]

    if len(clean_idx) > max_sample:
        rng = np.random.default_rng(42)
        clean_idx = rng.choice(clean_idx, size=max_sample, replace=False)

    idx = np.concatenate([launder_idx, clean_idx])
    embeds_sub = embeds[idx]
    labels_sub = labels[idx]

    print(f"\nRunning UMAP on {len(idx):,} nodes (dim={embeds.shape[1]}) "
          f"[{len(launder_idx)} laundering + {len(clean_idx):,} clean sampled]...")

    # No random_state → enables parallel execution (n_jobs=-1)
    reducer = umap.UMAP(
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        n_components=2,
        n_jobs=-1,
        low_memory=True,
    )
    projection = reducer.fit_transform(embeds_sub)

    mask_clean   = labels_sub == 0
    mask_launder = labels_sub == 1

    n_launder = mask_launder.sum()
    n_clean   = mask_clean.sum()
    print(f"  Laundering nodes: {n_launder:,}")
    print(f"  Clean nodes:      {n_clean:,} (sampled)")

    fig, ax = plt.subplots(figsize=(10, 8))

    # Background: clean accounts (small, semi-transparent)
    ax.scatter(
        projection[mask_clean, 0],
        projection[mask_clean, 1],
        c="#9ecae1",
        s=2,
        alpha=0.3,
        linewidths=0,
        label=f"Clean ({n_clean:,})",
        rasterized=True,   # keeps file size manageable at large N
    )

    # Foreground: laundering accounts (larger, opaque, distinct colour)
    ax.scatter(
        projection[mask_launder, 0],
        projection[mask_launder, 1],
        c="#e34a33",
        s=30,
        alpha=0.9,
        linewidths=0.4,
        edgecolors="darkred",
        label=f"Laundering ({n_launder:,})",
        zorder=5,
    )

    legend = [
        mpatches.Patch(color="#9ecae1", label=f"Clean ({n_clean:,})"),
        mpatches.Patch(color="#e34a33", label=f"Laundering ({n_launder:,})"),
    ]
    ax.legend(handles=legend, fontsize=11, loc="upper right")
    ax.set_title(title, fontsize=13, pad=12)
    ax.set_xlabel("UMAP-1", fontsize=10)
    ax.set_ylabel("UMAP-2", fontsize=10)
    ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved → {save_path}")
