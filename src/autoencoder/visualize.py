"""
visualize.py

Visualisation for the transaction VGAE pipeline.

1. UMAP of transaction latent vectors, coloured by Is_laundering
2. Reconstruction error histogram split by label
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import umap


def plot_umap_transactions(
    z: np.ndarray,
    labels: np.ndarray,
    save_path: str = 'results/autoencoder_umap.png',
    max_sample_genuine: int = 50_000,
    n_neighbors: int = 30,
    min_dist: float = 0.1,
    title: str = 'Transaction VGAE — UMAP of Latent Space',
):
    """
    UMAP scatter of transaction latent vectors.

    Subsamples genuine transactions (capped at max_sample_genuine)
    while keeping ALL fraud transactions visible.
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
        c='#9ecae1', s=2, alpha=0.3, rasterized=True, label='Genuine',
    )
    ax.scatter(
        emb[fraud_mask, 0], emb[fraud_mask, 1],
        c='#e34a33', s=30, alpha=0.9, zorder=5, label='Laundering',
    )

    ax.set_title(title, fontsize=14)
    ax.legend(fontsize=11, markerscale=3)
    ax.set_xticks([])
    ax.set_yticks([])

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"UMAP saved: {save_path}")


def plot_score_histogram(
    scores: np.ndarray,
    labels: np.ndarray,
    save_path: str = 'results/autoencoder_score_hist.png',
    percentile_cap: float = 99.5,
    title: str = 'Reconstruction Error Distribution',
):
    """
    Overlapping histograms + KDE of anomaly scores for genuine vs fraud.

    Uses density-normalised histograms so the two classes are visually
    comparable despite extreme class imbalance. KDE curves overlaid for
    smoother distribution shape. X-axis capped at the 99.5th percentile
    of genuine scores to prevent outlier compression.
    """
    from scipy.stats import gaussian_kde

    genuine_scores = scores[labels == 0]
    fraud_scores = scores[labels == 1]

    # Cap x-axis at percentile of genuine scores to avoid outlier compression
    x_max = np.percentile(genuine_scores, percentile_cap)

    fig, ax = plt.subplots(figsize=(10, 6))

    bins = np.linspace(0, x_max, 80)

    # Histograms (density-normalised for comparable scales)
    ax.hist(genuine_scores, bins=bins, density=True, alpha=0.4,
            color='#9ecae1', label='Genuine', edgecolor='white', linewidth=0.3)
    ax.hist(fraud_scores[fraud_scores <= x_max], bins=bins, density=True, alpha=0.5,
            color='#e34a33', label='Laundering', edgecolor='white', linewidth=0.3)

    # KDE overlays
    x_grid = np.linspace(0, x_max, 300)
    kde_genuine = gaussian_kde(genuine_scores[genuine_scores <= x_max])
    ax.plot(x_grid, kde_genuine(x_grid), color='#2171b5', linewidth=2)

    if len(fraud_scores[fraud_scores <= x_max]) > 1:
        kde_fraud = gaussian_kde(fraud_scores[fraud_scores <= x_max])
        ax.plot(x_grid, kde_fraud(x_grid), color='#cb181d', linewidth=2)

    # Mean lines
    ax.axvline(genuine_scores.mean(), color='#2171b5', linestyle='--', linewidth=1.2,
               label=f'Genuine mean ({genuine_scores.mean():.4f})')
    ax.axvline(fraud_scores.mean(), color='#cb181d', linestyle='--', linewidth=1.2,
               label=f'Fraud mean ({fraud_scores.mean():.4f})')

    ax.set_xlim(0, x_max)
    ax.set_xlabel('Reconstruction Error (MSE)', fontsize=12)
    ax.set_ylabel('Density', fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.legend(fontsize=10)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Histogram saved: {save_path}")


def plot_attention_analysis(
    attention: dict,
    anomaly_scores: np.ndarray,
    labels: np.ndarray,
    save_path: str = 'results/autoencoder_attention.png',
):
    """
    Attention weight analysis across edge types.

    Produces a 2x2 figure:
        Top-left:     Attention distribution per edge type (genuine vs fraud)
        Top-right:    Mean attention by edge type (genuine vs fraud bar chart)
        Bottom-left:  Attention vs anomaly score scatter (sampled)
        Bottom-right: Top-k highest attention edges: fraud rate vs baseline

    Args:
        attention: dict from model.get_attention_weights()
                   maps edge_type -> (edge_index [2, E], alpha [E])
        anomaly_scores: [Nt] numpy array
        labels: [Nt] numpy array (Is_laundering)
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 11))

    # ---- Panel 1: Attention distributions per edge type ----
    ax = axes[0, 0]
    for edge_type, (ei, alpha) in attention.items():
        alpha_np = alpha.cpu().numpy()
        rel_name = edge_type[1]
        # Use destination node labels for edges pointing to transactions
        dst_type = edge_type[2]
        if dst_type == 'transaction':
            dst_labels = labels[ei[1].cpu().numpy()]
        else:
            # For edges pointing to accounts, use source (transaction) labels
            dst_labels = labels[ei[0].cpu().numpy()]

        genuine_alpha = alpha_np[dst_labels == 0]
        fraud_alpha = alpha_np[dst_labels == 1]

        ax.hist(genuine_alpha, bins=50, density=True, alpha=0.3,
                label=f'{rel_name} (genuine)', histtype='stepfilled')
        if len(fraud_alpha) > 0:
            ax.hist(fraud_alpha, bins=50, density=True, alpha=0.5,
                    label=f'{rel_name} (fraud)', histtype='step', linewidth=2)

    ax.set_xlabel('Attention Weight', fontsize=11)
    ax.set_ylabel('Density', fontsize=11)
    ax.set_title('Attention Distribution by Edge Type', fontsize=12)
    ax.legend(fontsize=8)

    # ---- Panel 2: Mean attention bar chart ----
    ax = axes[0, 1]
    edge_names = []
    genuine_means = []
    fraud_means = []

    for edge_type, (ei, alpha) in attention.items():
        alpha_np = alpha.cpu().numpy()
        rel_name = edge_type[1]
        dst_type = edge_type[2]
        if dst_type == 'transaction':
            dst_labels = labels[ei[1].cpu().numpy()]
        else:
            dst_labels = labels[ei[0].cpu().numpy()]

        edge_names.append(rel_name)
        genuine_means.append(alpha_np[dst_labels == 0].mean())
        if (dst_labels == 1).sum() > 0:
            fraud_means.append(alpha_np[dst_labels == 1].mean())
        else:
            fraud_means.append(0.0)

    x = np.arange(len(edge_names))
    w = 0.35
    ax.bar(x - w/2, genuine_means, w, label='Genuine', color='#9ecae1')
    ax.bar(x + w/2, fraud_means, w, label='Fraud', color='#e34a33')
    ax.set_xticks(x)
    ax.set_xticklabels(edge_names, rotation=20, ha='right', fontsize=9)
    ax.set_ylabel('Mean Attention Weight', fontsize=11)
    ax.set_title('Mean Attention: Genuine vs Fraud', fontsize=12)
    ax.legend(fontsize=10)

    # ---- Panel 3: Attention vs anomaly score ----
    ax = axes[1, 0]
    # Use the "sends" edge type (account → transaction) — most interpretable
    sends_type = None
    for et in attention:
        if et[1] == 'sends':
            sends_type = et
            break
    if sends_type is None:
        sends_type = list(attention.keys())[0]

    ei, alpha = attention[sends_type]
    alpha_np = alpha.cpu().numpy()
    dst_idx = ei[1].cpu().numpy()

    # Compute mean attention received per transaction
    num_txns = len(anomaly_scores)
    txn_mean_attn = np.zeros(num_txns)
    txn_count = np.zeros(num_txns)
    np.add.at(txn_mean_attn, dst_idx, alpha_np)
    np.add.at(txn_count, dst_idx, 1)
    txn_count[txn_count == 0] = 1
    txn_mean_attn /= txn_count

    # Subsample for plotting
    rng = np.random.RandomState(42)
    n_plot = min(10_000, num_txns)
    plot_idx = rng.choice(num_txns, n_plot, replace=False)

    genuine_plot = plot_idx[labels[plot_idx] == 0]
    fraud_plot = plot_idx[labels[plot_idx] == 1]

    ax.scatter(txn_mean_attn[genuine_plot], anomaly_scores[genuine_plot],
               c='#9ecae1', s=2, alpha=0.3, rasterized=True, label='Genuine')
    ax.scatter(txn_mean_attn[fraud_plot], anomaly_scores[fraud_plot],
               c='#e34a33', s=30, alpha=0.8, zorder=5, label='Fraud')
    ax.set_xlabel(f'Mean Attention Received ({sends_type[1]})', fontsize=11)
    ax.set_ylabel('Anomaly Score (MSE)', fontsize=11)
    ax.set_title('Attention vs Reconstruction Error', fontsize=12)
    ax.legend(fontsize=10)

    # ---- Panel 4: Fraud enrichment in high-attention edges ----
    ax = axes[1, 1]
    # For each edge type, compute fraud rate in top-k% attention vs baseline
    percentiles = [100, 50, 25, 10, 5, 1]

    for edge_type, (ei, alpha_t) in attention.items():
        alpha_np = alpha_t.cpu().numpy()
        rel_name = edge_type[1]
        dst_type = edge_type[2]
        if dst_type == 'transaction':
            dst_labels = labels[ei[1].cpu().numpy()]
        else:
            dst_labels = labels[ei[0].cpu().numpy()]

        fraud_rates = []
        for pct in percentiles:
            threshold = np.percentile(alpha_np, 100 - pct)
            mask = alpha_np >= threshold
            if mask.sum() > 0:
                fraud_rates.append(dst_labels[mask].mean() * 100)
            else:
                fraud_rates.append(0.0)

        ax.plot(percentiles, fraud_rates, 'o-', label=rel_name, markersize=5)

    baseline = labels.mean() * 100
    ax.axhline(baseline, color='gray', linestyle='--', linewidth=1,
               label=f'Baseline ({baseline:.2f}%)')
    ax.set_xlabel('Top-k% Attention Edges', fontsize=11)
    ax.set_ylabel('Fraud Rate (%)', fontsize=11)
    ax.set_title('Fraud Enrichment in High-Attention Edges', fontsize=12)
    ax.legend(fontsize=9)
    ax.set_xscale('log')

    plt.suptitle('VGAE Attention Weight Analysis', fontsize=14, y=1.01)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Attention analysis saved: {save_path}")
