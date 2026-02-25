"""
train.py

VGAE training loop for transaction anomaly detection on SAML-D.

Training:
  - Full-batch on genuine transactions only
  - Loss = MSE reconstruction + beta * KL divergence
  - Early stopping on validation reconstruction error

Evaluation:
  - Anomaly score = per-transaction reconstruction MSE
  - AUROC and AUPRC against Is_laundering ground truth
"""

import datetime
import numpy as np
import torch
from sklearn.metrics import roc_auc_score, average_precision_score

from src.autoencoder.model import TransactionVGAE
from src.autoencoder.load_data import load_autoencoder_data
from src.utils.device import get_device


def train(args):
    """
    Full training pipeline.

    Returns:
        model:   trained TransactionVGAE
        scores:  per-transaction anomaly scores [Nt] numpy array
    """
    device = get_device()
    print(f"Device: {device}")

    # ------------------------------------------------------------------
    # Data
    # ------------------------------------------------------------------
    data, genuine_mask, idx_train, idx_val, idx_test = load_autoencoder_data(
        sample_ratio=getattr(args, 'sample_ratio', 0.1),
        n_days=getattr(args, 'n_days', None),
        use_cache=getattr(args, 'use_cache', True),
    )

    data = data.to(device)
    genuine_mask = genuine_mask.to(device)
    idx_val_dev = idx_val.to(device)

    x_dict = {nt: data[nt].x for nt in data.node_types}
    edge_index_dict = {et: data[et].edge_index for et in data.edge_types}

    account_feat_dim = data['account'].x.shape[1]
    txn_feat_dim = data['transaction'].x.shape[1]
    print(f"Account feat dim: {account_feat_dim}, Txn feat dim: {txn_feat_dim}")

    # ------------------------------------------------------------------
    # Model
    # ------------------------------------------------------------------
    model = TransactionVGAE(
        account_feat_dim=account_feat_dim,
        txn_feat_dim=txn_feat_dim,
        hidden_dim=args.hidden_dim,
        latent_dim=args.latent_dim,
        num_heads=getattr(args, 'num_heads', 4),
        encoder_dropout=args.encoder_dropout,
        decoder_dropout=args.decoder_dropout,
        beta=getattr(args, 'beta', 0.001),
        edge_types=list(data.edge_types),
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total_params:,}")

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )

    scheduler = None
    if getattr(args, 'scheduler', False):
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=args.epochs, eta_min=1e-5
        )

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------
    best_val_loss = float('inf')
    best_state = None
    cnt_wait = 0
    start = datetime.datetime.now()

    print(f"\nStarting training for {args.epochs} epochs (patience={args.patience})...")
    print("-" * 85)

    for epoch in range(args.epochs):
        model.train()
        optimizer.zero_grad()

        loss, recon_loss, kl_loss = model.training_loss(
            x_dict, edge_index_dict, genuine_mask
        )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        if scheduler:
            scheduler.step()

        # Validation every 5 epochs
        if (epoch + 1) % 5 == 0:
            model.eval()
            with torch.no_grad():
                scores = model.anomaly_scores(x_dict, edge_index_dict)
                val_genuine = genuine_mask[idx_val_dev]
                val_loss = scores[idx_val_dev][val_genuine].mean().item()

            lr = optimizer.param_groups[0]['lr']
            wait_str = f"wait {cnt_wait}/{args.patience}" if cnt_wait > 0 else "new best"
            elapsed = (datetime.datetime.now() - start).seconds
            print(f"[{epoch+1:4d}/{args.epochs}] "
                  f"loss {loss.item():.5f} (recon {recon_loss:.5f}, kl {kl_loss:.1f}) | "
                  f"val {val_loss:.5f} | lr {lr:.2e} | {wait_str} | {elapsed}s")

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
                cnt_wait = 0
            else:
                cnt_wait += 1

            if cnt_wait >= args.patience:
                print(f"\nEarly stopping at epoch {epoch+1} (no improvement for {args.patience} checks)")
                break

    elapsed = (datetime.datetime.now() - start).seconds
    print(f"\nTraining complete in {elapsed}s. Best val_loss: {best_val_loss:.5f}")

    # ------------------------------------------------------------------
    # Evaluate with best model
    # ------------------------------------------------------------------
    model.load_state_dict(best_state)
    model.eval()

    with torch.no_grad():
        all_scores = model.anomaly_scores(x_dict, edge_index_dict).cpu().numpy()

    labels = data['transaction'].is_laundering.cpu().numpy()

    for name, idx in [("Val", idx_val), ("Test", idx_test)]:
        _evaluate(all_scores, labels, idx.numpy(), name)

    # ------------------------------------------------------------------
    # Visualisation
    # ------------------------------------------------------------------
    if getattr(args, 'visualize', False):
        import os
        os.makedirs('results', exist_ok=True)

        with torch.no_grad():
            z = model.get_latents(x_dict, edge_index_dict).cpu().numpy()

        from src.autoencoder.visualize import (
            plot_umap_transactions, plot_score_histogram, plot_attention_analysis,
        )
        plot_umap_transactions(z, labels, save_path='results/autoencoder_umap.png')
        plot_score_histogram(all_scores, labels, save_path='results/autoencoder_score_hist.png')

        # Attention weight analysis
        attention = model.get_attention_weights(x_dict, edge_index_dict)
        plot_attention_analysis(
            attention, all_scores, labels,
            save_path='results/autoencoder_attention.png',
        )

    return model, all_scores


def _evaluate(scores: np.ndarray, labels: np.ndarray, idx: np.ndarray, split_name: str):
    """Print AUROC and AUPRC for a split."""
    s = scores[idx]
    y = labels[idx].astype(int)

    if len(np.unique(y)) < 2:
        print(f"\n{split_name}: only one class present — cannot compute metrics")
        return

    auroc = roc_auc_score(y, s)
    auprc = average_precision_score(y, s)

    print(f"\n{split_name} Anomaly Detection:")
    print(f"  AUROC:  {auroc:.4f}")
    print(f"  AUPRC:  {auprc:.4f}")
    print(f"  Mean score (genuine): {s[y==0].mean():.5f}")
    print(f"  Mean score (fraud):   {s[y==1].mean():.5f}")
