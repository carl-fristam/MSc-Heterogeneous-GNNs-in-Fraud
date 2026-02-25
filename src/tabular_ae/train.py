"""
train.py

Tabular autoencoder training loop for transaction anomaly detection.

Training:
  - Mini-batch on genuine transactions only
  - Loss = MSE reconstruction
  - Early stopping on validation reconstruction error

Evaluation:
  - Anomaly score = per-transaction reconstruction MSE
  - AUROC and AUPRC against Is_laundering ground truth
"""

import datetime
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import roc_auc_score, average_precision_score

from src.tabular_ae.model import TabularAutoencoder
from src.tabular_ae.load_data import load_tabular_data
from src.utils.device import get_device


def train(args):
    """
    Full training pipeline.

    Returns:
        model:   trained TabularAutoencoder
        scores:  per-transaction anomaly scores [N] numpy array
    """
    device = get_device()
    print(f"Device: {device}")

    # ------------------------------------------------------------------
    # Data
    # ------------------------------------------------------------------
    X, y, idx_train, idx_val, idx_test, scaler = load_tabular_data(
        sample_ratio=getattr(args, "sample_ratio", 0.1),
    )

    input_dim = X.shape[1]
    print(f"Input dim: {input_dim}")

    X = X.to(device)
    y = y.to(device)

    # DataLoader for mini-batch training on genuine transactions
    X_train = X[idx_train]
    train_dataset = TensorDataset(X_train)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=False,
    )

    # ------------------------------------------------------------------
    # Model
    # ------------------------------------------------------------------
    model = TabularAutoencoder(
        input_dim=input_dim,
        h1=args.h1,
        h2=args.h2,
        latent_dim=args.latent_dim,
        dropout=args.dropout,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total_params:,}")

    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )

    scheduler = None
    if getattr(args, "scheduler", False):
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=args.epochs, eta_min=1e-5
        )

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------
    best_val_loss = float("inf")
    best_state = None
    cnt_wait = 0
    start = datetime.datetime.now()

    # Pre-compute val genuine mask
    val_genuine_mask = y[idx_val] == 0

    print(f"\nStarting training for {args.epochs} epochs (patience={args.patience})...")
    print("-" * 80)

    for epoch in range(args.epochs):
        model.train()
        epoch_loss = 0.0
        n_batches = 0

        for (batch_x,) in train_loader:
            optimizer.zero_grad()
            loss = model.training_loss(batch_x)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            n_batches += 1

        if scheduler:
            scheduler.step()

        avg_loss = epoch_loss / n_batches

        # Validation every 5 epochs
        if (epoch + 1) % 5 == 0:
            model.eval()
            with torch.no_grad():
                scores = model.anomaly_scores(X[idx_val])
                val_loss = scores[val_genuine_mask].mean().item()

            lr = optimizer.param_groups[0]["lr"]
            wait_str = f"wait {cnt_wait}/{args.patience}" if cnt_wait > 0 else "new best"
            elapsed = (datetime.datetime.now() - start).seconds
            print(
                f"[{epoch+1:4d}/{args.epochs}] "
                f"train {avg_loss:.5f} | val {val_loss:.5f} | "
                f"lr {lr:.2e} | {wait_str} | {elapsed}s"
            )

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
                cnt_wait = 0
            else:
                cnt_wait += 1

            if cnt_wait >= args.patience:
                print(
                    f"\nEarly stopping at epoch {epoch+1} "
                    f"(no improvement for {args.patience} checks)"
                )
                break

    elapsed = (datetime.datetime.now() - start).seconds
    print(f"\nTraining complete in {elapsed}s. Best val_loss: {best_val_loss:.5f}")

    # ------------------------------------------------------------------
    # Evaluate with best model
    # ------------------------------------------------------------------
    model.load_state_dict(best_state)
    model.eval()

    with torch.no_grad():
        all_scores = model.anomaly_scores(X).cpu().numpy()

    labels = y.cpu().numpy()

    for name, idx in [("Val", idx_val), ("Test", idx_test)]:
        _evaluate(all_scores, labels, idx.numpy(), name)

    # ------------------------------------------------------------------
    # Visualisation
    # ------------------------------------------------------------------
    if getattr(args, "visualize", False):
        import os

        os.makedirs("results", exist_ok=True)

        with torch.no_grad():
            z = model.get_latents(X).cpu().numpy()

        from src.tabular_ae.visualize import plot_umap_tabular, plot_score_histogram

        plot_umap_tabular(z, labels, save_path="results/tabular_ae_umap.png")
        plot_score_histogram(
            all_scores, labels, save_path="results/tabular_ae_score_hist.png"
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
