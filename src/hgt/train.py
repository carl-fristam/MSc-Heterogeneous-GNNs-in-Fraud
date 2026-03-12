"""
HGT training loop with early stopping on validation AUPRC.
"""

import torch
import torch.nn as nn
import numpy as np
from copy import deepcopy
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix,
)

from src.hgt.model import HGT


def train(data, device, args):
    """Full train/val/test loop. Returns test metrics dict."""
    from src.utils.class_weights import compute_class_weights

    data = data.to(device)

    # --- Model ---
    model = HGT(
        data,
        hidden_dim=args.hidden_dim,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        dropout=args.dropout,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"HGT parameters: {total_params:,}")

    # --- Loss ---
    y = data["transaction"].y
    train_mask = data["transaction"].train_mask
    train_labels = y[train_mask]

    use_class_weight = not getattr(args, "no_class_weight", False)
    if use_class_weight:
        weights = compute_class_weights(train_labels)
        pos_weight = weights[1] / weights[0] if len(weights) > 1 else torch.tensor(1.0)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight.to(device))
        print(f"Class weights: legit={weights[0]:.4f}, fraud={weights[1]:.4f}, pos_weight={pos_weight:.2f}")
    else:
        criterion = nn.BCEWithLogitsLoss()
        print("No class weighting (plain BCE)")

    # --- Optimizer ---
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    scheduler = (
        torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
        if args.scheduler
        else None
    )

    # --- Training loop ---
    best_val_auprc = 0.0
    best_state = None
    cnt_wait = 0

    val_mask = data["transaction"].val_mask
    test_mask = data["transaction"].test_mask

    for epoch in range(1, args.epochs + 1):
        model.train()
        optimizer.zero_grad()

        logits = model(data)
        loss = criterion(logits[train_mask], y[train_mask])

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        if scheduler:
            scheduler.step()

        # --- Validation ---
        if epoch % 5 == 0 or epoch == 1:
            model.eval()
            with torch.no_grad():
                logits = model(data)
                val_probs = torch.sigmoid(logits[val_mask]).cpu().numpy()
                val_labels = y[val_mask].cpu().numpy()

                val_loss = criterion(logits[val_mask], y[val_mask]).item()

                if val_labels.sum() > 0:
                    val_auroc = roc_auc_score(val_labels, val_probs)
                    val_auprc = average_precision_score(val_labels, val_probs)
                else:
                    val_auroc = val_auprc = 0.0

            print(
                f"Epoch {epoch:3d} | "
                f"Train loss: {loss.item():.4f} | "
                f"Val loss: {val_loss:.4f} | "
                f"Val AUROC: {val_auroc:.4f} | "
                f"Val AUPRC: {val_auprc:.4f}"
            )

            if val_auprc > best_val_auprc:
                best_val_auprc = val_auprc
                best_state = deepcopy(model.state_dict())
                cnt_wait = 0
            else:
                cnt_wait += 1
                if cnt_wait >= args.patience:
                    print(f"Early stopping at epoch {epoch} (patience={args.patience})")
                    break

    # --- Test evaluation ---
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()

    with torch.no_grad():
        logits = model(data)
        test_probs = torch.sigmoid(logits[test_mask]).cpu().numpy()
        test_labels = y[test_mask].cpu().numpy()

    test_preds = (test_probs >= 0.5).astype(int)

    metrics = {
        "auroc": roc_auc_score(test_labels, test_probs) if test_labels.sum() > 0 else 0.0,
        "auprc": average_precision_score(test_labels, test_probs) if test_labels.sum() > 0 else 0.0,
        "f1": f1_score(test_labels, test_preds, zero_division=0),
        "precision": precision_score(test_labels, test_preds, zero_division=0),
        "recall": recall_score(test_labels, test_preds, zero_division=0),
        "confusion_matrix": confusion_matrix(test_labels, test_preds),
    }

    print("\n" + "=" * 50)
    print("TEST RESULTS")
    print("=" * 50)
    print(f"  AUROC:     {metrics['auroc']:.4f}")
    print(f"  AUPRC:     {metrics['auprc']:.4f}")
    print(f"  F1:        {metrics['f1']:.4f}")
    print(f"  Precision: {metrics['precision']:.4f}")
    print(f"  Recall:    {metrics['recall']:.4f}")
    print(f"  Confusion matrix:\n{metrics['confusion_matrix']}")

    return metrics
