"""
Shared evaluation metrics and data splitting utilities.
"""

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def compute_metrics(model, data, mask, device):
    """
    Compute classification metrics for a GNN model on masked nodes.

    Args:
        model: GNN model with forward(x, edge_index) signature
        data: PyG Data object (must already be on *device*)
        mask: Boolean mask selecting evaluation nodes
        device: torch device (unused but kept for API compat)

    Returns:
        Dictionary with accuracy, precision, recall, f1, auc, confusion_matrix
    """
    model.eval()
    with torch.no_grad():
        logits = model(data.x, data.edge_index)
        pred = logits[mask].argmax(dim=1)
        y_true = data.y[mask].cpu().numpy()
        y_pred = pred.cpu().numpy()
        y_prob = F.softmax(logits[mask], dim=1)[:, 1].cpu().numpy()

        acc = accuracy_score(y_true, y_pred)
        prec = precision_score(y_true, y_pred, zero_division=0)
        rec = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)

        if len(np.unique(y_true)) > 1:
            auc = roc_auc_score(y_true, y_prob)
        else:
            auc = 0.0

        cm = confusion_matrix(y_true, y_pred)

    return {
        'accuracy': acc,
        'precision': prec,
        'recall': rec,
        'f1': f1,
        'auc': auc,
        'confusion_matrix': cm,
    }


def print_metrics(metrics, split_name="Validation"):
    """Pretty-print a metrics dictionary."""
    print(f"\n{split_name} Metrics:")
    print(f"  Accuracy:  {metrics['accuracy']:.4f}")
    print(f"  Precision: {metrics['precision']:.4f}")
    print(f"  Recall:    {metrics['recall']:.4f}")
    print(f"  F1 Score:  {metrics['f1']:.4f}")
    print(f"  ROC AUC:   {metrics['auc']:.4f}")
    print(f"  Confusion Matrix:")
    print(f"    {metrics['confusion_matrix']}")


def create_splits(data, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, seed=42):
    """
    Create train/val/test masks with random splits.

    Args:
        data: PyG Data object
        train_ratio: Training set ratio
        val_ratio: Validation set ratio
        test_ratio: Test set ratio
        seed: Random seed

    Returns:
        train_mask, val_mask, test_mask (boolean tensors)
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    num_nodes = data.num_nodes
    indices = torch.randperm(num_nodes)

    train_size = int(train_ratio * num_nodes)
    val_size = int(val_ratio * num_nodes)

    train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    val_mask = torch.zeros(num_nodes, dtype=torch.bool)
    test_mask = torch.zeros(num_nodes, dtype=torch.bool)

    train_mask[indices[:train_size]] = True
    val_mask[indices[train_size:train_size + val_size]] = True
    test_mask[indices[train_size + val_size:]] = True

    print(f"\nDataset Split Statistics:")
    print(f"  Train: {train_mask.sum().item()} nodes ({train_mask.sum().item()/num_nodes*100:.1f}%)")
    print(f"    - Positive: {data.y[train_mask].sum().item()}")
    print(f"    - Negative: {(data.y[train_mask] == 0).sum().item()}")
    print(f"  Val:   {val_mask.sum().item()} nodes ({val_mask.sum().item()/num_nodes*100:.1f}%)")
    print(f"    - Positive: {data.y[val_mask].sum().item()}")
    print(f"    - Negative: {(data.y[val_mask] == 0).sum().item()}")
    print(f"  Test:  {test_mask.sum().item()} nodes ({test_mask.sum().item()/num_nodes*100:.1f}%)")
    print(f"    - Positive: {data.y[test_mask].sum().item()}")
    print(f"    - Negative: {(data.y[test_mask] == 0).sum().item()}")

    return train_mask, val_mask, test_mask
