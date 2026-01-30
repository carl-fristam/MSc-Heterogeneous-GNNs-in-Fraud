"""
Graph Convolutional Network (GCN) for Money Laundering Detection on SAML-D dataset.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
import numpy as np


class GCN(nn.Module):
    """
    Graph Convolutional Network for node classification.

    Args:
        in_channels: Number of input features per node
        hidden_channels: Hidden dimension size
        out_channels: Number of output classes (2 for binary classification)
        num_layers: Number of GCN layers
        dropout: Dropout rate
    """
    def __init__(self, in_channels, hidden_channels=64, out_channels=2,
                 num_layers=2, dropout=0.3):
        super(GCN, self).__init__()

        self.num_layers = num_layers
        self.dropout = dropout

        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()

        # First layer
        self.convs.append(GCNConv(in_channels, hidden_channels))
        self.bns.append(nn.BatchNorm1d(hidden_channels))

        # Hidden layers
        for _ in range(num_layers - 2):
            self.convs.append(GCNConv(hidden_channels, hidden_channels))
            self.bns.append(nn.BatchNorm1d(hidden_channels))

        # Last layer
        if num_layers > 1:
            self.convs.append(GCNConv(hidden_channels, out_channels))
        else:
            self.convs[-1] = GCNConv(in_channels, out_channels)

    def forward(self, x, edge_index):
        """
        Forward pass.

        Args:
            x: Node features [num_nodes, in_channels]
            edge_index: Edge indices [2, num_edges]

        Returns:
            Node embeddings [num_nodes, out_channels]
        """
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, edge_index)
            x = self.bns[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

        # Final layer (no activation, no dropout)
        x = self.convs[-1](x, edge_index)

        return x

    def predict(self, x, edge_index):
        """Get class predictions."""
        self.eval()
        with torch.no_grad():
            logits = self.forward(x, edge_index)
            return torch.argmax(logits, dim=1)


def compute_class_weights(labels):
    """
    Compute class weights for imbalanced datasets.

    Args:
        labels: Tensor of labels

    Returns:
        Tensor of class weights
    """
    unique, counts = torch.unique(labels, return_counts=True)
    weights = 1.0 / counts.float()
    weights = weights / weights.sum() * len(unique)
    return weights


def evaluate_model(model, data, mask, device):
    """
    Evaluate model performance.

    Args:
        model: GCN model
        data: PyG Data object
        mask: Boolean mask for evaluation set
        device: torch device

    Returns:
        Dictionary of metrics
    """
    model.eval()
    with torch.no_grad():
        logits = model(data.x, data.edge_index)
        pred = logits[mask].argmax(dim=1)
        y_true = data.y[mask].cpu().numpy()
        y_pred = pred.cpu().numpy()
        y_prob = F.softmax(logits[mask], dim=1)[:, 1].cpu().numpy()

        # Compute metrics
        acc = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)

        # ROC AUC (only if both classes present)
        if len(np.unique(y_true)) > 1:
            auc = roc_auc_score(y_true, y_prob)
        else:
            auc = 0.0

        cm = confusion_matrix(y_true, y_pred)

        metrics = {
            'accuracy': acc,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'auc': auc,
            'confusion_matrix': cm
        }

        return metrics


def print_metrics(metrics, split_name="Validation"):
    """Pretty print metrics."""
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
    Create train/val/test splits with stratification.

    Args:
        data: PyG Data object
        train_ratio: Training set ratio
        val_ratio: Validation set ratio
        test_ratio: Test set ratio
        seed: Random seed

    Returns:
        train_mask, val_mask, test_mask
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    num_nodes = data.num_nodes
    indices = torch.randperm(num_nodes)

    # Calculate split sizes
    train_size = int(train_ratio * num_nodes)
    val_size = int(val_ratio * num_nodes)

    # Create masks
    train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    val_mask = torch.zeros(num_nodes, dtype=torch.bool)
    test_mask = torch.zeros(num_nodes, dtype=torch.bool)

    train_mask[indices[:train_size]] = True
    val_mask[indices[train_size:train_size + val_size]] = True
    test_mask[indices[train_size + val_size:]] = True

    # Print split statistics
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
