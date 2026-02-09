"""
Optimized Training utilities for GraphSAGE on Apple Silicon (M4).
Includes Neighbor Sampling and MPS acceleration.
"""

import torch
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score
from tqdm import tqdm
from torch_geometric.loader import NeighborLoader

from src.utils.device import get_device

def create_splits(data, train_ratio=0.6, val_ratio=0.2):
    """Create train/val/test splits with stratification for imbalanced AML data."""
    num_nodes = data.num_nodes
    indices = torch.arange(num_nodes)

    # Split indices (stratify helps with the 1% 'Is Laundering' class)
    train_idx, temp_idx = train_test_split(
        indices, train_size=train_ratio, stratify=data.y, random_state=42
    )
    val_idx, test_idx = train_test_split(
        temp_idx, train_size=val_ratio/(1-train_ratio), stratify=data.y[temp_idx], random_state=42
    )

    # Create masks
    data.train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    data.val_mask = torch.zeros(num_nodes, dtype=torch.bool)
    data.test_mask = torch.zeros(num_nodes, dtype=torch.bool)

    data.train_mask[train_idx] = True
    data.val_mask[val_idx] = True
    data.test_mask[test_idx] = True

    print(f"Splits Created - Train: {train_idx.shape[0]}, Val: {val_idx.shape[0]}, Test: {test_idx.shape[0]}")
    return data

def train_epoch(model, loader, optimizer, device):
    """Train for one epoch using Neighbor Sampling and MPS."""
    model.train()
    total_loss = 0

    for batch in loader:
        # Move mini-batch to M4 GPU
        batch = batch.to(device)
        optimizer.zero_grad()

        # Forward pass
        out = model(batch.x, batch.edge_index)

        # NeighborLoader puts target nodes at the start of the batch
        # We only compute loss for these 'root' nodes
        loss = F.cross_entropy(out[:batch.batch_size], batch.y[:batch.batch_size])

        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    return total_loss / len(loader)

def evaluate(model, loader, device):
    """Evaluate model using mini-batches to save memory."""
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            out = model(batch.x, batch.edge_index)
            pred = out[:batch.batch_size].argmax(dim=1)

            all_preds.append(pred.cpu())
            all_labels.append(batch.y[:batch.batch_size].cpu())

    y_pred = torch.cat(all_preds).numpy()
    y_true = torch.cat(all_labels).numpy()

    acc = (y_pred == y_true).mean()
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    return acc, precision, recall, f1

def train_model(model, data, epochs=5, lr=0.01, batch_size=1024):
    """Full training loop optimized for M4 Air."""

    # 1. Setup Device
    device = get_device()
    print(f"Using Device: {device}")
    model = model.to(device)

    # 2. Setup Neighbor Loaders (Essential for large graphs)
    train_loader = NeighborLoader(
        data,
        num_neighbors=[10, 5], # Sample neighbors for 2 hops
        batch_size=batch_size,
        input_nodes=data.train_mask,
        shuffle=True,
        num_workers=4,
        persistent_workers=True
    )

    val_loader = NeighborLoader(
        data,
        num_neighbors=[15, 10],
        batch_size=batch_size,
        input_nodes=data.val_mask,
        shuffle=False,
        num_workers=4,
        persistent_workers=True
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=5e-4)
    best_val_f1 = 0
    best_model_state = None

    pbar = tqdm(range(1, epochs + 1), desc="Training GNN")

    for epoch in pbar:
        loss = train_epoch(model, train_loader, optimizer, device)

        if epoch % 5 == 0: # Check more frequently
            val_acc, val_prec, val_rec, val_f1 = evaluate(model, val_loader, device)

            pbar.set_postfix({
                'Loss': f'{loss:.4f}',
                'Val F1': f'{val_f1:.4f}',
                'Recall': f'{val_rec:.4f}' # Important for AML
            })

            if val_f1 > best_val_f1:
                best_val_f1 = val_f1
                best_model_state = model.state_dict().copy()

    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    return model
