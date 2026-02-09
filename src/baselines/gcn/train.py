"""
Training script for GCN on SAML-D money laundering detection.
"""

import torch
import torch.nn.functional as F
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
import os

from src.baselines.gcn.model import GCN
from src.data.saml_homo import load_graph
from src.utils.class_weights import compute_class_weights
from src.utils.evaluation import compute_metrics, print_metrics, create_splits
from src.utils.device import get_device
from src.utils.config import PROJECT_ROOT


def train_epoch(model, data, train_mask, optimizer, criterion, device):
    """Train for one epoch."""
    model.train()
    optimizer.zero_grad()

    # Forward pass
    logits = model(data.x, data.edge_index)
    loss = criterion(logits[train_mask], data.y[train_mask])

    # Backward pass
    loss.backward()
    optimizer.step()

    return loss.item()


def train_gcn(data, device, num_epochs=200, hidden_dim=64, num_layers=2,
              dropout=0.3, lr=0.005, weight_decay=5e-4,
              patience=20, use_class_weights=True):
    """
    Train GCN model.

    Args:
        data: PyG Data object
        device: torch device
        num_epochs: Number of training epochs
        hidden_dim: Hidden layer dimension
        num_layers: Number of GCN layers
        dropout: Dropout rate
        lr: Learning rate
        weight_decay: Weight decay for optimizer
        patience: Early stopping patience
        use_class_weights: Whether to use class weights for imbalanced data

    Returns:
        Trained model, training history
    """
    # Move data to device
    data = data.to(device)

    # Create train/val/test splits
    train_mask, val_mask, test_mask = create_splits(data)

    # Initialize model
    model = GCN(
        in_channels=data.num_node_features,
        hidden_channels=hidden_dim,
        out_channels=2,
        num_layers=num_layers,
        dropout=dropout
    ).to(device)

    print(f"\nModel Architecture:")
    print(model)
    print(f"\nTotal parameters: {sum(p.numel() for p in model.parameters())}")

    # Setup optimizer
    optimizer = Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=10)

    # Setup loss function with class weights
    if use_class_weights:
        class_weights = compute_class_weights(data.y[train_mask]).to(device)
        print(f"\nClass weights: {class_weights}")
    else:
        class_weights = None

    criterion = torch.nn.CrossEntropyLoss(weight=class_weights)

    # Training loop
    best_val_f1 = 0
    best_model_state = None
    epochs_without_improvement = 0
    history = {'train_loss': [], 'val_f1': [], 'val_auc': []}

    print(f"\n{'='*60}")
    print("Starting Training")
    print(f"{'='*60}\n")

    for epoch in range(1, num_epochs + 1):
        # Train
        train_loss = train_epoch(model, data, train_mask, optimizer, criterion, device)

        # Evaluate
        train_metrics = compute_metrics(model, data, train_mask, device)
        val_metrics = compute_metrics(model, data, val_mask, device)

        # Update scheduler
        scheduler.step(val_metrics['f1'])

        # Save history
        history['train_loss'].append(train_loss)
        history['val_f1'].append(val_metrics['f1'])
        history['val_auc'].append(val_metrics['auc'])

        # Print progress
        if epoch % 10 == 0 or epoch == 1:
            print(f"Epoch {epoch:3d}/{num_epochs} | "
                  f"Loss: {train_loss:.4f} | "
                  f"Train F1: {train_metrics['f1']:.4f} | "
                  f"Val F1: {val_metrics['f1']:.4f} | "
                  f"Val AUC: {val_metrics['auc']:.4f}")

        # Early stopping and model checkpointing
        if val_metrics['f1'] > best_val_f1:
            best_val_f1 = val_metrics['f1']
            best_model_state = model.state_dict().copy()
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= patience:
            print(f"\nEarly stopping at epoch {epoch}")
            break

    # Load best model
    model.load_state_dict(best_model_state)

    # Final evaluation
    print(f"\n{'='*60}")
    print("Training Complete - Final Evaluation")
    print(f"{'='*60}")

    train_metrics = compute_metrics(model, data, train_mask, device)
    val_metrics = compute_metrics(model, data, val_mask, device)
    test_metrics = compute_metrics(model, data, test_mask, device)

    print_metrics(train_metrics, "Train")
    print_metrics(val_metrics, "Validation")
    print_metrics(test_metrics, "Test")

    return model, history, (train_mask, val_mask, test_mask)


def main():
    """Main training function."""
    device = get_device()
    print(f"Using device: {device}")

    # Load preprocessed graph
    print("\nLoading graph data...")
    cache_path = str(PROJECT_ROOT / 'data' / 'processed' / 'saml_graph.pkl')

    if not os.path.exists(cache_path):
        print(f"Error: {cache_path} not found")
        print("Please run saml_homo.py first to create the preprocessed graph")
        return

    data, account_mapping = load_graph(cache_path)

    print(f"\nLoaded graph:")
    print(f"  Nodes: {data.num_nodes}")
    print(f"  Edges: {data.num_edges}")
    print(f"  Features: {data.num_node_features}")
    print(f"  Positive class: {data.y.sum().item()} ({data.y.sum().item()/len(data.y)*100:.2f}%)")

    # Train model
    print("\n" + "="*60)
    print("Training GCN Model")
    print("="*60)

    model, history, masks = train_gcn(
        data=data,
        device=device,
        num_epochs=200,
        hidden_dim=64,
        num_layers=2,
        dropout=0.3,
        lr=0.005,
        weight_decay=5e-4,
        patience=20,
        use_class_weights=True
    )

    # Save model
    model_path = str(PROJECT_ROOT / 'models' / 'gcn_saml.pt')
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    torch.save({
        'model_state_dict': model.state_dict(),
        'history': history
    }, model_path)
    print(f"\nModel saved to {model_path}")


if __name__ == '__main__':
    main()
