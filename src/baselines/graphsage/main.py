"""
Main pipeline for GraphSAGE AML detection.
"""

from src.utils.compat import apply_pyg_compat_patch
apply_pyg_compat_patch()

import torch
from torch_geometric.loader import NeighborLoader

from src.baselines.graphsage.data import load_and_prepare_data
from src.baselines.graphsage.model import GraphSAGE
from src.baselines.graphsage.train import create_splits, train_model, evaluate
from src.utils.config import PROJECT_ROOT
from src.utils.device import get_device


def main():
    print("="*60)
    print("GraphSAGE AML Detection Pipeline")
    print("="*60)

    # Load data (Full Dataset)
    data, account_to_id = load_and_prepare_data()

    # Create splits
    data = create_splits(data)

    # Initialize model
    model = GraphSAGE(
        in_channels=data.num_node_features,
        hidden_channels=64,
        num_layers=2,
        dropout=0.5
    )

    print(f"\nModel: {sum(p.numel() for p in model.parameters())} parameters")

    # Train
    print("\nTraining...")
    model = train_model(model, data, epochs=100, lr=0.01)

    # Final evaluation
    print("\n" + "="*60)
    print("Final Results")
    print("="*60)

    # Create loaders for final evaluation
    device = get_device()
    model = model.to(device)

    def get_loader(mask):
        return NeighborLoader(
            data,
            num_neighbors=[10, 5],
            batch_size=4096,
            input_nodes=mask,
            shuffle=False,
            num_workers=4
        )

    train_loader = get_loader(data.train_mask)
    val_loader = get_loader(data.val_mask)
    test_loader = get_loader(data.test_mask)

    train_acc, train_prec, train_rec, train_f1 = evaluate(model, train_loader, device)
    val_acc, val_prec, val_rec, val_f1 = evaluate(model, val_loader, device)
    test_acc, test_prec, test_rec, test_f1 = evaluate(model, test_loader, device)

    print(f"\nTrain - Acc: {train_acc:.4f}, Prec: {train_prec:.4f}, Rec: {train_rec:.4f}, F1: {train_f1:.4f}")
    print(f"Val   - Acc: {val_acc:.4f}, Prec: {val_prec:.4f}, Rec: {val_rec:.4f}, F1: {val_f1:.4f}")
    print(f"Test  - Acc: {test_acc:.4f}, Prec: {test_prec:.4f}, Rec: {test_rec:.4f}, F1: {test_f1:.4f}")

    # Save model
    save_path = str(PROJECT_ROOT / 'models' / 'graphsage_model.pt')
    torch.save(model.state_dict(), save_path)
    print(f"\nModel saved to {save_path}")


if __name__ == "__main__":
    main()
