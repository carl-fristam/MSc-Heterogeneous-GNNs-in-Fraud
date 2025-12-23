"""
Main pipeline for GraphSAGE AML detection.
"""

from data import load_and_prepare_data
from model import GraphSAGE
from train import create_splits, train_model, evaluate


def main():
    print("="*60)
    print("GraphSAGE AML Detection Pipeline")
    print("="*60)
    
    # Load data
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
    
    train_acc, train_prec, train_rec, train_f1 = evaluate(model, data, data.train_mask)
    val_acc, val_prec, val_rec, val_f1 = evaluate(model, data, data.val_mask)
    test_acc, test_prec, test_rec, test_f1 = evaluate(model, data, data.test_mask)
    
    print(f"\nTrain - Acc: {train_acc:.4f}, Prec: {train_prec:.4f}, Rec: {train_rec:.4f}, F1: {train_f1:.4f}")
    print(f"Val   - Acc: {val_acc:.4f}, Prec: {val_prec:.4f}, Rec: {val_rec:.4f}, F1: {val_f1:.4f}")
    print(f"Test  - Acc: {test_acc:.4f}, Prec: {test_prec:.4f}, Rec: {test_rec:.4f}, F1: {test_f1:.4f}")
    
    # Save model
    import torch
    torch.save(model.state_dict(), 'outputs/graphsage_model.pt')
    print("\nModel saved to outputs/graphsage_model.pt")


if __name__ == "__main__":
    main()
