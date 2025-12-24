"""
Main pipeline for GraphSAGE AML detection.
"""

# --- Monkeypatch for Python 3.14 Compatibility ---
import torch_geometric.inspector
import typing

def _patched_type_repr(obj, *args, **kwargs):
    if getattr(obj, '__module__', '') == 'typing':
        # Python 3.14 workaround: typing.Union doesn't have _name
        name = getattr(obj, '_name', None)
        if name is None:
            if hasattr(obj, '__origin__'):
                return str(obj.__origin__).split('.')[-1]
            return str(obj).replace('typing.', '')
    return torch_geometric.inspector._type_repr(obj, *args, **kwargs)

# Save original if needed (though we're likely replacing checking logic)
# Note: In torch_geometric 2.6.1, type_repr calls itself recursively.
# We need to rely on the module-level function.
# However, to avoid recursion depth issues if we wrap it, we should carefuly check implementation.
# The original function seems to be `type_repr`.
# Let's just redefine the one in the module.

# Re-implementing the specific typing check that fails:
def _safe_type_repr(obj, *args, **kwargs):
    if obj is typing.Union:
        return 'Union'
    
    # Safe access to _name
    if getattr(obj, '__module__', '') == 'typing':
        try:
            name = unicode(obj._name) if 'unicode' in locals() else str(obj._name)
        except AttributeError:
            name = None
            
        if name is None:
             if hasattr(obj, '__origin__'):
                return str(obj.__origin__).split('.')[-1]
             return str(obj).replace('typing.', '')
             
    # Fallback to original via a backup reference if we had one, 
    # but since I can't easily get the original code without copying it,
    # and I don't want to import the *internal* function logic...
    
    # Better approach: We catch the specific exception in the original function?
    # No, we can't wrap it internally. 
    
    # Let's COPY the small original function logic from the file read?
    # No, too brittle.
    
    # Best approach: Patch the attribute access on typing.Union?? 
    # No, can't patch C-implemented types easily.
    
    # Let's try to patch `torch_geometric.inspector.type_repr` by wrapping it
    # and catching the AttributeError.
    try:
        return _orig_type_repr(obj, *args, **kwargs)
    except AttributeError:
        if getattr(obj, '__module__', '') == 'typing':
            return str(obj).replace('typing.', '')
        raise

_orig_type_repr = torch_geometric.inspector.type_repr
torch_geometric.inspector.type_repr = _safe_type_repr
# -------------------------------------------------

import torch
from data import load_and_prepare_data
from model import GraphSAGE
from train import create_splits, train_model, evaluate


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
    
    print("\n" + "="*60)
    print("Final Results")
    print("="*60)
    
    # Create loaders for final evaluation
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model = model.to(device)
    
    from torch_geometric.loader import NeighborLoader
    
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
    torch.save(model.state_dict(), 'outputs/graphsage_model.pt')
    print("\nModel saved to outputs/graphsage_model.pt")


if __name__ == "__main__":
    main()
