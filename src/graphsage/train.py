"""
Training utilities for GraphSAGE.
"""

import torch
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from tqdm import tqdm


def create_splits(data, train_ratio=0.6, val_ratio=0.2):
    """Create train/val/test splits."""
    num_nodes = data.num_nodes
    indices = torch.arange(num_nodes)
    
    # Split indices
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
    
    print(f"Train: {train_idx.shape[0]}, Val: {val_idx.shape[0]}, Test: {test_idx.shape[0]}")
    
    return data


def train_epoch(model, data, optimizer):
    """Train for one epoch."""
    model.train()
    optimizer.zero_grad()
    
    out = model(data.x, data.edge_index)
    loss = F.cross_entropy(out[data.train_mask], data.y[data.train_mask])
    
    loss.backward()
    optimizer.step()
    
    return loss.item()


def evaluate(model, data, mask):
    """Evaluate model on given mask."""
    model.eval()
    
    with torch.no_grad():
        out = model(data.x, data.edge_index)
        pred = out.argmax(dim=1)
        
        correct = pred[mask] == data.y[mask]
        acc = correct.sum().item() / mask.sum().item()
        
        # Calculate per-class metrics
        y_true = data.y[mask].cpu().numpy()
        y_pred = pred[mask].cpu().numpy()
        
        from sklearn.metrics import precision_score, recall_score, f1_score
        
        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
    
    return acc, precision, recall, f1


def train_model(model, data, epochs=100, lr=0.01, weight_decay=5e-4):
    """Full training loop."""
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    
    best_val_f1 = 0
    best_model_state = None
    
    pbar = tqdm(range(1, epochs + 1), desc="Training")
    
    for epoch in pbar:
        loss = train_epoch(model, data, optimizer)
        
        if epoch % 10 == 0:
            train_acc, train_prec, train_rec, train_f1 = evaluate(model, data, data.train_mask)
            val_acc, val_prec, val_rec, val_f1 = evaluate(model, data, data.val_mask)
            
            pbar.set_postfix({
                'Loss': f'{loss:.4f}',
                'Train F1': f'{train_f1:.4f}',
                'Val F1': f'{val_f1:.4f}'
            })
            
            if val_f1 > best_val_f1:
                best_val_f1 = val_f1
                best_model_state = model.state_dict().copy()
    
    # Load best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    
    return model
