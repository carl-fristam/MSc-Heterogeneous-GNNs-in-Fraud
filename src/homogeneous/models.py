"""
Homogeneous GNN models for L1 edge classification.

Three architectures:
  gcn   — GCNConv layers (no edge features in message passing)
  sage  — SAGEConv layers (no edge features in message passing)
  gat   — GATConv layers (edge features used in attention)

All models:
  - Take a projected homogeneous PyG Data object
  - Run N conv layers to produce node embeddings
  - Score each edge by concatenating src + dst embeddings + edge_attr → MLP

Usage:
    from src.homogeneous.models import HomoGNN
    model = HomoGNN(data, conv_type="sage", hidden_dim=64, num_layers=2)
    logits = model(data)   # shape: (num_edges,)
"""

import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv, SAGEConv, GATConv


class HomoGNN(nn.Module):
    """
    Args:
        data:        PyG Data (homogeneous projection)
        conv_type:   "gcn" | "sage" | "gat"
        hidden_dim:  node embedding dimension
        num_layers:  number of conv layers
        num_heads:   attention heads (gat only)
        dropout:     dropout rate
    """

    def __init__(self, data, conv_type="sage", hidden_dim=64,
                 num_layers=2, num_heads=4, dropout=0.3):
        super().__init__()
        self.conv_type = conv_type

        in_dim       = data.x.shape[1]
        edge_feat_dim = data.edge_attr.shape[1] if (
            hasattr(data, "edge_attr") and data.edge_attr is not None
        ) else 0

        # Input projection
        self.input_proj = nn.Linear(in_dim, hidden_dim)

        # Conv layers
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        for _ in range(num_layers):
            if conv_type == "gcn":
                self.convs.append(GCNConv(hidden_dim, hidden_dim, add_self_loops=False))
            elif conv_type == "sage":
                self.convs.append(SAGEConv(hidden_dim, hidden_dim))
            elif conv_type == "gat":
                self.convs.append(GATConv(
                    hidden_dim, hidden_dim,
                    heads=num_heads,
                    edge_dim=edge_feat_dim if edge_feat_dim > 0 else None,
                    add_self_loops=False,
                    concat=False,   # average over heads → output is hidden_dim
                ))
            else:
                raise ValueError(f"Unknown conv_type: {conv_type!r}. Choose gcn | sage | gat")
            self.norms.append(nn.LayerNorm(hidden_dim))

        self.dropout = nn.Dropout(dropout)

        # Edge classifier: concat(src, dst, edge_attr) → 1
        classifier_in = hidden_dim * 2 + edge_feat_dim
        self.classifier = nn.Sequential(
            nn.Linear(classifier_in, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, data):
        x          = data.x
        edge_index = data.edge_index
        edge_attr  = getattr(data, "edge_attr", None)

        # Input projection
        x = self.dropout(torch.relu(self.input_proj(x)))

        # Conv layers
        for conv, norm in zip(self.convs, self.norms):
            if self.conv_type == "gat" and edge_attr is not None:
                x = conv(x, edge_index, edge_attr=edge_attr)
            else:
                x = conv(x, edge_index)
            x = self.dropout(torch.relu(norm(x)))

        # Edge scoring
        src_emb = x[edge_index[0]]
        dst_emb = x[edge_index[1]]

        if edge_attr is not None:
            edge_emb = torch.cat([src_emb, dst_emb, edge_attr], dim=1)
        else:
            edge_emb = torch.cat([src_emb, dst_emb], dim=1)

        return self.classifier(edge_emb).squeeze(-1)
