"""
L2 — Homogeneous GNN models: GCN and GraphSAGE.

Standard architectures on the unified (single node type, single edge type) graph.
Support both node classification and edge classification modes.
"""

import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv, SAGEConv


class HomoGNN(nn.Module):
    """
    Flexible homogeneous GNN supporting GCN or GraphSAGE backbone.

    Args:
        in_dim: input feature dimension
        hidden_dim: hidden layer dimension
        num_layers: number of GNN layers
        dropout: dropout rate
        conv_type: "gcn" or "sage"
        task: "node" or "edge"
    """

    def __init__(self, in_dim, hidden_dim=64, num_layers=2, dropout=0.3,
                 conv_type="sage", task="node"):
        super().__init__()
        self.task = task
        self.dropout = nn.Dropout(dropout)

        ConvClass = GCNConv if conv_type == "gcn" else SAGEConv

        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()

        # First layer
        self.convs.append(ConvClass(in_dim, hidden_dim))
        self.norms.append(nn.LayerNorm(hidden_dim))

        # Middle layers
        for _ in range(num_layers - 1):
            self.convs.append(ConvClass(hidden_dim, hidden_dim))
            self.norms.append(nn.LayerNorm(hidden_dim))

        if task == "node":
            self.classifier = nn.Linear(hidden_dim, 1)
        else:
            # Edge classification: concat src + dst embeddings
            self.classifier = nn.Sequential(
                nn.Linear(hidden_dim * 2, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, 1),
            )

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        for conv, norm in zip(self.convs, self.norms):
            x = conv(x, edge_index)
            x = norm(x)
            x = torch.relu(x)
            x = self.dropout(x)

        if self.task == "node":
            return self.classifier(x).squeeze(-1)
        else:
            # Edge classification: score each edge by concat(src, dst)
            src, dst = edge_index
            edge_emb = torch.cat([x[src], x[dst]], dim=1)
            return self.classifier(edge_emb).squeeze(-1)
