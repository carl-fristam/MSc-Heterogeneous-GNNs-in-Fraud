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

    Node mode: GNN produces per-node embeddings → linear classifier on
               transaction nodes.
    Edge mode: GNN produces per-node embeddings → concat(src, dst, edge_attr)
               → MLP classifier on edges. Edge features (transaction features)
               are concatenated with structural embeddings.
    """

    def __init__(self, in_dim, hidden_dim=64, num_layers=2, dropout=0.3,
                 conv_type="sage", task="node", edge_feat_dim=0):
        super().__init__()
        self.task = task
        self.dropout = nn.Dropout(dropout)

        ConvClass = GCNConv if conv_type == "gcn" else SAGEConv

        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()

        self.convs.append(ConvClass(in_dim, hidden_dim))
        self.norms.append(nn.LayerNorm(hidden_dim))

        for _ in range(num_layers - 1):
            self.convs.append(ConvClass(hidden_dim, hidden_dim))
            self.norms.append(nn.LayerNorm(hidden_dim))

        if task == "node":
            self.classifier = nn.Linear(hidden_dim, 1)
        else:
            clf_in = hidden_dim * 2 + edge_feat_dim
            self.classifier = nn.Sequential(
                nn.Linear(clf_in, hidden_dim),
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
            src, dst = edge_index
            parts = [x[src], x[dst]]
            if hasattr(data, "edge_attr") and data.edge_attr is not None:
                parts.append(data.edge_attr)
            edge_emb = torch.cat(parts, dim=1)
            return self.classifier(edge_emb).squeeze(-1)
