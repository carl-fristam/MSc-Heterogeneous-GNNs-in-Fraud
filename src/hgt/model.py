"""
Heterogeneous Graph Transformer (HGT) for transaction-level fraud classification.

Uses PyG's HGTConv on the bipartite Account ↔ Transaction graph.
"""

import torch
import torch.nn as nn
from torch_geometric.nn import HGTConv


class HGT(nn.Module):
    def __init__(self, data, hidden_dim=64, num_heads=4, num_layers=2, dropout=0.3):
        super().__init__()
        metadata = data.metadata()
        node_types = metadata[0]

        # Per-type input projection
        self.input_proj = nn.ModuleDict()
        for ntype in node_types:
            in_dim = data[ntype].x.size(1)
            self.input_proj[ntype] = nn.Linear(in_dim, hidden_dim)

        # HGT layers
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        for _ in range(num_layers):
            self.convs.append(HGTConv(hidden_dim, hidden_dim, metadata, heads=num_heads))
            self.norms.append(nn.LayerNorm(hidden_dim))

        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_dim, 1)

    def forward(self, data):
        x_dict = {
            ntype: self.dropout(torch.relu(self.input_proj[ntype](data[ntype].x)))
            for ntype in data.node_types
        }

        for conv, norm in zip(self.convs, self.norms):
            x_dict = conv(x_dict, data.edge_index_dict)
            x_dict = {
                ntype: self.dropout(torch.relu(norm(x)))
                for ntype, x in x_dict.items()
            }

        # Classify transaction nodes only
        return self.classifier(x_dict["transaction"]).squeeze(-1)
