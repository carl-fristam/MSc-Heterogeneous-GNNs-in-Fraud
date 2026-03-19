"""
Heterogeneous Graph Transformer (HGT) for fraud classification.

Supports both node classification (labels on transaction nodes) and
edge classification (labels on edges between account nodes).

Uses PyG's HGTConv with per-type input projections.
"""

import torch
import torch.nn as nn
from torch_geometric.nn import HGTConv


class HGT(nn.Module):
    """
    Args:
        data: HeteroData object (used to infer metadata and feature dims)
        hidden_dim: hidden dimension for all layers
        num_heads: number of attention heads per HGTConv layer
        num_layers: number of HGTConv layers
        dropout: dropout rate
        task: "node" or "edge"
            - "node": classify a target node type (e.g. "transaction")
            - "edge": classify edges by concatenating src + dst node embeddings
        target_node_type: which node type to classify (node mode only)
    """

    def __init__(self, data, hidden_dim=64, num_heads=4, num_layers=2,
                 dropout=0.3, task="node", target_node_type="transaction"):
        super().__init__()
        self.task = task
        self.target_node_type = target_node_type

        metadata = data.metadata()

        # Per-type input projection
        self.input_proj = nn.ModuleDict()
        for ntype in metadata[0]:
            in_dim = data[ntype].x.size(1)
            self.input_proj[ntype] = nn.Linear(in_dim, hidden_dim)

        # HGT layers
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        for _ in range(num_layers):
            self.convs.append(HGTConv(hidden_dim, hidden_dim, metadata, heads=num_heads))
            self.norms.append(nn.LayerNorm(hidden_dim))

        self.dropout = nn.Dropout(dropout)

        if task == "node":
            self.classifier = nn.Linear(hidden_dim, 1)
        else:
            self.classifier = nn.Sequential(
                nn.Linear(hidden_dim * 2, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, 1),
            )

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

        if self.task == "node":
            return self.classifier(x_dict[self.target_node_type]).squeeze(-1)
        else:
            # Return node embeddings — edge scoring done in training loop
            # because we need to iterate over edge types
            return x_dict
