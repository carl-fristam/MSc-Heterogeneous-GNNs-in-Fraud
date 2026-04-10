"""
Heterogeneous GAT (HeteroGAT) for edge classification.

Uses PyG's HeteroConv to apply a separate GATConv per edge type,
then aggregates messages at each node. This preserves edge-type
identity during message passing but uses simpler per-type attention
rather than HGT's cross-type transformer attention.

Contrast with HGT:
  HGT     — cross-type attention, type-specific K/Q/V projections
  HeteroGAT — per-type GAT, aggregated at nodes, type-aware but lighter

Both support the same Trainer interface (forward takes HeteroData,
returns x_dict for edge mode or logits for node mode).

Usage:
    from src.heterogeneous.hetero_gat.model import HeteroGAT
    model = HeteroGAT(data, hidden_dim=64, num_heads=4, num_layers=2)
"""

import torch
import torch.nn as nn
from torch_geometric.nn import HeteroConv, GATConv


class HeteroGAT(nn.Module):
    """
    Args:
        data:             HeteroData (used to infer metadata and feature dims)
        hidden_dim:       node embedding dimension
        num_heads:        GAT attention heads per layer
        num_layers:       number of HeteroConv layers
        dropout:          dropout rate
        task:             "node" or "edge"
        target_node_type: node type to classify (node task only)
    """

    def __init__(self, data, hidden_dim=64, num_heads=4, num_layers=2,
                 dropout=0.3, task="edge", target_node_type="internal_account"):
        super().__init__()
        self.task = task
        self.target_node_type = target_node_type

        metadata = data.metadata()
        node_types, edge_types = metadata

        # Per-type input projection (each node type has its own feature dim)
        self.input_proj = nn.ModuleDict({
            nt: nn.Linear(data[nt].x.shape[1], hidden_dim)
            for nt in node_types
        })

        # HeteroConv layers — one GATConv per edge type
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        for _ in range(num_layers):
            conv_dict = {}
            for et in edge_types:
                _, _, dst_type = et
                # edge_dim: use edge features in GAT attention if available
                edge_feat_dim = (
                    data[et].edge_attr.shape[1]
                    if hasattr(data[et], "edge_attr") and data[et].edge_attr is not None
                    else None
                )
                conv_dict[et] = GATConv(
                    hidden_dim, hidden_dim,
                    heads=num_heads,
                    edge_dim=edge_feat_dim,
                    add_self_loops=False,
                    concat=False,   # average over heads → output stays hidden_dim
                )
            self.convs.append(HeteroConv(conv_dict, aggr="sum"))
            self.norms.append(nn.ModuleDict({
                nt: nn.LayerNorm(hidden_dim) for nt in node_types
            }))

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
            nt: self.dropout(torch.relu(self.input_proj[nt](data[nt].x)))
            for nt in data.node_types
        }

        # Build edge_attr_dict for GATConv attention
        edge_attr_dict = {
            et: data[et].edge_attr
            for et in data.edge_types
            if hasattr(data[et], "edge_attr") and data[et].edge_attr is not None
        }

        for conv, norm_dict in zip(self.convs, self.norms):
            x_dict = conv(x_dict, data.edge_index_dict, edge_attr_dict=edge_attr_dict)
            x_dict = {
                nt: self.dropout(torch.relu(norm_dict[nt](x)))
                for nt, x in x_dict.items()
            }

        if self.task == "node":
            return self.classifier(x_dict[self.target_node_type]).squeeze(-1)
        else:
            return x_dict
