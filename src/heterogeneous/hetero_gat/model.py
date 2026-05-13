"""Per-relation GAT stack: a separate GATConv per edge type, aggregated
at the node with HeteroConv. Lighter than HGT — no cross-type attention,
just type-aware message passing with edge-feature-conditioned weights.

Same forward contract as HGT: HeteroData in, embedding dict out (edge task)
or logits out (node task).
"""

import torch
import torch.nn as nn
from torch_geometric.nn import HeteroConv, GATConv


class HeteroGAT(nn.Module):
    """HeteroConv(GATConv per edge type) + classifier head.

    Edge features are passed to GATConv via `edge_dim` so attention is
    conditioned on the transaction features, not just node embeddings.
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

        edge_feat_dim = 0
        if task == "edge":
            for et in edge_types:
                if hasattr(data[et], "y") and data[et].y is not None and \
                   hasattr(data[et], "edge_attr") and data[et].edge_attr is not None:
                    edge_feat_dim = data[et].edge_attr.shape[1]
                    break

        if task == "node":
            self.classifier = nn.Linear(hidden_dim, 1)
        else:
            self.classifier = nn.Sequential(
                nn.Linear(hidden_dim * 2 + edge_feat_dim, hidden_dim),
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

    @torch.no_grad()
    def extract_attention(self, data):
        """Extract per-edge-type attention weights from all layers."""
        self.eval()
        x_dict = {
            nt: self.dropout(torch.relu(self.input_proj[nt](data[nt].x)))
            for nt in data.node_types
        }

        edge_attr_dict = {
            et: data[et].edge_attr
            for et in data.edge_types
            if hasattr(data[et], "edge_attr") and data[et].edge_attr is not None
        }

        all_attention = {}

        for layer_idx, (conv, norm_dict) in enumerate(zip(self.convs, self.norms)):
            layer_attn = {}

            for et in data.edge_types:
                gat = conv.convs[et]
                src_type, _, dst_type = et

                src_x = x_dict[src_type]
                dst_x = x_dict[dst_type]
                edge_index = data[et].edge_index
                edge_attr = edge_attr_dict.get(et, None)

                _, (ei, alpha) = gat(
                    (src_x, dst_x), edge_index,
                    edge_attr=edge_attr,
                    return_attention_weights=True,
                )
                layer_attn[et] = alpha.cpu()

            all_attention[layer_idx] = layer_attn

            x_dict = conv(x_dict, data.edge_index_dict, edge_attr_dict=edge_attr_dict)
            x_dict = {
                nt: self.dropout(torch.relu(norm_dict[nt](x)))
                for nt, x in x_dict.items()
            }

        return all_attention
