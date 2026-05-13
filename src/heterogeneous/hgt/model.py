"""HGT (Hu et al., 2020) wrapped around PyG's HGTConv.

Each node type gets its own input projection. For edge classification
the model returns the per-type embedding dict and the Trainer scores
edges by concatenating src/dst embeddings with the raw edge features.
"""

import torch
import torch.nn as nn
from torch_geometric.nn import HGTConv


class HGT(nn.Module):
    """HGT backbone + classifier head.

    Pass a HeteroData; the model uses it once to infer node/edge metadata
    and per-type feature dims. `task="edge"` returns the embedding dict;
    `task="node"` returns logits for `target_node_type`.
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

        # Edge feature dim for the classifier (edge features are concatenated
        # with node embeddings when scoring edges in the Trainer)
        edge_feat_dim = 0
        if task == "edge":
            for et in data.edge_types:
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
            return x_dict

    @torch.no_grad()
    def extract_attention(self, data):
        """Run forward pass with hooks to capture HGTConv attention weights."""
        self.eval()
        attn_weights = {}

        def make_hook(layer_idx):
            def hook_fn(module, inputs, outputs):
                if hasattr(module, '_alpha') and module._alpha is not None:
                    attn_weights[layer_idx] = module._alpha.detach().cpu()
            return hook_fn

        hooks = []
        for i, conv in enumerate(self.convs):
            hooks.append(conv.register_forward_hook(make_hook(i)))

        self.forward(data)

        for h in hooks:
            h.remove()

        return attn_weights
