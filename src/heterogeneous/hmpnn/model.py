"""
HMPNN — Heterogeneous Message Passing Neural Network.
Based on Johannessen & Jullum (2023).

Architecture
------------
Each HMPNNLayer performs one full message-passing step across the full
heterogeneous graph: for every node type, it gathers edge-feature-weighted
messages from all incoming edge types and projects them to a common hidden dim.

NNConv (Gilmer et al., 2017) maps per-edge features to a weight matrix that
modulates the neighbour message, allowing transaction-level features (amount,
time delta, …) to influence propagation rather than treating all edges equally.

Forward contract (mirrors HGT)
-------------------------------
- node task : returns logits tensor of shape (N_target,)
- edge task  : returns x_dict; trainer scores edges via self.classifier
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import NNConv


class HMPNNLayer(nn.Module):
    """
    One message-passing step that updates every node type.

    For each incoming edge type (with edge_attr):
      1. NNConv uses edge features to weight the source-node message.
      2. Messages from all relations arriving at a node type are concatenated.
      3. A linear projection maps the concatenation to dim_out.

    Node types with no incoming edge-attributed edges receive a zero tensor.
    """

    def __init__(self, data, dim_in: dict, dim_out: int, dim_message: int):
        """
        Args:
            data:        HeteroData (read-only, used only here to infer shapes).
            dim_in:      {node_type: feature dim} for the inputs to this layer.
            dim_out:     output feature dim for every node type.
            dim_message: per-relation message size before the final projection.
        """
        super().__init__()
        self.dim_out = dim_out
        self.node_types = list(data.node_types)

        # One NNConv per edge type that carries edge attributes.
        # Key format: "src_type__rel__dst_type"
        self.convs = nn.ModuleDict()
        # incoming[dst_type] = [(src_type, rel, dst_type, conv_key), ...]
        self.incoming: dict = {nt: [] for nt in self.node_types}

        for src_type, rel, dst_type in data.edge_types:
            et = (src_type, rel, dst_type)
            if not (hasattr(data[et], "edge_attr") and data[et].edge_attr is not None):
                continue

            key = f"{src_type}__{rel}__{dst_type}"
            num_edge_feats = data[et].edge_attr.shape[1]
            dim_src = dim_in[src_type]
            dim_dst = dim_in[dst_type]

            # Maps each edge's features to the (dim_src × dim_message) weight matrix
            # used by NNConv to scale the source-node embedding.
            message_nn = nn.Sequential(
                nn.Linear(num_edge_feats, 32),
                nn.ReLU(),
                nn.Linear(32, dim_src * dim_message),
            )
            self.convs[key] = NNConv(
                in_channels=(dim_src, dim_dst),
                out_channels=dim_message,
                nn=message_nn,
                aggr="mean",
            )
            self.incoming[dst_type].append((src_type, rel, dst_type, key))

        # Per-node-type projection: concat of all incoming messages → dim_out
        self.projs = nn.ModuleDict()
        for nt in self.node_types:
            n_rels = max(len(self.incoming[nt]), 1)
            self.projs[nt] = nn.Linear(n_rels * dim_message, dim_out)

        self._dim_message = dim_message

    def forward(self, x_dict, edge_index_dict, edge_attr_dict):
        out = {}
        for nt in self.node_types:
            rels = self.incoming[nt]
            if not rels:
                # No incoming attributed edges — propagate zeros
                n = x_dict[nt].shape[0]
                out[nt] = x_dict[nt].new_zeros(n, self.dim_out)
                continue

            msgs = []
            for src_type, rel, dst_type, key in rels:
                et = (src_type, rel, dst_type)
                msg = self.convs[key](
                    (x_dict[src_type], x_dict[dst_type]),
                    edge_index_dict[et],
                    edge_attr_dict[et],
                )
                msgs.append(F.relu(msg))

            out[nt] = F.relu(self.projs[nt](torch.cat(msgs, dim=1)))

        return out


class HMPNN(nn.Module):
    """
    Full HMPNN: num_layers HMPNNLayer instances + task head.

    Args:
        data:             HeteroData object.
        target_node_type: node type to classify (node task) or produce
                          embeddings for (edge task).
        num_layers:       number of message-passing layers.
        hidden_dim:       node embedding dimension throughout.
        message_dim:      per-relation message size inside each layer.
        dropout:          dropout applied after each layer.
        task:             "node" or "edge".
    """

    def __init__(
        self,
        data,
        target_node_type: str = "transaction",
        num_layers: int = 2,
        hidden_dim: int = 16,
        message_dim: int = 8,
        dropout: float = 0.0,
        task: str = "node",
    ):
        super().__init__()
        self.target_node_type = target_node_type
        self.node_types = list(data.node_types)
        self.task = task

        dim_in = {nt: data[nt].x.shape[1] for nt in data.node_types}

        self.mp_layers = nn.ModuleList()
        for _ in range(num_layers):
            self.mp_layers.append(
                HMPNNLayer(data, dim_in=dim_in, dim_out=hidden_dim, dim_message=message_dim)
            )
            dim_in = {nt: hidden_dim for nt in data.node_types}

        self.dropout = nn.Dropout(dropout)

        if task == "node":
            self.classifier = nn.Linear(hidden_dim, 1)
        else:
            # Trainer scores edges via self.classifier using concat(src_emb, dst_emb)
            self.classifier = nn.Sequential(
                nn.Linear(hidden_dim * 2, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, 1),
            )

    def _edge_attr_dict(self, data):
        return {
            et: data[et].edge_attr
            for et in data.edge_types
            if hasattr(data[et], "edge_attr") and data[et].edge_attr is not None
        }

    def forward(self, data):
        x_dict = {nt: data[nt].x for nt in self.node_types}
        edge_index_dict = data.edge_index_dict
        edge_attr_dict = self._edge_attr_dict(data)

        for layer in self.mp_layers:
            x_dict = layer(x_dict, edge_index_dict, edge_attr_dict)
            x_dict = {nt: self.dropout(x) for nt, x in x_dict.items()}

        if self.task == "node":
            return self.classifier(x_dict[self.target_node_type]).squeeze(-1)
        else:
            return x_dict  # Trainer scores edges via self.classifier
