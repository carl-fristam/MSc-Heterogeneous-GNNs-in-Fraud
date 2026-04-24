"""
HMPNN — Heterogeneous Message Passing Neural Network.

Original:  Johannessen & Jullum (2023)
           https://github.com/fredjo89/heterogeneous-mpnn

---------------------------------------------------------------------------
What we kept from the original
---------------------------------------------------------------------------
- Two aggregation variants:
    CT  (default) — one NNConv per relation, messages concatenated across
                    relations then projected. Their main model.
    Sum           — all relations summed via a single HeteroConv. Ablation.
- aggr="sum" within each relation via NNConv — matches the original exactly.
- Sigmoid activations throughout each layer.
- NNConv as the message function: edge features produce a weight matrix that
  modulates the source node embedding (Gilmer et al., 2017).

---------------------------------------------------------------------------
What we changed
---------------------------------------------------------------------------
- Task: node classification → edge classification.
  The HMPNN layers still run identically and produce node embeddings.
  The Trainer then scores each transaction edge by concatenating
  (src_emb, dst_emb) and passing through an MLP classifier head.

- Forward signature: the original takes (x_dict, edge_index_dict,
  edge_attr_dict) separately. Ours accepts a HeteroData object directly
  and unpacks internally, matching the shared Trainer interface used by
  HGT and HeteroGAT.

- Classifier: self.classifier is an MLP on hidden_dim * 2 inputs for the
  edge task, or a single Linear for the node task.

---------------------------------------------------------------------------
Graph structure
---------------------------------------------------------------------------
- Two node types: internal_account, external_account
- V1: two edge types — onus_transfer (internal → internal),
                        external_transfer (internal → external)
- V2: five edge types — onus_transfer + four external types split by
                        payment rail (realtime, giro, future, other)
- Each edge carries transaction features (amount, channel, payment method,
  currency, destination, time encoding).
- Labels live on edges — each transaction is either fraud or not.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import NNConv, HeteroConv


class HMPNNLayer(nn.Module):
    """
    One message-passing step across all node types.

    CT variant:
        For each incoming relation, NNConv (aggr="sum") produces a
        dim_message vector per destination node. Vectors from all
        relations are concatenated, then a Linear projects to dim_out.
        Matches models_HMPNN_ct.py from the original repo.

    Sum variant:
        All relations passed into one HeteroConv(aggr="sum"), producing
        a single dim_message vector per node. Then projected to dim_out.
        Matches models_HMPNN_sum.py — used as ablation.
    """

    def __init__(self, data, dim_in: dict, dim_out: int, dim_message: int, aggr: str = "ct"):
        super().__init__()
        self.aggr = aggr
        self.node_types = list(data.node_types)
        self.dim_out = dim_out
        self._dim_message = dim_message

        self.incoming: dict = {nt: [] for nt in self.node_types}

        if aggr == "ct":
            self.convs = nn.ModuleDict()
            for src_type, rel, dst_type in data.edge_types:
                et = (src_type, rel, dst_type)
                if not (hasattr(data[et], "edge_attr") and data[et].edge_attr is not None):
                    continue
                key = f"{src_type}__{rel}__{dst_type}"
                num_edge_feats = data[et].edge_attr.shape[1]
                message_nn = nn.Sequential(
                    nn.Linear(num_edge_feats, 32),
                    nn.ReLU(),
                    nn.Linear(32, dim_in[src_type] * dim_message),
                )
                self.convs[key] = NNConv(
                    in_channels=(dim_in[src_type], dim_in[dst_type]),
                    out_channels=dim_message,
                    nn=message_nn,
                    aggr="sum",  # sum within relation — matches original
                )
                self.incoming[dst_type].append((src_type, rel, dst_type, key))

            # concat across relations → project to dim_out
            self.projs = nn.ModuleDict()
            for nt in self.node_types:
                n_rels = max(len(self.incoming[nt]), 1)
                self.projs[nt] = nn.Linear(n_rels * dim_message, dim_out)

        else:  # sum
            mp_dict = {}
            for src_type, rel, dst_type in data.edge_types:
                et = (src_type, rel, dst_type)
                if not (hasattr(data[et], "edge_attr") and data[et].edge_attr is not None):
                    continue
                num_edge_feats = data[et].edge_attr.shape[1]
                message_nn = nn.Sequential(
                    nn.Linear(num_edge_feats, 32),
                    nn.ReLU(),
                    nn.Linear(32, dim_in[src_type] * dim_message),
                )
                mp_dict[(src_type, rel, dst_type)] = NNConv(
                    in_channels=(dim_in[src_type], dim_in[dst_type]),
                    out_channels=dim_message,
                    nn=message_nn,
                    aggr="sum",
                )
                self.incoming[dst_type].append((src_type, rel, dst_type, None))
            self.hetero_conv = HeteroConv(mp_dict, aggr="sum")

            self.projs = nn.ModuleDict()
            for nt in self.node_types:
                self.projs[nt] = nn.Linear(dim_message, dim_out)

    def forward(self, x_dict, edge_index_dict, edge_attr_dict):
        out = {}

        if self.aggr == "ct":
            for nt in self.node_types:
                rels = self.incoming[nt]
                if not rels:
                    out[nt] = x_dict[nt].new_zeros(x_dict[nt].shape[0], self.dim_out)
                    continue
                msgs = []
                for src_type, rel, dst_type, key in rels:
                    et = (src_type, rel, dst_type)
                    msg = self.convs[key](
                        (x_dict[src_type], x_dict[dst_type]),
                        edge_index_dict[et],
                        edge_attr_dict[et],
                    )
                    msgs.append(torch.sigmoid(msg))  # sigmoid per relation — matches original
                out[nt] = torch.sigmoid(self.projs[nt](torch.cat(msgs, dim=1)))

        else:  # sum
            hetero_out = self.hetero_conv(x_dict, edge_index_dict, edge_attr_dict)
            for nt in self.node_types:
                if nt in hetero_out:
                    out[nt] = torch.sigmoid(self.projs[nt](hetero_out[nt]))
                else:
                    out[nt] = x_dict[nt].new_zeros(x_dict[nt].shape[0], self.dim_out)

        return out


class HMPNNLayerAnalysis:
    """Captured message norms from a single HMPNNLayer forward pass."""
    def __init__(self):
        self.norms = {}


class HMPNN(nn.Module):
    """
    Full HMPNN: stacked HMPNNLayer instances + edge classification head.

    Args:
        data:             HeteroData object.
        target_node_type: node type whose embeddings feed the edge classifier.
        num_layers:       number of message-passing layers.
        hidden_dim:       node embedding dimension.
        message_dim:      per-relation message size inside each layer.
        dropout:          dropout applied between layers.
        task:             "edge" (default) or "node".
        aggr:             "ct" (concatenation, main model) or "sum" (ablation).
    """

    def __init__(
        self,
        data,
        target_node_type: str = "internal_account",
        num_layers: int = 2,
        hidden_dim: int = 64,
        message_dim: int = 32,
        dropout: float = 0.3,
        task: str = "edge",
        aggr: str = "ct",
    ):
        super().__init__()
        self.target_node_type = target_node_type
        self.node_types = list(data.node_types)
        self.task = task
        self.aggr = aggr

        dim_in = {nt: data[nt].x.shape[1] for nt in data.node_types}

        self.mp_layers = nn.ModuleList()
        for _ in range(num_layers):
            self.mp_layers.append(
                HMPNNLayer(data, dim_in=dim_in, dim_out=hidden_dim,
                           dim_message=message_dim, aggr=aggr)
            )
            dim_in = {nt: hidden_dim for nt in data.node_types}

        self.dropout = nn.Dropout(dropout)

        edge_feat_dim = 0
        if task == "edge":
            for et in data.edge_types:
                if hasattr(data[et], "edge_attr") and data[et].edge_attr is not None:
                    edge_feat_dim = data[et].edge_attr.shape[1]
                    break

        if task == "node":
            self.classifier = nn.Linear(hidden_dim, 1)
        else:
            self.classifier = nn.Sequential(
                nn.Linear(hidden_dim * 2 + edge_feat_dim, hidden_dim),
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
            return x_dict

    @torch.no_grad()
    def extract_message_norms(self, data):
        """Run forward and capture L2 norm of messages per relation per layer."""
        self.eval()
        x_dict = {nt: data[nt].x for nt in self.node_types}
        edge_index_dict = data.edge_index_dict
        edge_attr_dict = self._edge_attr_dict(data)

        layer_norms = {}
        for layer_idx, layer in enumerate(self.mp_layers):
            if layer.aggr == "ct":
                norms = {}
                for src_type, rel, dst_type, key in [
                    r for rels in layer.incoming.values() for r in rels
                ]:
                    et = (src_type, rel, dst_type)
                    msg = layer.convs[key](
                        (x_dict[src_type], x_dict[dst_type]),
                        edge_index_dict[et],
                        edge_attr_dict[et],
                    )
                    norms[et] = msg.norm(dim=1).cpu().numpy()
                layer_norms[layer_idx] = norms

            x_dict = layer(x_dict, edge_index_dict, edge_attr_dict)
            x_dict = {nt: self.dropout(x) for nt, x in x_dict.items()}

        return layer_norms
