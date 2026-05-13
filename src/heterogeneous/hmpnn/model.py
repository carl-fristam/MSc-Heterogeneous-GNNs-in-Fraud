"""HMPNN: Johannessen & Jullum (2023), https://github.com/fredjo89/heterogeneous-mpnn

Two aggregation variants are exposed:
    "ct"  (default) one NNConv per relation, outputs concatenated and projected
    "sum"           all relations summed via a single HeteroConv (ablation)

NNConv (Gilmer et al., 2017) is the per-relation message function: edge
features produce a weight matrix that modulates the source embedding.
Within each relation we keep the original's aggr="sum" and per-layer sigmoid.

Differences from the original:
    - Edge classification instead of node classification. The layers still
      produce node embeddings; the Trainer scores edges by concatenating
      (src_emb, dst_emb, edge_attr) through an MLP head.
    - `forward` takes a HeteroData object, matching the shared Trainer
      interface used by HGT and HeteroGAT.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import NNConv, HeteroConv


class HMPNNLayer(nn.Module):
    """One message-passing step across all node types.

    "ct":  one NNConv per relation, dim_message vectors concatenated across
           relations then projected to dim_out. Matches models_HMPNN_ct.py.
    "sum": all relations passed through a single HeteroConv(aggr="sum")
           and projected. Matches models_HMPNN_sum.py (ablation).
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
    """Stacked HMPNNLayer + edge classification head.

    `aggr` selects the per-layer behaviour: "ct" is the paper's main model,
    "sum" is the ablation that drops cross-relation concatenation.
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
