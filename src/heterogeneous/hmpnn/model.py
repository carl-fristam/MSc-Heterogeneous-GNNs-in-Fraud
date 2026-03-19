"""
HMPNN — Heterogeneous Message Passing Neural Network.

Based on Johannessen & Jullum (2023).
Uses NNConv to incorporate edge features into message passing.
Supports both node and edge classification.
"""

from src.utils.compat import apply_pyg_compat_patch
apply_pyg_compat_patch()

import torch
import torch.nn as nn
from torch_geometric.nn import HeteroConv, NNConv


class HMPNNLayer(nn.Module):
    """
    Single HMPNN layer for one target node type.

    For each edge type ending at target_node_type:
    - Apply NNConv (message function uses edge features)
    - Concatenate all messages
    - Apply linear transformation
    """

    def __init__(self, data, target_node_type, dim_in=None, dim_message=16, dim_out=16):
        super().__init__()

        self.target_node_type = target_node_type

        if dim_in is None:
            self.dim_in = {nt: data[nt].x.shape[1] for nt in data.node_types}
        else:
            self.dim_in = dim_in

        self.dim_message = dim_message
        self.dim_out = dim_out

        self.convs = nn.ModuleList()
        self.edge_types_to_target = []

        for edge_type in data.edge_types:
            src_type, rel_type, dst_type = edge_type

            if dst_type != target_node_type:
                continue

            if not hasattr(data[edge_type], "edge_attr") or data[edge_type].edge_attr is None:
                continue

            self.edge_types_to_target.append(edge_type)

            num_edge_features = data[edge_type].edge_attr.shape[1]

            message_nn = nn.Sequential(
                nn.Linear(num_edge_features, 32),
                nn.ReLU(),
                nn.Linear(32, self.dim_in[src_type] * dim_message)
            )

            conv = NNConv(
                in_channels=(self.dim_in[src_type], self.dim_in[dst_type]),
                out_channels=dim_message,
                nn=message_nn,
                aggr='mean'
            )

            hetero_conv = HeteroConv({edge_type: conv}, aggr='sum')
            self.convs.append(hetero_conv)

        n_incoming = max(len(self.convs), 1)
        self.linear = nn.Linear(n_incoming * dim_message, dim_out)

    def forward(self, x_dict, edge_index_dict, edge_attr_dict):
        messages = []

        for conv in self.convs:
            out = conv(x_dict, edge_index_dict, edge_attr_dict)
            msg = torch.sigmoid(out[self.target_node_type])
            messages.append(msg)

        if not messages:
            n = x_dict[self.target_node_type].shape[0]
            device = x_dict[self.target_node_type].device
            return torch.zeros(n, self.dim_out, device=device)

        concat = torch.cat(messages, dim=1)
        return torch.sigmoid(self.linear(concat))


class HMPNN(nn.Module):
    """
    Full HMPNN model.

    Args:
        data: HeteroData object
        target_node_type: node type to produce embeddings for
        num_layers: number of HMPNN layers (1-3)
        hidden_dim: hidden dimension
        message_dim: message dimension in HMPNN layers
        task: "node" or "edge"
    """

    def __init__(self, data, target_node_type="transaction", num_layers=2,
                 hidden_dim=16, message_dim=8, task="node"):
        super().__init__()

        self.target_node_type = target_node_type
        self.num_layers = num_layers
        self.node_types = data.node_types
        self.task = task

        dim_in = {nt: data[nt].x.shape[1] for nt in data.node_types}

        if num_layers == 1:
            out_dim = 1 if task == "node" else hidden_dim
            self.layers = nn.ModuleList([
                HMPNNLayer(data, target_node_type, dim_in=dim_in,
                          dim_message=message_dim, dim_out=out_dim)
            ])
        else:
            self.layers = nn.ModuleList()

            # First layer: for ALL node types
            self.layer1_modules = nn.ModuleDict()
            for nt in data.node_types:
                self.layer1_modules[nt] = HMPNNLayer(
                    data, nt, dim_in=dim_in,
                    dim_message=message_dim, dim_out=hidden_dim
                )

            dim_in_next = {nt: hidden_dim for nt in data.node_types}

            # Middle layers (if num_layers > 2)
            if num_layers >= 3:
                self.layer2_modules = nn.ModuleDict()
                for nt in data.node_types:
                    self.layer2_modules[nt] = HMPNNLayer(
                        data, nt, dim_in=dim_in_next,
                        dim_message=message_dim, dim_out=hidden_dim
                    )

            # Final layer
            final_out = 1 if task == "node" else hidden_dim
            self.final_layer = HMPNNLayer(
                data, target_node_type, dim_in=dim_in_next,
                dim_message=message_dim * 2, dim_out=final_out
            )

        if task == "edge":
            self.edge_classifier = nn.Sequential(
                nn.Linear(hidden_dim * 2, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, 1),
            )

    def forward(self, x_dict, edge_index_dict, edge_attr_dict):
        if self.num_layers == 1:
            out = self.layers[0](x_dict, edge_index_dict, edge_attr_dict)
            if self.task == "node":
                return out
            else:
                return {self.target_node_type: out}

        # Layer 1: update all node types
        x_dict_new = {}
        for nt in self.node_types:
            x_dict_new[nt] = self.layer1_modules[nt](x_dict, edge_index_dict, edge_attr_dict)

        # Layer 2 (if exists)
        if self.num_layers >= 3:
            x_dict_tmp = {}
            for nt in self.node_types:
                x_dict_tmp[nt] = self.layer2_modules[nt](x_dict_new, edge_index_dict, edge_attr_dict)
            x_dict_new = x_dict_tmp

        out = self.final_layer(x_dict_new, edge_index_dict, edge_attr_dict)

        if self.task == "node":
            return out
        else:
            x_dict_new[self.target_node_type] = out
            return x_dict_new
