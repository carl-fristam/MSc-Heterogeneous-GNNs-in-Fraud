"""
HMPNN-ct model adapted for SAML-D dataset.
Based on: https://github.com/fredjo89/heterogeneous-mpnn

Heterogeneous Message Passing Neural Network with concatenation aggregation.
Uses NNConv to incorporate edge features into message passing.
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

        # Get input dimensions for each node type
        if dim_in is None:
            self.dim_in = {nt: data[nt].x.shape[1] for nt in data.node_types}
        else:
            self.dim_in = dim_in

        self.dim_message = dim_message
        self.dim_out = dim_out

        # Create message passing operators for each edge type ending at target
        self.convs = nn.ModuleList()
        self.edge_types_to_target = []

        for edge_type in data.edge_types:
            src_type, rel_type, dst_type = edge_type

            if dst_type != target_node_type:
                continue

            self.edge_types_to_target.append(edge_type)

            # Edge feature dimension
            num_edge_features = data[edge_type].edge_attr.shape[1]

            # NNConv: uses a NN to compute message weights from edge features
            # The NN maps edge_attr -> (dim_in[src] * dim_message) matrix
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

        # Final linear layer: concat all messages -> output
        self.linear = nn.Linear(len(self.convs) * dim_message, dim_out)

    def forward(self, x_dict, edge_index_dict, edge_attr_dict):
        messages = []

        for conv in self.convs:
            out = conv(x_dict, edge_index_dict, edge_attr_dict)
            msg = torch.sigmoid(out[self.target_node_type])
            messages.append(msg)

        # Concatenate all messages
        concat = torch.cat(messages, dim=1)

        # Final transformation
        return torch.sigmoid(self.linear(concat))


class HMPNN(nn.Module):
    """
    Full HMPNN model for node classification.

    Args:
        data: HeteroData object
        target_node_type: Node type to classify
        num_layers: Number of HMPNN layers (1-3)
        hidden_dim: Hidden dimension for intermediate layers
        message_dim: Message dimension in HMPNN layers
    """

    def __init__(self, data, target_node_type='account', num_layers=2,
                 hidden_dim=16, message_dim=8):
        super().__init__()

        self.target_node_type = target_node_type
        self.num_layers = num_layers
        self.node_types = data.node_types

        # Get input dimensions
        dim_in = {nt: data[nt].x.shape[1] for nt in data.node_types}

        if num_layers == 1:
            # Single layer directly to output
            self.layers = nn.ModuleList([
                HMPNNLayer(data, target_node_type, dim_in=dim_in,
                          dim_message=message_dim, dim_out=1)
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

            # Middle layers (if num_layers > 2)
            dim_in_next = {nt: hidden_dim for nt in data.node_types}

            if num_layers >= 3:
                self.layer2_modules = nn.ModuleDict()
                for nt in data.node_types:
                    self.layer2_modules[nt] = HMPNNLayer(
                        data, nt, dim_in=dim_in_next,
                        dim_message=message_dim, dim_out=hidden_dim
                    )

            # Final layer: only for target node type
            self.final_layer = HMPNNLayer(
                data, target_node_type, dim_in=dim_in_next,
                dim_message=message_dim * 2, dim_out=1
            )

    def forward(self, x_dict, edge_index_dict, edge_attr_dict):
        if self.num_layers == 1:
            return self.layers[0](x_dict, edge_index_dict, edge_attr_dict)

        # Layer 1: update all node types
        x_dict_new = {}
        for nt in self.node_types:
            x_dict_new[nt] = self.layer1_modules[nt](x_dict, edge_index_dict, edge_attr_dict)

        # Layer 2 (if exists): update all node types
        if self.num_layers >= 3:
            x_dict_tmp = {}
            for nt in self.node_types:
                x_dict_tmp[nt] = self.layer2_modules[nt](x_dict_new, edge_index_dict, edge_attr_dict)
            x_dict_new = x_dict_tmp

        # Final layer: only target node type
        return self.final_layer(x_dict_new, edge_index_dict, edge_attr_dict)


if __name__ == '__main__':
    from src.utils.config import PROJECT_ROOT
    from src.data.saml_hetero import load_hetero_saml_data

    # Load small sample
    data, _ = load_hetero_saml_data(
        sample_ratio=0.01,
        cache_path=str(PROJECT_ROOT / 'data' / 'processed' / 'saml_hetero_test.pkl')
    )

    print("\nTesting HMPNN model...")
    model = HMPNN(data, target_node_type='account', num_layers=2)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters())}")

    # Forward pass
    x_dict = {nt: data[nt].x for nt in data.node_types}
    edge_index_dict = {et: data[et].edge_index for et in data.edge_types}
    edge_attr_dict = {et: data[et].edge_attr for et in data.edge_types}

    out = model(x_dict, edge_index_dict, edge_attr_dict)
    print(f"Output shape: {out.shape}")
    print(f"Output range: [{out.min():.4f}, {out.max():.4f}]")
