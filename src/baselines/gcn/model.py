"""
Graph Convolutional Network (GCN) for Money Laundering Detection.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv


class GCN(nn.Module):
    """
    Graph Convolutional Network for node classification.

    Args:
        in_channels: Number of input features per node
        hidden_channels: Hidden dimension size
        out_channels: Number of output classes (2 for binary classification)
        num_layers: Number of GCN layers
        dropout: Dropout rate
    """
    def __init__(self, in_channels, hidden_channels=64, out_channels=2,
                 num_layers=2, dropout=0.3):
        super(GCN, self).__init__()

        self.num_layers = num_layers
        self.dropout = dropout

        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()

        # First layer
        self.convs.append(GCNConv(in_channels, hidden_channels))
        self.bns.append(nn.BatchNorm1d(hidden_channels))

        # Hidden layers
        for _ in range(num_layers - 2):
            self.convs.append(GCNConv(hidden_channels, hidden_channels))
            self.bns.append(nn.BatchNorm1d(hidden_channels))

        # Last layer
        if num_layers > 1:
            self.convs.append(GCNConv(hidden_channels, out_channels))
        else:
            self.convs[-1] = GCNConv(in_channels, out_channels)

    def forward(self, x, edge_index):
        """
        Forward pass.

        Args:
            x: Node features [num_nodes, in_channels]
            edge_index: Edge indices [2, num_edges]

        Returns:
            Node embeddings [num_nodes, out_channels]
        """
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, edge_index)
            x = self.bns[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

        # Final layer (no activation, no dropout)
        x = self.convs[-1](x, edge_index)

        return x

    def predict(self, x, edge_index):
        """Get class predictions."""
        self.eval()
        with torch.no_grad():
            logits = self.forward(x, edge_index)
            return torch.argmax(logits, dim=1)
