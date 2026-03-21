"""
LaundroGraph — Bipartite link-prediction encoder for self-supervised pretraining.

Encoder: HeteroConv with per-edge-type SAGEConv, 2 layers.
Pretext task: Link prediction with negative sampling on the bipartite account graph.
Output: Account embeddings → concat src+dst → MLP classifier for edge classification.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import HeteroConv, SAGEConv


class LaundroGraphEncoder(nn.Module):
    """HeteroConv encoder with per-edge-type SAGEConv layers."""

    def __init__(self, metadata, feat_dims, hidden_dim=64, num_layers=2, dropout=0.3):
        super().__init__()

        self.input_proj = nn.ModuleDict()
        for ntype in metadata[0]:
            self.input_proj[ntype] = nn.Linear(feat_dims[ntype], hidden_dim)

        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        for i in range(num_layers):
            conv_dict = {}
            for et in metadata[1]:
                conv_dict[et] = SAGEConv((-1, -1), hidden_dim)
            self.convs.append(HeteroConv(conv_dict, aggr="sum"))
            self.norms.append(nn.ModuleDict({
                ntype: nn.LayerNorm(hidden_dim) for ntype in metadata[0]
            }))

        self.dropout = nn.Dropout(dropout)

    def forward(self, x_dict, edge_index_dict):
        h_dict = {
            ntype: self.dropout(torch.relu(self.input_proj[ntype](x)))
            for ntype, x in x_dict.items()
        }
        for conv, norm_dict in zip(self.convs, self.norms):
            h_dict = conv(h_dict, edge_index_dict)
            h_dict = {
                ntype: self.dropout(torch.relu(norm_dict[ntype](h)))
                for ntype, h in h_dict.items()
            }
        return h_dict


class LaundroGraph(nn.Module):
    """
    Self-supervised link prediction model for heterogeneous graphs.

    Pretraining mode: call pretrain_forward() for link prediction loss.
    Inference mode: call forward(data) to get x_dict for downstream classifier.

    Args:
        data: HeteroData object
        hidden_dim: latent dimension
        num_layers: number of HeteroConv layers
        dropout: dropout rate
        num_neg_samples: negative samples per positive edge
    """

    def __init__(self, data, hidden_dim=64, num_layers=2, dropout=0.3,
                 num_neg_samples=5):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_neg_samples = num_neg_samples

        metadata = data.metadata()
        self.node_types = metadata[0]
        self.edge_types = metadata[1]

        feat_dims = {ntype: data[ntype].x.size(1) for ntype in self.node_types}

        self.encoder = LaundroGraphEncoder(
            metadata, feat_dims, hidden_dim, num_layers, dropout
        )

    def pretrain_forward(self, data):
        """Link prediction loss with negative sampling."""
        x_dict = {ntype: data[ntype].x for ntype in self.node_types}
        h_dict = self.encoder(x_dict, data.edge_index_dict)

        total_loss = 0.0
        count = 0

        for et in self.edge_types:
            src_type, _, dst_type = et
            edge_index = data[et].edge_index
            num_edges = edge_index.size(1)
            if num_edges == 0:
                continue

            src_emb = h_dict[src_type]
            dst_emb = h_dict[dst_type]

            # Positive scores
            pos_score = (src_emb[edge_index[0]] * dst_emb[edge_index[1]]).sum(dim=-1)

            # Negative sampling
            num_neg = self.num_neg_samples * num_edges
            neg_dst_idx = torch.randint(0, dst_emb.size(0), (num_neg,), device=edge_index.device)
            neg_src_idx = edge_index[0].repeat(self.num_neg_samples)
            neg_score = (src_emb[neg_src_idx] * dst_emb[neg_dst_idx]).sum(dim=-1)

            pos_loss = F.binary_cross_entropy_with_logits(
                pos_score, torch.ones_like(pos_score)
            )
            neg_loss = F.binary_cross_entropy_with_logits(
                neg_score, torch.zeros_like(neg_score)
            )
            total_loss += pos_loss + neg_loss
            count += 1

        return total_loss / max(count, 1)

    def encode(self, data):
        """Encode all nodes. Returns x_dict."""
        x_dict = {ntype: data[ntype].x for ntype in self.node_types}
        return self.encoder(x_dict, data.edge_index_dict)

    def forward(self, data):
        """Inference forward — returns x_dict."""
        return self.encode(data)
