"""
HGMAE — Heterogeneous Graph Masked Autoencoder.

PyG-native implementation inspired by the DGL reference in src/references/HGMAE/.
Two pretraining objectives:
  1. Attribute masking + restoration (SCE loss on masked node features)
  2. Edge reconstruction (sampled inner-product adjacency reconstruction)

After pretraining, the encoder produces per-type node embeddings that can be
used with a downstream classifier via PretrainTrainer.
"""

from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import HGTConv


def sce_loss(x, y, alpha=3):
    """Spectral contrastive error loss."""
    x = F.normalize(x, p=2, dim=-1)
    y = F.normalize(y, p=2, dim=-1)
    loss = (1 - (x * y).sum(dim=-1)).pow_(alpha)
    return loss.mean()


class HGMAEEncoder(nn.Module):
    """HGTConv-based encoder with per-type input projections."""

    def __init__(self, metadata, feat_dims, hidden_dim=64, num_heads=4,
                 num_layers=2, dropout=0.3):
        super().__init__()
        self.input_proj = nn.ModuleDict()
        for ntype in metadata[0]:
            self.input_proj[ntype] = nn.Linear(feat_dims[ntype], hidden_dim)

        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        for _ in range(num_layers):
            self.convs.append(HGTConv(hidden_dim, hidden_dim, metadata, heads=num_heads))
            self.norms.append(nn.LayerNorm(hidden_dim))

        self.dropout = nn.Dropout(dropout)

    def forward(self, x_dict, edge_index_dict):
        h_dict = {
            ntype: self.dropout(torch.relu(self.input_proj[ntype](x)))
            for ntype, x in x_dict.items()
        }
        for conv, norm in zip(self.convs, self.norms):
            h_dict = conv(h_dict, edge_index_dict)
            h_dict = {
                ntype: self.dropout(torch.relu(norm(h)))
                for ntype, h in h_dict.items()
            }
        return h_dict


class HGMAEDecoder(nn.Module):
    """Per-type MLP decoder projecting back to input dim."""

    def __init__(self, hidden_dim, feat_dims):
        super().__init__()
        self.decoders = nn.ModuleDict()
        for ntype, dim in feat_dims.items():
            self.decoders[ntype] = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, dim),
            )

    def forward(self, h_dict):
        return {ntype: self.decoders[ntype](h) for ntype, h in h_dict.items()}


class HGMAE(nn.Module):
    """
    Heterogeneous Graph Masked Autoencoder.

    Pretraining mode: call pretrain_forward() to get self-supervised loss.
    Inference mode: call forward(data) to get x_dict for downstream classifier.

    Args:
        data: HeteroData object
        hidden_dim: latent dimension
        num_heads: attention heads for HGTConv
        num_layers: number of HGTConv layers
        dropout: dropout rate
        feat_mask_rate: fraction of nodes to mask
        edge_mask_rate: fraction of edges to drop for edge recon
        alpha_l: exponent for SCE loss
        edge_recon_weight: weight for edge reconstruction loss
        num_neg_samples: negative samples per positive for edge recon
    """

    def __init__(self, data, hidden_dim=64, num_heads=4, num_layers=2,
                 dropout=0.3, feat_mask_rate=0.5, edge_mask_rate=0.3,
                 alpha_l=3, edge_recon_weight=1.0, num_neg_samples=5):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.feat_mask_rate = feat_mask_rate
        self.edge_mask_rate = edge_mask_rate
        self.alpha_l = alpha_l
        self.edge_recon_weight = edge_recon_weight
        self.num_neg_samples = num_neg_samples

        metadata = data.metadata()
        self.node_types = metadata[0]
        self.edge_types = metadata[1]

        feat_dims = {ntype: data[ntype].x.size(1) for ntype in self.node_types}
        self.feat_dims = feat_dims

        # Encoder
        self.encoder = HGMAEEncoder(
            metadata, feat_dims, hidden_dim, num_heads, num_layers, dropout
        )

        # Learnable mask tokens (one per node type)
        self.mask_tokens = nn.ParameterDict({
            ntype: nn.Parameter(torch.zeros(1, feat_dims[ntype]))
            for ntype in self.node_types
        })

        # Encoder-to-decoder projection
        self.enc_to_dec = nn.Linear(hidden_dim, hidden_dim, bias=False)

        # Decoder
        self.decoder = HGMAEDecoder(hidden_dim, feat_dims)

        # Attribute restoration loss
        self.attr_loss_fn = partial(sce_loss, alpha=alpha_l)

        self._init_mask_tokens()

    def _init_mask_tokens(self):
        for p in self.mask_tokens.values():
            nn.init.xavier_uniform_(p.unsqueeze(0))

    def _mask_features(self, x_dict):
        """Mask a fraction of node features with learnable mask tokens."""
        masked_x_dict = {}
        mask_nodes_dict = {}

        for ntype, x in x_dict.items():
            num_nodes = x.size(0)
            num_mask = max(1, int(self.feat_mask_rate * num_nodes))
            perm = torch.randperm(num_nodes, device=x.device)
            mask_nodes = perm[:num_mask]

            masked_x = x.clone()
            masked_x[mask_nodes] = 0.0
            masked_x[mask_nodes] += self.mask_tokens[ntype]

            masked_x_dict[ntype] = masked_x
            mask_nodes_dict[ntype] = mask_nodes

        return masked_x_dict, mask_nodes_dict

    def _drop_edges(self, edge_index_dict):
        """Randomly drop edges for edge reconstruction objective."""
        dropped_dict = {}
        kept_dict = {}

        for et, edge_index in edge_index_dict.items():
            num_edges = edge_index.size(1)
            num_keep = max(1, int((1 - self.edge_mask_rate) * num_edges))
            perm = torch.randperm(num_edges, device=edge_index.device)
            kept_dict[et] = edge_index[:, perm[:num_keep]]
            dropped_dict[et] = edge_index[:, perm[num_keep:]]

        return kept_dict, dropped_dict

    def _edge_recon_loss(self, h_dict, data):
        """Sampled edge reconstruction loss via inner product."""
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
            pos_src = src_emb[edge_index[0]]
            pos_dst = dst_emb[edge_index[1]]
            pos_score = (pos_src * pos_dst).sum(dim=-1)

            # Negative sampling
            num_neg = min(self.num_neg_samples * num_edges, dst_emb.size(0) * num_edges)
            num_neg = min(num_neg, num_edges * self.num_neg_samples)
            neg_dst_idx = torch.randint(0, dst_emb.size(0), (num_neg,), device=edge_index.device)
            neg_src_idx = edge_index[0].repeat(self.num_neg_samples)[:num_neg]
            neg_src = src_emb[neg_src_idx]
            neg_dst = dst_emb[neg_dst_idx]
            neg_score = (neg_src * neg_dst).sum(dim=-1)

            # BCE loss
            pos_loss = F.binary_cross_entropy_with_logits(
                pos_score, torch.ones_like(pos_score)
            )
            neg_loss = F.binary_cross_entropy_with_logits(
                neg_score, torch.zeros_like(neg_score)
            )
            total_loss += pos_loss + neg_loss
            count += 1

        return total_loss / max(count, 1)

    def pretrain_forward(self, data):
        """
        Self-supervised forward pass. Returns combined loss.

        Objective 1: Mask node features → encode → decode → SCE loss on masked nodes
        Objective 2: Edge reconstruction via inner product with negative sampling
        """
        x_dict = {ntype: data[ntype].x for ntype in self.node_types}

        # --- Objective 1: Attribute masking + restoration ---
        masked_x_dict, mask_nodes_dict = self._mask_features(x_dict)
        h_dict = self.encoder(masked_x_dict, data.edge_index_dict)

        # Project and decode
        h_dec = {ntype: self.enc_to_dec(h) for ntype, h in h_dict.items()}
        recon_dict = self.decoder(h_dec)

        attr_loss = 0.0
        attr_count = 0
        for ntype in self.node_types:
            mask = mask_nodes_dict[ntype]
            if mask.numel() == 0:
                continue
            attr_loss += self.attr_loss_fn(recon_dict[ntype][mask], x_dict[ntype][mask])
            attr_count += 1
        attr_loss = attr_loss / max(attr_count, 1)

        # --- Objective 2: Edge reconstruction ---
        # Encode with full features for edge recon
        h_full = self.encoder(x_dict, data.edge_index_dict)
        edge_loss = self._edge_recon_loss(h_full, data)

        loss = attr_loss + self.edge_recon_weight * edge_loss
        return loss

    def encode(self, data):
        """Encode with full (unmasked) features. Returns x_dict."""
        x_dict = {ntype: data[ntype].x for ntype in self.node_types}
        return self.encoder(x_dict, data.edge_index_dict)

    def forward(self, data):
        """Inference forward — returns x_dict (node embeddings per type)."""
        return self.encode(data)
