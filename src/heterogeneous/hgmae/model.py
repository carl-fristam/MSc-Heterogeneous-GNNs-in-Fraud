"""
HGMAE: Heterogeneous Graph Masked AutoEncoder (PyG + HGT backbone).

Adapted from Tan et al. (2023) — replaces the original DGL/HAN backbone
with PyG's HGTConv so no metapath definitions are needed. Works directly
on typed edge relations (onus_transfer, external_transfer).

Pretraining objective:
  1. Attribute restoration — mask ~30% of node features per type, encode
     with HGT, decode, reconstruct original features with SCE loss.
  2. (Optional) Edge reconstruction — drop edges, reconstruct adjacency.

After pretraining, call .encode(data) to get enriched node embeddings,
or .reconstruction_error(data) for per-node anomaly scores.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import HGTConv


def sce_loss(x, y, alpha=2):
    """Scaled cosine error — more robust than MSE for feature reconstruction."""
    x = F.normalize(x, p=2, dim=-1)
    y = F.normalize(y, p=2, dim=-1)
    return (1 - (x * y).sum(dim=-1)).pow(alpha).mean()


class HGMAEModel(nn.Module):
    """
    Args:
        data:              PyG HeteroData (used to infer metadata + feature dims)
        hidden_dim:        embedding dimension throughout encoder/decoder
        num_heads:         attention heads for HGTConv
        num_encoder_layers: depth of the encoder
        feat_mask_rate:    fraction of nodes whose features are masked per forward
        replace_rate:      fraction of masked nodes replaced with random features
                           (instead of mask token) — adds noise, improves robustness
        dropout:           dropout rate
        alpha_l:           SCE loss exponent (2 = standard)
    """

    def __init__(
        self,
        data,
        hidden_dim: int = 256,
        num_heads: int = 4,
        num_encoder_layers: int = 2,
        feat_mask_rate: float = 0.3,
        replace_rate: float = 0.1,
        dropout: float = 0.2,
        alpha_l: int = 2,
    ):
        super().__init__()
        self.feat_mask_rate = feat_mask_rate
        self.replace_rate   = replace_rate
        self.alpha_l        = alpha_l

        metadata         = data.metadata()
        self.node_types  = metadata[0]
        self.feat_dims   = {nt: data[nt].x.shape[1] for nt in self.node_types}

        # Per-type input projections (raw feat_dim → hidden_dim)
        self.input_proj = nn.ModuleDict({
            nt: nn.Linear(self.feat_dims[nt], hidden_dim)
            for nt in self.node_types
        })

        # Learnable mask tokens in raw feature space (one per node type)
        self.mask_tokens = nn.ParameterDict({
            nt: nn.Parameter(torch.zeros(1, self.feat_dims[nt]))
            for nt in self.node_types
        })

        # Encoder: stack of HGTConv layers
        self.encoder_convs = nn.ModuleList()
        self.encoder_norms = nn.ModuleList()
        for _ in range(num_encoder_layers):
            self.encoder_convs.append(HGTConv(hidden_dim, hidden_dim, metadata, heads=num_heads))
            self.encoder_norms.append(nn.LayerNorm(hidden_dim))

        # Bridge between encoder and decoder (linear, no bias — as in original)
        self.enc_to_dec = nn.Linear(hidden_dim, hidden_dim, bias=False)

        # Decoder: single HGTConv layer
        self.decoder_conv = HGTConv(hidden_dim, hidden_dim, metadata, heads=num_heads)
        self.decoder_norm = nn.LayerNorm(hidden_dim)

        # Per-type reconstruction heads (hidden_dim → original feat_dim)
        self.recon_heads = nn.ModuleDict({
            nt: nn.Linear(hidden_dim, self.feat_dims[nt])
            for nt in self.node_types
        })

        self.dropout = nn.Dropout(dropout)

    # ── Masking ───────────────────────────────────────────────────────────────

    def _mask(self, data):
        """
        Randomly mask node features. Returns masked feature dict and
        a dict of masked node indices per type.
        """
        masked_x    = {}
        mask_indices = {}

        for nt in self.node_types:
            x   = data[nt].x
            n   = x.shape[0]
            num_mask = int(self.feat_mask_rate * n)
            perm     = torch.randperm(n, device=x.device)
            mask_idx = perm[:num_mask]

            out = x.clone()
            # Replace with learnable mask token
            out[mask_idx] = self.mask_tokens[nt].expand(num_mask, -1)

            # Replace a fraction with random features (noise injection)
            if self.replace_rate > 0:
                num_replace = int(self.replace_rate * num_mask)
                if num_replace > 0:
                    replace_idx  = mask_idx[:num_replace]
                    random_idx   = torch.randperm(n, device=x.device)[:num_replace]
                    out[replace_idx] = x[random_idx]

            masked_x[nt]     = out
            mask_indices[nt] = mask_idx

        return masked_x, mask_indices

    # ── Encoder ───────────────────────────────────────────────────────────────

    def _encode(self, x_dict, edge_index_dict):
        """Project raw features then run HGT encoder layers."""
        h = {
            nt: self.dropout(torch.relu(self.input_proj[nt](x)))
            for nt, x in x_dict.items()
        }
        for conv, norm in zip(self.encoder_convs, self.encoder_norms):
            h = conv(h, edge_index_dict)
            h = {nt: self.dropout(torch.relu(norm(emb))) for nt, emb in h.items()}
        return h

    # ── Forward (training) ────────────────────────────────────────────────────

    def forward(self, data):
        """
        Mask → encode → re-mask → decode → reconstruct.
        Returns scalar SCE loss averaged across node types.
        """
        masked_x, mask_indices = self._mask(data)

        # Encode masked features
        enc_out = self._encode(masked_x, data.edge_index_dict)

        # Bridge + re-mask: zero out masked node positions before decoding
        # so the decoder must reconstruct them from neighbours alone
        dec_in = {nt: self.enc_to_dec(h) for nt, h in enc_out.items()}
        for nt, idx in mask_indices.items():
            dec_in[nt] = dec_in[nt].clone()
            dec_in[nt][idx] = 0.0

        # Decode
        dec_out = self.decoder_conv(dec_in, data.edge_index_dict)
        dec_out = {nt: self.dropout(torch.relu(self.decoder_norm(h))) for nt, h in dec_out.items()}

        # Reconstruct raw features
        recon = {nt: self.recon_heads[nt](h) for nt, h in dec_out.items()}

        # SCE loss on masked nodes only
        loss = sum(
            sce_loss(recon[nt][idx], data[nt].x[idx], alpha=self.alpha_l)
            for nt, idx in mask_indices.items()
        )
        return loss / len(mask_indices)

    # ── Inference ─────────────────────────────────────────────────────────────

    def encode(self, data):
        """
        Return node embeddings without masking.
        Used after pretraining to extract enriched features.

        Returns:
            dict {node_type: (N, hidden_dim) tensor}
        """
        self.eval()
        with torch.no_grad():
            return self._encode(
                {nt: data[nt].x for nt in self.node_types},
                data.edge_index_dict,
            )

    def reconstruction_error(self, data):
        """
        Per-node reconstruction error (MSE between original and reconstructed features).
        Used for anomaly scoring — higher error = more anomalous account.

        Returns:
            dict {node_type: (N,) tensor of per-node errors}
        """
        self.eval()
        with torch.no_grad():
            enc_out = self._encode(
                {nt: data[nt].x for nt in self.node_types},
                data.edge_index_dict,
            )
            dec_in  = {nt: self.enc_to_dec(h) for nt, h in enc_out.items()}
            dec_out = self.decoder_conv(dec_in, data.edge_index_dict)
            dec_out = {nt: torch.relu(self.decoder_norm(h)) for nt, h in dec_out.items()}
            recon   = {nt: self.recon_heads[nt](h) for nt, h in dec_out.items()}

            return {
                nt: ((data[nt].x - recon[nt]) ** 2).mean(dim=1)
                for nt in self.node_types
            }
