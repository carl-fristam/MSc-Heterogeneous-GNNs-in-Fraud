"""
model.py

Variational Graph Autoencoder (VGAE) for transaction-level anomaly detection.

Architecture (inspired by https://arxiv.org/abs/2410.08121):
    Encoder: 2-layer HeteroConv with GATConv per edge type
             → outputs mu and log_var for transaction nodes
             → reparameterization trick: z = mu + std * eps
    Decoder: 3-layer MLP z → reconstructed transaction features

Training:
    L = MSE(x_hat[genuine], x[genuine]) + beta * KL(q(z|x) || N(0,1))
    Only genuine (non-laundering) transactions contribute to the loss.

Anomaly detection:
    Per-transaction reconstruction error = MSE(x_hat[i], x[i])
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import HeteroConv, GATConv


class HeteroVAEEncoder(nn.Module):
    """
    Two-layer heterogeneous GATConv encoder with VAE output.

    Layer 1: raw features → hidden_dim (both node types)
    Layer 2: hidden_dim → mu [Nt, latent_dim] + log_var [Nt, latent_dim]
    """

    def __init__(
        self,
        account_feat_dim: int,
        txn_feat_dim: int,
        hidden_dim: int,
        latent_dim: int,
        num_heads: int = 4,
        dropout: float = 0.2,
        edge_types: list = None,
    ):
        super().__init__()
        self.dropout = dropout

        if edge_types is None:
            edge_types = [
                ('account', 'sends', 'transaction'),
                ('transaction', 'received_by', 'account'),
            ]

        # Determine input dims per node type
        dim_map = {'account': account_feat_dim, 'transaction': txn_feat_dim}

        # Layer 1: raw → hidden_dim
        # GATConv with bipartite inputs needs (src_dim, dst_dim) tuple
        # heads are concatenated, so output = hidden_dim (we set out_channels = hidden_dim // num_heads)
        head_dim = hidden_dim // num_heads
        conv1_dict = {}
        for src, rel, dst in edge_types:
            conv1_dict[(src, rel, dst)] = GATConv(
                (dim_map[src], dim_map[dst]),
                head_dim,
                heads=num_heads,
                dropout=dropout,
                add_self_loops=False,  # bipartite — self-loops don't apply
            )
        self.conv1 = HeteroConv(conv1_dict, aggr='sum')

        self.norm_txn = nn.LayerNorm(hidden_dim)
        self.norm_acc = nn.LayerNorm(hidden_dim)

        # Layer 2: hidden_dim → mu and log_var (separate convolution heads)
        # Output single-headed for clean latent_dim output
        conv_mu_dict = {}
        conv_logvar_dict = {}
        for src, rel, dst in edge_types:
            conv_mu_dict[(src, rel, dst)] = GATConv(
                (hidden_dim, hidden_dim),
                latent_dim,
                heads=1,
                dropout=dropout,
                add_self_loops=False,
            )
            conv_logvar_dict[(src, rel, dst)] = GATConv(
                (hidden_dim, hidden_dim),
                latent_dim,
                heads=1,
                dropout=dropout,
                add_self_loops=False,
            )
        self.conv_mu = HeteroConv(conv_mu_dict, aggr='sum')
        self.conv_logvar = HeteroConv(conv_logvar_dict, aggr='sum')

    def forward(self, x_dict: dict, edge_index_dict: dict):
        """
        Returns:
            mu:      [Nt, latent_dim]
            log_var: [Nt, latent_dim]
            z:       [Nt, latent_dim] (sampled during training, mu during eval)
        """
        # Layer 1
        h = self.conv1(x_dict, edge_index_dict)
        h['transaction'] = F.elu(self.norm_txn(h['transaction']))
        h['account'] = F.elu(self.norm_acc(h['account']))
        h = {k: F.dropout(v, p=self.dropout, training=self.training) for k, v in h.items()}

        # Layer 2 — separate heads for mu and log_var
        mu_dict = self.conv_mu(h, edge_index_dict)
        logvar_dict = self.conv_logvar(h, edge_index_dict)

        mu = mu_dict['transaction']
        log_var = logvar_dict['transaction']

        # Reparameterization
        if self.training:
            std = torch.exp(0.5 * log_var)
            eps = torch.randn_like(std)
            z = mu + std * eps
        else:
            z = mu

        return mu, log_var, z


class MLPDecoder(nn.Module):
    """MLP decoder: latent z → reconstructed transaction features."""

    def __init__(self, latent_dim: int, hidden_dim: int, output_dim: int, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)


class TransactionVGAE(nn.Module):
    """
    Full VGAE: encoder + decoder with training and inference methods.
    """

    def __init__(
        self,
        account_feat_dim: int,
        txn_feat_dim: int,
        hidden_dim: int = 128,
        latent_dim: int = 32,
        num_heads: int = 4,
        encoder_dropout: float = 0.2,
        decoder_dropout: float = 0.1,
        beta: float = 0.001,
        edge_types: list = None,
    ):
        super().__init__()
        self.beta = beta

        self.encoder = HeteroVAEEncoder(
            account_feat_dim=account_feat_dim,
            txn_feat_dim=txn_feat_dim,
            hidden_dim=hidden_dim,
            latent_dim=latent_dim,
            num_heads=num_heads,
            dropout=encoder_dropout,
            edge_types=edge_types,
        )
        self.decoder = MLPDecoder(
            latent_dim=latent_dim,
            hidden_dim=hidden_dim,
            output_dim=txn_feat_dim,
            dropout=decoder_dropout,
        )

    def forward(self, x_dict: dict, edge_index_dict: dict):
        """
        Returns:
            mu, log_var, z, x_hat
        """
        mu, log_var, z = self.encoder(x_dict, edge_index_dict)
        x_hat = self.decoder(z)
        return mu, log_var, z, x_hat

    def training_loss(
        self,
        x_dict: dict,
        edge_index_dict: dict,
        genuine_mask: torch.Tensor,
    ) -> tuple:
        """
        Compute VAE loss on genuine transactions only.

        Returns:
            (total_loss, recon_loss_item, kl_loss_item)
        """
        mu, log_var, z, x_hat = self.forward(x_dict, edge_index_dict)
        x_t = x_dict['transaction']

        # Reconstruction loss (genuine only)
        recon_loss = F.mse_loss(x_hat[genuine_mask], x_t[genuine_mask])

        # KL divergence (genuine only)
        kl = -0.5 * (1 + log_var[genuine_mask] - mu[genuine_mask].pow(2) - log_var[genuine_mask].exp())
        kl_loss = kl.sum(dim=1).mean()

        total = recon_loss + self.beta * kl_loss
        return total, recon_loss.item(), kl_loss.item()

    @torch.no_grad()
    def anomaly_scores(self, x_dict: dict, edge_index_dict: dict) -> torch.Tensor:
        """Per-transaction reconstruction error (MSE)."""
        self.eval()
        mu, log_var, z, x_hat = self.forward(x_dict, edge_index_dict)
        x_t = x_dict['transaction']
        return ((x_hat - x_t) ** 2).mean(dim=1)

    @torch.no_grad()
    def get_latents(self, x_dict: dict, edge_index_dict: dict) -> torch.Tensor:
        """Extract transaction latent vectors (mu, deterministic)."""
        self.eval()
        mu, _, _ = self.encoder(x_dict, edge_index_dict)
        return mu

    @torch.no_grad()
    def get_attention_weights(self, x_dict: dict, edge_index_dict: dict) -> dict:
        """
        Extract per-edge-type attention weights from Layer 1 GATConv.

        HeteroConv doesn't expose attention weights, so we call each
        GATConv manually with return_attention_weights=True.

        Returns:
            dict mapping edge_type -> (edge_index [2, E], alpha [E])
            where alpha is averaged across attention heads.
        """
        self.eval()
        attention = {}

        for edge_type, conv in self.encoder.conv1.convs.items():
            src_type, _, dst_type = edge_type
            edge_index = edge_index_dict[edge_type]
            x_src = x_dict[src_type]
            x_dst = x_dict[dst_type]

            # GATConv with bipartite input: (x_src, x_dst)
            _, (ei, alpha) = conv(
                (x_src, x_dst),
                edge_index,
                return_attention_weights=True,
            )
            # alpha shape: [E, heads] — average across heads for a single score
            attention[edge_type] = (ei, alpha.mean(dim=1))

        return attention
