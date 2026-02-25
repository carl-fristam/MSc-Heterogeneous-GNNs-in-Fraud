"""
model.py

Vanilla (non-variational) tabular autoencoder for anomaly detection.

Architecture follows the reference paper:
    3-layer encoder (input → h1 → h2 → latent)
    3-layer decoder (latent → h2 → h1 → input)
    ReLU activations, trained with MSE loss on genuine transactions only.

Anomaly score = per-sample reconstruction MSE.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class TabularAutoencoder(nn.Module):
    """
    Symmetric 3-layer autoencoder for tabular transaction features.

    Architecture:
        Encoder: input_dim → h1 → h2 → latent_dim
        Decoder: latent_dim → h2 → h1 → input_dim
    """

    def __init__(
        self,
        input_dim: int,
        h1: int = 64,
        h2: int = 32,
        latent_dim: int = 16,
        dropout: float = 0.2,
    ):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, h1),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(h1, h2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(h2, latent_dim),
            nn.ReLU(),
        )

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, h2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(h2, h1),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(h1, input_dim),
            # No activation — reconstructed values should match MinMax-scaled [0, 1] range
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Returns reconstructed input."""
        z = self.encoder(x)
        x_hat = self.decoder(z)
        return x_hat

    def training_loss(self, x: torch.Tensor) -> torch.Tensor:
        """MSE reconstruction loss."""
        x_hat = self.forward(x)
        return F.mse_loss(x_hat, x)

    @torch.no_grad()
    def anomaly_scores(self, x: torch.Tensor) -> torch.Tensor:
        """Per-sample reconstruction MSE."""
        self.eval()
        x_hat = self.forward(x)
        return ((x_hat - x) ** 2).mean(dim=1)

    @torch.no_grad()
    def get_latents(self, x: torch.Tensor) -> torch.Tensor:
        """Extract bottleneck representations."""
        self.eval()
        return self.encoder(x)
