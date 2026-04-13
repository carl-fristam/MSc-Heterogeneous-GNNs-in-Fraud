"""
HGMAE pretraining loop.

Trains the masked autoencoder on the full graph with no labels.
Saves the best checkpoint by reconstruction loss.
"""

import os
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path

import torch

from src.heterogeneous.hgmae.model import HGMAEModel
from src.utils.config import PROJECT_ROOT


@dataclass
class PretrainConfig:
    epochs:          int   = 300
    lr:              float = 1e-3
    weight_decay:    float = 1e-4
    hidden_dim:      int   = 256
    num_heads:       int   = 4
    num_encoder_layers: int = 2
    feat_mask_rate:  float = 0.3
    replace_rate:    float = 0.1
    dropout:         float = 0.2
    alpha_l:         int   = 2
    patience:        int   = 30
    checkpoint_dir:  str   = "models/hgmae"


def pretrain(data, config: PretrainConfig, device: torch.device) -> HGMAEModel:
    """
    Pretrain HGMAE on the full graph (no labels used).

    Args:
        data:    PyG HeteroData on device
        config:  PretrainConfig
        device:  torch device

    Returns:
        Trained HGMAEModel with best weights loaded.
    """
    model = HGMAEModel(
        data,
        hidden_dim         = config.hidden_dim,
        num_heads          = config.num_heads,
        num_encoder_layers = config.num_encoder_layers,
        feat_mask_rate     = config.feat_mask_rate,
        replace_rate       = config.replace_rate,
        dropout            = config.dropout,
        alpha_l            = config.alpha_l,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"HGMAE parameters: {total_params:,}")

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.lr, weight_decay=config.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config.epochs, eta_min=1e-6
    )

    best_loss  = float("inf")
    best_state = None
    cnt_wait   = 0

    print(f"\nPretraining HGMAE for {config.epochs} epochs ...")
    for epoch in range(1, config.epochs + 1):
        model.train()
        optimizer.zero_grad()

        loss = model(data)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()

        loss_val = loss.item()

        if epoch % 10 == 0 or epoch == 1:
            print(f"  Epoch {epoch:4d}  |  Loss: {loss_val:.6f}")

        if loss_val < best_loss:
            best_loss  = loss_val
            best_state = deepcopy(model.state_dict())
            cnt_wait   = 0
        else:
            cnt_wait += 1
            if cnt_wait >= config.patience:
                print(f"  Early stopping at epoch {epoch} (best loss: {best_loss:.6f})")
                break

    model.load_state_dict(best_state)
    print(f"\nPretraining complete. Best reconstruction loss: {best_loss:.6f}")

    _save_checkpoint(model, config)
    return model


def _save_checkpoint(model: HGMAEModel, config: PretrainConfig):
    save_dir = PROJECT_ROOT / config.checkpoint_dir
    save_dir.mkdir(parents=True, exist_ok=True)
    path = save_dir / "pretrained.pt"
    torch.save(model.state_dict(), path)
    print(f"Checkpoint saved: {path}")


def load_pretrained(data, checkpoint_path: str, config: PretrainConfig, device: torch.device) -> HGMAEModel:
    """Load a saved HGMAE checkpoint."""
    model = HGMAEModel(
        data,
        hidden_dim         = config.hidden_dim,
        num_heads          = config.num_heads,
        num_encoder_layers = config.num_encoder_layers,
        feat_mask_rate     = config.feat_mask_rate,
        replace_rate       = config.replace_rate,
        dropout            = config.dropout,
        alpha_l            = config.alpha_l,
    ).to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()
    print(f"Loaded HGMAE checkpoint: {checkpoint_path}")
    return model
