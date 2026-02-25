"""
scripts/train_autoencoder.py

Entry point for VGAE anomaly detection on SAML-D.

Usage:
    python scripts/train_autoencoder.py
    python scripts/train_autoencoder.py --sample_ratio 0.01 --epochs 10   # smoke test
    python scripts/train_autoencoder.py --n_days 30 --visualize
    python scripts/train_autoencoder.py --epochs 500 --latent_dim 64
"""

import argparse
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.autoencoder.train import train
from src.utils.config import load_config


def _flatten(d, out=None):
    """Recursively flatten nested config dict."""
    if out is None:
        out = {}
    for k, v in d.items():
        if isinstance(v, dict):
            _flatten(v, out)
        else:
            out[k] = v
    return out


def build_args():
    parser = argparse.ArgumentParser(description="Transaction VGAE on SAML-D")
    parser.set_defaults(**_flatten(load_config("autoencoder")))

    # Data
    parser.add_argument("--sample_ratio", type=float)
    parser.add_argument("--n_days", type=int, default=None)
    parser.add_argument("--use_cache", action="store_true", default=True)

    # Model
    parser.add_argument("--hidden_dim", type=int)
    parser.add_argument("--latent_dim", type=int)
    parser.add_argument("--num_heads", type=int)
    parser.add_argument("--encoder_dropout", type=float)
    parser.add_argument("--decoder_dropout", type=float)
    parser.add_argument("--beta", type=float)

    # Training
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--lr", type=float)
    parser.add_argument("--weight_decay", type=float)
    parser.add_argument("--patience", type=int)
    parser.add_argument("--scheduler", action="store_true")

    # Output
    parser.add_argument("--visualize", action="store_true", default=False)

    return parser.parse_args()


if __name__ == "__main__":
    args = build_args()
    print(args)
    train(args)
