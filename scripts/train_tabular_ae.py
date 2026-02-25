"""
scripts/train_tabular_ae.py

Entry point for tabular autoencoder anomaly detection on SAML-D.

Usage:
    python scripts/train_tabular_ae.py
    python scripts/train_tabular_ae.py --sample_ratio 0.01 --epochs 10   # smoke test
    python scripts/train_tabular_ae.py --visualize
    python scripts/train_tabular_ae.py --epochs 500 --latent_dim 32
"""

import argparse
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.tabular_ae.train import train
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
    parser = argparse.ArgumentParser(description="Tabular AE on SAML-D")
    parser.set_defaults(**_flatten(load_config("tabular_ae")))

    # Data
    parser.add_argument("--sample_ratio", type=float)

    # Model
    parser.add_argument("--h1", type=int)
    parser.add_argument("--h2", type=int)
    parser.add_argument("--latent_dim", type=int)
    parser.add_argument("--dropout", type=float)

    # Training
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--lr", type=float)
    parser.add_argument("--weight_decay", type=float)
    parser.add_argument("--batch_size", type=int)
    parser.add_argument("--patience", type=int)
    parser.add_argument("--scheduler", action="store_true")

    # Output
    parser.add_argument("--visualize", action="store_true", default=False)

    return parser.parse_args()


if __name__ == "__main__":
    args = build_args()
    print(args)
    train(args)
