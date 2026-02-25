"""
scripts/train_hgmae.py

Entry point for HGMAE pre-training on SAML-D.

Usage:
    python scripts/train_hgmae.py
    python scripts/train_hgmae.py --sample_ratio 0.01   # quick smoke test
    python scripts/train_hgmae.py --mae_epochs 100 --hidden_dim 64
"""

import argparse
import types
import sys
import os

# Ensure project root is on path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.hgmae.train import train
from src.utils.config import load_config


def _flatten(d, out=None):
    """Recursively flatten nested config dict into a single-level dict."""
    if out is None:
        out = {}
    for k, v in d.items():
        if isinstance(v, dict):
            _flatten(v, out)
        else:
            out[k] = v
    return out


def build_args():
    parser = argparse.ArgumentParser(description="HGMAE pre-training on SAML-D")
    parser.set_defaults(**_flatten(load_config("hgmae")))

    # Data
    parser.add_argument("--sample_ratio", type=float, default=1.0,
                        help="Fraction of SAML-D to use (0.01 for quick test)")
    parser.add_argument("--use_cache",    action="store_true", default=True)
    parser.add_argument("--graph_version", type=str, default="v1",
                        choices=["v1", "v2"],
                        help="v1=account-level, v2=transaction-level bipartite")

    # Model architecture
    parser.add_argument("--encoder",       type=str,   default="han")
    parser.add_argument("--decoder",       type=str,   default="han")
    parser.add_argument("--hidden_dim",    type=int,   default=64)
    parser.add_argument("--num_layers",    type=int,   default=2)
    parser.add_argument("--num_heads",     type=int,   default=4)
    parser.add_argument("--num_out_heads", type=int,   default=1)
    parser.add_argument("--activation",    type=str,   default="prelu")
    parser.add_argument("--norm",          type=str,   default="layernorm")
    parser.add_argument("--residual",      action="store_true", default=True)

    # Dropout
    parser.add_argument("--feat_drop",      type=float, default=0.2)
    parser.add_argument("--attn_drop",      type=float, default=0.2)
    parser.add_argument("--negative_slope", type=float, default=0.2)

    # Masking
    parser.add_argument("--feat_mask_rate",   type=str,   default="0.3")
    parser.add_argument("--replace_rate",     type=float, default=0.1)
    parser.add_argument("--leave_unchanged",  type=float, default=0.1)
    parser.add_argument("--loss_fn",          type=str,   default="sce")
    parser.add_argument("--alpha_l",          type=float, default=3.0)

    # Metapath edge reconstruction (optional auxiliary task)
    parser.add_argument("--use_mp_edge_recon",         action="store_true", default=False)
    parser.add_argument("--mp_edge_mask_rate",         type=str,   default="0.3")
    parser.add_argument("--mp_edge_recon_loss_weight", type=float, default=1.0)
    parser.add_argument("--mp_edge_alpha_l",           type=float, default=3.0)

    # Metapath2vec auxiliary task (optional)
    parser.add_argument("--use_mp2vec_feat_pred",       action="store_true", default=False)
    parser.add_argument("--mps_embedding_dim",          type=int,   default=64)
    parser.add_argument("--mp2vec_feat_pred_loss_weight", type=float, default=1.0)
    parser.add_argument("--mp2vec_feat_alpha_l",        type=float, default=3.0)
    parser.add_argument("--mp2vec_feat_drop",           type=float, default=0.2)

    # Training
    parser.add_argument("--mae_epochs", type=int,   default=200)
    parser.add_argument("--lr",         type=float, default=1e-3)
    parser.add_argument("--l2_coef",    type=float, default=1e-4)
    parser.add_argument("--patience",   type=int,   default=20)
    parser.add_argument("--scheduler",  action="store_true", default=False)
    parser.add_argument("--scheduler_gamma", type=float, default=0.99)

    # Visualisation
    parser.add_argument("--visualize", action="store_true", default=False,
                        help="Run UMAP on embeddings after training and save plot")

    return parser.parse_args()


if __name__ == "__main__":
    args = build_args()
    print(args)
    train(args)
