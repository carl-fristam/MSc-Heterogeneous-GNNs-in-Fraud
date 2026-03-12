"""Train HGT on the bipartite transaction graph."""

import argparse
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils.compat import apply_pyg_compat_patch

apply_pyg_compat_patch()

from src.utils.config import load_config
from src.utils.device import get_device
from src.graph_pipeline import build_graph
from src.hgt.train import train


def _flatten(d, out=None):
    if out is None:
        out = {}
    for k, v in d.items():
        if isinstance(v, dict):
            _flatten(v, out)
        else:
            out[k] = v
    return out


def build_args():
    parser = argparse.ArgumentParser(description="HGT transaction classification")
    parser.set_defaults(**_flatten(load_config("hgt")))
    parser.add_argument("--hidden_dim", type=int)
    parser.add_argument("--num_heads", type=int)
    parser.add_argument("--num_layers", type=int)
    parser.add_argument("--dropout", type=float)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--lr", type=float)
    parser.add_argument("--weight_decay", type=float)
    parser.add_argument("--patience", type=int)
    parser.add_argument("--scheduler", type=bool)
    parser.add_argument("--sample_ratio", type=float)
    parser.add_argument("--split", type=str, default="temporal", choices=["temporal", "random"],
                        help="Split method: 'temporal' or 'random' (stratified)")
    parser.add_argument("--no_class_weight", action="store_true",
                        help="Disable class weighting in loss (Jullum setup)")
    return parser.parse_args()


def main():
    args = build_args()
    device = get_device()
    print(f"Device: {device}")

    # Override graph pipeline config
    cfg = load_config("graph_pipeline")
    cfg["sample_ratio"] = args.sample_ratio
    cfg["split"]["method"] = args.split

    print(f"Building graph (sample_ratio={args.sample_ratio}, split={args.split})...")
    data, account_to_id = build_graph(cfg)

    print(f"Nodes: {data['account'].num_nodes:,} accounts, {data['transaction'].num_nodes:,} transactions")
    print(f"Features: account={data['account'].x.size(1)}, transaction={data['transaction'].x.size(1)}")
    print(f"Train: {data['transaction'].train_mask.sum():,}, Val: {data['transaction'].val_mask.sum():,}, Test: {data['transaction'].test_mask.sum():,}")

    train(data, device, args)


if __name__ == "__main__":
    main()
