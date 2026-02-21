"""
scripts/viz_umap.py

Standalone UMAP visualisation from saved embeddings.
Run this after train_hgmae_viz.py has saved results/hgmae_embeds.npy.

Usage:
    python scripts/viz_umap.py
    python scripts/viz_umap.py --embeds results/hgmae_embeds.npy --out results/umap_v2.png
"""

import argparse
import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.hgmae.visualize import plot_umap


def build_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--embeds", type=str, default="results/hgmae_embeds.npy")
    parser.add_argument("--labels", type=str, default="results/hgmae_labels.npy")
    parser.add_argument("--out",    type=str, default="results/hgmae_umap.png")
    parser.add_argument("--max_sample", type=int, default=50_000,
                        help="Max clean nodes to sample for UMAP (all laundering always kept)")
    return parser.parse_args()


if __name__ == "__main__":
    args = build_args()
    embeds    = np.load(args.embeds)
    labels_np = np.load(args.labels)
    print(f"Loaded embeddings: {embeds.shape}, labels: {labels_np.shape}")
    plot_umap(embeds, labels_np, save_path=args.out, max_sample=args.max_sample)
