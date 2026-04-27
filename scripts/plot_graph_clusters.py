"""
Visualise ego-network clusters around high-fraud accounts.

Produces a single-canvas figure in the style of Johannessen & Jullum (2023),
with multiple subgraph clusters laid out organically on a white background.

Runs on raw data after graph construction — no model needed.

Usage:
    python scripts/plot_graph_clusters.py
    python scripts/plot_graph_clusters.py --sample 0.5 --top-k 4 --hops 2
    python scripts/plot_graph_clusters.py --out figures/graph_clusters.png
"""

import argparse
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import networkx as nx
import numpy as np
import pandas as pd

from src.utils.config import load_variant
from src.data.prepare import prepare_data


COLORS = {
    "internal_account": "#2563eb",
    "external_account": "#f59e0b",
    "fraud_edge":       "#dc2626",
    "legit_edge":       "#94a3b8",
}


def build_nx_graph(df: pd.DataFrame, col_cfg: dict):
    G = nx.DiGraph()
    onus_col = col_cfg["onus_flag"]
    label_col = col_cfg["label"]

    senders = set(df["_sender"].unique())
    onus_receivers = set(df.loc[df[onus_col] == True, "_receiver"].unique())
    internal_ids = senders | onus_receivers

    for _, row in df.iterrows():
        src = row["_sender"]
        dst = row["_receiver"]

        src_type = "internal_account"
        dst_type = "internal_account" if dst in internal_ids else "external_account"

        if src not in G:
            G.add_node(src, node_type=src_type)
        if dst not in G:
            G.add_node(dst, node_type=dst_type)

        G.add_edge(src, dst, fraud=bool(row[label_col]))

    return G


def find_top_fraud_senders(df: pd.DataFrame, col_cfg: dict, top_k: int = 4):
    fraud_df = df[df[col_cfg["label"]] == True]
    counts = fraud_df.groupby("_sender").size().sort_values(ascending=False)
    return list(counts.head(top_k).index)


def extract_ego_subgraph(G: nx.DiGraph, center: str, hops: int = 2):
    undirected = G.to_undirected()
    ego_nodes = nx.ego_graph(undirected, center, radius=hops).nodes()
    return G.subgraph(ego_nodes).copy()


def plot_clusters(subgraphs: list, centers: list, out_path: str, top_k: int):
    n = len(subgraphs)

    # Grid offsets to spread clusters across the canvas
    if n <= 2:
        offsets = [(0, 0), (6, 0)]
    elif n <= 4:
        offsets = [(-3, 3), (3, 3), (-3, -3), (3, -3)]
    elif n <= 6:
        offsets = [(-4, 3.5), (0, 3.5), (4, 3.5),
                   (-4, -3.5), (0, -3.5), (4, -3.5)]
    else:
        cols = 3
        rows_n = (n + cols - 1) // cols
        offsets = []
        for r in range(rows_n):
            for c in range(cols):
                if len(offsets) < n:
                    offsets.append((c * 7 - 7, -r * 7 + 3.5))

    fig, ax = plt.subplots(figsize=(12, 10))
    ax.set_aspect("equal")
    ax.axis("off")
    fig.patch.set_facecolor("white")

    for idx, (sg, center) in enumerate(zip(subgraphs, centers)):
        ox, oy = offsets[idx]

        pos = nx.spring_layout(sg, k=1.2 / max(1, np.sqrt(sg.number_of_nodes())),
                               iterations=100, seed=42 + idx)

        # Normalise positions to roughly [-1, 1] and apply offset
        if pos:
            coords = np.array(list(pos.values()))
            cx, cy = coords.mean(axis=0)
            scale = max(coords.max() - coords.min(), 1e-6) / 2.0
            pos = {node: ((x - cx) / scale + ox, (y - cy) / scale + oy)
                   for node, (x, y) in pos.items()}

        # Draw legitimate edges first (underneath)
        legit_edges = [(u, v) for u, v in sg.edges()
                       if not sg.edges[u, v].get("fraud", False)]
        fraud_edges = [(u, v) for u, v in sg.edges()
                       if sg.edges[u, v].get("fraud", False)]

        nx.draw_networkx_edges(sg, pos, edgelist=legit_edges, ax=ax,
                               edge_color=COLORS["legit_edge"],
                               width=0.4, alpha=0.35,
                               arrows=True, arrowsize=4,
                               node_size=15, min_source_margin=2,
                               min_target_margin=2)

        nx.draw_networkx_edges(sg, pos, edgelist=fraud_edges, ax=ax,
                               edge_color=COLORS["fraud_edge"],
                               width=0.8, alpha=0.7,
                               arrows=True, arrowsize=5,
                               node_size=15, min_source_margin=2,
                               min_target_margin=2)

        # Separate nodes by type
        internal_nodes = [n for n in sg.nodes()
                          if sg.nodes[n].get("node_type") == "internal_account"]
        external_nodes = [n for n in sg.nodes()
                          if sg.nodes[n].get("node_type") == "external_account"]

        # Center node drawn slightly larger
        internal_sizes = [50 if n == center else 15 for n in internal_nodes]
        external_sizes = [15] * len(external_nodes)

        nx.draw_networkx_nodes(sg, pos, nodelist=internal_nodes, ax=ax,
                               node_color=COLORS["internal_account"],
                               node_size=internal_sizes,
                               alpha=0.7, linewidths=0,)

        nx.draw_networkx_nodes(sg, pos, nodelist=external_nodes, ax=ax,
                               node_color=COLORS["external_account"],
                               node_size=external_sizes,
                               alpha=0.7, linewidths=0,)

    # Legend
    legend_handles = [
        mlines.Line2D([], [], marker="o", color="none",
                      markerfacecolor=COLORS["internal_account"],
                      markersize=8, alpha=0.8, label="Internal Account"),
        mlines.Line2D([], [], marker="o", color="none",
                      markerfacecolor=COLORS["external_account"],
                      markersize=8, alpha=0.8, label="External Account"),
        mlines.Line2D([], [], color=COLORS["fraud_edge"],
                      linewidth=1.5, alpha=0.8, label="Fraud"),
        mlines.Line2D([], [], color=COLORS["legit_edge"],
                      linewidth=1.5, alpha=0.5, label="Legitimate"),
    ]

    ax.legend(handles=legend_handles, loc="lower center",
              bbox_to_anchor=(0.5, -0.04), ncol=4,
              fontsize=11, frameon=False,
              handletextpad=0.4, columnspacing=1.5)

    fig.tight_layout(pad=0.5)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=250, bbox_inches="tight",
                facecolor="white", edgecolor="none")
    plt.close(fig)
    print(f"Saved: {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample", type=float, default=0.1,
                        help="Sample fraction (default 0.1 for speed)")
    parser.add_argument("--top-k", type=int, default=4,
                        help="Number of clusters to plot")
    parser.add_argument("--hops", type=int, default=2,
                        help="Neighbourhood radius")
    parser.add_argument("--out", type=str,
                        default="figures/graph_clusters.png")
    args = parser.parse_args()

    config = load_variant("v1")
    col_cfg = config["columns"]

    print("Loading data...")
    prep = prepare_data(config, sample=args.sample)

    print("Building networkx graph...")
    G = build_nx_graph(prep.df, col_cfg)
    print(f"  {G.number_of_nodes():,} nodes, {G.number_of_edges():,} edges")

    print(f"Finding top {args.top_k} fraud senders...")
    centers = find_top_fraud_senders(prep.df, col_cfg, top_k=args.top_k)

    fraud_df = prep.df[prep.df[col_cfg["label"]] == True]
    fraud_per_center = fraud_df.groupby("_sender").size()

    subgraphs = []
    for c in centers:
        sg = extract_ego_subgraph(G, c, hops=args.hops)
        fc = int(fraud_per_center.get(c, 0))
        subgraphs.append(sg)
        print(f"  {c}: {sg.number_of_nodes()} nodes, "
              f"{sg.number_of_edges()} edges, {fc} fraud txns")

    plot_clusters(subgraphs, centers, args.out, args.top_k)


if __name__ == "__main__":
    main()
