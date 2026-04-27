"""
Visualise the transaction graph structure.

Extracts the largest connected component (capped for readability) and
plots it on a clean white canvas with fraud edges highlighted.

Usage:
    python scripts/plot_graph_clusters.py
    python scripts/plot_graph_clusters.py --sample 0.05 --max-nodes 800
    python scripts/plot_graph_clusters.py --out figures/graph_structure.png
"""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import networkx as nx
import numpy as np

from src.utils.config import load_variant
from src.data.prepare import prepare_data


COLORS = {
    "internal_account": "#2563eb",
    "external_account": "#f59e0b",
    "fraud_edge":       "#dc2626",
    "legit_edge":       "#c4cdd8",
}


def build_nx_graph(df, col_cfg):
    G = nx.DiGraph()
    onus_col = col_cfg["onus_flag"]
    label_col = col_cfg["label"]

    senders = set(df["_sender"].unique())
    onus_receivers = set(df.loc[df[onus_col] == True, "_receiver"].unique())
    internal_ids = senders | onus_receivers

    for _, row in df.iterrows():
        src = row["_sender"]
        dst = row["_receiver"]

        if src not in G:
            G.add_node(src, node_type="internal_account")
        if dst not in G:
            nt = "internal_account" if dst in internal_ids else "external_account"
            G.add_node(dst, node_type=nt)

        G.add_edge(src, dst, fraud=bool(row[label_col]))

    return G


def largest_component(G, max_nodes=600):
    undirected = G.to_undirected()
    components = sorted(nx.connected_components(undirected),
                        key=len, reverse=True)

    for comp in components:
        if len(comp) <= max_nodes:
            return G.subgraph(comp).copy()

    # Largest is too big — take a BFS subgraph from a high-degree node
    comp = components[0]
    sg = G.subgraph(comp)
    start = max(sg.nodes(), key=lambda n: sg.degree(n))
    bfs_nodes = []
    for node in nx.bfs_tree(sg.to_undirected(), start):
        bfs_nodes.append(node)
        if len(bfs_nodes) >= max_nodes:
            break
    return G.subgraph(bfs_nodes).copy()


def plot_graph(sg, out_path):
    fig, ax = plt.subplots(figsize=(14, 11))
    ax.axis("off")
    fig.patch.set_facecolor("white")

    pos = nx.spring_layout(sg, k=0.8 / max(1, np.sqrt(sg.number_of_nodes())),
                           iterations=120, seed=42)

    # Edges: legit first, fraud on top
    legit_edges = [(u, v) for u, v in sg.edges()
                   if not sg.edges[u, v].get("fraud", False)]
    fraud_edges = [(u, v) for u, v in sg.edges()
                   if sg.edges[u, v].get("fraud", False)]

    nx.draw_networkx_edges(sg, pos, edgelist=legit_edges, ax=ax,
                           edge_color=COLORS["legit_edge"],
                           width=0.3, alpha=0.3,
                           arrows=True, arrowsize=3,
                           node_size=12, min_source_margin=1,
                           min_target_margin=1)

    nx.draw_networkx_edges(sg, pos, edgelist=fraud_edges, ax=ax,
                           edge_color=COLORS["fraud_edge"],
                           width=0.9, alpha=0.75,
                           arrows=True, arrowsize=4,
                           node_size=12, min_source_margin=1,
                           min_target_margin=1)

    # Nodes by type
    internal = [n for n in sg.nodes()
                if sg.nodes[n].get("node_type") == "internal_account"]
    external = [n for n in sg.nodes()
                if sg.nodes[n].get("node_type") == "external_account"]

    nx.draw_networkx_nodes(sg, pos, nodelist=internal, ax=ax,
                           node_color=COLORS["internal_account"],
                           node_size=12, alpha=0.7, linewidths=0)

    nx.draw_networkx_nodes(sg, pos, nodelist=external, ax=ax,
                           node_color=COLORS["external_account"],
                           node_size=12, alpha=0.7, linewidths=0)

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
              bbox_to_anchor=(0.5, -0.03), ncol=4,
              fontsize=11, frameon=False,
              handletextpad=0.4, columnspacing=1.5)

    n_int = len(internal)
    n_ext = len(external)
    n_fraud = len(fraud_edges)
    n_legit = len(legit_edges)
    print(f"  Plotted: {n_int} internal + {n_ext} external nodes, "
          f"{n_fraud} fraud + {n_legit} legit edges")

    fig.tight_layout(pad=0.5)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=250, bbox_inches="tight",
                facecolor="white", edgecolor="none")
    plt.close(fig)
    print(f"Saved: {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample", type=float, default=0.05,
                        help="Sample fraction (default 0.05)")
    parser.add_argument("--max-nodes", type=int, default=600,
                        help="Max nodes in the plotted component")
    parser.add_argument("--out", type=str,
                        default="figures/graph_structure.png")
    args = parser.parse_args()

    config = load_variant("v1")
    col_cfg = config["columns"]

    print("Loading data...")
    prep = prepare_data(config, sample=args.sample)

    print("Building networkx graph...")
    G = build_nx_graph(prep.df, col_cfg)
    print(f"  Full graph: {G.number_of_nodes():,} nodes, "
          f"{G.number_of_edges():,} edges")

    print(f"Extracting component (max {args.max_nodes} nodes)...")
    sg = largest_component(G, max_nodes=args.max_nodes)

    print("Plotting...")
    plot_graph(sg, args.out)


if __name__ == "__main__":
    main()
