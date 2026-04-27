"""
Visualise the transaction graph structure.

Builds the graph from sampled data and plots a dense subgraph
with fraud edges highlighted.

Usage:
    PYTHONPATH=. python scripts/plot_graph_clusters.py
    PYTHONPATH=. python scripts/plot_graph_clusters.py --sample 0.1 --max-nodes 500
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
    "legit_edge":       "#94a3b8",
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


def dense_subgraph(G, max_nodes=500, min_nodes=100):
    """Pick the largest connected component, BFS-trim if needed."""
    undirected = G.to_undirected()
    components = sorted(nx.connected_components(undirected),
                        key=len, reverse=True)

    # Take the largest component
    comp = components[0]
    sg = G.subgraph(comp).copy()
    print(f"  Largest component: {sg.number_of_nodes()} nodes, "
          f"{sg.number_of_edges()} edges")

    if sg.number_of_nodes() <= max_nodes:
        return sg

    # BFS from highest-degree node to cap size
    start = max(sg.nodes(), key=lambda n: sg.degree(n))
    bfs_nodes = []
    for node in nx.bfs_tree(sg.to_undirected(), start):
        bfs_nodes.append(node)
        if len(bfs_nodes) >= max_nodes:
            break
    return G.subgraph(bfs_nodes).copy()


def plot_graph(sg, out_path):
    fig, ax = plt.subplots(figsize=(14, 12))
    ax.axis("off")
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    # Betweenness centrality for node sizing
    print("  Computing centrality...")
    bc = nx.betweenness_centrality(sg.to_undirected())

    # Use degree as fallback influence on size too
    deg = dict(sg.degree())
    deg_max = max(deg.values()) if deg else 1

    MIN_SIZE, MAX_SIZE = 20, 500
    bc_max = max(bc.values()) if max(bc.values()) > 0 else 1e-9

    def node_size(n):
        bc_norm = (bc.get(n, 0) / bc_max) ** 0.4
        deg_norm = (deg.get(n, 1) / deg_max) ** 0.5
        combined = 0.5 * bc_norm + 0.5 * deg_norm
        return MIN_SIZE + (MAX_SIZE - MIN_SIZE) * combined

    # Tight spring layout
    pos = nx.spring_layout(sg, k=2.0 / max(1, np.sqrt(sg.number_of_nodes())),
                           iterations=300, seed=42)

    # Separate edges
    legit_edges = [(u, v) for u, v in sg.edges()
                   if not sg.edges[u, v].get("fraud", False)]
    fraud_edges = [(u, v) for u, v in sg.edges()
                   if sg.edges[u, v].get("fraud", False)]

    # Draw legit edges
    nx.draw_networkx_edges(sg, pos, edgelist=legit_edges, ax=ax,
                           edge_color=COLORS["legit_edge"],
                           width=0.4, alpha=0.3,
                           arrows=False, node_size=MIN_SIZE)

    # Draw fraud edges on top
    nx.draw_networkx_edges(sg, pos, edgelist=fraud_edges, ax=ax,
                           edge_color=COLORS["fraud_edge"],
                           width=1.2, alpha=0.8,
                           arrows=False, node_size=MIN_SIZE)

    # Nodes by type
    internal = [n for n in sg.nodes()
                if sg.nodes[n].get("node_type") == "internal_account"]
    external = [n for n in sg.nodes()
                if sg.nodes[n].get("node_type") == "external_account"]

    int_sizes = [node_size(n) for n in internal]
    ext_sizes = [node_size(n) for n in external]

    nx.draw_networkx_nodes(sg, pos, nodelist=internal, ax=ax,
                           node_color=COLORS["internal_account"],
                           node_size=int_sizes,
                           alpha=0.8, linewidths=0.4,
                           edgecolors="#1e40af")

    nx.draw_networkx_nodes(sg, pos, nodelist=external, ax=ax,
                           node_color=COLORS["external_account"],
                           node_size=ext_sizes,
                           alpha=0.8, linewidths=0.4,
                           edgecolors="#b45309")

    # Legend
    legend_handles = [
        mlines.Line2D([], [], marker="o", color="none",
                      markerfacecolor=COLORS["internal_account"],
                      markeredgecolor="#1e40af",
                      markersize=10, label="Internal Account"),
        mlines.Line2D([], [], marker="o", color="none",
                      markerfacecolor=COLORS["external_account"],
                      markeredgecolor="#b45309",
                      markersize=10, label="External Account"),
        mlines.Line2D([], [], color=COLORS["fraud_edge"],
                      linewidth=2, label="Fraud"),
        mlines.Line2D([], [], color=COLORS["legit_edge"],
                      linewidth=1.5, alpha=0.5, label="Legitimate"),
    ]

    ax.legend(handles=legend_handles, loc="lower center",
              bbox_to_anchor=(0.5, -0.02), ncol=4,
              fontsize=12, frameon=False,
              handletextpad=0.5, columnspacing=2.0)

    n_int = len(internal)
    n_ext = len(external)
    n_fraud = len(fraud_edges)
    n_legit = len(legit_edges)
    print(f"  Plotted: {n_int} internal + {n_ext} external nodes, "
          f"{n_fraud} fraud + {n_legit} legit edges")

    # Pad tightly around actual content
    ax.margins(0.05)
    fig.tight_layout(pad=1.0)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=250, bbox_inches="tight",
                facecolor="white", edgecolor="none")
    plt.close(fig)
    print(f"Saved: {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample", type=float, default=0.1,
                        help="Sample fraction (default 0.1)")
    parser.add_argument("--max-nodes", type=int, default=500,
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

    print(f"Extracting dense subgraph (max {args.max_nodes} nodes)...")
    sg = dense_subgraph(G, max_nodes=args.max_nodes)

    print("Plotting...")
    plot_graph(sg, args.out)


if __name__ == "__main__":
    main()
