"""
Visualize the bipartite graph built by the graph pipeline.

Produces:
  1. Subgraph around a fraud transaction (local neighborhood)
  2. Degree distribution for accounts
  3. Feature distribution comparison (fraud vs legit transactions)
  4. Temporal split overview
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import networkx as nx
import numpy as np
import torch

from src.utils.config import load_config
from src.graph_pipeline import build_graph


def main():
    config = load_config("graph_pipeline")
    data, account_to_id = build_graph(config)

    fig, axes = plt.subplots(2, 2, figsize=(16, 14))
    fig.suptitle("Graph Pipeline Visualization — SAML-D (1% sample)", fontsize=16, fontweight="bold")

    plot_fraud_subgraph(data, axes[0, 0])
    plot_degree_distribution(data, axes[0, 1])
    plot_feature_comparison(data, axes[1, 0])
    plot_temporal_split(data, axes[1, 1])

    plt.tight_layout()
    plt.savefig("outputs/graph_pipeline_viz.png", dpi=150, bbox_inches="tight")
    print("\nSaved to outputs/graph_pipeline_viz.png")
    plt.show()


def plot_fraud_subgraph(data, ax):
    """Plot a 2-hop neighborhood around a fraud transaction."""
    y = data["transaction"].y
    fraud_idx = torch.where(y == 1)[0]

    if len(fraud_idx) == 0:
        ax.text(0.5, 0.5, "No fraud transactions\nin sample", ha="center", va="center", fontsize=14)
        ax.set_title("Fraud Transaction Neighborhood")
        return

    # Pick the first fraud transaction
    target = fraud_idx[0].item()

    # Get edges: account -> sends -> transaction
    sends_ei = data[("account", "sends", "transaction")].edge_index
    recv_ei = data[("transaction", "received_by", "account")].edge_index

    # Build a small networkx graph around this transaction
    G = nx.DiGraph()
    txn_node = f"T{target}"
    G.add_node(txn_node, node_type="fraud_txn")

    # Find sender account for this transaction
    mask = sends_ei[1] == target
    sender_accts = sends_ei[0][mask].tolist()

    # Find receiver account
    mask = recv_ei[0] == target
    receiver_accts = recv_ei[1][mask].tolist()

    # Add sender and receiver
    for s in sender_accts:
        G.add_node(f"A{s}", node_type="account")
        G.add_edge(f"A{s}", txn_node, edge_type="sends")

    for r in receiver_accts:
        G.add_node(f"A{r}", node_type="account")
        G.add_edge(txn_node, f"A{r}", edge_type="received_by")

    # 2nd hop: other transactions from these accounts
    hop2_txns = set()
    for acct in sender_accts + receiver_accts:
        # Transactions sent by this account
        mask = sends_ei[0] == acct
        neighbor_txns = sends_ei[1][mask].tolist()[:5]  # limit for readability
        for t in neighbor_txns:
            if t != target:
                label = "laundering" if y[t] == 1 else "legit"
                tname = f"T{t}"
                G.add_node(tname, node_type=f"{label}_txn")
                G.add_edge(f"A{acct}", tname, edge_type="sends")
                hop2_txns.add(t)

        # Transactions received by this account
        mask = recv_ei[1] == acct
        neighbor_txns = recv_ei[0][mask].tolist()[:5]
        for t in neighbor_txns:
            if t != target:
                label = "laundering" if y[t] == 1 else "legit"
                tname = f"T{t}"
                G.add_node(tname, node_type=f"{label}_txn")
                G.add_edge(tname, f"A{acct}", edge_type="received_by")

    # Layout and draw
    pos = nx.spring_layout(G, seed=42, k=2)

    color_map = {
        "account": "#4ECDC4",
        "fraud_txn": "#FF6B6B",
        "laundering_txn": "#FF6B6B",
        "legit_txn": "#95E1D3",
    }
    node_colors = [color_map.get(G.nodes[n].get("node_type", "account"), "#999") for n in G.nodes]
    node_sizes = [300 if "fraud" in G.nodes[n].get("node_type", "") else 150 for n in G.nodes]

    nx.draw(G, pos, ax=ax, node_color=node_colors, node_size=node_sizes,
            arrows=True, arrowsize=8, edge_color="#CCCCCC", width=0.8,
            font_size=0, with_labels=False)

    legend_elements = [
        mpatches.Patch(facecolor="#4ECDC4", label="Account"),
        mpatches.Patch(facecolor="#FF6B6B", label="Fraud txn"),
        mpatches.Patch(facecolor="#95E1D3", label="Legit txn"),
    ]
    ax.legend(handles=legend_elements, loc="upper left", fontsize=8)
    ax.set_title(f"2-Hop Neighborhood Around Fraud Txn #{target}")


def plot_degree_distribution(data, ax):
    """Plot in-degree and out-degree distributions for accounts."""
    sends_ei = data[("account", "sends", "transaction")].edge_index
    recv_ei = data[("transaction", "received_by", "account")].edge_index
    num_accounts = data["account"].num_nodes

    out_deg = torch.zeros(num_accounts, dtype=torch.long)
    in_deg = torch.zeros(num_accounts, dtype=torch.long)

    # Count degrees
    for idx in sends_ei[0]:
        out_deg[idx] += 1
    for idx in recv_ei[1]:
        in_deg[idx] += 1

    # Plot as log-log histogram
    bins = np.logspace(0, np.log10(max(out_deg.max().item(), in_deg.max().item()) + 1), 50)

    ax.hist(out_deg.numpy(), bins=bins, alpha=0.6, label="Out-degree (sends)", color="#FF6B6B")
    ax.hist(in_deg.numpy(), bins=bins, alpha=0.6, label="In-degree (receives)", color="#4ECDC4")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Degree")
    ax.set_ylabel("Count")
    ax.set_title("Account Degree Distribution (log-log)")
    ax.legend()


def plot_feature_comparison(data, ax):
    """Compare amount distribution for fraud vs legit transactions."""
    x = data["transaction"].x
    y = data["transaction"].y

    # Amount is the first feature (index 0), z-score normalized
    amount_fraud = x[y == 1, 0].numpy()
    amount_legit = x[y == 0, 0].numpy()

    bins = np.linspace(-3, 5, 60)
    ax.hist(amount_legit, bins=bins, alpha=0.6, label=f"Legit (n={len(amount_legit):,})",
            color="#4ECDC4", density=True)
    ax.hist(amount_fraud, bins=bins, alpha=0.7, label=f"Fraud (n={len(amount_fraud):,})",
            color="#FF6B6B", density=True)
    ax.set_xlabel("Amount (z-score normalized)")
    ax.set_ylabel("Density")
    ax.set_title("Transaction Amount: Fraud vs Legit")
    ax.legend()


def plot_temporal_split(data, ax):
    """Visualize the temporal split as a stacked bar showing txn counts over time."""
    train_mask = data["transaction"].train_mask
    val_mask = data["transaction"].val_mask
    test_mask = data["transaction"].test_mask
    y = data["transaction"].y

    counts = {
        "Train": train_mask.sum().item(),
        "Val": val_mask.sum().item(),
        "Test": test_mask.sum().item(),
    }
    fraud_counts = {
        "Train": y[train_mask].sum().item(),
        "Val": y[val_mask].sum().item(),
        "Test": y[test_mask].sum().item(),
    }

    x_pos = range(len(counts))
    labels = list(counts.keys())
    totals = list(counts.values())
    frauds = list(fraud_counts.values())

    bars = ax.bar(x_pos, totals, color=["#4ECDC4", "#FFD93D", "#FF6B6B"], alpha=0.8)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Transaction Count")
    ax.set_title("Temporal Split Distribution")

    # Annotate with fraud counts
    for i, (bar, total, fraud) in enumerate(zip(bars, totals, frauds)):
        pct = 100 * fraud / total if total > 0 else 0
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 200,
                f"{total:,}\n({int(fraud)} fraud, {pct:.2f}%)",
                ha="center", va="bottom", fontsize=9)


def plot_full_network(data, max_txns=8000, hops=5, seed=42):
    """
    Network visualization via BFS expansion from fraud transactions.

    Every edge is a real (account, sends, transaction) or
    (transaction, received_by, account) edge from the PyG edge_index.
    Directed arrows are drawn at edge midpoints showing money flow.

    Grows outward from every fraud transaction for `hops` steps,
    preserving the real graph topology so clusters emerge naturally.
    """
    from collections import deque
    from matplotlib.patches import FancyArrowPatch
    import random
    random.seed(seed)

    y = data["transaction"].y
    sends_ei = data[("account", "sends", "transaction")].edge_index
    recv_ei = data[("transaction", "received_by", "account")].edge_index

    # --- Build SEPARATE directed adjacency lookups ---
    # sends: account -> transaction  (sender account initiated this txn)
    # recv:  transaction -> account  (txn delivered to receiver account)
    sends_lookup = {}  # acct_id -> set of txn_ids they SENT
    recv_lookup = {}   # txn_id -> set of acct_ids that RECEIVED

    for i in range(sends_ei.shape[1]):
        a, t = sends_ei[0, i].item(), sends_ei[1, i].item()
        sends_lookup.setdefault(a, set()).add(t)

    for i in range(recv_ei.shape[1]):
        t, a = recv_ei[0, i].item(), recv_ei[1, i].item()
        recv_lookup.setdefault(t, set()).add(a)

    # Undirected neighbor lookup for BFS traversal
    txn_neighbors = {}  # txn_id -> set of acct_ids (any direction)
    acct_neighbors = {}  # acct_id -> set of txn_ids (any direction)
    for a, txns in sends_lookup.items():
        acct_neighbors.setdefault(a, set()).update(txns)
        for t in txns:
            txn_neighbors.setdefault(t, set()).add(a)
    for t, accts in recv_lookup.items():
        txn_neighbors.setdefault(t, set()).update(accts)
        for a in accts:
            acct_neighbors.setdefault(a, set()).add(t)

    # --- BFS from all fraud transactions ---
    fraud_txns = set(torch.where(y == 1)[0].tolist())
    visited_txns = set()
    visited_accts = set()

    queue = deque()
    for ft in fraud_txns:
        queue.append((ft, "txn", 0))
        visited_txns.add(ft)

    print(f"Starting BFS from {len(fraud_txns)} fraud transactions, max {hops} hops...")

    while queue and len(visited_txns) < max_txns:
        node_id, node_type, depth = queue.popleft()
        if depth >= hops:
            continue

        if node_type == "txn":
            for a in txn_neighbors.get(node_id, []):
                if a not in visited_accts:
                    visited_accts.add(a)
                    queue.append((a, "acct", depth + 1))
        else:  # acct
            neighbors = list(acct_neighbors.get(node_id, []))
            if len(neighbors) > 30:
                neighbors = random.sample(neighbors, 30)
            for t in neighbors:
                if t not in visited_txns:
                    visited_txns.add(t)
                    if len(visited_txns) >= max_txns:
                        break
                    queue.append((t, "txn", depth + 1))

    # --- Collect REAL directed edges (only between visited nodes) ---
    directed_edges = []  # (source_str, target_str)

    for a in visited_accts:
        for t in sends_lookup.get(a, []):
            if t in visited_txns:
                directed_edges.append((f"A{a}", f"T{t}"))  # account sends → txn

    for t in visited_txns:
        for a in recv_lookup.get(t, []):
            if a in visited_accts:
                directed_edges.append((f"T{t}", f"A{a}"))  # txn received_by → account

    print(f"BFS result: {len(visited_accts):,} accounts, {len(visited_txns):,} transactions, {len(directed_edges):,} directed edges")
    print(f"  Fraud in subgraph: {len(visited_txns & fraud_txns)}")

    # --- Build directed networkx graph ---
    G = nx.DiGraph()
    for a in visited_accts:
        G.add_node(f"A{a}", node_type="account")
    for t in visited_txns:
        label = "fraud" if t in fraud_txns else "legit"
        G.add_node(f"T{t}", node_type=label)
    G.add_edges_from(directed_edges)

    acct_nodes = [n for n in G.nodes if G.nodes[n]["node_type"] == "account"]
    legit_nodes = [n for n in G.nodes if G.nodes[n]["node_type"] == "legit"]
    fraud_nodes = [n for n in G.nodes if G.nodes[n]["node_type"] == "fraud"]

    acct_degrees = np.array([G.degree(n) for n in acct_nodes], dtype=float)
    acct_sizes = 10 + acct_degrees * 8

    # --- Layout ---
    print("Computing layout (this may take a minute)...")
    pos = nx.spring_layout(G, seed=seed, k=1.5 / np.sqrt(len(G.nodes)), iterations=100)

    # --- Draw ---
    fig, ax = plt.subplots(1, 1, figsize=(30, 30))
    ax.set_facecolor("white")
    fig.patch.set_facecolor("white")

    # Draw edges as lines
    edge_xs, edge_ys = [], []
    for u, v in G.edges():
        x0, y0 = pos[u]
        x1, y1 = pos[v]
        edge_xs.extend([x0, x1, None])
        edge_ys.extend([y0, y1, None])
    ax.plot(edge_xs, edge_ys, color="#444444", alpha=0.12, linewidth=0.3, zorder=1)

    # Draw midpoint arrows (only if graph is small enough, otherwise too slow)
    num_edges = G.number_of_edges()
    if num_edges <= 5000:
        print(f"Drawing {num_edges} midpoint arrows...")
        for u, v in G.edges():
            x0, y0 = pos[u]
            x1, y1 = pos[v]
            mx, my = (x0 + x1) / 2, (y0 + y1) / 2
            dx, dy = x1 - x0, y1 - y0
            ax.annotate("", xy=(mx + dx * 0.05, my + dy * 0.05),
                         xytext=(mx - dx * 0.05, my - dy * 0.05),
                         arrowprops=dict(arrowstyle="->", color="#666666",
                                         lw=0.6, mutation_scale=8),
                         zorder=2)
    else:
        print(f"Skipping midpoint arrows ({num_edges} edges — too many, would be slow)")

    # Accounts — blue, sized by degree
    nx.draw_networkx_nodes(G, pos, nodelist=acct_nodes, ax=ax,
                           node_color="#3399FF", node_size=acct_sizes, alpha=0.8,
                           linewidths=0, edgecolors="none")
    # Legit transactions — green
    nx.draw_networkx_nodes(G, pos, nodelist=legit_nodes, ax=ax,
                           node_color="#33CC33", node_size=12, alpha=0.7,
                           linewidths=0, edgecolors="none")
    # Fraud transactions — red, big, glowing
    nx.draw_networkx_nodes(G, pos, nodelist=fraud_nodes, ax=ax,
                           node_color="#FF2222", node_size=120, alpha=1.0,
                           linewidths=1.5, edgecolors="#FF6666")

    legend_elements = [
        mpatches.Patch(facecolor="#3399FF", label=f"Accounts ({len(acct_nodes):,})"),
        mpatches.Patch(facecolor="#33CC33", label=f"Legit txns ({len(legit_nodes):,})"),
        mpatches.Patch(facecolor="#FF2222", label=f"Fraud txns ({len(fraud_nodes):,})"),
    ]
    ax.legend(handles=legend_elements, loc="upper left", fontsize=13,
              framealpha=0.9, facecolor="white", labelcolor="black",
              edgecolor="#cccccc")
    ax.set_title(
        f"Transaction Network — {len(G.nodes):,} nodes, {len(G.edges):,} directed edges\n"
        f"BFS from {len(fraud_txns)} fraud txns, {hops} hops  |  "
        f"Account →sends→ Txn →received_by→ Account",
        fontsize=14, fontweight="bold", color="black", pad=20)
    ax.axis("off")

    plt.tight_layout()
    plt.savefig("outputs/full_network_viz.png", dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    print("Saved to outputs/full_network_viz.png")
    plt.show()
    return fig


def plot_ego_networks(data, hops_upper=7, hops_lower=15, seed=42):
    """
    2x2 panel of ego networks around 4 accounts.

    Upper: 2 fraud-connected accounts, expanded by hops_upper (bipartite) hops.
    Lower: 2 high-degree hub accounts, expanded by hops_lower (bipartite) hops.
    The starting node is enlarged.
    """
    from collections import deque
    import random
    random.seed(seed)

    y = data["transaction"].y
    sends_ei = data[("account", "sends", "transaction")].edge_index
    recv_ei = data[("transaction", "received_by", "account")].edge_index

    # Build adjacency
    sends_lookup = {}
    recv_lookup = {}
    for i in range(sends_ei.shape[1]):
        a, t = sends_ei[0, i].item(), sends_ei[1, i].item()
        sends_lookup.setdefault(a, set()).add(t)
    for i in range(recv_ei.shape[1]):
        t, a = recv_ei[0, i].item(), recv_ei[1, i].item()
        recv_lookup.setdefault(t, set()).add(a)

    txn_neighbors = {}
    acct_neighbors = {}
    for a, txns in sends_lookup.items():
        acct_neighbors.setdefault(a, set()).update(txns)
        for t in txns:
            txn_neighbors.setdefault(t, set()).add(a)
    for t, accts in recv_lookup.items():
        txn_neighbors.setdefault(t, set()).update(accts)
        for a in accts:
            acct_neighbors.setdefault(a, set()).add(t)

    fraud_txns = set(torch.where(y == 1)[0].tolist())

    # --- Pick 4 seed accounts ---
    # 2 fraud-connected: accounts with most fraud transaction neighbors
    acct_fraud_count = {}
    for a, txns in acct_neighbors.items():
        acct_fraud_count[a] = len(txns & fraud_txns)

    fraud_accts = sorted(
        [a for a, c in acct_fraud_count.items() if c > 0],
        key=lambda a: acct_fraud_count[a], reverse=True
    )

    # 2 high-degree hub accounts (non-fraud-connected for variety)
    acct_degree = {a: len(txns) for a, txns in acct_neighbors.items()}
    hub_accts = sorted(
        [a for a in acct_degree if acct_fraud_count.get(a, 0) == 0],
        key=lambda a: acct_degree[a], reverse=True
    )

    seeds = []
    if len(fraud_accts) >= 2:
        seeds.extend(fraud_accts[:2])
    elif len(fraud_accts) == 1:
        seeds.append(fraud_accts[0])

    # Fill remaining with hubs
    for h in hub_accts:
        if len(seeds) >= 4:
            break
        if h not in seeds:
            seeds.append(h)

    labels = [
        f"Fraud-connected account (degree {acct_degree.get(s, 0)}, {acct_fraud_count.get(s, 0)} fraud txns)"
        if acct_fraud_count.get(s, 0) > 0
        else f"Hub account (degree {acct_degree.get(s, 0)})"
        for s in seeds
    ]

    print(f"Selected seed accounts: {seeds}")
    for s, l in zip(seeds, labels):
        print(f"  Account {s}: {l}")

    # --- BFS from each seed ---
    def bfs_ego(seed_acct, max_hops):
        visited_txns = set()
        visited_accts = {seed_acct}
        queue = deque([(seed_acct, "acct", 0)])

        while queue:
            node_id, node_type, depth = queue.popleft()
            if depth >= max_hops:
                continue
            if node_type == "acct":
                for t in acct_neighbors.get(node_id, []):
                    if t not in visited_txns:
                        visited_txns.add(t)
                        queue.append((t, "txn", depth + 1))
            else:
                for a in txn_neighbors.get(node_id, []):
                    if a not in visited_accts:
                        visited_accts.add(a)
                        queue.append((a, "acct", depth + 1))

        return visited_accts, visited_txns

    # --- Build subgraphs ---
    fig, axes = plt.subplots(2, 2, figsize=(24, 24))
    fig.patch.set_facecolor("white")

    # Upper row (idx 0,1) = fraud-connected, lower row (idx 2,3) = hubs
    hops_per_panel = [hops_upper, hops_upper, hops_lower, hops_lower]

    for idx, (seed_acct, ax, label) in enumerate(zip(seeds, axes.flat, labels)):
        panel_hops = hops_per_panel[idx]
        v_accts, v_txns = bfs_ego(seed_acct, panel_hops)

        # Collect directed edges
        edges = []
        for a in v_accts:
            for t in sends_lookup.get(a, []):
                if t in v_txns:
                    edges.append((f"A{a}", f"T{t}"))
        for t in v_txns:
            for a in recv_lookup.get(t, []):
                if a in v_accts:
                    edges.append((f"T{t}", f"A{a}"))

        G = nx.DiGraph()
        for a in v_accts:
            G.add_node(f"A{a}", node_type="account")
        for t in v_txns:
            ntype = "fraud" if t in fraud_txns else "legit"
            G.add_node(f"T{t}", node_type=ntype)
        G.add_edges_from(edges)

        acct_nodes = [n for n in G.nodes if G.nodes[n]["node_type"] == "account"]
        legit_nodes = [n for n in G.nodes if G.nodes[n]["node_type"] == "legit"]
        fraud_nodes = [n for n in G.nodes if G.nodes[n]["node_type"] == "fraud"]

        # Degree-based sizing for accounts
        acct_degrees = np.array([G.degree(n) for n in acct_nodes], dtype=float)
        acct_sizes = 20 + acct_degrees * 10
        # Make seed node extra large
        seed_key = f"A{seed_acct}"
        for i, n in enumerate(acct_nodes):
            if n == seed_key:
                acct_sizes[i] = max(acct_sizes[i], 200)

        # Layout
        pos = nx.spring_layout(G, seed=seed + idx, k=1.8 / np.sqrt(max(len(G.nodes), 1)), iterations=80)

        ax.set_facecolor("white")

        # Edges
        edge_xs, edge_ys = [], []
        for u, v in G.edges():
            x0, y0 = pos[u]
            x1, y1 = pos[v]
            edge_xs.extend([x0, x1, None])
            edge_ys.extend([y0, y1, None])
        ax.plot(edge_xs, edge_ys, color="#999999", alpha=0.3, linewidth=0.5, zorder=1)

        # Midpoint arrows (ego networks are small enough)
        if G.number_of_edges() <= 3000:
            for u, v in G.edges():
                x0, y0 = pos[u]
                x1, y1 = pos[v]
                mx, my = (x0 + x1) / 2, (y0 + y1) / 2
                dx, dy = x1 - x0, y1 - y0
                ax.annotate("", xy=(mx + dx * 0.05, my + dy * 0.05),
                            xytext=(mx - dx * 0.05, my - dy * 0.05),
                            arrowprops=dict(arrowstyle="->", color="#888888",
                                            lw=0.5, mutation_scale=7),
                            zorder=2)

        # Nodes
        nx.draw_networkx_nodes(G, pos, nodelist=acct_nodes, ax=ax,
                               node_color="#3399FF", node_size=acct_sizes, alpha=0.85,
                               linewidths=0.5, edgecolors="#2277CC")
        nx.draw_networkx_nodes(G, pos, nodelist=legit_nodes, ax=ax,
                               node_color="#33CC33", node_size=20, alpha=0.75,
                               linewidths=0.3, edgecolors="#22AA22")
        if fraud_nodes:
            nx.draw_networkx_nodes(G, pos, nodelist=fraud_nodes, ax=ax,
                                   node_color="#FF2222", node_size=80, alpha=1.0,
                                   linewidths=1.5, edgecolors="#FF6666")

        n_fraud = len(fraud_nodes)
        ax.set_title(
            f"{label}\n{panel_hops}-hop egonet: {len(G.nodes):,} nodes, {len(G.edges):,} edges"
            f"{f', {n_fraud} fraud txns' if n_fraud > 0 else ''}",
            fontsize=11, fontweight="bold", pad=10)
        ax.axis("off")

    # Shared legend
    legend_elements = [
        mpatches.Patch(facecolor="#3399FF", edgecolor="#2277CC", label="Account"),
        mpatches.Patch(facecolor="#33CC33", edgecolor="#22AA22", label="Legit transaction"),
        mpatches.Patch(facecolor="#FF2222", edgecolor="#FF6666", label="Fraud transaction"),
    ]
    fig.legend(handles=legend_elements, loc="lower center", ncol=3, fontsize=13,
               framealpha=0.9, edgecolor="#cccccc", bbox_to_anchor=(0.5, 0.02))

    fig.suptitle(
        f"Bipartite ego networks for four accounts in the graph.\n"
        f"Upper: {hops_upper}-hop egonets of fraud-connected accounts. "
        f"Lower: {hops_lower}-hop egonets of high-degree hub accounts.\n"
        f"The starting node is enlarged.",
        fontsize=14, fontweight="bold", y=0.99)

    plt.tight_layout(rect=[0, 0.05, 1, 0.96])
    plt.savefig("outputs/ego_networks.png", dpi=150, bbox_inches="tight",
                facecolor="white")
    print("Saved to outputs/ego_networks.png")
    plt.show()
    return fig


if __name__ == "__main__":
    import os
    os.makedirs("outputs", exist_ok=True)
    main()
