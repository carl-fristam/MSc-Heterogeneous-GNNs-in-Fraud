"""
Attention case study for HeteroGAT.

Loads a trained HeteroGAT model, identifies the highest-scored fraud
transactions in the test set, and visualises what the model "saw":
which neighbouring accounts received the highest attention weights
when scoring those edges.

Usage:
    PYTHONPATH=. python scripts/attention_case_study.py results/defaults/hetero_gat/<run_dir>
    PYTHONPATH=. python scripts/attention_case_study.py results/defaults/hetero_gat/<run_dir> --top-k 5
"""

import argparse
import json
import pickle
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import networkx as nx
import numpy as np
import torch

from src.utils.config import load_variant, PROJECT_ROOT
from src.data.prepare import prepare_data
from src.graph_builder.assembler import build_graph
from src.heterogeneous.hetero_gat.model import HeteroGAT


def load_model_and_data(run_dir: Path, config, prep):
    """Load graph, rebuild model, load best weights."""
    device = torch.device("cpu")
    graph_result = build_graph(config, prep)
    data = graph_result["data"].to(device)
    node_maps = graph_result["node_maps"]

    # Read hyperparams from metrics.json
    metrics_path = run_dir / "metrics.json"
    with open(metrics_path) as f:
        saved = json.load(f)
    hp = saved.get("meta", {}).get("hyperparams", {})

    model = HeteroGAT(
        data,
        hidden_dim=hp.get("hidden_dim", 64),
        num_heads=hp.get("num_heads", 4),
        num_layers=hp.get("num_layers", 2),
        dropout=hp.get("dropout", 0.3),
        task="edge",
        target_node_type="internal_account",
    )

    # Find the state dict
    state_path = run_dir / "model_state.pt"
    if not state_path.exists():
        print("No model_state.pt found — running with current weights from a fresh forward pass.")
        print("For proper results, the model needs to be retrained with state saving enabled.")
    else:
        model.load_state_dict(torch.load(state_path, map_location=device))

    model.eval()
    return model, data, node_maps


def find_top_fraud_edges(model, data, edge_type_slices, test_mask, y, top_k=5):
    """Find the test fraud edges with the highest model scores."""
    with torch.no_grad():
        x_dict = model(data)

        all_logits = []
        for et, (start, end) in edge_type_slices.items():
            src_type, _, dst_type = et
            edge_index = data[et].edge_index
            src_emb = x_dict[src_type][edge_index[0]]
            dst_emb = x_dict[dst_type][edge_index[1]]
            parts = [src_emb, dst_emb]
            if hasattr(data[et], "edge_attr") and data[et].edge_attr is not None:
                parts.append(data[et].edge_attr)
            edge_emb = torch.cat(parts, dim=1)
            all_logits.append(model.classifier(edge_emb).squeeze(-1))
        logits = torch.cat(all_logits)
        probs = torch.sigmoid(logits).cpu().detach().numpy()

    test_idx = test_mask.cpu().detach().numpy().astype(bool)
    labels = y.cpu().detach().numpy()

    # Fraud edges in test set
    fraud_test = test_idx & (labels == 1)
    fraud_indices = np.where(fraud_test)[0]
    fraud_probs = probs[fraud_indices]

    # Top-k by model score
    top_k_local = np.argsort(fraud_probs)[-top_k:][::-1]
    top_indices = fraud_indices[top_k_local]
    top_scores = fraud_probs[top_k_local]

    return top_indices, top_scores


def get_edge_info(global_idx, edge_type_slices, data):
    """Map global edge index back to edge type and local index."""
    for et, (start, end) in edge_type_slices.items():
        if start <= global_idx < end:
            local_idx = global_idx - start
            edge_index = data[et].edge_index
            src_node = edge_index[0, local_idx].item()
            dst_node = edge_index[1, local_idx].item()
            src_type, rel, dst_type = et
            return {
                "edge_type": et,
                "local_idx": local_idx,
                "src_node": src_node,
                "dst_node": dst_node,
                "src_type": src_type,
                "dst_type": dst_type,
                "rel": rel,
            }
    return None


def get_neighbours_with_attention(node_idx, node_type, data, attention_weights, layer=1):
    """Get all neighbours of a node and their attention weights from the last layer."""
    neighbours = []
    layer_attn = attention_weights.get(layer, attention_weights.get(0, {}))

    for et, alpha in layer_attn.items():
        src_type, _, dst_type = et

        if dst_type == node_type:
            edge_index = data[et].edge_index
            dst_mask = edge_index[1] == node_idx
            if dst_mask.any():
                src_nodes = edge_index[0, dst_mask].cpu().numpy()
                attn_vals = alpha[dst_mask].cpu().numpy()
                if attn_vals.ndim > 1:
                    attn_vals = attn_vals.mean(axis=1)
                for s, a in zip(src_nodes, attn_vals):
                    neighbours.append({
                        "node": int(s),
                        "node_type": src_type,
                        "attention": float(a),
                        "edge_type": et,
                    })

        if src_type == node_type:
            edge_index = data[et].edge_index
            src_mask = edge_index[0] == node_idx
            if src_mask.any():
                dst_nodes = edge_index[1, src_mask].cpu().numpy()
                attn_vals = alpha[src_mask].cpu().numpy()
                if attn_vals.ndim > 1:
                    attn_vals = attn_vals.mean(axis=1)
                for d, a in zip(dst_nodes, attn_vals):
                    neighbours.append({
                        "node": int(d),
                        "node_type": dst_type,
                        "attention": float(a),
                        "edge_type": et,
                    })

    neighbours.sort(key=lambda x: x["attention"], reverse=True)
    return neighbours


def plot_case(edge_info, src_neighbours, dst_neighbours, score, case_idx, out_dir):
    """Plot a single fraud edge with its neighbourhood attention."""
    G = nx.DiGraph()

    src_id = f"{edge_info['src_type']}\n#{edge_info['src_node']}"
    dst_id = f"{edge_info['dst_type']}\n#{edge_info['dst_node']}"

    G.add_node(src_id, node_type=edge_info["src_type"], role="center")
    G.add_node(dst_id, node_type=edge_info["dst_type"], role="center")
    G.add_edge(src_id, dst_id, attention=1.0, is_target=True)

    # Add top neighbours of sender
    for i, nb in enumerate(src_neighbours[:8]):
        nb_id = f"{nb['node_type']}\n#{nb['node']}"
        if nb_id not in G:
            G.add_node(nb_id, node_type=nb["node_type"], role="neighbour")
        G.add_edge(nb_id, src_id, attention=nb["attention"], is_target=False)

    # Add top neighbours of receiver
    for i, nb in enumerate(dst_neighbours[:8]):
        nb_id = f"{nb['node_type']}\n#{nb['node']}"
        if nb_id not in G:
            G.add_node(nb_id, node_type=nb["node_type"], role="neighbour")
        G.add_edge(nb_id, dst_id, attention=nb["attention"], is_target=False)

    fig, ax = plt.subplots(figsize=(12, 9))
    ax.axis("off")
    fig.patch.set_facecolor("white")

    pos = nx.spring_layout(G, k=2.5, iterations=100, seed=42)

    # Node colors and sizes
    node_colors = []
    node_sizes = []
    for n in G.nodes():
        nt = G.nodes[n]["node_type"]
        role = G.nodes[n]["role"]
        if nt == "internal_account":
            node_colors.append("#2563eb")
        else:
            node_colors.append("#f59e0b")
        node_sizes.append(600 if role == "center" else 200)

    # Edge colors and widths by attention
    edge_colors = []
    edge_widths = []
    for u, v in G.edges():
        edata = G.edges[u, v]
        if edata["is_target"]:
            edge_colors.append("#dc2626")
            edge_widths.append(3.0)
        else:
            attn = edata["attention"]
            edge_colors.append("#64748b")
            edge_widths.append(0.5 + attn * 4.0)

    nx.draw_networkx_edges(G, pos, ax=ax,
                           edge_color=edge_colors,
                           width=edge_widths,
                           alpha=0.7,
                           arrows=True, arrowsize=15,
                           node_size=200,
                           connectionstyle="arc3,rad=0.1")

    nx.draw_networkx_nodes(G, pos, ax=ax,
                           node_color=node_colors,
                           node_size=node_sizes,
                           alpha=0.85,
                           edgecolors="white",
                           linewidths=1.5)

    # Attention labels on edges
    edge_labels = {}
    for u, v in G.edges():
        edata = G.edges[u, v]
        if not edata["is_target"]:
            edge_labels[(u, v)] = f"{edata['attention']:.3f}"
    nx.draw_networkx_edge_labels(G, pos, edge_labels, ax=ax,
                                 font_size=7, font_color="#475569")

    legend_handles = [
        mpatches.Patch(color="#2563eb", label="Internal Account"),
        mpatches.Patch(color="#f59e0b", label="External Account"),
        plt.Line2D([], [], color="#dc2626", linewidth=3, label="Fraud edge (target)"),
        plt.Line2D([], [], color="#64748b", linewidth=2, label="Neighbour (width = attention)"),
    ]
    ax.legend(handles=legend_handles, loc="lower center",
              bbox_to_anchor=(0.5, -0.04), ncol=4,
              fontsize=10, frameon=False)

    ax.set_title(
        f"Case {case_idx + 1}: {edge_info['rel']}  |  "
        f"Fraud score: {score:.4f}\n"
        f"Sender: {edge_info['src_type']} #{edge_info['src_node']}  →  "
        f"Receiver: {edge_info['dst_type']} #{edge_info['dst_node']}",
        fontsize=12, pad=15,
    )

    fig.tight_layout()
    out_path = out_dir / f"attention_case_{case_idx + 1}.png"
    fig.savefig(out_path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  Saved: {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=str,
                        help="Path to HeteroGAT run directory")
    parser.add_argument("--top-k", type=int, default=3,
                        help="Number of fraud cases to visualise")
    parser.add_argument("--sample", type=float, default=0.5)
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    config = load_variant("v1")
    prep = prepare_data(config, sample=args.sample)

    print("Loading model and data...")
    model, data, node_maps = load_model_and_data(run_dir, config, prep)

    # Build edge type slices (same logic as trainer)
    edge_type_slices = {}
    ys, trains, vals, tests = [], [], [], []
    offset = 0
    for et in data.edge_types:
        if hasattr(data[et], "y") and data[et].y is not None:
            n = data[et].y.shape[0]
            ys.append(data[et].y)
            trains.append(data[et].train_mask)
            vals.append(data[et].val_mask)
            tests.append(data[et].test_mask)
            edge_type_slices[et] = (offset, offset + n)
            offset += n

    y = torch.cat(ys)
    test_mask = torch.cat(tests)

    print(f"Finding top {args.top_k} fraud edges by model score...")
    top_indices, top_scores = find_top_fraud_edges(
        model, data, edge_type_slices, test_mask, y, top_k=args.top_k
    )

    print("Extracting attention weights...")
    attention_weights = model.extract_attention(data)

    out_dir = run_dir / "attention_cases"
    out_dir.mkdir(exist_ok=True)

    for i, (idx, score) in enumerate(zip(top_indices, top_scores)):
        edge_info = get_edge_info(idx, edge_type_slices, data)
        if edge_info is None:
            continue

        print(f"\nCase {i + 1}: {edge_info['rel']}  "
              f"src={edge_info['src_type']}#{edge_info['src_node']} → "
              f"dst={edge_info['dst_type']}#{edge_info['dst_node']}  "
              f"score={score:.4f}")

        src_nb = get_neighbours_with_attention(
            edge_info["src_node"], edge_info["src_type"],
            data, attention_weights
        )
        dst_nb = get_neighbours_with_attention(
            edge_info["dst_node"], edge_info["dst_type"],
            data, attention_weights
        )

        print(f"  Sender: {len(src_nb)} neighbours, "
              f"top attn: {src_nb[0]['attention']:.4f}" if src_nb else "  Sender: no neighbours")
        print(f"  Receiver: {len(dst_nb)} neighbours, "
              f"top attn: {dst_nb[0]['attention']:.4f}" if dst_nb else "  Receiver: no neighbours")

        plot_case(edge_info, src_nb, dst_nb, score, i, out_dir)

    print(f"\nDone — {args.top_k} cases saved to {out_dir}/")


if __name__ == "__main__":
    main()
