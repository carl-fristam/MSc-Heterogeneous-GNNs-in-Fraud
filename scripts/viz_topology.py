"""
Visualize graph topology for thesis figures.

Generates a schematic diagram showing node types, edge types,
and their counts for any graph variant.

Usage:
    python scripts/viz_topology.py --variant v1
    python scripts/viz_topology.py --variant v2
    python scripts/viz_topology.py --variant txn_v1
    python scripts/viz_topology.py --level 2 --mode node
    python scripts/viz_topology.py --level 2 --mode edge
    python scripts/viz_topology.py --all
"""

import argparse
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.utils.config import load_variant, PROJECT_ROOT

NODE_COLORS = {
    "internal_account": "#4C72B0",
    "external_account": "#DD8452",
    "transaction": "#55A868",
    "device": "#C44E52",
    "account": "#8172B3",
}

EDGE_COLORS = [
    "#333333", "#E24A33", "#348ABD", "#988ED5",
    "#777777", "#FBC15E", "#8EBA42", "#FFB5B8",
]


def draw_hetero_schema(config, variant, output_path=None):
    """Draw a schema-level topology (node types as big circles, edge types as arrows)."""
    edges_cfg = config["edges"]["relations"]
    node_types = list(config["nodes"].keys())

    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    ax.set_aspect("equal")
    ax.axis("off")

    # Layout: place nodes in a circle (or custom for 2-node case)
    positions = {}
    n = len(node_types)
    if n == 2:
        positions[node_types[0]] = (3, 5)
        positions[node_types[1]] = (9, 5)
    elif n == 3:
        positions[node_types[0]] = (6, 9)
        positions[node_types[1]] = (2, 3)
        positions[node_types[2]] = (10, 3)
    else:
        for i, nt in enumerate(node_types):
            angle = np.pi / 2 + 2 * np.pi * i / n
            x = 6 + 4 * np.cos(angle)
            y = 5 + 3.5 * np.sin(angle)
            positions[nt] = (x, y)

    radius = 1.0

    # Draw nodes
    for nt in node_types:
        x, y = positions[nt]
        color = NODE_COLORS.get(nt, "#999999")
        circle = plt.Circle((x, y), radius, color=color, alpha=0.85, zorder=3)
        ax.add_patch(circle)
        label = nt.replace("_", "\n")
        ax.text(x, y, label, ha="center", va="center", fontsize=10,
                fontweight="bold", color="white", zorder=4)

    # Group edges by (src, dst) pair to fan out parallel edges
    pair_edges = defaultdict(list)
    for edge in edges_cfg:
        key = (edge["src"], edge["dst"])
        pair_edges[key].append(edge["name"])

    edge_color_idx = 0
    for (src_type, dst_type), names in pair_edges.items():
        sx, sy = positions[src_type]
        dx, dy = positions[dst_type]
        n_edges = len(names)

        if src_type == dst_type:
            # Self-loops: stack vertically above the node
            for i, name in enumerate(names):
                color = EDGE_COLORS[edge_color_idx % len(EDGE_COLORS)]
                edge_color_idx += 1
                offset = 1.5 + i * 0.6
                ax.annotate(
                    "", xy=(sx + 0.7, sy + radius), xytext=(sx - 0.7, sy + radius),
                    arrowprops=dict(arrowstyle="->", color=color, lw=2.0,
                                    connectionstyle=f"arc3,rad=-{0.6 + i * 0.3}"))
                ax.text(sx + 1.5, sy + radius + 0.4 + i * 0.55, name,
                        ha="left", va="center", fontsize=8, style="italic", color=color,
                        fontweight="semibold")
        else:
            dist = np.sqrt((dx - sx)**2 + (dy - sy)**2)
            dx_norm = (dx - sx) / dist
            dy_norm = (dy - sy) / dist
            perp_x = -dy_norm
            perp_y = dx_norm

            # Fan spread: center the group of edges
            spread = 0.35
            offsets = [(i - (n_edges - 1) / 2) * spread for i in range(n_edges)]

            for i, (name, off) in enumerate(zip(names, offsets)):
                color = EDGE_COLORS[edge_color_idx % len(EDGE_COLORS)]
                edge_color_idx += 1

                ox = perp_x * off
                oy = perp_y * off

                start_x = sx + radius * dx_norm + ox
                start_y = sy + radius * dy_norm + oy
                end_x = dx - radius * dx_norm + ox
                end_y = dy - radius * dy_norm + oy

                ax.annotate(
                    "", xy=(end_x, end_y), xytext=(start_x, start_y),
                    arrowprops=dict(arrowstyle="-|>", color=color, lw=1.8,
                                    shrinkA=0, shrinkB=0))

                mid_x = (start_x + end_x) / 2 + perp_x * 0.25
                mid_y = (start_y + end_y) / 2 + perp_y * 0.25
                ax.text(mid_x, mid_y, name, ha="center", va="center",
                        fontsize=7.5, style="italic", color=color, fontweight="semibold",
                        bbox=dict(boxstyle="round,pad=0.15", facecolor="white",
                                  edgecolor="none", alpha=0.85))

    # Set limits with padding
    all_x = [p[0] for p in positions.values()]
    all_y = [p[1] for p in positions.values()]
    ax.set_xlim(min(all_x) - 3, max(all_x) + 3)
    ax.set_ylim(min(all_y) - 2.5, max(all_y) + 3.5)

    ax.set_title(f"Graph Topology — {variant}", fontsize=16, fontweight="bold", pad=20)

    if output_path is None:
        output_path = PROJECT_ROOT / "outputs" / "topology" / f"topology_{variant}.pdf"
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight", dpi=150)
    print(f"Saved: {output_path}")
    plt.close(fig)


def draw_homo_schema(mode, output_path=None):
    """Draw the homogeneous graph schema."""
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    ax.set_aspect("equal")
    ax.axis("off")

    if mode == "node":
        for x, y, label, color in [
            (3, 4, "account", NODE_COLORS["account"]),
            (9, 4, "transaction", NODE_COLORS["transaction"]),
        ]:
            circle = plt.Circle((x, y), 1.0, color=color, alpha=0.85, zorder=3)
            ax.add_patch(circle)
            ax.text(x, y, label, ha="center", va="center", fontsize=11,
                    fontweight="bold", color="white", zorder=4)

        ax.annotate("", xy=(8, 4.15), xytext=(4, 4.15),
                    arrowprops=dict(arrowstyle="-|>", color="#333", lw=2))
        ax.text(6, 4.5, "sends", ha="center", fontsize=9, style="italic", color="#333")
        ax.annotate("", xy=(4, 3.85), xytext=(8, 3.85),
                    arrowprops=dict(arrowstyle="-|>", color="#666", lw=2))
        ax.text(6, 3.4, "receives", ha="center", fontsize=9, style="italic", color="#666")

        ax.set_xlim(0, 12)
        ax.set_ylim(1, 7)
        ax.set_title("Homogeneous Topology — node mode", fontsize=14, fontweight="bold")
    else:
        circle = plt.Circle((6, 4), 1.0, color=NODE_COLORS["account"], alpha=0.85, zorder=3)
        ax.add_patch(circle)
        ax.text(6, 4, "account", ha="center", va="center", fontsize=11,
                fontweight="bold", color="white", zorder=4)

        ax.annotate("", xy=(6.7, 5.0), xytext=(5.3, 5.0),
                    arrowprops=dict(arrowstyle="-|>", color="#333", lw=2,
                                    connectionstyle="arc3,rad=-0.7"))
        ax.text(6, 6.2, "transaction", ha="center", fontsize=9, style="italic", color="#333")

        ax.set_xlim(2, 10)
        ax.set_ylim(1, 7.5)
        ax.set_title("Homogeneous Topology — edge mode", fontsize=14, fontweight="bold")

    if output_path is None:
        output_path = PROJECT_ROOT / "outputs" / "topology" / f"topology_homo_{mode}.pdf"
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight", dpi=150)
    print(f"Saved: {output_path}")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Visualize graph topology")
    parser.add_argument("--variant", type=str, help="Hetero variant (v1, v2, v3, txn_v1)")
    parser.add_argument("--level", type=int, choices=[2], help="Level 2 for homogeneous")
    parser.add_argument("--mode", type=str, default="node", choices=["node", "edge"])
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--all", action="store_true", help="Generate all topology figures")
    args = parser.parse_args()

    if args.all:
        for variant in ["v1", "v2", "v3", "txn_v1"]:
            config = load_variant(variant)
            draw_hetero_schema(config, variant)
        draw_homo_schema("node")
        draw_homo_schema("edge")
        return

    if args.variant:
        config = load_variant(args.variant)
        draw_hetero_schema(config, args.variant, args.output)
    elif args.level == 2:
        draw_homo_schema(args.mode, args.output)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
