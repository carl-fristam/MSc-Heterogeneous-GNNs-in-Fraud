"""
Graph visualization utilities.
"""

import networkx as nx
import matplotlib.pyplot as plt
from pathlib import Path


def visualize_subgraph(G: nx.DiGraph, output_path: str = 'outputs/subgraph.png', num_nodes: int = 50) -> None:
    """
    Visualize a small subgraph from the transaction network.
    
    Args:
        G: NetworkX graph
        output_path: Path to save visualization
        num_nodes: Number of nodes to include in visualization
    """
    print(f"\nCreating visualization of {num_nodes} nodes...")
    
    # Get subgraph from largest component
    largest_wcc = max(nx.weakly_connected_components(G), key=len)
    subgraph_nodes = list(largest_wcc)[:num_nodes]
    subgraph = G.subgraph(subgraph_nodes)
    
    # Create figure
    plt.figure(figsize=(15, 10))
    
    # Layout
    pos = nx.spring_layout(subgraph, k=2, iterations=50, seed=42)
    
    # Color nodes by laundering activity
    node_colors = _get_node_colors(subgraph)
    
    # Draw graph
    nx.draw_networkx_nodes(subgraph, pos, node_color=node_colors, 
                          node_size=300, alpha=0.7)
    nx.draw_networkx_edges(subgraph, pos, alpha=0.3, arrows=True, 
                          arrowsize=10, edge_color='gray')
    nx.draw_networkx_labels(subgraph, pos, font_size=8)
    
    plt.title(f'Transaction Network Subgraph ({num_nodes} accounts)\nRed = Has laundering transactions', 
              fontsize=14)
    plt.axis('off')
    plt.tight_layout()
    
    # Save
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved visualization to {output_path}")
    plt.close()


def _get_node_colors(subgraph: nx.DiGraph) -> list:
    """
    Determine node colors based on laundering activity.
    
    Args:
        subgraph: NetworkX subgraph
        
    Returns:
        List of colors for each node
    """
    node_colors = []
    for node in subgraph.nodes():
        # Check if node has any outgoing laundering transactions
        has_laundering = any(
            subgraph[node][neighbor]['laundering_count'] > 0 
            for neighbor in subgraph.successors(node)
        ) if subgraph.out_degree(node) > 0 else False
        
        node_colors.append('red' if has_laundering else 'lightblue')
    
    return node_colors


def create_degree_distribution_plot(G: nx.DiGraph, output_path: str = 'outputs/degree_distribution.png') -> None:
    """
    Create degree distribution histogram.
    
    Args:
        G: NetworkX graph
        output_path: Path to save plot
    """
    print("\nCreating degree distribution plot...")
    
    in_degrees = [d for n, d in G.in_degree()]
    out_degrees = [d for n, d in G.out_degree()]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # In-degree distribution
    ax1.hist(in_degrees, bins=50, alpha=0.7, color='blue', edgecolor='black')
    ax1.set_xlabel('In-Degree')
    ax1.set_ylabel('Frequency')
    ax1.set_title('In-Degree Distribution')
    ax1.set_yscale('log')
    
    # Out-degree distribution
    ax2.hist(out_degrees, bins=50, alpha=0.7, color='green', edgecolor='black')
    ax2.set_xlabel('Out-Degree')
    ax2.set_ylabel('Frequency')
    ax2.set_title('Out-Degree Distribution')
    ax2.set_yscale('log')
    
    plt.tight_layout()
    
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved degree distribution to {output_path}")
    plt.close()
