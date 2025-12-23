"""
Graph statistics computation utilities.
"""

import networkx as nx
import numpy as np
from typing import Dict, Any


def compute_basic_stats(G: nx.DiGraph) -> Dict[str, Any]:
    """
    Compute basic graph structure statistics.
    
    Args:
        G: NetworkX graph
        
    Returns:
        Dictionary of statistics
    """
    stats = {
        'num_nodes': G.number_of_nodes(),
        'num_edges': G.number_of_edges(),
        'density': nx.density(G)
    }
    return stats


def compute_degree_stats(G: nx.DiGraph) -> Dict[str, Any]:
    """
    Compute degree distribution statistics.
    
    Args:
        G: NetworkX graph
        
    Returns:
        Dictionary of degree statistics
    """
    in_degrees = [d for n, d in G.in_degree()]
    out_degrees = [d for n, d in G.out_degree()]
    
    stats = {
        'in_degree': {
            'mean': np.mean(in_degrees),
            'max': max(in_degrees),
            'min': min(in_degrees),
            'std': np.std(in_degrees)
        },
        'out_degree': {
            'mean': np.mean(out_degrees),
            'max': max(out_degrees),
            'min': min(out_degrees),
            'std': np.std(out_degrees)
        }
    }
    return stats


def compute_connectivity_stats(G: nx.DiGraph) -> Dict[str, Any]:
    """
    Compute connectivity statistics.
    
    Args:
        G: NetworkX graph
        
    Returns:
        Dictionary of connectivity statistics
    """
    weakly_connected = nx.number_weakly_connected_components(G)
    strongly_connected = nx.number_strongly_connected_components(G)
    largest_wcc = max(nx.weakly_connected_components(G), key=len)
    
    stats = {
        'weakly_connected_components': weakly_connected,
        'strongly_connected_components': strongly_connected,
        'largest_component_size': len(largest_wcc),
        'largest_component_ratio': len(largest_wcc) / G.number_of_nodes()
    }
    return stats


def compute_transaction_stats(G: nx.DiGraph) -> Dict[str, Any]:
    """
    Compute transaction-specific statistics.
    
    Args:
        G: NetworkX graph
        
    Returns:
        Dictionary of transaction statistics
    """
    weights = [G[u][v]['weight'] for u, v in G.edges()]
    amounts = [G[u][v]['total_amount'] for u, v in G.edges()]
    
    stats = {
        'transactions_per_edge': {
            'mean': np.mean(weights),
            'max': max(weights),
            'min': min(weights),
            'std': np.std(weights)
        },
        'amount_per_edge': {
            'mean': np.mean(amounts),
            'max': max(amounts),
            'min': min(amounts),
            'std': np.std(amounts)
        }
    }
    return stats


def compute_laundering_stats(G: nx.DiGraph) -> Dict[str, Any]:
    """
    Compute money laundering statistics.
    
    Args:
        G: NetworkX graph
        
    Returns:
        Dictionary of laundering statistics
    """
    laundering_ratios = [G[u][v]['laundering_ratio'] for u, v in G.edges()]
    laundering_edges = sum(1 for r in laundering_ratios if r > 0)
    
    stats = {
        'edges_with_laundering': laundering_edges,
        'edges_with_laundering_ratio': laundering_edges / G.number_of_edges(),
        'avg_laundering_ratio': np.mean(laundering_ratios),
        'max_laundering_ratio': max(laundering_ratios)
    }
    return stats


def compute_all_statistics(G: nx.DiGraph) -> Dict[str, Any]:
    """
    Compute all graph statistics.
    
    Args:
        G: NetworkX graph
        
    Returns:
        Dictionary containing all statistics
    """
    return {
        'basic': compute_basic_stats(G),
        'degree': compute_degree_stats(G),
        'connectivity': compute_connectivity_stats(G),
        'transactions': compute_transaction_stats(G),
        'laundering': compute_laundering_stats(G)
    }


def print_statistics(stats: Dict[str, Any]) -> None:
    """
    Print formatted statistics to console.
    
    Args:
        stats: Dictionary of statistics from compute_all_statistics
    """
    print("\n" + "="*60)
    print("GRAPH STATISTICS")
    print("="*60)
    
    # Basic stats
    print(f"\nBasic Structure:")
    print(f"  Nodes (Accounts): {stats['basic']['num_nodes']:,}")
    print(f"  Edges (Transaction pairs): {stats['basic']['num_edges']:,}")
    print(f"  Density: {stats['basic']['density']:.6f}")
    
    # Degree statistics
    print(f"\nDegree Statistics:")
    print(f"  In-degree  - Mean: {stats['degree']['in_degree']['mean']:.2f}, "
          f"Max: {stats['degree']['in_degree']['max']}, "
          f"Min: {stats['degree']['in_degree']['min']}")
    print(f"  Out-degree - Mean: {stats['degree']['out_degree']['mean']:.2f}, "
          f"Max: {stats['degree']['out_degree']['max']}, "
          f"Min: {stats['degree']['out_degree']['min']}")
    
    # Connectivity
    print(f"\nConnectivity:")
    print(f"  Weakly connected components: {stats['connectivity']['weakly_connected_components']}")
    print(f"  Strongly connected components: {stats['connectivity']['strongly_connected_components']}")
    print(f"  Largest component size: {stats['connectivity']['largest_component_size']:,} nodes "
          f"({stats['connectivity']['largest_component_ratio']*100:.1f}%)")
    
    # Transactions
    print(f"\nTransaction Statistics:")
    print(f"  Transactions per edge - Mean: {stats['transactions']['transactions_per_edge']['mean']:.2f}, "
          f"Max: {stats['transactions']['transactions_per_edge']['max']}")
    print(f"  Total amount per edge - Mean: ${stats['transactions']['amount_per_edge']['mean']:,.2f}, "
          f"Max: ${stats['transactions']['amount_per_edge']['max']:,.2f}")
    
    # Laundering
    print(f"\nMoney Laundering:")
    print(f"  Edges with laundering: {stats['laundering']['edges_with_laundering']:,} "
          f"({stats['laundering']['edges_with_laundering_ratio']*100:.1f}%)")
    print(f"  Average laundering ratio: {stats['laundering']['avg_laundering_ratio']:.4f}")
    
    print("="*60)
