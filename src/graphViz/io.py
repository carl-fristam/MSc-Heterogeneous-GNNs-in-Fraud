"""
Graph I/O utilities for saving and loading graphs.
"""

import networkx as nx
import pickle
from pathlib import Path
from typing import Dict


def save_graph(G: nx.DiGraph, account_to_id: Dict[str, int], output_dir: str = 'outputs') -> None:
    """
    Save graph and mappings to disk.
    
    Args:
        G: NetworkX graph
        account_to_id: Account to ID mapping
        output_dir: Output directory
    """
    print("\nSaving graph...")
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save graph as pickle
    graph_path = output_dir / 'transaction_graph.gpickle'
    with open(graph_path, 'wb') as f:
        pickle.dump(G, f, pickle.HIGHEST_PROTOCOL)
    print(f"Saved graph to {graph_path}")
    
    # Save as GraphML (for Gephi, Cytoscape, etc.)
    graphml_path = output_dir / 'transaction_graph.graphml'
    nx.write_graphml(G, graphml_path)
    print(f"Saved GraphML to {graphml_path}")
    
    # Save account mapping
    mapping_path = output_dir / 'account_to_id.pkl'
    with open(mapping_path, 'wb') as f:
        pickle.dump(account_to_id, f)
    print(f"Saved account mapping to {mapping_path}")


def load_graph(graph_path: str = 'outputs/transaction_graph.gpickle') -> nx.DiGraph:
    """
    Load graph from disk.
    
    Args:
        graph_path: Path to graph file
        
    Returns:
        NetworkX graph
    """
    print(f"Loading graph from {graph_path}...")
    with open(graph_path, 'rb') as f:
        G = pickle.load(f)
    print(f"Loaded graph with {G.number_of_nodes():,} nodes and {G.number_of_edges():,} edges")
    return G


def load_account_mapping(mapping_path: str = 'outputs/account_to_id.pkl') -> Dict[str, int]:
    """
    Load account mapping from disk.
    
    Args:
        mapping_path: Path to mapping file
        
    Returns:
        Account to ID mapping dictionary
    """
    print(f"Loading account mapping from {mapping_path}...")
    with open(mapping_path, 'rb') as f:
        account_to_id = pickle.load(f)
    print(f"Loaded mapping for {len(account_to_id):,} accounts")
    return account_to_id
