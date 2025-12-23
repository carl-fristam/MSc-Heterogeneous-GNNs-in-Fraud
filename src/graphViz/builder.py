"""
Graph construction utilities for building NetworkX graphs from transaction data.
"""

import networkx as nx
import pandas as pd
from typing import Dict


def build_transaction_graph(trans_df: pd.DataFrame, account_to_id: Dict[str, int]) -> nx.DiGraph:
    """
    Build directed graph from transaction data.
    
    Args:
        trans_df: Transaction DataFrame
        account_to_id: Mapping from account identifiers to integer IDs
        
    Returns:
        NetworkX DiGraph with transaction edges
    """
    print("\nBuilding transaction graph...")
    
    # Create directed graph
    G = nx.DiGraph()
    
    # Add nodes (accounts)
    G.add_nodes_from(range(len(account_to_id)))
    
    # Add edges (transactions)
    edge_data = []
    
    for idx, row in trans_df.iterrows():
        from_acc = f"{str(row['From Bank'])}_{str(row['Account'])}"
        to_acc = f"{str(row['To Bank'])}_{str(row['Account.1'])}"
        
        if from_acc in account_to_id and to_acc in account_to_id:
            from_id = account_to_id[from_acc]
            to_id = account_to_id[to_acc]
            
            edge_data.append({
                'from': from_id,
                'to': to_id,
                'amount': row['Amount Received'],
                'currency': row['Receiving Currency'],
                'timestamp': row['Timestamp'],
                'payment_format': row['Payment Format'],
                'is_laundering': row['Is Laundering']
            })
    
    # Aggregate edges
    edge_dict = _aggregate_edges(edge_data)
    
    # Add edges to graph
    for (from_id, to_id), attrs in edge_dict.items():
        G.add_edge(
            from_id, 
            to_id,
            weight=attrs['count'],
            total_amount=attrs['total_amount'],
            avg_amount=attrs['total_amount'] / attrs['count'],
            transaction_count=attrs['count'],
            laundering_count=attrs['laundering_count'],
            laundering_ratio=attrs['laundering_count'] / attrs['count']
        )
    
    print(f"Created graph with {G.number_of_nodes():,} nodes and {G.number_of_edges():,} edges")
    
    return G


def _aggregate_edges(edge_data: list) -> dict:
    """
    Aggregate multiple transactions between same account pairs.
    
    Args:
        edge_data: List of edge dictionaries
        
    Returns:
        Dictionary mapping (from, to) tuples to aggregated attributes
    """
    edge_dict = {}
    
    for edge in edge_data:
        key = (edge['from'], edge['to'])
        if key not in edge_dict:
            edge_dict[key] = {
                'weight': 0,
                'count': 0,
                'total_amount': 0,
                'laundering_count': 0,
                'timestamps': [],
                'payment_formats': []
            }
        
        edge_dict[key]['count'] += 1
        edge_dict[key]['total_amount'] += edge['amount']
        edge_dict[key]['laundering_count'] += edge['is_laundering']
        edge_dict[key]['timestamps'].append(edge['timestamp'])
        edge_dict[key]['payment_formats'].append(edge['payment_format'])
    
    return edge_dict


def add_account_attributes(G: nx.DiGraph, accounts_df: pd.DataFrame, account_to_id: Dict[str, int]) -> nx.DiGraph:
    """
    Add account metadata as node attributes.
    
    Args:
        G: NetworkX graph
        accounts_df: Accounts DataFrame
        account_to_id: Mapping from account identifiers to integer IDs
        
    Returns:
        Graph with added node attributes
    """
    print("\nAdding account attributes...")
    
    # Create reverse mapping
    id_to_account = {v: k for k, v in account_to_id.items()}
    
    # Create account lookup
    accounts_df['account_id'] = accounts_df['Bank ID'].astype(str) + '_' + accounts_df['Account Number'].astype(str)
    account_info = accounts_df.set_index('account_id').to_dict('index')
    
    # Add attributes to nodes
    for node_id in G.nodes():
        account_id = id_to_account.get(node_id)
        if account_id and account_id in account_info:
            info = account_info[account_id]
            G.nodes[node_id]['bank_name'] = info.get('Bank Name', 'Unknown')
            G.nodes[node_id]['bank_id'] = info.get('Bank ID', 'Unknown')
            G.nodes[node_id]['entity_id'] = info.get('Entity ID', 'Unknown')
            G.nodes[node_id]['entity_name'] = info.get('Entity Name', 'Unknown')
    
    attributed_nodes = len([n for n in G.nodes() if 'bank_name' in G.nodes[n]])
    print(f"Added attributes to {attributed_nodes} nodes")
    
    return G
