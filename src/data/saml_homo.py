"""
Data preparation for SAML-D dataset.
Converts SAML-D transaction CSV to PyTorch Geometric format for GNN.

Usage:
    # First time - processes CSV and saves to cache
    data, account_mapping = load_and_prepare_saml_data()

    # Subsequent times - loads instantly from cache
    data, account_mapping = load_and_prepare_saml_data()

    # Force reprocessing (ignore cache)
    data, account_mapping = load_and_prepare_saml_data(use_cache=False)

    # Different cache file for experiments
    data, account_mapping = load_and_prepare_saml_data(
        sample_ratio=0.1,
        cache_path='data/processed/saml_10percent.pkl'
    )
"""

import pandas as pd
import torch
from torch_geometric.data import Data
import numpy as np
from tqdm import tqdm
import pickle
import os

from src.utils.config import PROJECT_ROOT


def load_and_prepare_saml_data(data_path=str(PROJECT_ROOT / 'datasets' / 'SAML-D.csv'),
                                sample_ratio=1.0,
                                cache_path=str(PROJECT_ROOT / 'data' / 'processed' / 'saml_graph.pkl'),
                                use_cache=True):
    """
    Load SAML-D CSV and create PyG Data object.

    Args:
        data_path: Path to SAML-D.csv file
        sample_ratio: Fraction of data to use (1.0 = full dataset)
        cache_path: Path to save/load preprocessed graph
        use_cache: If True, load from cache if available; save to cache after processing

    Returns:
        data: PyTorch Geometric Data object
        account_to_id: Dictionary mapping account IDs to node indices
    """
    # Try to load from cache
    if use_cache and os.path.exists(cache_path):
        print(f"Loading preprocessed graph from cache: {cache_path}")
        with open(cache_path, 'rb') as f:
            cache_data = pickle.load(f)
        print(f"Loaded graph with {cache_data['data'].num_nodes} nodes and {cache_data['data'].num_edges} edges")
        return cache_data['data'], cache_data['account_to_id']

    print("Loading SAML-D data...")
    df = pd.read_csv(data_path)

    print(f"Loaded {len(df)} transactions")
    print(f"Columns: {df.columns.tolist()}")
    print(f"Laundering transactions: {df['Is_laundering'].sum()} ({df['Is_laundering'].sum()/len(df)*100:.2f}%)")

    if sample_ratio < 1.0:
        print(f"Sampling {sample_ratio*100}% of transactions...")
        df = df.sample(frac=sample_ratio, random_state=42).reset_index(drop=True)
        print(f"Sampled {len(df)} transactions")

    # Create account mapping
    account_to_id = _create_account_mapping(df)

    # Build edge index and labels
    edge_index, edge_labels = _build_edges(df, account_to_id)

    # Create node features and labels
    node_features = _create_node_features(df, account_to_id)
    node_labels = _create_node_labels(df, account_to_id)

    # Create PyG Data object
    data = Data(
        x=node_features,
        edge_index=edge_index,
        y=node_labels,
        edge_attr=edge_labels
    )

    print(f"\nGraph Statistics:")
    print(f"  Nodes: {data.num_nodes}")
    print(f"  Edges: {data.num_edges}")
    print(f"  Node features: {data.num_node_features} dimensions")
    print(f"  Positive node labels: {node_labels.sum().item()} ({node_labels.sum().item()/len(node_labels)*100:.2f}%)")

    # Save to cache
    if use_cache:
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        print(f"Saving preprocessed graph to cache: {cache_path}")
        with open(cache_path, 'wb') as f:
            pickle.dump({'data': data, 'account_to_id': account_to_id}, f)
        print(f"Cache saved successfully")

    return data, account_to_id


def _create_account_mapping(df):
    """Map account identifiers to integer node IDs."""
    sender_accounts = df['Sender_account'].astype(str)
    receiver_accounts = df['Receiver_account'].astype(str)

    all_accounts = pd.concat([sender_accounts, receiver_accounts]).unique()
    account_to_id = {acc: idx for idx, acc in enumerate(all_accounts)}

    print(f"Found {len(account_to_id)} unique accounts")
    return account_to_id


def _build_edges(df, account_to_id):
    """Build edge index tensor and edge labels from transactions."""
    edge_list = []
    edge_labels = []

    print("Building edges...")
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Processing transactions"):
        sender = str(row['Sender_account'])
        receiver = str(row['Receiver_account'])

        if sender in account_to_id and receiver in account_to_id:
            edge_list.append([account_to_id[sender], account_to_id[receiver]])
            edge_labels.append(row['Is_laundering'])

    edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
    edge_labels = torch.tensor(edge_labels, dtype=torch.float)

    return edge_index, edge_labels


def _create_node_features(df, account_to_id):
    """
    Create node features from transaction patterns.

    Features:
        0: Out-degree (number of outgoing transactions)
        1: In-degree (number of incoming transactions)
        2: Total amount sent
        3: Average amount sent
        4: Total amount received
        5: Average amount received
        6: Number of unique recipient accounts
        7: Number of unique sender accounts
    """
    print("Creating node features...")
    num_nodes = len(account_to_id)
    features = torch.zeros((num_nodes, 8))

    # Convert account columns to string for consistency
    df['sender_str'] = df['Sender_account'].astype(str)
    df['receiver_str'] = df['Receiver_account'].astype(str)

    # --- Outgoing Statistics (Sender perspective) ---
    out_stats = df.groupby('sender_str').agg({
        'receiver_str': ['count', 'nunique'],
        'Amount': ['sum', 'mean']
    })
    out_stats.columns = ['out_degree', 'unique_receivers', 'total_sent', 'avg_sent']

    valid_out = out_stats[out_stats.index.isin(account_to_id.keys())]
    if not valid_out.empty:
        indices = torch.tensor([account_to_id[acc] for acc in valid_out.index], dtype=torch.long)
        features[indices, 0] = torch.tensor(valid_out['out_degree'].values, dtype=torch.float)
        features[indices, 2] = torch.tensor(valid_out['total_sent'].values, dtype=torch.float)
        features[indices, 3] = torch.tensor(valid_out['avg_sent'].values, dtype=torch.float)
        features[indices, 6] = torch.tensor(valid_out['unique_receivers'].values, dtype=torch.float)

    # --- Incoming Statistics (Receiver perspective) ---
    in_stats = df.groupby('receiver_str').agg({
        'sender_str': ['count', 'nunique'],
        'Amount': ['sum', 'mean']
    })
    in_stats.columns = ['in_degree', 'unique_senders', 'total_received', 'avg_received']

    valid_in = in_stats[in_stats.index.isin(account_to_id.keys())]
    if not valid_in.empty:
        indices = torch.tensor([account_to_id[acc] for acc in valid_in.index], dtype=torch.long)
        features[indices, 1] = torch.tensor(valid_in['in_degree'].values, dtype=torch.float)
        features[indices, 4] = torch.tensor(valid_in['total_received'].values, dtype=torch.float)
        features[indices, 5] = torch.tensor(valid_in['avg_received'].values, dtype=torch.float)
        features[indices, 7] = torch.tensor(valid_in['unique_senders'].values, dtype=torch.float)

    # Normalize features (z-score normalization)
    std = features.std(dim=0)
    std[std == 0] = 1.0  # Avoid division by zero
    features = (features - features.mean(dim=0)) / std

    return features


def _create_node_labels(df, account_to_id):
    """
    Create binary node labels.
    Label = 1 if account was involved in any laundering transaction (as sender).
    """
    print("Creating node labels...")
    labels = torch.zeros(len(account_to_id), dtype=torch.long)

    df['sender_str'] = df['Sender_account'].astype(str)

    # Find accounts that sent laundering transactions
    laundering_sums = df.groupby('sender_str')['Is_laundering'].sum()
    laundering_accounts = laundering_sums[laundering_sums > 0]

    # Filter for accounts in our graph
    valid_accounts = laundering_accounts[laundering_accounts.index.isin(account_to_id.keys())]

    # Set labels
    if len(valid_accounts) > 0:
        indices = torch.tensor([account_to_id[acc] for acc in valid_accounts.index], dtype=torch.long)
        labels[indices] = 1

    return labels


def save_graph(data, account_to_id, save_path):
    """Save preprocessed graph to disk."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, 'wb') as f:
        pickle.dump({'data': data, 'account_to_id': account_to_id}, f)
    print(f"Graph saved to {save_path}")


def load_graph(load_path):
    """Load preprocessed graph from disk."""
    with open(load_path, 'rb') as f:
        cache_data = pickle.load(f)
    return cache_data['data'], cache_data['account_to_id']


if __name__ == '__main__':
    # Test the data loading with caching
    print("=== First run (will process and cache) ===")
    data, account_mapping = load_and_prepare_saml_data(
        sample_ratio=0.1,
        cache_path='data/processed/saml_graph_sample.pkl'
    )
    print("\nData object:", data)
    print(f"Sample account mapping: {list(account_mapping.items())[:5]}")

    print("\n=== Second run (will load from cache) ===")
    data2, account_mapping2 = load_and_prepare_saml_data(
        sample_ratio=0.1,
        cache_path='data/processed/saml_graph_sample.pkl'
    )
    print("Loaded from cache successfully!")
