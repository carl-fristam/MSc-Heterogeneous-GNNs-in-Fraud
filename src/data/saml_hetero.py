"""
Heterogeneous data preparation for SAML-D dataset.
Converts SAML-D to PyTorch Geometric HeteroData for HMPNN.

Edge types are based on Payment_type:
- Credit card, Debit card, Cheque, ACH, Cross-border, Cash Withdrawal, Cash Deposit

Usage:
    data = load_hetero_saml_data()
    print(data.edge_types)  # List of edge types
    print(data['account', 'credit_card', 'account'].edge_index)
"""

import pandas as pd
import torch
from torch_geometric.data import HeteroData
import numpy as np
from tqdm import tqdm
import pickle
import os

from src.utils.config import PROJECT_ROOT


# Map Payment_type values to valid edge type names (no spaces, lowercase)
PAYMENT_TYPE_MAP = {
    'Credit card': 'credit_card',
    'Debit card': 'debit_card',
    'Cheque': 'cheque',
    'ACH': 'ach',
    'Cross-border': 'cross_border',
    'Cash Withdrawal': 'cash_withdrawal',
    'Cash Deposit': 'cash_deposit',
}


def load_hetero_saml_data(
    data_path=str(PROJECT_ROOT / 'datasets' / 'SAML-D.csv'),
    sample_ratio=1.0,
    cache_path=str(PROJECT_ROOT / 'data' / 'processed' / 'saml_hetero.pkl'),
    use_cache=True
):
    """
    Load SAML-D CSV and create PyG HeteroData object.

    Args:
        data_path: Path to SAML-D.csv file
        sample_ratio: Fraction of data to use (1.0 = full dataset)
        cache_path: Path to save/load preprocessed graph
        use_cache: If True, load from cache if available

    Returns:
        data: PyTorch Geometric HeteroData object
        account_to_id: Dictionary mapping account IDs to node indices
    """
    if use_cache and os.path.exists(cache_path):
        print(f"Loading from cache: {cache_path}")
        with open(cache_path, 'rb') as f:
            cache = pickle.load(f)
        return cache['data'], cache['account_to_id']

    print("Loading SAML-D data...")
    df = pd.read_csv(data_path)
    print(f"Loaded {len(df)} transactions")

    if sample_ratio < 1.0:
        df = df.sample(frac=sample_ratio, random_state=42).reset_index(drop=True)
        print(f"Sampled {len(df)} transactions")

    # Create account mapping
    account_to_id = _create_account_mapping(df)
    num_accounts = len(account_to_id)

    # Create HeteroData
    data = HeteroData()

    # Node features (same as homogeneous version)
    node_features = _create_node_features(df, account_to_id)
    node_labels = _create_node_labels(df, account_to_id)
    data['account'].x = node_features
    data['account'].y = node_labels

    # Build edges per payment type
    print("Building heterogeneous edges...")
    for payment_type, edge_type_name in tqdm(PAYMENT_TYPE_MAP.items(), desc="Edge types"):
        df_type = df[df['Payment_type'] == payment_type]

        if len(df_type) == 0:
            continue

        edge_index, edge_attr = _build_edges_for_type(df_type, account_to_id)

        # Store as (src_type, edge_type, dst_type)
        data['account', edge_type_name, 'account'].edge_index = edge_index
        data['account', edge_type_name, 'account'].edge_attr = edge_attr

    # Print stats
    print(f"\nHeteroData Statistics:")
    print(f"  Account nodes: {num_accounts}")
    print(f"  Node features: {node_features.shape[1]} dimensions")
    print(f"  Edge types: {len(data.edge_types)}")
    for edge_type in data.edge_types:
        num_edges = data[edge_type].edge_index.shape[1]
        print(f"    {edge_type}: {num_edges} edges")

    # Cache
    if use_cache:
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        with open(cache_path, 'wb') as f:
            pickle.dump({'data': data, 'account_to_id': account_to_id}, f)
        print(f"Saved to cache: {cache_path}")

    return data, account_to_id


def _create_account_mapping(df):
    """Map account identifiers to integer node IDs."""
    sender_accounts = df['Sender_account'].astype(str)
    receiver_accounts = df['Receiver_account'].astype(str)
    all_accounts = pd.concat([sender_accounts, receiver_accounts]).unique()
    account_to_id = {acc: idx for idx, acc in enumerate(all_accounts)}
    print(f"Found {len(account_to_id)} unique accounts")
    return account_to_id


def _build_edges_for_type(df, account_to_id):
    """
    Build edge_index and edge_attr for a specific payment type.

    Edge features:
        0: Amount (normalized)
        1: Hour of day (0-23, normalized)
        2: Is cross-border (sender_bank != receiver_bank)
        3: Is same currency (payment_currency == received_currency)

    Note: Is_laundering is intentionally excluded — it is the label and
    including it as an edge feature would cause data leakage.
    """
    # Vectorised account ID lookup — avoids slow iterrows() on 9.5M rows
    sender_ids = df['Sender_account'].astype(str).map(account_to_id)
    receiver_ids = df['Receiver_account'].astype(str).map(account_to_id)

    # Drop rows where either account is not in the mapping
    valid = sender_ids.notna() & receiver_ids.notna()
    sender_ids = sender_ids[valid].astype(int)
    receiver_ids = receiver_ids[valid].astype(int)
    df = df[valid]

    edge_index = torch.tensor(
        np.stack([sender_ids.values, receiver_ids.values], axis=0), dtype=torch.long
    )

    hour = df['Time'].str.split(':').str[0].astype(float, errors='ignore').fillna(0) / 23.0
    is_cross_border = (df['Sender_bank_location'] != df['Receiver_bank_location']).astype(float)
    is_same_currency = (df['Payment_currency'] == df['Received_currency']).astype(float)

    edge_attr = torch.tensor(
        np.stack([df['Amount'].values, hour.values, is_cross_border.values, is_same_currency.values], axis=1),
        dtype=torch.float
    )

    # Normalize amount (first feature)
    if edge_attr.shape[0] > 0:
        amount_std = edge_attr[:, 0].std()
        if amount_std > 0:
            edge_attr[:, 0] = (edge_attr[:, 0] - edge_attr[:, 0].mean()) / amount_std

    return edge_index, edge_attr


def _create_node_features(df, account_to_id):
    """
    Create node features from transaction patterns.
    Same as homogeneous version for compatibility.
    """
    num_nodes = len(account_to_id)
    features = torch.zeros((num_nodes, 8))

    df['sender_str'] = df['Sender_account'].astype(str)
    df['receiver_str'] = df['Receiver_account'].astype(str)

    # Outgoing stats
    out_stats = df.groupby('sender_str').agg({
        'receiver_str': ['count', 'nunique'],
        'Amount': ['sum', 'mean']
    })
    out_stats.columns = ['out_degree', 'unique_receivers', 'total_sent', 'avg_sent']

    for acc in out_stats.index:
        if acc in account_to_id:
            idx = account_to_id[acc]
            features[idx, 0] = out_stats.loc[acc, 'out_degree']
            features[idx, 2] = out_stats.loc[acc, 'total_sent']
            features[idx, 3] = out_stats.loc[acc, 'avg_sent']
            features[idx, 6] = out_stats.loc[acc, 'unique_receivers']

    # Incoming stats
    in_stats = df.groupby('receiver_str').agg({
        'sender_str': ['count', 'nunique'],
        'Amount': ['sum', 'mean']
    })
    in_stats.columns = ['in_degree', 'unique_senders', 'total_received', 'avg_received']

    for acc in in_stats.index:
        if acc in account_to_id:
            idx = account_to_id[acc]
            features[idx, 1] = in_stats.loc[acc, 'in_degree']
            features[idx, 4] = in_stats.loc[acc, 'total_received']
            features[idx, 5] = in_stats.loc[acc, 'avg_received']
            features[idx, 7] = in_stats.loc[acc, 'unique_senders']

    # Normalize
    std = features.std(dim=0)
    std[std == 0] = 1.0
    features = (features - features.mean(dim=0)) / std

    return features


def _create_node_labels(df, account_to_id):
    """Binary node labels: 1 if account sent any laundering transaction."""
    labels = torch.zeros(len(account_to_id), dtype=torch.long)
    df['sender_str'] = df['Sender_account'].astype(str)

    laundering_sums = df.groupby('sender_str')['Is_laundering'].sum()
    for acc, count in laundering_sums.items():
        if count > 0 and acc in account_to_id:
            labels[account_to_id[acc]] = 1

    return labels


if __name__ == '__main__':
    # Test with small sample
    data, mapping = load_hetero_saml_data(
        sample_ratio=0.01,
        cache_path='data/processed/saml_hetero_test.pkl',
        use_cache=False
    )
    print("\nHeteroData object:")
    print(data)