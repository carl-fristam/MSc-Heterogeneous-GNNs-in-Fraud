"""
Data preparation for GraphSAGE.
Converts transaction CSV to PyTorch Geometric format.
"""

import pandas as pd
import torch
from torch_geometric.data import Data
import numpy as np
from tqdm import tqdm

from src.utils.config import PROJECT_ROOT


def load_and_prepare_data(trans_path=str(PROJECT_ROOT / 'data' / 'HI-Small_Trans.csv'),
                          accounts_path=str(PROJECT_ROOT / 'data' / 'HI-Small_accounts.csv'),
                          sample_ratio=1.0): # Default to Full Dataset
    """Load CSVs and create PyG Data object."""
    print("Loading data...")
    trans_df = pd.read_csv(trans_path)
    accounts_df = pd.read_csv(accounts_path)

    if sample_ratio < 1.0:
        print(f"Sampling {sample_ratio*100}% of accounts (preserving structure)...")

        # Create helper columns for filtering
        trans_df['from_id'] = trans_df['From Bank'].astype(str) + '_' + trans_df['Account'].astype(str)
        trans_df['to_id'] = trans_df['To Bank'].astype(str) + '_' + trans_df['Account.1'].astype(str)

        # Get all unique accounts per class
        # (We need to look at 'Is Laundering' to know which accounts are bad)
        # Note: In this dataset, the label is on the EDGE (Transaction).
        # We consider an account 'bad' if it ever initiated a laundering transaction.

        laundering_trans = trans_df[trans_df['Is Laundering'] == 1]
        bad_accounts = laundering_trans['from_id'].unique()

        all_accounts = pd.concat([trans_df['from_id'], trans_df['to_id']]).unique()
        clean_accounts = np.setdiff1d(all_accounts, bad_accounts)

        # Sample accounts
        # Keep ALL bad accounts (to preserve signal)
        # Sample random subset of clean accounts
        num_clean_to_sample = int(len(clean_accounts) * sample_ratio)
        sampled_clean = np.random.choice(clean_accounts, size=num_clean_to_sample, replace=False)

        target_accounts = set(bad_accounts) | set(sampled_clean)
        print(f"Selected {len(target_accounts)} target accounts ({len(bad_accounts)} laundering, {len(sampled_clean)} clean)")

        # Filter transactions: Keep if EITHER side is in our target set
        # This preserves the 1-hop neighborhood for all target accounts
        mask = trans_df['from_id'].isin(target_accounts) | trans_df['to_id'].isin(target_accounts)
        trans_df = trans_df[mask].reset_index(drop=True)

        print(f"Retained {len(trans_df)} transactions involving target accounts")

    # Create account mapping
    account_to_id = _create_account_mapping(trans_df)

    # Build edge index and labels
    edge_index, edge_labels = _build_edges(trans_df, account_to_id)

    # Create node features and labels
    node_features = _create_node_features(account_to_id, accounts_df, trans_df)
    node_labels = _create_node_labels(trans_df, account_to_id)

    # Create PyG Data object
    data = Data(
        x=node_features,
        edge_index=edge_index,
        y=node_labels,
        edge_attr=edge_labels
    )

    print(f"Graph: {data.num_nodes} nodes, {data.num_edges} edges")
    print(f"Features: {data.num_node_features} dimensions")
    print(f"Positive labels: {node_labels.sum().item()} ({node_labels.sum().item()/len(node_labels)*100:.2f}%)")

    return data, account_to_id


def _create_account_mapping(trans_df):
    """Map account identifiers to integer IDs."""
    from_accounts = trans_df['From Bank'].astype(str) + '_' + trans_df['Account'].astype(str)
    to_accounts = trans_df['To Bank'].astype(str) + '_' + trans_df['Account.1'].astype(str)

    all_accounts = pd.concat([from_accounts, to_accounts]).unique()
    account_to_id = {acc: idx for idx, acc in enumerate(all_accounts)}

    print(f"Found {len(account_to_id)} unique accounts")
    return account_to_id


def _build_edges(trans_df, account_to_id):
    """Build edge index and edge labels."""
    edge_list = []
    edge_labels = []

    print("Building edges...")
    for _, row in tqdm(trans_df.iterrows(), total=len(trans_df), desc="Processing transactions"):
        from_acc = f"{row['From Bank']}_{row['Account']}"
        to_acc = f"{row['To Bank']}_{row['Account.1']}"

        if from_acc in account_to_id and to_acc in account_to_id:
            edge_list.append([account_to_id[from_acc], account_to_id[to_acc]])
            edge_labels.append(row['Is Laundering'])

    edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
    edge_labels = torch.tensor(edge_labels, dtype=torch.float)

    return edge_index, edge_labels


def _create_node_features(account_to_id, accounts_df, trans_df):
    """Create simple node features using vectorized operations."""
    print("Creating node features...")
    num_nodes = len(account_to_id)

    # Initialize feature matrix
    # 0: Out-degree
    # 1: In-degree
    # 2: Total Amount Sent
    # 3: Avg Amount Sent
    # 4: Total Amount Received
    features = torch.zeros((num_nodes, 5))

    # Create helper columns
    # (Already computed during sampling or need to be recomputed if full logic changed)
    # Recomputing to be safe if sample_ratio=1.0 case skipped the block above
    if 'from_id' not in trans_df.columns:
        trans_df['from_id'] = trans_df['From Bank'].astype(str) + '_' + trans_df['Account'].astype(str)
    if 'to_id' not in trans_df.columns:
        trans_df['to_id'] = trans_df['To Bank'].astype(str) + '_' + trans_df['Account.1'].astype(str)

    # --- Outgoing Statistics (Sender) ---
    out_stats = trans_df.groupby('from_id').agg({
        'Amount Received': ['count', 'sum', 'mean']
    })
    out_stats.columns = ['out_degree', 'total_sent', 'avg_sent']

    valid_out = out_stats[out_stats.index.isin(account_to_id.keys())]
    if not valid_out.empty:
        indices = [account_to_id[acc] for acc in valid_out.index]
        indices_tensor = torch.tensor(indices, dtype=torch.long)

        features[indices_tensor, 0] = torch.tensor(valid_out['out_degree'].values, dtype=torch.float)
        features[indices_tensor, 2] = torch.tensor(valid_out['total_sent'].values, dtype=torch.float)
        features[indices_tensor, 3] = torch.tensor(valid_out['avg_sent'].values, dtype=torch.float)

    # --- Incoming Statistics (Receiver) ---
    in_stats = trans_df.groupby('to_id').agg({
        'Amount Received': ['count', 'sum']
    })
    in_stats.columns = ['in_degree', 'total_received']

    valid_in = in_stats[in_stats.index.isin(account_to_id.keys())]
    if not valid_in.empty:
        indices = [account_to_id[acc] for acc in valid_in.index]
        indices_tensor = torch.tensor(indices, dtype=torch.long)

        features[indices_tensor, 1] = torch.tensor(valid_in['in_degree'].values, dtype=torch.float)
        features[indices_tensor, 4] = torch.tensor(valid_in['total_received'].values, dtype=torch.float)

    # Normalize features
    std = features.std(dim=0)
    std[std == 0] = 1.0
    features = (features - features.mean(dim=0)) / std

    return features


def _create_node_labels(trans_df, account_to_id):
    """Create binary node labels (1 if account has any laundering transactions) using vectorized operations."""
    print("Creating node labels...")
    labels = torch.zeros(len(account_to_id), dtype=torch.long)

    if 'from_id' not in trans_df.columns:
        trans_df['from_id'] = trans_df['From Bank'].astype(str) + '_' + trans_df['Account'].astype(str)

    # Group by account and sum laundering flag
    laundering_sums = trans_df.groupby('from_id')['Is Laundering'].sum()

    # Filter for accounts that have at least one laundering transaction
    laundering_accounts = laundering_sums[laundering_sums > 0]

    # Filter for accounts in our graph
    valid_accounts = laundering_accounts[laundering_accounts.index.isin(account_to_id.keys())]

    # Get indices for these accounts
    indices = [account_to_id[acc] for acc in valid_accounts.index]

    if indices:
        indices_tensor = torch.tensor(indices, dtype=torch.long)
        labels[indices_tensor] = 1

    return labels
