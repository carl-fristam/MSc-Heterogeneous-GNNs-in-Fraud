"""
Data preparation for GraphSAGE.
Converts transaction CSV to PyTorch Geometric format.
"""

import pandas as pd
import torch
from torch_geometric.data import Data
import numpy as np
from tqdm import tqdm


def load_and_prepare_data(trans_path='data/HI-Small_Trans.csv', 
                          accounts_path='data/HI-Small_accounts.csv'):
    """Load CSVs and create PyG Data object."""
    print("Loading data...")
    trans_df = pd.read_csv(trans_path)
    accounts_df = pd.read_csv(accounts_path)
    
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
    """Create simple node features."""
    num_nodes = len(account_to_id)
    
    # Initialize feature matrix
    features = torch.zeros((num_nodes, 5))
    
    # Feature 1-2: In/Out degree (will be computed from edges)
    # Feature 3: Total transaction amount (outgoing)
    # Feature 4: Average transaction amount
    # Feature 5: Laundering ratio
    
    # Compute transaction statistics per account
    trans_df['from_id'] = trans_df['From Bank'].astype(str) + '_' + trans_df['Account'].astype(str)
    
    for acc, idx in account_to_id.items():
        acc_trans = trans_df[trans_df['from_id'] == acc]
        
        if len(acc_trans) > 0:
            features[idx, 2] = acc_trans['Amount Received'].sum()  # Total amount
            features[idx, 3] = acc_trans['Amount Received'].mean()  # Avg amount
            features[idx, 4] = acc_trans['Is Laundering'].mean()  # Laundering ratio
    
    # Normalize features
    features = (features - features.mean(dim=0)) / (features.std(dim=0) + 1e-8)
    
    return features


def _create_node_labels(trans_df, account_to_id):
    """Create binary node labels (1 if account has any laundering transactions)."""
    labels = torch.zeros(len(account_to_id), dtype=torch.long)
    
    trans_df['from_id'] = trans_df['From Bank'].astype(str) + '_' + trans_df['Account'].astype(str)
    
    for acc, idx in account_to_id.items():
        acc_trans = trans_df[trans_df['from_id'] == acc]
        if len(acc_trans) > 0 and acc_trans['Is Laundering'].sum() > 0:
            labels[idx] = 1
    
    return labels
