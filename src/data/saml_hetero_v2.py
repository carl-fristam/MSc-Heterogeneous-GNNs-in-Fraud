"""
saml_hetero_v2.py

Bipartite heterogeneous graph for SAML-D with TWO node types:
  - account nodes:     senders/receivers (14-dim aggregated features)
  - transaction nodes: individual transactions (76-dim per-event features)

Edge types:
  - (account, sends, transaction)       : sender → transaction
  - (transaction, received_by, account) : transaction → receiver
  - (transaction, sent_by, account)     : reverse of sends
  - (account, receives, transaction)    : reverse of received_by

This is parallel to saml_hetero.py (v1) — v1 is untouched.
"""

import os
import math
import pickle
import numpy as np
import pandas as pd
import torch
from torch_geometric.data import HeteroData
from tqdm import tqdm

from src.utils.config import PROJECT_ROOT

# ---------------------------------------------------------------------------
# Fixed vocabularies (derived from full SAML-D dataset)
# ---------------------------------------------------------------------------

CURRENCY_VOCAB = [
    'Albanian lek', 'Dirham', 'Euro', 'Indian rupee', 'Mexican Peso',
    'Moroccan dirham', 'Naira', 'Pakistani rupee', 'Swiss franc',
    'Turkish lira', 'UK pounds', 'US dollar', 'Yen',
]

BANK_LOC_VOCAB = [
    'Albania', 'Austria', 'France', 'Germany', 'India', 'Italy',
    'Japan', 'Mexico', 'Morocco', 'Netherlands', 'Nigeria', 'Pakistan',
    'Spain', 'Switzerland', 'Turkey', 'UAE', 'UK', 'USA',
]

PAYMENT_TYPE_VOCAB = [
    'ACH', 'Cash Deposit', 'Cash Withdrawal', 'Cheque',
    'Credit card', 'Cross-border', 'Debit card',
]


def load_hetero_v2(
    data_path: str = str(PROJECT_ROOT / 'datasets' / 'SAML-D.csv'),
    sample_ratio: float = 0.1,
    n_days: int = None,
    cache_path: str = None,
    use_cache: bool = True,
    add_reverse_edges: bool = True,
):
    """
    Load SAML-D and build bipartite HeteroData.

    Returns:
        data:           HeteroData with account and transaction node types
        account_to_id:  dict mapping account string → int node id
    """
    if cache_path is None:
        suffix = f"d{n_days}" if n_days else f"sr{sample_ratio:.2f}"
        cache_path = str(PROJECT_ROOT / 'data' / 'processed' / f'saml_hetero_v2_{suffix}.pkl')

    if use_cache and os.path.exists(cache_path):
        print(f"Loading from cache: {cache_path}")
        with open(cache_path, 'rb') as f:
            cache = pickle.load(f)
        return cache['data'], cache['account_to_id']

    print("Loading SAML-D data...")
    df = pd.read_csv(data_path)
    print(f"Loaded {len(df)} transactions")

    df = _sample_dataframe(df, sample_ratio, n_days)
    df = df.reset_index(drop=True)

    # String-typed account columns for consistent mapping
    df['sender_str'] = df['Sender_account'].astype(str)
    df['receiver_str'] = df['Receiver_account'].astype(str)

    account_to_id = _build_account_mapping(df)
    num_accounts = len(account_to_id)
    num_txns = len(df)

    data = HeteroData()

    # Account features
    print("Building account features...")
    data['account'].x = _build_account_features(df, account_to_id)

    # Transaction features
    print("Building transaction features...")
    data['transaction'].x = _build_transaction_features(df)

    # Transaction labels (not used for training, only evaluation)
    data['transaction'].is_laundering = torch.tensor(
        df['Is_laundering'].values, dtype=torch.float
    )

    # Edges
    print("Building edges...")
    edge_dict = _build_edges(df, account_to_id, add_reverse_edges)
    for edge_type, edge_index in edge_dict.items():
        data[edge_type].edge_index = edge_index

    # Print stats
    print(f"\nHeteroData v2 Statistics:")
    print(f"  Account nodes:     {num_accounts}")
    print(f"  Transaction nodes: {num_txns}")
    print(f"  Account feat dim:  {data['account'].x.shape[1]}")
    print(f"  Txn feat dim:      {data['transaction'].x.shape[1]}")
    print(f"  Edge types:        {len(data.edge_types)}")
    for et in data.edge_types:
        print(f"    {et}: {data[et].edge_index.shape[1]} edges")
    pos = int(data['transaction'].is_laundering.sum().item())
    print(f"  Laundering txns:   {pos} ({100*pos/num_txns:.2f}%)")

    # Cache
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    with open(cache_path, 'wb') as f:
        pickle.dump({'data': data, 'account_to_id': account_to_id}, f)
    print(f"Saved to cache: {cache_path}")

    return data, account_to_id


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _sample_dataframe(df: pd.DataFrame, sample_ratio: float, n_days: int) -> pd.DataFrame:
    """Apply sampling: temporal window (n_days) takes priority over random ratio."""
    if n_days is not None:
        df['_date'] = pd.to_datetime(df['Date'])
        cutoff = df['_date'].min() + pd.Timedelta(days=n_days)
        df = df[df['_date'] < cutoff].drop(columns=['_date'])
        print(f"Temporal sample: first {n_days} days → {len(df)} transactions")
    elif sample_ratio < 1.0:
        df = df.sample(frac=sample_ratio, random_state=42)
        print(f"Random sample: {sample_ratio:.0%} → {len(df)} transactions")
    return df


def _build_account_mapping(df: pd.DataFrame) -> dict:
    """Union of sender and receiver accounts → sequential int ids."""
    all_accounts = pd.concat([df['sender_str'], df['receiver_str']]).unique()
    account_to_id = {acc: idx for idx, acc in enumerate(all_accounts)}
    print(f"Found {len(account_to_id)} unique accounts")
    return account_to_id


def _build_account_features(df: pd.DataFrame, account_to_id: dict) -> torch.Tensor:
    """
    14-dim account features, z-score normalised.

    0: out_degree          1: in_degree
    2: total_sent          3: total_received
    4: mean_sent           5: mean_received
    6: unique_receivers    7: unique_senders
    8: n_payment_types     9: n_bank_locs_reached
    10: cross_border_ratio_sender  11: cross_border_ratio_receiver
    12: night_ratio_sender         13: n_currencies_used
    """
    num_nodes = len(account_to_id)
    features = np.zeros((num_nodes, 14), dtype=np.float32)

    is_cross = (df['Sender_bank_location'] != df['Receiver_bank_location'])
    hour = df['Time'].str.split(':').str[0].astype(float, errors='ignore').fillna(0)
    is_night = ((hour >= 22) | (hour < 6)).astype(float)

    # Sender-side aggregates
    sender_grp = df.groupby('sender_str')
    out_stats = sender_grp.agg(
        out_degree=('receiver_str', 'count'),
        total_sent=('Amount', 'sum'),
        mean_sent=('Amount', 'mean'),
        unique_receivers=('receiver_str', 'nunique'),
        n_payment_types=('Payment_type', 'nunique'),
        n_bank_locs=('Receiver_bank_location', 'nunique'),
        n_currencies=('Payment_currency', 'nunique'),
    )

    # Cross-border ratio and night ratio need manual groupby
    df_sender_extra = df.assign(is_cross=is_cross.astype(float), is_night=is_night)
    sender_extra = df_sender_extra.groupby('sender_str').agg(
        cross_border_ratio=('is_cross', 'mean'),
        night_ratio=('is_night', 'mean'),
    )

    for acc in out_stats.index:
        if acc in account_to_id:
            i = account_to_id[acc]
            features[i, 0] = out_stats.loc[acc, 'out_degree']
            features[i, 2] = out_stats.loc[acc, 'total_sent']
            features[i, 4] = out_stats.loc[acc, 'mean_sent']
            features[i, 6] = out_stats.loc[acc, 'unique_receivers']
            features[i, 8] = out_stats.loc[acc, 'n_payment_types']
            features[i, 9] = out_stats.loc[acc, 'n_bank_locs']
            features[i, 13] = out_stats.loc[acc, 'n_currencies']
    for acc in sender_extra.index:
        if acc in account_to_id:
            i = account_to_id[acc]
            features[i, 10] = sender_extra.loc[acc, 'cross_border_ratio']
            features[i, 12] = sender_extra.loc[acc, 'night_ratio']

    # Receiver-side aggregates
    recv_grp = df.groupby('receiver_str')
    in_stats = recv_grp.agg(
        in_degree=('sender_str', 'count'),
        total_received=('Amount', 'sum'),
        mean_received=('Amount', 'mean'),
        unique_senders=('sender_str', 'nunique'),
    )
    recv_extra = df_sender_extra.groupby('receiver_str').agg(
        cross_border_ratio_recv=('is_cross', 'mean'),
    )

    for acc in in_stats.index:
        if acc in account_to_id:
            i = account_to_id[acc]
            features[i, 1] = in_stats.loc[acc, 'in_degree']
            features[i, 3] = in_stats.loc[acc, 'total_received']
            features[i, 5] = in_stats.loc[acc, 'mean_received']
            features[i, 7] = in_stats.loc[acc, 'unique_senders']
    for acc in recv_extra.index:
        if acc in account_to_id:
            i = account_to_id[acc]
            features[i, 11] = recv_extra.loc[acc, 'cross_border_ratio_recv']

    # Z-score normalise
    features = torch.tensor(features, dtype=torch.float)
    std = features.std(dim=0)
    std[std == 0] = 1.0
    features = (features - features.mean(dim=0)) / std

    return features


def _one_hot(series: pd.Series, vocab: list) -> np.ndarray:
    """One-hot encode a pandas Series against a fixed vocabulary."""
    val_to_idx = {v: i for i, v in enumerate(vocab)}
    indices = series.map(val_to_idx).fillna(-1).astype(int).values
    result = np.zeros((len(series), len(vocab)), dtype=np.float32)
    valid = indices >= 0
    result[valid, indices[valid]] = 1.0
    return result


def _build_transaction_features(df: pd.DataFrame) -> torch.Tensor:
    """
    76-dim transaction features:
      amount(1) + payment_curr(13) + received_curr(13) +
      sender_loc(18) + receiver_loc(18) + payment_type(7) +
      hour_sin(1) + hour_cos(1) + dow_sin(1) + dow_cos(1) +
      is_cross_border(1) + is_same_currency(1)
    """
    parts = []

    # Amount (z-score)
    amount = df['Amount'].values.astype(np.float32)
    amt_std = amount.std()
    if amt_std > 0:
        amount = (amount - amount.mean()) / amt_std
    parts.append(amount.reshape(-1, 1))

    # One-hot categoricals
    parts.append(_one_hot(df['Payment_currency'], CURRENCY_VOCAB))
    parts.append(_one_hot(df['Received_currency'], CURRENCY_VOCAB))
    parts.append(_one_hot(df['Sender_bank_location'], BANK_LOC_VOCAB))
    parts.append(_one_hot(df['Receiver_bank_location'], BANK_LOC_VOCAB))
    parts.append(_one_hot(df['Payment_type'], PAYMENT_TYPE_VOCAB))

    # Cyclical time encoding
    hour = df['Time'].str.split(':').str[0].astype(float, errors='ignore').fillna(0).values
    parts.append(np.sin(2 * math.pi * hour / 24).reshape(-1, 1).astype(np.float32))
    parts.append(np.cos(2 * math.pi * hour / 24).reshape(-1, 1).astype(np.float32))

    # Day of week (0=Monday, 6=Sunday)
    dow = pd.to_datetime(df['Date']).dt.dayofweek.values.astype(np.float32)
    parts.append(np.sin(2 * math.pi * dow / 7).reshape(-1, 1).astype(np.float32))
    parts.append(np.cos(2 * math.pi * dow / 7).reshape(-1, 1).astype(np.float32))

    # Binary flags
    is_cross = (df['Sender_bank_location'] != df['Receiver_bank_location']).values.astype(np.float32)
    is_same_curr = (df['Payment_currency'] == df['Received_currency']).values.astype(np.float32)
    parts.append(is_cross.reshape(-1, 1))
    parts.append(is_same_curr.reshape(-1, 1))

    features = np.concatenate(parts, axis=1)
    return torch.tensor(features, dtype=torch.float)


def _build_edges(df: pd.DataFrame, account_to_id: dict, add_reverse: bool) -> dict:
    """
    Build edge_index tensors. Each transaction i creates:
      - (account, sends, transaction):       [sender_id, i]
      - (transaction, received_by, account): [i, receiver_id]
    Plus reverse edges if add_reverse=True.
    """
    sender_ids = df['sender_str'].map(account_to_id).values.astype(np.int64)
    receiver_ids = df['receiver_str'].map(account_to_id).values.astype(np.int64)
    txn_ids = np.arange(len(df), dtype=np.int64)

    edges = {}

    # Forward edges
    edges[('account', 'sends', 'transaction')] = torch.tensor(
        np.stack([sender_ids, txn_ids]), dtype=torch.long
    )
    edges[('transaction', 'received_by', 'account')] = torch.tensor(
        np.stack([txn_ids, receiver_ids]), dtype=torch.long
    )

    if add_reverse:
        edges[('transaction', 'sent_by', 'account')] = torch.tensor(
            np.stack([txn_ids, sender_ids]), dtype=torch.long
        )
        edges[('account', 'receives', 'transaction')] = torch.tensor(
            np.stack([receiver_ids, txn_ids]), dtype=torch.long
        )

    return edges
