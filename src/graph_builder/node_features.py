"""
Selects pre-computed node features from the transaction DataFrame.

Node features are account-level properties (target encodings, format flags).
Since the DataFrame has one row per transaction, we aggregate per account
using the mean across training rows. For target-encoded columns this is a
no-op (same value per account), but mean is safe regardless.

Accounts not seen in training rows get all-zero feature vectors.
"""

import numpy as np
import torch
import pandas as pd


def build_internal_node_features(
    df: pd.DataFrame,
    node_map: dict,
    train_mask: pd.Series,
    feature_cols: list,
) -> torch.Tensor:

    n = len(node_map)
    train_df = df[train_mask]

    features = np.zeros((n, len(feature_cols)), dtype=np.float32)
    grouped = train_df.groupby("_sender")[feature_cols].mean()
    for acc_id, row in grouped.iterrows():
        if acc_id in node_map:
            features[node_map[acc_id]] = row.values.astype(np.float32)

    print(f"  internal_account: {n:,} nodes, {len(feature_cols)} features")
    return torch.tensor(features, dtype=torch.float32)


def build_external_node_features(
    df: pd.DataFrame,
    col_cfg: dict,
    node_map: dict,
    train_mask: pd.Series,
    feature_cols: list,
) -> torch.Tensor:

    n = len(node_map)
    onus_col = col_cfg["onus_flag"]
    train_df = df[train_mask & (df[onus_col] == False)]

    features = np.zeros((n, len(feature_cols)), dtype=np.float32)
    grouped = train_df.groupby("_receiver")[feature_cols].mean()
    for acc_id, row in grouped.iterrows():
        if acc_id in node_map:
            features[node_map[acc_id]] = row.values.astype(np.float32)

    print(f"  external_account: {n:,} nodes, {len(feature_cols)} features")
    return torch.tensor(features, dtype=torch.float32)
