"""
Computes feature vectors for each node type from static (account-level) columns.

These are one-hot encoded properties of the account itself — not computed aggregations.
Aggregations (degree, amount stats, diversity, time behaviour) belong in the tabular
feature engineering stage so XGBoost and the GNN see the same inputs.

All lookups use training rows only — nodes not seen in training get all-zero features.
"""

import numpy as np
import torch
import pandas as pd


def build_internal_node_features(
    df: pd.DataFrame,
    col_cfg: dict,
    node_map: dict,
    train_mask: pd.Series,
) -> torch.Tensor:

    n        = len(node_map)
    train_df = df[train_mask]
    parts    = []

    # branch OHE: which Danske Bank branch this account belongs to (static per account)
    branch_cols = [c.strip() for c in col_cfg["ohe_groups"]["branch_tbe"] if c.strip() in train_df.columns]
    if branch_cols:
        branch_arr = np.zeros((n, len(branch_cols)), dtype=np.float32)
        first_rows = train_df.drop_duplicates("_sender").set_index("_sender")[branch_cols]
        for acc_id, row in first_rows.iterrows():
            if acc_id in node_map:
                branch_arr[node_map[acc_id]] = row.values.astype(np.float32)
        parts.append(branch_arr)

    # sender bank OHE: which Danske Bank entity processed transactions (static per account)
    bank_cols = [c.strip() for c in col_cfg["ohe_groups"]["sender_bank"] if c.strip() in train_df.columns]
    if bank_cols:
        bank_arr   = np.zeros((n, len(bank_cols)), dtype=np.float32)
        first_rows = train_df.drop_duplicates("_sender").set_index("_sender")[bank_cols]
        for acc_id, row in first_rows.iterrows():
            if acc_id in node_map:
                bank_arr[node_map[acc_id]] = row.values.astype(np.float32)
        parts.append(bank_arr)

    features = np.concatenate(parts, axis=1) if parts else np.zeros((n, 1), dtype=np.float32)

    print(f"  internal_account: {features.shape[0]:,} nodes, {features.shape[1]} features")
    return torch.tensor(features, dtype=torch.float32)


def build_external_node_features(
    df: pd.DataFrame,
    col_cfg: dict,
    node_map: dict,
    train_mask: pd.Series,
) -> torch.Tensor:

    n        = len(node_map)
    onus_col = col_cfg["onus_flag"]
    train_df = df[train_mask & (df[onus_col] == False)]
    parts    = []

    # counter agent OHE: which bank the external account belongs to (static per account)
    ca_cols = [c.strip() for c in col_cfg["ohe_groups"]["counter_agent"] if c.strip() in train_df.columns]
    if ca_cols:
        ca_arr     = np.zeros((n, len(ca_cols)), dtype=np.float32)
        first_rows = train_df.drop_duplicates("_receiver").set_index("_receiver")[ca_cols]
        for acc_id, row in first_rows.iterrows():
            if acc_id in node_map:
                ca_arr[node_map[acc_id]] = row.values.astype(np.float32)
        parts.append(ca_arr)

    # counter ID format OHE: BBAN or IBAN (static per account)
    fmt_cols = [c.strip() for c in col_cfg["ohe_groups"]["counter_id_format"] if c.strip() in train_df.columns]
    if fmt_cols:
        fmt_arr    = np.zeros((n, len(fmt_cols)), dtype=np.float32)
        first_rows = train_df.drop_duplicates("_receiver").set_index("_receiver")[fmt_cols]
        for acc_id, row in first_rows.iterrows():
            if acc_id in node_map:
                fmt_arr[node_map[acc_id]] = row.values.astype(np.float32)
        parts.append(fmt_arr)

    features = np.concatenate(parts, axis=1) if parts else np.zeros((n, 1), dtype=np.float32)

    print(f"  external_account: {features.shape[0]:,} nodes, {features.shape[1]} features")
    return torch.tensor(features, dtype=torch.float32)
