"""
Computes feature vectors for each node type.
All aggregations use training rows only — no leakage into val/test.
Nodes that never appear in training get all-zero features.
"""

import numpy as np
import torch
import pandas as pd


# ── Helpers ───────────────────────────────────────────────────────────────────

def _place(series: pd.Series, node_map: dict, out: np.ndarray, col: int):
    """Write a {account_id: scalar} series into the correct rows of the output array."""
    for acc_id, val in series.items():
        if acc_id in node_map:
            out[node_map[acc_id], col] = float(val)


def _zscore(arr: np.ndarray) -> np.ndarray:
    mean = arr.mean(axis=0)
    std  = arr.std(axis=0)
    std[std == 0] = 1.0
    return (arr - mean) / std


# ── Internal account features ─────────────────────────────────────────────────

def build_internal_node_features(
    df: pd.DataFrame,
    col_cfg: dict,
    node_map: dict,
    train_mask: pd.Series,
) -> torch.Tensor:

    n        = len(node_map)
    train_df = df[train_mask]
    val_col  = col_cfg["base_value"]   # BASEVALUE
    parts    = []

    # out-degree: how many transactions this account sent
    out_degree = np.zeros((n, 1), dtype=np.float32)
    _place(train_df.groupby("_sender").size(), node_map, out_degree, 0)
    parts.append(out_degree)

    # amount stats: mean, std, total sent (log1p to compress the scale)
    amount = np.zeros((n, 3), dtype=np.float32)
    grp = train_df.groupby("_sender")[val_col]
    _place(np.log1p(grp.mean()),             node_map, amount, 0)
    _place(np.log1p(grp.std().fillna(0)),    node_map, amount, 1)
    _place(np.log1p(grp.sum()),              node_map, amount, 2)
    parts.append(amount)

    # counterparty diversity: how many unique receivers this account has sent to
    cp_div = np.zeros((n, 1), dtype=np.float32)
    _place(train_df.groupby("_sender")["_receiver"].nunique(), node_map, cp_div, 0)
    parts.append(cp_div)

    # channel diversity: how many distinct payment channels this account has used
    ch_cols = [c.strip() for c in col_cfg["ohe_groups"]["channel"] if c.strip() in train_df.columns]
    ch_div  = np.zeros((n, 1), dtype=np.float32)
    if ch_cols:
        diversity = train_df.groupby("_sender")[ch_cols].max().sum(axis=1)
        _place(diversity, node_map, ch_div, 0)
    parts.append(ch_div)

    # time behavior: fraction of transactions at night (22-6) and on weekends
    time_feat = np.zeros((n, 2), dtype=np.float32)
    tmp = train_df.copy()
    tmp["_night"]   = ((tmp["_datetime"].dt.hour >= 22) | (tmp["_datetime"].dt.hour < 6)).astype(float)
    tmp["_weekend"] = (tmp["_datetime"].dt.dayofweek >= 5).astype(float)
    grp = tmp.groupby("_sender")
    _place(grp["_night"].mean(),   node_map, time_feat, 0)
    _place(grp["_weekend"].mean(), node_map, time_feat, 1)
    parts.append(time_feat)

    # branch OHE: which Danske Bank branch this account belongs to
    # static per account so we just take the first observed row
    branch_cols = [c.strip() for c in col_cfg["ohe_groups"]["branch_tbe"] if c.strip() in train_df.columns]
    if branch_cols:
        branch_arr  = np.zeros((n, len(branch_cols)), dtype=np.float32)
        first_rows  = train_df.drop_duplicates("_sender").set_index("_sender")[branch_cols]
        for acc_id, row in first_rows.iterrows():
            if acc_id in node_map:
                branch_arr[node_map[acc_id]] = row.values.astype(np.float32)
        parts.append(branch_arr)

    # sender bank OHE: which Danske Bank entity (country subsidiary) processed the transaction
    # also static per account in practice
    bank_cols = [c.strip() for c in col_cfg["ohe_groups"]["sender_bank"] if c.strip() in train_df.columns]
    if bank_cols:
        bank_arr   = np.zeros((n, len(bank_cols)), dtype=np.float32)
        first_rows = train_df.drop_duplicates("_sender").set_index("_sender")[bank_cols]
        for acc_id, row in first_rows.iterrows():
            if acc_id in node_map:
                bank_arr[node_map[acc_id]] = row.values.astype(np.float32)
        parts.append(bank_arr)

    features = np.concatenate(parts, axis=1)
    features = _zscore(features)

    print(f"  internal_account: {features.shape[0]:,} nodes, {features.shape[1]} features")
    return torch.tensor(features, dtype=torch.float32)


# ── External account features ─────────────────────────────────────────────────

def build_external_node_features(
    df: pd.DataFrame,
    col_cfg: dict,
    node_map: dict,
    train_mask: pd.Series,
) -> torch.Tensor:

    n        = len(node_map)
    val_col  = col_cfg["base_value"]
    onus_col = col_cfg["onus_flag"]

    # external accounts only appear in non-onus transactions
    train_df = df[train_mask & (df[onus_col] == False)]
    parts    = []

    # in-degree: how many transactions this account has received
    in_degree = np.zeros((n, 1), dtype=np.float32)
    _place(train_df.groupby("_receiver").size(), node_map, in_degree, 0)
    parts.append(in_degree)

    # received amount stats: mean and std (log1p)
    amount = np.zeros((n, 2), dtype=np.float32)
    grp = train_df.groupby("_receiver")[val_col]
    _place(np.log1p(grp.mean()),          node_map, amount, 0)
    _place(np.log1p(grp.std().fillna(0)), node_map, amount, 1)
    parts.append(amount)

    # sender diversity: how many unique internal accounts sent money here
    # high value is an AML signal (funds funnelled from many sources)
    sd = np.zeros((n, 1), dtype=np.float32)
    _place(train_df.groupby("_receiver")["_sender"].nunique(), node_map, sd, 0)
    parts.append(sd)

    # sender bank diversity: how many distinct Danske Bank entities sent here
    # high value means money arriving from multiple bank subsidiaries
    bank_col = col_cfg["sender_bank"]   # raw ACCOUNTAGENTID column
    if bank_col in train_df.columns:
        sbd = np.zeros((n, 1), dtype=np.float32)
        _place(train_df.groupby("_receiver")[bank_col].nunique(), node_map, sbd, 0)
        parts.append(sbd)

    # counter agent OHE: which bank the external account belongs to
    # static per account — take first observed row
    ca_cols = [c.strip() for c in col_cfg["ohe_groups"]["counter_agent"] if c.strip() in train_df.columns]
    if ca_cols:
        ca_arr     = np.zeros((n, len(ca_cols)), dtype=np.float32)
        first_rows = train_df.drop_duplicates("_receiver").set_index("_receiver")[ca_cols]
        for acc_id, row in first_rows.iterrows():
            if acc_id in node_map:
                ca_arr[node_map[acc_id]] = row.values.astype(np.float32)
        parts.append(ca_arr)

    # counter ID format OHE: BBAN or IBAN — property of the external account
    fmt_cols = [c.strip() for c in col_cfg["ohe_groups"]["counter_id_format"] if c.strip() in train_df.columns]
    if fmt_cols:
        fmt_arr    = np.zeros((n, len(fmt_cols)), dtype=np.float32)
        first_rows = train_df.drop_duplicates("_receiver").set_index("_receiver")[fmt_cols]
        for acc_id, row in first_rows.iterrows():
            if acc_id in node_map:
                fmt_arr[node_map[acc_id]] = row.values.astype(np.float32)
        parts.append(fmt_arr)

    features = np.concatenate(parts, axis=1)
    features = _zscore(features)

    print(f"  external_account: {features.shape[0]:,} nodes, {features.shape[1]} features")
    return torch.tensor(features, dtype=torch.float32)
