"""
Builds the feature vector for each transaction (edge).
These are per-row — no aggregation, just pulling and transforming columns.

The only thing that requires training data is the amount normalization:
we compute mean/std of log(amount) on training rows, then apply those
same stats to val/test rows so the scale is consistent.
"""

import math
import numpy as np
import pandas as pd


def fit_edge_stats(df: pd.DataFrame, col_cfg: dict, train_mask: pd.Series) -> dict:
    """
    Compute normalization stats from training rows.
    Call this once before building edge features.
    """
    val_col     = col_cfg["base_value"]
    train_vals  = np.log1p(df.loc[train_mask, val_col].fillna(0).values.astype(np.float32))
    return {
        "log_amount_mean": float(train_vals.mean()),
        "log_amount_std":  float(train_vals.std()) if train_vals.std() > 0 else 1.0,
    }


def build_edge_features(df: pd.DataFrame, col_cfg: dict, stats: dict) -> np.ndarray:
    """
    Build the edge feature matrix for a set of transaction rows.

    Args:
        df:       subset of the transaction dataframe for one edge type
        col_cfg:  columns section from master.yaml
        stats:    output of fit_edge_stats (normalization constants)

    Returns:
        np.ndarray of shape (num_edges, num_features), float32
    """
    parts = []

    # transaction amount — log1p to compress the scale, then z-score using training stats
    vals = np.log1p(df[col_cfg["base_value"]].fillna(0).values.astype(np.float32))
    vals = (vals - stats["log_amount_mean"]) / stats["log_amount_std"]
    parts.append(vals.reshape(-1, 1))

    # pre-encoded OHE groups — just pull the columns directly
    for group in ["channel", "submethod", "clearing", "currency", "destination"]:
        cols = [c.strip() for c in col_cfg["ohe_groups"][group] if c.strip() in df.columns]
        if cols:
            parts.append(df[cols].values.astype(np.float32))

    # international flag — convert the boolean/string column to 0/1
    intl_col = col_cfg["intl_flag"]
    if intl_col in df.columns:
        raw  = df[intl_col].astype(str).str.strip().str.lower()
        flag = raw.map({"true": 1.0, "false": 0.0, "1": 1.0, "0": 0.0}).fillna(0)
        parts.append(flag.values.astype(np.float32).reshape(-1, 1))

    # time encoding — sin/cos of hour-of-day and day-of-week
    # cyclical encoding so the model sees midnight and 23:00 as adjacent
    h   = df["_datetime"].dt.hour.values.astype(np.float32)
    dow = df["_datetime"].dt.dayofweek.values.astype(np.float32)
    time_enc = np.column_stack([
        np.sin(2 * math.pi * h   / 24),
        np.cos(2 * math.pi * h   / 24),
        np.sin(2 * math.pi * dow / 7),
        np.cos(2 * math.pi * dow / 7),
    ]).astype(np.float32)
    parts.append(time_enc)

    features = np.concatenate(parts, axis=1)
    print(f"    edge features: {features.shape[0]:,} edges, {features.shape[1]} dims")
    return features
