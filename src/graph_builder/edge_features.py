"""
Selects pre-computed edge features from the transaction DataFrame.

All feature engineering (log-transform, OHE, cyclical encoding, velocity
features) is done in the external data pipeline. This module just picks
the configured columns and returns them as a numpy array.
"""

import numpy as np
import pandas as pd


def build_edge_features(df: pd.DataFrame, edge_feature_cols: list) -> np.ndarray:
    """
    Select pre-computed edge features for a set of transaction rows.

    Args:
        df:                 subset of the transaction DataFrame for one edge type
        edge_feature_cols:  list of column names from config["edge_features"]

    Returns:
        np.ndarray of shape (num_edges, num_features), float32
    """
    present = [c for c in edge_feature_cols if c in df.columns]
    missing = [c for c in edge_feature_cols if c not in df.columns]
    if missing:
        print(f"    WARNING: {len(missing)} edge feature columns not found: {missing[:5]}...")

    features = df[present].fillna(0).values.astype(np.float32)
    print(f"    edge features: {features.shape[0]:,} edges, {features.shape[1]} dims")
    return features
