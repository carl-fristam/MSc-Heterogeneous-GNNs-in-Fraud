"""
Account-level feature extractors with a decorator registry.

Each feature is a function that receives the FULL DataFrame but a train_mask
to restrict aggregation to training-period data only (prevents temporal leakage).

Signature: fn(df, schema, account_to_id, train_mask) -> np.ndarray | None
    Returns shape (num_accounts, dim) or None if required columns are missing.

To add a new feature:
    @register_acct_feature("my_feature", dim=2)
    def my_feature(df, schema, account_to_id, train_mask):
        train_df = df[train_mask]
        ...
        return np.ndarray of shape (num_accounts, 2)
"""

import numpy as np
import pandas as pd

from src.graph_pipeline.schema import DatasetSchema
from src.graph_pipeline.normalize import zscore_tensor

import torch

_REGISTRY: dict[str, callable] = {}


def register_acct_feature(name: str, dim: int):
    """Decorator that registers an account feature extractor."""
    def decorator(fn):
        fn._feat_name = name
        fn._feat_dim = dim
        _REGISTRY[name] = fn
        return fn
    return decorator


def get_registered_features() -> dict[str, callable]:
    """Return a copy of the registry."""
    return dict(_REGISTRY)


def build_account_mapping(df: pd.DataFrame) -> dict[str, int]:
    """
    Build a mapping from account string ID to integer node index.

    Unions all senders and receivers into a single node pool —
    an account that both sends and receives is ONE node.
    """
    all_accounts = pd.concat([df["_sender"], df["_receiver"]]).unique()
    account_to_id = {acc: idx for idx, acc in enumerate(all_accounts)}
    print(f"  Account mapping: {len(account_to_id):,} unique accounts")
    return account_to_id


def build_acct_features(
    df: pd.DataFrame,
    schema: DatasetSchema,
    account_to_id: dict[str, int],
    train_mask: pd.Series,
    enabled: list[str] | None = None,
) -> torch.Tensor:
    """
    Build the full account feature matrix.

    IMPORTANT: All aggregations use only training-period rows (train_mask)
    to prevent temporal leakage.

    Returns:
        torch.Tensor of shape (num_accounts, total_dim), z-score normalized
    """
    registry = get_registered_features()

    if enabled is not None:
        registry = {k: registry[k] for k in enabled if k in registry}

    parts = []
    for name, fn in registry.items():
        result = fn(df, schema, account_to_id, train_mask)
        if result is None:
            continue
        parts.append(result)
        print(f"    acct feature '{name}': {result.shape[1]} dims")

    features = np.concatenate(parts, axis=1)
    print(f"  Total account features: {features.shape[1]} dims (before normalization)")

    # Z-score normalize the entire account feature matrix
    features = zscore_tensor(torch.tensor(features, dtype=torch.float32))
    return features


# Helper: place per-account aggregates into the feature matrix

def _place(agg_series: pd.Series, account_to_id: dict, num_accounts: int, col: int, features: np.ndarray):
    """
    Place a pandas Series (indexed by account string) into the correct
    rows of the feature matrix.
    """
    for acc, val in agg_series.items():
        if acc in account_to_id:
            features[account_to_id[acc], col] = val


# Built-in feature extractors

@register_acct_feature("degree", dim=2)
def _degree(df, schema, account_to_id, train_mask):
    """Out-degree (sent count) and in-degree (received count)."""
    num = len(account_to_id)
    features = np.zeros((num, 2), dtype=np.float32)
    train_df = df[train_mask]

    out_deg = train_df.groupby("_sender").size()
    in_deg = train_df.groupby("_receiver").size()

    _place(out_deg, account_to_id, num, 0, features)
    _place(in_deg, account_to_id, num, 1, features)
    return features


@register_acct_feature("amount_stats", dim=4)
def _amount_stats(df, schema, account_to_id, train_mask):
    """total_sent, total_received, mean_sent, mean_received."""
    num = len(account_to_id)
    features = np.zeros((num, 4), dtype=np.float32)
    train_df = df[train_mask]

    sender_agg = train_df.groupby("_sender")[schema.amount].agg(["sum", "mean"])
    recv_agg = train_df.groupby("_receiver")[schema.amount].agg(["sum", "mean"])

    _place(sender_agg["sum"], account_to_id, num, 0, features)
    _place(recv_agg["sum"], account_to_id, num, 1, features)
    _place(sender_agg["mean"], account_to_id, num, 2, features)
    _place(recv_agg["mean"], account_to_id, num, 3, features)
    return features


@register_acct_feature("counterparty_diversity", dim=2)
def _counterparty_diversity(df, schema, account_to_id, train_mask):
    """unique_receivers (as sender) and unique_senders (as receiver)."""
    num = len(account_to_id)
    features = np.zeros((num, 2), dtype=np.float32)
    train_df = df[train_mask]

    unique_recv = train_df.groupby("_sender")["_receiver"].nunique()
    unique_send = train_df.groupby("_receiver")["_sender"].nunique()

    _place(unique_recv, account_to_id, num, 0, features)
    _place(unique_send, account_to_id, num, 1, features)
    return features


@register_acct_feature("categorical_diversity", dim=2)
def _categorical_diversity(df, schema, account_to_id, train_mask):
    """n_payment_types and n_currencies_used (as sender)."""
    num = len(account_to_id)
    features = np.zeros((num, 2), dtype=np.float32)
    train_df = df[train_mask]

    if schema.payment_type is not None:
        n_ptypes = train_df.groupby("_sender")[schema.payment_type].nunique()
        _place(n_ptypes, account_to_id, num, 0, features)

    if schema.payment_currency is not None:
        n_curr = train_df.groupby("_sender")[schema.payment_currency].nunique()
        _place(n_curr, account_to_id, num, 1, features)

    return features


@register_acct_feature("geo_behavior", dim=3)
def _geo_behavior(df, schema, account_to_id, train_mask):
    """n_bank_locations_reached, cross_border_ratio (sender), cross_border_ratio (receiver)."""
    num = len(account_to_id)
    features = np.zeros((num, 3), dtype=np.float32)
    train_df = df[train_mask]

    if schema.sender_location is None or schema.receiver_location is None:
        return None

    # Unique bank locations reached as sender
    n_locs = train_df.groupby("_sender")[schema.receiver_location].nunique()
    _place(n_locs, account_to_id, num, 0, features)

    # Cross-border ratio
    is_cross = (train_df[schema.sender_location] != train_df[schema.receiver_location]).astype(float)
    train_with_cross = train_df.assign(_is_cross=is_cross)

    sender_cross = train_with_cross.groupby("_sender")["_is_cross"].mean()
    recv_cross = train_with_cross.groupby("_receiver")["_is_cross"].mean()

    _place(sender_cross, account_to_id, num, 1, features)
    _place(recv_cross, account_to_id, num, 2, features)
    return features


@register_acct_feature("time_behavior", dim=1)
def _time_behavior(df, schema, account_to_id, train_mask):
    """Night transaction ratio (22:00-06:00) as sender."""
    num = len(account_to_id)
    features = np.zeros((num, 1), dtype=np.float32)
    train_df = df[train_mask]

    hour = train_df["_datetime"].dt.hour
    is_night = ((hour >= 22) | (hour < 6)).astype(float)
    train_with_night = train_df.assign(_is_night=is_night)

    night_ratio = train_with_night.groupby("_sender")["_is_night"].mean()
    _place(night_ratio, account_to_id, num, 0, features)
    return features