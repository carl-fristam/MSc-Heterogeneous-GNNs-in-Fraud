"""
Transaction-level feature extractors with a decorator registry.

Each feature is a function decorated with @register_txn_feature.
The pipeline collects all registered features, calls them, and
concatenates the results into a single feature matrix.

To add a new feature:
    @register_txn_feature("my_feature", dim=3)
    def my_feature(df, schema):
        ...
        return np.ndarray of shape (len(df), 3)
"""

import math
import numpy as np
import pandas as pd

from src.graph_pipeline.schema import DatasetSchema
from src.graph_pipeline.normalize import zscore, one_hot


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

_REGISTRY: dict[str, callable] = {}


def register_txn_feature(name: str, dim: int):
    """
    Decorator that registers a transaction feature extractor.

    Args:
        name: unique identifier for this feature
        dim:  number of columns it produces
    """
    def decorator(fn):
        fn._feat_name = name
        fn._feat_dim = dim
        _REGISTRY[name] = fn
        return fn
    return decorator


def get_registered_features() -> dict[str, callable]:
    """Return a copy of the registry."""
    return dict(_REGISTRY)


def build_txn_features(
    df: pd.DataFrame,
    schema: DatasetSchema,
    enabled: list[str] | None = None,
) -> np.ndarray:
    """
    Build the full transaction feature matrix by calling all registered extractors.

    Args:
        df:      DataFrame with raw transaction data
        schema:  column name mapping
        enabled: list of feature names to include (None = all registered)

    Returns:
        np.ndarray of shape (len(df), total_dim)
    """
    registry = get_registered_features()

    if enabled is not None:
        # Only use the requested features, in the specified order
        registry = {k: registry[k] for k in enabled if k in registry}

    parts = []
    for name, fn in registry.items():
        result = fn(df, schema)
        if result is None:
            # Feature not available for this dataset (missing column)
            continue
        parts.append(result)
        print(f"    txn feature '{name}': {result.shape[1]} dims")

    features = np.concatenate(parts, axis=1)
    print(f"  Total transaction features: {features.shape[1]} dims")
    return features


# ---------------------------------------------------------------------------
# Fixed vocabularies (derived from SAML-D, extensible for bank data)
# ---------------------------------------------------------------------------

CURRENCY_VOCAB = [
    "Albanian lek", "Dirham", "Euro", "Indian rupee", "Mexican Peso",
    "Moroccan dirham", "Naira", "Pakistani rupee", "Swiss franc",
    "Turkish lira", "UK pounds", "US dollar", "Yen",
]

BANK_LOC_VOCAB = [
    "Albania", "Austria", "France", "Germany", "India", "Italy",
    "Japan", "Mexico", "Morocco", "Netherlands", "Nigeria", "Pakistan",
    "Spain", "Switzerland", "Turkey", "UAE", "UK", "USA",
]

PAYMENT_TYPE_VOCAB = [
    "ACH", "Cash Deposit", "Cash Withdrawal", "Cheque",
    "Credit card", "Cross-border", "Debit card",
]


# ---------------------------------------------------------------------------
# Built-in feature extractors
# ---------------------------------------------------------------------------

@register_txn_feature("amount_zscore", dim=1)
def _amount_zscore(df: pd.DataFrame, schema: DatasetSchema) -> np.ndarray:
    """Z-score normalized transaction amount."""
    return zscore(df[schema.amount].values).reshape(-1, 1)


@register_txn_feature("payment_currency", dim=13)
def _payment_currency(df: pd.DataFrame, schema: DatasetSchema) -> np.ndarray | None:
    if schema.payment_currency is None:
        return None
    return one_hot(df[schema.payment_currency], CURRENCY_VOCAB)


@register_txn_feature("received_currency", dim=13)
def _received_currency(df: pd.DataFrame, schema: DatasetSchema) -> np.ndarray | None:
    if schema.received_currency is None:
        return None
    return one_hot(df[schema.received_currency], CURRENCY_VOCAB)


@register_txn_feature("sender_location", dim=18)
def _sender_location(df: pd.DataFrame, schema: DatasetSchema) -> np.ndarray | None:
    if schema.sender_location is None:
        return None
    return one_hot(df[schema.sender_location], BANK_LOC_VOCAB)


@register_txn_feature("receiver_location", dim=18)
def _receiver_location(df: pd.DataFrame, schema: DatasetSchema) -> np.ndarray | None:
    if schema.receiver_location is None:
        return None
    return one_hot(df[schema.receiver_location], BANK_LOC_VOCAB)


@register_txn_feature("payment_type", dim=7)
def _payment_type(df: pd.DataFrame, schema: DatasetSchema) -> np.ndarray | None:
    if schema.payment_type is None:
        return None
    return one_hot(df[schema.payment_type], PAYMENT_TYPE_VOCAB)


@register_txn_feature("time_cyclical", dim=4)
def _time_cyclical(df: pd.DataFrame, schema: DatasetSchema) -> np.ndarray:
    """
    Cyclical encoding of hour-of-day and day-of-week.

    Why cyclical? Because 23:00 and 01:00 are 2 hours apart, but numerically
    they're 22 apart. Sin/cos encoding wraps around so the model sees the
    true circular distance.
    """
    dt = df["_datetime"]
    hour = dt.dt.hour.values.astype(np.float32)
    dow = dt.dt.dayofweek.values.astype(np.float32)

    return np.column_stack([
        np.sin(2 * math.pi * hour / 24),
        np.cos(2 * math.pi * hour / 24),
        np.sin(2 * math.pi * dow / 7),
        np.cos(2 * math.pi * dow / 7),
    ]).astype(np.float32)


@register_txn_feature("cross_border_flags", dim=2)
def _cross_border_flags(df: pd.DataFrame, schema: DatasetSchema) -> np.ndarray | None:
    """Binary flags: is_cross_border and is_same_currency."""
    if schema.sender_location is None or schema.receiver_location is None:
        return None
    is_cross = (df[schema.sender_location] != df[schema.receiver_location]).values.astype(np.float32)
    is_same_curr = (df[schema.payment_currency] == df[schema.received_currency]).values.astype(np.float32)
    return np.column_stack([is_cross, is_same_curr])
