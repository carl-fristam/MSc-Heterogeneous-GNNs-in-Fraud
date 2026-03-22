"""
Edge (transaction) feature extractors for the bank heterogeneous graph.

Each feature is a function decorated with @register_edge_feature.
Features are per-row — no train-only restriction needed since they don't
aggregate across transactions.

OHE columns are pre-encoded in the dataset. The `ohe_groups` config maps
group names to their column lists. Extractors simply pull the columns.

Decorator usage:
    @register_edge_feature("my_feature", dim=2)
    def _my_feature(df, col_cfg, vocabs):
        ...
        return np.ndarray of shape (len(df), 2)
"""

import math
import numpy as np
import pandas as pd

from src.graph_pipeline_bank.normalize import zscore


# ── Registry ──────────────────────────────────────────────────────────────────

_REGISTRY: dict[str, callable] = {}


def register_edge_feature(name: str, dim: int):
    def decorator(fn):
        fn._feat_name = name
        fn._feat_dim  = dim
        _REGISTRY[name] = fn
        return fn
    return decorator


def get_registry() -> dict[str, callable]:
    return dict(_REGISTRY)


# ── Build entry point ─────────────────────────────────────────────────────────

def build_edge_features(
    df: pd.DataFrame,
    col_cfg: dict,
    vocabs: dict[str, list[str]],
    enabled: list[str] | None = None,
) -> np.ndarray:
    """
    Build edge feature matrix for a subset of transaction rows.

    Args:
        df:       DataFrame subset (rows for one relation type)
        col_cfg:  config["columns"]
        vocabs:   {feature_name: vocab_list} fitted on training data (legacy, unused for pre-OHE)
        enabled:  feature names to include (None = all registered)

    Returns:
        np.ndarray of shape (len(df), total_dim), float32
    """
    registry = get_registry()
    if enabled is not None:
        registry = {k: registry[k] for k in enabled if k in registry}

    parts = []
    for name, fn in registry.items():
        result = fn(df, col_cfg, vocabs)
        if result is not None:
            parts.append(result)

    if not parts:
        return np.zeros((len(df), 1), dtype=np.float32)

    return np.concatenate(parts, axis=1).astype(np.float32)


def fit_vocabs(df: pd.DataFrame, col_cfg: dict, train_mask: pd.Series) -> dict[str, list[str]]:
    """
    Legacy vocab fitting. With pre-OHE'd data this is a no-op but kept
    for API compatibility with the graph builder.
    """
    return {}


# ── Helpers ───────────────────────────────────────────────────────────────────

def _grab_ohe_group(df, col_cfg, group_name):
    """
    Pull a pre-OHE'd column group from the DataFrame.
    Returns np.ndarray of shape (len(df), num_cols) or None if group not configured.
    """
    ohe_groups = col_cfg.get("ohe_groups", {})
    cols = ohe_groups.get(group_name)
    if not cols:
        return None
    # Only use columns that exist in the DataFrame
    present = [c for c in cols if c in df.columns]
    if not present:
        return None
    return df[present].values.astype(np.float32)


# ── Feature extractors ────────────────────────────────────────────────────────

@register_edge_feature("log_base_value", dim=1)
def _log_base_value(df, col_cfg, vocabs):
    """log1p(BASEVALUE), z-score normalised."""
    bval_col = col_cfg.get("base_value")
    if not bval_col or bval_col not in df.columns:
        return None
    vals = np.log1p(df[bval_col].fillna(0).values.astype(np.float32))
    return zscore(vals).reshape(-1, 1)


@register_edge_feature("channel_ohe", dim=None)
def _channel_ohe(df, col_cfg, vocabs):
    """Pre-encoded CHANNEL_* columns."""
    return _grab_ohe_group(df, col_cfg, "channel")


@register_edge_feature("submethod_ohe", dim=None)
def _submethod_ohe(df, col_cfg, vocabs):
    """Pre-encoded PAYMENTSUBMETHOD_* columns."""
    return _grab_ohe_group(df, col_cfg, "submethod")


@register_edge_feature("clearing_ohe", dim=None)
def _clearing_ohe(df, col_cfg, vocabs):
    """Pre-encoded PAYMENTCLEARING_* columns."""
    return _grab_ohe_group(df, col_cfg, "clearing")


@register_edge_feature("currency_ohe", dim=None)
def _currency_ohe(df, col_cfg, vocabs):
    """Pre-encoded CURRENCY_TBE_* columns."""
    return _grab_ohe_group(df, col_cfg, "currency")


@register_edge_feature("counter_agent_ohe", dim=None)
def _counter_agent_ohe(df, col_cfg, vocabs):
    """Pre-encoded COUNTERAGENT_TBE_* columns."""
    return _grab_ohe_group(df, col_cfg, "counter_agent")


@register_edge_feature("sender_bank_ohe", dim=None)
def _sender_bank_ohe(df, col_cfg, vocabs):
    """Pre-encoded ACCOUNTAGENTID_* columns."""
    return _grab_ohe_group(df, col_cfg, "sender_bank")


@register_edge_feature("counter_id_format", dim=None)
def _counter_id_format(df, col_cfg, vocabs):
    """Pre-encoded COUNTERIDFORMAT_* columns."""
    return _grab_ohe_group(df, col_cfg, "counter_id_format")


@register_edge_feature("destination_ohe", dim=None)
def _destination_ohe(df, col_cfg, vocabs):
    """Pre-encoded DESTINATION_TBE_* columns."""
    return _grab_ohe_group(df, col_cfg, "destination")


@register_edge_feature("branch_tbe_ohe", dim=None)
def _branch_tbe_ohe(df, col_cfg, vocabs):
    """Pre-encoded ACCOUNTBRANCH_TBE_* columns."""
    return _grab_ohe_group(df, col_cfg, "branch_tbe")


@register_edge_feature("international_flag", dim=1)
def _international_flag(df, col_cfg, vocabs):
    """INTERNATIONALFLAG as binary float."""
    col = col_cfg.get("intl_flag")
    if not col or col not in df.columns:
        return None
    raw = df[col].astype(str).str.strip().str.lower()
    return raw.map({"true": 1.0, "false": 0.0, "1": 1.0, "0": 0.0}).fillna(0).astype(np.float32).values.reshape(-1, 1)


@register_edge_feature("time_encoding", dim=4)
def _time_encoding(df, col_cfg, vocabs):
    """
    Cyclical sin/cos encoding of hour-of-day and day-of-week.
    Dims: [sin_hour, cos_hour, sin_dow, cos_dow]
    """
    dt  = df["_datetime"]
    h   = dt.dt.hour.values.astype(np.float32)
    dow = dt.dt.dayofweek.values.astype(np.float32)
    return np.column_stack([
        np.sin(2 * math.pi * h   / 24),
        np.cos(2 * math.pi * h   / 24),
        np.sin(2 * math.pi * dow / 7),
        np.cos(2 * math.pi * dow / 7),
    ]).astype(np.float32)
