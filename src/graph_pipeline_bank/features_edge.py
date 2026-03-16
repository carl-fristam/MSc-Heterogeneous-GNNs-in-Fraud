"""
Edge (transaction) feature extractors for the bank heterogeneous graph.

Each feature is a function decorated with @register_edge_feature.
Features are per-row — no train-only restriction needed since they don't
aggregate across transactions.

Vocabularies for OHE columns are fitted on training rows and passed in
via the `vocabs` dict (built once in edge_builder, stored in cache).

Decorator usage:
    @register_edge_feature("my_feature", dim=2)
    def _my_feature(df, col_cfg, vocabs):
        ...
        return np.ndarray of shape (len(df), 2)
"""

import math
import numpy as np
import pandas as pd

from src.graph_pipeline_bank.normalize import zscore, one_hot, one_hot_from_training


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
        vocabs:   {feature_name: vocab_list} fitted on training data
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
    Fit OHE vocabularies from training data for all categorical edge features.
    Call once during graph build; store in cache alongside the graph.
    """
    vocabs = {}
    from src.graph_pipeline_bank.normalize import vocab_from_training

    for key in ("channel", "method", "submethod", "clearing"):
        col = col_cfg.get(key)
        if col and col in df.columns:
            vocabs[key] = vocab_from_training(df.loc[train_mask, col])
            print(f"    vocab '{key}': {vocabs[key]}")

    return vocabs


# ── Feature extractors ────────────────────────────────────────────────────────

@register_edge_feature("log_value", dim=1)
def _log_value(df, col_cfg, vocabs):
    """log1p(VALUE), z-score normalised. Handles heavy-tailed amount distributions."""
    val_col = col_cfg.get("value")
    if not val_col or val_col not in df.columns:
        return None
    vals = np.log1p(df[val_col].fillna(0).values.astype(np.float32))
    return zscore(vals).reshape(-1, 1)


@register_edge_feature("log_base_value", dim=1)
def _log_base_value(df, col_cfg, vocabs):
    """log1p(BASEVALUE), z-score normalised. Amount in reporting currency."""
    bval_col = col_cfg.get("base_value")
    if not bval_col or bval_col not in df.columns:
        return None
    vals = np.log1p(df[bval_col].fillna(0).values.astype(np.float32))
    return zscore(vals).reshape(-1, 1)


@register_edge_feature("currency_mismatch", dim=1)
def _currency_mismatch(df, col_cfg, vocabs):
    """
    Binary: CURRENCY != BASECURRENCY.
    Flags FX conversion — cross-currency transactions are higher risk.
    """
    c1 = col_cfg.get("currency")
    c2 = col_cfg.get("base_currency")
    if not c1 or not c2 or c1 not in df.columns or c2 not in df.columns:
        return None
    mismatch = (df[c1] != df[c2]).astype(np.float32).values.reshape(-1, 1)
    return mismatch


@register_edge_feature("channel_ohe", dim=None)  # dim set dynamically
def _channel_ohe(df, col_cfg, vocabs):
    """OHE of CHANNEL (e.g. mobile, internet, branch, atm). Vocab from training."""
    col = col_cfg.get("channel")
    if not col or col not in df.columns or "channel" not in vocabs:
        return None
    return one_hot(df[col].astype(str).fillna("__null__"), vocabs["channel"])


@register_edge_feature("method_ohe", dim=None)
def _method_ohe(df, col_cfg, vocabs):
    """OHE of PAYMENTMETHOD (online / file / bulk). Vocab from training."""
    col = col_cfg.get("method")
    if not col or col not in df.columns or "method" not in vocabs:
        return None
    return one_hot(df[col].astype(str).fillna("__null__"), vocabs["method"])


@register_edge_feature("submethod_ohe", dim=None)
def _submethod_ohe(df, col_cfg, vocabs):
    """
    OHE of PAYMENTSUBMETHOD (realTime, bankGiro, plusGiro, futurePayment,
    salary, accountClosure, chaps). Vocab from training.
    Each submethod captures a structurally different transaction type.
    """
    col = col_cfg.get("submethod")
    if not col or col not in df.columns or "submethod" not in vocabs:
        return None
    return one_hot(df[col].astype(str).fillna("__null__"), vocabs["submethod"])


@register_edge_feature("clearing_express", dim=1)
def _clearing_express(df, col_cfg, vocabs):
    """
    Binary: PAYMENTCLEARING != 'default'.
    Express/instant clearing reduces the fraud interception window.
    """
    col = col_cfg.get("clearing")
    if not col or col not in df.columns:
        return None
    is_express = (df[col].astype(str).str.lower() != "default").astype(np.float32)
    return is_express.values.reshape(-1, 1)


@register_edge_feature("international_flag", dim=1)
def _international_flag(df, col_cfg, vocabs):
    """INTERNATIONALFLAG as binary float. Cross-border transactions are higher risk."""
    col = col_cfg.get("intl_flag")
    if not col or col not in df.columns:
        return None
    return df[col].fillna(0).astype(np.float32).values.reshape(-1, 1)


@register_edge_feature("time_encoding", dim=4)
def _time_encoding(df, col_cfg, vocabs):
    """
    Cyclical sin/cos encoding of hour-of-day and day-of-week.
    Captures periodic patterns without the boundary artefact of raw integers
    (e.g. 23:00 and 01:00 are 2 hours apart, not 22).
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
