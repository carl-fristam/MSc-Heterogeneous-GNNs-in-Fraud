"""
Node feature extractors for the bank heterogeneous graph.

Three separate registries:
  _INTERNAL_REGISTRY — features for InternalAccount nodes (aggregated as sender)
  _EXTERNAL_REGISTRY — features for ExternalAccount nodes (aggregated as receiver)
  _DEVICE_REGISTRY   — features for Device nodes (V3 only)

All aggregations are computed on TRAINING data only to prevent temporal leakage.

Decorator usage:
    @register_internal_feature("my_feature", dim=2)
    def _my_feature(df, col_cfg, node_to_id, train_mask):
        train_df = df[train_mask]
        ...
        return np.ndarray of shape (len(node_to_id), 2)

    @register_external_feature("my_feature", dim=1)
    def _my_feature(df, col_cfg, node_to_id, train_mask): ...

    @register_device_feature("my_feature", dim=1)
    def _my_feature(df, col_cfg, node_to_id, train_mask): ...
"""

import numpy as np
import pandas as pd
import torch

from src.graph_pipeline_bank.normalize import zscore_cols, one_hot_from_training


# ── Registries ────────────────────────────────────────────────────────────────

_INTERNAL_REGISTRY: dict[str, callable] = {}
_EXTERNAL_REGISTRY: dict[str, callable] = {}
_DEVICE_REGISTRY:   dict[str, callable] = {}

_REGISTRIES = {
    "internal_account": _INTERNAL_REGISTRY,
    "external_account": _EXTERNAL_REGISTRY,
    "device":           _DEVICE_REGISTRY,
}


def _make_decorator(registry: dict):
    def decorator(name: str, dim: int):
        def inner(fn):
            fn._feat_name = name
            fn._feat_dim  = dim
            registry[name] = fn
            return fn
        return inner
    return decorator


register_internal_feature = _make_decorator(_INTERNAL_REGISTRY)
register_external_feature = _make_decorator(_EXTERNAL_REGISTRY)
register_device_feature   = _make_decorator(_DEVICE_REGISTRY)


# ── Build entry point ─────────────────────────────────────────────────────────

def build_node_features(
    df: pd.DataFrame,
    node_type: str,
    node_to_id: dict[str, int],
    col_cfg: dict,
    train_mask: pd.Series,
    enabled: list[str] | None = None,
) -> torch.Tensor:
    """
    Build the feature matrix for one node type.

    Args:
        df:         full cleaned DataFrame
        node_type:  "internal_account" | "external_account" | "device"
        node_to_id: {raw_id: int_index} mapping for this node type
        col_cfg:    config["columns"]
        train_mask: boolean Series, True for training rows
        enabled:    feature names to include (None = all registered)

    Returns:
        torch.Tensor of shape (num_nodes, total_dim), z-score normalised
    """
    registry = dict(_REGISTRIES[node_type])
    if enabled is not None:
        registry = {k: registry[k] for k in enabled if k in registry}

    parts = []
    for name, fn in registry.items():
        result = fn(df, col_cfg, node_to_id, train_mask)
        if result is None:
            print(f"    skipped '{name}' (column missing)")
            continue
        parts.append(result)
        print(f"    {node_type} feature '{name}': {result.shape[1]} dims")

    if not parts:
        # Fallback: single zero feature so the node type isn't featureless
        n = len(node_to_id)
        return torch.zeros((n, 1), dtype=torch.float32)

    feat = np.concatenate(parts, axis=1).astype(np.float32)
    print(f"  {node_type}: {feat.shape[1]} total feature dims (before normalisation)")

    feat_tensor = torch.tensor(feat, dtype=torch.float32)

    # Compute normalisation stats from training-present nodes only.
    # Nodes unseen in training have all-zero features; including them would
    # bias the mean toward zero and underestimate std.
    if node_type == "internal_account":
        train_ids = set(df.loc[train_mask, "_sender"].astype(str).unique())
    elif node_type == "external_account":
        onus_col = col_cfg.get("onus_flag")
        train_df = df[train_mask]
        if onus_col and onus_col in train_df.columns:
            train_df = train_df[~train_df[onus_col]]
        train_ids = set(train_df["_receiver"].astype(str).unique())
    else:
        train_ids = set(node_to_id.keys())

    train_indices = [node_to_id[nid] for nid in train_ids if nid in node_to_id]
    if train_indices:
        train_feat = feat_tensor[train_indices]
        mean = train_feat.mean(dim=0)
        std  = train_feat.std(dim=0)
        std[std == 0] = 1.0
        return (feat_tensor - mean) / std
    else:
        return zscore_cols(feat_tensor)


# ── Shared helper ─────────────────────────────────────────────────────────────

def _place(series: pd.Series, node_to_id: dict, out: np.ndarray, col: int):
    """Place a {node_id: value} Series into the correct rows of `out`."""
    for node_id, val in series.items():
        if node_id in node_to_id:
            out[node_to_id[node_id], col] = float(val)


# ══════════════════════════════════════════════════════════════════════════════
# InternalAccount features
# ══════════════════════════════════════════════════════════════════════════════

@register_internal_feature("out_degree", dim=1)
def _out_degree(df, col_cfg, node_to_id, train_mask):
    """Number of transactions sent (training period)."""
    n   = len(node_to_id)
    out = np.zeros((n, 1), dtype=np.float32)
    _place(df[train_mask].groupby("_sender").size(), node_to_id, out, 0)
    return out


@register_internal_feature("amount_stats", dim=3)
def _amount_stats(df, col_cfg, node_to_id, train_mask):
    """mean_sent, std_sent, total_sent (log1p applied to amounts)."""
    val_col = col_cfg.get("value") or col_cfg.get("base_value")
    if not val_col or val_col not in df.columns:
        return None
    n        = len(node_to_id)
    out      = np.zeros((n, 3), dtype=np.float32)
    train_df = df[train_mask]
    grp      = train_df.groupby("_sender")[val_col]
    _place(np.log1p(grp.mean()),  node_to_id, out, 0)
    _place(np.log1p(grp.std().fillna(0)), node_to_id, out, 1)
    _place(np.log1p(grp.sum()),   node_to_id, out, 2)
    return out


@register_internal_feature("counterparty_diversity", dim=1)
def _counterparty_diversity(df, col_cfg, node_to_id, train_mask):
    """Number of unique receiver accounts (as sender)."""
    n   = len(node_to_id)
    out = np.zeros((n, 1), dtype=np.float32)
    _place(df[train_mask].groupby("_sender")["_receiver"].nunique(), node_to_id, out, 0)
    return out


@register_internal_feature("device_diversity", dim=1)
def _device_diversity(df, col_cfg, node_to_id, train_mask):
    """
    Number of unique devices used (as sender).
    High value = suspicious (multiple devices per account = money mule signal).
    """
    dev_col = col_cfg.get("device")
    if not dev_col or dev_col not in df.columns:
        return None
    n   = len(node_to_id)
    out = np.zeros((n, 1), dtype=np.float32)
    _place(df[train_mask].groupby("_sender")[dev_col].nunique(), node_to_id, out, 0)
    return out


@register_internal_feature("channel_diversity", dim=1)
def _channel_diversity(df, col_cfg, node_to_id, train_mask):
    """Number of unique channel OHE columns active per sender (proxy for channel variety)."""
    ch_cols = col_cfg.get("ohe_groups", {}).get("channel", [])
    ch_cols = [c.strip() for c in ch_cols if c.strip() in df.columns]
    if not ch_cols:
        return None
    n        = len(node_to_id)
    out      = np.zeros((n, 1), dtype=np.float32)
    train_df = df[train_mask]
    # Count how many distinct channel columns are ever 1 for each sender
    diversity = train_df.groupby("_sender")[ch_cols].max().sum(axis=1)
    _place(diversity, node_to_id, out, 0)
    return out


@register_internal_feature("time_behavior", dim=2)
def _time_behavior(df, col_cfg, node_to_id, train_mask):
    """
    night_ratio: fraction of transactions between 22:00–06:00.
    weekend_ratio: fraction on Saturday/Sunday.
    Both are AML signals (unusual timing for legitimate activity).
    """
    n        = len(node_to_id)
    out      = np.zeros((n, 2), dtype=np.float32)
    train_df = df[train_mask].copy()
    hour     = train_df["_datetime"].dt.hour
    dow      = train_df["_datetime"].dt.dayofweek  # 0=Mon, 6=Sun

    train_df["_night"]   = ((hour >= 22) | (hour < 6)).astype(float)
    train_df["_weekend"] = (dow >= 5).astype(float)

    grp = train_df.groupby("_sender")
    _place(grp["_night"].mean(),   node_to_id, out, 0)
    _place(grp["_weekend"].mean(), node_to_id, out, 1)
    return out


@register_internal_feature("customer_type_ohe", dim=4)
def _customer_type_ohe(df, col_cfg, node_to_id, train_mask):
    """
    One-hot encoding of CUSTOMERTYPE (top values from training data).
    dim=4: up to 4 distinct types; unknown/rare mapped to all-zeros.
    """
    ct_col = col_cfg.get("customer_type")
    if not ct_col or ct_col not in df.columns:
        return None

    n        = len(node_to_id)
    train_df = df[train_mask]

    # One row per account (take first occurrence — type shouldn't change)
    acct_type = (
        train_df.drop_duplicates("_sender")
        .set_index("_sender")[ct_col]
        .fillna("unknown")
    )

    # Vocab = top 4 by frequency in training data (excluding "unknown")
    top_types = (
        acct_type[acct_type != "unknown"]
        .value_counts()
        .head(4)
        .index.tolist()
    )
    if not top_types:
        return None

    out = np.zeros((n, len(top_types)), dtype=np.float32)
    for i, t in enumerate(top_types):
        for acc, val in acct_type.items():
            if acc in node_to_id and val == t:
                out[node_to_id[acc], i] = 1.0

    return out


# ══════════════════════════════════════════════════════════════════════════════
# ExternalAccount features
# ══════════════════════════════════════════════════════════════════════════════

@register_external_feature("in_degree", dim=1)
def _ext_in_degree(df, col_cfg, node_to_id, train_mask):
    """Number of transactions received from internal accounts (training)."""
    onus_col  = col_cfg.get("onus_flag")
    n         = len(node_to_id)
    out       = np.zeros((n, 1), dtype=np.float32)
    train_df  = df[train_mask]
    # External accounts only receive non-onus transactions
    if onus_col and onus_col in train_df.columns:
        train_df = train_df[~train_df[onus_col]]
    _place(train_df.groupby("_receiver").size(), node_to_id, out, 0)
    return out


@register_external_feature("received_amount_stats", dim=2)
def _ext_received_amount_stats(df, col_cfg, node_to_id, train_mask):
    """mean_received, std_received (log1p applied)."""
    val_col  = col_cfg.get("value") or col_cfg.get("base_value")
    onus_col = col_cfg.get("onus_flag")
    if not val_col or val_col not in df.columns:
        return None
    n        = len(node_to_id)
    out      = np.zeros((n, 2), dtype=np.float32)
    train_df = df[train_mask]
    if onus_col and onus_col in train_df.columns:
        train_df = train_df[~train_df[onus_col]]
    grp = train_df.groupby("_receiver")[val_col]
    _place(np.log1p(grp.mean()),              node_to_id, out, 0)
    _place(np.log1p(grp.std().fillna(0)),     node_to_id, out, 1)
    return out


@register_external_feature("sender_diversity", dim=1)
def _ext_sender_diversity(df, col_cfg, node_to_id, train_mask):
    """Number of unique internal accounts that sent money to this account."""
    onus_col = col_cfg.get("onus_flag")
    n        = len(node_to_id)
    out      = np.zeros((n, 1), dtype=np.float32)
    train_df = df[train_mask]
    if onus_col and onus_col in train_df.columns:
        train_df = train_df[~train_df[onus_col]]
    _place(train_df.groupby("_receiver")["_sender"].nunique(), node_to_id, out, 0)
    return out


@register_external_feature("sender_bank_diversity", dim=1)
def _ext_sender_bank_diversity(df, col_cfg, node_to_id, train_mask):
    """
    Number of unique Danske Bank entities (ACCOUNTAGENTID) that sent to this external account.
    High diversity = money flowing in from multiple bank branches/subsidiaries (suspicious pattern).
    """
    bank_col = col_cfg.get("sender_bank")  # ACCOUNTAGENTID — the Danske Bank entity that sent the money
    onus_col = col_cfg.get("onus_flag")
    if not bank_col or bank_col not in df.columns:
        return None
    n        = len(node_to_id)
    out      = np.zeros((n, 1), dtype=np.float32)
    train_df = df[train_mask]
    if onus_col and onus_col in train_df.columns:
        train_df = train_df[~train_df[onus_col]]
    _place(train_df.groupby("_receiver")[bank_col].nunique(), node_to_id, out, 0)
    return out


# ══════════════════════════════════════════════════════════════════════════════
# Device features (V3)
# ══════════════════════════════════════════════════════════════════════════════

@register_device_feature("device_tx_count", dim=1)
def _device_tx_count(df, col_cfg, node_to_id, train_mask):
    """Total transactions through this device (training)."""
    dev_col = col_cfg.get("device")
    if not dev_col or dev_col not in df.columns:
        return None
    n   = len(node_to_id)
    out = np.zeros((n, 1), dtype=np.float32)
    counts = df[train_mask].groupby(dev_col).size()
    counts.index = counts.index.astype(str)
    _place(counts, node_to_id, out, 0)
    return out


@register_device_feature("device_acct_diversity", dim=1)
def _device_acct_diversity(df, col_cfg, node_to_id, train_mask):
    """
    Number of unique accounts that used this device.
    Key AML signal: high value = shared device across accounts = money mule ring.
    """
    dev_col = col_cfg.get("device")
    if not dev_col or dev_col not in df.columns:
        return None
    n   = len(node_to_id)
    out = np.zeros((n, 1), dtype=np.float32)
    div = df[train_mask].groupby(dev_col)["_sender"].nunique()
    div.index = div.index.astype(str)
    _place(div, node_to_id, out, 0)
    return out
