"""
Build heterogeneous graph from bank payment dataset.

Graph design:
  - Two node types: InternalAccount (ACCOUNTID) and ExternalAccount (COUNTERENTITYID not in ACCOUNTID)
  - Transactions are directed EDGES (edge classification)
  - Two edge relation types:
      ('internal_account', 'onus_transfer',    'internal_account')  — TRANSACTIONONUS=True
      ('internal_account', 'external_transfer', 'external_account') — TRANSACTIONONUS=False
  - Labels (CONFIRMEDRISK) and train/val/test masks live on edges
  - Node features: rich for internal accounts, sparse for external
  - Edge features: amount, currency, channel, method, flags, time encodings

Output: PyG HeteroData saved to OUTPUT_PATH

Usage:
    python scripts/build_graph_bank.py
"""

# ─────────────────────────────────────────────────────────────
# CONFIG — edit these paths before running
# ─────────────────────────────────────────────────────────────
DATA_PATH   = "datasets/bank_transactions.parquet"  # or .csv
OUTPUT_PATH = "data/processed/graph_bank_hetero.pt"
SAMPLE      = 1.0        # fraction of rows to load (0.01 for dev)
TRAIN_END   = "2023-04-01"
VAL_END     = "2023-06-01"
# ─────────────────────────────────────────────────────────────

import os, sys
from pathlib import Path

# Ensure project root is on path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import HeteroData


# ── Columns ──────────────────────────────────────────────────

COL_SENDER      = "ACCOUNTID"
COL_RECEIVER    = "COUNTERENTITYID"
COL_ONUS        = "TRANSACTIONONUS"      # bool: True = on-us (internal-internal)
COL_TIME        = "EVENTTIME"
COL_LABEL       = "CONFIRMEDRISK"
COL_VALUE       = "VALUE"
COL_BASEVALUE   = "BASEVALUE"
COL_CURRENCY    = "CURRENCY"
COL_BASECURRENCY= "BASECURRENCY"
COL_CHANNEL     = "CHANNEL"
COL_METHOD      = "PAYMENTMETHOD"
COL_SUBMETHOD   = "PAYMENTSUBMETHOD"
COL_INTL        = "INTERNATIONALFLAG"
COL_DESTCOUNTRY = "DESTINATIONCOUNTRY"
COL_CUSTOMER    = "CUSTOMERID"
COL_CUSTTYPE    = "CUSTOMERTYPE"
COL_SENDER_BANK = "ACCOUNTAGENTID"
COL_RECV_BANK   = "COUNTERAGENTID"
COL_DEVICE      = "DEVICEID"


# ── Load ──────────────────────────────────────────────────────

def load_data(path: str, sample: float) -> pd.DataFrame:
    path = str(PROJECT_ROOT / path)
    print(f"Loading {path} ...")
    if path.endswith(".parquet"):
        df = pd.read_parquet(path)
    else:
        df = pd.read_csv(path, low_memory=False)

    if sample < 1.0:
        df = df.sample(frac=sample, random_state=42).reset_index(drop=True)
        print(f"  Sampled {sample:.0%} → {len(df):,} rows")
    else:
        print(f"  Loaded {len(df):,} rows")

    # Parse datetime
    df[COL_TIME] = pd.to_datetime(df[COL_TIME], errors="coerce")
    n_bad = df[COL_TIME].isna().sum()
    if n_bad:
        print(f"  Warning: {n_bad:,} rows have unparseable {COL_TIME} — dropping")
        df = df.dropna(subset=[COL_TIME]).reset_index(drop=True)

    # Ensure TRANSACTIONONUS is bool
    if COL_ONUS in df.columns:
        df[COL_ONUS] = df[COL_ONUS].astype(str).str.lower().map(
            {"true": True, "false": False, "1": True, "0": False}
        ).fillna(False).astype(bool)

    return df


# ── Temporal split ─────────────────────────────────────────────

def temporal_split(df: pd.DataFrame):
    t_train = pd.Timestamp(TRAIN_END)
    t_val   = pd.Timestamp(VAL_END)
    times   = df[COL_TIME]
    train_mask = times < t_train
    val_mask   = (times >= t_train) & (times < t_val)
    test_mask  = times >= t_val
    print(f"\nSplit:")
    for name, mask in [("train", train_mask), ("val", val_mask), ("test", test_mask)]:
        n = mask.sum()
        pos = df.loc[mask, COL_LABEL].sum() if COL_LABEL in df.columns else 0
        print(f"  {name:5s}: {n:,} edges,  {int(pos)} positive ({100*pos/n:.2f}%)" if n > 0 else f"  {name}: empty")
    return train_mask, val_mask, test_mask


# ── Node mappings ─────────────────────────────────────────────

def build_node_mappings(df: pd.DataFrame):
    """
    InternalAccount = all unique ACCOUNTID values.
    ExternalAccount = COUNTERENTITYID values that never appear as ACCOUNTID.
    On-us transactions (TRANSACTIONONUS=True) link InternalAccount → InternalAccount.
    External transactions link InternalAccount → ExternalAccount.
    """
    internal_ids = set(df[COL_SENDER].unique())

    # For on-us transactions, the receiver is also an internal account
    onus_receivers = set(df.loc[df[COL_ONUS], COL_RECEIVER].unique()) if COL_ONUS in df.columns else set()
    all_internal = internal_ids | onus_receivers

    all_receivers = set(df[COL_RECEIVER].unique())
    external_ids  = all_receivers - all_internal

    internal_to_id = {acc: i for i, acc in enumerate(sorted(all_internal))}
    external_to_id = {acc: i for i, acc in enumerate(sorted(external_ids))}

    print(f"\nNode mapping:")
    print(f"  InternalAccount: {len(internal_to_id):,}")
    print(f"  ExternalAccount: {len(external_to_id):,}")
    print(f"  On-us receivers promoted to internal: {len(onus_receivers - internal_ids):,}")

    return internal_to_id, external_to_id


# ── Node features ─────────────────────────────────────────────

def _zscore(arr: np.ndarray) -> np.ndarray:
    mu, sigma = arr.mean(axis=0), arr.std(axis=0)
    sigma = np.where(sigma == 0, 1.0, sigma)
    return (arr - mu) / sigma


def _one_hot(series: pd.Series, vocab: list) -> np.ndarray:
    return np.stack([
        (series == v).astype(np.float32).values for v in vocab
    ], axis=1)


def build_internal_features(df: pd.DataFrame, internal_to_id: dict, train_mask: pd.Series) -> torch.Tensor:
    """
    Aggregate features for InternalAccount nodes from training data only.
    Features: tx count, mean/std amount, fraud rate, customer type OHE, unique devices.
    """
    n = len(internal_to_id)
    train_df = df[train_mask].copy()

    # Per-sender aggregations (training only)
    grp = train_df.groupby(COL_SENDER)

    count     = grp[COL_VALUE].count().rename("count")
    amt_mean  = grp[COL_VALUE].mean().rename("amt_mean")
    amt_std   = grp[COL_VALUE].std().fillna(0).rename("amt_std")
    fraud_rate= grp[COL_LABEL].mean().rename("fraud_rate") if COL_LABEL in train_df.columns else None

    stats = pd.concat([count, amt_mean, amt_std], axis=1)
    if fraud_rate is not None:
        stats = pd.concat([stats, fraud_rate], axis=1)
    stats = stats.reindex(list(internal_to_id.keys())).fillna(0)

    # Numeric block
    numeric = _zscore(stats.values.astype(np.float32))

    # Customer type OHE (optional)
    extras = []
    if COL_CUSTTYPE in train_df.columns:
        ctype = train_df.drop_duplicates(COL_SENDER).set_index(COL_SENDER)[COL_CUSTTYPE]
        ctype = ctype.reindex(list(internal_to_id.keys())).fillna("unknown")
        vocab = [v for v in ctype.value_counts().index if v != "unknown"][:8]
        ohe   = _one_hot(ctype, vocab)
        extras.append(ohe)

    # Device diversity
    if COL_DEVICE in train_df.columns:
        dev_diversity = train_df.groupby(COL_SENDER)[COL_DEVICE].nunique().reindex(
            list(internal_to_id.keys())
        ).fillna(0).values.reshape(-1, 1).astype(np.float32)
        dev_diversity = _zscore(dev_diversity)
        extras.append(dev_diversity)

    parts = [numeric] + extras
    feat  = np.concatenate(parts, axis=1).astype(np.float32)

    print(f"  InternalAccount features: {feat.shape}")
    return torch.tensor(feat, dtype=torch.float)


def build_external_features(df: pd.DataFrame, external_to_id: dict, train_mask: pd.Series) -> torch.Tensor:
    """
    Sparse features for ExternalAccount nodes (only appear as receivers).
    Features: tx received count, mean received amount, unique sender banks.
    """
    n = len(external_to_id)
    train_df = df[train_mask & ~df[COL_ONUS]] if COL_ONUS in df.columns else df[train_mask]

    grp = train_df.groupby(COL_RECEIVER)

    count    = grp[COL_VALUE].count().rename("count")
    amt_mean = grp[COL_VALUE].mean().rename("amt_mean")
    stats    = pd.concat([count, amt_mean], axis=1)

    if COL_RECV_BANK in train_df.columns:
        bank_div = train_df.groupby(COL_RECEIVER)[COL_RECV_BANK].nunique().rename("bank_diversity")
        stats    = pd.concat([stats, bank_div], axis=1)

    stats = stats.reindex(list(external_to_id.keys())).fillna(0)
    feat  = _zscore(stats.values.astype(np.float32))

    print(f"  ExternalAccount features: {feat.shape}")
    return torch.tensor(feat, dtype=torch.float)


# ── Edge features ─────────────────────────────────────────────

CHANNEL_VOCAB   = ["mobile", "internet", "branch", "atm", "api", "pos"]
METHOD_VOCAB    = ["sepa", "swift", "domestic", "instant", "standing_order", "direct_debit"]
SUBMETHOD_VOCAB = ["realTime", "batch", "manual"]

def build_edge_features(df_subset: pd.DataFrame) -> np.ndarray:
    """Build edge feature matrix for a subset of transactions."""
    parts = []

    # Log-amount
    val = np.log1p(df_subset[COL_VALUE].fillna(0).values).reshape(-1, 1).astype(np.float32)
    parts.append(val)

    if COL_BASEVALUE in df_subset.columns:
        bval = np.log1p(df_subset[COL_BASEVALUE].fillna(0).values).reshape(-1, 1).astype(np.float32)
        parts.append(bval)

    # Currency mismatch flag
    if COL_CURRENCY in df_subset.columns and COL_BASECURRENCY in df_subset.columns:
        mismatch = (df_subset[COL_CURRENCY] != df_subset[COL_BASECURRENCY]).astype(np.float32).values.reshape(-1, 1)
        parts.append(mismatch)

    # Channel OHE
    if COL_CHANNEL in df_subset.columns:
        ch = df_subset[COL_CHANNEL].str.lower().fillna("unknown")
        parts.append(_one_hot(ch, CHANNEL_VOCAB).astype(np.float32))

    # Payment method OHE
    if COL_METHOD in df_subset.columns:
        pm = df_subset[COL_METHOD].str.lower().fillna("unknown")
        parts.append(_one_hot(pm, METHOD_VOCAB).astype(np.float32))

    # Payment submethod OHE
    if COL_SUBMETHOD in df_subset.columns:
        psm = df_subset[COL_SUBMETHOD].fillna("unknown")
        parts.append(_one_hot(psm, SUBMETHOD_VOCAB).astype(np.float32))

    # International flag
    if COL_INTL in df_subset.columns:
        intl = df_subset[COL_INTL].fillna(0).astype(np.float32).values.reshape(-1, 1)
        parts.append(intl)

    # Time-of-day and day-of-week (sin/cos encoding)
    t = df_subset[COL_TIME]
    tod = t.dt.hour / 24.0 * 2 * np.pi
    dow = t.dt.dayofweek / 7.0 * 2 * np.pi
    time_enc = np.stack([
        np.sin(tod.values), np.cos(tod.values),
        np.sin(dow.values), np.cos(dow.values),
    ], axis=1).astype(np.float32)
    parts.append(time_enc)

    return np.concatenate(parts, axis=1)


# ── Main build ────────────────────────────────────────────────

def build_graph(df: pd.DataFrame, train_mask, val_mask, test_mask,
                internal_to_id, external_to_id) -> HeteroData:

    data = HeteroData()

    # ── Node features ──
    print("\nBuilding node features...")
    data["internal_account"].x = build_internal_features(df, internal_to_id, train_mask)
    data["internal_account"].num_nodes = len(internal_to_id)
    data["external_account"].x = build_external_features(df, external_to_id, train_mask)
    data["external_account"].num_nodes = len(external_to_id)

    # ── Edge types ──
    print("\nBuilding edges...")

    has_onus = COL_ONUS in df.columns
    if has_onus:
        onus_mask = df[COL_ONUS].values
        ext_mask  = ~onus_mask
    else:
        # Fallback: treat all as external
        onus_mask = np.zeros(len(df), dtype=bool)
        ext_mask  = np.ones(len(df), dtype=bool)
        print("  Warning: TRANSACTIONONUS not found — treating all edges as external_transfer")

    def _make_edge_type(row_mask, src_col, dst_col, src_map, dst_map, rel_name):
        idx   = np.where(row_mask)[0]
        sub   = df.iloc[idx]
        src   = sub[src_col].map(src_map).values.astype(np.int64)
        dst   = sub[dst_col].map(dst_map).values.astype(np.int64)

        # Drop rows where mapping failed (account not in map)
        valid = ~(np.isnan(src.astype(float)) | np.isnan(dst.astype(float)))
        if not valid.all():
            print(f"    {rel_name}: dropped {(~valid).sum():,} rows with unknown accounts")
            idx, src, dst = idx[valid], src[valid], dst[valid]
            sub = df.iloc[idx]

        edge_index = torch.tensor(np.stack([src, dst]), dtype=torch.long)

        # Edge features
        edge_feat = build_edge_features(sub)

        # Labels
        if COL_LABEL in sub.columns:
            edge_label = torch.tensor(sub[COL_LABEL].values, dtype=torch.float)
        else:
            edge_label = torch.zeros(len(sub), dtype=torch.float)

        # Masks (index into this edge type's edges)
        row_idx_set = set(idx.tolist())
        sub_train = torch.tensor([i in row_idx_set and train_mask.iloc[i] for i in idx])
        sub_val   = torch.tensor([i in row_idx_set and val_mask.iloc[i]   for i in idx])
        sub_test  = torch.tensor([i in row_idx_set and test_mask.iloc[i]  for i in idx])

        print(f"    ({rel_name}): {edge_index.shape[1]:,} edges, feat_dim={edge_feat.shape[1]}")
        return edge_index, edge_feat, edge_label, sub_train, sub_val, sub_test

    # On-us: internal → internal
    onus_idx = np.where(onus_mask)[0]
    onus_sub  = df.iloc[onus_idx]
    # Receivers for on-us should be in internal_to_id (they were promoted above)
    onus_ei, onus_feat, onus_y, onus_tr, onus_val, onus_te = _make_edge_type(
        onus_mask, COL_SENDER, COL_RECEIVER,
        internal_to_id, internal_to_id, "onus_transfer"
    )
    et_onus = ("internal_account", "onus_transfer", "internal_account")
    data[et_onus].edge_index  = onus_ei
    data[et_onus].edge_attr   = torch.tensor(onus_feat, dtype=torch.float)
    data[et_onus].y           = onus_y
    data[et_onus].train_mask  = onus_tr
    data[et_onus].val_mask    = onus_val
    data[et_onus].test_mask   = onus_te

    # External: internal → external
    ext_ei, ext_feat, ext_y, ext_tr, ext_val, ext_te = _make_edge_type(
        ext_mask, COL_SENDER, COL_RECEIVER,
        internal_to_id, external_to_id, "external_transfer"
    )
    et_ext = ("internal_account", "external_transfer", "external_account")
    data[et_ext].edge_index  = ext_ei
    data[et_ext].edge_attr   = torch.tensor(ext_feat, dtype=torch.float)
    data[et_ext].y           = ext_y
    data[et_ext].train_mask  = ext_tr
    data[et_ext].val_mask    = ext_val
    data[et_ext].test_mask   = ext_te

    return data


def _print_summary(data: HeteroData):
    print(f"\n{'='*60}")
    print("Graph Summary")
    print(f"{'='*60}")
    for nt in data.node_types:
        print(f"  {nt}: {data[nt].num_nodes:,} nodes, feat_dim={data[nt].x.shape[1]}")
    for et in data.edge_types:
        n = data[et].edge_index.shape[1]
        feat_dim = data[et].edge_attr.shape[1] if hasattr(data[et], 'edge_attr') else 0
        print(f"  {et}: {n:,} edges, feat_dim={feat_dim}")
        if hasattr(data[et], 'y'):
            y = data[et].y
            for split in ["train", "val", "test"]:
                mask = data[et][f"{split}_mask"]
                ns   = mask.sum().item()
                if ns > 0:
                    pos = y[mask].sum().item()
                    print(f"    {split:5s}: {ns:,}  pos={int(pos)} ({100*pos/ns:.2f}%)")


def main():
    df = load_data(DATA_PATH, SAMPLE)
    train_mask, val_mask, test_mask = temporal_split(df)
    internal_to_id, external_to_id = build_node_mappings(df)
    data = build_graph(df, train_mask, val_mask, test_mask, internal_to_id, external_to_id)
    _print_summary(data)

    out = PROJECT_ROOT / OUTPUT_PATH
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"data": data, "internal_to_id": internal_to_id, "external_to_id": external_to_id}, str(out))
    print(f"\nSaved → {out}")


if __name__ == "__main__":
    main()
