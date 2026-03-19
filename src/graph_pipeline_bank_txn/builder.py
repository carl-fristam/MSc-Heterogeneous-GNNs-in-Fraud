"""
Graph builder — transactions-as-nodes variant.

Each row in the dataset becomes a Transaction node.
Accounts are InternalAccount or ExternalAccount nodes.
Edges encode the sender→transaction and transaction→receiver relationships.

Can be called standalone or with a PreparedData object.

Entry point:
    # Standalone
    from src.graph_pipeline_bank_txn import build_graph
    result = build_graph(load_config("graph_bank_txn_v1"))

    # Shared data (recommended)
    from src.data.prepare import prepare_data
    prep = prepare_data(config)
    result = build_graph(config, prep=prep)
"""

import hashlib
import json
import os
import pickle

import numpy as np
import torch
from torch_geometric.data import HeteroData

from src.utils.config import PROJECT_ROOT
from src.graph_pipeline_bank.loader import load_raw
from src.graph_pipeline_bank.node_builder import build_node_maps
from src.graph_pipeline_bank.features_node import build_node_features
from src.graph_pipeline_bank.features_edge import build_edge_features, fit_vocabs
from src.utils.split import temporal_split, random_stratified_split


def build_graph(config: dict, prep=None) -> dict:
    """
    Build a PyG HeteroData graph with transactions as nodes.

    Args:
        config: dict loaded from configs/graph_bank_txn_v1.yaml
        prep: optional PreparedData instance

    Returns:
        dict with "data" (HeteroData), "node_maps", "vocabs"
    """
    # ── Cache check ───────────────────────────────────────────────────────────
    cache_path = _cache_path(config)
    if config.get("cache", {}).get("enabled", True):
        cached = _load_cache(cache_path)
        if cached is not None:
            return cached

    variant = config.get("variant", "txn_v1")
    print(f"\n{'='*60}")
    print(f"Building bank graph (txn-as-node)  |  variant={variant}")
    print(f"{'='*60}")

    # ── Load & split (reuse PreparedData if available) ────────────────────────
    if prep is not None:
        df = prep.df.reset_index(drop=True)
        train_mask = prep.train_mask.reset_index(drop=True)
        val_mask = prep.val_mask.reset_index(drop=True)
        test_mask = prep.test_mask.reset_index(drop=True)
        vocabs = prep.vocabs
    else:
        data_path = str(PROJECT_ROOT / config["data_path"])
        df = load_raw(data_path, config)

        split_cfg = config["split"]
        col_cfg = config["columns"]

        if split_cfg.get("method", "temporal") == "temporal":
            train_mask, val_mask, test_mask = temporal_split(
                df, train_end=split_cfg["train_end"], val_end=split_cfg["val_end"],
            )
        else:
            train_mask, val_mask, test_mask = random_stratified_split(
                df, label_col=col_cfg["label"],
                train_ratio=split_cfg.get("train_ratio", 0.7),
                val_ratio=split_cfg.get("val_ratio", 0.15),
                seed=split_cfg.get("seed", 42),
            )

        df = df.reset_index(drop=True)
        train_mask = train_mask.reset_index(drop=True)
        val_mask = val_mask.reset_index(drop=True)
        test_mask = test_mask.reset_index(drop=True)
        vocabs = fit_vocabs(df, config["columns"], train_mask)

    col_cfg = config["columns"]
    n_txn = len(df)
    txn_idx = np.arange(n_txn, dtype=np.int64)

    # ── Account node mappings ─────────────────────────────────────────────────
    print("\nBuilding account node mappings...")
    node_maps = build_node_maps(df, config)

    # ── Account node features ─────────────────────────────────────────────────
    print("\nBuilding account node features (train-only aggregation)...")
    node_features: dict[str, torch.Tensor] = {}
    for node_type, node_cfg in config["nodes"].items():
        if node_type == "transaction":
            continue
        enabled = node_cfg.get("features", None)
        node_features[node_type] = build_node_features(
            df=df,
            node_type=node_type,
            node_to_id=node_maps[node_type],
            col_cfg=col_cfg,
            train_mask=train_mask,
            enabled=enabled,
        )

    # ── Transaction node features ─────────────────────────────────────────────
    print("\nBuilding transaction node features...")
    txn_feat_cfg = config["nodes"].get("transaction", {})
    enabled_txn_feats = txn_feat_cfg.get("features", None)

    if prep is not None:
        # Reuse pre-computed features
        txn_x = torch.tensor(prep.txn_features, dtype=torch.float32)
    else:
        feat_arr = build_edge_features(df, col_cfg, vocabs, enabled=enabled_txn_feats)
        txn_x = torch.tensor(feat_arr, dtype=torch.float32)
    print(f"  transaction feature dim: {txn_x.shape[1]}")

    # ── Transaction labels and masks ──────────────────────────────────────────
    label_col = col_cfg.get("label")
    if prep is not None:
        txn_y = torch.tensor(prep.labels, dtype=torch.float32)
    else:
        txn_y = torch.tensor(
            df[label_col].fillna(0).values, dtype=torch.float32
        )
    txn_train = torch.tensor(train_mask.values, dtype=torch.bool)
    txn_val = torch.tensor(val_mask.values, dtype=torch.bool)
    txn_test = torch.tensor(test_mask.values, dtype=torch.bool)

    pos_total = int(txn_y.sum().item())
    print(f"\n  Transaction nodes: {n_txn:,}  |  fraud: {pos_total:,} ({100*pos_total/n_txn:.3f}%)")
    for split_name, mask in [("train", txn_train), ("val", txn_val), ("test", txn_test)]:
        n = int(mask.sum().item())
        pos = int(txn_y[mask].sum().item())
        print(f"  {split_name:5s}: {n:,}  pos={pos:,} ({100*pos/n:.3f}%)" if n > 0 else f"  {split_name:5s}: 0")

    # ── Build edge indices ────────────────────────────────────────────────────
    print("\nBuilding edge indices...")

    sender_col = col_cfg["sender"]
    receiver_col = col_cfg["receiver"]
    onus_col = col_cfg["onus_flag"]

    int_map = node_maps["internal_account"]
    ext_map = node_maps["external_account"]

    sender_int_idx = df[sender_col].astype(str).map(int_map).values.astype(np.int64)

    # 1. sends: internal_account → transaction
    sends_edge_index = torch.tensor(
        np.stack([sender_int_idx, txn_idx]), dtype=torch.long
    )
    print(f"  sends:                {sends_edge_index.shape[1]:,} edges")

    # 2. received_by_internal: transaction → internal_account (on-us only)
    onus_mask = df[onus_col].astype(bool).values
    onus_rows = np.where(onus_mask)[0]
    recv_int_ids = df.loc[onus_rows, receiver_col].astype(str).map(int_map)
    valid_int = recv_int_ids.notna().values
    recv_int_src = onus_rows[valid_int]
    recv_int_dst = recv_int_ids.dropna().values.astype(np.int64)
    recv_internal_edge_index = torch.tensor(
        np.stack([recv_int_src, recv_int_dst]), dtype=torch.long
    )
    print(f"  received_by_internal: {recv_internal_edge_index.shape[1]:,} edges")

    # 3. received_by_external: transaction → external_account (non-onus only)
    ext_mask = ~onus_mask
    ext_rows = np.where(ext_mask)[0]
    recv_ext_ids = df.loc[ext_rows, receiver_col].astype(str).map(ext_map)
    valid_ext = recv_ext_ids.notna().values
    recv_ext_src = ext_rows[valid_ext]
    recv_ext_dst = recv_ext_ids.dropna().values.astype(np.int64)
    recv_external_edge_index = torch.tensor(
        np.stack([recv_ext_src, recv_ext_dst]), dtype=torch.long
    )
    print(f"  received_by_external: {recv_external_edge_index.shape[1]:,} edges")

    # ── Assemble HeteroData ───────────────────────────────────────────────────
    data = HeteroData()

    data["transaction"].x = txn_x
    data["transaction"].y = txn_y
    data["transaction"].train_mask = txn_train
    data["transaction"].val_mask = txn_val
    data["transaction"].test_mask = txn_test
    data["transaction"].num_nodes = n_txn

    for node_type, feat in node_features.items():
        data[node_type].x = feat
        data[node_type].num_nodes = len(node_maps[node_type])

    data[("internal_account", "sends", "transaction")].edge_index = sends_edge_index
    data[("transaction", "received_by_internal", "internal_account")].edge_index = recv_internal_edge_index
    data[("transaction", "received_by_external", "external_account")].edge_index = recv_external_edge_index

    _print_summary(data)

    # ── Cache ─────────────────────────────────────────────────────────────────
    result = {"data": data, "node_maps": node_maps, "vocabs": vocabs}
    if config.get("cache", {}).get("enabled", True):
        _save_cache(cache_path, result)

    return result


# ── Cache helpers ──────────────────────────────────────────────────────────────

def _cache_path(config: dict) -> str:
    cache_dir = config.get("cache", {}).get("dir", "data/processed/bank")
    variant = config.get("variant", "txn_v1")
    sr = config.get("sample_ratio", 1.0)
    sig = json.dumps({
        "variant": variant,
        "split": config.get("split", {}),
        "nodes": {k: v.get("features") for k, v in config.get("nodes", {}).items()},
    }, sort_keys=True)
    h = hashlib.md5(sig.encode()).hexdigest()[:6]
    return str(PROJECT_ROOT / cache_dir / f"graph_bank_{variant}_sr{sr:.2f}_{h}.pkl")


def _load_cache(path: str):
    if os.path.exists(path):
        print(f"Loading from cache: {path}")
        with open(path, "rb") as f:
            return pickle.load(f)
    return None


def _save_cache(path: str, data: dict):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(data, f)
    print(f"\nSaved to cache: {path}")


# ── Summary ────────────────────────────────────────────────────────────────────

def _print_summary(data: HeteroData):
    print(f"\n{'='*60}")
    print("Graph Summary (txn-as-node)")
    print(f"{'='*60}")

    for nt in data.node_types:
        n = data[nt].num_nodes
        fdim = data[nt].x.shape[1] if hasattr(data[nt], "x") and data[nt].x is not None else 0
        print(f"  {nt:<25} {n:>10,} nodes   feat_dim={fdim}")
        if hasattr(data[nt], "y") and data[nt].y is not None:
            y = data[nt].y
            for split in ["train", "val", "test"]:
                attr = f"{split}_mask"
                if hasattr(data[nt], attr):
                    mask = data[nt][attr]
                    ns = mask.sum().item()
                    if ns > 0:
                        pos = y[mask].sum().item()
                        print(f"    {split:5s}: {ns:>8,}  pos={int(pos):>6} ({100*pos/ns:.2f}%)")

    print()
    for et in data.edge_types:
        n = data[et].edge_index.shape[1]
        print(f"  {str(et):<65} {n:>8,} edges")
