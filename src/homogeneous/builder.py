"""
Homogeneous graph builder for the bank dataset.

Collapses all accounts into a single node type and all transactions into a
single edge type. This is the L2 baseline: graph structure without
heterogeneous typing.

Two modes supported:
  - node: transactions are nodes with labels (node classification)
  - edge: transactions are edges with labels (edge classification)

Usage:
    from src.data.prepare import prepare_data
    from src.homogeneous.builder import build_homogeneous_graph

    prep = prepare_data(config)
    result = build_homogeneous_graph(prep, mode="node")
    data = result["data"]  # PyG Data
"""

import hashlib
import json
import os
import pickle

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data

from src.utils.config import PROJECT_ROOT
from src.graph_pipeline_bank.normalize import zscore_cols
from src.data.prepare import PreparedData


def _cache_path(config: dict, mode: str) -> str:
    cache_dir = config.get("cache", {}).get("dir", "data/processed/bank")
    sr = config.get("sample_ratio", 1.0)
    return str(PROJECT_ROOT / cache_dir / f"graph_homo_{mode}_sr{sr:.2f}.pkl")


def build_homogeneous_graph(prep: PreparedData, mode: str = "node", config: dict = None) -> dict:
    """
    Build a homogeneous PyG Data object from PreparedData.

    Args:
        prep: PreparedData instance (shared across all levels)
        mode: "node" (transactions as nodes) or "edge" (transactions as edges)
        config: optional config dict (used for cache path)

    Returns:
        dict with "data" (PyG Data), "node_map"
    """
    if config is not None and config.get("cache", {}).get("enabled", True):
        cp = _cache_path(config, mode)
        if os.path.exists(cp):
            print(f"Loading from cache: {cp}")
            with open(cp, "rb") as f:
                return pickle.load(f)

    df = prep.df
    col_cfg = prep.col_cfg

    print(f"\n{'='*60}")
    print(f"Building HOMOGENEOUS graph  |  mode={mode}")
    print(f"{'='*60}")

    # ── Unified account mapping (all senders + all receivers → one pool) ──────
    sender_col = col_cfg["sender"]
    receiver_col = col_cfg["receiver"]

    all_accounts = pd.concat([
        df[sender_col].astype(str),
        df[receiver_col].astype(str),
    ]).unique()
    node_map = {acc: i for i, acc in enumerate(sorted(all_accounts))}
    n_accounts = len(node_map)
    print(f"  Accounts (unified): {n_accounts:,}")

    # ── Account node features (train-only aggregation) ────────────────────────
    train_df = df[prep.train_mask]
    value_col = col_cfg["value"]

    out_deg = train_df.groupby(sender_col).size()
    in_deg = train_df.groupby(receiver_col).size()
    mean_sent = train_df.groupby(sender_col)[value_col].mean()
    mean_recv = train_df.groupby(receiver_col)[value_col].mean()
    std_sent = train_df.groupby(sender_col)[value_col].std().fillna(0)
    unique_cpty = train_df.groupby(sender_col)[receiver_col].nunique()

    feat_dim = 6
    node_feats = np.zeros((n_accounts, feat_dim), dtype=np.float32)
    for acc, idx in node_map.items():
        node_feats[idx] = [
            np.log1p(out_deg.get(acc, 0)),
            np.log1p(in_deg.get(acc, 0)),
            np.log1p(mean_sent.get(acc, 0)),
            np.log1p(mean_recv.get(acc, 0)),
            np.log1p(std_sent.get(acc, 0)),
            np.log1p(unique_cpty.get(acc, 0)),
        ]

    node_x = zscore_cols(torch.tensor(node_feats, dtype=torch.float32))
    print(f"  Node feature dim: {node_x.shape[1]}")
    print(f"  Transaction feature dim: {prep.txn_features.shape[1]}")

    # ── Build graph ───────────────────────────────────────────────────────────
    sender_idx = df[sender_col].astype(str).map(node_map).values.astype(np.int64)
    receiver_idx = df[receiver_col].astype(str).map(node_map).values.astype(np.int64)

    train_m = torch.tensor(prep.train_mask.values, dtype=torch.bool)
    val_m = torch.tensor(prep.val_mask.values, dtype=torch.bool)
    test_m = torch.tensor(prep.test_mask.values, dtype=torch.bool)

    if mode == "node":
        result = _build_node_mode(
            node_x, prep.txn_features, sender_idx, receiver_idx,
            prep.labels, train_m, val_m, test_m, n_accounts,
        )
    else:
        result = _build_edge_mode(
            node_x, prep.txn_features, sender_idx, receiver_idx,
            prep.labels, train_m, val_m, test_m, n_accounts,
        )

    result["node_map"] = node_map
    _print_summary(result["data"], mode)

    if config is not None and config.get("cache", {}).get("enabled", True):
        cp = _cache_path(config, mode)
        os.makedirs(os.path.dirname(cp), exist_ok=True)
        with open(cp, "wb") as f:
            pickle.dump(result, f)
        print(f"\nSaved to cache: {cp}")

    return result


def _build_node_mode(node_x, txn_feats, sender_idx, receiver_idx,
                     labels, train_m, val_m, test_m, n_accounts):
    """Transactions as nodes. Account→Transaction and Transaction→Account edges."""
    n_txn = len(labels)
    txn_offset = n_accounts

    txn_x = torch.tensor(txn_feats, dtype=torch.float32)
    max_dim = max(node_x.shape[1], txn_x.shape[1])
    if node_x.shape[1] < max_dim:
        node_x = torch.cat([node_x, torch.zeros(node_x.shape[0], max_dim - node_x.shape[1])], dim=1)
    if txn_x.shape[1] < max_dim:
        txn_x = torch.cat([txn_x, torch.zeros(txn_x.shape[0], max_dim - txn_x.shape[1])], dim=1)

    x = torch.cat([node_x, txn_x], dim=0)

    # Edges: sender → txn, txn → receiver
    txn_nodes = np.arange(n_txn) + txn_offset
    src = np.concatenate([sender_idx, txn_nodes])
    dst = np.concatenate([txn_nodes, receiver_idx])
    edge_index = torch.tensor(np.stack([src, dst]), dtype=torch.long)

    y = torch.full((x.shape[0],), -1, dtype=torch.float32)
    y[txn_offset:] = torch.tensor(labels)

    full_train = torch.zeros(x.shape[0], dtype=torch.bool)
    full_val = torch.zeros(x.shape[0], dtype=torch.bool)
    full_test = torch.zeros(x.shape[0], dtype=torch.bool)
    full_train[txn_offset:] = train_m
    full_val[txn_offset:] = val_m
    full_test[txn_offset:] = test_m

    data = Data(x=x, edge_index=edge_index, y=y)
    data.train_mask = full_train
    data.val_mask = full_val
    data.test_mask = full_test
    data.num_account_nodes = n_accounts
    data.num_txn_nodes = n_txn

    return {"data": data}


def _build_edge_mode(node_x, txn_feats, sender_idx, receiver_idx,
                     labels, train_m, val_m, test_m, n_accounts):
    """Transactions as edges between account nodes."""
    edge_index = torch.tensor(
        np.stack([sender_idx, receiver_idx]), dtype=torch.long
    )
    edge_attr = torch.tensor(txn_feats, dtype=torch.float32)

    data = Data(x=node_x, edge_index=edge_index, edge_attr=edge_attr)
    data.edge_y = torch.tensor(labels, dtype=torch.float32)
    data.edge_train_mask = train_m
    data.edge_val_mask = val_m
    data.edge_test_mask = test_m

    return {"data": data}


def _print_summary(data, mode):
    print(f"\n{'='*60}")
    print(f"Homogeneous Graph Summary (mode={mode})")
    print(f"{'='*60}")
    print(f"  Nodes: {data.num_nodes:,}  |  feat_dim={data.x.shape[1]}")
    print(f"  Edges: {data.edge_index.shape[1]:,}")
    if mode == "node":
        y = data.y
        for name, mask in [("train", data.train_mask), ("val", data.val_mask), ("test", data.test_mask)]:
            n = mask.sum().item()
            if n > 0:
                pos = int(y[mask].clamp(min=0).sum().item())
                print(f"  {name:5s}: {n:,}  pos={pos}")
    else:
        y = data.edge_y
        for name, mask in [("train", data.edge_train_mask), ("val", data.edge_val_mask), ("test", data.edge_test_mask)]:
            n = mask.sum().item()
            if n > 0:
                pos = int(y[mask].sum().item())
                print(f"  {name:5s}: {n:,}  pos={pos}")
