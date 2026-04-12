"""
assembler.py

Orchestrates the full graph build and caches the result.

Steps:
    1. Build node ID mappings
    2. Compute node features (training data only)
    3. Fit edge normalization stats (training data only)
    4. Build edges (index + features + labels + masks) per edge type
    5. Assemble into a PyG HeteroData object
    6. Cache to disk so we don't rebuild every run

The cache is keyed by a hash of the config so any change to features,
filters, or split dates automatically triggers a rebuild.
"""

import hashlib
import json
import os
import pickle
from pathlib import Path

import torch
from torch_geometric.data import HeteroData

from src.utils.config import PROJECT_ROOT
from src.graph_builder.node_builder  import build_node_maps
from src.graph_builder.node_features import build_internal_node_features, build_external_node_features
from src.graph_builder.edge_features import fit_edge_stats
from src.graph_builder.edge_builder  import build_all_edges


def build_graph(config: dict, prep) -> dict:
    """
    Build the heterogeneous graph for a given variant.

    Args:
        config: full config dict from master.yaml (with variant already selected)
        prep:   PreparedData object — provides df, masks (avoids reloading data)

    Returns:
        dict with:
            "data"      — PyG HeteroData object
            "node_maps" — {node_type: {acc_id: int_index}}
    """
    # --- Cache check ---
    cache_file = _cache_path(config)
    if config.get("cache", {}).get("enabled", True):
        cached = _load_cache(cache_file)
        if cached is not None:
            return cached

    variant = config.get("variant", "unknown")
    print(f"\nBuilding graph  |  variant = {variant}")

    df         = prep.df
    train_mask = prep.train_mask
    val_mask   = prep.val_mask
    test_mask  = prep.test_mask
    col_cfg    = config["columns"]

    # --- Step 1: node ID mappings ---
    print("\n[1/4] Building node maps...")
    node_maps = build_node_maps(df, col_cfg)

    # --- Step 2: node features ---
    print("\n[2/4] Computing node features...")
    internal_features = build_internal_node_features(df, col_cfg, node_maps["internal_account"], train_mask)
    external_features = build_external_node_features(df, col_cfg, node_maps["external_account"], train_mask)

    # --- Step 3: fit edge normalization stats on training data ---
    print("\n[3/4] Building edges...")
    stats = fit_edge_stats(df, col_cfg, train_mask)

    # edge definitions come from config["edges"]["relations"] (flattened by load_variant)
    edge_defs = config["edges"]["relations"]

    bundles = build_all_edges(
        df         = df,
        node_maps  = node_maps,
        edge_defs  = edge_defs,
        col_cfg    = col_cfg,
        stats      = stats,
        train_mask = train_mask,
        val_mask   = val_mask,
        test_mask  = test_mask,
    )

    # --- Step 4: assemble into HeteroData ---
    print("\n[4/4] Assembling HeteroData object...")
    data = HeteroData()

    data["internal_account"].x         = internal_features
    data["internal_account"].num_nodes = len(node_maps["internal_account"])

    data["external_account"].x         = external_features
    data["external_account"].num_nodes = len(node_maps["external_account"])

    for b in bundles:
        et = b.edge_type
        data[et].edge_index = b.edge_index
        data[et].edge_attr  = b.edge_attr
        data[et].y          = b.y
        data[et].train_mask = b.train_mask
        data[et].val_mask   = b.val_mask
        data[et].test_mask  = b.test_mask

    _print_summary(data)

    # --- Cache ---
    result = {"data": data, "node_maps": node_maps}
    if config.get("cache", {}).get("enabled", True):
        _save_cache(cache_file, result)

    return result


# ── Cache helpers ─────────────────────────────────────────────────────────────

def _cache_path(config: dict) -> str:
    cache_dir = config.get("cache", {}).get("dir", "data/processed/bank")
    variant   = config.get("variant", "unknown")
    sr        = config.get("sample_ratio", 1.0)

    # hash covers everything that would change the graph output
    sig = json.dumps({
        "variant":        variant,
        "edges":          config["edges"]["relations"],
        "edge_features":  config["edges"]["features"],
        "truncate_after": config.get("truncate_after"),
        "split":          config["split"],
    }, sort_keys=True)

    h = hashlib.md5(sig.encode()).hexdigest()[:8]
    return str(PROJECT_ROOT / cache_dir / f"graph_{variant}_sr{sr:.2f}_{h}.pkl")


def _load_cache(path: str):
    if os.path.exists(path):
        print(f"Loading graph from cache: {path}")
        with open(path, "rb") as f:
            return pickle.load(f)
    return None


def _save_cache(path: str, result: dict):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(result, f)
    print(f"Graph cached at: {path}")


# ── Summary ───────────────────────────────────────────────────────────────────

def _print_summary(data: HeteroData):
    print(f"\n{'='*55}")
    print("Graph summary")
    print(f"{'='*55}")
    for nt in data.node_types:
        n    = data[nt].num_nodes
        fdim = data[nt].x.shape[1] if hasattr(data[nt], "x") else 0
        print(f"  {nt:<25} {n:>10,} nodes   feat_dim={fdim}")
    print()
    for et in data.edge_types:
        n    = data[et].edge_index.shape[1]
        fdim = data[et].edge_attr.shape[1] if data[et].edge_attr is not None else 0
        print(f"  {str(et):<50} {n:>8,} edges  feat_dim={fdim}")
