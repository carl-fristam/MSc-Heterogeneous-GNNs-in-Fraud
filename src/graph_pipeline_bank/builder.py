"""
Graph builder — orchestrates the full bank pipeline (edge classification).

Can be called standalone with a config, or with a PreparedData object
to share data loading/splitting with other experiment levels.

Entry point:
    # Standalone (loads data itself)
    from src.utils.config import load_config
    from src.graph_pipeline_bank.builder import build_graph
    result = build_graph(load_config("graph_bank_v1"))

    # Shared data (recommended — guarantees same split)
    from src.data.prepare import prepare_data
    from src.graph_pipeline_bank.builder import build_graph
    prep = prepare_data(config)
    result = build_graph(config, prep=prep)
"""

import hashlib
import json
import pickle
import os
from pathlib import Path

import torch
from torch_geometric.data import HeteroData

from src.utils.config import PROJECT_ROOT, load_config
from src.graph_pipeline_bank.loader import load_raw
from src.graph_pipeline_bank.node_builder import build_node_maps
from src.graph_pipeline_bank.features_node import build_node_features
from src.graph_pipeline_bank.edge_builder import build_edges
from src.utils.split import temporal_split, random_stratified_split


def build_graph(config: dict, prep=None) -> dict:
    """
    Build a PyG HeteroData graph for the bank payment dataset.

    Args:
        config: dict loaded from configs/graph_bank_v*.yaml
        prep: optional PreparedData instance. If provided, reuses its
              df/masks/vocabs instead of loading from scratch.

    Returns:
        dict with keys:
            "data"      — PyG HeteroData
            "node_maps" — {node_type: {raw_id: int_index}}
            "vocabs"    — {feature_name: vocab_list} for OHE reproducibility
    """
    # ── Cache check ──────────────────────────────────────────────────────────
    cache_path = _cache_path(config)
    if config.get("cache", {}).get("enabled", True):
        cached = _load_cache(cache_path)
        if cached is not None:
            return cached

    variant = config.get("variant", "unknown")
    print(f"\n{'='*60}")
    print(f"Building bank graph  |  variant={variant}")
    print(f"{'='*60}")

    # ── Load & split (reuse PreparedData if available) ────────────────────────
    if prep is not None:
        df = prep.df
        train_mask = prep.train_mask
        val_mask = prep.val_mask
        test_mask = prep.test_mask
    else:
        data_path = str(PROJECT_ROOT / config["data_path"])
        df = load_raw(data_path, config)

        split_cfg = config["split"]
        col_cfg = config["columns"]

        if split_cfg.get("method", "temporal") == "temporal":
            train_mask, val_mask, test_mask = temporal_split(
                df,
                train_end=split_cfg["train_end"],
                val_end=split_cfg["val_end"],
            )
        else:
            train_mask, val_mask, test_mask = random_stratified_split(
                df,
                label_col=col_cfg["label"],
                train_ratio=split_cfg.get("train_ratio", 0.7),
                val_ratio=split_cfg.get("val_ratio", 0.15),
                seed=split_cfg.get("seed", 42),
            )

    # ── Node mappings ─────────────────────────────────────────────────────────
    print("\nBuilding node mappings...")
    node_maps = build_node_maps(df, config)

    # ── Node features ─────────────────────────────────────────────────────────
    print("\nBuilding node features (train-only aggregation)...")
    node_features: dict[str, torch.Tensor] = {}
    for node_type, node_cfg in config["nodes"].items():
        enabled = node_cfg.get("features", None)
        node_features[node_type] = build_node_features(
            df=df,
            node_type=node_type,
            node_to_id=node_maps[node_type],
            col_cfg=config["columns"],
            train_mask=train_mask,
            enabled=enabled,
        )

    # ── Edges ─────────────────────────────────────────────────────────────────
    print("\nBuilding edges...")
    bundles, vocabs = build_edges(df, node_maps, config, train_mask, val_mask, test_mask)

    # ── Assemble HeteroData ───────────────────────────────────────────────────
    data = HeteroData()

    for node_type, feat in node_features.items():
        data[node_type].x         = feat
        data[node_type].num_nodes = len(node_maps[node_type])

    for b in bundles:
        et = b.edge_type
        data[et].edge_index = b.edge_index
        if b.edge_attr  is not None: data[et].edge_attr  = b.edge_attr
        if b.y          is not None: data[et].y          = b.y
        if b.train_mask is not None: data[et].train_mask = b.train_mask
        if b.val_mask   is not None: data[et].val_mask   = b.val_mask
        if b.test_mask  is not None: data[et].test_mask  = b.test_mask

    _print_summary(data)

    # ── Cache ─────────────────────────────────────────────────────────────────
    result = {"data": data, "node_maps": node_maps, "vocabs": vocabs}
    if config.get("cache", {}).get("enabled", True):
        _save_cache(cache_path, result)

    return result


# ── Cache helpers ─────────────────────────────────────────────────────────────

def _cache_path(config: dict) -> str:
    cache_dir = config.get("cache", {}).get("dir", "data/processed/bank")
    variant   = config.get("variant", "unknown")
    sr        = config.get("sample_ratio", 1.0)
    rel_sig = json.dumps(config.get("edges", {}).get("relations", []), sort_keys=True)
    h = hashlib.md5(rel_sig.encode()).hexdigest()[:6]
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
    print("Graph Summary")
    print(f"{'='*60}")

    for nt in data.node_types:
        n    = data[nt].num_nodes
        fdim = data[nt].x.shape[1] if hasattr(data[nt], "x") else 0
        print(f"  {nt:<25} {n:>10,} nodes   feat_dim={fdim}")

    print()
    for et in data.edge_types:
        n    = data[et].edge_index.shape[1]
        fdim = data[et].edge_attr.shape[1] if hasattr(data[et], "edge_attr") and data[et].edge_attr is not None else 0
        print(f"  {str(et):<55} {n:>8,} edges  feat_dim={fdim}")
        if hasattr(data[et], "y") and data[et].y is not None:
            y = data[et].y
            for split in ["train", "val", "test"]:
                attr = f"{split}_mask"
                if hasattr(data[et], attr):
                    mask = data[et][attr]
                    ns   = mask.sum().item()
                    if ns > 0:
                        pos = y[mask].sum().item()
                        print(f"    {split:5s}: {ns:>8,}  pos={int(pos):>6} ({100*pos/ns:.2f}%)")
