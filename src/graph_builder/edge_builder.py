"""
Builds the graph connectivity for each edge type.

For each edge type defined in the config variant we:
  1. Filter the dataframe to get the relevant transaction rows
  2. Map sender/receiver account IDs to integer node indices
  3. Build the edge_index tensor — shape (2, num_edges)
  4. Build edge features and labels
  5. Attach train/val/test masks

The result is a list of EdgeBundle objects, one per edge type.
The assembler then puts these into a PyG HeteroData object.
"""

import numpy as np
import pandas as pd
import torch
from dataclasses import dataclass

from src.graph_builder.edge_features import build_edge_features


@dataclass
class EdgeBundle:
    edge_type:  tuple          # (src_node_type, relation_name, dst_node_type)
    edge_index: torch.Tensor   # shape (2, num_edges) — integer node indices
    edge_attr:  torch.Tensor   # shape (num_edges, num_features)
    y:          torch.Tensor   # shape (num_edges,) — fraud labels
    amounts:    torch.Tensor   # shape (num_edges,) — raw BASEVALUE (for threshold table)
    train_mask: torch.Tensor
    val_mask:   torch.Tensor
    test_mask:  torch.Tensor


def build_all_edges(
    df:         pd.DataFrame,
    node_maps:  dict,
    edge_defs:  list,
    col_cfg:    dict,
    stats:      dict,
    train_mask: pd.Series,
    val_mask:   pd.Series,
    test_mask:  pd.Series,
) -> list[EdgeBundle]:
    """
    Build all edge bundles for a given graph variant.

    Args:
        df:        full transaction dataframe
        node_maps: output of build_node_maps — {node_type: {acc_id: int_index}}
        edge_defs: list of edge definitions from config["variants"][v]["edges"]
        col_cfg:   columns section from master.yaml
        stats:     output of fit_edge_stats — normalization constants for amount
        train/val/test_mask: boolean pd.Series aligned with df index

    Returns:
        list of EdgeBundle, one per edge type
    """
    label_col = col_cfg["label"]
    bundles   = []

    for edge_def in edge_defs:
        name     = edge_def["name"]
        src_type = edge_def["src"]
        dst_type = edge_def["dst"]
        filt     = edge_def.get("filter")

        print(f"\n  Building edge type: {name}  ({src_type} → {dst_type})")

        # --- Step 1: filter rows for this edge type ---
        if filt:
            sub = df.query(filt)
        else:
            sub = df

        print(f"    rows after filter: {len(sub):,}")

        if len(sub) == 0:
            print(f"    skipping — no rows matched filter")
            continue

        # --- Step 2: map account IDs to integer node indices ---
        src_map = node_maps[src_type]
        dst_map = node_maps[dst_type]

        src_idx = sub["_sender"].map(src_map)
        dst_idx = sub["_receiver"].map(dst_map)

        # drop any rows where the account ID wasn't in the node map
        # (should be rare, but can happen at boundaries)
        valid   = src_idx.notna() & dst_idx.notna()
        if not valid.all():
            print(f"    dropped {(~valid).sum():,} rows with unmapped IDs")
            sub     = sub[valid]
            src_idx = src_idx[valid]
            dst_idx = dst_idx[valid]

        src_idx = src_idx.astype(np.int64).values
        dst_idx = dst_idx.astype(np.int64).values

        # --- Step 3: build edge_index tensor ---
        # PyG expects shape (2, num_edges) where row 0 = sources, row 1 = destinations
        edge_index = torch.tensor(np.stack([src_idx, dst_idx]), dtype=torch.long)

        # --- Step 4: edge features and labels ---
        feat_arr  = build_edge_features(sub, col_cfg, stats)
        edge_attr = torch.tensor(feat_arr, dtype=torch.float32)

        y       = torch.tensor(sub[label_col].fillna(0).values, dtype=torch.float32)
        amounts = torch.tensor(sub[col_cfg["base_value"]].fillna(0).values, dtype=torch.float32)

        # --- Step 5: train/val/test masks aligned to this edge subset ---
        train_m = torch.tensor(train_mask.loc[sub.index].values, dtype=torch.bool)
        val_m   = torch.tensor(val_mask.loc[sub.index].values,   dtype=torch.bool)
        test_m  = torch.tensor(test_mask.loc[sub.index].values,  dtype=torch.bool)

        # print a quick sanity check on fraud rate per split
        for split_name, mask in [("train", train_m), ("val", val_m), ("test", test_m)]:
            n   = mask.sum().item()
            pos = y[mask].sum().item() if n > 0 else 0
            print(f"    {split_name}: {n:,} edges  |  fraud: {int(pos)} ({100*pos/n:.2f}%)" if n > 0 else f"    {split_name}: 0 edges")

        bundles.append(EdgeBundle(
            edge_type  = (src_type, name, dst_type),
            edge_index = edge_index,
            edge_attr  = edge_attr,
            y          = y,
            amounts    = amounts,
            train_mask = train_m,
            val_mask   = val_m,
            test_mask  = test_m,
        ))

    return bundles
