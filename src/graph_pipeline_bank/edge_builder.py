"""
Edge construction for the bank heterogeneous graph.

For each relation declared in config["edges"]["relations"]:
  1. Apply the filter expression to select the relevant rows
  2. Map sender/receiver IDs → integer node indices
  3. Build edge features from the selected rows
  4. Attach labels (CONFIRMED_RISK) and train/val/test masks

The result is a list of EdgeBundle dicts, one per relation type.
The builder.py then assembles these into a PyG HeteroData object.

Filter expressions use pandas .query() syntax, e.g.:
  "TRANSACTIONONUS == True"
  "TRANSACTIONONUS == False and PAYMENTSUBMETHOD == 'realTime'"

Special relation type: uses_device (V3)
  No filter — covers all rows.
  Uses src_col / dst_col overrides instead of _sender / _receiver.
  deduplicate=True: one edge per unique (src, dst) pair.
  has_label=False: no classification target on these edges.
"""

import numpy as np
import pandas as pd
import torch
from dataclasses import dataclass, field

from src.graph_pipeline_bank.features_edge import build_edge_features, fit_vocabs
from src.graph_pipeline_bank.node_builder import NodeMaps


@dataclass
class EdgeBundle:
    """All data for one relation type."""
    edge_type:  tuple[str, str, str]   # (src_node_type, relation_name, dst_node_type)
    edge_index: torch.Tensor           # shape (2, E)
    edge_attr:  torch.Tensor | None    # shape (E, F) or None
    y:          torch.Tensor | None    # shape (E,) float labels or None
    train_mask: torch.Tensor | None    # shape (E,) bool
    val_mask:   torch.Tensor | None
    test_mask:  torch.Tensor | None


def build_edges(
    df: pd.DataFrame,
    node_maps: NodeMaps,
    config: dict,
    train_mask: pd.Series,
    val_mask: pd.Series,
    test_mask: pd.Series,
) -> tuple[list[EdgeBundle], dict]:
    """
    Build all edge bundles declared in config["edges"]["relations"].

    Args:
        df:         full cleaned DataFrame (all rows)
        node_maps:  {node_type: {raw_id: int_index}}
        config:     full pipeline config
        train/val/test_mask: boolean pd.Series aligned with df.index

    Returns:
        (bundles, vocabs)
          bundles: list of EdgeBundle, one per relation
          vocabs:  OHE vocabularies fitted on training data (save in cache)
    """
    edge_cfg = config["edges"]
    col_cfg  = config["columns"]
    label_col = col_cfg.get("label")

    # Fit OHE vocabularies once from training data
    print("\n  Fitting edge feature vocabularies from training data...")
    vocabs = fit_vocabs(df, col_cfg, train_mask)

    enabled_feats = edge_cfg.get("features", None)
    bundles: list[EdgeBundle] = []

    for rel in edge_cfg["relations"]:
        name      = rel["name"]
        src_type  = rel["src"]
        dst_type  = rel["dst"]
        has_label = rel.get("has_label", True)
        dedup     = rel.get("deduplicate", False)
        feat_list = rel.get("features", enabled_feats)  # per-relation override or global

        print(f"\n  Building relation: ({src_type}, {name}, {dst_type})")

        # ── Row selection ────────────────────────────────────────────────────
        filt = rel.get("filter")
        if filt:
            try:
                sub_idx = df.query(filt).index
            except Exception as e:
                raise ValueError(f"Bad filter for relation '{name}': {filt!r}\n{e}")
        elif "src_col" in rel:
            # Structural edge (e.g. uses_device) — use all rows
            sub_idx = df.index
        else:
            sub_idx = df.index

        sub = df.loc[sub_idx]
        print(f"    rows selected: {len(sub):,}")

        if len(sub) == 0:
            print(f"    Warning: no rows matched filter — skipping")
            continue

        # ── Node ID → integer index ──────────────────────────────────────────
        src_col_name = rel.get("src_col", "_sender")
        dst_col_name = rel.get("dst_col", "_receiver")

        src_map = node_maps[src_type]
        dst_map = node_maps[dst_type]

        src_ids = sub[src_col_name].astype(str)
        dst_ids = sub[dst_col_name].astype(str)

        src_idx = src_ids.map(src_map)
        dst_idx = dst_ids.map(dst_map)

        # Drop rows where mapping failed (ID not in node pool)
        valid = src_idx.notna() & dst_idx.notna()
        if not valid.all():
            n_drop = (~valid).sum()
            print(f"    Dropped {n_drop:,} rows with unmapped node IDs")
            sub     = sub[valid]
            src_idx = src_idx[valid].astype(np.int64)
            dst_idx = dst_idx[valid].astype(np.int64)
        else:
            src_idx = src_idx.astype(np.int64)
            dst_idx = dst_idx.astype(np.int64)

        # ── Deduplication (structural edges only) ────────────────────────────
        if dedup:
            pairs   = pd.DataFrame({"src": src_idx.values, "dst": dst_idx.values})
            pairs   = pairs.drop_duplicates()
            src_idx_arr = pairs["src"].values
            dst_idx_arr = pairs["dst"].values
            edge_index  = torch.tensor(
                np.stack([src_idx_arr, dst_idx_arr]), dtype=torch.long
            )
            edge_attr   = None
            y = train_m = val_m = test_m = None
            print(f"    edges after dedup: {edge_index.shape[1]:,}")
        else:
            edge_index = torch.tensor(
                np.stack([src_idx.values, dst_idx.values]), dtype=torch.long
            )

            # ── Edge features ────────────────────────────────────────────────
            if feat_list is not None and len(feat_list) > 0:
                feat_arr = build_edge_features(sub, col_cfg, vocabs, enabled=feat_list)
                edge_attr = torch.tensor(feat_arr, dtype=torch.float32)
                print(f"    edge feature dim: {edge_attr.shape[1]}")
            else:
                edge_attr = None

            # ── Labels ───────────────────────────────────────────────────────
            if has_label and label_col and label_col in sub.columns:
                y = torch.tensor(sub[label_col].fillna(0).values, dtype=torch.float32)
                pos = int(y.sum().item())
                print(f"    labels: {len(y):,} edges, {pos} positive ({100*pos/len(y):.2f}%)")
            else:
                y = None

            # ── Masks (aligned to this relation's edges) ─────────────────────
            if has_label:
                train_m = torch.tensor(train_mask.loc[sub.index].values, dtype=torch.bool)
                val_m   = torch.tensor(val_mask.loc[sub.index].values,   dtype=torch.bool)
                test_m  = torch.tensor(test_mask.loc[sub.index].values,  dtype=torch.bool)
                for split_name, mask in [("train", train_m), ("val", val_m), ("test", test_m)]:
                    n = mask.sum().item()
                    if y is not None and n > 0:
                        pos = y[mask].sum().item()
                        print(f"    {split_name:5s}: {n:,}  pos={int(pos)} ({100*pos/n:.2f}%)")
            else:
                train_m = val_m = test_m = None

        bundles.append(EdgeBundle(
            edge_type  = (src_type, name, dst_type),
            edge_index = edge_index,
            edge_attr  = edge_attr,
            y          = y,
            train_mask = train_m,
            val_mask   = val_m,
            test_mask  = test_m,
        ))

    return bundles, vocabs
