"""
Shared data preparation layer.

Loads the bank dataset, cleans it, splits it once. All experiment levels
(tabular baseline, graph builder) consume the same PreparedData object,
guaranteeing identical data splits.

Usage:
    from src.data.prepare import prepare_data

    prep = prepare_data(config)

    prep.df          # cleaned DataFrame
    prep.train_mask  # pd.Series[bool]
    prep.val_mask
    prep.test_mask
    prep.labels      # np.ndarray (N,)
    prep.col_cfg     # config["columns"]
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.graph_builder.loader import load_raw
from src.utils.split import temporal_split, random_stratified_split


@dataclass
class PreparedData:
    df:         pd.DataFrame
    train_mask: pd.Series
    val_mask:   pd.Series
    test_mask:  pd.Series
    labels:     np.ndarray   # (N,) float32
    col_cfg:    dict


def prepare_data(config: dict) -> PreparedData:
    from src.utils.config import PROJECT_ROOT
    data_path = str(PROJECT_ROOT / config["data_path"])
    df        = load_raw(data_path, config)

    split_cfg = config["split"]
    col_cfg   = config["columns"]

    if split_cfg.get("method", "temporal") == "temporal":
        train_mask, val_mask, test_mask = temporal_split(
            df,
            train_end = split_cfg["train_end"],
            val_end   = split_cfg["val_end"],
        )
    else:
        train_mask, val_mask, test_mask = random_stratified_split(
            df,
            label_col   = col_cfg["label"],
            train_ratio = split_cfg.get("train_ratio", 0.7),
            val_ratio   = split_cfg.get("val_ratio",   0.15),
            seed        = split_cfg.get("seed",        42),
        )

    labels = df[col_cfg["label"]].fillna(0).values.astype(np.float32)

    print(f"\nPreparedData ready:")
    print(f"  Rows: {len(df):,}  |  Train: {train_mask.sum():,}  |  Val: {val_mask.sum():,}  |  Test: {test_mask.sum():,}")
    print(f"  Fraud: {int(labels.sum()):,} ({100 * labels.mean():.3f}%)")

    return PreparedData(
        df         = df,
        train_mask = train_mask,
        val_mask   = val_mask,
        test_mask  = test_mask,
        labels     = labels,
        col_cfg    = col_cfg,
    )
