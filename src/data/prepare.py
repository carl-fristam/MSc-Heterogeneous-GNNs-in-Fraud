"""
Shared data preparation layer.

Loads the bank dataset, cleans it, splits it, and computes transaction-level
features ONCE. Everything downstream (tabular baselines, graph builders,
models) consumes the same PreparedData object.

Usage:
    from src.data.prepare import prepare_data

    prep = prepare_data(config)

    prep.df              # cleaned DataFrame
    prep.train_mask      # pd.Series[bool]
    prep.val_mask
    prep.test_mask
    prep.txn_features    # np.ndarray (N, F) — transaction feature matrix
    prep.vocabs          # OHE vocabularies fitted on training data
    prep.labels          # np.ndarray (N,)
    prep.col_cfg         # config["columns"]
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.utils.config import PROJECT_ROOT
from src.graph_pipeline_bank.loader import load_raw
from src.graph_pipeline_bank.features_edge import build_edge_features, fit_vocabs
from src.utils.split import temporal_split, random_stratified_split


@dataclass
class PreparedData:
    """Single source of truth for all experiment levels."""
    df: pd.DataFrame
    train_mask: pd.Series
    val_mask: pd.Series
    test_mask: pd.Series
    txn_features: np.ndarray    # (N, F) transaction-level feature matrix
    vocabs: dict                # OHE vocabularies from training
    labels: np.ndarray          # (N,) float32
    col_cfg: dict               # config["columns"]


def prepare_data(config: dict) -> PreparedData:
    """
    Load, clean, split, and featurize the bank dataset.

    This is the single entry point for data preparation. All experiment
    levels consume this same object, guaranteeing identical data splits
    and feature engineering.

    Args:
        config: dict loaded from any graph_bank config

    Returns:
        PreparedData instance
    """
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

    vocabs = fit_vocabs(df, col_cfg, train_mask)
    txn_features = build_edge_features(df, col_cfg, vocabs, enabled=None)

    labels = df[col_cfg["label"]].fillna(0).values.astype(np.float32)

    print(f"\nPreparedData ready:")
    print(f"  Rows: {len(df):,}  |  Features: {txn_features.shape[1]}")
    print(f"  Train: {train_mask.sum():,}  |  Val: {val_mask.sum():,}  |  Test: {test_mask.sum():,}")
    print(f"  Fraud: {int(labels.sum()):,} ({100 * labels.mean():.3f}%)")

    return PreparedData(
        df=df,
        train_mask=train_mask,
        val_mask=val_mask,
        test_mask=test_mask,
        txn_features=txn_features,
        vocabs=vocabs,
        labels=labels,
        col_cfg=col_cfg,
    )
