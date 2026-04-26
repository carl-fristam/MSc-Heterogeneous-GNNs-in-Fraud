"""
Shared data preparation layer.

Loads pre-split DataFrames (train/val/test), optionally samples them with
stratified temporal sampling, concatenates into a single DataFrame with
boolean masks. All experiment levels (tabular baseline, graph builder)
consume the same PreparedData object.

Usage:
    from src.data.prepare import prepare_data

    prep = prepare_data(config)
    prep = prepare_data(config, sample=0.5)  # stratified 50% sample
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from src.utils.config import PROJECT_ROOT


@dataclass
class PreparedData:
    df:         pd.DataFrame
    train_mask: pd.Series
    val_mask:   pd.Series
    test_mask:  pd.Series
    labels:     np.ndarray   # (N,) float32
    col_cfg:    dict


def prepare_data(config: dict, df_train=None, df_val=None, df_test=None,
                 sample: float = None) -> PreparedData:
    """
    Build PreparedData from three pre-split DataFrames.

    Args:
        config:    full config dict
        df_train:  optional — pass DataFrames directly (e.g. from notebook)
        df_val:    optional
        df_test:   optional
        sample:    optional — fraction to sample (stratified by month + label)
    """
    col_cfg = config["columns"]

    if df_train is None:
        df_train, df_val, df_test = _load_splits(config)

    if sample is not None and sample < 1.0:
        label_col = col_cfg["label"]
        time_col = col_cfg.get("timestamp", "EVENTTIME")
        df_train = _stratified_temporal_sample(df_train, sample, label_col, time_col)
        df_val   = _stratified_temporal_sample(df_val,   sample, label_col, time_col)
        df_test  = _stratified_temporal_sample(df_test,  sample, label_col, time_col)

    n_train, n_val, n_test = len(df_train), len(df_val), len(df_test)
    df = pd.concat([df_train, df_val, df_test], ignore_index=True)

    df.columns = df.columns.str.strip()

    df["_sender"]   = df[col_cfg["sender"]].astype(str)
    df["_receiver"] = df[col_cfg["receiver"]].astype(str)

    train_mask = pd.Series([True]*n_train + [False]*n_val + [False]*n_test)
    val_mask   = pd.Series([False]*n_train + [True]*n_val + [False]*n_test)
    test_mask  = pd.Series([False]*n_train + [False]*n_val + [True]*n_test)

    labels = df[col_cfg["label"]].fillna(0).values.astype(np.float32)

    _print_summary(df, train_mask, val_mask, test_mask, labels, col_cfg)

    return PreparedData(
        df         = df,
        train_mask = train_mask,
        val_mask   = val_mask,
        test_mask  = test_mask,
        labels     = labels,
        col_cfg    = col_cfg,
    )


def _stratified_temporal_sample(df: pd.DataFrame, frac: float,
                                 label_col: str, time_col: str) -> pd.DataFrame:
    """
    Sample a fraction of rows, preserving monthly distribution and fraud ratio.
    Groups by (month, label), samples frac from each group.
    """
    dt = pd.to_datetime(df[time_col])
    month = dt.dt.to_period("M")

    sampled = df.groupby([month, df[label_col]], group_keys=False).apply(
        lambda g: g.sample(frac=frac, random_state=42) if len(g) > 1
                  else g  # keep single-row groups (rare fraud months)
    )
    sampled = sampled.sort_values(time_col).reset_index(drop=True)
    return sampled


def _print_summary(df, train_mask, val_mask, test_mask, labels, col_cfg):
    time_col = col_cfg.get("timestamp", "EVENTTIME")
    label_col = col_cfg["label"]
    has_time = time_col in df.columns

    print(f"\n{'='*65}")
    print("PreparedData summary")
    print(f"{'='*65}")
    print(f"  Total rows: {len(df):,}  |  Fraud: {int(labels.sum()):,} ({100 * labels.mean():.3f}%)")
    print()

    for name, mask in [("Train", train_mask), ("Val", val_mask), ("Test", test_mask)]:
        n = mask.sum()
        sub = df[mask]
        pos = sub[label_col].sum() if n > 0 else 0
        pct = 100 * pos / n if n > 0 else 0

        if has_time and n > 0:
            dates = pd.to_datetime(sub[time_col])
            t_min = dates.min().strftime("%Y-%m-%d")
            t_max = dates.max().strftime("%Y-%m-%d")
            print(f"  {name:<5}  {n:>10,} rows  |  {t_min} → {t_max}  |  fraud: {int(pos):,} ({pct:.2f}%)")
        else:
            print(f"  {name:<5}  {n:>10,} rows  |  fraud: {int(pos):,} ({pct:.2f}%)")

    print(f"{'='*65}")


def _load_splits(config: dict):
    split_cfg = config["split"]
    split_dir = Path(PROJECT_ROOT) / split_cfg["dir"]

    print(f"\nLoading splits from {split_dir}")
    dfs = []
    for name in ("train", "val", "test"):
        parquet_path = split_dir / f"{name}.parquet"
        csv_path     = split_dir / f"{name}.csv"
        if parquet_path.exists():
            dfs.append(pd.read_parquet(parquet_path))
            print(f"  {name}: {parquet_path.name} ({len(dfs[-1]):,} rows)")
        elif csv_path.exists():
            dfs.append(pd.read_csv(csv_path))
            print(f"  {name}: {csv_path.name} ({len(dfs[-1]):,} rows)")
        else:
            raise FileNotFoundError(
                f"No {name}.parquet or {name}.csv found in {split_dir}"
            )
    return dfs[0], dfs[1], dfs[2]
