"""
Data loading for the bank payment pipeline.

Assumes the parquet is already clean. This module only does what is
strictly necessary for graph construction:

  - Read the parquet
  - Parse EVENTTIME → _datetime (no-op if already a datetime dtype)
  - Sort by time (required for temporal split correctness)
  - Truncate rows beyond the fraud-label coverage window
  - Optional temporal sampling for dev runs
  - Expose _sender, _receiver, _datetime as canonical internal column names
"""

import pandas as pd


def load_raw(data_path: str, config: dict) -> pd.DataFrame:
    """
    Load the bank transaction parquet and prepare it for graph construction.

    Args:
        data_path:  absolute path to .parquet
        config:     full pipeline config dict

    Returns:
        DataFrame with _sender, _receiver, _datetime columns added.
    """
    col_cfg = config["columns"]
    sample  = config.get("sample_ratio", 1.0)

    print(f"Loading {data_path} ...")
    df = pd.read_parquet(data_path)
    print(f"  Rows: {len(df):,}  |  Columns: {len(df.columns)}")

    # ── Timestamp ─────────────────────────────────────────────────────────────
    ts_col = col_cfg["timestamp"]
    df["_datetime"] = pd.to_datetime(df[ts_col])

    # ── Sort by time (required for temporal split) ────────────────────────────
    df = df.sort_values("_datetime").reset_index(drop=True)

    # ── Truncate to fraud-label coverage window ───────────────────────────────
    truncate = config.get("truncate_after")
    if truncate:
        before = len(df)
        df = df[df["_datetime"] <= pd.Timestamp(truncate)].reset_index(drop=True)
        print(f"  Truncated to {truncate}: {before:,} → {len(df):,} rows")

    # ── Optional temporal sampling (dev / debug runs) ─────────────────────────
    if sample < 1.0:
        n_rows = int(len(df) * sample)
        df = df.head(n_rows).reset_index(drop=True)
        print(f"  Sample {sample:.0%}: {n_rows:,} rows  "
              f"({df['_datetime'].min().date()} → {df['_datetime'].max().date()})")

    # ── Canonical aliases ─────────────────────────────────────────────────────
    df["_sender"]   = df[col_cfg["sender"]].astype(str)
    df["_receiver"] = df[col_cfg["receiver"]].astype(str)

    print(f"  Ready: {len(df):,} rows  "
          f"({df['_datetime'].min().date()} → {df['_datetime'].max().date()})")
    return df
