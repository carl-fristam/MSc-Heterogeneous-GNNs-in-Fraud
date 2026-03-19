"""
Data loading for the bank payment pipeline.

- Reads CSV or parquet
- Drops redundant / leaky / zero-variance columns declared in config
- Parses EVENTTIME → _datetime
- Sorts by time (required for temporal split)
- Applies optional sampling
- Normalises TRANSACTIONONUS to Python bool
- Exposes _sender, _receiver, _datetime as canonical internal column names
"""

import pandas as pd


def load_raw(data_path: str, config: dict) -> pd.DataFrame:
    """
    Load and clean the raw bank transaction file.

    Args:
        data_path:  absolute path to .parquet or .csv
        config:     full pipeline config dict (reads columns + sample_ratio)

    Returns:
        Cleaned DataFrame with added _sender, _receiver, _datetime columns.
    """
    col_cfg = config["columns"]
    sample  = config.get("sample_ratio", 1.0)

    # ── Load ────────────────────────────────────────────────────────────────
    print(f"Loading {data_path} ...")
    if data_path.endswith(".parquet"):
        df = pd.read_parquet(data_path)
    else:
        df = pd.read_csv(data_path, low_memory=False)
    print(f"  Raw rows: {len(df):,}  |  columns: {len(df.columns)}")

    # ── Drop declared columns ───────────────────────────────────────────────
    to_drop = [c for c in col_cfg.get("drop", []) if c in df.columns]
    if to_drop:
        df = df.drop(columns=to_drop)
        print(f"  Dropped {len(to_drop)} columns: {to_drop}")

    # ── Parse timestamp ──────────────────────────────────────────────────────
    ts_col = col_cfg["timestamp"]
    df["_datetime"] = pd.to_datetime(df[ts_col], errors="coerce")
    n_bad = df["_datetime"].isna().sum()
    if n_bad:
        print(f"  Warning: {n_bad:,} rows with unparseable {ts_col} — dropping")
        df = df.dropna(subset=["_datetime"])

    # ── Sort by time (required for temporal split correctness) ───────────────
    df = df.sort_values("_datetime").reset_index(drop=True)

    # ── Sample ───────────────────────────────────────────────────────────────
    if sample < 1.0:
        n_days = config.get("n_days", None)
        if n_days is not None:
            cutoff = df["_datetime"].min() + pd.Timedelta(days=n_days)
            df = df[df["_datetime"] < cutoff].reset_index(drop=True)
            print(f"  Temporal sample: first {n_days} days → {len(df):,} rows")
        else:
            n_rows = int(len(df) * sample)
            df = df.head(n_rows).reset_index(drop=True)
            print(f"  Temporal sample: first {sample:.0%} ({n_rows:,}) rows → {df['_datetime'].min()} to {df['_datetime'].max()}")

    # ── Normalise TRANSACTIONONUS to bool ────────────────────────────────────
    onus_col = col_cfg.get("onus_flag")
    if onus_col and onus_col in df.columns:
        df[onus_col] = (
            df[onus_col].astype(str).str.strip().str.lower()
            .map({"true": True, "false": False, "1": True, "0": False})
            .fillna(False)
            .astype(bool)
        )

    # ── Canonical aliases ────────────────────────────────────────────────────
    df["_sender"]   = df[col_cfg["sender"]].astype(str)
    df["_receiver"] = df[col_cfg["receiver"]].astype(str)

    print(f"  Final: {len(df):,} rows  |  {df['_datetime'].min()} → {df['_datetime'].max()}")
    return df
