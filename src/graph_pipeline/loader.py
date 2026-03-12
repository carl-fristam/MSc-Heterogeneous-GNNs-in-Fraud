"""
Raw data loading for the graph pipeline.

- Reads CSV/parquet
- parses dates
- applies optional sampling.
- Adds a unified `_datetime` column regardless of input format.

"""



import pandas as pd

from src.graph_pipeline.schema import DatasetSchema

def load_raw(
    data_path: str,
    schema: DatasetSchema,
    sample_ratio: float = 1.0,
    n_days: int | None = None,
) -> pd.DataFrame:
    """
    Load raw transaction data and prepare it for the pipeline.

    Steps:
        1. Read CSV (or parquet if path ends with .parquet)
        2. Parse dates into a unified '_datetime' column
        3. Sort by datetime (critical for temporal splitting)
        4. Apply optional sampling (n_days takes priority over sample_ratio)

    Args:
        data_path:    path to the raw data file
        schema:       DatasetSchema mapping column names
        sample_ratio: fraction of rows to keep (1.0 = all)
        n_days:       if set, keep only the first n_days of data

    Returns:
        DataFrame with all original columns plus '_datetime' (pd.Timestamp)
    """

    # Step 1: load the data
    print(f"Loading data from {data_path}")
    if data_path.endswith(".parquet"):
        df = pd.read_parquet(data_path)
    else:
        df = pd.read_csv(data_path)
    print(f"Loaded {len(df)} rows")

    # Step 2: parse dates into a unified '_datetime' column
    df["_datetime"] = _parse_datetime(df, schema)

    # Step 3: sort by time
    df = df.sort_values("_datetime").reset_index(drop=True)

    # Step 4: sample
    df = _apply_sampling(df, sample_ratio, n_days)

    # Step 5: Ensure string-typed ID columns
    df["_sender"] = df[schema.sender_id].astype(str)
    df["_receiver"] = df[schema.receiver_id].astype(str)

    print(f"  Final: {len(df):,} rows, date range: {df['_datetime'].min()} → {df['_datetime'].max()}")
    return df

def _parse_datetime(df: pd.DataFrame, schema: DatasetSchema) -> pd.Series:
    """
    Combine date (and optional time) columns into a single datetime Series.

    SAML-D has separate Date and Time columns:  "2022-10-07" + "10:35:19"
    Bank data has a single EVENTTIME column:    "2022-10-07 10:35:19"
    """
    if schema.timestamp_time is not None:
        # Two separate columns — combine them
        combined = df[schema.timestamp].astype(str) + " " + df[schema.timestamp_time].astype(str)
        return pd.to_datetime(combined)
    else:
        # Single datetime column
        return pd.to_datetime(df[schema.timestamp])
    
def _apply_sampling(df: pd.DataFrame, sample_ratio: float, n_days: int | None) -> pd.DataFrame:
    """
    Reduce dataset size for development.

    n_days takes priority: keeps first n_days from the earliest date.

    Otherwise sample_ratio randomly samples rows.
    """

    if n_days is not None:

        cutoff = df["_datetime"].min() + pd.Timedelta(days=n_days)
        df = df[df["_datetime"] < cutoff].reset_index(drop=True)
        print(f"  Temporal sample: first {n_days} days → {len(df):,} rows")

    elif sample_ratio < 1.0:

        df = df.sample(frac=sample_ratio, random_state=42).reset_index(drop=True)
        print(f"  Random sample: {sample_ratio:.0%} → {len(df):,} rows")

    return df