"""
Export 50% sampled splits as parquet files.

Uses the exact same sampling logic as prepare_data with sample=0.5
and keep_all_fraud=True: all fraud transactions are kept, legitimate
transactions are downsampled 50% per month per split.

Output: datasets/splits_50pct/train.parquet, val.parquet, test.parquet

Usage:
    PYTHONPATH=. python scripts/export_50pct_splits.py
"""

from pathlib import Path

import pandas as pd

from src.utils.config import load_variant, PROJECT_ROOT


SAMPLE = 0.5
OUTPUT_DIR = PROJECT_ROOT / "datasets" / "splits_50pct"


def stratified_temporal_sample(df, frac, label_col, time_col):
    """Same logic as src/data/prepare._stratified_temporal_sample with keep_all_fraud=True."""
    fraud = df[df[label_col] == True]
    legit = df[df[label_col] == False]
    dt = pd.to_datetime(legit[time_col])
    month = dt.dt.to_period("M")
    legit_sampled = legit.groupby(month, group_keys=False).apply(
        lambda g: g.sample(frac=frac, random_state=42) if len(g) > 1
                  else g
    )
    sampled = pd.concat([fraud, legit_sampled], ignore_index=True)
    sampled = sampled.sort_values(time_col).reset_index(drop=True)
    return sampled


def main():
    config = load_variant("v1")
    col_cfg = config["columns"]
    split_cfg = config["split"]
    split_dir = Path(PROJECT_ROOT) / split_cfg["dir"]

    label_col = col_cfg["label"]
    time_col = col_cfg.get("timestamp", "EVENTTIME")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for name in ("train", "val", "test"):
        parquet_path = split_dir / f"{name}.parquet"
        csv_path = split_dir / f"{name}.csv"

        if parquet_path.exists():
            df = pd.read_parquet(parquet_path)
        elif csv_path.exists():
            df = pd.read_csv(csv_path)
        else:
            raise FileNotFoundError(f"No {name}.parquet or {name}.csv in {split_dir}")

        n_before = len(df)
        fraud_before = df[label_col].sum()

        sampled = stratified_temporal_sample(df, SAMPLE, label_col, time_col)

        n_after = len(sampled)
        fraud_after = sampled[label_col].sum()

        out_path = OUTPUT_DIR / f"{name}.parquet"
        sampled.to_parquet(out_path, index=False)

        print(f"{name}: {n_before:,} → {n_after:,} rows  |  "
              f"fraud: {int(fraud_before):,} → {int(fraud_after):,}  |  "
              f"saved to {out_path.relative_to(PROJECT_ROOT)}")

    # Combined file
    all_dfs = []
    for name in ("train", "val", "test"):
        all_dfs.append(pd.read_parquet(OUTPUT_DIR / f"{name}.parquet"))
    combined = pd.concat(all_dfs, ignore_index=True)
    combined_path = OUTPUT_DIR / "combined.parquet"
    combined.to_parquet(combined_path, index=False)
    print(f"\nCombined: {len(combined):,} rows  |  saved to {combined_path.relative_to(PROJECT_ROOT)}")

    print(f"\nDone — splits saved to {OUTPUT_DIR.relative_to(PROJECT_ROOT)}/")


if __name__ == "__main__":
    main()
