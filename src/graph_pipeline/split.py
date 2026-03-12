import pandas as pd
import numpy as np


def random_stratified_split(
        df: pd.DataFrame,
        label_col: str,
        train_ratio: float = 0.7,
        val_ratio: float = 0.15,
        seed: int = 42,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """
    Random stratified split preserving class balance across splits.

    Matches Johannessen & Jullum (2023) experimental setup.

    Args:
        df:          DataFrame with a label column
        label_col:   name of the binary label column
        train_ratio: fraction for training
        val_ratio:   fraction for validation (rest goes to test)
        seed:        random seed for reproducibility

    Returns:
        (train_mask, val_mask, test_mask) — boolean pd.Series
    """
    rng = np.random.default_rng(seed)
    n = len(df)
    train_mask = pd.Series(False, index=df.index)
    val_mask = pd.Series(False, index=df.index)
    test_mask = pd.Series(False, index=df.index)

    # Split each class independently to preserve balance
    for label in df[label_col].unique():
        idx = df.index[df[label_col] == label].to_numpy()
        rng.shuffle(idx)

        n_cls = len(idx)
        n_train = int(n_cls * train_ratio)
        n_val = int(n_cls * val_ratio)

        train_mask.iloc[idx[:n_train]] = True
        val_mask.iloc[idx[n_train:n_train + n_val]] = True
        test_mask.iloc[idx[n_train + n_val:]] = True

    total = train_mask.sum() + val_mask.sum() + test_mask.sum()
    assert total == n, f"Split masks don't cover all rows: {total} vs {n}"

    _print_split_stats(df, train_mask, val_mask, test_mask, title="Random stratified split")

    return train_mask, val_mask, test_mask


def temporal_split(
        df: pd.DataFrame,
        train_end: str,
        val_end: str,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """
    Split a DataFrame into train/val/test by date thresholds.

    All transactions before train_end   → train
    Between train_end and val_end       → val
    After val_end                       → test

    Args:
        df:         DataFrame with a '_datetime' column (created by loader)
        train_end:  date string like "2023-04-01" — exclusive upper bound for train
        val_end:    date string like "2023-06-01" — exclusive upper bound for val

    Returns:
        (train_mask, val_mask, test_mask) — boolean pd.Series, same index as df
    """

    dates = df["_datetime"]
    train_cutoff = pd.Timestamp(train_end)
    val_cutoff = pd.Timestamp(val_end)

    train_mask = dates < train_cutoff
    val_mask = (dates >= train_cutoff) & (dates < val_cutoff)
    test_mask = dates >= val_cutoff

    # Sanity checks
    total = train_mask.sum() + val_mask.sum() + test_mask.sum()
    assert total == len(df), f"Split masks don't cover all rows: {total} vs {len(df)}"

    _print_split_stats(df, train_mask, val_mask, test_mask, title="Temporal split")

    return train_mask, val_mask, test_mask

def _print_split_stats(
    df: pd.DataFrame,
    train_mask: pd.Series,
    val_mask: pd.Series,
    test_mask: pd.Series,
    title: str = "Split",
) -> None:
    """ Print row counts, date ranges, and label distributions for each split """

    label_col = None

    for col in ("Is_laundering", "CONFIRMEDRISK"):
        if col in df.columns:
            label_col = col
            break

    print(f"\n{title}:")
    
    for name, mask in [("Train", train_mask), ("Val", val_mask), ("Test", test_mask)]:
        n = mask.sum()
        dates = df.loc[mask, "_datetime"]
        date_range = f"{dates.min()} → {dates.max()}" if n > 0 else "empty"

        if label_col and n > 0:
            pos = df.loc[mask, label_col].sum()
            pct = 100 * pos / n
            print(f"  {name:5s}: {n:>10,} rows  |  {date_range}  |  {int(pos)} pos ({pct:.2f}%)")

        else:

            print(f"  {name:5s}: {n:>10,} rows  |  {date_range}")