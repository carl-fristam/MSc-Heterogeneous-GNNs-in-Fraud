"""
Normalisation utilities for the bank pipeline.
"""

import numpy as np
import pandas as pd
import torch


def zscore_cols(tensor: torch.Tensor) -> torch.Tensor:
    """Column-wise z-score normalisation. Columns with zero std are left as-is."""
    mean = tensor.mean(dim=0)
    std  = tensor.std(dim=0)
    std[std == 0] = 1.0
    return (tensor - mean) / std


def zscore(values: np.ndarray) -> np.ndarray:
    """Row-vector z-score."""
    values = values.astype(np.float32)
    std = values.std()
    return (values - values.mean()) / std if std > 0 else np.zeros_like(values)


def one_hot(series: pd.Series, vocab: list[str]) -> np.ndarray:
    """
    One-hot encode a Series given a fixed vocabulary.
    Values not in vocab produce an all-zeros row.
    """
    val_to_idx = {v: i for i, v in enumerate(vocab)}
    indices    = series.map(val_to_idx).fillna(-1).astype(int).values
    result     = np.zeros((len(series), len(vocab)), dtype=np.float32)
    valid      = indices >= 0
    result[valid, indices[valid]] = 1.0
    return result


def vocab_from_training(series: pd.Series, top_k: int | None = None) -> list[str]:
    """
    Derive a vocabulary from a training-data Series.
    Returns values sorted by frequency descending, optionally capped at top_k.
    Excludes NaN.
    """
    counts = series.dropna().value_counts()
    if top_k is not None:
        counts = counts.head(top_k)
    return counts.index.astype(str).tolist()


def one_hot_from_training(
    full_series: pd.Series,
    train_mask: pd.Series,
    top_k: int | None = None,
) -> tuple[np.ndarray, list[str]]:
    """
    Fit vocabulary on training rows, then OHE the full series.

    Returns:
        (encoded array shape (N, K), vocab list used)
    """
    vocab   = vocab_from_training(full_series[train_mask], top_k=top_k)
    encoded = one_hot(full_series.astype(str).fillna("__null__"), vocab)
    return encoded, vocab
