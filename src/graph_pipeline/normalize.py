import numpy as np
import pandas as pd
import torch

def zscore(values:np.ndarray) -> np.ndarray:
    """Normalize the values using z-score normalization."""
    
    values = values.astype(np.float32)
    mean = values.mean()
    std = values.std()
    if std > 0:
        return (values - mean) / std
    return np.zeros_like(values)

def zscore_tensor(tensor:torch.Tensor) -> torch.Tensor:
    """Normalize the values using z-score normalization."""
    
    mean = tensor.mean(dim=0)
    std = tensor.std(dim=0)
    std[std == 0] = 1.0  # Avoid division by zero
    return (tensor - mean) / std

def one_hot(series: pd.Series, vocab: list) -> np.ndarray:
    """One-hot encode the values in the series based on the provided vocabulary."""
    
    val_to_idx = {v: i for i, v in enumerate(vocab)}

    indices = series.map(val_to_idx).fillna(-1).astype(int)

    result = np.zeros((len(series), len(vocab)), dtype = np.float32)

    valid = indices >= 0

    result[valid, indices[valid]] = 1.0

    return result