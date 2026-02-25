"""
load_data.py

Data loading for the VGAE anomaly detection pipeline.

Loads the v2 bipartite graph, creates stratified train/val/test splits,
and restricts the training set to genuine (non-laundering) transactions only.
"""

import torch
from torch import Tensor
from torch_geometric.data import HeteroData
from typing import Tuple

from src.data.saml_hetero_v2 import load_hetero_v2


def load_autoencoder_data(
    sample_ratio: float = 0.1,
    n_days: int = None,
    use_cache: bool = True,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
) -> Tuple[HeteroData, Tensor, Tensor, Tensor, Tensor]:
    """
    Load v2 graph and prepare splits for autoencoder training.

    Returns:
        data:          HeteroData (CPU)
        genuine_mask:  BoolTensor [Nt] — True where Is_laundering == 0
        idx_train:     LongTensor — genuine-only transaction indices for training
        idx_val:       LongTensor — mixed transaction indices for validation
        idx_test:      LongTensor — mixed transaction indices for test
    """
    data, _ = load_hetero_v2(
        sample_ratio=sample_ratio,
        n_days=n_days,
        use_cache=use_cache,
    )

    is_laundering = data['transaction'].is_laundering
    genuine_mask = (is_laundering == 0)

    # Stratified split preserving fraud ratio in val/test
    idx_train, idx_val, idx_test = _stratified_split(
        is_laundering, val_ratio, test_ratio, seed
    )

    # Restrict training to genuine transactions only
    idx_train = idx_train[genuine_mask[idx_train]]

    _print_stats(is_laundering, genuine_mask, idx_train, idx_val, idx_test)

    return data, genuine_mask, idx_train, idx_val, idx_test


def _stratified_split(
    is_laundering: Tensor,
    val_ratio: float,
    test_ratio: float,
    seed: int,
) -> Tuple[Tensor, Tensor, Tensor]:
    """Split transaction indices, preserving fraud ratio in each split."""
    torch.manual_seed(seed)

    fraud_idx = torch.where(is_laundering == 1)[0]
    genuine_idx = torch.where(is_laundering == 0)[0]

    def split(idx):
        perm = idx[torch.randperm(len(idx))]
        n_val = int(val_ratio * len(idx))
        n_test = int(test_ratio * len(idx))
        return perm[n_val + n_test:], perm[:n_val], perm[n_val:n_val + n_test]

    g_train, g_val, g_test = split(genuine_idx)
    f_train, f_val, f_test = split(fraud_idx)

    return (
        torch.cat([g_train, f_train]),
        torch.cat([g_val, f_val]),
        torch.cat([g_test, f_test]),
    )


def _print_stats(is_laundering, genuine_mask, idx_train, idx_val, idx_test):
    """Print split statistics."""
    total = len(is_laundering)
    total_fraud = int((is_laundering == 1).sum().item())
    print(f"\nSplit statistics (total: {total} txns, {total_fraud} fraud):")
    for name, idx in [("Train (genuine only)", idx_train), ("Val", idx_val), ("Test", idx_test)]:
        n = len(idx)
        fraud = int(is_laundering[idx].sum().item())
        print(f"  {name}: {n} txns — {fraud} fraud ({100*fraud/max(n,1):.2f}%)")
