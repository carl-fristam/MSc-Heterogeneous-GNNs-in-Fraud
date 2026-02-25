"""
load_data_v2.py

Transaction-level data loader for HGMAE pre-training using the v2 bipartite graph.

Wraps load_hetero_v2 and metapath_builder_v2 to produce the exact inputs
PreModelPyG expects:

    feats       - [transaction_features]  (list with one tensor, shape [N_txn, 76])
    mps         - metapath adjacency matrices (list of sparse [N_txn, N_txn] tensors)
    label       - transaction labels, shape [N_txn]
    idx_train   - training transaction indices
    idx_val     - validation transaction indices
    idx_test    - test transaction indices
"""

import torch
from torch import Tensor
from typing import Tuple, List

from src.data.saml_hetero_v2 import load_hetero_v2
from src.hgmae.metapath_builder_v2 import build_metapath_adjs_v2


def load_saml_v2_for_hgmae(
    sample_ratio: float = 0.10,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    seed: int = 42,
    use_cache: bool = True,
) -> Tuple[List[Tensor], List[Tensor], Tensor, Tensor, Tensor, Tensor]:
    """
    Load SAML-D v2 bipartite graph and prepare transaction-level inputs for HGMAE.

    Returns:
        feats       - list containing one tensor [N_txn, 76]
        mps         - list of sparse [N_txn, N_txn] metapath adjacency tensors
        label       - LongTensor [N_txn]
        idx_train   - LongTensor of training indices
        idx_val     - LongTensor of validation indices
        idx_test    - LongTensor of test indices
    """
    # Step 1: load bipartite graph
    print("=" * 50)
    print("Loading SAML-D v2 bipartite graph...")
    data, _ = load_hetero_v2(sample_ratio=sample_ratio, use_cache=use_cache)

    # Step 2: build transaction-level metapath adjacencies
    print("\nBuilding transaction-level metapath adjacencies...")
    mps = build_metapath_adjs_v2(data)

    # Step 3: extract transaction features and labels
    feats = [data['transaction'].x]
    label = data['transaction'].is_laundering.long()

    # Step 4: stratified train/val/test split
    print("\nCreating stratified splits...")
    idx_train, idx_val, idx_test = _stratified_split(
        label, train_ratio, val_ratio, seed
    )

    _print_split_stats(label, idx_train, idx_val, idx_test)

    return feats, mps, label, idx_train, idx_val, idx_test


def _stratified_split(
    label: Tensor,
    train_ratio: float,
    val_ratio: float,
    seed: int,
) -> Tuple[Tensor, Tensor, Tensor]:
    """Stratified split preserving positive class ratio in each split."""
    torch.manual_seed(seed)

    pos_idx = torch.where(label == 1)[0]
    neg_idx = torch.where(label == 0)[0]

    pos_idx = pos_idx[torch.randperm(len(pos_idx))]
    neg_idx = neg_idx[torch.randperm(len(neg_idx))]

    def split(idx):
        n_train = int(train_ratio * len(idx))
        n_val = int(val_ratio * len(idx))
        return idx[:n_train], idx[n_train:n_train + n_val], idx[n_train + n_val:]

    pos_train, pos_val, pos_test = split(pos_idx)
    neg_train, neg_val, neg_test = split(neg_idx)

    return (
        torch.cat([pos_train, neg_train]),
        torch.cat([pos_val, neg_val]),
        torch.cat([pos_test, neg_test]),
    )


def _print_split_stats(label, idx_train, idx_val, idx_test):
    """Print class distribution for each split."""
    for name, idx in [("Train", idx_train), ("Val", idx_val), ("Test", idx_test)]:
        n = len(idx)
        pos = label[idx].sum().item()
        print(f"  {name}: {n} nodes — {pos} positive ({100*pos/n:.2f}%)")
