"""
load_data.py

SAML-D data loader for HGMAE pre-training.

Wraps the existing HeteroData loader and metapath builder to produce
the exact inputs PreModelPyG expects:

    feats       - [node_features]  (list with one tensor, shape [N, F])
    mps         - metapath adjacency matrices (list of sparse [N, N] tensors)
    label       - node labels, shape [N]
    idx_train   - training node indices
    idx_val     - validation node indices
    idx_test    - test node indices
"""

import torch
from torch import Tensor
from typing import Tuple, List

from src.data.saml_hetero import load_hetero_saml_data
from src.hgmae.metapath_builder import build_metapath_adjs


def load_saml_for_hgmae(
    sample_ratio: float = 1.0,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    seed: int = 42,
    use_cache: bool = True,
) -> Tuple[List[Tensor], List[Tensor], Tensor, Tensor, Tensor, Tensor]:
    """
    Load SAML-D and prepare all inputs for PreModelPyG.

    Args:
        sample_ratio: Fraction of raw data to use (1.0 = full ~9.5M transactions)
        train_ratio:  Fraction of nodes for training
        val_ratio:    Fraction of nodes for validation (remainder goes to test)
        seed:         Random seed for reproducibility
        use_cache:    Whether to use cached HeteroData if available

    Returns:
        feats       - list containing one tensor of shape [N, F]
        mps         - list of sparse metapath adjacency tensors, each [N, N]
        label       - LongTensor of shape [N]
        idx_train   - LongTensor of training node indices
        idx_val     - LongTensor of validation node indices
        idx_test    - LongTensor of test node indices
    """
    # Step 1: load heterogeneous graph
    print("=" * 50)
    print("Loading SAML-D heterogeneous graph...")
    data, _ = load_hetero_saml_data(sample_ratio=sample_ratio, use_cache=use_cache)

    # Step 2: build metapath adjacency matrices
    print("\nBuilding metapath adjacency matrices...")
    mps = build_metapath_adjs(data)

    # Step 3: extract features and labels
    feats = [data['account'].x]     # list of one tensor — HGMAE expects a list
    label = data['account'].y

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
    """
    Split node indices into train/val/test, preserving the positive class ratio
    in each split (stratified). This is important because laundering nodes are
    only ~0.14% of all accounts.

    Args:
        label:       LongTensor of shape [N] with binary labels (0 or 1)
        train_ratio: Fraction of nodes for training
        val_ratio:   Fraction of nodes for validation
        seed:        Random seed

    Returns:
        idx_train, idx_val, idx_test — LongTensors of node indices

    

    """
    torch.manual_seed(seed)

    pos_idx = torch.where(label == 1)[0]
    neg_idx = torch.where(label == 0)[0]

    pos_idx = pos_idx[torch.randperm(len(pos_idx))]
    neg_idx = neg_idx[torch.randperm(len(neg_idx))]

    def split(idx):
        n_train = int(train_ratio * len(idx))
        n_val   = int(val_ratio   * len(idx))
        return idx[:n_train], idx[n_train:n_train + n_val], idx[n_train + n_val:]

    pos_train, pos_val, pos_test = split(pos_idx)
    neg_train, neg_val, neg_test = split(neg_idx)

    return (
        torch.cat([pos_train, neg_train]),
        torch.cat([pos_val,   neg_val]),
        torch.cat([pos_test,  neg_test]),
    )



def _print_split_stats(label, idx_train, idx_val, idx_test):
    """Print class distribution for each split."""
    for name, idx in [("Train", idx_train), ("Val", idx_val), ("Test", idx_test)]:
        n = len(idx)
        pos = label[idx].sum().item()
        print(f"  {name}: {n} nodes — {pos} positive ({100*pos/n:.2f}%)")
