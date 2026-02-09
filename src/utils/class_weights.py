"""
Inverse-frequency class weights for imbalanced datasets.
"""

import torch


def compute_class_weights(labels: torch.Tensor) -> torch.Tensor:
    """
    Compute class weights inversely proportional to class frequency.

    Args:
        labels: Tensor of integer class labels

    Returns:
        Tensor of per-class weights (length = number of unique classes)
    """
    unique, counts = torch.unique(labels, return_counts=True)
    weights = 1.0 / counts.float()
    weights = weights / weights.sum() * len(unique)
    return weights
