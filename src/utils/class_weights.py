"""
Class weight computation for imbalanced binary labels.
"""

import torch


def compute_class_weights(y: torch.Tensor) -> torch.Tensor:
    """
    Compute inverse-frequency class weights for binary labels.

    Returns a (2,) tensor [weight_neg, weight_pos] suitable for
    torch.nn.BCEWithLogitsLoss(pos_weight=...) or CrossEntropyLoss(weight=...).

    Args:
        y: 1-D float or long tensor of binary labels (0/1)

    Returns:
        tensor([n / (2 * n_neg), n / (2 * n_pos)])
    """
    y = y.float()
    n = len(y)
    n_pos = y.sum().item()
    n_neg = n - n_pos

    if n_pos == 0 or n_neg == 0:
        return torch.ones(2)

    w_neg = n / (2.0 * n_neg)
    w_pos = n / (2.0 * n_pos)
    return torch.tensor([w_neg, w_pos], dtype=torch.float32)
