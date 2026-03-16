"""
Device selection utility.
"""

import torch


def get_device() -> torch.device:
    """Return MPS (Apple Silicon), CUDA, or CPU — in that priority order."""
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")
