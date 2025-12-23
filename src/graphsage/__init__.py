"""
GraphSAGE package for AML detection.
"""

from .data import load_and_prepare_data
from .model import GraphSAGE
from .train import create_splits, train_model, evaluate

__all__ = [
    'load_and_prepare_data',
    'GraphSAGE',
    'create_splits',
    'train_model',
    'evaluate'
]
