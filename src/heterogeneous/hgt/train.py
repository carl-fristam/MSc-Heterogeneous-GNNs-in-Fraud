"""
HGT training — thin wrapper around the unified Trainer.

Kept for backwards compatibility. Prefer using src.training.trainer directly.
"""

from src.heterogeneous.hgt.model import HGT
from src.training.trainer import Trainer, TrainConfig
from src.utils.device import get_device


def train(data, task="node", target_node_type="transaction", **kwargs):
    """
    Train HGT on a HeteroData graph.

    Args:
        data: PyG HeteroData
        task: "node" or "edge"
        target_node_type: which node type has labels (node mode)
        **kwargs: overrides for TrainConfig fields

    Returns:
        dict of test metrics
    """
    device = get_device()

    config = TrainConfig(
        task=task,
        graph_type="hetero",
        target_node_type=target_node_type,
        **kwargs,
    )

    model = HGT(
        data,
        hidden_dim=kwargs.get("hidden_dim", 64),
        num_heads=kwargs.get("num_heads", 4),
        num_layers=kwargs.get("num_layers", 2),
        dropout=config.dropout,
        task=task,
        target_node_type=target_node_type,
    )

    trainer = Trainer(model, data, config, device)
    return trainer.run()
