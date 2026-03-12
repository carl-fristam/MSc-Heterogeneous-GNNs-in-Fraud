"""
Graph pipeline — extensible heterogeneous graph construction.

Usage:
    from src.utils.config import load_config
    from src.graph_pipeline import build_graph

    config = load_config("graph_pipeline")
    data, account_to_id = build_graph(config)
"""

from src.graph_pipeline.graph_builder import build_graph, print_feature_inventory, feature_table, inspect_features

__all__ = ["build_graph", "print_feature_inventory", "feature_table", "inspect_features"]
