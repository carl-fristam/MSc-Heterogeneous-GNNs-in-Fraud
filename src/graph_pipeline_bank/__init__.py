"""
Bank payment dataset graph pipeline.

Usage:
    from src.utils.config import load_config
    from src.graph_pipeline_bank import build_graph

    config = load_config("graph_bank_v1")   # loads configs/graph_bank_v1.yaml
    result = build_graph(config)

    data      = result["data"]        # PyG HeteroData
    node_maps = result["node_maps"]   # {node_type: {raw_id: int_index}}
    vocabs    = result["vocabs"]      # OHE vocabularies used during build
"""

from src.graph_pipeline_bank.builder import build_graph

__all__ = ["build_graph"]
