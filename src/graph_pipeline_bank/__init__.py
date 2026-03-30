"""
Bank payment dataset graph pipeline.

Usage:
    from src.utils.config import load_variant
    from src.graph_pipeline_bank import build_graph

    config = load_variant("v1")   # loads variant from configs/master.yaml
    result = build_graph(config)

    data      = result["data"]        # PyG HeteroData
    node_maps = result["node_maps"]   # {node_type: {raw_id: int_index}}
    vocabs    = result["vocabs"]      # OHE vocabularies used during build
"""

from src.graph_pipeline_bank.builder import build_graph

__all__ = ["build_graph"]
