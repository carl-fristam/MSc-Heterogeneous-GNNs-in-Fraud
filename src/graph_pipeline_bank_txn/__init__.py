"""
Bank payment dataset — transactions-as-nodes pipeline.

Mirrors the xFraud architecture: transactions and accounts are both node types.
Labels live on Transaction nodes (node classification, not edge classification).

Usage:
    from src.utils.config import load_config
    from src.graph_pipeline_bank_txn import build_graph

    config = load_config("graph_bank_txn_v1")
    result = build_graph(config)

    data      = result["data"]        # PyG HeteroData
    node_maps = result["node_maps"]   # {node_type: {raw_id: int_index}}
    vocabs    = result["vocabs"]      # OHE vocabularies used during build

Graph topology:
    InternalAccount ──[sends]──────────────► Transaction
    Transaction     ──[received_by_internal]► InternalAccount   (TRANSACTIONONUS=True)
    Transaction     ──[received_by_external]► ExternalAccount   (TRANSACTIONONUS=False)
"""

from src.graph_pipeline_bank_txn.builder import build_graph

__all__ = ["build_graph"]
