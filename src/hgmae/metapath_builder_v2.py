"""
metapath_builder_v2.py

Transaction-level metapath adjacency matrices for the v2 bipartite graph.

All metapaths are 2-hop: transaction -> account -> transaction
This captures how transactions are linked through shared accounts
(same sender or same receiver), which is essential for detecting
laundering chains.

Four metapaths:
  1. sent_by + sends:       txn_i shares sender with txn_j
  2. sent_by + receives:    txn_i's sender is txn_j's receiver (chain link!)
  3. received_by + sends:   txn_i's receiver is txn_j's sender (chain link!)
  4. received_by + receives: txn_i shares receiver with txn_j

Metapaths 2 and 3 are the most important for AML — they capture the
directional flow of funds through accounts (A sends to B, B sends to C).
"""

import torch
from torch import Tensor
from torch_geometric.data import HeteroData
from typing import List, Tuple

Metapath = List[Tuple[str, str, str]]

# Transaction-level metapaths via account nodes
TXN_METAPATHS: List[Metapath] = [
    # Same sender (co-sending transactions)
    [("transaction", "sent_by", "account"), ("account", "sends", "transaction")],
    # Chain forward: txn_i's sender is txn_j's receiver
    [("transaction", "sent_by", "account"), ("account", "receives", "transaction")],
    # Chain backward: txn_i's receiver is txn_j's sender
    [("transaction", "received_by", "account"), ("account", "sends", "transaction")],
    # Same receiver (co-receiving transactions)
    [("transaction", "received_by", "account"), ("account", "receives", "transaction")],
]


def build_metapath_adjs_v2(
    data: HeteroData,
    metapaths: List[Metapath] = None,
) -> List[Tensor]:
    """
    Build sparse [N_txn, N_txn] metapath adjacency matrices from v2 HeteroData.

    Args:
        data:      PyG HeteroData with account and transaction nodes (v2 schema)
        metapaths: List of 2-hop metapaths. Defaults to TXN_METAPATHS.

    Returns:
        List of sparse tensors, each [N_txn, N_txn].
    """
    if metapaths is None:
        metapaths = TXN_METAPATHS

    adjs = []
    for mp in metapaths:
        adj = _compute_metapath_adj(data, mp)
        if adj is not None:
            adjs.append(adj)

    print(f"Built {len(adjs)} transaction-level metapath adjacency matrices")
    return adjs


def _compute_metapath_adj(data: HeteroData, metapath: Metapath) -> Tensor:
    """Compose single-hop adjacencies via sparse matrix multiplication."""
    adj = _single_hop_adj(data, metapath[0])
    if adj is None:
        return None

    for edge_triple in metapath[1:]:
        next_adj = _single_hop_adj(data, edge_triple)
        if next_adj is None:
            return None
        adj = torch.sparse.mm(adj, next_adj).coalesce()

    # Remove self-loops (a transaction connecting to itself through an account)
    indices = adj.indices()
    values = adj.values()
    mask = indices[0] != indices[1]
    indices = indices[:, mask]
    values = values[mask]

    # Binarise (we only care about connectivity, not multiplicity)
    values = torch.ones_like(values)

    return torch.sparse_coo_tensor(
        indices, values, adj.shape
    ).coalesce()


def _single_hop_adj(data: HeteroData, edge_triple: Tuple[str, str, str]) -> Tensor:
    """Build sparse adjacency for one edge type."""
    if edge_triple not in data.edge_types:
        return None

    src_type, _, dst_type = edge_triple
    edge_index = data[edge_triple].edge_index

    n_src = data[src_type].x.shape[0]
    n_dst = data[dst_type].x.shape[0]
    values = torch.ones(edge_index.shape[1])

    return torch.sparse_coo_tensor(edge_index, values, (n_src, n_dst)).coalesce()
