"""
metapath_builder.py

Converts a PyG HeteroData object into metapath adjacency matrices
suitable for HGMAE's PreModel.

Metapaths are expressed as sequences of edge type triples:
    (src_node_type, relation, dst_node_type)

A 1-hop metapath is a single triple.
A 2-hop metapath is a list of two triples, composed via sparse matrix multiplication.

Example — SAML-D schema:
    METAPATHS = [
        [("account", "credit_card", "account")],
        [("account", "cross_border", "account")],
        [("account", "cash_deposit", "account"), ("account", "cross_border", "account")],
    ]

Example — bank data with richer schema:
    METAPATHS = [
        [("customer", "owns", "account")],
        [("account", "wire_transfer", "account")],
        [("customer", "owns", "account"), ("account", "wire_transfer", "account")],
    ]

Each metapath produces a sparse adjacency matrix of shape [N_src, N_dst]
where N_src/N_dst are the number of nodes of the respective types.
These are the `mps` tensors fed into PreModelPyG.
"""

import torch
from torch import Tensor
from torch_geometric.data import HeteroData
from typing import List, Tuple

# A metapath is a list of edge type triples, e.g.:
#   [("account", "credit_card", "account")]                          # 1-hop
#   [("account", "cash_deposit", "account"),
#    ("account", "cross_border", "account")]                         # 2-hop
Metapath = List[Tuple[str, str, str]]


# ---------------------------------------------------------------------------
# Default SAML-D metapaths — override by passing your own to build_metapath_adjs
# ---------------------------------------------------------------------------

SAML_METAPATHS: List[Metapath] = [
    [("account", "credit_card",     "account")],
    [("account", "debit_card",      "account")],
    [("account", "cheque",          "account")],
    [("account", "ach",             "account")],
    [("account", "cross_border",    "account")],
    [("account", "cash_withdrawal", "account")],
    [("account", "cash_deposit",    "account")],
    # 2-hop examples — uncomment when ready:
    # [("account", "cash_deposit", "account"), ("account", "cross_border", "account")],
    # [("account", "cross_border", "account"), ("account", "cash_withdrawal", "account")],
]


def build_metapath_adjs(
    data: HeteroData,
    metapaths: List[Metapath] = None,
) -> List[Tensor]:
    """
    Build a list of sparse metapath adjacency matrices from a HeteroData object.

    Args:
        data:      PyG HeteroData object with node features and edge indices.
        metapaths: List of metapaths to compute. Each metapath is a list of
                   (src_type, relation, dst_type) triples. If None, defaults
                   to SAML_METAPATHS.

    Returns:
        List of sparse tensors, one per metapath. Shape varies by metapath:
        [N_src, N_dst] where N_src/N_dst are node counts for the endpoint types.
        Passed directly as `mps` to PreModelPyG.
    """
    if metapaths is None:
        metapaths = SAML_METAPATHS

    adjs = []
    for mp in metapaths:
        adj = _compute_metapath_adj(data, mp)
        if adj is not None:
            adjs.append(adj)

    print(f"Built {len(adjs)} metapath adjacency matrices")
    return adjs


def _compute_metapath_adj(data: HeteroData, metapath: Metapath) -> Tensor:
    """
    Compute the adjacency matrix for a metapath by composing single-hop adjacencies.
    """
    adj = _single_hop_adj(data, metapath[0])
    if adj is None:
        return None

    for edge_triple in metapath[1:]:
        next_adj = _single_hop_adj(data, edge_triple)
        if next_adj is None:
            return None
        adj = torch.sparse.mm(adj, next_adj).coalesce()

    return adj


def _single_hop_adj(data: HeteroData, edge_triple: Tuple[str, str, str]) -> Tensor:
    """
    Build a sparse adjacency matrix for one edge type triple.

    Args:
        data:        HeteroData object
        edge_triple: (src_node_type, relation, dst_node_type)

    Returns:
        Sparse tensor of shape [N_src, N_dst], or None if edge type absent.
    """
    if edge_triple not in data.edge_types:
        return None

    src_type, _, dst_type = edge_triple
    edge_index = data[edge_triple].edge_index

    n_src = data[src_type].x.shape[0]
    n_dst = data[dst_type].x.shape[0]
    values = torch.ones(edge_index.shape[1])

    return torch.sparse_coo_tensor(edge_index, values, (n_src, n_dst)).coalesce()
