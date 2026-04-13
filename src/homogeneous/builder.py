"""
Homogeneous graph projection for L1 baselines.

Projects the heterogeneous HeteroData graph into a single-type PyG Data
object by:
  1. Stacking all node feature matrices (zero-padded to the same width)
     and appending a one-hot node-type indicator column.
  2. Concatenating all edge_index tensors with correct node-index offsets.
  3. Concatenating edge features, labels, and masks for all labelled edge types.

The resulting graph loses edge-type information but preserves all transaction
edges with their features and labels — exactly the comparison we want for
"hetero structure vs. same graph treated as homogeneous."

Usage:
    from src.graph_builder.assembler import build_graph
    from src.homogeneous.builder import project_to_homo

    hetero_result = build_graph(config, prep=prep)
    homo_result   = project_to_homo(hetero_result["data"])
"""

import torch
from torch_geometric.data import Data


def project_to_homo(hetero_data) -> Data:
    """
    Project a HeteroData graph to a homogeneous PyG Data object.

    Args:
        hetero_data: PyG HeteroData built by src.graph_builder.assembler

    Returns:
        PyG Data with fields:
            x               (N_total, max_feat_dim + num_node_types)
            num_nodes       int
            edge_index      (2, E_total)
            edge_attr       (E_total, F_edge)  — all labelled edges
            edge_y          (E_labelled,)
            edge_train_mask (E_labelled,)
            edge_val_mask   (E_labelled,)
            edge_test_mask  (E_labelled,)
    """
    node_types = hetero_data.node_types

    # ── Node feature stacking ─────────────────────────────────────────────────
    # Find the maximum feature dimension across node types.
    feat_dims = {nt: hetero_data[nt].x.shape[1] for nt in node_types}
    max_dim = max(feat_dims.values())

    node_offsets = {}
    x_parts = []
    offset = 0

    for i, nt in enumerate(node_types):
        x = hetero_data[nt].x          # (N_t, F_t)
        n, f = x.shape

        # Zero-pad to max_dim
        if f < max_dim:
            pad = torch.zeros(n, max_dim - f, device=x.device, dtype=x.dtype)
            x = torch.cat([x, pad], dim=1)

        # One-hot node-type indicator
        type_indicator = torch.zeros(n, len(node_types), device=x.device, dtype=x.dtype)
        type_indicator[:, i] = 1.0

        x_parts.append(torch.cat([x, type_indicator], dim=1))
        node_offsets[nt] = offset
        offset += n

    x_homo = torch.cat(x_parts, dim=0)  # (N_total, max_dim + num_node_types)

    # ── Edge concatenation ────────────────────────────────────────────────────
    ei_parts     = []
    ea_parts     = []
    y_parts      = []
    amount_parts = []
    train_parts  = []
    val_parts    = []
    test_parts   = []

    for et in hetero_data.edge_types:
        src_type, _, dst_type = et
        ei = hetero_data[et].edge_index.clone()
        ei[0] += node_offsets[src_type]
        ei[1] += node_offsets[dst_type]

        ea = getattr(hetero_data[et], "edge_attr", None)
        y  = getattr(hetero_data[et], "y", None)

        # Only include edge types that have features and labels —
        # otherwise edge_index and edge_attr dimensions won't align
        if ea is None or y is None:
            continue

        ei_parts.append(ei)
        ea_parts.append(ea)
        y_parts.append(y)
        train_parts.append(hetero_data[et].train_mask)
        val_parts.append(hetero_data[et].val_mask)
        test_parts.append(hetero_data[et].test_mask)
        amt = getattr(hetero_data[et], "amounts", None)
        if amt is not None:
            amount_parts.append(amt)

    data = Data()
    data.x = x_homo
    data.num_nodes = offset
    data.edge_index = torch.cat(ei_parts, dim=1)

    if ea_parts:
        data.edge_attr = torch.cat(ea_parts, dim=0)

    if y_parts:
        data.edge_y          = torch.cat(y_parts,     dim=0)
        data.edge_train_mask = torch.cat(train_parts, dim=0)
        data.edge_val_mask   = torch.cat(val_parts,   dim=0)
        data.edge_test_mask  = torch.cat(test_parts,  dim=0)
        if amount_parts:
            data.amounts = torch.cat(amount_parts, dim=0)

    _print_summary(data, node_offsets, hetero_data)
    return data


def _print_summary(data: Data, node_offsets: dict, hetero_data):
    print(f"\n{'='*60}")
    print("Homogeneous Projection Summary")
    print(f"{'='*60}")
    for nt, off in node_offsets.items():
        n = hetero_data[nt].num_nodes
        print(f"  {nt:<25} {n:>10,} nodes  (offset={off})")
    print(f"  {'TOTAL':<25} {data.num_nodes:>10,} nodes")
    print(f"  node feat dim:  {data.x.shape[1]}")
    print(f"  total edges:    {data.edge_index.shape[1]:,}")
    if hasattr(data, "edge_attr") and data.edge_attr is not None:
        print(f"  edge feat dim:  {data.edge_attr.shape[1]}")
    if hasattr(data, "edge_y") and data.edge_y is not None:
        y = data.edge_y
        for split, mask in [("train", data.edge_train_mask),
                             ("val",   data.edge_val_mask),
                             ("test",  data.edge_test_mask)]:
            n = mask.sum().item()
            pos = y[mask].sum().item()
            print(f"  {split:5s}: {n:>8,}  pos={int(pos):>6} ({100*pos/n:.2f}%)")
