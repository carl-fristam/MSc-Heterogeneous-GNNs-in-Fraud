"""
han_pyg.py

PyG-based replacement for the DGL HAN encoder used in HGMAE.

Wraps PyG's HANConv to match the calling interface PreModel expects:

    forward(gs, x, return_hidden=False) -> (embeddings, att_mp)

where gs is a list of edge_index tensors (one per metapath) and x is
the node feature tensor [N, F].

Internally, HANConv works with dicts:
    x_dict          = {"account": x}
    edge_index_dict = {("account", "mp_0", "account"): gs[0], ...}

We build those dicts on the fly so the rest of PreModel never needs to change.
"""

import torch
import torch.nn as nn
from torch import Tensor
from typing import List, Optional

from torch_geometric.nn import HANConv

# hgmae.utils is from the reference code — available via sys.path set in premodel_adapter.py.
# Import lazily inside __init__ if this module is ever used standalone.
try:
    from hgmae.utils import create_activation  # type: ignore[import-not-found]
except ImportError:
    import sys, os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../references/HGMAE"))
    from hgmae.utils import create_activation  # type: ignore[import-not-found]


class HANPyG(nn.Module):
    """
    PyG HANConv wrapper that matches the original DGL HAN interface.

    Two-level attention (as in the paper):
      1. Node-level:     GAT over each metapath-induced subgraph
      2. Semantic-level: learned softmax weighting across metapaths

    Both levels are handled internally by PyG's HANConv.
    """

    def __init__(
        self,
        num_metapath: int,
        in_dim: int,
        num_hidden: int,
        out_dim: int,
        num_layers: int,
        nhead: int,
        nhead_out: int,
        activation,
        feat_drop: float,
        attn_drop: float,
        negative_slope: float,
        residual: bool,
        norm,
        concat_out: bool = False,
        encoding: bool = False,
        node_type: str = "account",
    ):
        super().__init__()

        self.num_metapath = num_metapath
        self.encoding = encoding

        # One synthetic edge type per metapath — named mp_0, mp_1, ...
        # node_type defaults to "account" (v1) but can be "transaction" (v2).
        self._node_type = node_type
        self._edge_types = [
            (node_type, f"mp_{i}", node_type) for i in range(num_metapath)
        ]
        metadata = ([node_type], self._edge_types)

        self.han_layers = nn.ModuleList()
        self.norms = nn.ModuleList()

        # Dimension schedule: in_dim → num_hidden (×layers-1) → out_dim
        dims = [in_dim] + [num_hidden] * (num_layers - 1) + [out_dim]

        for i in range(num_layers):
            is_last = (i == num_layers - 1)
            heads = nhead_out if is_last else nhead

            self.han_layers.append(HANConv(
                in_channels=dims[i],
                out_channels=dims[i + 1],
                metadata=metadata,
                heads=heads,
                dropout=attn_drop,
                negative_slope=negative_slope,
            ))

            # Apply norm on every layer during encoding; skip on decoder's last layer
            # norm is already a constructor (e.g. nn.LayerNorm) passed in from outside,
            # NOT a string — do not call create_norm() again here.
            # HANConv outputs [N, out_channels] — heads are pooled internally, not concatenated.
            if norm is not None and (encoding or not is_last):
                self.norms.append(norm(dims[i + 1]))
            else:
                self.norms.append(nn.Identity())

        self.activation = create_activation(activation)
        self.last_activation = create_activation(activation) if encoding else create_activation(None)
        self.feat_drop = nn.Dropout(feat_drop)

        # HANConv pools across heads internally — output dim is just out_dim
        self.output_dim = out_dim

    def forward(
        self,
        gs: List[Tensor],
        x: Tensor,
        return_hidden: bool = False,
    ):
        """
        Args:
            gs:            list of edge_index tensors [2, E], one per metapath
            x:             node features [N, F]
            return_hidden: kept for interface compat with original HAN; returns
                           intermediate layer outputs if True

        Returns:
            (out [N, out_dim], att_mp)
            att_mp is None — PyG's HANConv does not expose semantic weights
            in its public API. Implement a custom SemanticAttention layer
            if these weights are needed for xAI (see H4 in hypotheses).
        """
        x = self.feat_drop(x)
        x_dict = {self._node_type: x}
        edge_index_dict = {et: gs[i] for i, et in enumerate(self._edge_types)}

        hidden_list = []
        for i, layer in enumerate(self.han_layers):
            x_dict = layer(x_dict, edge_index_dict)
            h = x_dict[self._node_type]
            h = self.norms[i](h)
            is_last = (i == len(self.han_layers) - 1)
            h = self.last_activation(h) if is_last else self.activation(h)
            x_dict[self._node_type] = h
            hidden_list.append(h)

        out = x_dict[self._node_type]

        if return_hidden:
            return out, hidden_list
        return out, None  # att_mp=None; semantic weights not exposed by HANConv
