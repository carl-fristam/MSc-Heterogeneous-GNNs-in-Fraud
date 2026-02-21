"""
premodel_adapter.py

Adapts the reference HGMAE PreModel (DGL-based) to run without DGL installed,
using PyG-compatible edge_index tensors instead of DGL graph objects.

Strategy:
  1. Inject a minimal DGL stub into sys.modules so that `import dgl` in the
     reference code does not crash at import time.
  2. Subclass PreModel and override only the methods that call DGL at runtime:
       - mps_to_gs()              converts metapath sparse tensors → graph objects
       - mask_mp_edge_reconstruction()  uses DropEdge and add_self_loop

Everything else (masking logic, loss computation, encoder/decoder) is inherited
unchanged from the reference implementation.
"""

import sys
import types
import torch
from torch import Tensor
from typing import List

# ---------------------------------------------------------------------------
# 1. DGL stub — inserted before any reference imports touch sys.modules
# ---------------------------------------------------------------------------

def _make_dgl_stub():
    """Build a minimal fake dgl module that satisfies reference import statements."""
    dgl = types.ModuleType("dgl")

    # dgl.graph() is called inside mps_to_gs() in the original PreModel.
    # Our override replaces mps_to_gs() entirely, so this never runs —
    # but the class body still needs the name to exist at definition time.
    dgl.graph = lambda *a, **kw: None
    dgl.add_self_loop = lambda g, *a, **kw: g

    # DropEdge is imported at module level: `from dgl import DropEdge`
    class _DropEdge:
        def __init__(self, p=0.0):
            self.p = p
        def __call__(self, g):
            return g  # no-op stub; real impl is in our subclass

    dgl.DropEdge = _DropEdge

    # Submodules referenced at import time in gat.py / han.py / gcn.py / gin.py
    for submod in ("ops", "function", "utils", "nn"):
        dgl.__dict__[submod] = types.ModuleType(f"dgl.{submod}")

    # expand_as_pair: imported from dgl.utils in gat/gcn/gin/dot_gat at module level.
    # Handles bipartite graphs — for our homogeneous case, src and dst features are the same.
    dgl.utils.expand_as_pair = lambda input_, g=None: (
        input_ if isinstance(input_, tuple) else (input_, input_)
    )

    # edge_softmax: imported from dgl.ops in gat.py at module level.
    # Only called during forward() which we replace — stub just needs to exist.
    dgl.ops.edge_softmax = lambda graph, logits: logits

    # dgl.nn.functional.edge_softmax: imported in dot_gat.py
    dgl.nn.functional = types.ModuleType("dgl.nn.functional")
    dgl.nn.functional.edge_softmax = lambda graph, logits: logits

    sys.modules["dgl"] = dgl
    sys.modules["dgl.ops"] = dgl.ops
    sys.modules["dgl.function"] = dgl.function
    sys.modules["dgl.utils"] = dgl.utils
    sys.modules["dgl.nn"] = dgl.nn
    sys.modules["dgl.nn.functional"] = dgl.nn.functional

_make_dgl_stub()

# ---------------------------------------------------------------------------
# 2. Now safe to import the reference PreModel
# ---------------------------------------------------------------------------

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../references/HGMAE"))

from hgmae.models.edcoder import PreModel  # noqa: E402


# ---------------------------------------------------------------------------
# 3. PyG-compatible adapter
# ---------------------------------------------------------------------------

class PreModelPyG(PreModel):
    """
    Replacement for PreModel that works without DGL.

    On init, calls super().__init__() to let the reference code set up all
    components (mask token, loss functions, projection layers), then immediately
    replaces self.encoder and self.decoder with PyG-based HANPyG instances.

    Only three methods are overridden:
        __init__                     — swaps encoder/decoder
        mps_to_gs()                  — sparse tensor → edge_index
        mask_mp_edge_reconstruction() — DropEdge via PyG
    """

    def __init__(self, args, num_metapath: int, focused_feature_dim: int):
        # Let reference code build everything: mask token, loss fns, projections
        super().__init__(args, num_metapath, focused_feature_dim)

        # HANConv (PyG) outputs `out_channels` directly — heads are pooled internally,
        # NOT concatenated like DGL's GAT. The reference assumed concat (16 * 4 = 64),
        # so we target hidden_dim straight to keep encoder_to_decoder(64→64) intact.
        enc_num_hidden = args.hidden_dim   # encoder output dim
        enc_nhead = args.num_heads         # attention heads (internal to HANConv)
        dec_num_hidden = args.hidden_dim   # decoder hidden dim
        dec_nhead = args.num_out_heads

        from src.hgmae.han_pyg import HANPyG
        from hgmae.utils import create_norm

        # Replace encoder with PyG HAN
        self.encoder = HANPyG(
            num_metapath=num_metapath,
            in_dim=focused_feature_dim,
            num_hidden=enc_num_hidden,
            out_dim=enc_num_hidden,
            num_layers=args.num_layers,
            nhead=enc_nhead,
            nhead_out=enc_nhead,
            activation=args.activation,
            feat_drop=args.feat_drop,
            attn_drop=args.attn_drop,
            negative_slope=args.negative_slope,
            residual=args.residual,
            norm=create_norm(args.norm),
            concat_out=True,
            encoding=True,
        )

        # Replace decoder with PyG HAN (1 layer, no activation — standard MAE decoder)
        if args.decoder == "han":
            self.decoder = HANPyG(
                num_metapath=num_metapath,
                in_dim=args.hidden_dim,
                num_hidden=dec_num_hidden,
                out_dim=focused_feature_dim,
                num_layers=1,
                nhead=enc_nhead,
                nhead_out=dec_nhead,
                activation=args.activation,
                feat_drop=args.feat_drop,
                attn_drop=args.attn_drop,
                negative_slope=args.negative_slope,
                residual=args.residual,
                norm=create_norm(args.norm),
                concat_out=True,
                encoding=False,
            )

    def mps_to_gs(self, mps: List[Tensor]) -> List[Tensor]:
        """
        Convert a list of metapath sparse adjacency matrices to a list of
        edge_index tensors (shape [2, num_edges]), one per metapath.

        Original (DGL):
            builds a dgl.graph((row_indices, col_indices)) per metapath

        Ours (PyG):
            extracts the non-zero indices from the sparse tensor directly
            and returns them as a [2, E] LongTensor — the standard PyG format.

        Args:
            mps: list of torch sparse tensors, each of shape [N, N]

        Returns:
            list of edge_index tensors, each of shape [2, E]
        """        

        return [mp._indices() for mp in mps]

    def mask_mp_edge_reconstruction(self, feat: Tensor, mps: List[Tensor], epoch) -> Tensor:
        """
        Edge reconstruction pretraining task — re-implemented without DGL's DropEdge.

        Randomly removes a fraction of edges from each metapath edge_index,
        encodes the corrupted graph, reconstructs the adjacency via dot product,
        and computes loss against the original adjacency.
        """
        from torch_geometric.utils import add_self_loops, dropout_edge

        edge_index_list = self.mps_to_gs(mps)
        cur_rate = self.get_mask_rate(self.mp_edge_mask_rate, epoch=epoch)

        masked_edge_index_list = []
        for ei in edge_index_list:
            ei_dropped, _ = dropout_edge(ei, p=cur_rate, training=self.training)
            num_nodes = mps[0].shape[0]
            ei_dropped, _ = add_self_loops(ei_dropped, num_nodes=num_nodes)
            masked_edge_index_list.append(ei_dropped)

        enc_rep, _ = self.encoder(masked_edge_index_list, feat, return_hidden=False)
        rep = self.encoder_to_decoder_edge_recon(enc_rep)

        if self.decoder_type == "mlp":
            feat_recon = self.decoder(rep)
            att_mp = [1.0 / len(mps)] * len(mps)
        else:
            feat_recon, att_mp = self.decoder(masked_edge_index_list, rep)

        gs_recon = torch.mm(feat_recon, feat_recon.T)

        loss = None
        for i in range(len(mps)):
            w = att_mp[i]
            mp_dense = mps[i].to_dense()
            term = w * self.mp_edge_recon_loss(gs_recon, mp_dense)
            loss = term if loss is None else loss + term

        return loss
