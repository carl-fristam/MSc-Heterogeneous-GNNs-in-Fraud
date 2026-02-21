"""
scripts/train_hgmae_viz.py

HGMAE pre-training + UMAP visualisation. No downstream evaluation.

Usage:
    python scripts/train_hgmae_viz.py
    python scripts/train_hgmae_viz.py --sample_ratio 0.01 --mae_epochs 5
    python scripts/train_hgmae_viz.py --mae_epochs 200 --out results/umap_full.png
"""

import argparse
import datetime
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from src.hgmae.premodel_adapter import PreModelPyG
from src.hgmae.load_data import load_saml_for_hgmae
from src.hgmae.visualize import plot_umap
from src.utils.device import get_device


def build_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample_ratio", type=float, default=1.0)
    parser.add_argument("--use_cache",    action="store_true", default=True)
    parser.add_argument("--mae_epochs",   type=int,   default=200)
    parser.add_argument("--hidden_dim",   type=int,   default=64)
    parser.add_argument("--num_layers",   type=int,   default=2)
    parser.add_argument("--num_heads",    type=int,   default=4)
    parser.add_argument("--num_out_heads",type=int,   default=1)
    parser.add_argument("--activation",   type=str,   default="prelu")
    parser.add_argument("--norm",         type=str,   default="layernorm")
    parser.add_argument("--residual",     action="store_true", default=True)
    parser.add_argument("--feat_drop",    type=float, default=0.2)
    parser.add_argument("--attn_drop",    type=float, default=0.2)
    parser.add_argument("--negative_slope", type=float, default=0.2)
    parser.add_argument("--feat_mask_rate",  type=str,   default="0.3")
    parser.add_argument("--replace_rate",    type=float, default=0.1)
    parser.add_argument("--leave_unchanged", type=float, default=0.1)
    parser.add_argument("--loss_fn",         type=str,   default="sce")
    parser.add_argument("--alpha_l",         type=float, default=3.0)
    parser.add_argument("--use_mp_edge_recon",         action="store_true", default=False)
    parser.add_argument("--mp_edge_mask_rate",         type=str,   default="0.3")
    parser.add_argument("--mp_edge_recon_loss_weight", type=float, default=1.0)
    parser.add_argument("--mp_edge_alpha_l",           type=float, default=3.0)
    parser.add_argument("--use_mp2vec_feat_pred",      action="store_true", default=False)
    parser.add_argument("--mps_embedding_dim",         type=int,   default=64)
    parser.add_argument("--mp2vec_feat_pred_loss_weight", type=float, default=1.0)
    parser.add_argument("--mp2vec_feat_alpha_l",       type=float, default=3.0)
    parser.add_argument("--mp2vec_feat_drop",          type=float, default=0.2)
    parser.add_argument("--lr",       type=float, default=1e-3)
    parser.add_argument("--l2_coef",  type=float, default=1e-4)
    parser.add_argument("--patience", type=int,   default=20)
    parser.add_argument("--scheduler",       action="store_true", default=False)
    parser.add_argument("--scheduler_gamma", type=float, default=0.99)
    parser.add_argument("--encoder", type=str, default="han")
    parser.add_argument("--decoder", type=str, default="han")
    parser.add_argument("--out", type=str, default="results/hgmae_umap.png",
                        help="Output path for UMAP plot")
    return parser.parse_args()


if __name__ == "__main__":
    args = build_args()
    device = get_device()
    print(f"Device: {device}")

    # Data
    feats, mps, label, idx_train, _, _ = load_saml_for_hgmae(
        sample_ratio=args.sample_ratio,
        use_cache=args.use_cache,
    )
    feats = [f.to(device) for f in feats]
    mps   = [mp.to(device) for mp in mps]
    label = label.to(device)

    num_mp = len(mps)
    feat_dim = feats[0].shape[1]
    print(f"Metapaths: {num_mp}  |  Feature dim: {feat_dim}")

    # Model
    model = PreModelPyG(args, num_mp, feat_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.l2_coef)

    if args.scheduler:
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=args.scheduler_gamma)

    # Training
    best_loss, best_state, cnt_wait = float("inf"), None, 0
    start = datetime.datetime.now()

    for epoch in range(args.mae_epochs):
        model.train()
        optimizer.zero_grad()
        loss, loss_item = model(feats, mps, epoch=epoch)
        loss.backward()
        optimizer.step()
        if args.scheduler:
            scheduler.step()

        print(f"Epoch {epoch:4d} | loss {loss_item:.4f}")

        if loss_item < best_loss:
            best_loss = loss_item
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            cnt_wait = 0
        else:
            cnt_wait += 1

        if cnt_wait >= args.patience:
            print(f"Early stopping at epoch {epoch}")
            break

    elapsed = (datetime.datetime.now() - start).seconds
    print(f"\nDone in {elapsed}s. Best loss: {best_loss:.4f}")

    # Embeddings
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        embeds = model.get_embeds(feats, mps).cpu().numpy()

    labels_np = label.cpu().numpy()

    # Save embeddings + labels so UMAP can be re-run without retraining
    os.makedirs("results", exist_ok=True)
    import numpy as np
    np.save("results/hgmae_embeds.npy", embeds)
    np.save("results/hgmae_labels.npy", labels_np)
    print("Embeddings saved → results/hgmae_embeds.npy")

    # UMAP
    os.makedirs(os.path.dirname(args.out) or "results", exist_ok=True)
    plot_umap(embeds, labels_np, save_path=args.out)
