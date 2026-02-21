"""
train.py

HGMAE pre-training loop for SAML-D.

Mirrors references/HGMAE/main.py but uses:
  - PreModelPyG       instead of PreModel        (no DGL)
  - load_saml_for_hgmae() instead of load_data() (our data pipeline)
  - src.utils.device  for device selection

Training is self-supervised: no labels are used during pre-training.
After pre-training, frozen embeddings are evaluated by fitting a logistic
regression on the train split and scoring on the test split.
"""

import datetime
import torch
import torch.nn as nn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
import numpy as np

from src.hgmae.premodel_adapter import PreModelPyG
from src.hgmae.load_data import load_saml_for_hgmae
from src.utils.device import get_device


def train(args):
    device = get_device()
    print(f"Device: {device}")

    # ------------------------------------------------------------------
    # Data
    # ------------------------------------------------------------------
    feats, mps, label, idx_train, idx_val, idx_test = load_saml_for_hgmae(
        sample_ratio=getattr(args, "sample_ratio", 1.0),
        use_cache=getattr(args, "use_cache", True),
    )

    feats = [f.to(device) for f in feats]
    mps   = [mp.to(device) for mp in mps]
    label = label.to(device)

    num_mp = len(mps)
    focused_feature_dim = feats[0].shape[1]
    print(f"Metapaths: {num_mp}  |  Node feature dim: {focused_feature_dim}")

    # ------------------------------------------------------------------
    # Model
    # ------------------------------------------------------------------
    model = PreModelPyG(args, num_mp, focused_feature_dim).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.lr, weight_decay=args.l2_coef
    )

    if getattr(args, "scheduler", False):
        scheduler = torch.optim.lr_scheduler.ExponentialLR(
            optimizer, gamma=args.scheduler_gamma
        )
    else:
        scheduler = None

    # ------------------------------------------------------------------
    # Pre-training loop
    # ------------------------------------------------------------------
    best_loss = float("inf")
    best_state = None
    cnt_wait = 0
    start = datetime.datetime.now()

    for epoch in range(args.mae_epochs):
        model.train()
        optimizer.zero_grad()

        loss, loss_item = model(feats, mps, epoch=epoch)
        loss.backward()
        optimizer.step()
        if scheduler:
            scheduler.step()

        print(f"Epoch {epoch:4d} | loss {loss_item:.4f} | "
              f"lr {optimizer.param_groups[0]['lr']:.2e}")

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
    print(f"\nPre-training complete in {elapsed}s. Best loss: {best_loss:.4f}")

    # ------------------------------------------------------------------
    # Extract embeddings with best model
    # ------------------------------------------------------------------
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        embeds = model.get_embeds(feats, mps).cpu().numpy()

    labels_np = label.cpu().numpy()

    # ------------------------------------------------------------------
    # Downstream evaluation: logistic regression on frozen embeddings
    # Standard protocol for self-supervised graph learning benchmarks.
    # ------------------------------------------------------------------
    # Baseline: LR on raw node features (no graph, no pre-training)
    # ------------------------------------------------------------------
    print("\nBaseline evaluation (logistic regression on raw node features)...")
    raw_feats = feats[0].cpu().numpy()
    labels_np = label.cpu().numpy()

    raw_scaler = StandardScaler()
    X_raw_train = raw_scaler.fit_transform(raw_feats[idx_train.cpu()])
    X_raw_val   = raw_scaler.transform(raw_feats[idx_val.cpu()])
    X_raw_test  = raw_scaler.transform(raw_feats[idx_test.cpu()])

    y_train = labels_np[idx_train.cpu()]
    y_val   = labels_np[idx_val.cpu()]
    y_test  = labels_np[idx_test.cpu()]

    clf_raw = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42)
    clf_raw.fit(X_raw_train, y_train)

    for split_name, X, y in [("Val", X_raw_val, y_val), ("Test", X_raw_test, y_test)]:
        preds = clf_raw.predict(X)
        probs = clf_raw.predict_proba(X)[:, 1]
        f1  = f1_score(y, preds, zero_division=0)
        auc = roc_auc_score(y, probs) if len(np.unique(y)) > 1 else 0.0
        print(f"  {split_name}: F1={f1:.4f}  AUC={auc:.4f}")

    # ------------------------------------------------------------------
    print("\nDownstream evaluation (logistic regression on frozen HGMAE embeddings)...")

    scaler = StandardScaler()
    X_train = scaler.fit_transform(embeds[idx_train.cpu()])
    X_val   = scaler.transform(embeds[idx_val.cpu()])
    X_test  = scaler.transform(embeds[idx_test.cpu()])

    y_train = labels_np[idx_train.cpu()]
    y_val   = labels_np[idx_val.cpu()]
    y_test  = labels_np[idx_test.cpu()]

    clf = LogisticRegression(
        max_iter=1000,
        class_weight="balanced",   # critical given 0.14% positive rate
        random_state=42,
    )
    clf.fit(X_train, y_train)

    for split_name, X, y in [("Val", X_val, y_val), ("Test", X_test, y_test)]:
        preds = clf.predict(X)
        probs = clf.predict_proba(X)[:, 1]
        f1  = f1_score(y, preds, zero_division=0)
        auc = roc_auc_score(y, probs) if len(np.unique(y)) > 1 else 0.0
        print(f"  {split_name}: F1={f1:.4f}  AUC={auc:.4f}")

    # ------------------------------------------------------------------
    # Optional: UMAP visualisation of embeddings
    # ------------------------------------------------------------------
    if getattr(args, "visualize", False):
        import os
        from src.hgmae.visualize import plot_umap
        os.makedirs("results", exist_ok=True)
        plot_umap(embeds, labels_np, save_path="results/hgmae_umap.png")

    return embeds, clf
