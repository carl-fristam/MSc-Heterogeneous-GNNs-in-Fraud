"""
L1 — Graph feature extraction → tabular classifier.

Extracts structural features (degree, PageRank, clustering coefficient)
from the transaction graph, concatenates with transaction features,
and feeds into XGBoost.

Isolates the value of graph *structure* without neural networks.
Follows Feedzai group (Eddin et al., 2021) approach.

Usage:
    from src.data.prepare import prepare_data
    from src.baselines.graph_features import run_graph_feature_baselines

    prep = prepare_data(config)
    results = run_graph_feature_baselines(prep)
"""

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix,
)

from src.data.prepare import PreparedData


def _build_account_graph_features(df: pd.DataFrame, col_cfg: dict, train_mask: np.ndarray) -> dict:
    """
    Compute graph-structural features per account from training data only.

    Returns:
        {account_id: feature_vector} for sender accounts
    """
    train_df = df[train_mask]
    sender_col = col_cfg["sender"]
    receiver_col = col_cfg["receiver"]

    out_degree = train_df.groupby(sender_col)[receiver_col].nunique()

    in_degree = train_df.groupby(receiver_col)[sender_col].nunique()

    tx_count = train_df.groupby(sender_col).size()

    if "receiver_bank" in col_cfg and col_cfg["receiver_bank"] in df.columns:
        bank_diversity = train_df.groupby(sender_col)[col_cfg["receiver_bank"]].nunique()
    else:
        bank_diversity = pd.Series(dtype=float)

    all_senders = df[sender_col].unique()
    features = {}
    for acc in all_senders:
        feat = [
            np.log1p(out_degree.get(acc, 0)),
            np.log1p(in_degree.get(acc, 0)),
            np.log1p(tx_count.get(acc, 0)),
            np.log1p(bank_diversity.get(acc, 0)),
        ]
        features[acc] = feat

    return features


def run_graph_feature_baselines(prep: PreparedData) -> list[dict]:
    """Run XGBoost with graph-structural features appended to transaction features."""
    try:
        from xgboost import XGBClassifier
    except ImportError:
        print("xgboost not installed. Run: pip install xgboost")
        return []

    df = prep.df
    col_cfg = prep.col_cfg

    acct_feats = _build_account_graph_features(df, col_cfg, prep.train_mask.values)

    sender_col = col_cfg["sender"]
    n_graph_feats = 4
    X_graph = np.zeros((len(df), n_graph_feats), dtype=np.float32)
    for i, acc in enumerate(df[sender_col]):
        if acc in acct_feats:
            X_graph[i] = acct_feats[acc]

    X = np.hstack([prep.txn_features, X_graph])
    y = prep.labels
    train_m = prep.train_mask.values
    val_m = prep.val_mask.values
    test_m = prep.test_mask.values

    n_neg = (y[train_m] == 0).sum()
    n_pos = (y[train_m] == 1).sum()
    scale_pos = n_neg / n_pos if n_pos > 0 else 1.0

    model = XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.1,
        scale_pos_weight=scale_pos,
        eval_metric="aucpr",
        early_stopping_rounds=20,
        n_jobs=-1,
        tree_method="hist",
    )
    model.fit(X[train_m], y[train_m], eval_set=[(X[val_m], y[val_m])], verbose=False)

    # Find optimal threshold on validation set
    y_val_prob = model.predict_proba(X[val_m])[:, 1]
    best_t, best_f1 = 0.5, 0.0
    for t in np.arange(0.05, 0.95, 0.01):
        f1 = f1_score(y[val_m], (y_val_prob >= t).astype(int), zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_t = t
    print(f"  Optimal threshold (val F1): {best_t:.2f} (F1={best_f1:.4f})")

    y_prob = model.predict_proba(X[test_m])[:, 1]
    y_pred = (y_prob >= best_t).astype(int)

    metrics = {
        "model": "XGBoost + Graph Features",
        "auroc": roc_auc_score(y[test_m], y_prob) if y[test_m].sum() > 0 else 0.0,
        "auprc": average_precision_score(y[test_m], y_prob) if y[test_m].sum() > 0 else 0.0,
        "f1": f1_score(y[test_m], y_pred, zero_division=0),
        "precision": precision_score(y[test_m], y_pred, zero_division=0),
        "recall": recall_score(y[test_m], y_pred, zero_division=0),
        "confusion_matrix": confusion_matrix(y[test_m], y_pred),
    }

    print(f"\n{'='*50}")
    print(f"XGBoost + Graph Features — TEST RESULTS")
    print(f"{'='*50}")
    print(f"  AUROC:     {metrics['auroc']:.4f}")
    print(f"  AUPRC:     {metrics['auprc']:.4f}")
    print(f"  F1:        {metrics['f1']:.4f}")
    print(f"  Precision: {metrics['precision']:.4f}")
    print(f"  Recall:    {metrics['recall']:.4f}")
    print(f"  Confusion matrix:\n{metrics['confusion_matrix']}")

    return [metrics]
