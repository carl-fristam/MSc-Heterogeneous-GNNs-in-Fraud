"""
L0 — Tabular baselines: Logistic Regression and XGBoost on transaction features.

No graph structure. Establishes the floor that graph-based models must beat.

Usage:
    from src.data.prepare import prepare_data
    from src.baselines.tabular import run_tabular_baselines

    prep = prepare_data(config)
    results = run_tabular_baselines(prep)
"""

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix,
)

from src.data.prepare import PreparedData
from src.utils.threshold_table import print_threshold_table


def _evaluate(y_true, y_prob, y_pred, name: str) -> dict:
    """Compute and print standard metrics."""
    metrics = {
        "model": name,
        "auroc": roc_auc_score(y_true, y_prob) if y_true.sum() > 0 else 0.0,
        "auprc": average_precision_score(y_true, y_prob) if y_true.sum() > 0 else 0.0,
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "confusion_matrix": confusion_matrix(y_true, y_pred),
    }

    print(f"\n{'='*50}")
    print(f"{name} — TEST RESULTS")
    print(f"{'='*50}")
    print(f"  AUROC:     {metrics['auroc']:.4f}")
    print(f"  AUPRC:     {metrics['auprc']:.4f}")
    print(f"  F1:        {metrics['f1']:.4f}")
    print(f"  Precision: {metrics['precision']:.4f}")
    print(f"  Recall:    {metrics['recall']:.4f}")
    print(f"  Confusion matrix:\n{metrics['confusion_matrix']}")

    return metrics


def run_logistic_regression(prep: PreparedData) -> dict:
    """Train and evaluate logistic regression on transaction features."""
    X, y = prep.txn_features, prep.labels
    train_m, test_m = prep.train_mask.values, prep.test_mask.values

    model = LogisticRegression(
        class_weight="balanced",
        max_iter=5000,
        solver="saga",
        n_jobs=-1,
    )
    model.fit(X[train_m], y[train_m])

    y_prob = model.predict_proba(X[test_m])[:, 1]
    y_pred = (y_prob >= 0.5).astype(int)

    return _evaluate(y[test_m], y_prob, y_pred, "Logistic Regression")


def run_xgboost(prep: PreparedData) -> dict:
    """Train and evaluate XGBoost on transaction features."""
    try:
        from xgboost import XGBClassifier
    except ImportError:
        print("xgboost not installed. Run: pip install xgboost")
        return {}

    X, y = prep.txn_features, prep.labels
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

    metrics = _evaluate(y[test_m], y_prob, y_pred, "XGBoost")

    base_val_col = prep.col_cfg.get("base_value")
    amounts = prep.df[base_val_col].values[test_m] if base_val_col else None
    print_threshold_table(y[test_m], y_prob, amounts=amounts, model_name="XGBoost")

    return metrics


def run_tabular_baselines(prep: PreparedData) -> list[dict]:
    """Run all L0 baselines and return results."""
    return [
        # run_logistic_regression(prep),
        run_xgboost(prep),
    ]
