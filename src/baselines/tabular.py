"""
Usage:
    from src.data.prepare import prepare_data
    from src.baselines.tabular import run_tabular_baselines

    prep = prepare_data(config)
    results = run_tabular_baselines(prep)
"""

import numpy as np
import pandas as pd
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


def _tabular_X(prep: PreparedData,
               feature_cols: list[str] | None = None) -> tuple[np.ndarray, list[str]]:
    """
    Build the feature matrix for tabular models.

    If *feature_cols* is given, use exactly those columns (must be present in df).
    Otherwise fall back to all numeric columns minus identifiers/label/timestamp.
    Returns (feature_matrix, feature_names).
    """
    if feature_cols is not None:
        missing = [c for c in feature_cols if c not in prep.df.columns]
        if missing:
            raise KeyError(f"Feature columns not found in dataframe: {missing}")
        num_cols = feature_cols
    else:
        col_cfg = prep.col_cfg
        exclude = {
            col_cfg.get("label"),
            col_cfg.get("sender"),
            col_cfg.get("receiver"),
            col_cfg.get("timestamp"),
            col_cfg.get("transaction_id"),
            col_cfg.get("customer_id"),
            "COUNTERBRANCHID",
            "_sender", "_receiver",
        }
        exclude.discard(None)
        num_cols = [c for c in prep.df.columns
                    if c not in exclude and pd.api.types.is_numeric_dtype(prep.df[c])]
    return prep.df[num_cols].fillna(0).values.astype(np.float32), num_cols

# Hyperparameter search space for Bayesian optimisation
_XGB_SEARCH_SPACE = {
    "max_depth":        ("int",   3,    10),
    "learning_rate":    ("float", 0.01, 0.3,  {"log": True}),
    "n_estimators":     ("int",   100,  600),
    "subsample":        ("float", 0.6,  1.0),
    "colsample_bytree": ("float", 0.6,  1.0),
    "min_child_weight": ("int",   1,    10),
    "gamma":            ("float", 0.0,  5.0),
    "reg_alpha":        ("float", 1e-8, 1.0,  {"log": True}),
    "reg_lambda":       ("float", 1e-8, 3.0,  {"log": True}),
}


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

def run_xgboost(prep: PreparedData,
                feature_cols: list[str] | None = None) -> dict:
    """Train and evaluate XGBoost on transaction features."""
    try:
        from xgboost import XGBClassifier
    except ImportError:
        print("xgboost not installed. Run: pip install xgboost")
        return {}

    (X, feat_names), y = _tabular_X(prep, feature_cols), prep.labels
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
    val_auprc = average_precision_score(y[val_m], y_val_prob)
    val_auroc = roc_auc_score(y[val_m], y_val_prob)
    print(f"  Val AUPRC: {val_auprc:.4f}")
    print(f"  Val AUROC: {val_auroc:.4f}")
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
    metrics["threshold_table"] = print_threshold_table(
        y[test_m], y_prob, amounts=amounts, model_name="XGBoost",
        optimal_threshold=best_t,
    )
    metrics["_y_true"] = y[test_m]
    metrics["_y_prob"] = y_prob
    metrics["_xgb_model"] = model
    metrics["_feature_names"] = feat_names

    return metrics


def run_xgboost_bayes(prep: PreparedData, n_trials: int = 50,
                      feature_cols: list[str] | None = None) -> dict:
    """
    XGBoost with Bayesian hyperparameter optimisation via optuna.

    Runs n_trials smart trials on the validation set (maximising AUPRC),
    then trains a final model with the best params and evaluates on test.
    """
    try:
        import optuna
        from xgboost import XGBClassifier
    except ImportError as e:
        print(f"Missing dependency: {e}. Run: pip install optuna xgboost")
        return {}

    (X, feat_names), y = _tabular_X(prep, feature_cols), prep.labels
    train_m = prep.train_mask.values
    val_m   = prep.val_mask.values
    test_m  = prep.test_mask.values

    n_neg     = (y[train_m] == 0).sum()
    n_pos     = (y[train_m] == 1).sum()
    scale_pos = n_neg / n_pos if n_pos > 0 else 1.0

    def _suggest(trial, name, spec):
        kind = spec[0]
        if kind == "int":
            return trial.suggest_int(name, spec[1], spec[2])
        kwargs = spec[3] if len(spec) > 3 else {}
        return trial.suggest_float(name, spec[1], spec[2], **kwargs)

    def objective(trial):
        params = {name: _suggest(trial, name, spec) for name, spec in _XGB_SEARCH_SPACE.items()}
        params.update({"scale_pos_weight": scale_pos, "tree_method": "hist",
                       "eval_metric": "aucpr", "n_jobs": -1})
        model = XGBClassifier(**params)
        model.fit(X[train_m], y[train_m],
                  eval_set=[(X[val_m], y[val_m])],
                  verbose=False)
        return average_precision_score(y[val_m], model.predict_proba(X[val_m])[:, 1])

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction="maximize",
                                sampler=optuna.samplers.TPESampler(seed=42))

    print(f"\nBayesian optimisation: {n_trials} trials on val AUPRC ...")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    print(f"\n  Best val AUPRC : {study.best_value:.4f}")
    print(f"  Best params    :")
    for k, v in study.best_params.items():
        print(f"    {k}: {v}")

    # ── Final model with best params ──────────────────────────────────────────
    best = {**study.best_params,
            "scale_pos_weight": scale_pos,
            "tree_method": "hist",
            "eval_metric": "aucpr",
            "n_jobs": -1}

    final_model = XGBClassifier(**best)
    final_model.fit(X[train_m], y[train_m],
                    eval_set=[(X[val_m], y[val_m])],
                    verbose=False)

    # Threshold optimisation on val
    y_val_prob = final_model.predict_proba(X[val_m])[:, 1]
    val_auprc = average_precision_score(y[val_m], y_val_prob)
    val_auroc = roc_auc_score(y[val_m], y_val_prob)
    print(f"  Val AUPRC: {val_auprc:.4f}")
    print(f"  Val AUROC: {val_auroc:.4f}")
    best_t, best_f1 = 0.5, 0.0
    for t in np.arange(0.05, 0.95, 0.01):
        f1 = f1_score(y[val_m], (y_val_prob >= t).astype(int), zero_division=0)
        if f1 > best_f1:
            best_f1, best_t = f1, t
    print(f"  Optimal threshold (val F1): {best_t:.2f} (F1={best_f1:.4f})")

    y_prob = final_model.predict_proba(X[test_m])[:, 1]
    y_pred = (y_prob >= best_t).astype(int)

    metrics = _evaluate(y[test_m], y_prob, y_pred, "XGBoost (Bayesian tuned)")
    base_val_col = prep.col_cfg.get("base_value")
    amounts = prep.df[base_val_col].values[test_m] if base_val_col else None
    metrics["threshold_table"] = print_threshold_table(
        y[test_m], y_prob, amounts=amounts,
        model_name="XGBoost (Bayesian tuned)",
        optimal_threshold=best_t,
    )
    metrics["_y_true"] = y[test_m]
    metrics["_y_prob"] = y_prob
    metrics["_xgb_model"] = final_model
    metrics["_feature_names"] = feat_names

    return metrics


def run_tabular_baselines(prep: PreparedData, tune: bool = False,
                          n_trials: int = 50,
                          feature_cols: list[str] | None = None) -> list[dict]:
    """Run tabular baselines. Pass tune=True to use Bayesian optimisation for XGBoost."""
    if tune:
        return [run_xgboost_bayes(prep, n_trials=n_trials, feature_cols=feature_cols)]
    return [run_xgboost(prep, feature_cols=feature_cols)]
