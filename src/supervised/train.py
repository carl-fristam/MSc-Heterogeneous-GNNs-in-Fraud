from src.supervised.data_prep import load_data_preprocess, split, scale_features, resample
from src.supervised.models import get_models
from src.supervised.eval import (evaluate, find_optimal_threshold, print_feature_importance,
                                  plot_confusion_matrix, plot_feature_importance)
from xgboost import XGBClassifier



# CONFIG

MODELS_TO_RUN = ["XGBoost"]  # ["LogisticRegression", "RandomForest", "XGBoost"] or subset
RESAMPLE_STRATEGY = None     # None, "smote", "undersample", "smote_tomek"
RESAMPLE_RATIO = 0.1         # Target minority:majority ratio (if resampling)
POS_WEIGHT = None            # None=auto, or manual value like 100
TUNE_THRESHOLD = True        # Whether to find optimal threshold
TARGET_METRIC = 0.9         # "f1" for max F1, or float like 0.8 for target recall
SHOW_PLOTS = True            # Whether to show confusion matrix and feature importance plots


def main():

    # Load and split data

    print("Loading data")
    df = load_data_preprocess()
    X_train, X_test, y_train, y_test = split(df, target="Is_laundering")

    print(f"Train size: {len(X_train):,} | Test size: {len(X_test):,}")
    print(f"Train class distribution: {y_train.value_counts().to_dict()}")

    # Resample training data

    if RESAMPLE_STRATEGY:
        X_train_resampled, y_train_resampled = resample(
            X_train, y_train,
            strategy=RESAMPLE_STRATEGY,
            ratio=RESAMPLE_RATIO
        )
    else:
        X_train_resampled, y_train_resampled = X_train, y_train


    # Scale features (for LR)

    X_train_scaled, X_test_scaled, scaler = scale_features(X_train_resampled, X_test)

    # Get models with weight config

    if POS_WEIGHT is None and RESAMPLE_STRATEGY is None:
        # Auto-compute weight from original data
        auto_weight = (y_train == 0).sum() / (y_train == 1).sum()
        print(f"Auto-computed pos_weight: {auto_weight:.2f}")
    else:
        auto_weight = POS_WEIGHT if POS_WEIGHT else 1.0

    models = get_models(
        pos_weight=POS_WEIGHT,
        auto_balance=(POS_WEIGHT is None and RESAMPLE_STRATEGY is None)
    )

    # Update XGBoost weight if auto-balanced without resampling
    if POS_WEIGHT is None and RESAMPLE_STRATEGY is None:
        models["XGBoost"] = XGBClassifier(
            scale_pos_weight=auto_weight,
            n_estimators=100, learning_rate=0.1,
            max_depth=6, n_jobs=-1, random_state=42
        )

    # Train and evaluate

    results = []

    for name, model in models.items():
        if MODELS_TO_RUN and name not in MODELS_TO_RUN:
            continue
        print(f"\n# Training {name}")

        # Use scaled data for LR
        if "Logistic" in name:
            X_tr, X_te = X_train_scaled, X_test_scaled
        else:
            X_tr, X_te = X_train_resampled, X_test

        model.fit(X_tr, y_train_resampled)

        # Feature importance
        feature_names = X_train.columns.tolist()
        importance_df = print_feature_importance(model, feature_names, top_n=15)

        if SHOW_PLOTS and importance_df is not None:
            plot_feature_importance(importance_df, name, top_n=15)

        # Threshold tuning

        if TUNE_THRESHOLD:
            optimal_thresh = find_optimal_threshold(
                model, X_te, y_test,
                metric=TARGET_METRIC
            )
        else:
            optimal_thresh = 0.5

        # Evaluate with default and optimal threshold
        res_default = evaluate(model, X_te, y_test, f"{name} (default)", threshold=0.5)
        results.append(res_default)

        if SHOW_PLOTS:
            plot_confusion_matrix(res_default['cm'], f"{name} (default)")

        if TUNE_THRESHOLD and optimal_thresh != 0.5:
            res_tuned = evaluate(model, X_te, y_test, f"{name} (tuned)", threshold=optimal_thresh)
            results.append(res_tuned)

            if SHOW_PLOTS:
                plot_confusion_matrix(res_tuned['cm'], f"{name} (tuned)")


    # Summary

    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"{'Model':<35} {'ROC-AUC':>10} {'PR-AUC':>10} {'F1':>10}")
    print("-"*60)
    for r in results:
        print(f"{r['name']:<35} {r['roc_auc']:>10.4f} {r['pr_auc']:>10.4f} {r['f1']:>10.4f}")

    return results


if __name__ == "__main__":
    main()
