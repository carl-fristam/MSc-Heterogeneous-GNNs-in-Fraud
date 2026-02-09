import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report,
    roc_auc_score,
    average_precision_score,
    precision_recall_curve,
    f1_score,
    confusion_matrix,
    ConfusionMatrixDisplay
)


def evaluate(model, X, y, name, threshold=0.5):
    """
    Evaluate model with custom threshold.

    :param model: Trained model
    :param X: Test features
    :param y: Test labels
    :param name: Model name for printing
    :param threshold: Classification threshold (default 0.5)
    """
    y_prob = model.predict_proba(X)[:, 1]
    y_pred = (y_prob >= threshold).astype(int)

    # Confusion matrix
    cm = confusion_matrix(y, y_pred)
    tn, fp, fn, tp = cm.ravel()

    print(f"\n{'='*60}")
    print(f"{name} (threshold={threshold})")
    print('='*60)

    # Confusion matrix display
    print("\nConfusion Matrix:")
    print(f"                 Predicted")
    print(f"                 Neg      Pos")
    print(f"Actual Neg    {tn:>8,}  {fp:>8,}")
    print(f"Actual Pos    {fn:>8,}  {tp:>8,}")
    print(f"\nTP: {tp:,} | FP: {fp:,} | TN: {tn:,} | FN: {fn:,}")

    print(f"\n{classification_report(y, y_pred, digits=4)}")
    print(f"ROC-AUC: {roc_auc_score(y, y_prob):.4f}")
    print(f"PR-AUC:  {average_precision_score(y, y_prob):.4f}")

    return {
        "name": name,
        "threshold": threshold,
        "roc_auc": roc_auc_score(y, y_prob),
        "pr_auc": average_precision_score(y, y_prob),
        "f1": f1_score(y, y_pred),
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "cm": cm,
        "y_pred": y_pred,
        "y_prob": y_prob,
    }


def plot_confusion_matrix(cm, name, save_path=None):
    """
    Plot confusion matrix as a heatmap.

    :param cm: Confusion matrix from sklearn
    :param name: Model name for title
    :param save_path: Optional path to save the figure
    """
    plt.figure(figsize=(8, 6))

    # Use logarithmic scale for annotations due to class imbalance
    sns.heatmap(
        cm,
        annot=True,
        fmt=',d',
        cmap='Blues',
        xticklabels=['Normal', 'Laundering'],
        yticklabels=['Normal', 'Laundering'],
        cbar_kws={'label': 'Count'}
    )

    plt.title(f'Confusion Matrix: {name}')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved confusion matrix to {save_path}")

    plt.show()


def plot_feature_importance(importance_df, name, top_n=15, save_path=None):
    """
    Plot feature importance as a horizontal bar chart.

    :param importance_df: DataFrame with 'feature' and 'importance' columns
    :param name: Model name for title
    :param top_n: Number of top features to show
    :param save_path: Optional path to save the figure
    """
    plt.figure(figsize=(10, 8))

    top_features = importance_df.head(top_n).sort_values('importance', ascending=True)

    plt.barh(top_features['feature'], top_features['importance'], color='steelblue')
    plt.xlabel('Importance')
    plt.title(f'Top {top_n} Feature Importances: {name}')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved feature importance plot to {save_path}")

    plt.show()


def print_feature_importance(model, feature_names, top_n=15):
    """
    Print top feature importances for a model.

    :param model: Trained model
    :param feature_names: List of feature names
    :param top_n: Number of top features to show
    """
    print(f"\n{'='*60}")
    print(f"Feature Importance (top {top_n})")
    print('='*60)

    # Get importances based on model type
    if hasattr(model, 'feature_importances_'):
        # Tree-based models (RF, XGBoost)
        importances = model.feature_importances_
    elif hasattr(model, 'coef_'):
        # Linear models (LR)
        importances = np.abs(model.coef_[0])
    else:
        print("Model does not support feature importance")
        return None

    # Create dataframe and sort
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': importances
    }).sort_values('importance', ascending=False)

    # Print top features
    print(f"\n{'Feature':<40} {'Importance':>15}")
    print("-" * 55)
    for _, row in importance_df.head(top_n).iterrows():
        print(f"{row['feature']:<40} {row['importance']:>15.4f}")

    return importance_df


def find_optimal_threshold(model, X, y, metric="f1"):
    """
    Find optimal classification threshold.

    :param model: Trained model
    :param X: Validation features
    :param y: Validation labels
    :param metric: 'f1' or target recall value (e.g., 0.8)
    """
    y_prob = model.predict_proba(X)[:, 1]
    precisions, recalls, thresholds = precision_recall_curve(y, y_prob)

    if metric == "f1":
        # Maximize F1
        f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-8)
        optimal_idx = np.argmax(f1_scores)
        optimal_threshold = thresholds[optimal_idx] if optimal_idx < len(thresholds) else 0.5
        print(f"Optimal threshold (max F1): {optimal_threshold:.4f}")
        print(f"  -> Precision: {precisions[optimal_idx]:.4f}, Recall: {recalls[optimal_idx]:.4f}")

    elif isinstance(metric, float):
        # Find threshold for target recall
        target_recall = metric
        idx = np.argmin(np.abs(recalls - target_recall))
        optimal_threshold = thresholds[idx] if idx < len(thresholds) else 0.5
        print(f"Threshold for {target_recall:.0%} recall: {optimal_threshold:.4f}")
        print(f"  -> Precision: {precisions[idx]:.4f}, Recall: {recalls[idx]:.4f}")

    else:
        optimal_threshold = 0.5

    return optimal_threshold
