"""
Operational threshold analysis table.

Prints a table showing the precision/recall tradeoff at 5 fixed thresholds.
Used by both tabular baselines and GNN trainer after test evaluation.

Columns:
  Threshold    — score cutoff above which a transaction is flagged as fraud
  Recall       — fraction of actual fraud caught
  Precision    — of all flagged transactions, fraction that are actually fraud
  Flag rate    — fraction of ALL transactions flagged (operational load)
  Fraud missed — fraction of actual fraud value NOT caught (needs amounts)
                 shown as % of cases if amounts not provided
"""

import numpy as np
from sklearn.metrics import recall_score, precision_score

THRESHOLDS = [0.50, 0.65, 0.75, 0.90, 0.99]


def print_threshold_table(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    amounts: np.ndarray = None,
    model_name: str = "",
):
    """
    Print an operational threshold table for a set of test predictions.

    Args:
        y_true:     ground-truth binary labels (numpy array)
        y_prob:     predicted probabilities (numpy array)
        amounts:    transaction values (BASEVALUE) aligned to y_true.
                    If provided, "Fraud lost" shows % of total fraud VALUE missed.
                    If None, shows % of fraud CASES missed instead.
        model_name: optional label shown in the header
    """
    header = f"Threshold analysis — {model_name}" if model_name else "Threshold analysis"

    has_amounts = amounts is not None
    fraud_col_label = "Fraud lost (value)" if has_amounts else "Fraud missed"
    total_fraud_value = amounts[y_true == 1].sum() if has_amounts else None

    width = 80 if has_amounts else 70
    print(f"\n{header}")
    print(f"{'─' * width}")
    print(
        f"  {'Threshold':>9}  {'Recall':>7}  {'Precision':>9}  "
        f"{'Flag rate':>9}  {fraud_col_label:>18}"
    )
    print(f"{'─' * width}")

    n_total = len(y_true)

    for t in THRESHOLDS:
        y_pred = (y_prob >= t).astype(int)

        recall    = recall_score(y_true, y_pred, zero_division=0)
        precision = precision_score(y_true, y_pred, zero_division=0)
        flag_rate = y_pred.sum() / n_total

        if has_amounts:
            missed_mask = (y_true == 1) & (y_pred == 0)
            fraud_lost  = amounts[missed_mask].sum()
            fraud_lost_str = f"€{fraud_lost:>16,.0f}"
        else:
            fraud_lost_str = f"{1.0 - recall:>17.2%}"

        print(
            f"  {t:>9.2f}  "
            f"{recall:>7.3f}  "
            f"{precision:>9.3f}  "
            f"{flag_rate:>8.2%}  "
            f"{fraud_lost_str}"
        )

    print(f"{'─' * width}")
