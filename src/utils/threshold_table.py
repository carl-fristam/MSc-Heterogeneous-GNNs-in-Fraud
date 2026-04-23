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


def compute_threshold_rows(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    amounts: np.ndarray = None,
) -> list[dict]:
    n_total = len(y_true)
    rows = []
    for t in THRESHOLDS:
        y_pred = (y_prob >= t).astype(int)
        rec  = recall_score(y_true, y_pred, zero_division=0)
        prec = precision_score(y_true, y_pred, zero_division=0)
        row = {
            "threshold": t,
            "recall": float(rec),
            "precision": float(prec),
            "flag_rate": float(y_pred.sum() / n_total),
        }
        if amounts is not None:
            missed_mask = (y_true == 1) & (y_pred == 0)
            row["fraud_lost_value"] = float(amounts[missed_mask].sum())
        rows.append(row)
    return rows


def print_threshold_table(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    amounts: np.ndarray = None,
    model_name: str = "",
) -> list[dict]:
    """
    Print an operational threshold table and return the row data.
    """
    rows = compute_threshold_rows(y_true, y_prob, amounts)

    header = f"Threshold analysis — {model_name}" if model_name else "Threshold analysis"
    has_amounts = amounts is not None
    fraud_col_label = "Fraud lost (value)" if has_amounts else "Fraud missed"

    width = 80 if has_amounts else 70
    print(f"\n{header}")
    print(f"{'─' * width}")
    print(
        f"  {'Threshold':>9}  {'Recall':>7}  {'Precision':>9}  "
        f"{'Flag rate':>9}  {fraud_col_label:>18}"
    )
    print(f"{'─' * width}")

    for row in rows:
        if has_amounts:
            fraud_lost_str = f"€{row['fraud_lost_value']:>16,.0f}"
        else:
            fraud_lost_str = f"{1.0 - row['recall']:>17.2%}"

        print(
            f"  {row['threshold']:>9.2f}  "
            f"{row['recall']:>7.3f}  "
            f"{row['precision']:>9.3f}  "
            f"{row['flag_rate']:>8.2%}  "
            f"{fraud_lost_str}"
        )

    print(f"{'─' * width}")
    return rows
