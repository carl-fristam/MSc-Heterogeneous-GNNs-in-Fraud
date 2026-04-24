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
from sklearn.metrics import recall_score, precision_score, f1_score, accuracy_score

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
    optimal_threshold: float = None,
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

    # Second table: classification metrics per threshold
    perf_header = f"Classification metrics — {model_name}" if model_name else "Classification metrics"
    perf_width = 65
    print(f"\n{perf_header}")
    print(f"{'─' * perf_width}")
    print(f"  {'Threshold':>9}  {'Accuracy':>9}  {'F1':>7}  {'Precision':>9}  {'Recall':>7}")
    print(f"{'─' * perf_width}")

    for t in THRESHOLDS:
        y_pred = (y_prob >= t).astype(int)
        acc  = accuracy_score(y_true, y_pred)
        f1   = f1_score(y_true, y_pred, zero_division=0)
        prec = precision_score(y_true, y_pred, zero_division=0)
        rec  = recall_score(y_true, y_pred, zero_division=0)
        print(
            f"  {t:>9.2f}  "
            f"{acc:>9.4f}  "
            f"{f1:>7.4f}  "
            f"{prec:>9.4f}  "
            f"{rec:>7.3f}"
        )

    print(f"{'─' * perf_width}")

    # Third table: optimal threshold summary
    if optimal_threshold is not None:
        y_pred_opt = (y_prob >= optimal_threshold).astype(int)
        acc_opt  = accuracy_score(y_true, y_pred_opt)
        f1_opt   = f1_score(y_true, y_pred_opt, zero_division=0)
        prec_opt = precision_score(y_true, y_pred_opt, zero_division=0)
        rec_opt  = recall_score(y_true, y_pred_opt, zero_division=0)

        has_amounts = amounts is not None
        opt_width = 90 if has_amounts else 75
        opt_header = f"Optimal threshold — {model_name}" if model_name else "Optimal threshold"
        print(f"\n{opt_header}")
        print(f"{'─' * opt_width}")

        if has_amounts:
            missed_mask = (y_true == 1) & (y_pred_opt == 0)
            fraud_lost = amounts[missed_mask].sum()
            print(f"  {'Threshold':>9}  {'Accuracy':>9}  {'F1':>7}  {'Precision':>9}  {'Recall':>7}  {'Fraud lost':>18}")
            print(f"{'─' * opt_width}")
            print(
                f"  {optimal_threshold:>9.2f}  "
                f"{acc_opt:>9.4f}  "
                f"{f1_opt:>7.4f}  "
                f"{prec_opt:>9.4f}  "
                f"{rec_opt:>7.3f}  "
                f"€{fraud_lost:>16,.0f}"
            )
        else:
            print(f"  {'Threshold':>9}  {'Accuracy':>9}  {'F1':>7}  {'Precision':>9}  {'Recall':>7}")
            print(f"{'─' * opt_width}")
            print(
                f"  {optimal_threshold:>9.2f}  "
                f"{acc_opt:>9.4f}  "
                f"{f1_opt:>7.4f}  "
                f"{prec_opt:>9.4f}  "
                f"{rec_opt:>7.3f}"
            )

        print(f"{'─' * opt_width}")

    return rows
