"""
Compare experiment results across models.

Scans results directories for metrics.json files and prints a comparison table.

Usage:
    python compare.py              # latest run per model
    python compare.py --all        # all runs
"""

import argparse
import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent

RESULTS_DIRS = {
    "XGBoost":    PROJECT_ROOT / "src" / "baselines" / "tabular" / "results",
    "HGT":        PROJECT_ROOT / "src" / "heterogeneous" / "hgt" / "results",
    "HMPNN":      PROJECT_ROOT / "src" / "heterogeneous" / "hmpnn" / "results",
    "HeteroGAT":  PROJECT_ROOT / "src" / "heterogeneous" / "hetero_gat" / "results",
}

METRIC_COLS = [
    ("auprc",     "PR-AUC"),
    ("auroc",     "AUROC"),
    ("f1",        "F1"),
    ("precision", "Precision"),
    ("recall",    "Recall"),
]


def load_runs(results_dir: Path) -> list[dict]:
    runs = []
    if not results_dir.exists():
        return runs
    for mf in sorted(results_dir.glob("*/metrics.json")):
        with open(mf) as f:
            data = json.load(f)
        data["_path"] = mf.parent.name
        runs.append(data)
    return runs


def latest_run(runs: list[dict]) -> dict | None:
    if not runs:
        return None
    return runs[-1]


def fmt(val, width=9):
    if val is None or val == "N/A":
        return f"{'—':>{width}}"
    return f"{val:>{width}.4f}"


def print_table(rows: list[tuple[str, dict]]):
    name_w = max(len(name) for name, _ in rows)
    name_w = max(name_w, 5)
    col_w = 10

    header = f"  {'Model':<{name_w}}"
    for _, label in METRIC_COLS:
        header += f"  {label:>{col_w}}"
    sep = "  " + "─" * (name_w + (col_w + 2) * len(METRIC_COLS))

    print(f"\n{sep}")
    print(header)
    print(sep)

    for name, metrics in rows:
        line = f"  {name:<{name_w}}"
        for key, _ in METRIC_COLS:
            line += f"  {fmt(metrics.get(key), col_w)}"
        print(line)

    print(sep)


def print_threshold_tables(rows: list[tuple[str, dict]]):
    for name, metrics in rows:
        tt = metrics.get("threshold_table")
        if not tt:
            continue
        has_amounts = "fraud_lost_value" in tt[0]
        fraud_col = "Fraud lost (value)" if has_amounts else "Fraud missed"
        width = 80 if has_amounts else 70

        print(f"\n  Threshold analysis — {name}")
        print(f"  {'─' * width}")
        print(
            f"    {'Threshold':>9}  {'Recall':>7}  {'Precision':>9}  "
            f"{'Flag rate':>9}  {fraud_col:>18}"
        )
        print(f"  {'─' * width}")
        for row in tt:
            if has_amounts:
                fraud_str = f"€{row['fraud_lost_value']:>16,.0f}"
            else:
                fraud_str = f"{1.0 - row['recall']:>17.2%}"
            print(
                f"    {row['threshold']:>9.2f}  "
                f"{row['recall']:>7.3f}  "
                f"{row['precision']:>9.3f}  "
                f"{row['flag_rate']:>8.2%}  "
                f"{fraud_str}"
            )
        print(f"  {'─' * width}")


def print_confusion_matrices(rows: list[tuple[str, dict]]):
    for name, metrics in rows:
        cm = metrics.get("confusion_matrix")
        if cm is None:
            continue
        print(f"\n  {name} — Confusion Matrix")
        print(f"  {'':>12} Pred 0    Pred 1")
        print(f"  {'Actual 0':>12} {cm[0][0]:>8}  {cm[0][1]:>8}")
        print(f"  {'Actual 1':>12} {cm[1][0]:>8}  {cm[1][1]:>8}")


def main():
    parser = argparse.ArgumentParser(description="Compare experiment results")
    parser.add_argument("--all", action="store_true", help="Show all runs, not just latest")
    args = parser.parse_args()

    rows = []
    for model_name, results_dir in RESULTS_DIRS.items():
        runs = load_runs(results_dir)
        if not runs:
            continue
        if args.all:
            for run in runs:
                ts = run.get("meta", {}).get("timestamp", "")
                label = f"{model_name} ({ts})" if ts else model_name
                rows.append((label, run["metrics"]))
        else:
            run = latest_run(runs)
            if run:
                rows.append((model_name, run["metrics"]))

    if not rows:
        print("\nNo results found. Run experiments first:")
        print("  python run.py --mode tab")
        print("  python run.py --mode het --model hgt")
        print("  python run.py --mode het --model hmpnn")
        print("  python run.py --mode het --model hetero_gat")
        return

    print_table(rows)
    print_threshold_tables(rows)
    print_confusion_matrices(rows)
    print()


if __name__ == "__main__":
    main()
