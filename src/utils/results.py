"""
results.py

Saves experiment output (metrics + a markdown report) to a timestamped
folder next to the model that produced it.
"""

import json
from datetime import datetime
from pathlib import Path

from src.utils.config import PROJECT_ROOT


def save_results(metrics: dict, level: int, model: str = None, **kwargs):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir   = _results_dir(level, model=model) / f"{_run_name(model)}_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)

    # make sure numpy values are serialisable
    serializable = {
        k: v.tolist() if hasattr(v, "tolist") else v
        for k, v in metrics.items()
    }

    meta = {"level": level, "timestamp": timestamp, "model": model, **kwargs}
    with open(run_dir / "metrics.json", "w") as f:
        json.dump({"meta": meta, "metrics": serializable}, f, indent=2)

    with open(run_dir / "report.md", "w") as f:
        f.write(_build_report(model, level, timestamp, serializable, kwargs))

    print(f"\nResults saved to: {run_dir.relative_to(PROJECT_ROOT)}/")


# ── Helpers ───────────────────────────────────────────────────────────────────

def _results_dir(level: int, model: str = None) -> Path:
    if level == 0:
        return PROJECT_ROOT / "src" / "baselines" / "tabular" / "results"
    elif level == 1:
        return PROJECT_ROOT / "src" / "homogeneous" / (model or "unknown") / "results"
    elif level == 2:
        return PROJECT_ROOT / "src" / "heterogeneous" / (model or "unknown") / "results"
    return PROJECT_ROOT / "results"


def _run_name(model: str = None) -> str:
    return model if model else "run"


def _build_report(model, level, timestamp, metrics, kwargs) -> str:
    cm     = metrics.get("confusion_matrix")
    cm_str = ""
    if cm:
        cm_str = (
            f"\n## Confusion Matrix\n\n"
            f"|  | Pred 0 | Pred 1 |\n|---|---|---|\n"
            f"| **Actual 0** | {cm[0][0]} | {cm[0][1]} |\n"
            f"| **Actual 1** | {cm[1][0]} | {cm[1][1]} |\n"
        )

    header = f"# {model or 'run'}\n\n**Date:** {timestamp}  \n**Level:** {level}  \n"
    for k, v in kwargs.items():
        if v is not None:
            header += f"**{k.title()}:** {v}  \n"

    body = (
        f"\n## Metrics\n\n"
        f"| Metric | Value |\n|---|---|\n"
        f"| AUROC     | {metrics.get('auroc',     'N/A'):.4f} |\n"
        f"| AUPRC     | {metrics.get('auprc',     'N/A'):.4f} |\n"
        f"| F1        | {metrics.get('f1',        'N/A'):.4f} |\n"
        f"| Precision | {metrics.get('precision', 'N/A'):.4f} |\n"
        f"| Recall    | {metrics.get('recall',    'N/A'):.4f} |\n"
        f"{cm_str}"
    )

    return header + body
