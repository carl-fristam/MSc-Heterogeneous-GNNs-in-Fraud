"""
Run all three GNN models with default params.
50% sample with proportional fraud (~7.5k fraud transactions).

Usage:
    python scripts/run_defaults_50pct_proportional.py
"""

import subprocess
import sys
from pathlib import Path

SAMPLE = 0.5
RESULTS_DIR = "results/defaults_proportional"

MODELS = ["hgt", "hmpnn", "hetero_gat"]

Path(RESULTS_DIR).mkdir(parents=True, exist_ok=True)

for i, model in enumerate(MODELS, 1):
    print(f"\n{'=' * 50}")
    print(f"[{i}/3] {model} | hidden=64 layers=2 lr=1e-3")
    print("=" * 50)

    cmd = [
        sys.executable, "-u", "run.py",
        "--mode", "het",
        "--model", model,
        "--sample", str(SAMPLE),
        "--proportional-sample",
        "--hidden-dim", "64",
        "--num-layers", "2",
        "--lr", "1e-3",
        "--patience", "15",
        "--epochs", "200",
        "--results-dir", RESULTS_DIR,
    ]

    result = subprocess.run(cmd)

    if result.returncode != 0:
        print(f"FAILED: {model} (exit code {result.returncode})")

print(f"\n{'=' * 50}")
print(f"DONE — results in {RESULTS_DIR}/")
print("=" * 50)
