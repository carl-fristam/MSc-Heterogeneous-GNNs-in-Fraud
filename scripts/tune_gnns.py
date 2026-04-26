"""
Hyperparameter tuning grid for all three GNN architectures.

Results go to results/tuning/<model>/
Killed runs are logged to results/tuning/killed.log

Usage:
    python scripts/tune_gnns.py
"""

import subprocess
import sys
from datetime import datetime
from pathlib import Path
from itertools import product

SAMPLE = 0.5
RESULTS_DIR = "results/tuning"
KILLED_LOG = Path(RESULTS_DIR) / "killed.log"

MODELS = ["hgt", "hmpnn", "hetero_gat"]
HIDDEN_DIMS = [32, 64, 128]
NUM_LAYERS = [2, 3]
LRS = [1e-3, 5e-4]

grid = list(product(MODELS, HIDDEN_DIMS, NUM_LAYERS, LRS))
total = len(grid)
killed = 0

Path(RESULTS_DIR).mkdir(parents=True, exist_ok=True)
with open(KILLED_LOG, "w") as f:
    f.write(f"Tuning started: {datetime.now()}\n")
    f.write("=" * 50 + "\n")

for i, (model, hd, nl, lr) in enumerate(grid, 1):
    desc = f"{model} | hidden={hd} layers={nl} lr={lr}"
    print(f"\n{'=' * 50}")
    print(f"[{i}/{total}] {desc}")
    print("=" * 50)

    cmd = [
        sys.executable, "-u", "run.py",
        "--mode", "het",
        "--model", model,
        "--sample", str(SAMPLE),
        "--hidden-dim", str(hd),
        "--num-layers", str(nl),
        "--lr", str(lr),
        "--patience", "15",
        "--epochs", "200",
        "--results-dir", RESULTS_DIR,
    ]

    result = subprocess.run(cmd)

    if result.returncode != 0:
        killed += 1
        msg = f"KILLED: {desc} (exit code {result.returncode})"
        print(msg)
        with open(KILLED_LOG, "a") as f:
            f.write(msg + "\n")

print(f"\n{'=' * 50}")
print("TUNING COMPLETE")
print(f"  Total runs:  {total}")
print(f"  Killed:      {killed}")
print(f"  Results in:  {RESULTS_DIR}/")
print(f"  Kill log:    {KILLED_LOG}")
print("=" * 50)
