"""
Run XGBoost + all three GNN models with default params.
50% sample, all fraud kept.
Each GNN is run at 1 and 2 hidden layers for ablation.

Usage:
    nohup python -u scripts/run_all_defaults.py > results/defaults/run.log 2>&1 &

Then:
    tail -f results/defaults/run.log
"""

import subprocess
import sys
from pathlib import Path

SAMPLE = 0.5
RESULTS_DIR = "results/defaults"

Path(RESULTS_DIR).mkdir(parents=True, exist_ok=True)

GNN_MODELS = ["hgt", "hmpnn", "hetero_gat"]
GNN_LABELS = {"hgt": "HGT", "hmpnn": "HMPNN", "hetero_gat": "HeteroGAT"}
LAYERS = [1, 2]

RUNS = [
    {"label": "XGBoost", "cmd": [
        sys.executable, "-u", "run.py",
        "--mode", "tab",
        "--results-dir", RESULTS_DIR,
        "--sample", str(SAMPLE),
    ]},
]

for model in GNN_MODELS:
    for n_layers in LAYERS:
        RUNS.append({
            "label": f"{GNN_LABELS[model]} ({n_layers}L)",
            "cmd": [
                sys.executable, "-u", "run.py",
                "--mode", "het", "--model", model,
                "--sample", str(SAMPLE),
                "--hidden-dim", "64", "--num-layers", str(n_layers),
                "--lr", "1e-3", "--patience", "15", "--epochs", "200",
                "--results-dir", RESULTS_DIR,
            ],
        })

total = len(RUNS)
for i, run in enumerate(RUNS, 1):
    print(f"\n{'=' * 60}")
    print(f"[{i}/{total}] {run['label']}")
    print(f"{'=' * 60}", flush=True)

    result = subprocess.run(run["cmd"])

    if result.returncode != 0:
        print(f"FAILED: {run['label']} (exit code {result.returncode})")

# Plots
print(f"\n{'=' * 60}")
print("Generating plots...")
print(f"{'=' * 60}", flush=True)
subprocess.run([sys.executable, "scripts/plot_pr_comparison.py", RESULTS_DIR])
subprocess.run([sys.executable, "scripts/plot_layer_ablation.py", RESULTS_DIR])
subprocess.run([sys.executable, "scripts/plot_threshold_analysis.py", RESULTS_DIR])

print(f"\n{'=' * 60}")
print(f"ALL DONE — results in {RESULTS_DIR}/")
print(f"{'=' * 60}")
