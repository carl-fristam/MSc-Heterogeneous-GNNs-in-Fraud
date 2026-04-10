# CLAUDE.md

This file provides guidance to Claude Code when working with code in this repository.

## Project

MSc thesis: **Heterogeneous Graph Neural Networks for Transaction-Level Fraud Detection** on Danske Bank retail payment data.

Core research question: _Does preserving heterogeneous structure in a transaction graph improve fraud detection over simpler representations?_

The experimental design is a three-level ladder. Each rung isolates one design decision:

| Level | What runs | Models | Isolates |
|-------|-----------|--------|----------|
| L0 | Tabular XGBoost | — | Floor: no graph at all |
| L1 | Homo GNN | GCN, GraphSAGE, GAT | Does any graph structure help? |
| L2 | Hetero GNN | HGT, HMPNN, HeteroGAT | Does heterogeneous structure specifically help? |

All GNN experiments use **edge classification** on the bank payment graph with `internal_account` and `external_account` node types. Graph variants (v1, v2) vary the edge-type granularity. The ladder and current models are reflected in `run.py` and `src/`.

## Running things

`run.py` is the single entry point. Use `--sample 0.05` for dev/debug runs.

## Data

- Dataset: `datasets/TRANSACTIONS_almost_clean.parquet`
- Label column: `CONFIRMEDRISK` (boolean) — fraud labels only cover a narrow confirmed-fraud window, not the full dataset history. Data is truncated to stay within that window.
- Temporal split reflects deployment reality (train → val → test in time order), not random split.

## Conventions

- **Single config**: `configs/master.yaml` is the source of truth for data paths, column mappings, features, split dates, and graph variants. Use `load_variant()` to access it — never hardcode variant details.
- **Entry point for data**: `src/data/prepare.py` (`PreparedData`) loads and prepares data once. All levels consume the same object.
- **Results live next to code**: each model directory has a `results/` subfolder with timestamped runs (`metrics.json` + `report.md`).
- **Graph cache**: built graphs are pickled to `data/processed/bank/`. Clear cache when data or features change.
- **Primary metric**: PR-AUC. Accuracy and AUROC are misleading under ~0.3% fraud rate. Secondary: AUROC, operational metrics.
