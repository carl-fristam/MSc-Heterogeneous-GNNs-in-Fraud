# CLAUDE.md

This file provides guidance to Claude Code when working with code in this repository.

## Project

MSc thesis: **Heterogeneous Graph Neural Networks for Transaction-Level Fraud Detection** on Danske Bank retail payment data.

Core research question: _Does preserving heterogeneous structure in a transaction graph improve fraud detection over simpler representations?_

The experimental design compares two levels:

| Mode | What runs | Models | Isolates |
|------|-----------|--------|----------|
| tab  | Tabular XGBoost | — | Production-style baseline: no graph |
| het  | Hetero GNN | HGT, HMPNN, HeteroGAT | Does heterogeneous graph structure improve over tabular? |

Homogeneous GNNs and self-supervised pretraining (LaundroGraph, HGMAE) are covered in the literature review as related work but are **not part of the experimental evaluation**.

All GNN experiments use **edge classification** (transactions are edges) on the bank payment graph with `internal_account` and `external_account` node types.

## Running things

`run.py` is the single entry point.

```
python run.py --mode tab                          # tabular baseline
python run.py --mode het --model hgt              # hetero GNN
python run.py --mode het --model hmpnn            # hetero GNN (HMPNN)
python run.py --mode het --model hetero_gat       # hetero GNN (HeteroGAT)
```

## Data

- **No feature engineering or splitting is done in this repo.** Data arrives as pre-split parquet files (train/val/test) from an external pipeline.
- Split files: `datasets/splits/train.parquet`, `val.parquet`, `test.parquet`
- Label column: `CONFIRMEDRISK` (boolean)
- Temporal split: train ~2024-11 to 2025-08, val ~2025-09 to 2025-11, test ~2025-12 to 2026-03
- All features (OHE, target encodings, velocity features, cyclical time) are pre-computed in the external pipeline.

## Graph structure

One active variant: **V1** (hardcoded in `run.py` via `load_variant("v1")`).

**Node types:**
- `internal_account` — Danske Bank accounts (senders). Features: `accountbranchid_te`, `accagentcountry_te` (2 dims, target-encoded).
- `external_account` — counterparty accounts (receivers only). Features: `counteragentid_te`, `counterpartyid_te`, `COUNTERIDFORMAT_IBAN` (3 dims).

**Edge types (V1):**
- `onus_transfer` — internal -> internal (TRANSACTIONONUS == True)
- `external_transfer` — internal -> external (TRANSACTIONONUS == False)

**Edge features** (91 dims): log amount, OHE groups (channel, clearing, submethod, currency, destination), international flag, cyclical time encoding, temporal context, novelty flags, customer velocity features (1D/7D/30D windows), transaction ratios, safe pair flag.

Node features are account-level properties (target encodings). Edge features are per-transaction and include customer velocity features because they vary per transaction.

## Key source files

- `configs/master.yaml` — single source of truth for columns, features, graph topology
- `src/data/prepare.py` — loads pre-split data, produces `PreparedData` (df + masks + labels)
- `src/graph_builder/` — graph construction pipeline:
  - `node_builder.py` — assigns account IDs to node types and integer indices
  - `node_features.py` — selects pre-computed node features from training rows
  - `edge_features.py` — selects pre-computed edge feature columns
  - `edge_builder.py` — builds edge index tensors and attaches features/labels/masks
  - `assembler.py` — orchestrates the above, assembles PyG `HeteroData`, caches to disk
- `src/heterogeneous/` — het model implementations (`hgt/`, `hmpnn/`, `hetero_gat/`)
- `src/training/trainer.py` — unified training loop for all GNN models
- `src/utils/results.py` — saves metrics.json and report.md per run

**Removed / inactive:** `src/self_supervised/`, `src/heterogeneous/hgmae/`, `src/homogeneous/` are not part of the experimental scope. `src/graph_builder/loader.py` is no longer used (data loading moved to `prepare.py`).

## Conventions

- **Single config**: `configs/master.yaml` is the source of truth. Use `load_variant("v1")` to load it — never hardcode variant details.
- **Entry point for data**: `src/data/prepare.py` (`PreparedData`) loads and prepares data once. All levels consume the same object.
- **No feature engineering**: all encoding, scaling, and feature computation is done externally. This repo only selects columns.
- **Results live next to code**: each model directory has a `results/` subfolder with timestamped runs (`metrics.json` + `report.md`).
- **Graph cache**: built graphs are pickled to `data/processed/bank/`. Clear cache when data or features change.
- **Primary metric**: PR-AUC. Accuracy and AUROC are misleading under ~0.3% fraud rate. Secondary: AUROC, operational metrics.
- **No node task code**: everything is edge classification. Node task paths in the trainer are dead code.
