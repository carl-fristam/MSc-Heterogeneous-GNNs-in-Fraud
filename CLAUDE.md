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

All GNN experiments use **edge classification** on the bank payment graph with `internal_account` and `external_account` node types. The ladder and current models are reflected in `run.py` and `src/`.

## Running things

`run.py` is the single entry point. Use `--sample 0.05` for dev/debug runs.

```
python run.py --level 0                          # tabular baseline
python run.py --level 1 --model sage             # homo GNN
python run.py --level 2 --model hgt              # hetero GNN
python run.py --level 2 --model hgt --sample 0.05
```

For local testing without the real dataset: `python mock_run.py` and `python sanity_check.py`.

## Data

- Dataset: `datasets/TRANSACTIONS_almost_clean.parquet`
- Label column: `CONFIRMEDRISK` (boolean) — fraud labels only cover a narrow confirmed-fraud window, not the full dataset history. Data is truncated to stay within that window.
- Temporal split reflects deployment reality (train → val → test in time order), not random split.

## Graph structure

One active variant: **V1** (hardcoded in `run.py` via `load_variant("v1")`).

**Node types:**
- `internal_account` — Danske Bank accounts (senders). Features: out-degree, amount stats, counterparty diversity, channel diversity, time behavior, branch OHE, sender bank OHE.
- `external_account` — counterparty accounts (receivers only, never senders). Features: in-degree, received amount stats, sender diversity, sender bank diversity, counterparty bank OHE, account format OHE.

**Edge types (V1):**
- `onus_transfer` — internal → internal (TRANSACTIONONUS == True)
- `external_transfer` — internal → external (TRANSACTIONONUS == False)

**Edge features** (34 dims): log amount, channel, payment submethod, clearing type, currency, destination country, international flag, cyclical time encoding.

Note: branch, sender bank, counterparty bank, and account format are **node features**, not edge features. They describe the account entity, not the transaction.

## Key source files

- `configs/master.yaml` — single source of truth for columns, features, split dates, graph topology
- `src/data/prepare.py` — loads data once, produces `PreparedData` (df + masks + labels)
- `src/graph_builder/` — graph construction pipeline:
  - `loader.py` — loads parquet, creates `_sender`/`_receiver` aliases
  - `node_builder.py` — assigns account IDs to node types and integer indices
  - `node_features.py` — aggregates node features from training rows only
  - `edge_features.py` — builds per-transaction edge feature vectors
  - `edge_builder.py` — builds edge index tensors and attaches features/labels/masks
  - `assembler.py` — orchestrates the above, assembles PyG `HeteroData`, caches to disk
- `src/training/trainer.py` — unified training loop for all GNN models
- `src/utils/results.py` — saves metrics.json and report.md per run

## Conventions

- **Single config**: `configs/master.yaml` is the source of truth. Use `load_variant("v1")` to load it — never hardcode variant details.
- **Entry point for data**: `src/data/prepare.py` (`PreparedData`) loads and prepares data once. All levels consume the same object.
- **Results live next to code**: each model directory has a `results/` subfolder with timestamped runs (`metrics.json` + `report.md`).
- **Graph cache**: built graphs are pickled to `data/processed/bank/`. Clear cache when data or features change.
- **Primary metric**: PR-AUC. Accuracy and AUROC are misleading under ~0.3% fraud rate. Secondary: AUROC, operational metrics.
- **No node task code**: everything is edge classification. Node task paths in the trainer are dead code.
