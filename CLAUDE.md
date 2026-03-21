# CLAUDE.md

This file provides guidance to Claude Code when working with code in this repository.

## Project Overview

MSc thesis project: **Heterogeneous Graph Neural Networks for Transaction-Level Fraud Detection** on Danske Bank retail payment data (~3M transactions, ~0.3% fraud rate).

Core research question: _Does preserving heterogeneous structure in a transaction graph improve fraud detection over simpler representations?_

## Graph Signal

Fraud network analysis confirms exploitable graph structure:
- **12.4x enrichment**: accounts in the 1-hop fraud neighborhood are 12x more likely to be fraud senders vs baseline
- **31.6% repeat offenders**: nearly a third of fraud senders commit multiple fraud transactions
- **32k accounts** send to a fraud receiver; 1,049 of those are fraud senders themselves
- This justifies the GNN approach — neighbor context carries signal

## Experimental Ladder

| Level  | What                                         | Code                                         |
| ------ | -------------------------------------------- | -------------------------------------------- |
| **L0** | Tabular baselines (LR, XGBoost)              | `src/baselines/tabular.py`                   |
| **L1** | Graph features → XGBoost                     | `src/baselines/graph_features.py`            |
| **L2** | Homogeneous (GCN, SAGE, TransE, DistMult)    | `src/homogeneous/`                           |
| **L3** | Heterogeneous GNN (HGT, HMPNN)               | `src/heterogeneous/hgt/`, `src/heterogeneous/hmpnn/` |

L0→L1: does graph structure help at all?
L1→L2: do GNNs/KGE learn better representations than hand-crafted graph features?
L2→L3: does heterogeneous typing improve over homogeneous?
V1→V2: which heterogeneous design choices matter?

Both **node classification** (transactions as nodes) and **edge classification** (transactions as edges) are supported.

## Running Experiments

```bash
# L0: Tabular baselines
python run.py --level 0

# L1: Graph features → XGBoost
python run.py --level 1

# L2: Homogeneous GNN
python run.py --level 2 --task edge --conv sage --epochs 200
python run.py --level 2 --task edge --conv gcn --epochs 200
python run.py --level 2 --task node --conv sage --epochs 200

# L2: KGE baselines (edge classification only)
python run.py --level 2 --task edge --conv transe --epochs 500 --patience 50
python run.py --level 2 --task edge --conv distmult --epochs 500 --patience 50

# L3: Heterogeneous GNN
python run.py --level 3 --task edge --model hgt --variant v1 --epochs 300 --patience 40
python run.py --level 3 --task edge --model hgt --variant v2 --epochs 300 --patience 40
python run.py --level 3 --task edge --model hmpnn --variant v1 --epochs 300 --patience 40
python run.py --level 3 --task node --model hgt --variant txn_v1 --epochs 300 --patience 40

# Dev mode (5% sample)
python run.py --level 2 --task edge --conv sage --sample 0.05 --epochs 10
```

Primary metric: **PR-AUC** (precision-recall area under curve). Secondary: AUROC, operational metrics (false positives per true positive).

## Code Structure & Conventions

### Directory layout — code and results live together

```
src/
  baselines/
    tabular.py                # L0: LR, XGBoost
    tabular/results/          # L0 results land here
    graph_features.py         # L1: graph features → XGBoost
    graph_features/results/   # L1 results land here
  homogeneous/                # L2
    builder.py                # homogeneous graph construction (node or edge mode)
    models.py                 # GCN, GraphSAGE (supports edge_attr in edge mode)
    kge_models.py             # TransE, DistMult
    gcn/results/              # GCN results
    sage/results/             # SAGE results
    transe/results/           # TransE results
    distmult/results/         # DistMult results
  heterogeneous/              # L3
    hgt/
      model.py                # HGTConv wrapper
      train.py                # thin wrapper around unified Trainer
      results/                # HGT results
    hmpnn/
      model.py                # NNConv + HeteroConv
      results/                # HMPNN results
  graph_pipeline_bank/        # hetero edge classification graph construction
    loader.py                 # load parquet, parse dates, truncate, sample
    node_builder.py           # build node ID mappings
    features_node.py          # @register_*_feature decorators for node features
    features_edge.py          # @register_edge_feature for transaction features
    edge_builder.py           # filter → edge_index + edge_attr + labels + masks
    normalize.py              # zscore, one_hot, vocab utilities
    builder.py                # orchestrator → HeteroData
  graph_pipeline_bank_txn/    # hetero node classification graph construction
    builder.py                # transactions as nodes variant
  training/
    trainer.py                # unified Trainer: all model/task/graph combos
  data/
    prepare.py                # PreparedData — single entry point for all levels
  utils/
    config.py                 # load_config(), load_variant(), PROJECT_ROOT
    device.py                 # get_device() — MPS/CUDA/CPU
    class_weights.py          # inverse-frequency weighting
    split.py                  # temporal_split(), random_stratified_split()
    compat.py                 # PyG API compatibility patches
scripts/
  viz_topology.py             # generate topology figures for thesis
  fraud_network_analysis.py   # analyze graph signal in fraud data
configs/
  master.yaml                 # single source of truth for all variants
```

### Key conventions

- **Results live next to code**: each model type has a `results/` subfolder. Results are saved as timestamped folders containing `metrics.json` + `report.md`.
- **New models/pipelines follow the same pattern**: code in the relevant `src/` subfolder, results in a `results/` subfolder next to it.
- **Single config**: `configs/master.yaml` is the command center. All data paths, column mappings, feature lists, split dates, and graph variants are defined there. `load_variant()` merges shared settings with variant-specific topology.
- **PreparedData**: `src/data/prepare.py` loads and prepares data once. All levels consume the same `PreparedData` object.
- **Graph caching**: built graphs are pickled to `data/processed/bank/` with names encoding variant and sample ratio. Clear cache when data or features change.
- **Topology figures**: `python scripts/viz_topology.py --all` generates PDFs in `outputs/topology/`.

## Config — `configs/master.yaml`

The master config defines:
- `data_path`: path to parquet file
- `truncate_after`: drop rows after this date (fraud labels only cover early period)
- `split`: temporal split cutoff dates
- `columns`: mapping of semantic roles to column names
- `edge_features`: list of transaction-level features
- `node_features`: per-node-type feature definitions
- `variants`: graph topology definitions (v1, v2, txn_v1)

### Current variants

**V1** — onus vs external (2 edge types):
```
InternalAccount ──[onus_transfer]────► InternalAccount
InternalAccount ──[external_transfer]► ExternalAccount
```

**V2** — payment rail typed edges (6 edge types):
```
InternalAccount ──[onus_transfer]────► InternalAccount
InternalAccount ──[ext_realtime]─────► ExternalAccount
InternalAccount ──[ext_giro]─────────► ExternalAccount
InternalAccount ──[ext_future]───────► ExternalAccount
InternalAccount ──[ext_salary]───────► ExternalAccount
InternalAccount ──[ext_other]────────► ExternalAccount
```

**TXN_V1** — transactions as nodes (node classification):
```
InternalAccount ──[sends]──────────────► Transaction
Transaction ──[received_by_internal]──► InternalAccount
Transaction ──[received_by_external]──► ExternalAccount
```

## Data

- Dataset: `datasets/TRANSACTIONS_almost_clean.parquet` (~3M rows)
- Fraud labels only cover Sep–Dec 2024; data truncated to 2024-12-31
- Temporal split: train before Nov 20, val Nov 20–Dec 10, test Dec 10–Dec 31
- Label column: `CONFIRMEDRISK` (boolean True/False)

### Current columns in cleaned dataset

`ACCOUNTAGENTID`, `ACCOUNTBRANCHID`, `ACCOUNTENTITYID`, `ACCOUNTID`, `ACCOUNTIDFORMAT`, `CURRENCY`, `BASEVALUE`, `CHANNEL`, `COUNTERAGENTID`, `COUNTERBRANCHID`, `COUNTERENTITYID`, `COUNTERPARTYID`, `COUNTERIDFORMAT`, `CUSTOMERENTITYID`, `CUSTOMERID`, `DESTINATIONCOUNTRY`, `EVENTTIME`, `PAYMENTCLEARING`, `PAYMENTSUBMETHOD`, `TRANSACTIONONUS`, `INTERNATIONALFLAG`, `CONFIRMEDRISK`, `ACCOUNTBRANCH_TBE`

### Edge / transaction features

log1p BASEVALUE (z-scored), channel OHE, submethod OHE, clearing express flag, international flag, branch TBE OHE, sin/cos time encoding (hour + day-of-week)

### Node features (train-only aggregation)

- **InternalAccount**: out-degree, amount stats (mean/std/total sent, log1p), counterparty diversity, channel diversity, time behavior (night/weekend ratios)
- **ExternalAccount**: in-degree, received amount stats, sender diversity, sender bank diversity

## Data & Output Paths (gitignored)

| Path                    | Contents                       |
| ----------------------- | ------------------------------ |
| `datasets/`             | Parquet data files             |
| `data/processed/bank/`  | Cached PyG graph objects (.pkl)|
| `outputs/topology/`     | Topology visualization PDFs    |
| `src/*/results/`        | Experiment results per model   |
