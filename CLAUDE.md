# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

MSc thesis project: **Heterogeneous Graph Neural Networks for Transaction-Level Fraud Detection** on Danske Bank retail payment data (~1.5M transactions, 0.14% fraud rate).

Core research question: *Does preserving heterogeneous structure in a transaction graph improve fraud detection over simpler representations?*

## Experimental Ladder

| Level | What | Code |
|---|---|---|
| **L0** | Tabular baselines (LR, XGBoost) — no graph | `src/baselines/tabular.py` |
| **L1** | Graph features → XGBoost — structure without NNs | `src/baselines/graph_features.py` |
| **L2** | Homogeneous GNN (GCN, GraphSAGE) | `src/homogeneous/` |
| **L3** | Heterogeneous GNN (HGT, HMPNN) | `src/heterogeneous/hgt/`, `src/heterogeneous/hmpnn/` |

L0→L1: does graph structure help at all?
L1→L2: do GNNs learn better representations than hand-crafted graph features?
L2→L3: does heterogeneous typing improve over homogeneous?
V1→V2→V3: which heterogeneous design choices matter?

Both **node classification** (transactions as nodes) and **edge classification** (transactions as edges) are supported. We test both formulations and commit to whichever performs better.

## Running Experiments

```bash
source .venv/bin/activate   # Python 3.14

# L0: Tabular baselines
python run.py --level 0

# L1: Graph features → XGBoost
python run.py --level 1

# L2: Homogeneous GNN
python run.py --level 2 --task node --conv sage
python run.py --level 2 --task edge --conv gcn

# L3: Heterogeneous GNN
python run.py --level 3 --task node --model hgt --variant txn_v1
python run.py --level 3 --task edge --model hgt --variant v1
python run.py --level 3 --task edge --model hmpnn --variant v2

# Dev mode (1% sample)
python run.py --level 3 --task edge --model hgt --variant v1 --sample 0.01
```

Key dependencies: `torch`, `torch-geometric`, `torch_scatter`, `torch_sparse`, `scikit-learn`, `pandas`, `numpy`, `xgboost`.

Primary metric: **PR-AUC** (precision-recall area under curve). Secondary: operational metrics (false positives per true positive).

---

## Bank Dataset — Danske Bank Retail Payments

### Data dictionary

1,502,051 retail payment transactions. Labels joined from a separate fraud table: `CONFIRMED_RISK = 1` means confirmed fraud (2,051 cases, 0.14%), `0` means absent from the fraud table (not a confirmed negative). Official column definitions:

| Column | Type | Remarks |
|---|---|---|
| `ACCAGENTCOUNTRY` | CHAR(2) | Debit country (ISO2) |
| `ACCOUNTAGENTID` | CHAR(8) | BIC — hardcoded values per country (8 unique values) |
| `ACCOUNTBRANCHID` | CHAR(4) | Branch of account |
| `ACCOUNTENTITYID` | CHAR(34) | Debit account (IBAN format in most cases) — same as ACCOUNTID |
| `ACCOUNTID` | CHAR(34) | Debit account (IBAN format in most cases) — primary sender key |
| `ACCOUNTIDFORMAT` | VARCHAR(5) | Debit account format (IBAN or BBAN) |
| `VALUE` | DECIMAL(15,2) | Original transaction value |
| `CURRENCY` | CHAR(3) | Original currency |
| `BASEVALUE` | DECIMAL(15,2) | Value converted to EUR |
| `BASECURRENCY` | CHAR(3) | Always EUR |
| `CHANNEL` | VARCHAR(15) | Payment channel |
| `COUNTERAGENTID` | CHAR(11) | Counterparty bank BIC (3,502 unique values) |
| `COUNTERBRANCHID` | CHAR(11) | Counterparty branch ID |
| `COUNTERENTITYID` | CHAR(34) | Counterparty account (IBAN where possible) — primary receiver key |
| `COUNTERPARTYID` | CHAR(34) | Same as COUNTERENTITYID — redundant, drop |
| `COUNTERIDFORMAT` | CHAR(4) | Counterparty account format (IBAN or BBAN) |
| `CUSTOMERENTITYID` | CHAR(10) | Customer number (internal) — same as CUSTOMERID |
| `CUSTOMERID` | CHAR(10) | Customer number (internal) — near 1:1 with ACCOUNTID |
| `CUSTOMERTYPE` | CHAR(6) | retail or busine (business) |
| `DESTINATIONCOUNTRY` | CHAR(2) | Destination country — creditor (ISO2) |
| `IPADDRESS` | CHAR(15) | IP address (IPv4) |
| `USERAGENTSTRING` | VARCHAR(384) | Browser/app user agent string |
| `DEVICEENTITYID` | VARCHAR(88) | Device ID — same as DEVICEID, redundant |
| `DEVICEID` | VARCHAR(88) | Device used to initiate the transaction (475k unique, no nulls) |
| `EVENTTIME` | TIMESTAMP | When payment was initiated — primary timestamp |
| `MSGSTATUS` | CHAR(3) | Hardcoded "new" — zero variance, drop |
| `PAYMENTCLEARING` | CHAR(7) | Clearing speed/urgency. Express = customer pays premium for speed |
| `PAYMENTMETHOD` | CHAR(6) | Payment method (online / file / bulk) |
| `PAYMENTSUBMETHOD` | VARCHAR(15) | Payment rail (realTime, bankGiro, plusGiro, futurePayment, salary, chaps, accountClosure) |
| `TRANSACTIONID` | CHAR(26) | Unique transaction identifier — join key only |
| `TRANSACTIONONUS` | VARCHAR(5) | true = Danske-to-Danske transfer; false = external bank |
| `EXCEPTIONRULE` | VARCHAR(78) | Hardcoded "noException" — zero variance, drop |
| `INTERNATIONALFLAG` | CHAR(5) | true if debit country ≠ destination country |
| `CONFIRMED_RISK` | INT | **Label.** 1 = confirmed fraud (from fraud labels table); 0 = not in fraud table |

**Key observations:**
- `TRANSACTIONONUS = false` means external bank — confirmed. Fraud rate: 0.15% external vs 0.087% on-us (~1.7× riskier)
- `BASECURRENCY` is always EUR — `BASEVALUE` is always the EUR-converted amount
- `EXCEPTIONRULE` and `MSGSTATUS` are hardcoded constants — zero variance, safe to drop
- `COUNTERPARTYID` = `COUNTERENTITYID` and `DEVICEENTITYID` = `DEVICEID` — confirmed redundant pairs
- `CONFIRMED_RISK = 0` is absence from the fraud table, not a confirmed negative

---

## Graph Pipelines

### Heterogeneous edge classification — `src/graph_pipeline_bank/`

Transactions are **edges** between account nodes. Config-driven pipeline.

```python
from src.utils.config import load_config
from src.graph_pipeline_bank import build_graph
result = build_graph(load_config("graph_bank_v1"))
data = result["data"]       # PyG HeteroData
```

**Pipeline steps:**
1. Loads parquet/CSV, drops redundant/zero-variance/leaky columns declared in config
2. Parses `EVENTTIME`, normalises `TRANSACTIONONUS` to bool, sorts chronologically
3. Applies optional sampling (random fraction or first N days)
4. Splits chronologically into train/val/test by cutoff dates
5. Builds node mappings: `InternalAccount`, `ExternalAccount`, optionally `Device`
6. Computes node features — **all aggregations use training rows only** to prevent temporal leakage
7. Routes transaction rows to edge relation types by `.query()` filters in config
8. Builds per-relation `edge_index`, `edge_attr`, `y`, masks
9. Assembles PyG `HeteroData` and pickles to cache

### Heterogeneous node classification — `src/graph_pipeline_bank_txn/`

Transactions are **nodes**. Account nodes are structural.

```python
from src.graph_pipeline_bank_txn import build_graph
result = build_graph(load_config("graph_bank_txn_v1"))
```

### Homogeneous graph — `src/homogeneous/`

All accounts collapsed into one type, all transactions into one edge/node type.

```python
from src.homogeneous.builder import build_homogeneous_graph
result = build_homogeneous_graph(load_config("graph_bank_v1"), mode="node")  # or "edge"
```

### Heterogeneous graph variants (edge classification)

**V1 — Baseline: onus / external split (`graph_bank_v1.yaml`)**
```
InternalAccount ──[onus_transfer]────► InternalAccount   TRANSACTIONONUS=True  (21.2%)
InternalAccount ──[external_transfer]► ExternalAccount   TRANSACTIONONUS=False (78.8%)
```

**V2 — Payment rail typed edges (`graph_bank_v2.yaml`)**
```
InternalAccount ──[onus_transfer]────► InternalAccount   TRANSACTIONONUS=True
InternalAccount ──[ext_realtime]─────► ExternalAccount   realTime
InternalAccount ──[ext_giro]─────────► ExternalAccount   bankGiro|plusGiro
InternalAccount ──[ext_future]───────► ExternalAccount   futurePayment
InternalAccount ──[ext_salary]───────► ExternalAccount   salary
InternalAccount ──[ext_other]────────► ExternalAccount   remaining submethods
```

**V3 — Device node type added (`graph_bank_v3.yaml`)**
```
InternalAccount ──[onus_transfer]────► InternalAccount
InternalAccount ──[external_transfer]► ExternalAccount
InternalAccount ──[uses_device]──────► Device            (structural, no labels)
```

---

## Code Structure

```
src/
  baselines/              # L0 + L1
    tabular.py            # LR, XGBoost on transaction features
    graph_features.py     # graph-structural features → XGBoost
  homogeneous/            # L2
    builder.py            # homogeneous graph construction (node or edge mode)
    models.py             # GCN, GraphSAGE
  heterogeneous/          # L3
    hgt/                  # Heterogeneous Graph Transformer
      model.py            # HGTConv wrapper (node + edge classification)
      train.py            # thin wrapper around unified Trainer
    hmpnn/                # Heterogeneous MPNN (Johannessen & Jullum)
      model.py            # NNConv + HeteroConv (node + edge classification)
  graph_pipeline_bank/    # L3 edge classification graph construction
    loader.py             # load parquet, parse dates, sample
    node_builder.py       # build node ID mappings
    features_node.py      # @register_*_feature decorators for node features
    features_edge.py      # @register_edge_feature for transaction features
    edge_builder.py       # filter → edge_index + edge_attr + labels + masks
    normalize.py          # zscore, one_hot, vocab utilities
    builder.py            # orchestrator → HeteroData
  graph_pipeline_bank_txn/  # L3 node classification graph construction
    builder.py            # transactions as nodes variant
  training/               # unified training loop
    trainer.py            # Trainer class: handles all model/task/graph combos
  utils/
    config.py             # load_config(), PROJECT_ROOT
    device.py             # get_device() — MPS/CUDA/CPU
    class_weights.py      # inverse-frequency weighting
    split.py              # temporal_split(), random_stratified_split()
    compat.py             # PyG API compatibility patches
  references/             # read-only upstream implementations (do not modify)
configs/
  graph_bank_v1.yaml      # hetero edge classification: onus/external (2 types)
  graph_bank_v2.yaml      # hetero edge classification: payment rail (6 types)
  graph_bank_v3.yaml      # hetero edge classification: + device nodes (3 types)
  graph_bank_txn_v1.yaml  # hetero node classification: transactions as nodes
run.py                    # main experiment runner (CLI)
```

### `src/utils/`
- `config.py` — `load_config(name)` loads `configs/{name}.yaml`; exports `PROJECT_ROOT`
- `device.py` — `get_device()` for MPS/CUDA/CPU
- `class_weights.py` — `compute_class_weights()` for imbalanced labels
- `split.py` — `temporal_split()`, `random_stratified_split()`
- `compat.py` — `apply_pyg_compat_patch()` for PyG API fixes

### `src/references/` — read-only upstream implementations
Do not modify. Reference only.

---

## Data & Output Paths (gitignored)

| Path | Contents |
|---|---|
| `datasets/bank_transactions.parquet` | Bank dataset |
| `data/processed/` | Cached PyG graph objects |
| `outputs/` | EDA outputs, data dictionary |
| `models/` | Saved model checkpoints |
| `results/` | Experiment result JSON logs |

## Node features (train-only aggregation)

- `InternalAccount`: out-degree, sent amount stats (mean/std/total, log1p), unique receivers, unique devices used, unique channels, night/weekend transaction ratios, customer type OHE
- `ExternalAccount`: in-degree, received amount stats, unique senders, unique sending banks
- `Device` (V3 only): transaction count, unique accounts per device

## Edge / transaction features (per-row, all splits)

log1p VALUE and BASEVALUE (z-scored), currency mismatch flag, channel/method/submethod/clearing OHE (vocab from training), international flag, sin/cos time encoding (hour-of-day + day-of-week)

## Columns dropped in all variants

| Column | Reason |
|---|---|
| `COUNTERPARTYID` | Identical to `COUNTERENTITYID` |
| `ACCOUNTENTITYID` | Same as `ACCOUNTID` |
| `CUSTOMERENTITYID` | Same as `CUSTOMERID` |
| `DEVICEENTITYID` | Same as `DEVICEID` |
| `CUSTOMERID` | Near 1:1 with `ACCOUNTID`; no Customer node type |
| `MSGSTATUS` | Hardcoded "new" |
| `EXCEPTIONRULE` | Hardcoded "noException" |
| `ACCOUNTIDFORMAT` / `COUNTERIDFORMAT` | Format metadata, not predictive |
| `ACCOUNTBRANCHID` / `COUNTERBRANCHID` | Subsumed by agent IDs |
