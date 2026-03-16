# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

MSc thesis project applying Heterogeneous Graph Neural Networks to Anti-Money Laundering (AML) fraud detection. Two datasets are used:

1. **SAML-D** — synthetic AML dataset. Modelled as a bipartite `Account ↔ Transaction` heterogeneous graph (node classification on transactions). Pipeline in `src/graph_pipeline/`.
2. **Danske Bank retail payments** — real bank transaction data (~1.5M rows). Modelled as a heterogeneous directed multigraph with transactions as **edges** (edge classification). Pipeline in `src/graph_pipeline_bank/`.

Three GNN architectures are compared: HGT, HGMAE, and HMPNN.

## Environment

```bash
source .venv/bin/activate   # Python 3.14 venv with all deps installed
```

Key dependencies: `torch`, `torch-geometric`, `torch_scatter`, `torch_sparse`, `scikit-learn`, `pandas`, `numpy`, `umap-learn`.

Datasets are gitignored. Processed graph caches go to `data/processed/` (also gitignored).

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

### Bank pipeline — `src/graph_pipeline_bank/`

Config-driven pipeline for the bank dataset. Entry point:
```python
from src.utils.config import load_config
from src.graph_pipeline_bank import build_graph
result = build_graph(load_config("graph_bank_v1"))
data = result["data"]       # PyG HeteroData
maps = result["node_maps"]  # {node_type: {raw_id: int_index}}
```

**What the pipeline does:**
1. Loads parquet/CSV, drops redundant/zero-variance/leaky columns declared in config
2. Parses `EVENTTIME`, normalises `TRANSACTIONONUS` to bool, sorts chronologically
3. Applies optional sampling (random fraction or first N days)
4. Splits chronologically into train/val/test by cutoff dates
5. Builds node mappings: `InternalAccount` (all `ACCOUNTID` values + on-us receivers), `ExternalAccount` (receivers never seen as senders), optionally `Device`
6. Computes node features using decorator registries — **all aggregations use training rows only** to prevent temporal leakage
7. Routes transaction rows to edge relation types by applying pandas `.query()` filters declared in config
8. Builds per-relation `edge_index`, `edge_attr`, `y`, `train_mask`, `val_mask`, `test_mask`
9. Assembles PyG `HeteroData` and pickles to cache

**Graph topology (V1):**
```
InternalAccount ──[onus_transfer]────► InternalAccount   (TRANSACTIONONUS=True)
InternalAccount ──[external_transfer]► ExternalAccount   (TRANSACTIONONUS=False)
```
Labels (`y`) and masks live on each edge type. This is **edge classification**, not node classification.

**Columns dropped in all variants:**

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

**Node features (train-only aggregation):**
- `InternalAccount`: out-degree, sent amount stats (mean/std/total, log1p), unique receivers, unique devices used, unique channels, night/weekend transaction ratios, customer type OHE
- `ExternalAccount`: in-degree, received amount stats, unique senders, unique sending banks (COUNTERAGENTID)
- `Device` (V3 only): transaction count, unique accounts per device

**Edge features (per-row, all splits):**
- log1p VALUE and BASEVALUE (z-scored), currency mismatch flag, channel/method/submethod/clearing OHE (vocab from training), international flag, sin/cos time encoding (hour-of-day + day-of-week)

**Graph variants:**

All variants share the same node types (InternalAccount, ExternalAccount) and the same node/edge feature sets. Only the edge relation typing changes.

**V1 — Baseline: onus / external split (`graph_bank_v1.yaml`)**
```
InternalAccount ──[onus_transfer]────► InternalAccount   TRANSACTIONONUS=True  (21.2%)
InternalAccount ──[external_transfer]► ExternalAccount   TRANSACTIONONUS=False (78.8%)
```
2 edge types. Hypothesis: structural separation of internal and external transactions improves edge classification over a homogeneous baseline. Motivated by external transactions being ~1.7× riskier than on-us.

**V2 — Payment rail typed edges (`graph_bank_v2.yaml`)**
```
InternalAccount ──[onus_transfer]────► InternalAccount   TRANSACTIONONUS=True
InternalAccount ──[ext_realtime]─────► ExternalAccount   TRANSACTIONONUS=False, PAYMENTSUBMETHOD=realTime
InternalAccount ──[ext_giro]─────────► ExternalAccount   TRANSACTIONONUS=False, PAYMENTSUBMETHOD=bankGiro|plusGiro
InternalAccount ──[ext_future]───────► ExternalAccount   TRANSACTIONONUS=False, PAYMENTSUBMETHOD=futurePayment
InternalAccount ──[ext_salary]───────► ExternalAccount   TRANSACTIONONUS=False, PAYMENTSUBMETHOD=salary
InternalAccount ──[ext_other]────────► ExternalAccount   TRANSACTIONONUS=False, remaining submethods
```
6 edge types. On-us kept as one type (structurally different, low fraud rate). External split by payment rail. bankGiro + plusGiro merged (both Nordic giro rails). Hypothesis: payment rail provides inductive bias — realTime (no interception window), salary (near-zero fraud), futurePayment (scheduled, different temporal profile) are structurally distinct. Trade-off: sparser graphs per relation type.

**V3 — Device node type added (`graph_bank_v3.yaml`)**
```
InternalAccount ──[onus_transfer]────► InternalAccount   (same as V1)
InternalAccount ──[external_transfer]► ExternalAccount   (same as V1)
InternalAccount ──[uses_device]──────► Device            one edge per unique (ACCOUNTID, DEVICEID) pair
```
3 edge types + Device node type (475k nodes). uses_device edges are deduplicated (one per unique account-device pair), structural-only (no edge features, no label). Hypothesis: device-linkage connectivity reveals multi-account fraud rings invisible in transaction-only graphs — a single device used across many accounts indicates one actor controlling multiple accounts. Only InternalAccount nodes have DEVICEID; ExternalAccount nodes do not.

---

## SAML-D Pipeline — `src/graph_pipeline/`

Orchestrated by `graph_builder.py`. Reads raw CSV → temporal train/val/test splits → extracts features → builds PyG `HeteroData`.

- `schema.py` — maps logical column roles to CSV column names. **Change dataset here.**
- `loader.py` — reads CSV/parquet, parses dates, applies `sample_ratio`
- `split.py` — temporal masks (`train_mask`, `val_mask`, `test_mask` on transaction nodes)
- `features_txn.py` — per-row transaction features via `@register_txn_feature` decorator
- `features_acct.py` — aggregated account features via `@register_acct_feature` decorator. **Always filter to training data.**
- `normalize.py` — `zscore()`, `one_hot()` utilities
- `cache.py` — pickle-based cache

**Graph topology:**
```
Account ──[sends]──────────▶ Transaction ──[received_by]──▶ Account
Account ◀──[receives]────── Transaction ◀──[sent_by]────── Account
```
Labels (`y`) and masks live on `data['transaction']`. This is **node classification**.

Config: `configs/graph_pipeline.yaml`. Set `sample_ratio: 0.01` for dev.

```bash
python -c "from src.utils.config import load_config; from src.graph_pipeline import build_graph; data, acct = build_graph(load_config('graph_pipeline'))"
python -c "from src.graph_pipeline import print_feature_inventory; print_feature_inventory()"
```

---

## GNN Models

### `src/hgt/` — Heterogeneous Graph Transformer
Supervised node classification. Wraps PyG's `HGTConv` with per-type input projections and a transaction-node classification head. Train/val/test loop with early stopping on AUPRC.

### `src/hgmae/` — Heterogeneous Graph Masked Autoencoder
Self-supervised pre-training (masked feature reconstruction), then frozen embeddings + logistic regression probe. Ported from DGL reference in `src/references/HGMAE/`.

### `src/hmpnn/` — Heterogeneous MPNN
Edge-feature-aware message passing via `NNConv` + `HeteroConv`. Adapted from `src/references/heterogeneous-mpnn/`.

### `src/utils/`
- `config.py` — `load_config(name)` loads `configs/{name}.yaml`; exports `PROJECT_ROOT`
- `device.py` — `get_device()` for MPS/CUDA/CPU
- `class_weights.py` — `compute_class_weights()` for imbalanced labels
- `compat.py` — `apply_pyg_compat_patch()` for PyG API fixes

### `src/references/` — read-only upstream implementations
Do not modify. Adapt into `src/hgmae/` or `src/hmpnn/` instead.

---

## Data & Output Paths (gitignored)

| Path | Contents |
|---|---|
| `datasets/SAML-D.csv` | SAML-D raw dataset |
| `datasets/bank_transactions.parquet` | Bank dataset |
| `data/processed/` | Cached PyG graph objects |
| `outputs/` | EDA outputs, thesis writing, data dictionary |
| `models/` | Saved model checkpoints |
| `results/**/*.md` | Experiment result logs (tracked) |

## Key outputs
- `outputs/data_dictionary.md` — full column reference + graph design decisions
- `outputs/thesis_data_section.tex` — LaTeX data chapter (embeddable in full thesis)
- `outputs/pipeline_todo.md` — theoretical framework notes (transductive/inductive, directed multigraph, Node2Vec, LaundroGraph)
