# Graph Pipeline

Extensible heterogeneous graph construction for AML fraud detection.
Builds a bipartite `Account ↔ Transaction` graph from tabular data using PyG `HeteroData`.

## Quick Start

```bash
source .venv/bin/activate

# Build the graph (uses configs/graph_pipeline.yaml)
python -c "
from src.utils.config import load_config
from src.graph_pipeline import build_graph
data, account_to_id = build_graph(load_config('graph_pipeline'))
"

# Sanity check: see all registered features and their dimensions
python -c "from src.graph_pipeline import print_feature_inventory; print_feature_inventory()"

# Same but as pandas DataFrames (nicer in notebooks)
python -c "from src.graph_pipeline import feature_table; feature_table()"

# Export feature table (for thesis, docs, or spreadsheets)
python -c "from src.graph_pipeline import print_feature_inventory; print_feature_inventory('outputs/features.html')"
python -c "from src.graph_pipeline import print_feature_inventory; print_feature_inventory('outputs/features.md')"

# Visualize the graph
python scripts/visualize_graph.py
```

## Config: `configs/graph_pipeline.yaml`

```yaml
dataset: saml-d                # which schema to use (saml-d | bank-retail)
data_path: datasets/SAML-D.csv
sample_ratio: 0.01             # 1% for dev, 1.0 for full run

split:
  train_end_date: "2023-04-01" # everything before this = train
  val_end_date: "2023-06-01"   # train_end → val_end = val, after = test

graph:
  add_reverse_edges: true      # adds sent_by and receives edges

features:
  transaction: null             # null = all registered features
  account: null                 # or list specific ones: [degree, amount_stats]

cache:
  enabled: true
  dir: data/processed
```

## Architecture

```
configs/graph_pipeline.yaml
        │
        ▼
  graph_builder.py  ◄── orchestrator, calls everything below
        │
        ├── loader.py         read CSV/parquet, parse dates, sample
        ├── split.py          temporal train/val/test masks
        ├── features_txn.py   transaction feature extractors (registry)
        ├── features_acct.py  account feature extractors (registry, train-only)
        ├── normalize.py      zscore, one_hot utilities
        ├── schema.py         column name mapping (SAML-D ↔ bank data)
        └── cache.py          pickle save/load
```

## Features

Features are registered dynamically — the actual dims depend on which features are registered and which dataset you're using. To see the current inventory:

```bash
python -c "from src.graph_pipeline import print_feature_inventory; print_feature_inventory()"
python -c "from src.graph_pipeline import feature_table; feature_table()"
```

**Transaction features** are per-row properties (amount, currencies, locations, time encodings, flags).
Each extractor checks if the required columns exist in the schema — if not, it returns `None` and is skipped.

**Account features** are aggregated from transactions (degree, amount stats, diversity metrics, etc.).
**All account features are computed from training data only** to prevent temporal leakage.

## How to Add a New Feature

### Transaction feature (per-row)

In `features_txn.py`:

```python
@register_txn_feature("channel", dim=5)
def _channel(df, schema):
    if schema.channel is None:
        return None                  # skipped if column doesn't exist
    return one_hot(df[schema.channel], CHANNEL_VOCAB)
```

### Account feature (aggregated)

In `features_acct.py`:

```python
@register_acct_feature("velocity_7d", dim=2)
def _velocity_7d(df, schema, account_to_id, train_mask):
    train_df = df[train_mask]        # ALWAYS filter to training data
    cutoff = train_df["_datetime"].max() - pd.Timedelta(days=7)
    recent = train_df[train_df["_datetime"] >= cutoff]
    # ... aggregate per account ...
    return features                  # shape (num_accounts, 2)
```

That's it. The registry discovers it automatically. Run `print_feature_inventory()` to verify.

## How to Switch Datasets

The `schema.py` file maps logical column roles to actual column names:

```
SAML-D:        Sender_account  →  schema.sender_id
Bank data:     ACCOUNTID       →  schema.sender_id
```

To use bank data:
1. Add/update the schema in `schema.py`
2. Set `dataset: bank-retail` in the YAML config
3. Add new feature extractors for bank-specific columns (CHANNEL, DEVICEID, etc.)
4. Run `build_graph()` — same code, different data

## Graph Topology

```
Account ──[sends]──────────▶ Transaction ──[received_by]──▶ Account
Account ◀──[receives]────── Transaction ◀──[sent_by]────── Account
                            (reverse edges, optional)
```

- **Account nodes**: unified sender + receiver pool (same account = same node)
- **Transaction nodes**: one per row in the dataset
- **Edges**: each transaction creates 2 forward edges (+ 2 reverse if enabled)
- **Labels**: `data['transaction'].y` — binary fraud/laundering flag
- **Masks**: `data['transaction'].train_mask`, `val_mask`, `test_mask`

## Output: PyG HeteroData

```python
data['account'].x              # (num_accounts, acct_feat_dim)  float tensor
data['transaction'].x          # (num_txns, txn_feat_dim)       float tensor
data['transaction'].y          # (num_txns,)                    fraud labels
data['transaction'].train_mask # (num_txns,)                    bool
data['transaction'].val_mask   # (num_txns,)                    bool
data['transaction'].test_mask  # (num_txns,)                    bool
data[('account', 'sends', 'transaction')].edge_index            # (2, num_txns)
data[('transaction', 'received_by', 'account')].edge_index      # (2, num_txns)
# + reverse edges if enabled
```

This object is passed directly to GNN models (HGT, HAN, HMPNN, etc.).
