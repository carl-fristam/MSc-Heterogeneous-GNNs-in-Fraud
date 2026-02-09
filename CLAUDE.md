# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

MSc thesis project at Copenhagen Business School investigating Graph Neural Networks (GNNs) and Heterogeneous Graph Masked Autoencoders (HGMAE) for detecting money laundering patterns in financial transaction networks.

**Current Status**: Baseline GNN models (GCN, GraphSAGE) and supervised ML baselines (XGBoost, RF, LR) implemented. HMPNN heterogeneous model ported. Transitioning to heterogeneous graph approaches for edge (transaction) classification.

## Commands

### Environment Setup
```bash
source .venv/bin/activate
pip install -r requirements.txt
```

### Training Models
```bash
# GCN model (node classification on SAML-D)
python scripts/train_gcn.py

# GraphSAGE model (node classification on HI-Small)
python scripts/train_graphsage.py

# Supervised ML baselines (XGBoost, RF, LR on SAML-D)
python scripts/train_supervised.py

# HMPNN heterogeneous model (SAML-D)
python scripts/train_hmpnn.py
```

### Data Preparation
```bash
# Generate SAML-D homogeneous graph cache
python scripts/prepare_data.py

# Generate SAML-D heterogeneous graph cache
python scripts/prepare_data.py --hetero

# Both
python scripts/prepare_data.py --all

# With sampling (for testing)
python scripts/prepare_data.py --sample 0.01
```

## Project Structure

```
MSc-GNNs-in-AML/
├── configs/                        # YAML experiment configs
│   ├── gcn.yaml
│   ├── graphsage.yaml
│   ├── supervised.yaml
│   └── hmpnn.yaml
├── scripts/                        # Top-level entry points
│   ├── train_gcn.py
│   ├── train_graphsage.py
│   ├── train_supervised.py
│   ├── train_hmpnn.py
│   └── prepare_data.py
├── src/
│   ├── utils/                      # Shared utilities
│   │   ├── compat.py               # Python 3.14 PyG monkeypatch
│   │   ├── device.py               # Unified MPS/CUDA/CPU device selection
│   │   ├── evaluation.py           # Shared metrics + data splits
│   │   ├── class_weights.py        # Inverse-frequency class weights
│   │   └── config.py               # YAML loader + PROJECT_ROOT
│   ├── data/                       # Graph data loaders
│   │   ├── saml_homo.py            # SAML-D → PyG Data (homogeneous)
│   │   ├── saml_hetero.py          # SAML-D → PyG HeteroData (heterogeneous)
│   │   └── inspect_graph.py        # Graph inspection utility
│   ├── baselines/                  # GNN baseline models
│   │   ├── gcn/                    # GCN (GCNConv + BatchNorm + Dropout)
│   │   │   ├── model.py
│   │   │   └── train.py
│   │   └── graphsage/              # GraphSAGE (NeighborLoader mini-batching)
│   │       ├── model.py
│   │       ├── data.py             # HI-Small dataset loader
│   │       ├── train.py
│   │       └── main.py
│   ├── supervised/                 # Tabular ML baselines (XGBoost, RF, LR)
│   │   ├── data_prep.py
│   │   ├── models.py
│   │   ├── train.py
│   │   └── eval.py
│   └── hmpnn/                      # Heterogeneous MPNN
│       └── model.py
├── results/                        # Experiment result logs
│   └── supervised/
├── datasets/                       # Raw data (gitignored)
├── notebooks/                      # Jupyter notebooks
├── paper/                          # Thesis LaTeX
└── references/                     # Literature & reference code
```

## Architecture

### Data Pipeline

**SAML-D Dataset** (~9.5M transactions, ~0.14% laundering rate):
- `src/data/saml_homo.py` - Homogeneous graph loader with caching. Creates PyG Data object with:
  - Node features (8-dim): in/out degree, total/avg amounts sent/received, unique counterparty counts (z-score normalized)
  - Edge index from sender->receiver account pairs
  - Node labels: binary (1 if account initiated any laundering transaction)
- `src/data/saml_hetero.py` - Heterogeneous graph loader. Creates PyG HeteroData with:
  - Edge types based on Payment_type (credit_card, debit_card, cheque, ach, cross_border, cash_withdrawal, cash_deposit)
  - Edge features: amount, hour, is_cross_border, is_same_currency, is_laundering
- Cache location: `data/processed/`

**HI-Small Dataset** (alternative, used by GraphSAGE):
- `src/baselines/graphsage/data.py` - Separate loader for smaller dataset

### Model Implementations

**GCN** (`src/baselines/gcn/`):
- Homogeneous node classification
- Architecture: GCNConv layers + BatchNorm + Dropout
- Uses shared `compute_class_weights()` for imbalanced data handling

**GraphSAGE** (`src/baselines/graphsage/`):
- Inductive learning with neighbor sampling (NeighborLoader)
- Mini-batch training (batch_size=1024)

**Supervised ML** (`src/supervised/`):
- XGBoost, RandomForest, LogisticRegression
- SMOTE/undersampling for class imbalance
- Threshold tuning for precision-recall trade-off

**HMPNN** (`src/hmpnn/`):
- Heterogeneous Message Passing Neural Network
- Uses NNConv with edge features per payment type
- Supports 1-3 layer architectures

### Shared Utilities (`src/utils/`)

- **`device.py`**: `get_device()` — CUDA > MPS > CPU
- **`compat.py`**: `apply_pyg_compat_patch()` — Python 3.14 fix for PyG typing (idempotent)
- **`evaluation.py`**: `compute_metrics()`, `print_metrics()`, `create_splits()` — shared across GNN models
- **`class_weights.py`**: `compute_class_weights()` — inverse-frequency weighting
- **`config.py`**: `PROJECT_ROOT`, `load_config()` — YAML config loading

### Configuration

Experiment hyperparameters live in `configs/*.yaml`. Load with:
```python
from src.utils.config import load_config
cfg = load_config("gcn")  # loads configs/gcn.yaml
```

### Device Support

All models use unified device selection via `src/utils/device.py`:
```python
from src.utils.device import get_device
device = get_device()  # CUDA > MPS > CPU
```

## Key Technical Details

- PyTorch Geometric (PyG) is the GNN framework
- Class imbalance is severe (~99.86% negative) — always use weighted loss or evaluation metrics that account for this
- Python 3.14 compatibility patch in `src/utils/compat.py` — call `apply_pyg_compat_patch()` before importing PyG conv layers
- Data files (*.csv, *.pkl, *.pt) are gitignored — models and processed data stay local
- All paths use `PROJECT_ROOT` from `src/utils/config` — no hardcoded absolute paths
