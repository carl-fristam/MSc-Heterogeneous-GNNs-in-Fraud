# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

MSc thesis project at Copenhagen Business School investigating Graph Neural Networks (GNNs) and Heterogeneous Graph Masked Autoencoders (HGMAE) for detecting money laundering patterns in financial transaction networks.

**Current Status**: Baseline GNN models (GCN, GraphSAGE) implemented; transitioning to heterogeneous graph approaches for edge (transaction) classification.

## Commands

### Environment Setup
```bash
source .venv/bin/activate
pip install -r requirements.txt
```

### Training Models
```bash
# GCN model (node classification on SAML-D)
python src/train_gcn.py
# Output: src/models/gcn_saml.pt

# GraphSAGE model
cd src/graphsage && python main.py
# Output: outputs/graphsage_model.pt
```

### Data Preparation
```bash
# Generate SAML-D graph cache (required before training GCN)
python src/data/saml_data.py
# Output: data/processed/saml_graph.pkl

# Inspect cached graph
python src/data/inspect-saml.py
```

## Architecture

### Data Pipeline

**SAML-D Dataset** (~9.5M transactions, ~0.14% laundering rate):
- `src/data/saml_data.py` - Main loader with caching. Creates PyG Data object with:
  - Node features (8-dim): in/out degree, total/avg amounts sent/received, unique counterparty counts (z-score normalized)
  - Edge index from sender→receiver account pairs
  - Node labels: binary (1 if account initiated any laundering transaction)
- Cache location: `data/processed/saml_graph.pkl`

**HI-Small Dataset** (alternative, used by GraphSAGE):
- `src/graphsage/data.py` - Separate loader for smaller dataset

### Model Implementations

**GCN** (`src/gcn_model.py`, `src/train_gcn.py`):
- Homogeneous node classification
- Architecture: GCNConv layers + BatchNorm + Dropout
- Includes `compute_class_weights()` for imbalanced data handling
- Evaluation metrics: Accuracy, Precision, Recall, F1, ROC-AUC

**GraphSAGE** (`src/graphsage/`):
- Inductive learning with neighbor sampling (NeighborLoader)
- Mini-batch training (batch_size=4096)
- MPS (Apple Silicon) acceleration with CPU fallback

### Device Support

Models are optimized for Apple Silicon MPS. The GraphSAGE pipeline includes:
```python
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
```

## Key Technical Details

- PyTorch Geometric (PyG) is the GNN framework
- Class imbalance is severe (~99.86% negative) - always use weighted loss or evaluation metrics that account for this
- GraphSAGE includes a Python 3.14 compatibility fix for torch_geometric typing issues
- Data files (*.csv, *.pkl, *.pt) are gitignored - models and processed data stay local
