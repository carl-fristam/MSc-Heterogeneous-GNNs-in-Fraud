# CLAUDE.md

MSc thesis (CBS) — GNNs and Heterogeneous Graph Masked Autoencoders (HGMAE) for anti-money laundering on financial transaction networks.

## Quick Reference

```bash
source .venv/bin/activate              # Environment
python scripts/prepare_data.py --all   # Build graph caches (data/processed/)
python scripts/train_<model>.py        # Train: gcn | graphsage | hmpnn | hgmae | supervised
```

## Layout

- `configs/*.yaml` — experiment hyperparameters, loaded via `load_config("name")`
- `scripts/` — CLI entry points (one per model + `prepare_data.py`)
- `src/data/` — SAML-D graph loaders (homogeneous & heterogeneous)
- `src/baselines/` — GCN, GraphSAGE (homogeneous node classification)
- `src/supervised/` — XGBoost, RF, LR tabular baselines
- `src/hmpnn/` — Heterogeneous MPNN (NNConv + edge features)
- `src/hgmae/` — HGMAE self-supervised pretraining; custom PyG-native HAN encoder (`han_pyg.py`) + DGL-stubbed adapter (`premodel_adapter.py`) wrapping reference HGMAE code
- `src/utils/` — shared: device selection, metrics, class weights, config, PyG compat patch
- `src/references/` — vendored reference implementations (read-only)

## Conventions

- **Framework**: PyTorch Geometric (PyG)
- **Device**: always use `get_device()` from `src/utils/device` (CUDA > MPS > CPU)
- **Paths**: use `PROJECT_ROOT` from `src/utils/config` — no hardcoded absolute paths
- **Compat**: call `apply_pyg_compat_patch()` before importing PyG conv layers (Python 3.14 fix)
- **Class imbalance**: SAML-D is ~0.14% positive — always use weighted loss or appropriate metrics
- **Data files**: *.csv, *.pkl, *.pt are gitignored — stay local
