# Heterogeneous Graph Neural Networks for Transaction-Level Fraud Detection at Danske Bank

MSc in Data Science — Copenhagen Business School, 2026.

Compares tabular baselines (XGBoost) against heterogeneous GNN architectures (HGT, HMPNN, HeteroGAT) for edge-level fraud classification on retail payment data.

## Project structure

```
run.py                  # single entry point for all experiments
configs/master.yaml     # feature lists, column mappings, graph topology
src/
  data/prepare.py       # loads pre-split data into PreparedData
  graph_builder/        # builds PyG HeteroData from transactions
    node_builder.py     # account ID -> integer index mappings
    node_features.py    # per-account feature tensors
    edge_builder.py     # edge index, features, labels, masks
    assembler.py        # orchestrates build, caches to disk
  heterogeneous/        # GNN model implementations
    hgt/                # Heterogeneous Graph Transformer
    hmpnn/              # Heterogeneous Message Passing Neural Network
    hetero_gat/         # Heterogeneous GAT
  baselines/tabular.py  # XGBoost baseline
  training/trainer.py   # unified training loop for all GNN models
  utils/                # config, device, results, threshold table
```

## Data

Data is not included in this repository. It consists of pre-split parquet files produced by an external feature engineering pipeline and must be placed in:

```
datasets/splits/
  train.parquet
  val.parquet
  test.parquet
```

Each file contains one row per transaction with pre-computed features (OHE, target encodings, velocity features, cyclical time encodings), a binary label column (`CONFIRMEDRISK`), and account identifiers. The splits are temporal: training covers earlier months, validation and test cover successively later periods. This pre-processing and feature engineering was made in a separate pipeline on internal compute at DB, due to the sensitivity of the data.

## Usage

The following commands were used to run the pipeline during testing. Ad-hoc scripts were made to orchestrate on the compute network within DB.

```bash
# Tabular baseline (XGBoost, lean features)
python run.py --mode tab

# Tabular baseline (full features, Bayesian optimisation)
python run.py --mode tab --full-features --tune --n-trials 50

# Heterogeneous GNN
python run.py --mode het --model hgt
python run.py --mode het --model hmpnn
python run.py --mode het --model hetero_gat

# With customer nodes (adds owns/owned_by structural edges)
python run.py --mode het --model hmpnn --customer-nodes

# Downsample to 50% (keeps all fraud, samples legitimate transactions)
python run.py --mode het --model hgt --sample 0.5
```

## Requirements

```bash
pip install -r requirements.txt
```