"""
Main experiment runner.

Experiment ladder:

  tab   — Tabular baseline (XGBoost, no graph)
  het   — Heterogeneous GNN (hgt | hmpnn | hetero_gat)

Data arrives as pre-split parquet/csv files (train, val, test).
No feature engineering or splitting is done in this repo.

Usage:
    python run.py --mode tab
    python run.py --mode het --model hgt
    python run.py --mode het --model hmpnn --sample 0.5
    python run.py --mode het --model hgt --full-features
"""

import argparse
import io
import sys

from src.utils.config import load_variant, PROJECT_ROOT
from src.utils.device import get_device
from src.utils.results import save_results
from src.data.prepare import prepare_data


class _Tee:
    """Write to both stdout and a StringIO buffer."""
    def __init__(self, original):
        self.original = original
        self.buffer = io.StringIO()

    def write(self, msg):
        self.original.write(msg)
        self.buffer.write(msg)

    def flush(self):
        self.original.flush()
        self.buffer.flush()

    def getvalue(self):
        return self.buffer.getvalue()


def run_tab(prep, tune: bool = False, n_trials: int = 50):
    from src.baselines.tabular import run_tabular_baselines
    return run_tabular_baselines(prep, tune=tune, n_trials=n_trials)


def run_het(prep, config, model_name="hgt", **kwargs):
    from src.graph_builder.assembler import build_graph
    from src.training.trainer import Trainer, TrainConfig

    device           = get_device()
    target_node_type = "internal_account"
    data             = build_graph(config, prep)["data"].to(device)

    if model_name == "hgt":
        from src.heterogeneous.hgt.model import HGT
        model = HGT(
            data,
            hidden_dim       = kwargs.get("hidden_dim", 64),
            num_heads        = kwargs.get("num_heads",  4),
            num_layers       = kwargs.get("num_layers", 2),
            dropout          = kwargs.get("dropout",    0.3),
            task             = "edge",
            target_node_type = target_node_type,
        )
    elif model_name == "hmpnn":
        from src.heterogeneous.hmpnn.model import HMPNN
        model = HMPNN(
            data,
            target_node_type = target_node_type,
            num_layers       = kwargs.get("num_layers",  2),
            hidden_dim       = kwargs.get("hidden_dim",  64),
            message_dim      = kwargs.get("message_dim", 32),
            dropout          = kwargs.get("dropout",     0.3),
            task             = "edge",
            aggr             = kwargs.get("aggr",        "ct"),
        )
    elif model_name == "hetero_gat":
        from src.heterogeneous.hetero_gat.model import HeteroGAT
        model = HeteroGAT(
            data,
            hidden_dim       = kwargs.get("hidden_dim", 64),
            num_heads        = kwargs.get("num_heads",  4),
            num_layers       = kwargs.get("num_layers", 2),
            dropout          = kwargs.get("dropout",    0.3),
            task             = "edge",
            target_node_type = target_node_type,
        )
    else:
        raise ValueError(f"Unknown het model: {model_name!r}. Choose hgt | hmpnn | hetero_gat")

    trainer = Trainer(model, data, TrainConfig(
        task             = "edge",
        graph_type       = "hetero",
        target_node_type = target_node_type,
        epochs           = kwargs.get("epochs",   200),
        lr               = kwargs.get("lr",       1e-3),
        patience         = kwargs.get("patience", 15),
    ), device)

    metrics = trainer.run()
    if trainer._best_state is not None:
        metrics["_model_state"] = trainer._best_state
    return metrics


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Run fraud detection experiments")
    parser.add_argument("--mode",   type=str,
                        choices=["tab", "het"], required=True,
                        help="tab | het")
    parser.add_argument("--model",  type=str, default="hgt",
                        choices=["hgt", "hmpnn", "hetero_gat"])

    # Data
    parser.add_argument("--sample", type=float, default=None,
                        help="Stratified temporal sample fraction (e.g. 0.5)")
    parser.add_argument("--proportional-sample", action="store_true",
                        help="Sample fraud proportionally too (default: keep all fraud)")
    parser.add_argument("--full-features", action="store_true",
                        help="Use all 91 edge features instead of lean 30 (GNN only)")

    # Tabular tuning
    parser.add_argument("--tune",     action="store_true")
    parser.add_argument("--n-trials", type=int, default=50)

    # Hyperparameters
    parser.add_argument("--hidden-dim",  type=int,   default=64)
    parser.add_argument("--num-layers",  type=int,   default=2)
    parser.add_argument("--num-heads",   type=int,   default=4)
    parser.add_argument("--epochs",      type=int,   default=200)
    parser.add_argument("--lr",          type=float, default=1e-3)
    parser.add_argument("--patience",    type=int,   default=15)

    # Output
    parser.add_argument("--results-dir", type=str, default=None,
                        help="Override results directory (e.g. results/tuning)")

    args = parser.parse_args()

    tee = _Tee(sys.stdout)
    sys.stdout = tee

    config = load_variant("v1")

    # Select edge feature set for GNNs
    if args.full_features:
        config["edge_features"] = config["edge_features_full"]
        print(f"\nEdge features: FULL ({len(config['edge_features'])} dims)")
    else:
        print(f"\nEdge features: LEAN ({len(config['edge_features'])} dims)")

    prep = prepare_data(config, sample=args.sample,
                        keep_all_fraud=not args.proportional_sample)

    model_kwargs = {
        "hidden_dim":  args.hidden_dim,
        "num_layers":  args.num_layers,
        "num_heads":   args.num_heads,
        "epochs":      args.epochs,
        "lr":          args.lr,
        "patience":    args.patience,
    }

    if args.mode == "tab":
        results = run_tab(prep, tune=args.tune, n_trials=args.n_trials)
    elif args.mode == "het":
        results = run_het(prep, config, model_name=args.model, **model_kwargs)

    console_log = tee.getvalue()
    sys.stdout = tee.original

    save_kwargs = {"model": args.model if args.mode != "tab" else None,
                   "console_log": console_log}
    if args.results_dir:
        save_kwargs["results_dir_override"] = args.results_dir
    if args.mode == "het":
        save_kwargs["hyperparams"] = model_kwargs
        save_kwargs["sample"] = args.sample
        save_kwargs["full_features"] = args.full_features
    if isinstance(results, list):
        for r in results:
            if r: _save(r, args.mode, **save_kwargs)
    elif results:
        _save(results, args.mode, **save_kwargs)


def _save(metrics: dict, mode: str, **kwargs):
    extra = {}
    for key in ("_y_true", "_y_prob", "_xgb_model", "_feature_names", "_analysis", "_model_state"):
        if key in metrics:
            extra[key.lstrip("_")] = metrics.pop(key)
    save_results(metrics, mode, **extra, **kwargs)


if __name__ == "__main__":
    main()
