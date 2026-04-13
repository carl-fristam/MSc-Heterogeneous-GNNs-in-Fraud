"""
Main experiment runner.

Experiment ladder:

  tab  — Tabular baseline (XGBoost, no graph)
  homo — Homogeneous GNN  (gcn | sage | gat)
  het  — Heterogeneous GNN (hgt | hmpnn | hetero_gat)

Usage:
    python run.py --mode tab
    python run.py --mode homo --model sage
    python run.py --mode het --model hgt
    python run.py --mode het --model hgt --sample 0.05   # dev run
"""

import argparse

from src.utils.config import load_variant, PROJECT_ROOT
from src.utils.device import get_device
from src.utils.results import save_results
from src.data.prepare import prepare_data


def run_tab(prep, tune: bool = False, n_trials: int = 50):
    from src.baselines.tabular import run_tabular_baselines
    return run_tabular_baselines(prep, tune=tune, n_trials=n_trials)


def run_homo(prep, config, model_name="sage", **kwargs):
    from src.graph_builder.assembler import build_graph
    from src.homogeneous.builder import project_to_homo
    from src.homogeneous.models import HomoGNN
    from src.training.trainer import Trainer, TrainConfig

    device = get_device()
    data   = project_to_homo(build_graph(config, prep)["data"]).to(device)

    model = HomoGNN(
        data,
        conv_type  = model_name,
        hidden_dim = kwargs.get("hidden_dim", 64),
        num_layers = kwargs.get("num_layers", 2),
        num_heads  = kwargs.get("num_heads",  4),
        dropout    = kwargs.get("dropout",    0.3),
    )

    trainer = Trainer(model, data, TrainConfig(
        task        = "edge",
        graph_type  = "homo",
        epochs      = kwargs.get("epochs",   200),
        lr          = kwargs.get("lr",       1e-3),
        patience    = kwargs.get("patience", 15),
    ), device)

    return trainer.run()


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

    return trainer.run()


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Run fraud detection experiments")
    parser.add_argument("--mode",   type=str, choices=["tab", "homo", "het"], required=True,
                        help="tab=tabular  homo=homo GNN  het=hetero GNN")
    parser.add_argument("--model",  type=str, default="hgt",
                        choices=["gcn", "sage", "gat", "hgt", "hmpnn", "hetero_gat"])
    parser.add_argument("--sample", type=float, default=None,
                        help="Fraction of data to use (e.g. 0.05 for dev runs)")
    parser.add_argument("--data-path", type=str, default=None)

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

    args   = parser.parse_args()
    config = load_variant("v1")

    # ── Dataset selection ─────────────────────────────────────────────────────
    datasets_dir = PROJECT_ROOT / "datasets"
    if args.data_path is not None:
        config["data_path"] = args.data_path
    else:
        print("\nAvailable datasets:")
        files = sorted(datasets_dir.glob("*.parquet"))
        if not files:
            print("  No .parquet files found in datasets/")
            raise SystemExit(1)
        for i, f in enumerate(files, 1):
            print(f"  {i}. {f.name}")
        choice = input("\nSelect dataset (number or filename): ").strip()
        if choice.isdigit() and 1 <= int(choice) <= len(files):
            config["data_path"] = str(files[int(choice) - 1])
        else:
            config["data_path"] = str(datasets_dir / choice)

    if args.sample is not None:
        config["sample_ratio"] = args.sample

    prep = prepare_data(config)

    model_kwargs = {
        "hidden_dim": args.hidden_dim,
        "num_layers": args.num_layers,
        "num_heads":  args.num_heads,
        "epochs":     args.epochs,
        "lr":         args.lr,
        "patience":   args.patience,
    }

    if args.mode == "tab":
        results = run_tab(prep, tune=args.tune, n_trials=args.n_trials)
    elif args.mode == "homo":
        results = run_homo(prep, config, model_name=args.model, **model_kwargs)
    elif args.mode == "het":
        results = run_het(prep, config, model_name=args.model, **model_kwargs)

    save_kwargs = {"model": args.model if args.mode != "tab" else None}
    if isinstance(results, list):
        for r in results:
            if r: save_results(r, args.mode, **save_kwargs)
    elif results:
        save_results(results, args.mode, **save_kwargs)


if __name__ == "__main__":
    main()
