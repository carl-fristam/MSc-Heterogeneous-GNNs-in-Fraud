"""
Main experiment runner.

Experiment ladder:

  L0 — Tabular baseline (XGBoost, no graph)
        Establishes the floor. Everything above must beat this.

  L1 — Homogeneous GNN (edge classification on projected graph)
        Same transactions, same features — but heterogeneous structure
        collapsed to a single node/edge type.
        Models: gcn | sage | gat
        Isolates: does any graph representation help?

  L2 — Heterogeneous GNN (edge classification on full hetero graph)
        Full heterogeneous structure with typed nodes and edges.
        Models: hgt | hmpnn | hetero_gat
        Isolates: does preserving heterogeneous structure help over homo?

Usage:
    # L0: Tabular XGBoost
    python run.py --level 0

    # L1: Homogeneous GNN
    python run.py --level 1 --model sage --variant v1
    python run.py --level 1 --model gat  --variant v1
    python run.py --level 1 --model gcn  --variant v1

    # L2: Heterogeneous GNN
    python run.py --level 2 --model hgt       --variant v1
    python run.py --level 2 --model hmpnn     --variant v2
    python run.py --level 2 --model hetero_gat --variant v1

    # Dev mode (5% sample — fast sanity check)
    python run.py --level 2 --model hgt --variant v1 --sample 0.05

    # Run all levels with a single model per level
    python run.py --all --variant v1
"""

import argparse
import json
from datetime import datetime
from pathlib import Path

from src.utils.config import load_config, load_variant, PROJECT_ROOT
from src.utils.device import get_device
from src.data.prepare import prepare_data


def run_l0(prep, tune: bool = False, n_trials: int = 50):
    """L0: Tabular XGBoost — no graph."""
    from src.baselines.tabular import run_tabular_baselines
    return run_tabular_baselines(prep, tune=tune, n_trials=n_trials)


def run_l1(prep, config, model_name="sage", **kwargs):
    """L1: Homogeneous GNN on projected graph."""
    from src.graph_pipeline_bank import build_graph
    from src.homogeneous.builder import project_to_homo
    from src.homogeneous.models import HomoGNN
    from src.training.trainer import Trainer, TrainConfig

    device = get_device()

    hetero_result = build_graph(config, prep=prep)
    data = project_to_homo(hetero_result["data"]).to(device)

    model = HomoGNN(
        data,
        conv_type=model_name,
        hidden_dim=kwargs.get("hidden_dim", 64),
        num_layers=kwargs.get("num_layers", 2),
        num_heads=kwargs.get("num_heads", 4),
        dropout=kwargs.get("dropout", 0.3),
    )

    train_config = TrainConfig(
        task="edge",
        graph_type="homo",
        epochs=kwargs.get("epochs", 200),
        lr=kwargs.get("lr", 1e-3),
        patience=kwargs.get("patience", 15),
    )

    trainer = Trainer(model, data, train_config, device)
    return trainer.run()


def run_l2(prep, config, model_name="hgt", **kwargs):
    """L2: Heterogeneous GNN."""
    from src.graph_pipeline_bank import build_graph
    from src.training.trainer import Trainer, TrainConfig

    device = get_device()
    target_node_type = "internal_account"

    result = build_graph(config, prep=prep)
    data = result["data"]

    if model_name == "hgt":
        from src.heterogeneous.hgt.model import HGT
        model = HGT(
            data,
            hidden_dim=kwargs.get("hidden_dim", 64),
            num_heads=kwargs.get("num_heads", 4),
            num_layers=kwargs.get("num_layers", 2),
            dropout=kwargs.get("dropout", 0.3),
            task="edge",
            target_node_type=target_node_type,
        )
    elif model_name == "hmpnn":
        from src.heterogeneous.hmpnn.model import HMPNN
        model = HMPNN(
            data,
            target_node_type=target_node_type,
            num_layers=kwargs.get("num_layers", 2),
            hidden_dim=kwargs.get("hidden_dim", 64),
            message_dim=kwargs.get("message_dim", 32),
            dropout=kwargs.get("dropout", 0.3),
            task="edge",
        )
    elif model_name == "hetero_gat":
        from src.heterogeneous.hetero_gat.model import HeteroGAT
        model = HeteroGAT(
            data,
            hidden_dim=kwargs.get("hidden_dim", 64),
            num_heads=kwargs.get("num_heads", 4),
            num_layers=kwargs.get("num_layers", 2),
            dropout=kwargs.get("dropout", 0.3),
            task="edge",
            target_node_type=target_node_type,
        )
    else:
        raise ValueError(f"Unknown L2 model: {model_name!r}. Choose hgt | hmpnn | hetero_gat")

    train_config = TrainConfig(
        task="edge",
        graph_type="hetero",
        target_node_type=target_node_type,
        epochs=kwargs.get("epochs", 200),
        lr=kwargs.get("lr", 1e-3),
        patience=kwargs.get("patience", 15),
    )

    trainer = Trainer(model, data, train_config, device)
    return trainer.run()


# ── Results persistence ───────────────────────────────────────────────────────

def _results_dir(level, **kwargs):
    if level == 0:
        return PROJECT_ROOT / "src" / "baselines" / "tabular" / "results"
    elif level == 1:
        model = kwargs.get("model", "sage")
        return PROJECT_ROOT / "src" / "homogeneous" / model / "results"
    elif level == 2:
        model = kwargs.get("model", "hgt")
        return PROJECT_ROOT / "src" / "heterogeneous" / model / "results"
    return PROJECT_ROOT / "results"


def _run_name(level, **kwargs):
    parts = []
    if kwargs.get("model"):
        parts.append(kwargs["model"])
    if kwargs.get("variant"):
        parts.append(kwargs["variant"])
    return "_".join(parts) if parts else f"L{level}"


def save_results(metrics, level, **kwargs):
    name      = _run_name(level, **kwargs)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_dir  = _results_dir(level, **kwargs)
    run_dir   = base_dir / f"{name}_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)

    meta = {"level": level, "timestamp": timestamp, **kwargs}

    serializable = {}
    for k, v in metrics.items():
        serializable[k] = v.tolist() if hasattr(v, "tolist") else v

    with open(run_dir / "metrics.json", "w") as f:
        json.dump({"meta": meta, "metrics": serializable}, f, indent=2)

    cm = serializable.get("confusion_matrix")
    cm_str = ""
    if cm:
        cm_str = (
            f"\n## Confusion Matrix\n\n"
            f"|  | Pred 0 | Pred 1 |\n|---|---|---|\n"
            f"| **Actual 0** | {cm[0][0]} | {cm[0][1]} |\n"
            f"| **Actual 1** | {cm[1][0]} | {cm[1][1]} |\n"
        )

    md = f"# {name}\n\n**Date:** {timestamp}  \n**Level:** {level}  \n"
    for k, v in kwargs.items():
        if v is not None:
            md += f"**{k.title()}:** {v}  \n"
    md += (
        f"\n## Metrics\n\n"
        f"| Metric | Value |\n|---|---|\n"
        f"| AUROC | {serializable.get('auroc', 'N/A'):.4f} |\n"
        f"| AUPRC | {serializable.get('auprc', 'N/A'):.4f} |\n"
        f"| F1    | {serializable.get('f1',    'N/A'):.4f} |\n"
        f"| Precision | {serializable.get('precision', 'N/A'):.4f} |\n"
        f"| Recall    | {serializable.get('recall',    'N/A'):.4f} |\n"
        f"{cm_str}"
    )

    with open(run_dir / "report.md", "w") as f:
        f.write(md)

    print(f"\nResults saved to: {run_dir.relative_to(PROJECT_ROOT)}/")


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Run fraud detection experiments")
    parser.add_argument("--level",   type=int, choices=[0, 1, 2],
                        help="Experiment level (0=tabular, 1=homo GNN, 2=hetero GNN)")
    parser.add_argument("--all",     action="store_true", help="Run L0→L1→L2 sequentially")
    parser.add_argument("--model",   type=str, default="sage",
                        choices=["gcn", "sage", "gat", "hgt", "hmpnn", "hetero_gat"],
                        help="Model (L1: gcn|sage|gat  L2: hgt|hmpnn|hetero_gat)")
    parser.add_argument("--variant", type=str, default="v1",
                        help="Graph variant from master.yaml (v1 | v2)")
    parser.add_argument("--config",  type=str, default=None,
                        help="Override: load a named config file instead of a variant")
    parser.add_argument("--sample",  type=float, default=None,
                        help="Fraction of data to use (e.g. 0.05 for dev runs)")
    parser.add_argument("--data-path", type=str, default=None,
                        help="Override dataset path")

    # L0 Bayesian optimisation
    parser.add_argument("--tune",     action="store_true",
                        help="Use Bayesian optimisation for XGBoost (L0 only)")
    parser.add_argument("--n-trials", type=int, default=50,
                        help="Number of Bayesian optimisation trials (L0 --tune only)")

    # Hyperparameters
    parser.add_argument("--hidden-dim",  type=int,   default=64)
    parser.add_argument("--num-layers",  type=int,   default=2)
    parser.add_argument("--num-heads",   type=int,   default=4)
    parser.add_argument("--epochs",      type=int,   default=200)
    parser.add_argument("--lr",          type=float, default=1e-3)
    parser.add_argument("--patience",    type=int,   default=15)

    args = parser.parse_args()

    # ── Config ────────────────────────────────────────────────────────────────
    if args.config:
        config = load_config(args.config)
    else:
        config = load_variant(args.variant)

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

    model_kwargs = {
        "hidden_dim": args.hidden_dim,
        "num_layers": args.num_layers,
        "num_heads":  args.num_heads,
        "epochs":     args.epochs,
        "lr":         args.lr,
        "patience":   args.patience,
    }

    prep = prepare_data(config)

    # ── Run all levels ────────────────────────────────────────────────────────
    if args.all:
        print("\n" + "=" * 70)
        print("RUNNING ALL EXPERIMENT LEVELS")
        print("=" * 70)
        for level, model in [(0, None), (1, "sage"), (2, "hgt")]:
            print(f"\n\n{'#'*70}\n# LEVEL {level}\n{'#'*70}")
            if level == 0:
                results = run_l0(prep, tune=args.tune, n_trials=args.n_trials)
            elif level == 1:
                results = run_l1(prep, config, model_name=model, **model_kwargs)
            else:
                results = run_l2(prep, config, model_name=model, **model_kwargs)
            save_kwargs = {"model": model, "variant": args.variant if level > 0 else None}
            if isinstance(results, list):
                for r in results:
                    if r: save_results(r, level, **save_kwargs)
            elif results:
                save_results(results, level, **save_kwargs)
        return

    if args.level is None:
        parser.print_help()
        return

    # ── Single level ──────────────────────────────────────────────────────────
    if args.level == 0:
        results = run_l0(prep, tune=args.tune, n_trials=args.n_trials)
    elif args.level == 1:
        results = run_l1(prep, config, model_name=args.model, **model_kwargs)
    elif args.level == 2:
        results = run_l2(prep, config, model_name=args.model, **model_kwargs)

    save_kwargs = {
        "model":   args.model if args.level > 0 else None,
        "variant": args.variant if args.level > 0 else None,
    }
    if isinstance(results, list):
        for r in results:
            if r: save_results(r, args.level, **save_kwargs)
    elif results:
        save_results(results, args.level, **save_kwargs)


if __name__ == "__main__":
    main()
