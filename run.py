"""
Main experiment runner.

Usage:
    # L0: Tabular baselines
    python run.py --level 0

    # L1: Graph features → XGBoost
    python run.py --level 1

    # L2: Homogeneous GNN (node or edge mode)
    python run.py --level 2 --task node --conv sage
    python run.py --level 2 --task edge --conv gcn

    # L3: Heterogeneous GNN
    python run.py --level 3 --task node --model hgt --variant txn_v1
    python run.py --level 3 --task edge --model hgt --variant v1
    python run.py --level 3 --task edge --model hmpnn --variant v2

    # Dev mode (1% sample)
    python run.py --level 0 --sample 0.01

    # Run all levels sequentially
    python run.py --all
"""

import argparse
import json
from datetime import datetime
from pathlib import Path

from src.utils.config import load_config, load_variant, PROJECT_ROOT
from src.utils.device import get_device
from src.data.prepare import prepare_data


def run_l0(prep):
    """L0: Tabular baselines (no graph)."""
    from src.baselines.tabular import run_tabular_baselines
    return run_tabular_baselines(prep)


def run_l1(prep):
    """L1: Graph features → XGBoost."""
    from src.baselines.graph_features import run_graph_feature_baselines
    return run_graph_feature_baselines(prep)


def run_l2(prep, config, task="node", conv_type="sage", **kwargs):
    """L2: Homogeneous GNN (or KGE models)."""
    from src.homogeneous.builder import build_homogeneous_graph
    from src.training.trainer import Trainer, TrainConfig

    result = build_homogeneous_graph(prep, mode=task, config=config)
    data = result["data"]
    device = get_device()

    edge_feat_dim = data.edge_attr.shape[1] if hasattr(data, "edge_attr") and data.edge_attr is not None else 0

    if conv_type in ("transe", "distmult"):
        from src.homogeneous.kge_models import TransE, DistMult
        if task != "edge":
            raise ValueError(f"{conv_type} only supports edge classification")
        ModelClass = TransE if conv_type == "transe" else DistMult
        model = ModelClass(
            num_nodes=data.num_nodes,
            embedding_dim=kwargs.get("hidden_dim", 64),
            edge_feat_dim=edge_feat_dim,
            dropout=kwargs.get("dropout", 0.3),
        )
    else:
        from src.homogeneous.models import HomoGNN
        model = HomoGNN(
            in_dim=data.x.shape[1],
            hidden_dim=kwargs.get("hidden_dim", 64),
            num_layers=kwargs.get("num_layers", 2),
            dropout=kwargs.get("dropout", 0.3),
            conv_type=conv_type,
            task=task,
            edge_feat_dim=edge_feat_dim,
        )

    train_config = TrainConfig(
        task=task,
        graph_type="homo",
        epochs=kwargs.get("epochs", 200),
        lr=kwargs.get("lr", 1e-3),
        patience=kwargs.get("patience", 15),
    )

    trainer = Trainer(model, data, train_config, device)
    return trainer.run()


def run_l4(prep, config, task="node", model_name="hgmae", **kwargs):
    """L4: Self-supervised pretraining → downstream classifier."""
    from src.self_supervised.pretrain_trainer import PretrainTrainer

    device = get_device()

    if task == "node":
        from src.graph_pipeline_bank_txn import build_graph
        result = build_graph(config, prep=prep)
        target_node_type = "transaction"
    else:
        from src.graph_pipeline_bank import build_graph
        result = build_graph(config, prep=prep)
        target_node_type = "internal_account"

    data = result["data"]
    hidden_dim = kwargs.get("hidden_dim", 64)

    if model_name == "hgmae":
        from src.self_supervised.hgmae.model import HGMAE
        l4_cfg = config.get("l4", {}).get("hgmae", {})
        ssl_model = HGMAE(
            data,
            hidden_dim=hidden_dim,
            num_heads=kwargs.get("num_heads", 4),
            num_layers=kwargs.get("num_layers", 2),
            dropout=kwargs.get("dropout", 0.3),
            feat_mask_rate=l4_cfg.get("feat_mask_rate", 0.5),
            edge_mask_rate=l4_cfg.get("edge_mask_rate", 0.3),
            edge_recon_weight=l4_cfg.get("edge_recon_weight", 1.0),
        )
    elif model_name == "laundrograph":
        from src.self_supervised.laundrograph.model import LaundroGraph
        ssl_model = LaundroGraph(
            data,
            hidden_dim=hidden_dim,
            num_layers=kwargs.get("num_layers", 2),
            dropout=kwargs.get("dropout", 0.3),
        )
    else:
        raise ValueError(f"Unknown L4 model: {model_name}")

    pretrain_epochs = kwargs.get("pretrain_epochs", 200)
    freeze = kwargs.get("freeze", True)

    trainer = PretrainTrainer(
        ssl_model, data, device,
        task=task,
        target_node_type=target_node_type,
        hidden_dim=hidden_dim,
        pretrain_epochs=pretrain_epochs,
        pretrain_lr=kwargs.get("lr", 1e-3),
        pretrain_patience=kwargs.get("patience", 20),
        classify_epochs=kwargs.get("epochs", 200),
        classify_lr=kwargs.get("lr", 1e-3),
        classify_patience=kwargs.get("patience", 15),
        freeze=freeze,
        dropout=kwargs.get("dropout", 0.3),
    )
    return trainer.run()


def run_l3(prep, config, task="node", model_name="hgt", **kwargs):
    """L3: Heterogeneous GNN."""
    from src.training.trainer import Trainer, TrainConfig

    device = get_device()

    if task == "node":
        from src.graph_pipeline_bank_txn import build_graph
        result = build_graph(config, prep=prep)
        target_node_type = "transaction"
    else:
        from src.graph_pipeline_bank import build_graph
        result = build_graph(config, prep=prep)
        target_node_type = "internal_account"

    data = result["data"]

    if model_name == "hgt":
        from src.heterogeneous.hgt.model import HGT
        model = HGT(
            data,
            hidden_dim=kwargs.get("hidden_dim", 64),
            num_heads=kwargs.get("num_heads", 4),
            num_layers=kwargs.get("num_layers", 2),
            dropout=kwargs.get("dropout", 0.3),
            task=task,
            target_node_type=target_node_type,
        )
    elif model_name == "hmpnn":
        from src.heterogeneous.hmpnn.model import HMPNN
        model = HMPNN(
            data,
            target_node_type=target_node_type,
            num_layers=kwargs.get("num_layers", 2),
            hidden_dim=kwargs.get("hidden_dim", 16),
            message_dim=kwargs.get("message_dim", 8),
            task=task,
        )
    else:
        raise ValueError(f"Unknown model: {model_name}")

    train_config = TrainConfig(
        task=task,
        graph_type="hetero",
        target_node_type=target_node_type,
        epochs=kwargs.get("epochs", 200),
        lr=kwargs.get("lr", 1e-3),
        patience=kwargs.get("patience", 15),
    )

    trainer = Trainer(model, data, train_config, device)
    return trainer.run()


def _results_dir(level, **kwargs):
    """Map experiment level to a results subfolder colocated with the code."""
    if level == 0:
        return PROJECT_ROOT / "src" / "baselines" / "tabular" / "results"
    elif level == 1:
        return PROJECT_ROOT / "src" / "baselines" / "graph_features" / "results"
    elif level == 2:
        conv = kwargs.get("conv", "sage")
        return PROJECT_ROOT / "src" / "homogeneous" / conv / "results"
    elif level == 3:
        model = kwargs.get("model", "hgt")
        return PROJECT_ROOT / "src" / "heterogeneous" / model / "results"
    elif level == 4:
        model = kwargs.get("model", "hgmae")
        return PROJECT_ROOT / "src" / "self_supervised" / model / "results"
    return PROJECT_ROOT / "results"


def _run_name(level, **kwargs):
    """Build a descriptive run name like hgt_edge_v1."""
    parts = []
    if kwargs.get("task"):
        parts.append(kwargs["task"])
    if kwargs.get("variant"):
        parts.append(kwargs["variant"])
    return "_".join(parts) if parts else f"L{level}"


def save_results(metrics, level, **kwargs):
    """Save experiment results as JSON and a human-readable markdown report."""
    name = _run_name(level, **kwargs)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    base_dir = _results_dir(level, **kwargs)
    run_dir = base_dir / f"{name}_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)

    meta = {"level": level, "timestamp": timestamp, **kwargs}

    serializable = {}
    for k, v in metrics.items():
        if hasattr(v, "tolist"):
            serializable[k] = v.tolist()
        else:
            serializable[k] = v

    # JSON
    result = {"meta": meta, "metrics": serializable}
    with open(run_dir / "metrics.json", "w") as f:
        json.dump(result, f, indent=2)

    # Markdown report
    cm = serializable.get("confusion_matrix")
    cm_str = ""
    if cm:
        cm_str = (
            f"\n## Confusion Matrix\n\n"
            f"|  | Pred 0 | Pred 1 |\n|---|---|---|\n"
            f"| **Actual 0** | {cm[0][0]} | {cm[0][1]} |\n"
            f"| **Actual 1** | {cm[1][0]} | {cm[1][1]} |\n"
        )

    md = (
        f"# {name}\n\n"
        f"**Date:** {timestamp}  \n"
        f"**Level:** {level}  \n"
    )
    for k, v in kwargs.items():
        if v is not None:
            md += f"**{k.title()}:** {v}  \n"

    md += (
        f"\n## Metrics\n\n"
        f"| Metric | Value |\n|---|---|\n"
        f"| AUROC | {serializable.get('auroc', 'N/A'):.4f} |\n"
        f"| AUPRC | {serializable.get('auprc', 'N/A'):.4f} |\n"
        f"| F1 | {serializable.get('f1', 'N/A'):.4f} |\n"
        f"| Precision | {serializable.get('precision', 'N/A'):.4f} |\n"
        f"| Recall | {serializable.get('recall', 'N/A'):.4f} |\n"
        f"{cm_str}"
    )

    with open(run_dir / "report.md", "w") as f:
        f.write(md)

    print(f"\nResults saved to: {run_dir.relative_to(PROJECT_ROOT)}/")


def main():
    parser = argparse.ArgumentParser(description="Run fraud detection experiments")
    parser.add_argument("--level", type=int, choices=[0, 1, 2, 3, 4], help="Experiment level")
    parser.add_argument("--all", action="store_true", help="Run all levels")
    parser.add_argument("--task", type=str, default="node", choices=["node", "edge"])
    parser.add_argument("--model", type=str, default="hgt", choices=["hgt", "hmpnn", "hgmae", "laundrograph"])
    parser.add_argument("--conv", type=str, default="sage", choices=["gcn", "sage", "transe", "distmult"])
    parser.add_argument("--variant", type=str, default="v1", help="Config variant (v1, v2, v3, txn_v1)")
    parser.add_argument("--config", type=str, default=None, help="Config name override")
    parser.add_argument("--sample", type=float, default=None, help="Override sample_ratio")
    parser.add_argument("--data-path", type=str, default=None, help="Override dataset path")

    # Model hyperparameters
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--pretrain-epochs", type=int, default=200)
    parser.add_argument("--freeze", action="store_true", default=True,
                        help="Freeze encoder in stage 2 (linear probe)")
    parser.add_argument("--no-freeze", dest="freeze", action="store_false",
                        help="Fine-tune encoder in stage 2")

    args = parser.parse_args()

    if args.config:
        config = load_config(args.config)
    else:
        config = load_variant(args.variant)

    datasets_dir = PROJECT_ROOT / "datasets"
    if args.data_path is not None:
        config["data_path"] = args.data_path
    else:
        print(f"\nAvailable datasets:")
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
        "epochs": args.epochs,
        "lr": args.lr,
        "patience": args.patience,
        "pretrain_epochs": args.pretrain_epochs,
        "freeze": args.freeze,
    }

    prep = prepare_data(config)

    if args.all:
        print("\n" + "="*70)
        print("RUNNING ALL EXPERIMENT LEVELS")
        print("="*70)

        for level in [0, 1, 2, 3]:
            print(f"\n\n{'#'*70}")
            print(f"# LEVEL {level}")
            print(f"{'#'*70}")

            if level == 0:
                results = run_l0(prep)
            elif level == 1:
                results = run_l1(prep)
            elif level == 2:
                results = run_l2(prep, config, task=args.task, conv_type=args.conv, **model_kwargs)
            else:
                results = run_l3(prep, config, task=args.task, model_name=args.model, **model_kwargs)
        return

    if args.level is None:
        parser.print_help()
        return

    if args.level == 0:
        results = run_l0(prep)
    elif args.level == 1:
        results = run_l1(prep)
    elif args.level == 2:
        results = run_l2(prep, config, task=args.task, conv_type=args.conv, **model_kwargs)
    elif args.level == 3:
        results = run_l3(prep, config, task=args.task, model_name=args.model, **model_kwargs)
    elif args.level == 4:
        results = run_l4(prep, config, task=args.task, model_name=args.model, **model_kwargs)

    # Save
    save_kwargs = {
        "task": args.task,
        "model": args.model if args.level in (3, 4) else None,
        "conv": args.conv if args.level == 2 else None,
        "variant": args.variant if args.level in (3, 4) else None,
    }
    if isinstance(results, list):
        for r in results:
            if r:
                save_results(r, args.level, **save_kwargs)
    elif isinstance(results, dict):
        save_results(results, args.level, **save_kwargs)


if __name__ == "__main__":
    main()
