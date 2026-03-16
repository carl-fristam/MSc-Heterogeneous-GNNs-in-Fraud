"""
Build bank payment graphs.

Usage:
    # Full build — both pipelines
    python scripts/build_graphs.py

    # Single pipeline
    python scripts/build_graphs.py --pipeline edge
    python scripts/build_graphs.py --pipeline txn

    # Dev mode (1% sample, no cache)
    python scripts/build_graphs.py --dev
    python scripts/build_graphs.py --pipeline txn --dev
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.utils.config import load_config
from src.graph_pipeline_bank import build_graph as build_edge
from src.graph_pipeline_bank_txn import build_graph as build_txn


def run(pipeline: str, dev: bool):
    if pipeline == "edge":
        cfg = load_config("graph_bank_v1")
        builder = build_edge
    else:
        cfg = load_config("graph_bank_txn_v1")
        builder = build_txn

    if dev:
        cfg["sample_ratio"] = 0.01
        cfg["cache"]["enabled"] = False

    result = builder(cfg)
    print(result["data"])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pipeline", choices=["edge", "txn", "both"], default="both")
    parser.add_argument("--dev", action="store_true", help="1%% sample, no cache")
    args = parser.parse_args()

    pipelines = ["edge", "txn"] if args.pipeline == "both" else [args.pipeline]
    for p in pipelines:
        print(f"\n{'='*60}")
        print(f"Pipeline: {p}")
        print(f"{'='*60}")
        run(p, args.dev)


if __name__ == "__main__":
    main()
