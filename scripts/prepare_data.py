"""
Generate cached graph files from raw datasets.

Usage:
    python scripts/prepare_data.py              # SAML-D homogeneous
    python scripts/prepare_data.py --hetero     # SAML-D heterogeneous
    python scripts/prepare_data.py --all        # Both
"""

import argparse

from src.utils.compat import apply_pyg_compat_patch
apply_pyg_compat_patch()

from src.utils.config import PROJECT_ROOT


def prepare_homo(sample_ratio=1.0):
    from src.data.saml_homo import load_and_prepare_saml_data
    print("Preparing SAML-D homogeneous graph...")
    data, mapping = load_and_prepare_saml_data(sample_ratio=sample_ratio, use_cache=False)
    print(f"Done: {data.num_nodes} nodes, {data.num_edges} edges")


def prepare_hetero(sample_ratio=1.0):
    from src.data.saml_hetero import load_hetero_saml_data
    print("Preparing SAML-D heterogeneous graph...")
    data, mapping = load_hetero_saml_data(sample_ratio=sample_ratio, use_cache=False)
    print(f"Done: {data.num_nodes} nodes, edge types: {len(data.edge_types)}")


def main():
    parser = argparse.ArgumentParser(description="Prepare graph data caches")
    parser.add_argument("--hetero", action="store_true", help="Prepare heterogeneous graph")
    parser.add_argument("--all", action="store_true", help="Prepare both homo and hetero graphs")
    parser.add_argument("--sample", type=float, default=1.0, help="Sample ratio (default: 1.0)")
    args = parser.parse_args()

    if args.all:
        prepare_homo(args.sample)
        prepare_hetero(args.sample)
    elif args.hetero:
        prepare_hetero(args.sample)
    else:
        prepare_homo(args.sample)


if __name__ == "__main__":
    main()
