"""
Config loading and project root utilities.
"""

from copy import deepcopy
from pathlib import Path
import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


def load_config(name: str) -> dict:
    """
    Load a YAML config by name from configs/{name}.yaml.
    Falls back to legacy per-variant files if master.yaml doesn't have the variant.
    """
    path = PROJECT_ROOT / "configs" / f"{name}.yaml"
    if not path.exists():
        raise FileNotFoundError(f"Config not found: {path}")
    with open(path) as f:
        return yaml.safe_load(f)


def load_variant(variant: str) -> dict:
    """
    Load a variant from configs/master.yaml.

    Merges shared settings (data, columns, split, features) with the
    variant-specific graph topology. Returns a config dict in the same
    format the pipeline expects.
    """
    master_path = PROJECT_ROOT / "configs" / "master.yaml"
    with open(master_path) as f:
        master = yaml.safe_load(f)

    if variant not in master["variants"]:
        raise ValueError(f"Unknown variant '{variant}'. Available: {list(master['variants'].keys())}")

    vdef = master["variants"][variant]

    config = {
        "variant": variant,
        "split": deepcopy(master["split"]),
        "cache": deepcopy(master["cache"]),
        "columns": deepcopy(master["columns"]),
        "edge_features": deepcopy(master["edge_features_lean"]),
        "edge_features_full": deepcopy(master["edge_features_full"]),
        "node_features": deepcopy(master["node_features"]),
    }

    config["edges"] = {
        "relations": deepcopy(vdef["edges"]),
    }

    return config
