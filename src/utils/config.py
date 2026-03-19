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

    # Build the config the pipeline expects
    config = {
        "variant": variant,
        "data_path": master["data_path"],
        "sample_ratio": master["sample_ratio"],
        "truncate_after": master.get("truncate_after"),
        "split": deepcopy(master["split"]),
        "cache": deepcopy(master["cache"]),
        "columns": deepcopy(master["columns"]),
    }

    # Nodes: pick from shared node_features based on variant's node list
    config["nodes"] = {}
    for node_type in vdef["nodes"]:
        if node_type == "transaction":
            config["nodes"]["transaction"] = {
                "features": vdef.get("transaction_features", master["edge_features"]),
            }
        elif node_type in master["node_features"]:
            config["nodes"][node_type] = deepcopy(master["node_features"][node_type])

    # Edges
    config["edges"] = {
        "features": deepcopy(master["edge_features"]),
        "relations": deepcopy(vdef["edges"]),
    }

    return config
