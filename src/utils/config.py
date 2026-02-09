"""
Project configuration utilities.
"""

import os
from pathlib import Path

import yaml

# Root of the project (directory containing this src/ package)
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


def load_config(name: str) -> dict:
    """
    Load a YAML experiment config by name.

    Args:
        name: Config name without extension (e.g. 'gcn', 'graphsage')

    Returns:
        Dictionary of config values
    """
    config_path = PROJECT_ROOT / "configs" / f"{name}.yaml"
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")
    with open(config_path) as f:
        return yaml.safe_load(f)
