"""
Config loading and project root utilities.
"""

from pathlib import Path
import yaml

# Project root = directory containing src/, configs/, datasets/
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


def load_config(name: str) -> dict:
    """
    Load a YAML config by name from configs/{name}.yaml.

    Args:
        name: config filename without extension, e.g. "graph_bank_v1"

    Returns:
        dict of config values
    """
    path = PROJECT_ROOT / "configs" / f"{name}.yaml"
    if not path.exists():
        raise FileNotFoundError(f"Config not found: {path}")
    with open(path) as f:
        return yaml.safe_load(f)
