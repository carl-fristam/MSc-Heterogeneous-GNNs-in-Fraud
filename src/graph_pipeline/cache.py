"""
Thin pickle cache wrapper for processed graphs.
"""

import os
import pickle


def load_cache(path: str):
    """Load cached data from a pickle file, or return None if not found."""
    if os.path.exists(path):
        print(f"Loading from cache: {path}")
        with open(path, "rb") as f:
            return pickle.load(f)
    return None


def save_cache(path: str, data: dict):
    """Save data to a pickle file."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(data, f)
    print(f"Saved to cache: {path}")
