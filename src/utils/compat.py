"""
PyG API compatibility patches for version mismatches.
"""


def apply_pyg_compat_patch():
    """
    Apply any necessary monkey-patches for PyG API changes.

    Called once at the start of training scripts to smooth over
    breaking changes between PyG versions.
    """
    try:
        import torch_geometric
        # Add any version-specific patches here as needed.
        # Example: some PyG versions changed HeteroData attribute access.
        _ = torch_geometric.__version__
    except ImportError:
        pass
