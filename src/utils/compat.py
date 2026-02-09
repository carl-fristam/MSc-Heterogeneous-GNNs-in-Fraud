"""
Python 3.14 compatibility patch for PyTorch Geometric.

PyG's inspector module accesses typing._name which was removed in Python 3.14.
Call apply_pyg_compat_patch() before importing any PyG conv layers.
"""

import sys

_patched = False


def apply_pyg_compat_patch():
    """Apply the PyG typing compatibility patch (idempotent)."""
    global _patched
    if _patched:
        return

    if sys.version_info < (3, 14):
        _patched = True
        return

    import typing

    import torch_geometric.inspector

    _orig_type_repr = torch_geometric.inspector.type_repr

    def _safe_type_repr(obj, *args, **kwargs):
        if obj is typing.Union:
            return 'Union'
        if getattr(obj, '__module__', '') == 'typing':
            try:
                str(obj._name)
            except AttributeError:
                if hasattr(obj, '__origin__'):
                    return str(obj.__origin__).split('.')[-1]
                return str(obj).replace('typing.', '')
        try:
            return _orig_type_repr(obj, *args, **kwargs)
        except AttributeError:
            if getattr(obj, '__module__', '') == 'typing':
                return str(obj).replace('typing.', '')
            raise

    torch_geometric.inspector.type_repr = _safe_type_repr
    _patched = True
