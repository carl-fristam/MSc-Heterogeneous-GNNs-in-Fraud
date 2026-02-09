"""Entry point for GraphSAGE training."""

from src.utils.compat import apply_pyg_compat_patch
apply_pyg_compat_patch()

from src.baselines.graphsage.main import main

if __name__ == "__main__":
    main()
