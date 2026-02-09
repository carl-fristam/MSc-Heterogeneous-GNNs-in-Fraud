"""Entry point for HMPNN training."""

from src.utils.compat import apply_pyg_compat_patch
apply_pyg_compat_patch()

from src.hmpnn.model import HMPNN
from src.data.saml_hetero import load_hetero_saml_data
from src.utils.device import get_device
from src.utils.config import PROJECT_ROOT, load_config


def main():
    cfg = load_config("hmpnn")

    print("Loading heterogeneous graph data...")
    data, account_to_id = load_hetero_saml_data(
        sample_ratio=cfg["data"]["sample_ratio"],
        cache_path=str(PROJECT_ROOT / cfg["data"]["cache_path"]),
    )

    device = get_device()
    print(f"Using device: {device}")

    model = HMPNN(
        data,
        target_node_type=cfg["model"]["target_node_type"],
        num_layers=cfg["model"]["num_layers"],
        hidden_dim=cfg["model"]["hidden_dim"],
        message_dim=cfg["model"]["message_dim"],
    )
    print(f"Model parameters: {sum(p.numel() for p in model.parameters())}")

    # Forward pass test
    x_dict = {nt: data[nt].x for nt in data.node_types}
    edge_index_dict = {et: data[et].edge_index for et in data.edge_types}
    edge_attr_dict = {et: data[et].edge_attr for et in data.edge_types}

    out = model(x_dict, edge_index_dict, edge_attr_dict)
    print(f"Output shape: {out.shape}")
    print(f"Output range: [{out.min():.4f}, {out.max():.4f}]")


if __name__ == "__main__":
    main()
