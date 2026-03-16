"""
Node mapping construction for the bank heterogeneous graph.

Builds integer-index mappings for each node type declared in the config:

  internal_account  — unique ACCOUNTID values (sender pool).
                      On-us receivers are promoted into this pool too,
                      since TRANSACTIONONUS=True means both ends are
                      internal accounts.

  external_account  — COUNTERENTITYID values that NEVER appear as senders.
                      (exclude_if_sender: true in config)

  device            — unique DEVICEID values (V3 only).

Each mapping is a plain dict {raw_id_str: int_index}.
"""

import pandas as pd


NodeMaps = dict[str, dict[str, int]]


def build_node_maps(df: pd.DataFrame, config: dict) -> NodeMaps:
    """
    Build {node_type: {raw_id: int_index}} for every node type in config.

    Args:
        df:      cleaned DataFrame from loader (has _sender, _receiver columns)
        config:  full pipeline config dict

    Returns:
        NodeMaps dict, e.g. {
            "internal_account": {"ACC001": 0, "ACC002": 1, ...},
            "external_account": {"EXT001": 0, ...},
            "device":           {"DEV001": 0, ...},   # V3 only
        }
    """
    node_cfg = config["nodes"]
    col_cfg  = config["columns"]
    maps: NodeMaps = {}

    # ── InternalAccount pool ────────────────────────────────────────────────
    # All senders are internal by definition.
    # On-us receivers (TRANSACTIONONUS=True) are also internal.
    if "internal_account" in node_cfg:
        sender_ids = set(df["_sender"].unique())

        onus_col = col_cfg.get("onus_flag")
        if onus_col and onus_col in df.columns:
            onus_receivers = set(df.loc[df[onus_col], "_receiver"].unique())
        else:
            onus_receivers = set()

        all_internal = sender_ids | onus_receivers
        maps["internal_account"] = {acc: i for i, acc in enumerate(sorted(all_internal))}

        n_promoted = len(onus_receivers - sender_ids)
        print(f"  internal_account: {len(maps['internal_account']):,} nodes"
              f"  ({n_promoted:,} on-us receivers promoted)")

    # ── ExternalAccount pool ────────────────────────────────────────────────
    if "external_account" in node_cfg:
        cfg = node_cfg["external_account"]
        all_receivers = set(df["_receiver"].unique())

        if cfg.get("exclude_if_sender", True):
            # Only receivers that NEVER appear as senders
            internal_ids = set(maps.get("internal_account", {}).keys())
            external_ids = all_receivers - internal_ids
        else:
            external_ids = all_receivers

        maps["external_account"] = {acc: i for i, acc in enumerate(sorted(external_ids))}
        print(f"  external_account: {len(maps['external_account']):,} nodes")

    # ── Device pool (V3) ────────────────────────────────────────────────────
    if "device" in node_cfg:
        dev_col = col_cfg.get("device")
        if dev_col and dev_col in df.columns:
            device_ids = df[dev_col].astype(str).unique()
            maps["device"] = {d: i for i, d in enumerate(sorted(device_ids))}
            print(f"  device:           {len(maps['device']):,} nodes")
        else:
            print(f"  Warning: device node type requested but column '{dev_col}' not found")

    return maps
