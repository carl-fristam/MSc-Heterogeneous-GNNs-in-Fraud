"""
Figures out which accounts are internal vs external, and assigns each
account a unique integer index. This integer index is what the GNN uses
internally — it never sees raw account ID strings.

Two node types:
    internal_account — Danske Bank accounts. These are always the senders.
                       Receivers of on-us transactions are also internal,
                       because both sides of an on-us transfer are Danske accounts.

    external_account — Counterparty accounts at other banks. These only
                       appear as receivers and never as senders in our data.

Output: a dict of dicts, e.g.
    {
        "internal_account": {"ACC001": 0, "ACC002": 1, ...},
        "external_account": {"EXT999": 0, "EXT888": 1, ...}
    }
"""

import pandas as pd


def build_node_maps(df: pd.DataFrame, col_cfg: dict) -> dict:
    """
    Assign every account in the dataset to a node type and give it an integer index.

    Args:
        df:       the full transaction dataframe (output of loader.py)
                  must have _sender and _receiver columns
        col_cfg:  the columns section from master.yaml

    Returns:
        node_maps: dict with keys "internal_account" and "external_account",
                   each mapping raw account ID string -> integer index
    """

    onus_col = col_cfg["onus_flag"]  # TRANSACTIONONUS

    # --- Internal accounts ---
    # Every sender is by definition an internal (Danske Bank) account
    sender_ids = set(df["_sender"].unique())

    # On-us transactions connect two internal accounts.
    # The receiver in those transactions is also internal, but may never
    # appear as a sender in the data (e.g. a dormant account that only receives).
    # So we promote those receivers into the internal pool too.
    onus_receiver_ids = set(df.loc[df[onus_col] == True, "_receiver"].unique())

    internal_ids = sender_ids | onus_receiver_ids

    # --- External accounts ---
    # Any receiver that is NOT in the internal pool is external
    all_receiver_ids = set(df["_receiver"].unique())
    external_ids = all_receiver_ids - internal_ids

    # --- Build integer index mappings ---
    # We sort the IDs before enumerating so the mapping is deterministic
    # (same data always produces the same index for the same account)
    internal_map = {acc_id: idx for idx, acc_id in enumerate(sorted(internal_ids))}
    external_map = {acc_id: idx for idx, acc_id in enumerate(sorted(external_ids))}

    # Print a summary so we can sanity check
    promoted = len(onus_receiver_ids - sender_ids)
    print(f"Node mapping built:")
    print(f"  internal_account: {len(internal_map):,} nodes  ({promoted:,} added from on-us receivers)")
    print(f"  external_account: {len(external_map):,} nodes")

    return {
        "internal_account": internal_map,
        "external_account": external_map,
    }
