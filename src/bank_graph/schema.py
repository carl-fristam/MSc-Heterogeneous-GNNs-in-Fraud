"""
Graph schema definition for bank transaction data.

Source tables:
  - bj.fraud_data_ext  — Retail: ALL transactions (fraud + non-fraud) — main transaction table
  - bj.fraud_data_ret  — Retail: FRAUD ONLY — serves as labels for ext
  - bj.fraud_data_bus  — Business payment transactions (label source TBC)

Maps these relational tables to a PyG HeteroData heterogeneous graph.
"""

# --------------------------------------------------------------------------- #
# Table column inventory (for reference)
# --------------------------------------------------------------------------- #
#
# bj.fraud_data_ext (retail — ALL transactions, main table):
#   IDs:      ACCOUNTAGENTID, ACCOUNTBRANCHID, ACCOUNTENTITYID, ACCOUNTID,
#             COUNTERAGENTID, COUNTERBRANCHID, COUNTERENTITYID, COUNTERPARTYID,
#             CUSTOMERENTITYID, CUSTOMERID, DEVICEENTITYID, DEVICEID,
#             TRANSACTIONID
#   Features: VALUE, BASEVALUE, CURRENCY, BASECURRENCY, CHANNEL,
#             PAYMENTMETHOD, PAYMENTSUBMETHOD, PAYMENTCLEARING,
#             MSGSTATUS, TRANSACTIONONUS, EXCEPTIONRULE,
#             INTERNATIONALFLAG, DESTINATIONCOUNTRY, ACCAGENTCOUNTRY,
#             CUSTOMERTYPE, IPADDRESS, USERAGENTSTRING
#   Time:     EVENTTIME (x2?), ACCOUNTIDFORMAT, COUNTERIDFORMAT
#
# bj.fraud_data_ret (retail — FRAUD ONLY, used as labels):
#   IDs:      ACCOUNTAGENTID, ACCOUNTBRANCHID, ACCOUNTID,
#             COUNTERAGENTID, COUNTERENTITYID, COUNTERPARTYID,
#             CUSTOMERID, DEVICEID
#   Label:    CONFIRMEDRISK
#   Features: VALUE, BASEVALUE, CURRENCY, BASECURRENCY,
#             INTERNATIONALFLAG, TRANSDIRECTION
#   Time:     EVENTTIME, ORGEVENTTIME
#   Ref:      ORGTRANSID, MSGSTATUS
#
# bj.fraud_data_bus (business transactions):
#   IDs:      ACCOUNTAGENTID, ACCOUNTBRANCHID, ACCOUNTENTITYID, ACCOUNTID,
#             COUNTERAGENTID, COUNTERENTITYID, COUNTERPARTYID, CUSTOMERID
#   Features: VALUE, BASEVALUE, CURRENCY, BASECURRENCY, CHANNEL,
#             MSGSTATUS, INTERNATIONALFLAG, ACCAGENTCOUNTRY
#   Time:     EVENTTIME
#
# --------------------------------------------------------------------------- #

# --------------------------------------------------------------------------- #
# Graph schema — DRAFT v2 (transaction-as-node / bipartite)
# --------------------------------------------------------------------------- #
#
# Design decisions:
#   - Transactions are NODES (not edges). This is the classification target.
#   - ACCOUNTID and COUNTERPARTYID map to the SAME "account" node type.
#   - Graph is bipartite-like: Account ──[sends]──▶ Txn ──[received_by]──▶ Account
#     This lets the GNN learn per-transaction embeddings AND trace chains.
#   - Retail (ext) and business (bus) produce separate transaction node types
#     so the model can handle different feature sets.
#   - Labels come from ret, joined via ORGTRANSID → TRANSACTIONID.
#
# Topology:
#   Account ──[sends]──────────▶ RetailTxn ──[received_by]──▶ Account
#   Account ──[sends_bus]──────▶ BusTxn    ──[received_by_bus]──▶ Account
#
# TODO(human): Review and adjust. Key open questions marked with (?).
# --------------------------------------------------------------------------- #

GRAPH_SCHEMA = {
    # ------------------------------------------------------------------- #
    # Node types
    # ------------------------------------------------------------------- #
    "node_types": {
        # Unified account node: both ACCOUNTID (sender) and COUNTERPARTYID
        # (receiver) map into this single pool. An account that sends and
        # receives will be the SAME node, enabling chain detection.
        "account": {
            "id_columns": {
                "sender": "ACCOUNTID",
                "receiver": "COUNTERPARTYID",
            },
            "features": [
                "ACCAGENTCOUNTRY",
                # (?) ACCOUNTBRANCHID — could be a feature or its own node type
            ],
        },

        # Each row in ext becomes a transaction node. This is the
        # classification target — fraud labels attach here.
        "retail_txn": {
            "source_table": "bj.fraud_data_ext",
            "id_column": "TRANSACTIONID",
            "features": [
                # Monetary
                "VALUE", "BASEVALUE", "CURRENCY", "BASECURRENCY",
                # Transaction metadata
                "CHANNEL", "PAYMENTMETHOD", "PAYMENTSUBMETHOD",
                "PAYMENTCLEARING", "MSGSTATUS",
                "TRANSACTIONONUS", "EXCEPTIONRULE",
                # Geographic / risk
                "INTERNATIONALFLAG", "DESTINATIONCOUNTRY",
                # Time
                "EVENTTIME",
            ],
            # Available but not yet assigned:
            #   IPADDRESS, USERAGENTSTRING — could be txn features or drive Device nodes
            #   CUSTOMERID — used for customer→account edges if customer is a node
        },

        # Each row in bus becomes a business transaction node.
        "bus_txn": {
            "source_table": "bj.fraud_data_bus",
            "id_column": None,              # (?) does bus have a TRANSACTIONID?
            "features": [
                "VALUE", "BASEVALUE", "CURRENCY", "BASECURRENCY",
                "CHANNEL", "MSGSTATUS",
                "INTERNATIONALFLAG", "ACCAGENTCOUNTRY",
                "EVENTTIME",
            ],
        },

        # (?) Optional node types — uncomment if you want them:
        #
        # "customer": {
        #     "id_columns": "CUSTOMERID",
        #     "features": ["CUSTOMERTYPE"],
        #     "note": "Only useful if customer:account is not 1:1",
        # },
        #
        # "device": {
        #     "id_columns": "DEVICEID",
        #     "features": [],
        #     "note": "Creates links between txns sharing a device",
        # },
    },

    # ------------------------------------------------------------------- #
    # Edge types — structural links, NO features on edges
    # ------------------------------------------------------------------- #
    "edge_types": {
        # Retail: Account --[sends]--> RetailTxn
        ("account", "sends", "retail_txn"): {
            "source_table": "bj.fraud_data_ext",
            "src_col": "ACCOUNTID",
            "dst_col": "TRANSACTIONID",
        },
        # Retail: RetailTxn --[received_by]--> Account
        ("retail_txn", "received_by", "account"): {
            "source_table": "bj.fraud_data_ext",
            "src_col": "TRANSACTIONID",
            "dst_col": "COUNTERPARTYID",
        },

        # Business: Account --[sends_bus]--> BusTxn
        ("account", "sends_bus", "bus_txn"): {
            "source_table": "bj.fraud_data_bus",
            "src_col": "ACCOUNTID",
            "dst_col": None,                # (?) need bus txn ID
        },
        # Business: BusTxn --[received_by_bus]--> Account
        ("bus_txn", "received_by_bus", "account"): {
            "source_table": "bj.fraud_data_bus",
            "src_col": None,                # (?) need bus txn ID
            "dst_col": "COUNTERPARTYID",
        },

        # (?) Optional:
        # ("customer", "owns", "account"): { ... },
        # ("retail_txn", "uses_device", "device"): { ... },
    },

    # ------------------------------------------------------------------- #
    # Labels — fraud flags from ret, joined onto retail_txn nodes
    # ------------------------------------------------------------------- #
    "label": {
        "source_table": "bj.fraud_data_ret",
        "join_key": "ORGTRANSID",           # ret.ORGTRANSID → ext.TRANSACTIONID (?)
        "target_col": "CONFIRMEDRISK",
        "target_node_type": "retail_txn",   # labels live on transaction NODES
        "default": 0,                       # txns not in ret are assumed non-fraud
    },
}
