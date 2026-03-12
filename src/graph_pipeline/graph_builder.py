"""
Graph builder — orchestrates the full pipeline.

Calls loader, splitter, feature extractors, and edge builder,
then assembles everything into a PyG HeteroData object.
"""

import numpy as np
import torch
from torch_geometric.data import HeteroData

from src.graph_pipeline.schema import DatasetSchema, get_schema
from src.graph_pipeline.loader import load_raw
from src.graph_pipeline.split import temporal_split, random_stratified_split
from src.graph_pipeline.features_txn import build_txn_features, get_registered_features as get_txn_registry
from src.graph_pipeline.features_acct import build_account_mapping, build_acct_features, get_registered_features as get_acct_registry
from src.graph_pipeline.cache import load_cache, save_cache
from src.utils.config import PROJECT_ROOT


def build_graph(config: dict) -> tuple[HeteroData, dict]:
    """
    Main entry point: build a heterogeneous graph from config.

    Args:
        config: dictionary from load_config("graph_pipeline")

    Returns:
        (data, account_to_id)
            data:           PyG HeteroData with account + transaction nodes
            account_to_id:  dict mapping account string → int node index
    """
    # --- Check cache first ---
    cache_path = _resolve_cache_path(config)
    if config.get("cache", {}).get("enabled", True):
        cached = load_cache(cache_path)
        if cached is not None:
            return cached["data"], cached["account_to_id"]

    # --- Schema ---
    schema = get_schema(config["dataset"])
    print(f"\n{'='*60}")
    print(f"Building graph for dataset: {config['dataset']}")
    print(f"{'='*60}")

    # --- Load raw data ---
    df = load_raw(
        data_path=str(PROJECT_ROOT / config["data_path"]),
        schema=schema,
        sample_ratio=config.get("sample_ratio", 1.0),
        n_days=config.get("n_days", None),
    )

    # --- Split ---
    split_cfg = config["split"]
    split_method = split_cfg.get("method", "temporal")

    if split_method == "random":
        train_mask, val_mask, test_mask = random_stratified_split(
            df,
            label_col=schema.label,
            train_ratio=split_cfg.get("train_ratio", 0.7),
            val_ratio=split_cfg.get("val_ratio", 0.15),
            seed=split_cfg.get("seed", 42),
        )
    else:
        train_mask, val_mask, test_mask = temporal_split(
            df,
            train_end=split_cfg["train_end_date"],
            val_end=split_cfg["val_end_date"],
        )

    # --- Account mapping ---
    account_to_id = build_account_mapping(df)

    # --- Features ---
    print("\nBuilding account features (train-only aggregation)...")
    feat_cfg = config.get("features", {})
    acct_features = build_acct_features(
        df, schema, account_to_id, train_mask,
        enabled=feat_cfg.get("account", None),
    )

    print("\nBuilding transaction features...")
    txn_features = build_txn_features(
        df, schema,
        enabled=feat_cfg.get("transaction", None),
    )

    # --- Edges ---
    print("\nBuilding edges...")
    add_reverse = config.get("graph", {}).get("add_reverse_edges", True)
    edge_dict = _build_edges(df, account_to_id, add_reverse)

    # --- Labels ---
    if schema.label is not None:
        labels = torch.tensor(df[schema.label].values, dtype=torch.float)
    else:
        labels = torch.zeros(len(df), dtype=torch.float)

    # --- Assemble HeteroData ---
    data = HeteroData()

    data["account"].x = acct_features
    data["account"].num_nodes = len(account_to_id)

    data["transaction"].x = torch.tensor(txn_features, dtype=torch.float)
    data["transaction"].y = labels
    data["transaction"].train_mask = torch.tensor(train_mask.values)
    data["transaction"].val_mask = torch.tensor(val_mask.values)
    data["transaction"].test_mask = torch.tensor(test_mask.values)

    for edge_type, edge_index in edge_dict.items():
        data[edge_type].edge_index = edge_index

    # --- Print summary ---
    _print_summary(data)

    # --- Cache ---
    if config.get("cache", {}).get("enabled", True):
        save_cache(cache_path, {"data": data, "account_to_id": account_to_id})

    return data, account_to_id


# Edge construction

def _build_edges(df, account_to_id, add_reverse):
    """
    Build edge_index tensors for the bipartite graph.

    Each transaction i creates:
      account --[sends]--> transaction i
      transaction i --[received_by]--> account
    Plus reverse edges if requested.
    """
    sender_ids = df["_sender"].map(account_to_id).values.astype(np.int64)
    receiver_ids = df["_receiver"].map(account_to_id).values.astype(np.int64)
    txn_ids = np.arange(len(df), dtype=np.int64)

    edges = {}

    # Forward edges
    edges[("account", "sends", "transaction")] = torch.tensor(
        np.stack([sender_ids, txn_ids]), dtype=torch.long
    )
    edges[("transaction", "received_by", "account")] = torch.tensor(
        np.stack([txn_ids, receiver_ids]), dtype=torch.long
    )

    if add_reverse:
        edges[("transaction", "sent_by", "account")] = torch.tensor(
            np.stack([txn_ids, sender_ids]), dtype=torch.long
        )
        edges[("account", "receives", "transaction")] = torch.tensor(
            np.stack([receiver_ids, txn_ids]), dtype=torch.long
        )

    for etype, ei in edges.items():
        print(f"    {etype}: {ei.shape[1]:,} edges")

    return edges


# Helpers

def _resolve_cache_path(config: dict) -> str:
    """Build a cache file path from config parameters."""
    cache_dir = config.get("cache", {}).get("dir", "data/processed")
    dataset = config["dataset"]
    sr = config.get("sample_ratio", 1.0)
    n_days = config.get("n_days", None)
    suffix = f"d{n_days}" if n_days else f"sr{sr:.2f}"
    return str(PROJECT_ROOT / cache_dir / f"graph_pipeline_{dataset}_{suffix}.pkl")


def _print_summary(data: HeteroData):
    """Print a summary of the assembled graph."""
    print(f"\n{'='*60}")
    print("Graph Summary")
    print(f"{'='*60}")
    print(f"  Account nodes:     {data['account'].num_nodes:,}")
    print(f"  Transaction nodes: {data['transaction'].x.shape[0]:,}")
    print(f"  Account feat dim:  {data['account'].x.shape[1]}")
    print(f"  Txn feat dim:      {data['transaction'].x.shape[1]}")
    print(f"  Edge types:        {len(data.edge_types)}")
    for et in data.edge_types:
        print(f"    {et}: {data[et].edge_index.shape[1]:,} edges")

    # Label stats per split
    y = data["transaction"].y
    for name in ["train", "val", "test"]:
        mask = data["transaction"][f"{name}_mask"]
        n = mask.sum().item()
        pos = y[mask].sum().item()
        print(f"  {name:5s}: {n:,} txns, {int(pos)} positive ({100*pos/n:.2f}%)" if n > 0 else f"  {name}: empty")


def feature_table():
    """
    Return the full feature inventory as two pandas DataFrames.

    Usage:
        from src.graph_pipeline import feature_table
        txn_df, acct_df = feature_table()

    In a terminal / notebook, just call it and pandas will render the table:
        feature_table()[0]   # transaction features
        feature_table()[1]   # account features
    """
    import pandas as pd

    txn_reg = get_txn_registry()
    acct_reg = get_acct_registry()

    def _to_df(registry, node_type):
        rows = []
        cumul = 0
        for name, fn in registry.items():
            dim = fn._feat_dim
            cumul += dim
            doc = (fn.__doc__ or "").strip().split("\n")[0]
            rows.append({"node_type": node_type, "feature": name, "dims": dim, "cumulative": cumul, "description": doc})
        return pd.DataFrame(rows)

    txn_df = _to_df(txn_reg, "transaction")
    acct_df = _to_df(acct_reg, "account")

    # Print both for convenience
    print("\n=== TRANSACTION FEATURES ===")
    print(txn_df.to_string(index=False))
    print(f"\n=== ACCOUNT FEATURES ===")
    print(acct_df.to_string(index=False))

    return txn_df, acct_df


def print_feature_inventory(export: str | None = None):
    """
    Print a clean table of all registered features — useful for sanity checks.

    Args:
        export: optional file path to export the inventory.
                Supports .html, .csv, and .md extensions.
                If None, just prints to terminal.

    Usage:
        from src.graph_pipeline import print_feature_inventory
        print_feature_inventory()                                    # terminal only
        print_feature_inventory("outputs/features.html")             # open in browser
        print_feature_inventory("outputs/features.csv")              # for spreadsheets
        print_feature_inventory("outputs/features.md")               # for docs/thesis
    """
    txn_reg = get_txn_registry()
    acct_reg = get_acct_registry()

    # Build structured data
    txn_rows = []
    cumul = 0
    for name, fn in txn_reg.items():
        dim = fn._feat_dim
        cumul += dim
        doc = (fn.__doc__ or "").strip().split("\n")[0]
        txn_rows.append({"name": name, "dims": dim, "cumulative": cumul, "description": doc})

    acct_rows = []
    cumul = 0
    for name, fn in acct_reg.items():
        dim = fn._feat_dim
        cumul += dim
        doc = (fn.__doc__ or "").strip().split("\n")[0]
        acct_rows.append({"name": name, "dims": dim, "cumulative": cumul, "description": doc})

    # --- Terminal output ---
    _print_table_terminal("TRANSACTION FEATURES", txn_rows)
    _print_table_terminal("ACCOUNT FEATURES", acct_rows)

    # --- Export if requested ---
    if export is not None:
        import os
        os.makedirs(os.path.dirname(export) if os.path.dirname(export) else ".", exist_ok=True)

        if export.endswith(".html"):
            _export_html(export, txn_rows, acct_rows)
        elif export.endswith(".csv"):
            _export_csv(export, txn_rows, acct_rows)
        elif export.endswith(".md"):
            _export_markdown(export, txn_rows, acct_rows)
        else:
            print(f"  Unknown export format: {export} (use .html, .csv, or .md)")


def _print_table_terminal(title, rows):
    """Print a feature table to the terminal."""
    total = sum(r["dims"] for r in rows)
    print(f"\n{'─'*80}")
    print(f"  {title}")
    print(f"{'─'*80}")
    print(f"  {'#':<4} {'Name':<25} {'Dims':<6} {'Cumul':<7} {'Description'}")
    print(f"  {'─'*4} {'─'*25} {'─'*6} {'─'*7} {'─'*30}")
    for i, r in enumerate(rows, 1):
        print(f"  {i:<4} {r['name']:<25} {r['dims']:<6} {r['cumulative']:<7} {r['description']}")
    print(f"  {'':4} {'TOTAL':<25} {total:<6}")


def _export_html(path, txn_rows, acct_rows):
    """Export feature inventory as a styled HTML file."""
    def _table_html(title, rows):
        total = sum(r["dims"] for r in rows)
        html = f"<h2>{title}</h2>\n"
        html += "<table>\n<tr><th>#</th><th>Name</th><th>Dims</th><th>Cumulative</th><th>Description</th></tr>\n"
        for i, r in enumerate(rows, 1):
            html += f"<tr><td>{i}</td><td><code>{r['name']}</code></td><td>{r['dims']}</td><td>{r['cumulative']}</td><td>{r['description']}</td></tr>\n"
        html += f"<tr class='total'><td></td><td><strong>TOTAL</strong></td><td><strong>{total}</strong></td><td></td><td></td></tr>\n"
        html += "</table>\n"
        return html

    html = """<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>Feature Inventory</title>
<style>
  body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; max-width: 900px; margin: 40px auto; padding: 0 20px; background: #fafafa; }
  h1 { color: #2d3436; border-bottom: 3px solid #6c5ce7; padding-bottom: 10px; }
  h2 { color: #6c5ce7; margin-top: 30px; }
  table { border-collapse: collapse; width: 100%; margin: 15px 0; background: white; box-shadow: 0 1px 3px rgba(0,0,0,0.1); }
  th { background: #6c5ce7; color: white; padding: 10px 14px; text-align: left; }
  td { padding: 8px 14px; border-bottom: 1px solid #eee; }
  tr:hover { background: #f0f0f0; }
  tr.total { background: #f8f8f8; font-weight: bold; }
  code { background: #f1f0ff; padding: 2px 6px; border-radius: 3px; color: #6c5ce7; }
  .meta { color: #636e72; font-size: 0.9em; }
</style></head><body>
<h1>Feature Inventory</h1>
<p class="meta">Generated by <code>print_feature_inventory()</code> from <code>src/graph_pipeline</code></p>
"""
    html += _table_html("Transaction Features", txn_rows)
    html += _table_html("Account Features", acct_rows)
    html += "</body></html>"

    with open(path, "w") as f:
        f.write(html)
    print(f"\n  Exported HTML → {path}")


def _export_csv(path, txn_rows, acct_rows):
    """Export feature inventory as CSV."""
    import csv
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["type", "name", "dims", "cumulative", "description"])
        for r in txn_rows:
            w.writerow(["transaction", r["name"], r["dims"], r["cumulative"], r["description"]])
        for r in acct_rows:
            w.writerow(["account", r["name"], r["dims"], r["cumulative"], r["description"]])
    print(f"\n  Exported CSV → {path}")


def _export_markdown(path, txn_rows, acct_rows):
    """Export feature inventory as Markdown (for thesis/docs)."""
    def _table_md(title, rows):
        total = sum(r["dims"] for r in rows)
        md = f"### {title}\n\n"
        md += "| # | Name | Dims | Cumulative | Description |\n"
        md += "|---|------|------|------------|-------------|\n"
        for i, r in enumerate(rows, 1):
            md += f"| {i} | `{r['name']}` | {r['dims']} | {r['cumulative']} | {r['description']} |\n"
        md += f"| | **TOTAL** | **{total}** | | |\n\n"
        return md

    md = "# Feature Inventory\n\n"
    md += _table_md("Transaction Features", txn_rows)
    md += _table_md("Account Features", acct_rows)

    with open(path, "w") as f:
        f.write(md)
    print(f"\n  Exported Markdown → {path}")

def inspect_features(data, account_to_id, node_type="account"):
    """
    Return a pandas DataFrame of actual computed feature values.

    Usage:
        data, account_to_id = build_graph(config)
        inspect_features(data, account_to_id, "account")
        inspect_features(data, account_to_id, "transaction")
    """
    import pandas as pd

    if node_type == "account":
        registry = get_acct_registry()
        tensor = data["account"].x.numpy()
        id_to_label = {v: k for k, v in account_to_id.items()}
        index = [id_to_label[i] for i in range(len(tensor))]
        index_name = "account_id"
    else:
        registry = get_txn_registry()
        tensor = data["transaction"].x.numpy()
        index = range(len(tensor))
        index_name = "txn_id"

    # Expand feature names by dims
    cols = []
    for name, fn in registry.items():
        dim = fn._feat_dim
        if dim == 1:
            cols.append(name)
        else:
            cols.extend([f"{name}_{i}" for i in range(dim)])

    df = pd.DataFrame(tensor, columns=cols, index=index)
    df.index.name = index_name
    return df