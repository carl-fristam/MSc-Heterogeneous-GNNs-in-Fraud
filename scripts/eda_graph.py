"""
eda_graph.py — Universal EDA for heterogeneous graph discovery.

Edit the CONFIG block below, then run:
    python scripts/eda_graph.py
"""

# ─── CONFIG ────────────────────────────────────────────────────────────────────
DATA_PATH = "/path/to/your/dataset.parquet"   # csv / parquet / xlsx
SAMPLE    = None          # e.g. 50_000 to analyse a subset; None = all rows
OUT       = None          # e.g. "outputs/eda_report.md"; None = no file saved
# ───────────────────────────────────────────────────────────────────────────────

import sys
import warnings

from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# ─── ANSI colours ──────────────────────────────────────────────────────────────
R = "\033[91m"; G = "\033[92m"; Y = "\033[93m"; B = "\033[94m"
M = "\033[95m"; C = "\033[96m"; W = "\033[97m"; DIM = "\033[2m"; X = "\033[0m"
BOLD = "\033[1m"

def h1(s): return f"\n{BOLD}{W}{'━'*70}\n  {s}\n{'━'*70}{X}"
def h2(s): return f"\n{BOLD}{C}── {s} {X}"
def h3(s): return f"\n{Y}   {s}{X}"
def ok(s):  return f"  {G}✓{X} {s}"
def warn(s):return f"  {Y}⚠{X} {s}"
def info(s):return f"  {B}·{X} {s}"
def flag(s):return f"  {M}★{X} {s}"

# ─── Heuristics ────────────────────────────────────────────────────────────────

ID_PATTERNS = [
    "id", "_id", "id_", "uuid", "key", "code", "no", "num", "number",
    "account", "acct", "user", "customer", "client", "merchant", "sender",
    "receiver", "recipient", "party", "entity", "node",
    "counterparty", "counter", "device", "transaction", "agent", "branch",
]

AMOUNT_PATTERNS = ["amount", "value", "sum", "total", "price", "fee", "balance", "cost", "basevalue"]
TIMESTAMP_PATTERNS = ["date", "time", "datetime", "timestamp", "at", "created", "updated", "when", "eventtime"]
LABEL_PATTERNS = ["label", "fraud", "target", "is_fraud", "y", "class", "flag", "laundering", "illicit", "suspicious", "anomaly",
                  "confirmed_risk", "confirmedrisk", "confirmed risk", "risk"]
TYPE_PATTERNS = ["type", "category", "kind", "channel", "method", "submethod", "currency", "currency_code", "status",
                 "clearing", "format", "idformat", "msgstatus", "exceptionrule"]
GEO_PATTERNS = ["country", "city", "region", "location", "address", "zip", "lat", "lon", "longitude", "latitude",
                "destinationcountry", "agentcountry"]
DEVICE_PATTERNS = ["device", "ipaddress", "useragent", "ip_address", "user_agent"]


def column_role(col: str, series: pd.Series) -> dict:
    """Infer the semantic role of a column."""
    c = col.lower().replace(" ", "_")
    n_unique = series.nunique()
    n_total = len(series)
    null_frac = series.isna().mean()
    dtype = series.dtype

    roles = []

    # --- Label?
    if any(p in c for p in LABEL_PATTERNS):
        roles.append("LABEL")

    # --- Timestamp?
    if dtype in ["datetime64[ns]", "datetime64[us]"] or any(p in c for p in TIMESTAMP_PATTERNS):
        roles.append("TIMESTAMP")

    # --- ID / foreign-key?
    high_cardinality = n_unique / max(n_total, 1) > 0.3
    near_unique = n_unique / max(n_total, 1) > 0.8
    if any(p in c for p in ID_PATTERNS) or (near_unique and dtype == object):
        roles.append("ID")

    # --- Amount / numeric feature?
    if any(p in c for p in AMOUNT_PATTERNS):
        roles.append("AMOUNT")
    elif dtype in [np.float64, np.float32, np.int64, np.int32] and not roles:
        roles.append("NUMERIC")

    # --- Categorical?
    if dtype == object and not any(r in roles for r in ["ID", "TIMESTAMP"]):
        if n_unique <= 200 or any(p in c for p in TYPE_PATTERNS + GEO_PATTERNS):
            roles.append("CATEGORICAL")

    # --- Geography?
    if any(p in c for p in GEO_PATTERNS):
        roles.append("GEO")

    # --- Boolean?
    if set(series.dropna().unique()).issubset({0, 1, True, False, "0", "1", "true", "false", "True", "False"}):
        roles.append("BOOLEAN")

    return {
        "roles": roles or ["UNKNOWN"],
        "n_unique": n_unique,
        "null_frac": null_frac,
        "dtype": str(dtype),
        "high_cardinality": high_cardinality,
        "near_unique": near_unique,
    }


def detect_node_types(meta: dict[str, dict]) -> list[dict]:
    """
    Heuristic: columns that look like entity IDs are candidate node types.
    Pairs of (sender_id, receiver_id) suggest a transaction edge table.
    """
    id_cols = [c for c, m in meta.items() if "ID" in m["roles"]]
    label_cols = [c for c, m in meta.items() if "LABEL" in m["roles"]]
    timestamp_cols = [c for c, m in meta.items() if "TIMESTAMP" in m["roles"]]
    amount_cols = [c for c, m in meta.items() if "AMOUNT" in m["roles"]]

    suggestions = []

    # Pair detection: look for sender/receiver, source/target, from/to patterns
    def find_pair(cols, patterns_a, patterns_b):
        a = [c for c in cols if any(p in c.lower() for p in patterns_a)]
        b = [c for c in cols if any(p in c.lower() for p in patterns_b)]
        return a, b

    senders, receivers = find_pair(id_cols,
        ["sender", "source", "from", "origin", "payer", "debtor", "accountid", "account"],
        ["receiver", "target", "to", "dest", "payee", "creditor", "beneficiary",
         "counterpartyid", "counterparty", "counterentityid"])

    if senders and receivers:
        suggestions.append({
            "pattern": "BIPARTITE_TRANSACTION",
            "sender_cols": senders,
            "receiver_cols": receivers,
            "edge_table": True,
            "label_cols": label_cols,
            "timestamp_cols": timestamp_cols,
            "amount_cols": amount_cols,
        })

    # Generic: if ≥2 distinct ID columns
    if len(id_cols) >= 2 and not suggestions:
        suggestions.append({
            "pattern": "MULTI_ENTITY",
            "id_cols": id_cols,
            "label_cols": label_cols,
            "timestamp_cols": timestamp_cols,
            "amount_cols": amount_cols,
        })

    if not suggestions:
        suggestions.append({
            "pattern": "FLAT",
            "id_cols": id_cols,
            "label_cols": label_cols,
            "note": "No clear entity relationships detected. May need manual schema definition.",
        })

    return suggestions


def describe_numeric(series: pd.Series) -> str:
    s = series.dropna()
    if len(s) == 0:
        return "all null"
    p = np.percentile(s, [0, 1, 25, 50, 75, 99, 100])
    skew = s.skew()
    return (f"min={p[0]:.3g}  p1={p[1]:.3g}  p25={p[2]:.3g}  "
            f"median={p[3]:.3g}  p75={p[4]:.3g}  p99={p[5]:.3g}  max={p[6]:.3g}  "
            f"mean={s.mean():.3g}  skew={skew:.2f}")


def top_values(series: pd.Series, n=10) -> str:
    vc = series.value_counts(dropna=False).head(n)
    total = len(series)
    parts = [f"{str(v)!r}:{c}({100*c/total:.1f}%)" for v, c in vc.items()]
    return "  |  ".join(parts)


def try_parse_datetime(series: pd.Series) -> pd.Series | None:
    try:
        parsed = pd.to_datetime(series, infer_datetime_format=True, errors="coerce")
        if parsed.notna().mean() > 0.7:
            return parsed
    except Exception:
        pass
    return None


# ─── Main EDA ──────────────────────────────────────────────────────────────────

def run_eda(path: Path, sample: int | None, out: Path | None):
    lines = []  # for markdown output

    def emit(s: str):
        """Print and collect (strip ANSI for file)."""
        print(s)
        # strip ANSI
        import re
        lines.append(re.sub(r"\033\[[0-9;]*m", "", s))

    # ── Load ──────────────────────────────────────────────────────────────────
    emit(h1(f"EDA — {path.name}"))
    suffix = path.suffix.lower()
    if suffix == ".csv":
        df_full = pd.read_csv(path, low_memory=False)
    elif suffix == ".parquet":
        df_full = pd.read_parquet(path)
    elif suffix in (".xlsx", ".xls"):
        df_full = pd.read_excel(path)
    else:
        emit(warn(f"Unknown extension {suffix}, attempting CSV read."))
        df_full = pd.read_csv(path, low_memory=False)

    n_rows, n_cols = df_full.shape
    emit(ok(f"Loaded  {n_rows:,} rows × {n_cols} columns"))

    if sample and n_rows > sample:
        df = df_full.sample(sample, random_state=42)
        emit(info(f"Sampled {sample:,} rows for analysis"))
    else:
        df = df_full

    # ── Auto-parse dates ──────────────────────────────────────────────────────
    for col in df.columns:
        if df[col].dtype == object and any(p in col.lower() for p in TIMESTAMP_PATTERNS):
            parsed = try_parse_datetime(df[col])
            if parsed is not None:
                df[col] = parsed

    # ── Column metadata ───────────────────────────────────────────────────────
    emit(h2("COLUMN OVERVIEW"))
    meta = {}
    for col in df.columns:
        meta[col] = column_role(col, df[col])

    # Table header
    header = f"  {'Column':<35} {'Type':<12} {'Roles':<30} {'Unique':>8} {'Null%':>6}"
    emit(f"\n{DIM}{header}{X}")
    emit(f"  {'-'*95}")

    for col, m in meta.items():
        roles_str = ",".join(m["roles"])
        null_str = f"{m['null_frac']*100:.1f}%"
        null_col = R if m["null_frac"] > 0.2 else (Y if m["null_frac"] > 0.02 else G)
        role_col = M if "LABEL" in m["roles"] else (C if "ID" in m["roles"] else (Y if "TIMESTAMP" in m["roles"] else W))
        unique_str = f"{m['n_unique']:,}"
        emit(f"  {col:<35} {DIM}{m['dtype']:<12}{X} {role_col}{roles_str:<30}{X} {unique_str:>8} {null_col}{null_str:>6}{X}")

    # ── Per-column detail ─────────────────────────────────────────────────────
    emit(h2("PER-COLUMN DETAIL"))

    for col, m in meta.items():
        s = df[col]
        null_info = f"{m['null_frac']*100:.1f}% null" if m["null_frac"] > 0 else "no nulls"
        roles_str = "/".join(m["roles"])
        emit(h3(f"{col}  [{roles_str}]  ({null_info})"))

        if "TIMESTAMP" in m["roles"] or pd.api.types.is_datetime64_any_dtype(s):
            valid = s.dropna()
            if len(valid):
                emit(info(f"Range: {valid.min()} → {valid.max()}"))
                span = valid.max() - valid.min()
                emit(info(f"Span:  {span}"))
                if hasattr(span, 'days') and span.days > 0:
                    emit(info(f"Transactions per day (approx): {len(valid)/span.days:.1f}"))

        elif "LABEL" in m["roles"]:
            vc = s.value_counts(dropna=False).head(20)
            total = len(s)
            for val, cnt in vc.items():
                bar = "█" * int(30 * cnt / total)
                pct = 100 * cnt / total
                colour = R if pct < 10 else G
                emit(info(f"  {str(val):<15} {cnt:>8,}  ({pct:5.2f}%)  {colour}{bar}{X}"))
            if len(vc) == 2:
                minority = vc.min()
                majority = vc.max()
                emit(flag(f"Class imbalance ratio: 1 : {majority/minority:.0f}"))

        elif "ID" in m["roles"]:
            emit(info(f"Unique values: {m['n_unique']:,}  ({m['n_unique']/len(s)*100:.1f}% of rows)"))
            if m["near_unique"]:
                emit(ok("Near-unique → good primary key / node ID candidate"))
            else:
                emit(warn("Not near-unique → may be a foreign key / repeated entity"))
            # only show samples for low-cardinality IDs (e.g. account type codes)
            if m["n_unique"] <= 50:
                emit(info(f"Values: {top_values(s, n=10)}"))

        elif pd.api.types.is_numeric_dtype(s):
            emit(info(describe_numeric(s)))
            zeros = (s == 0).sum()
            negatives = (s < 0).sum()
            if zeros > 0:
                emit(info(f"Zeros: {zeros:,} ({100*zeros/len(s):.1f}%)"))
            if negatives > 0:
                emit(warn(f"Negative values: {negatives:,} ({100*negatives/len(s):.1f}%)"))

        elif "CATEGORICAL" in m["roles"] and m["n_unique"] <= 500:
            emit(info(f"Cardinality: {m['n_unique']:,}"))
            emit(info(f"Top values: {top_values(s)}"))

        elif s.dtype == object and m["n_unique"] > 500:
            emit(info(f"High-cardinality text/ID column ({m['n_unique']:,} unique) — skipping value listing"))

    # ── Correlations between numeric columns ──────────────────────────────────
    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    label_cols_found = [c for c, m in meta.items() if "LABEL" in m["roles"]]

    if label_cols_found and numeric_cols:
        emit(h2("NUMERIC CORRELATIONS WITH LABEL"))
        for lc in label_cols_found:
            if pd.api.types.is_numeric_dtype(df[lc]):
                corrs = df[numeric_cols].corrwith(df[lc]).abs().sort_values(ascending=False)
                emit(info(f"Label: {lc}"))
                for col, corr in corrs.head(15).items():
                    bar = "█" * int(corr * 30)
                    colour = G if corr > 0.1 else DIM
                    emit(info(f"  {col:<35} r={corr:.3f}  {colour}{bar}{X}"))

    # ── Missing value patterns ────────────────────────────────────────────────
    missing = df.isna().sum()
    missing = missing[missing > 0].sort_values(ascending=False)
    if len(missing):
        emit(h2("MISSING VALUES"))
        for col, cnt in missing.items():
            pct = 100 * cnt / len(df)
            colour = R if pct > 20 else Y
            bar = "█" * int(pct / 2)
            emit(info(f"  {col:<35} {cnt:>8,}  ({colour}{pct:.1f}%{X})  {colour}{bar}{X}"))

    # ── Duplicate rows ────────────────────────────────────────────────────────
    n_dupes = df.duplicated().sum()
    if n_dupes:
        emit(warn(f"Duplicate rows: {n_dupes:,} ({100*n_dupes/len(df):.2f}%)"))
    else:
        emit(ok("No duplicate rows"))

    # ── Internal vs External node analysis ───────────────────────────────────
    emit(h1("INTERNAL vs EXTERNAL ENTITY ANALYSIS"))
    _c = {c.upper(): c for c in df.columns}  # uppercase lookup

    # Sender / receiver ID columns (best guesses)
    sender_col   = next((c for k, c in _c.items() if k in ("ACCOUNTID", "SENDERID", "SOURCEID")), None)
    receiver_col = next((c for k, c in _c.items() if k in ("COUNTERPARTYID", "RECEIVERID", "DESTID", "COUNTERENTITYID")), None)
    onus_col     = next((c for k, c in _c.items() if k in ("TRANSACTIONONUS", "ONUS", "INTERNAL")), None)
    sagent_col   = next((c for k, c in _c.items() if k in ("ACCOUNTAGENTID", "SENDERAGENTID", "SOURCEAGENTID")), None)
    cagent_col   = next((c for k, c in _c.items() if k in ("COUNTERAGENTID", "RECEIVERAGENTID")), None)
    customer_col = next((c for k, c in _c.items() if k in ("CUSTOMERID", "CUSTOMERENTITYID")), None)
    label_col_found = next((c for k, c in _c.items() if any(p in k.lower() for p in LABEL_PATTERNS)), None)

    if sender_col and receiver_col:
        sender_ids   = set(df[sender_col].dropna().unique())
        receiver_ids = set(df[receiver_col].dropna().unique())
        overlap      = sender_ids & receiver_ids
        only_sender  = sender_ids - receiver_ids
        only_receiver= receiver_ids - sender_ids

        emit(h2("Account pool overlap"))
        emit(info(f"Unique senders   ({sender_col}):   {len(sender_ids):,}"))
        emit(info(f"Unique receivers ({receiver_col}): {len(receiver_ids):,}"))
        emit(info(f"Appear as BOTH sender & receiver:  {len(overlap):,}  "
                  f"({100*len(overlap)/max(len(sender_ids|receiver_ids),1):.1f}% of all accounts)"))
        emit(info(f"Only ever sender:   {len(only_sender):,}"))
        emit(info(f"Only ever receiver: {len(only_receiver):,}"))

        if len(overlap) / max(len(sender_ids | receiver_ids), 1) > 0.3:
            emit(flag("High overlap → unified account node pool is appropriate"))
        else:
            emit(warn("Low overlap → sender and receiver populations are distinct; "
                      "consider separate node types (e.g. internal vs external)"))

    if onus_col:
        emit(h2(f"On-us flag  ({onus_col})"))
        vc = df[onus_col].value_counts(dropna=False)
        total = len(df)
        for val, cnt in vc.items():
            bar = "█" * int(30 * cnt / total)
            emit(info(f"  {str(val):<10} {cnt:>8,}  ({100*cnt/total:.1f}%)  {bar}"))
        n_internal = vc.get(True, vc.get("true", vc.get("True", vc.get(1, 0))))
        n_external = total - n_internal
        emit(flag(f"~{n_internal:,} internal (on-us) transactions  |  ~{n_external:,} external"))
        emit(info("Internal = both accounts at the same bank → denser subgraph"))
        emit(info("External = counterparty is an outside bank → sparser, higher AML risk signal"))

        if label_col_found and pd.api.types.is_numeric_dtype(df[label_col_found]):
            emit(h2(f"Fraud rate by on-us flag"))
            grouped = df.groupby(onus_col)[label_col_found].mean()
            for val, rate in grouped.items():
                bar = "█" * int(rate * 200)
                emit(info(f"  {str(val):<10}  fraud rate = {rate:.4f}  {R if rate > 0.01 else DIM}{bar}{X}"))

    if sagent_col and cagent_col:
        emit(h2(f"Agent (bank) diversity  ({sagent_col} vs {cagent_col})"))
        emit(info(f"Unique sender agents:   {df[sagent_col].nunique():,}"))
        emit(info(f"Unique receiver agents: {df[cagent_col].nunique():,}"))
        cross = df[df[sagent_col] != df[cagent_col]]
        emit(info(f"Cross-bank transactions: {len(cross):,}  ({100*len(cross)/len(df):.1f}%)"))
        emit(info(f"Top receiver agents: {top_values(df[cagent_col], n=5)}"))

    if customer_col and sender_col:
        emit(h2(f"Customer ↔ Account linkage  ({customer_col})"))
        n_customers = df[customer_col].nunique()
        n_accounts  = df[sender_col].nunique()
        ratio = n_accounts / max(n_customers, 1)
        emit(info(f"Unique customers: {n_customers:,}"))
        emit(info(f"Unique accounts:  {n_accounts:,}"))
        emit(info(f"Accounts per customer (avg): {ratio:.2f}"))
        if ratio > 1.1:
            emit(flag("Multi-account customers present → Customer node type adds value"))
        else:
            emit(info("Mostly 1-to-1 customer↔account → Customer node may be redundant"))

    emit(h2("Homogeneous vs Heterogeneous recommendation"))
    emit(info("HOMOGENEOUS  — use if:"))
    emit(info("  · Only one meaningful entity type (e.g. all nodes are accounts/customers)"))
    emit(info("  · No rich per-type features; internal/external split is negligible"))
    emit(info("  · Simplicity is preferred; baseline model"))
    emit("")
    emit(info("HETEROGENEOUS — use if:"))
    emit(info("  · Distinct node types with different feature spaces"))
    emit(info("    e.g. Account (IBAN features) vs Customer (type, country) vs Device (IP, UA)"))
    emit(info("  · Internal vs external accounts have structurally different connectivity"))
    emit(info("  · You want to model Customer→Account→Transaction as separate relation types"))
    emit(info("  · Evidence: low sender/receiver overlap, significant cross-bank volume,"))
    emit(info("    or multi-account customers above"))

    # ── Graph schema suggestions ──────────────────────────────────────────────
    emit(h1("HETEROGENEOUS GRAPH SCHEMA SUGGESTIONS"))
    suggestions = detect_node_types(meta)

    for sg in suggestions:
        pattern = sg["pattern"]

        if pattern == "BIPARTITE_TRANSACTION":
            emit(flag(f"Pattern detected: BIPARTITE TRANSACTION GRAPH"))
            emit(info("This table looks like an edge list. Suggested schema:"))
            emit("")
            for s_col in sg["sender_cols"]:
                for r_col in sg["receiver_cols"]:
                    entity_a = s_col.lower().replace("_id","").replace("id_","").replace("id","").strip("_") or "entity_a"
                    entity_b = r_col.lower().replace("_id","").replace("id_","").replace("id","").strip("_") or "entity_b"
                    same_pool = any(kw in entity_a for kw in ["account","user","node","party"]) and \
                                any(kw in entity_b for kw in ["account","user","node","party"])
                    emit(f"    {B}Node type A{X}:  {G}{entity_a}{X}  (from column '{s_col}')")
                    emit(f"    {B}Node type B{X}:  {G}{entity_b}{X}  (from column '{r_col}')")
                    if same_pool:
                        emit(f"    {Y}→ A and B may be the SAME node type (unified account pool){X}")
                    emit(f"    {B}Edge{X}:         ({entity_a}) ──[sends]──▶ transaction ──[received_by]──▶ ({entity_b})")
                    emit(f"    {B}or simpler{X}:   ({entity_a}) ──[transacts_with]──▶ ({entity_b})")
                    emit("")

            if sg["label_cols"]:
                emit(info(f"Labels found → likely NODE or EDGE classification task"))
                for lc in sg["label_cols"]:
                    vc = df[lc].value_counts()
                    emit(info(f"  '{lc}': {dict(vc.head(5))}"))

            if sg["timestamp_cols"]:
                emit(info(f"Timestamp columns → temporal graph possible: {sg['timestamp_cols']}"))
                emit(info("  Consider: train/val/test splits by time (avoid data leakage)"))

            if sg["amount_cols"]:
                emit(info(f"Amount columns → natural edge features: {sg['amount_cols']}"))

        elif pattern == "MULTI_ENTITY":
            emit(flag("Pattern detected: MULTI-ENTITY TABLE"))
            emit(info(f"ID columns found: {sg['id_cols']}"))
            emit(info("Suggested approach: treat each ID column as a separate node type,"))
            emit(info("and rows as hyper-edges or explicit transaction nodes."))
            emit("")
            emit(info("PyG HeteroData skeleton:"))
            for i, id_col in enumerate(sg["id_cols"][:4]):
                emit(info(f"  data['{id_col}'].x  # node features"))
            emit(info("  data[(..., 'connects', ...)].edge_index"))

        else:
            emit(warn(f"Pattern: FLAT — no clear entity relationships detected."))
            emit(info("Options:"))
            emit(info("  1. Each row = a node (homogeneous graph, similarity edges)"))
            emit(info("  2. Manually define entity columns in schema.py"))
            emit(info(f"  3. Known ID cols: {sg['id_cols']}"))

    # ── PyG skeleton ─────────────────────────────────────────────────────────
    emit(h2("SUGGESTED PyG HeteroData SKELETON"))
    id_cols_list = [c for c, m in meta.items() if "ID" in m["roles"]]
    numeric_feat_cols = [c for c, m in meta.items() if any(r in m["roles"] for r in ["NUMERIC", "AMOUNT", "CATEGORICAL", "BOOLEAN", "GEO"]) and "ID" not in m["roles"] and "LABEL" not in m["roles"] and "TIMESTAMP" not in m["roles"]]
    lc = label_cols_found[0] if label_cols_found else "label_col"
    ts = [c for c, m in meta.items() if "TIMESTAMP" in m["roles"]]
    ts_hint = f"  # temporal split on '{ts[0]}'" if ts else ""

    emit(f"""
  {DIM}# ── Build graph ──────────────────────────────────────────────────{X}
  from torch_geometric.data import HeteroData
  import torch

  data = HeteroData()

  {DIM}# Node features (example — adapt per your entity columns){X}
  {DIM}# data['account'].x      = torch.tensor(acct_features,  dtype=torch.float){X}
  {DIM}# data['transaction'].x  = torch.tensor(txn_features,   dtype=torch.float){X}

  {DIM}# Edge indices — shape (2, num_edges){X}
  {DIM}# data[('account', 'sends', 'transaction')].edge_index = ..{X}
  {DIM}# data[('transaction', 'received_by', 'account')].edge_index = ..{X}

  {DIM}# Labels & masks on transaction nodes{X}
  {DIM}# data['transaction'].y          = torch.tensor(df['{lc}'].values){X}{ts_hint}
  {DIM}# data['transaction'].train_mask = ...{X}

  {DIM}# Feature columns identified in this dataset:{X}
  {DIM}# Potential node IDs:  {id_cols_list[:6]}{X}
  {DIM}# Feature columns:     {numeric_feat_cols[:8]}{X}
""")

    # ── Summary ───────────────────────────────────────────────────────────────
    emit(h2("QUICK SUMMARY"))
    emit(info(f"Shape:       {n_rows:,} rows × {n_cols} columns"))
    emit(info(f"ID columns:  {[c for c,m in meta.items() if 'ID' in m['roles']]}"))
    emit(info(f"Labels:      {label_cols_found}"))
    emit(info(f"Timestamps:  {[c for c,m in meta.items() if 'TIMESTAMP' in m['roles']]}"))
    emit(info(f"Amounts:     {[c for c,m in meta.items() if 'AMOUNT' in m['roles']]}"))
    emit(info(f"Categoricals:{[c for c,m in meta.items() if 'CATEGORICAL' in m['roles']]}"))
    emit(info(f"Missing:     {int((df.isna().any(axis=1)).sum()):,} rows have ≥1 null"))

    # ── Write markdown report ─────────────────────────────────────────────────
    if out:
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w") as f:
            f.write("\n".join(lines))
        print(f"\n{G}Report saved → {out}{X}")


if __name__ == "__main__":
    p = Path(DATA_PATH)
    if not p.exists():
        print(f"{R}File not found: {p}{X}")
        sys.exit(1)
    run_eda(p, SAMPLE, Path(OUT) if OUT else None)
