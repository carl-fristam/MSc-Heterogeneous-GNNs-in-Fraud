"""
eda_hetero_graph.py
-------------------
Universal EDA script for tabular datasets, focused on identifying structure
suitable for modelling as a heterogeneous graph.

Usage:
    python scripts/eda_hetero_graph.py path/to/dataset.csv
    python scripts/eda_hetero_graph.py path/to/dataset.csv --sample 50000
    python scripts/eda_hetero_graph.py path/to/dataset.csv --out outputs/eda_report.md
    python scripts/eda_hetero_graph.py path/to/dataset.parquet --sep ","

What it produces:
    1. Shape & memory
    2. Column type audit (numeric / categorical / datetime / boolean / id-like / free-text)
    3. Per-column statistics (nulls, cardinality, sample values, distributions)
    4. Correlation snapshot (numeric columns)
    5. Temporal column detection
    6. Candidate graph elements:
         - Node type candidates  (high-cardinality ID-like columns)
         - Edge type candidates  (pairs of ID columns that co-occur on a row)
         - Label / target candidates (low-cardinality, imbalanced, flag-like columns)
         - Feature candidates    (everything else)
    7. Suggested HeteroData sketch
"""

import argparse
import sys
import warnings
from pathlib import Path
from textwrap import indent

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# ─── ANSI colour helpers (stripped in file output) ─────────────────────────────

RESET  = "\033[0m"
BOLD   = "\033[1m"
DIM    = "\033[2m"
GREEN  = "\033[32m"
YELLOW = "\033[33m"
CYAN   = "\033[36m"
RED    = "\033[31m"
MAGENTA = "\033[35m"

def c(text, *codes):
    return "".join(codes) + str(text) + RESET

def strip_ansi(s):
    import re
    return re.sub(r"\033\[[0-9;]*m", "", s)


# ─── Utilities ─────────────────────────────────────────────────────────────────

def load(path: str, sample: int | None, sep: str) -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        sys.exit(f"File not found: {path}")
    ext = p.suffix.lower()
    print(c(f"\n► Loading {p.name} …", BOLD, CYAN))
    if ext in (".parquet", ".pq"):
        df = pd.read_parquet(p)
    elif ext in (".csv", ".tsv", ".txt"):
        df = pd.read_csv(p, sep=sep, low_memory=False)
    elif ext in (".xlsx", ".xls"):
        df = pd.read_excel(p)
    elif ext == ".json":
        df = pd.read_json(p)
    else:
        df = pd.read_csv(p, sep=sep, low_memory=False)

    if sample and len(df) > sample:
        df = df.sample(sample, random_state=42).reset_index(drop=True)
        print(c(f"  (sampled {sample:,} rows)", DIM))
    return df


def human_bytes(n: int) -> str:
    for unit in ["B", "KB", "MB", "GB"]:
        if n < 1024:
            return f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} TB"


# ─── Column classification ──────────────────────────────────────────────────────

ID_KEYWORDS    = ["id", "uuid", "key", "code", "ref", "no", "num", "number",
                  "account", "acct", "user", "customer", "transaction", "txn",
                  "order", "entity", "node", "src", "dst", "source", "target",
                  "sender", "receiver", "from", "to"]
DATE_KEYWORDS  = ["date", "time", "ts", "timestamp", "datetime", "created",
                  "updated", "at", "on", "start", "end"]
LABEL_KEYWORDS = ["label", "fraud", "flag", "is_", "has_", "target", "class",
                  "laundering", "suspicious", "alert", "anomaly", "y", "churn"]
AMOUNT_KEYWORDS = ["amount", "value", "price", "sum", "total", "balance",
                   "fee", "rate", "qty", "quantity", "count", "cnt", "vol"]


def col_name_matches(col: str, keywords: list[str]) -> bool:
    col_l = col.lower()
    return any(kw in col_l for kw in keywords)


def classify_columns(df: pd.DataFrame) -> dict[str, list[str]]:
    """Return a dict mapping category → list of column names."""
    cats: dict[str, list[str]] = {
        "datetime":    [],
        "boolean":     [],
        "id_like":     [],
        "categorical": [],
        "numeric":     [],
        "free_text":   [],
        "constant":    [],
    }

    for col in df.columns:
        s = df[col]
        nuniq = s.nunique(dropna=True)
        n     = len(s)

        # constants
        if nuniq <= 1:
            cats["constant"].append(col)
            continue

        # already datetime dtype
        if pd.api.types.is_datetime64_any_dtype(s):
            cats["datetime"].append(col)
            continue

        # boolean
        if pd.api.types.is_bool_dtype(s) or set(s.dropna().unique()).issubset({0, 1, True, False}):
            cats["boolean"].append(col)
            continue

        # numeric
        if pd.api.types.is_numeric_dtype(s):
            # could still be a numeric id
            if nuniq == n and col_name_matches(col, ID_KEYWORDS):
                cats["id_like"].append(col)
            else:
                cats["numeric"].append(col)
            continue

        # object / string columns
        # try parsing as datetime
        if col_name_matches(col, DATE_KEYWORDS):
            try:
                pd.to_datetime(s.dropna().head(200), infer_datetime_format=True)
                cats["datetime"].append(col)
                continue
            except Exception:
                pass

        # cardinality ratio
        ratio = nuniq / n

        # high-cardinality string → id-like or free-text
        if ratio > 0.5 or (nuniq > 5000):
            if col_name_matches(col, ID_KEYWORDS):
                cats["id_like"].append(col)
            else:
                # check average token length
                avg_len = s.dropna().astype(str).str.split().str.len().mean()
                if avg_len and avg_len > 4:
                    cats["free_text"].append(col)
                else:
                    cats["id_like"].append(col)
            continue

        # low cardinality → categorical / boolean
        if nuniq <= 2 or col_name_matches(col, ["flag", "is_", "has_", "bool"]):
            cats["boolean"].append(col)
        else:
            cats["categorical"].append(col)

    return cats


# ─── Section printers ──────────────────────────────────────────────────────────

def section(title: str, lines: list[str]) -> list[str]:
    bar = "─" * 70
    out = [
        "",
        c(bar, BOLD, CYAN),
        c(f"  {title}", BOLD, CYAN),
        c(bar, BOLD, CYAN),
    ]
    out.extend(lines)
    return out


def shape_section(df: pd.DataFrame) -> list[str]:
    mem = human_bytes(df.memory_usage(deep=True).sum())
    lines = [
        f"  Rows    : {c(f'{len(df):,}', BOLD)}",
        f"  Columns : {c(df.shape[1], BOLD)}",
        f"  Memory  : {c(mem, BOLD)}",
        f"  Dtypes  : {dict(df.dtypes.astype(str).value_counts())}",
    ]
    null_total = df.isnull().sum().sum()
    null_pct   = 100 * null_total / df.size
    lines.append(f"  Nulls   : {c(f'{null_total:,}', YELLOW)} ({null_pct:.1f}% of all cells)")
    return section("1. SHAPE & MEMORY", lines)


def type_audit_section(cats: dict[str, list[str]]) -> list[str]:
    lines = []
    icons = {
        "datetime":    "📅",
        "boolean":     "🔲",
        "id_like":     "🔑",
        "categorical": "🏷 ",
        "numeric":     "📊",
        "free_text":   "📝",
        "constant":    "⛔",
    }
    for cat, cols in cats.items():
        if cols:
            label = c(f"{icons[cat]} {cat:<12}", BOLD)
            lines.append(f"  {label}  ({len(cols)})  {c(', '.join(cols[:12]), DIM)}"
                         + (f"  …+{len(cols)-12}" if len(cols) > 12 else ""))
    return section("2. COLUMN TYPE AUDIT", lines)


def per_column_section(df: pd.DataFrame, cats: dict[str, list[str]]) -> list[str]:
    lines = []

    def stat_block(col: str, kind: str) -> list[str]:
        s = df[col]
        n_null  = s.isnull().sum()
        null_pct = 100 * n_null / len(s)
        nuniq   = s.nunique(dropna=True)
        samples = s.dropna().unique()[:5]

        out = [f"  {c(col, BOLD, YELLOW)}  {c(f'[{kind}]', DIM)}"]
        out.append(f"    nulls={c(f'{n_null:,} ({null_pct:.1f}%)', RED if null_pct > 20 else DIM)}  "
                   f"unique={c(nuniq, BOLD)}  "
                   f"sample={c(list(samples), DIM)}")

        if kind == "numeric":
            desc = s.describe()
            out.append(f"    min={desc['min']:.4g}  "
                       f"p25={desc['25%']:.4g}  "
                       f"median={desc['50%']:.4g}  "
                       f"p75={desc['75%']:.4g}  "
                       f"max={desc['max']:.4g}  "
                       f"skew={s.skew():.2f}")

        elif kind in ("categorical", "boolean"):
            vc = s.value_counts(normalize=True).head(5)
            bar_parts = [f"{v!r}:{100*p:.1f}%" for v, p in vc.items()]
            out.append(f"    top-5 → {c('  |  '.join(bar_parts), DIM)}")

        elif kind == "datetime":
            parsed = pd.to_datetime(s, errors="coerce")
            valid  = parsed.dropna()
            if len(valid):
                span = valid.max() - valid.min()
                out.append(f"    range: {valid.min()} → {valid.max()}  "
                           f"span={span.days} days")

        elif kind == "id_like":
            vc    = s.value_counts()
            dupes = (vc > 1).sum()
            out.append(f"    duplicated values: {c(dupes, RED if dupes > 0 else GREEN)}  "
                       f"(uniqueness ratio: {nuniq/len(s):.3f})")

        return out

    for kind, cols in cats.items():
        if kind == "constant" or not cols:
            continue
        for col in cols:
            lines.extend(stat_block(col, kind))
            lines.append("")

    return section("3. PER-COLUMN STATISTICS", lines)


def correlation_section(df: pd.DataFrame, cats: dict[str, list[str]]) -> list[str]:
    num_cols = cats["numeric"]
    if len(num_cols) < 2:
        return section("4. CORRELATIONS", ["  (fewer than 2 numeric columns — skipped)"])

    corr = df[num_cols].corr()
    pairs = []
    for i in range(len(num_cols)):
        for j in range(i + 1, len(num_cols)):
            pairs.append((abs(corr.iloc[i, j]), num_cols[i], num_cols[j], corr.iloc[i, j]))
    pairs.sort(reverse=True)

    lines = [f"  Top correlated numeric pairs (|r| > 0.3):"]
    shown = 0
    for abs_r, a, b, r in pairs:
        if abs_r < 0.3:
            break
        color = RED if abs_r > 0.8 else YELLOW if abs_r > 0.5 else DIM
        lines.append(f"    {c(f'{r:+.3f}', color)}  {a}  ↔  {b}")
        shown += 1
    if shown == 0:
        lines.append(c("    No strongly correlated pairs found.", DIM))
    return section("4. CORRELATIONS (numeric)", lines)


def graph_candidates_section(df: pd.DataFrame, cats: dict[str, list[str]]) -> list[str]:
    lines = []
    n = len(df)

    # ── Node candidates ──────────────────────────────────────────────────────
    lines.append(c("  ► NODE TYPE CANDIDATES  (high-cardinality ID-like columns)", BOLD, GREEN))
    lines.append(c("    These likely map to distinct entity types in your graph.", DIM))
    lines.append("")

    node_cols = []
    for col in cats["id_like"]:
        nuniq = df[col].nunique(dropna=True)
        ratio = nuniq / n
        note  = ""
        if col_name_matches(col, ["account", "acct", "user", "customer", "entity"]):
            note = c("  ← likely ACCOUNT node", MAGENTA, BOLD)
        elif col_name_matches(col, ["transaction", "txn", "order", "transfer"]):
            note = c("  ← likely TRANSACTION node", MAGENTA, BOLD)
        lines.append(f"    {c(col, BOLD)}  unique={nuniq:,}  ratio={ratio:.3f}{note}")
        node_cols.append(col)
    if not node_cols:
        lines.append(c("    (none detected — check if IDs are numeric and classified under 'numeric')", YELLOW))
    lines.append("")

    # ── Edge candidates ───────────────────────────────────────────────────────
    lines.append(c("  ► EDGE TYPE CANDIDATES  (pairs of ID columns on the same row)", BOLD, GREEN))
    lines.append(c("    Each pair (A → B) naturally encodes a relationship / edge type.", DIM))
    lines.append("")

    if len(node_cols) >= 2:
        for i, a in enumerate(node_cols):
            for b in node_cols[i + 1:]:
                # co-occurrence check: rows where both are non-null
                joint = df[[a, b]].dropna()
                pct   = 100 * len(joint) / n
                lines.append(f"    ({c(a, BOLD)}, {c(b, BOLD)})  co-occur on {pct:.1f}% of rows"
                             f"  → edge type  c{c(f\"'{a}' → '{b}'\", CYAN)}")
    else:
        lines.append(c("    (need ≥ 2 node candidates to suggest edges)", YELLOW))
    lines.append("")

    # ── Label candidates ─────────────────────────────────────────────────────
    lines.append(c("  ► LABEL / TARGET CANDIDATES  (low-cardinality, flag-like columns)", BOLD, GREEN))
    lines.append("")

    label_cols = (
        [c for c in cats["boolean"]   if col_name_matches(c, LABEL_KEYWORDS)] +
        [c for c in cats["categorical"] if col_name_matches(c, LABEL_KEYWORDS)] +
        [c for c in cats["numeric"]    if col_name_matches(c, LABEL_KEYWORDS)]
    )
    # also flag very imbalanced boolean columns
    for col in cats["boolean"]:
        if col not in label_cols:
            vc = df[col].value_counts(normalize=True)
            if len(vc) == 2 and vc.min() < 0.1:
                label_cols.append(col)

    for col in label_cols:
        vc = df[col].value_counts()
        imb = vc.min() / vc.sum()
        color = RED if imb < 0.05 else YELLOW
        lines.append(f"    {c(col, BOLD)}  dist={dict(vc.head(4))}  "
                     f"minority={c(f'{100*imb:.2f}%', color)}")
    if not label_cols:
        lines.append(c("    (none detected by name heuristic)", YELLOW))
    lines.append("")

    # ── Temporal column ───────────────────────────────────────────────────────
    lines.append(c("  ► TEMPORAL COLUMN  (used for train/val/test splits)", BOLD, GREEN))
    lines.append("")
    for col in cats["datetime"]:
        parsed = pd.to_datetime(df[col], errors="coerce")
        valid  = parsed.dropna()
        if len(valid):
            lines.append(f"    {c(col, BOLD)}  {valid.min()} → {valid.max()}  "
                         f"(span={int((valid.max()-valid.min()).days)} days)")
    if not cats["datetime"]:
        lines.append(c("    (no datetime columns detected — check 'id_like' for epoch timestamps)", YELLOW))
    lines.append("")

    # ── Feature candidates ────────────────────────────────────────────────────
    lines.append(c("  ► FEATURE CANDIDATES  (suitable as node/edge attributes)", BOLD, GREEN))
    lines.append("")
    used = set(node_cols + label_cols + cats["datetime"] + cats["constant"])
    feat_cols = (
        [c for c in cats["numeric"]     if c not in used] +
        [c for c in cats["categorical"] if c not in used] +
        [c for c in cats["boolean"]     if c not in used]
    )
    for col in feat_cols:
        kind = "numeric" if col in cats["numeric"] else (
               "categorical" if col in cats["categorical"] else "boolean")
        icon = {"numeric": "📊", "categorical": "🏷 ", "boolean": "🔲"}.get(kind, "  ")
        lines.append(f"    {icon} {c(col, BOLD)}  [{kind}]")

    return section("5. CANDIDATE GRAPH ELEMENTS", lines)


def sketch_section(df: pd.DataFrame, cats: dict[str, list[str]]) -> list[str]:
    """Print a suggested PyG HeteroData construction sketch."""
    node_cols = cats["id_like"]
    datetime_cols = cats["datetime"]
    label_cols = [c for c in cats["boolean"] if col_name_matches(c, LABEL_KEYWORDS)]

    account_cols = [c for c in node_cols if col_name_matches(c, ["account", "acct", "user", "customer"])]
    txn_cols     = [c for c in node_cols if col_name_matches(c, ["transaction", "txn", "order", "transfer"])]

    # Fall back to first two id columns
    if not account_cols and len(node_cols) >= 1:
        account_cols = [node_cols[0]]
    if not txn_cols and len(node_cols) >= 2:
        txn_cols = [node_cols[1]]

    a = account_cols[0] if account_cols else "ENTITY_A"
    b = txn_cols[0]     if txn_cols     else "ENTITY_B"
    t = datetime_cols[0] if datetime_cols else None
    y = label_cols[0]   if label_cols    else "LABEL_COL"

    sketch = f"""\
# ── Suggested HeteroData sketch (adapt to your actual schema) ──────────────

from torch_geometric.data import HeteroData
import torch

# After building account_to_id and txn_to_id lookup dicts:
data = HeteroData()

# Node features
data['account'].x    = account_feat_tensor        # (num_accounts, acct_feat_dim)
data['{b}'].x    = txn_feat_tensor            # (num_txns, txn_feat_dim)

# Labels & masks on the target node type
data['{b}'].y          = torch.tensor(df['{y}'].values, dtype=torch.long)
data['{b}'].train_mask = torch.tensor(train_mask, dtype=torch.bool)
data['{b}'].val_mask   = torch.tensor(val_mask,   dtype=torch.bool)
data['{b}'].test_mask  = torch.tensor(test_mask,  dtype=torch.bool)

# Edges  — adjust relation names to match domain semantics
src = df['{a}'].map(account_to_id).values
dst = df['{b}'].map(txn_to_id).values
data[('account', 'initiates', '{b}')].edge_index = torch.tensor([src, dst])
data[('{b}', 'initiated_by', 'account')].edge_index = torch.tensor([dst, src])  # reverse
"""

    if t:
        sketch += f"\n# Temporal split: use df['{t}'] to define train/val/test cutoff dates\n"

    lines = [c(indent(sketch, "  "), DIM)]
    return section("6. SUGGESTED PyG HeteroData SKETCH", lines)


# ─── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="EDA for heterogeneous graph modelling."
    )
    parser.add_argument("path",              help="Path to CSV / Parquet / Excel / JSON")
    parser.add_argument("--sample", "-n",   type=int, default=None,
                        help="Row sample cap (default: all rows)")
    parser.add_argument("--sep",            default=",",
                        help="CSV separator (default: ',')")
    parser.add_argument("--out", "-o",      default=None,
                        help="Write report to this file (markdown / txt)")
    parser.add_argument("--no-corr",        action="store_true",
                        help="Skip correlation matrix (slow on wide datasets)")
    args = parser.parse_args()

    df   = load(args.path, args.sample, args.sep)
    cats = classify_columns(df)

    all_lines: list[str] = []
    all_lines += shape_section(df)
    all_lines += type_audit_section(cats)
    all_lines += per_column_section(df, cats)
    if not args.no_corr:
        all_lines += correlation_section(df, cats)
    all_lines += graph_candidates_section(df, cats)
    all_lines += sketch_section(df, cats)
    all_lines.append("")

    # Print to terminal (with colour)
    for line in all_lines:
        print(line)

    # Optionally write to file (strip ANSI)
    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            for line in all_lines:
                f.write(strip_ansi(line) + "\n")
        print(c(f"\n✓ Report written to {out_path}", BOLD, GREEN))


if __name__ == "__main__":
    main()
