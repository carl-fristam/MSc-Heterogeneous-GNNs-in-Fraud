# Phase 1: Graph Representation - Complete! ✅

## What Was Built

Successfully refactored the graph builder into a clean, modular architecture:

### **Modular Structure**

```
src/
├── build_graph.py          # Main entry point
└── graph/                  # Graph package
    ├── __init__.py        # Package exports
    ├── data_loader.py     # CSV loading & account mapping
    ├── builder.py         # Graph construction
    ├── statistics.py      # Graph metrics computation
    ├── visualization.py   # Plotting & visualization
    └── io.py             # Save/load utilities
```

### **Graph Statistics**

Your AML transaction network:
- **515,088 accounts** (nodes)
- **1,015,736 transaction relationships** (edges)
- **5,078,345 total transactions** processed
- **72.2%** of accounts in largest connected component
- **0.5%** of edges involve money laundering
- Average **5 transactions** per account pair

### **Generated Outputs**

Located in `outputs/`:
- `transaction_graph.gpickle` - NetworkX graph (Python)
- `transaction_graph.graphml` - Graph for Gephi/Cytoscape
- `account_to_id.pkl` - Account ID mappings
- `subgraph.png` - Network visualization (50 nodes)
- `degree_distribution.png` - Degree distribution plots

## How to Use

### **Load and Explore**

```python
from graph import load_graph, load_account_mapping

# Load the graph
G = load_graph('outputs/transaction_graph.gpickle')

# Load account mappings
account_to_id = load_account_mapping('outputs/account_to_id.pkl')

# Explore
print(f"Nodes: {G.number_of_nodes()}")
print(f"Edges: {G.number_of_edges()}")

# Get node attributes
node_id = 0
print(G.nodes[node_id])  # bank_name, entity_id, etc.

# Get edge attributes
for u, v in list(G.edges())[:5]:
    print(f"Edge {u}->{v}: {G[u][v]}")  # weight, total_amount, laundering_ratio
```

### **Rebuild from Scratch**

```bash
python3 src/build_graph.py
```

### **Import as Module**

```python
from graph import (
    load_data,
    build_transaction_graph,
    compute_all_statistics,
    visualize_subgraph
)

# Custom pipeline
trans_df, accounts_df = load_data()
# ... process as needed
```

## Next Steps

### **Phase 2: GraphSAGE Baseline**

Ready to implement a simple GNN baseline:
1. Convert NetworkX graph to PyTorch Geometric format
2. Define node features (transaction statistics, bank info)
3. Implement GraphSAGE model
4. Train for binary classification (laundering detection)
5. Evaluate performance metrics

### **Phase 3: HGMAE (Heterogeneous)**

After baseline:
1. Create heterogeneous graph (Account, Bank, Entity nodes)
2. Generate HGMAE data format
3. Run pre-training with masked autoencoding
4. Compare with GraphSAGE baseline

## Key Insights

- **Sparse network**: Density of 0.000004 indicates highly sparse connections
- **Power-law distribution**: Max out-degree of 14,230 suggests hub accounts
- **Low laundering signal**: Only 0.5% of edges flagged - class imbalance challenge
- **Large connected component**: 72% connectivity good for message passing GNNs
