# Transaction Monitoring with Heterogeneous GNNs - Implementation Roadmap

## Executive Summary

Transform the current account-level fraud detection into a **transaction monitoring system** using heterogeneous graph neural networks. This approach better aligns with real-world AML systems where the goal is to flag suspicious transactions for review, not just identify bad accounts.

---

## Phase 1: Heterogeneous Graph Construction

### Objective
Convert SAML-D dataset from homogeneous (accounts only) to heterogeneous graph with multiple node and edge types.

### Node Types to Model

1. **Account nodes**
   - Features: Transaction statistics (in/out degree, volumes, velocities)
   - ~554k nodes in full dataset

2. **Bank nodes** (optional but valuable)
   - One node per bank location (UK, UAE, Spain, etc.)
   - Features: Total transaction volume, number of accounts, country risk score
   - Enables modeling: "Account → Bank → Account" paths

3. **Currency nodes** (optional)
   - UK pounds, USD, Euro, Dirham, etc.
   - Features: Exchange rate volatility, usage frequency
   - Enables: Currency conversion pattern detection

4. **Time window nodes** (advanced)
   - Hourly/daily buckets
   - Enables: Temporal pattern detection (burst activity)

### Edge Types to Model

**Priority 1 (Start here):**
- `account --[Cash_Deposit]--> account`
- `account --[Wire_Transfer]--> account`
- `account --[ACH]--> account`
- `account --[Cheque]--> account`
- `account --[Credit_Card]--> account`
- `account --[Cross_Border]--> account`

**Priority 2 (Add later):**
- `account --[located_at]--> bank`
- `account --[sends_currency]--> currency`
- `account --[active_in]--> time_window`

### Edge Features (Critical!)

Each transaction edge should have:
- **Amount** (normalized)
- **Timestamp** (hour of day, day of week, time since account creation)
- **Currency pair** (one-hot or embedding)
- **Is cross-border** (boolean)
- **Payment type** (one-hot)
- **Velocity features**:
  - Transactions in last 1 hour, 24 hours, 7 days
  - Amount percentile for this account
  - Deviation from account's typical amount

### Implementation

**File**: `src/hetero_data.py`

```python
def load_hetero_saml_data(data_path='data/SAML-D.csv'):
    """
    Convert SAML-D to PyG HeteroData object.

    Returns:
        data: HeteroData with multiple node/edge types
        edge_to_transaction_id: Mapping for evaluation
    """
    pass
```

**Key functions:**
- `_create_edge_types()` - Group transactions by Payment_type
- `_create_edge_features()` - Temporal and structural features
- `_create_bank_nodes()` - Optional bank location nodes
- `_add_metapaths()` - Define important metapaths for attention

---

## Phase 2: Edge Classification Model

### Objective
Build a model that predicts `Is_laundering` for each **transaction** (edge), not account.

### Model Architecture Options

#### Option A: Simple Heterogeneous GNN (Start here)
```
1. Node embeddings per type (account, bank, etc.)
2. Heterogeneous message passing (HeteroConv)
3. Edge classifier head
```

**Pros**: Easier to implement, faster training
**Cons**: Less expressive than transformers

#### Option B: Heterogeneous Graph Transformer (HGT)
```
1. Type-specific attention heads
2. Multi-head attention over different edge types
3. Edge prediction with learned attention weights
```

**Pros**: State-of-the-art, learns which attributes matter
**Cons**: More complex, slower training, needs more data

#### Option C: HGMAE Pre-training + Fine-tuning
```
1. Pre-train with masked edge/attribute prediction
2. Fine-tune for transaction classification
3. Transfer learning from unlabeled patterns
```

**Pros**: Best performance, handles imbalance well
**Cons**: Two-stage training, most complex

### Edge Classification Head

```python
class EdgeClassifier(nn.Module):
    """
    Predict laundering probability for each edge.

    Input:
        - Source node embedding
        - Target node embedding
        - Edge features
        - Attention weights (optional)

    Output:
        - Laundering probability [0, 1]
    """
```

### Implementation Strategy

**Start Simple → Add Complexity**

1. **Week 1**: Homogeneous edge classification baseline
2. **Week 2**: Add edge types (heterogeneous)
3. **Week 3**: Add attention mechanisms
4. **Week 4**: Add bank/currency nodes
5. **Week 5**: HGMAE pre-training (optional)

**Files:**
- `src/models/hetero_gnn.py` - HeteroConv model
- `src/models/hgt_model.py` - Graph transformer
- `src/models/edge_classifier.py` - Edge prediction head
- `src/train_edge_classification.py` - Training script

---

## Phase 3: Transaction Monitoring Evaluation

### Why Traditional Metrics Fail

❌ **Accuracy**: Useless with 99.86% negative class
❌ **F1 Score**: Doesn't reflect operational constraints
❌ **Recall/Precision alone**: Ignores ranking quality

### Proper Evaluation Metrics

#### 1. **Precision@k** (Primary metric)
"Of the top-k flagged transactions, what % are actually laundering?"

```python
def precision_at_k(scores, labels, k=[10, 50, 100, 500, 1000]):
    """
    Scores: Model's suspicion scores for each transaction
    Labels: Ground truth (1=laundering, 0=clean)
    k: Number of alerts to generate

    Returns: Precision for each k value
    """
```

**Why this matters**: You can only review ~100-1000 transactions per day. If Precision@100 = 0.15, then 15 of your daily alerts are real laundering (85 false positives).

#### 2. **Recall@k**
"Of all laundering transactions, what % are in your top-k?"

Tradeoff with Precision@k. Goal: Maximize recall while keeping precision acceptable (≥10%).

#### 3. **Alert Rate**
"What % of all transactions get flagged?"

Operational constraint: Most banks can only review 0.1-1% of transactions.

#### 4. **Detection Curves**

- **Precision-Recall Curve**: At various thresholds
- **Alert-Rate vs. Recall**: "To catch 80% of laundering, I need to review 2% of transactions"
- **Lift Curve**: Model's improvement over random

#### 5. **Temporal Evaluation**

Split by time, not randomly:
- Train: Months 1-6
- Val: Month 7
- Test: Month 8

**Why**: Patterns change over time. Model must generalize to future.

#### 6. **Per-Pattern Performance**

SAML-D has labeled patterns (Fan-in, Fan-out, Cycles, etc.). Report:
- Precision@100 for each pattern type
- Which patterns are hard to detect?

### Implementation

**File**: `src/evaluation/transaction_monitoring.py`

```python
def evaluate_transaction_monitoring(model, data, k_values=[10, 50, 100, 500, 1000]):
    """
    Comprehensive transaction monitoring evaluation.

    Returns:
        - Precision@k for each k
        - Recall@k for each k
        - Alert rate curves
        - Per-pattern breakdowns
        - Confusion matrices at different thresholds
    """
```

**File**: `src/evaluation/visualization.py`

```python
def plot_detection_curves(results):
    """
    Visualize:
    - Precision-Recall curve
    - Alert rate vs. Recall
    - Lift chart
    - Per-pattern performance heatmap
    """
```

---

## Phase 4: Attention Analysis

### Why Attention Matters

Attention weights show **which attributes/edges the model focuses on** for flagging transactions. This provides:

1. **Explainability**: "This transaction was flagged because it's a large cross-border wire following 10 rapid cash deposits"
2. **Model debugging**: Are we learning real patterns or shortcuts?
3. **Feature engineering**: Discover new suspicious patterns
4. **Compliance**: Regulators want explanations

### What to Analyze

#### 1. **Edge Type Attention**
"Which transaction types are most important for detection?"

```python
# For each flagged transaction, extract attention weights
attention_weights = {
    'Cash_Deposit': 0.45,
    'Wire_Transfer': 0.30,
    'Cross_Border': 0.20,
    'ACH': 0.05
}
```

#### 2. **Feature Attention**
"Which edge features matter most?"

- Amount vs. timestamp vs. currency?
- Velocity features vs. static features?

#### 3. **Metapath Attention**
"Which multi-hop patterns are suspicious?"

Example suspicious metapaths:
- Account A → [many cash deposits] → Account B → [wire] → Account C
- Account A → [cross-border] → Bank X → Account B

#### 4. **Temporal Attention**
"Does the model focus on recent or historical transactions?"

### Implementation

**File**: `src/analysis/attention_analysis.py`

```python
def extract_attention_weights(model, transaction_id):
    """
    For a specific flagged transaction, extract:
    - Edge type attention
    - Feature attention
    - Neighbor contribution
    """

def visualize_attention_graph(transaction_id, attention_weights):
    """
    Draw subgraph around flagged transaction with:
    - Edge thickness = attention weight
    - Node color = laundering probability
    - Annotations = key features
    """

def aggregate_attention_patterns():
    """
    Across all flagged transactions:
    - Top-k most important edge types
    - Top-k most important features
    - Common suspicious patterns
    """
```

---

## Phase 5: Advanced Techniques

### 5.1 Temporal Modeling

Add explicit time modeling:
- **Recurrent GNN**: Process transactions in chronological order
- **Temporal Point Process**: Model transaction arrival times
- **Sliding Window**: Graph changes over time

### 5.2 Contrastive Learning

Learn embeddings where:
- Laundering transactions are close in embedding space
- Clean transactions are far
- Improves recall for rare patterns

### 5.3 Active Learning

Model suggests which transactions to label next:
- Query transactions with high uncertainty
- Iteratively improve model with less labeling effort

### 5.4 Anomaly Detection

Complement supervised learning with:
- Autoencoder on transaction embeddings
- Flag transactions different from training distribution
- Catch zero-day laundering patterns

---

## Success Metrics

### Technical Metrics
- **Precision@100 ≥ 0.15** (15% of top-100 flags are real)
- **Recall@1000 ≥ 0.50** (catch 50% of laundering with 1000 daily alerts)
- **AUC ≥ 0.95**

### Operational Metrics
- **Alert rate ≤ 1%** (review at most 1% of transactions)
- **Detection delay ≤ 1 day** (flag suspicious transactions quickly)
- **False positive rate ≤ 1%**

### Research Metrics
- Beat homogeneous baseline by ≥20% on Precision@100
- Attention weights align with known laundering patterns
- Model generalizes to future time periods

---

## Implementation Timeline

### Sprint 1 (Week 1-2): Foundation
- [ ] Convert SAML-D to HeteroData with edge types
- [ ] Implement edge classification baseline (homogeneous)
- [ ] Build Precision@k evaluation pipeline

### Sprint 2 (Week 3-4): Heterogeneous GNN
- [ ] HeteroConv model with multiple edge types
- [ ] Add edge features (velocity, temporal)
- [ ] Improve Precision@100 vs. baseline

### Sprint 3 (Week 5-6): Attention Mechanisms
- [ ] Implement attention-based edge classifier
- [ ] Extract and visualize attention weights
- [ ] Analyze which patterns the model learns

### Sprint 4 (Week 7-8): Advanced Models
- [ ] Add bank/currency nodes
- [ ] Heterogeneous Graph Transformer (HGT)
- [ ] HGMAE pre-training (optional)

### Sprint 5 (Week 9-10): Production-Ready
- [ ] Temporal evaluation (time-split)
- [ ] Per-pattern performance analysis
- [ ] Documentation and deployment prep

---

## References & Resources

### Papers
1. **HGT**: "Heterogeneous Graph Transformer" (WWW 2020)
2. **HGMAE**: "Heterogeneous Graph Masked Autoencoders" (AAAI 2023)
3. **Edge Classification**: "Inductive Representation Learning on Large Graphs" (NeurIPS 2017)
4. **AML Detection**: "Anti-Money Laundering in Bitcoin: Experimenting with Graph Convolutional Networks for Financial Forensics" (KDD 2019)

### Code Examples
- PyG HeteroData: https://pytorch-geometric.readthedocs.io/en/latest/tutorial/heterogeneous.html
- HGT Implementation: `torch_geometric.nn.HGTConv`
- Edge Classification: Custom implementation needed

### Datasets
- SAML-D: Current dataset (transaction-level labels)
- Elliptic Bitcoin: Comparison benchmark
- IBM AML Dataset: Alternative if needed

---

## Key Decisions to Make

1. **Which edge types to include first?**
   - Start with Payment_type only?
   - Add bank/currency nodes immediately?

2. **Edge features vs. node features?**
   - Heavy edge features (velocity, temporal)
   - Light node features (aggregated stats)

3. **Model complexity tradeoff?**
   - Start simple (HeteroConv) or jump to HGT?
   - Pre-training worth the complexity?

4. **Evaluation focus?**
   - Optimize for Precision@100 (fewer alerts)?
   - Optimize for Recall@1000 (catch more laundering)?

5. **Temporal modeling?**
   - Static graph or temporal evolution?
   - Batch processing or online learning?

---

## Next Steps

1. **Review this roadmap** - Any changes/priorities?
2. **Start Phase 1** - Build heterogeneous graph constructor
3. **Set baseline** - Simple edge classification model
4. **Iterate** - Add complexity incrementally

Questions? Ready to start Phase 1?
