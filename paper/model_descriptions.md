# Technical Model Descriptions

## Data Representation

### Transaction Features (Edge Features)

All models that consume transaction-level features use the same engineered feature vector per transaction:

| Feature | Dim | Description |
|---------|-----|-------------|
| `log_base_value` | 1 | log1p of transaction amount (BASEVALUE) in reporting currency, z-score normalised |
| `channel_ohe` | K | One-hot encoding of payment channel (e.g. mobile, internet, branch). Vocabulary fitted on training data; unseen categories mapped to all-zeros |
| `submethod_ohe` | K | One-hot encoding of payment sub-method (realTime, bankGiro, plusGiro, futurePayment, salary, etc.). Vocabulary from training |
| `clearing_express` | 1 | Binary indicator: 1 if PAYMENTCLEARING != "default" (express/instant clearing reduces the fraud interception window) |
| `international_flag` | 1 | Binary indicator for cross-border transactions |
| `branch_tbe_ohe` | K | One-hot encoding of account branch (ACCOUNTBRANCH_TBE). Vocabulary from training |
| `time_encoding` | 4 | Cyclical sin/cos encoding of hour-of-day and day-of-week: [sin(2πh/24), cos(2πh/24), sin(2πd/7), cos(2πd/7)] |

### Node Features (Heterogeneous Models)

All node features are aggregated from **training data only** to prevent temporal leakage.

**InternalAccount** (sender accounts):

| Feature | Dim | Description |
|---------|-----|-------------|
| `out_degree` | 1 | Number of transactions sent during training period |
| `amount_stats` | 3 | log1p of mean, std, and total amount sent |
| `counterparty_diversity` | 1 | Number of unique receiver accounts |
| `channel_diversity` | 1 | Number of unique payment channels used |
| `time_behavior` | 2 | Fraction of transactions at night (22:00–06:00) and on weekends (Sat/Sun) |

**ExternalAccount** (receiver accounts, non-onus only):

| Feature | Dim | Description |
|---------|-----|-------------|
| `in_degree` | 1 | Number of transactions received during training |
| `received_amount_stats` | 2 | log1p of mean and std of received amounts |
| `sender_diversity` | 1 | Number of unique internal accounts sending to this account |
| `sender_bank_diversity` | 1 | Number of unique sender banks (COUNTERAGENTID) — high diversity signals potential money mule |

All node feature matrices are z-score normalised column-wise after construction.

### Graph Topologies

**V1 — Onus vs External** (2 edge types):
- `InternalAccount →[onus_transfer]→ InternalAccount`: intra-bank transfers (TRANSACTIONONUS = True)
- `InternalAccount →[external_transfer]→ ExternalAccount`: inter-bank transfers (TRANSACTIONONUS = False)

**V2 — Payment Rail Typed** (6 edge types):
- `InternalAccount →[onus_transfer]→ InternalAccount`
- `InternalAccount →[ext_realtime]→ ExternalAccount`: real-time external payments
- `InternalAccount →[ext_giro]→ ExternalAccount`: bankGiro / plusGiro payments
- `InternalAccount →[ext_future]→ ExternalAccount`: scheduled future payments
- `InternalAccount →[ext_salary]→ ExternalAccount`: salary payments
- `InternalAccount →[ext_other]→ ExternalAccount`: remaining external payments

**TXN_V1 — Transactions as Nodes** (3 edge types, for node classification):
- `InternalAccount →[sends]→ Transaction`
- `Transaction →[received_by_internal]→ InternalAccount`
- `Transaction →[received_by_external]→ ExternalAccount`

### Data Split

Temporal split reflecting deployment realism:
- **Train**: all transactions before 2024-11-20
- **Validation**: 2024-11-20 to 2024-12-10
- **Test**: 2024-12-10 to 2024-12-31

### Class Imbalance Handling

All supervised models use inverse-frequency class weighting. For BCE-based models, `pos_weight = n_negative / n_positive` is applied to `BCEWithLogitsLoss`. XGBoost uses the equivalent `scale_pos_weight` parameter.

---

## L0: Tabular Baselines

### Logistic Regression

A linear model trained on the transaction feature vector with L2 regularisation. Serves as the simplest baseline to establish whether any signal exists in the transaction-level features alone.

- **Input**: Transaction feature vector (edge features described above)
- **Architecture**: Single linear layer with sigmoid activation
- **Optimisation**: SAGA solver, `class_weight="balanced"`, max 1000 iterations
- **Output**: Fraud probability per transaction

### XGBoost (Tabular)

Gradient-boosted decision tree ensemble on transaction features. Represents the strongest non-graph baseline — if XGBoost on flat features already performs well, the marginal value of graph structure is limited.

- **Input**: Transaction feature vector
- **Architecture**: Ensemble of 300 decision trees (max depth 6), trained via histogram-based gradient boosting
- **Hyperparameters**: learning rate 0.1, `scale_pos_weight` for class imbalance, early stopping on validation AUPRC (patience 20)
- **Output**: Fraud probability per transaction

---

## L1: Graph Features + XGBoost

Hand-crafted graph-structural features are extracted from the transaction graph and concatenated with the transaction feature vector. The augmented feature set is fed to XGBoost. This isolates the value of graph structure without neural message passing, following the approach of Eddin et al. (2021).

### Graph Feature Extraction

Four structural features are computed per sender account from training data:

| Feature | Description |
|---------|-------------|
| log1p(out_degree) | Number of unique receivers the account sent to |
| log1p(in_degree) | Number of unique senders that sent to this account |
| log1p(tx_count) | Total number of transactions sent |
| log1p(bank_diversity) | Number of unique counterparty banks (COUNTERAGENTID) |

Each transaction inherits the graph features of its sender account, producing a per-transaction feature vector of dimension F_txn + 4.

### Classifier

Same XGBoost configuration as L0. The comparison L0 → L1 quantifies the additive value of graph-structural information when consumed by a non-neural model.

---

## L2: Homogeneous GNN

### Graph Construction

The heterogeneous account graph is collapsed into a single-type graph. All accounts (internal and external) become nodes of one type; all transactions become edges of one type. Node features are the account-level aggregations (internal or external features as appropriate, zero-padded to a common dimension). Edge features are the transaction feature vector.

Two tasks are supported:
- **Node classification**: Each node receives a fraud label (1 if the account sent any fraudulent transaction in that split, 0 otherwise). The model classifies nodes.
- **Edge classification**: Each edge carries the fraud label of the corresponding transaction. The model classifies edges.

### GCN (Graph Convolutional Network)

Kipf & Welling (2017). Spectral-domain convolution approximated by first-order Chebyshev polynomials.

**Message passing rule** (layer $l$):

$$H^{(l+1)} = \sigma\!\left(\tilde{D}^{-1/2}\,\tilde{A}\,\tilde{D}^{-1/2}\,H^{(l)}\,W^{(l)}\right)$$

where $\tilde{A} = A + I$ is the adjacency with self-loops and $\tilde{D}$ its degree matrix. Each node's updated representation is a weighted mean of its neighbours' features (including itself), transformed by a learnable weight matrix $W^{(l)}$.

- **Architecture**: $L$ GCNConv layers, each followed by LayerNorm, ReLU, and dropout
- **Node classification**: Final node embeddings → linear layer → sigmoid
- **Edge classification**: For each edge $(u, v)$, concatenate $[h_u \| h_v \| e_{uv}]$ (source embedding, destination embedding, edge features) → 2-layer MLP → sigmoid
- **Defaults**: hidden_dim=64, num_layers=2, dropout=0.3

### GraphSAGE (Sample and Aggregate)

Hamilton et al. (2017). Inductive node embedding via neighbourhood sampling and aggregation.

**Message passing rule** (layer $l$):

$$h_v^{(l+1)} = \sigma\!\left(W^{(l)} \cdot \text{CONCAT}\!\left(h_v^{(l)},\; \text{AGG}\!\left(\{h_u^{(l)} : u \in \mathcal{N}(v)\}\right)\right)\right)$$

where AGG is a mean aggregator over the neighbourhood $\mathcal{N}(v)$. Unlike GCN, SAGE concatenates the node's own embedding with the aggregated neighbourhood, preserving the node's identity signal.

- **Architecture**: $L$ SAGEConv layers with LayerNorm, ReLU, dropout
- **Classification heads**: Same as GCN (linear for node, MLP with edge features for edge)
- **Defaults**: hidden_dim=64, num_layers=2, dropout=0.3

### TransE

Bordes et al. (2013). Knowledge graph embedding where relationships are modelled as translations in embedding space.

**Scoring function**:

$$\text{score}(h, r, t) = -\|h + r - t\|_2$$

where $h$, $t$ are learned entity (account) embeddings and $r$ is a learned relation embedding. For positive (real) edges, $h + r$ should be close to $t$; the negated L2 distance serves as a plausibility score. Higher scores indicate more plausible triples.

- **Architecture**: Entity embedding table (num_nodes × 64) + relation embedding table (1 × 64). The TransE score is concatenated with the transaction edge feature vector and passed through a 2-layer MLP classifier
- **Training**: BCEWithLogitsLoss on fraud labels (not the standard contrastive KGE loss), enabling direct optimisation for the downstream task
- **Defaults**: embedding_dim=64, dropout=0.3

### DistMult

Yang et al. (2015). Bilinear knowledge graph embedding using element-wise products.

**Scoring function**:

$$\text{score}(h, r, t) = \langle h, r, t \rangle = \sum_i h_i \cdot r_i \cdot t_i$$

The bilinear formulation captures pairwise interactions between entity dimensions modulated by the relation vector. Unlike TransE, DistMult naturally models symmetric relations.

- **Architecture**: Same structure as TransE — entity and relation embedding tables, score concatenated with edge features, fed to 2-layer MLP
- **Training**: BCEWithLogitsLoss on fraud labels
- **Defaults**: embedding_dim=64, dropout=0.3

---

## L3: Heterogeneous GNN

### HGT (Heterogeneous Graph Transformer)

Hu et al. (2020). Extends the Transformer attention mechanism to heterogeneous graphs with type-specific projections.

**Per-type input projection**: Each node type has a dedicated linear layer projecting its features to a shared hidden dimension $d$:

$$h_v^{(0)} = \text{ReLU}(W_{\tau(v)}\,x_v + b_{\tau(v)})$$

where $\tau(v)$ is the type of node $v$.

**Heterogeneous multi-head attention** (layer $l$): For each target node $t$ with neighbour $s$ connected by edge type $\phi(e)$:

$$\text{Attention}(s, e, t) = \text{softmax}_{s}\!\left(\frac{(K^{(l)}_{\tau(s)}\,h_s^{(l)})^\top\;\mu_{\phi(e)}\;(Q^{(l)}_{\tau(t)}\,h_t^{(l)})}{\sqrt{d/H}}\right)$$

$$\text{Message}(s, e, t) = V^{(l)}_{\tau(s)}\,h_s^{(l)}$$

$$\tilde{h}_t^{(l+1)} = \bigoplus_{\phi(e)} \sum_{s \in \mathcal{N}_\phi(t)} \text{Attention}(s, e, t) \cdot \text{Message}(s, e, t)$$

where $K$, $Q$, $V$ are type-specific key, query, value projection matrices, $\mu_{\phi(e)}$ is a per-edge-type attention weight, $H$ is the number of heads, and $\bigoplus$ denotes aggregation across edge types.

Each layer is followed by LayerNorm, ReLU, and dropout.

- **Node classification** (TXN_V1 topology): Final transaction node embeddings → linear layer → sigmoid
- **Edge classification** (V1/V2 topology): Final node embeddings for all labelled edge types → for each edge $(u, v)$, concatenate $[h_u \| h_v]$ → 2-layer MLP → sigmoid
- **Defaults**: hidden_dim=64, num_heads=4, num_layers=2, dropout=0.3

### HMPNN (Heterogeneous Message Passing Neural Network)

Johannessen & Jullum (2023). Uses NNConv to incorporate edge features directly into the message function.

**Edge-conditioned message passing**: For each edge type ending at target node type $\tau$, a neural network $\text{NN}_\phi$ maps edge features to a weight matrix that transforms the source node's embedding:

$$m_{s \to t}^{(l)} = \text{NN}_\phi(e_{st}) \cdot h_s^{(l)}$$

where $\text{NN}_\phi: \mathbb{R}^{d_e} \to \mathbb{R}^{d_{\text{in}} \times d_{\text{msg}}}$ is a 2-layer MLP that produces a transformation matrix from the edge feature vector $e_{st}$. The edge feature neural network has architecture: Linear($d_e$, 32) → ReLU → Linear(32, $d_{\text{in}} \times d_{\text{msg}}$).

**Multi-relation aggregation**: Messages from all edge types targeting the same node type are concatenated and projected:

$$h_t^{(l+1)} = \sigma\!\left(W^{(l)}\,\text{CONCAT}\!\left(\sigma(m_{\phi_1}),\;\sigma(m_{\phi_2}),\;\ldots\right)\right)$$

where $\sigma$ is the sigmoid activation and each $m_{\phi_i}$ is the mean-aggregated message from edge type $\phi_i$. The sigmoid activation on messages bounds representations to $[0, 1]$.

**Multi-layer architecture**:
- Layer 1 updates all node types in parallel, each receiving messages from their incoming edge types
- Intermediate layers (if num_layers ≥ 3) repeat this for all node types
- Final layer updates only the target node type

- **Node classification**: Final target node embedding (1-dim output) → sigmoid
- **Edge classification**: Final node embeddings → for each edge, concatenate $[h_u \| h_v]$ → 2-layer MLP → sigmoid
- **Defaults**: hidden_dim=16, message_dim=8, num_layers=2

---

## L4: Self-Supervised Graph Pretraining

Both L4 models use a two-stage training procedure:
1. **Stage 1 (Pretraining)**: Train the encoder with a self-supervised objective on the graph structure and node features, using no fraud labels
2. **Stage 2 (Classification)**: Freeze the pretrained encoder (linear probe) or fine-tune it, and train an MLP classifier for edge-level fraud classification

The hypothesis is that self-supervised objectives force the encoder to learn richer structural representations than what BCE loss alone can discover from the highly imbalanced label distribution (~0.14% positive rate).

### Training Protocol

**Stage 1**: AdamW optimiser with cosine annealing learning rate schedule. Early stopping on pretraining loss (patience 20). Gradient clipping at max norm 1.0.

**Stage 2**: The pretrained encoder is wrapped with an edge classifier. Two modes:
- **Frozen probe** (default): Encoder parameters are frozen; only the MLP classifier is trained. This evaluates the quality of the learned representations in isolation.
- **Fine-tune**: All parameters (encoder + classifier) are updated. This allows the encoder to adapt its representations to the downstream task.

The stage 2 classifier and training loop are identical to the L3 supervised pipeline (same `Trainer` class, same BCEWithLogitsLoss with class weighting, same early stopping on validation AUPRC).

### HGMAE (Heterogeneous Graph Masked Autoencoder)

Adapted from Tian et al. (2023). A masked autoencoder for heterogeneous graphs with two self-supervised objectives.

**Encoder**: Identical architecture to the L3 HGT encoder — per-type linear input projections followed by $L$ HGTConv layers with LayerNorm, ReLU, and dropout. This reuse ensures the pretraining and supervised baselines use the same representational capacity, isolating the effect of the training objective.

**Objective 1 — Attribute Masking and Restoration**:

For each node type, a fraction $p_{\text{feat}}$ of nodes are selected uniformly at random. Their features are replaced with a learnable mask token $m_\tau \in \mathbb{R}^{d_\tau}$ (one per node type):

$$\tilde{x}_v = \begin{cases} m_{\tau(v)} & \text{if } v \in \mathcal{M} \\ x_v & \text{otherwise} \end{cases}$$

The corrupted features are passed through the encoder, then a per-type MLP decoder (Linear → ReLU → Linear) reconstructs the original features. The loss is computed only on masked nodes using **Scaled Cosine Error (SCE)**:

$$\mathcal{L}_{\text{attr}} = \frac{1}{|\mathcal{M}|}\sum_{v \in \mathcal{M}} \left(1 - \frac{\hat{x}_v \cdot x_v}{\|\hat{x}_v\| \cdot \|x_v\|}\right)^\alpha$$

where $\hat{x}_v$ is the reconstructed feature vector and $\alpha=3$ controls the sharpness of the loss. SCE is preferred over MSE because it is scale-invariant and focuses on directional similarity.

**Objective 2 — Edge Reconstruction**:

The encoder processes the full (unmasked) graph and produces node embeddings. For each edge type, positive edges are scored via inner product of source and destination embeddings:

$$s^+_{uv} = h_u^\top h_v \quad \text{for } (u, v) \in \mathcal{E}$$

Negative edges are sampled by corrupting destination nodes (5 negatives per positive):

$$s^-_{uj} = h_u^\top h_j \quad \text{where } j \sim \text{Uniform}(\mathcal{V}_{\text{dst}})$$

The loss is binary cross-entropy on the positive and negative scores:

$$\mathcal{L}_{\text{edge}} = -\frac{1}{|\mathcal{E}|}\sum \log\sigma(s^+) - \frac{1}{|\mathcal{E}^-|}\sum \log(1 - \sigma(s^-))$$

**Combined loss**:

$$\mathcal{L} = \mathcal{L}_{\text{attr}} + \lambda\,\mathcal{L}_{\text{edge}}$$

where $\lambda$ (`edge_recon_weight`) balances the two objectives (default 1.0).

**Decoder**: Per-type 2-layer MLP projecting from hidden dimension back to input feature dimension. Used only during pretraining; discarded at stage 2.

- **Defaults**: hidden_dim=64, num_heads=4, num_layers=2, dropout=0.3, feat_mask_rate=0.5, edge_mask_rate=0.3, alpha=3, edge_recon_weight=1.0, num_neg_samples=5

### LaundroGraph (Bipartite Link Prediction Encoder)

A simpler self-supervised model using link prediction on the heterogeneous bipartite account graph as the pretext task.

**Encoder**: Per-type linear input projections followed by $L$ HeteroConv layers. Each HeteroConv wraps a per-edge-type SAGEConv, aggregated across edge types via summation:

$$h_v^{(l+1)} = \sum_{\phi \in \mathcal{R}(v)} \text{SAGEConv}_\phi\!\left(\{h_u^{(l)} : u \in \mathcal{N}_\phi(v)\},\; h_v^{(l)}\right)$$

Each SAGEConv applies the standard SAGE aggregation (concatenation of self-embedding with mean-aggregated neighbourhood) independently per edge type, and the results are summed. Each layer is followed by per-type LayerNorm, ReLU, and dropout.

**Pretext Task — Link Prediction**:

For each edge type, the model scores observed edges (positives) and randomly sampled non-edges (negatives) via inner product:

$$s_{uv} = h_u^\top h_v$$

The loss is binary cross-entropy with 5 negative samples per positive edge, summed across all edge types:

$$\mathcal{L} = \sum_{\phi \in \mathcal{R}} \left[-\frac{1}{|\mathcal{E}_\phi|}\sum_{(u,v) \in \mathcal{E}_\phi} \log\sigma(s_{uv}) - \frac{1}{|\mathcal{E}_\phi^-|}\sum_{(u,j) \in \mathcal{E}_\phi^-} \log(1 - \sigma(s_{uj}))\right]$$

The intuition is that predicting which accounts transact with each other forces the encoder to learn structural roles (hubs, bridges, isolated nodes) and behavioural similarity — patterns that correlate with fraud without requiring labels.

- **Defaults**: hidden_dim=64, num_layers=2, dropout=0.3, num_neg_samples=5

---

## Shared Training Details

All neural models (L2–L4) share the following training protocol unless otherwise noted:

- **Optimiser**: AdamW with weight decay $10^{-4}$
- **Learning rate**: $10^{-3}$ with cosine annealing to $10^{-6}$
- **Gradient clipping**: Max norm 1.0
- **Early stopping**: On validation AUPRC with configurable patience (default 15 epochs)
- **Loss**: BCEWithLogitsLoss with inverse-frequency `pos_weight`
- **Evaluation frequency**: Every 5 epochs
- **Primary metric**: AUPRC (precision-recall area under curve), chosen because it is more informative than AUROC under severe class imbalance
- **Secondary metrics**: AUROC, F1, precision, recall, confusion matrix
