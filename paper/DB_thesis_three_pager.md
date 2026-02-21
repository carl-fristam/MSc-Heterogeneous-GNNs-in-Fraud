## Thesis "positioning" one-pager

Hi guys! As discussed on our recent check-in, please read below an overview of how we are positioning the thesis at the moment. Key architecture and their readings are linked, and the general concerns and ideas we have in mind are included.

### Heterogeneous Graph Neural Networks for Anti-Money Laundering Detection

---

### Official Thesis "topic delimitation"

This thesis investigates data-driven approaches to Anti-Money Laundering (AML) using operational-grade transaction-monitoring data from a collaborating bank. The study applies data science methods to compare unsupervised and graph-based modeling approaches in their ability to capture transaction patterns relative to the bank’s existing supervised AML models and operational framework. In addition, the study explores graph-based representations of transaction data and the potential of graph neural network (GNN) methods in enhancing fraud detection and explainability within AML. The analysis focuses on comparative performance metrics and model-specific explainability techniques.

### General thesis direction (heterogenous graph neural nets)

Our thesis investigates whether heterogeneous GNN architectures can improve transaction-level AML detection compared to conventional supervised approaches. The core hypothesis as evident by several papers released in the last approx. 3 years is that modeling different nodes and edge types in a _heterogeneous_ graph neural net preserves structural information that homogeneous graphs and tabular (tree) models lose out on. So that is why this is the direction you have heard us referring to the most.

### Hypotheses:

**H1 — Heterogeneous graph structure improves detection quality beyond accuracy:** We hypothesize that heterogeneous GNN architectures achieve a more favorable precision-recall trade-off than conventional supervised (tabular) models, meaning more true money laundering cases are surfaced per investigator-hour spent reviewing alerts. The goal is not to outperform on accuracy, but to reduce false positive volume while retaining high recall, thereby lowering the operational investigation burden.

**H2 — Self-supervised pre-training on unlabeled transactions yields useful representations despite ground truth limitations:** We hypothesize that a self-supervised approach (e.g. HGMAE) learning the structural patterns of "normal" transaction behavior can produce embeddings that generalize to the detection of anomalous laundering patterns — even accounting for the fact that ground truth labels in AML data are inherently incomplete, reflecting only _detected_ laundering rather than the true underlying illicit activity.

**H3 — Self-supervised heterogeneous embeddings improve the precision-recall trade-off under label scarcity:** We hypothesize that combining self-supervised pre-training (HGMAE) with a downstream heterogeneous classifier (HMPNN or HGT) yields a better operational trade-off than either approach alone — particularly in settings where labeled laundering cases are scarce and ground truth is incomplete. By first learning rich structural representations from the full transaction graph and then fine-tuning on limited labels, the model is expected to surface more genuine laundering patterns with fewer spurious alerts than supervised-only baselines.

**H4 — Heterogeneous GNN explanations provide more actionable investigator guidance than feature importance from tabular models:** We hypothesize that graph-native explainability methods (e.g. attention weights from HGT) surface the specific transaction paths and counterparty relationships that constitute suspicious behavior. Information that per-feature importance scores from tree-based models cannot structurally represent. This richer, topology-aware explanation is expected to reduce the cognitive load on investigators by pointing directly to the subgraph of interest rather than a ranked list of features.

---

#### Papers of interest:

The main papers whose abstracts are of the most interest are these 5. Have a read if you have the time! The two "literature review" papers (bottom 2) give a good overview of the area. The three at the top are the main ones related to the heterogenous direction we want to go in.

**Heterogenous Graph Transformer:**

Key architecture related to attention

https://arxiv.org/abs/2003.01332

**Heterogenous Graph Masked Auto Encoder:**

https://arxiv.org/abs/2208.09957

**Finding Money Launderers Using Heterogeneous Graph Neural Networks:**

(real collaboration with a Norwegian bank. Our thesis supervisor Julie Gerlings has worked with one of these authors before so we are also setting up a meeting with him to talk)

https://arxiv.org/abs/2307.13499

**Graph Neural Networks for Financial Fraud Detection: A Review:**

https://scispace.com/pdf/graph-neural-networks-for-financial-fraud-detection-a-review-4edsb46s1rxq.pdf

**Financial fraud detection using graph neural networks: A systematic review:**

https://www.sciencedirect.com/science/article/abs/pii/S0957417423026581

**Overview**:

So, the architectures:

- HMPNN (heterogeneous message passing, Jullum & Johannessen),
- HGT (transformers, Hu et al.),
- HGMAE (self-supervised masked autoencoders, Tian et al.)

The self-supervised angle (HGMAE) is particularly compelling given how labeled AML data is scarce and also interestingly looks into the unsupervised angle you guys have mentioned too (clustering and outlier detection in terms of masking and learning the patterns of "normal customers" in a transaction environment and thus being able to use learned embeddings in conjunction with a classifier GNN, either HMPNN or HGT for example)

## Current Status

- **Kaggle data work:** Working on a kaggle "SAML-D" (~9.5M transactions, 0.14% illicit rate) dataset as we mentioned.
- **DB data and code:** Data prep of the data we have access to from you into a graph structure is the biggest thing and what we are excited to get done the most - Can be more directly and efficiently done when Carl's access is fully there to the data. Then we can jump into pipelines set up for fitting the different architectures.
- **Other architectures:** We fitted some standard tree based models (RF, XGB) to the Kaggle data but yielded poor results. We think more knowledge on feature engineering (or maybe the real data you work with in production for these models are just more substantial in features) are key to this so a further talk (interview?) on the technicalities of your process could prove very interesting.

## Open Questions / Scope Considerations

- **Explainability:** We are aware of the xAI angle in AML as we have discussed briefly, but it may exceed our scope — would value your input on whether interpretability and xAI should be a priority here. That kind of angle/framing is also something we discuss with our thesis supervisor.
- **Latency:** The performance in terms of latency are also a factor we discussed with you and are aware of. Will be relevant to cover as well.
- **Architecture selection:** We are interested in the three architectures mentioned (HMPNN, HGT, HGMAE) and are open to narrow down to fewer depending on time constraints. Opinions/concerns on this are also welcome.
- **Heterogenous vs supervised:** Building a shared evaluation framework for apples-to-apples comparison across graph and non-graph approaches is something we deem highly interesting for the project in terms of business value impact.
