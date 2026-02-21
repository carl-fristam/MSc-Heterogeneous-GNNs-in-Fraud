# Literature Context Library — GNNs in AML

> Compiled for MSc thesis: Graph Neural Networks and Heterogeneous Graph Masked Autoencoders for Money Laundering Detection
> Last updated: 2026-02-20

---

## Table of Contents

1. [Core GNN-for-AML Papers](#core-gnn-for-aml-papers)
2. [Datasets & Benchmarks](#datasets--benchmarks)
3. [Heterogeneous & Knowledge Graph Approaches](#heterogeneous--knowledge-graph-approaches)
4. [Self-Supervised & Unsupervised Methods](#self-supervised--unsupervised-methods)
5. [Surveys & Reviews](#surveys--reviews)
6. [Traditional ML for AML](#traditional-ml-for-aml)
7. [Industry & Practitioner Resources](#industry--practitioner-resources)
8. [Explainability & Compliance](#explainability--compliance)
9. [Tooling & Methodology](#tooling--methodology)

---

## Core GNN-for-AML Papers

### 1. Anti-Money Laundering in Bitcoin: Experimenting with GCNs for Financial Forensics
- **Authors:** Mark Weber, Giacomo Domeniconi, Jie Chen, Daniel Karl I. Weidele, Claudio Bellei, Tom Robinson, Charles E. Leiserson
- **Year:** 2020
- **Source:** [arXiv:2003.01332](https://arxiv.org/abs/2003.01332) · PDF in `ARXIV_2003.01332.pdf`
- **Key Points:**
  - Introduces the **Elliptic dataset** — ~200K Bitcoin transactions labeled as licit, illicit, or unknown
  - Compares GCN vs. traditional ML baselines (Random Forest, Logistic Regression)
  - GCNs achieve competitive/superior performance by leveraging neighborhood structure, improving recall of illicit transactions
  - Temporal analysis shows model degradation as laundering patterns evolve
- **Relevance:** Foundational benchmark paper; Elliptic dataset is a standard AML-on-graphs benchmark

### 2. Finding Money Launderers Using Heterogeneous Graph Neural Networks
- **Authors:** Fredrik Johannessen, Martin Jullum
- **Year:** 2023
- **Source:** [arXiv:2307.13499](https://arxiv.org/abs/2307.13499) · PDF in `ARXIV_2307.13499.pdf`
- **Key Points:**
  - First published work applying GNNs to a **large real-world heterogeneous network** for AML (DNB, Norway's largest bank)
  - Extended Message Passing Neural Network framework for heterogeneous graphs with multiple relationship types
  - Novel method for aggregating messages across different edge types
  - GNN architecture selection is crucial for heterogeneous financial data
- **Relevance:** Directly relevant — heterogeneous GNNs on real banking data, same direction as thesis

### 3. Anti-Money Laundering by Group-Aware Deep Graph Learning
- **Authors:** Liu, H., Zuo, Y., Zhu, X., Yin, H., & Zhang, M.
- **Year:** 2023 (KDD 2021 / IEEE TKDE 2023)
- **Source:** [IJRASET](https://www.ijraset.com/research-paper/anti-money-laundering-by-group-aware-deep-graph-learning) / [IEEE](https://ieeexplore.ieee.org/document/10114503/)
- **Key Points:**
  - Addresses **organized gang-level money laundering** rather than individual accounts
  - Community-centric encoder for user transaction graphs to derive adjacent gang behaviors
  - Local enhancement scheme aggregating nodes with similar transaction features into gangs
  - Outperforms SOTA on real-world data from one of the world's largest bank card alliances
- **Relevance:** Group-aware detection is an important direction beyond individual node classification

### 4. Anti-Money Laundering Alert Optimization Using Machine Learning with Graphs
- **Authors:** Ahmad Naser Eddin, Jacopo Bono, David Aparício, David Polido, João Tiago Ascensão, Pedro Bizarro, Pedro Ribeiro
- **Year:** 2021
- **Source:** [arXiv:2112.07508](https://arxiv.org/abs/2112.07508) · PDF in `ARXIV_2112.07508.pdf`
- **Key Points:**
  - ML triage model complementing traditional rule-based AML systems
  - Entity-specific features + graph-based features characterizing entity relationships
  - Time windows for dynamic graph construction
  - **80% reduction in false positives** while detecting >90% of true positives on real banking data
- **Relevance:** Practical alert optimization; demonstrates graph features improve over entity-only features

### 5. Heterogeneous Graph Auto-Encoder for Credit Card Fraud Detection
- **Authors:** Moirangthem Tiken Singh, Rabinder Kumar Prasad, et al.
- **Year:** 2024
- **Source:** [arXiv:2410.08121](https://arxiv.org/abs/2410.08121) · PDF in `ARXIV_2410.08121.pdf`
- **Key Points:**
  - Heterogeneous graphs capturing cardholder-merchant-transaction relationships
  - Autoencoder trained on legitimate transactions; flags anomalous reconstruction patterns as fraud
  - AUC-PR of 0.89, F1-score of 0.81 — outperforms GraphSAGE and FI-GRL
- **Relevance:** Heterogeneous graph + autoencoder approach overlaps with HGMAE direction

### 6. Research on Anti-Money Laundering Technology Based on Graph Attention Mechanism
- **Authors:** Qian Zhang, Yihua Zhu, Ruiheng Zhang, Ruidong Chen, Tian Lan
- **Year:** 2025 (SPIE Proceedings)
- **Source:** [SPIE Digital Library](https://www.spiedigitallibrary.org/conference-proceedings-of-spie/13511/135111T/Research-on-anti-money-laundering-technology-based-on-graph-attention/10.1117/12.3056070.full)
- **Key Points:**
  - Proposes **TAGA (Temporal Aware Graph Attention)** model building on TGAT
  - Uses graph structures + variational autoencoders for temporal encoding
  - Captures time-sensitive dependencies for classifying suspicious activities
  - Applied to cryptocurrency/blockchain transaction analysis
- **Relevance:** Temporal-aware graph attention relevant to dynamic transaction patterns

### 7. Detecting and Preventing Money Laundering Using Deep Learning
- **Authors:** (International American University, LA, CA)
- **Year:** 2025
- **Source:** [IJACSA Vol.16 No.6](https://thesai.org/Downloads/Volume16No6/Paper_1-Detecting_and_Preventing_Money_Laundering.pdf) · PDF in `THESAI_DETECTING_PREVENTING_ML.pdf`
- **Key Points:**
  - Hybrid **LSTM-GraphSAGE** model combining temporal sequence modeling with graph-based anomaly detection
  - 95.4% accuracy on simulated financial transactions
  - Reduces false positives and improves detection of advanced AML operations
- **Relevance:** Hybrid temporal+graph approach; combines two architectures used in thesis baselines

### 8. Powerful Graph Neural Networks for Money Laundering Detection
- **Author:** Stan (Utrecht University thesis)
- **Year:** 2024
- **Source:** [Utrecht Student Theses](https://studenttheses.uu.nl/bitstream/handle/20.500.12932/50600/thesis_Stan.pdf) · PDF in `THESIS_STAN.pdf`
- **Key Points:**
  - Tests GNNs on large financial networks (~1M nodes, ~9M edges) using AMLSim data
  - Builds on Egressy et al. (2024) provably powerful directed multigraph GNNs
  - Proposes techniques: ego IDs, heterogeneous message passing, port numbering
  - Can transform any MPNN into powerful directed multigraph neural networks with provable properties
- **Relevance:** Very close to thesis scope — GNNs on SAML-D-scale synthetic data, heterogeneous approaches

---

## Datasets & Benchmarks

### 9. Realistic Synthetic Financial Transactions for Anti-Money Laundering Models (SAML-D / AMLSim)
- **Authors:** Erik Altman, Jovan Blanusa, Luc von Niederhauser, Kubilay Atasu, Andreea Anghel, Nathaniel Tucker
- **Year:** 2022
- **Source:** [arXiv:2208.09957](https://arxiv.org/abs/2208.09957) · PDF in `ARXIV_2208.09957.pdf`
- **Key Points:**
  - Configurable simulator producing transaction networks with realistic properties (heavy-tailed degree distributions, temporal patterns)
  - Injects known ML typologies: fan-in, fan-out, cycle, scatter-gather
  - ~9.5M transactions with ~0.14% laundering rate reflecting real-world class imbalance
  - Enables reproducible AML research without privacy concerns
- **Relevance:** **Primary dataset used in this thesis** — understanding its construction is essential

---

## Heterogeneous & Knowledge Graph Approaches

### 10. Enhancing Anti-Money Laundering Systems Using Knowledge Graphs and Graph Neural Networks
- **Authors:** (3rd Intl. Conf. on Financial Technology and Business Analysis, 2024)
- **Year:** 2024
- **Source:** [ResearchGate](https://www.researchgate.net/publication/387449610_Enhancing_Anti-Money_Laundering_Systems_Using_Knowledge_Graphs_and_Graph_Neural_Networks)
- **Key Points:**
  - Constructs knowledge graphs from financial transactional data
  - Applies GCN and GAT architectures to detect suspicious patterns
  - GAT encounters overfitting/generalization issues on larger test sets
  - Transaction timestamps and payment formats are critical features
- **Relevance:** Knowledge graph construction for AML; GAT performance issues worth noting

### 11. Knowledge Graphs for Anti-Money Laundering and Transaction Monitoring
- **Author:** Rob Chang (FI Consulting)
- **Year:** 2023
- **Source:** [FI Consulting](https://ficonsulting.com/insight-post/knowledge-graphs-for-anti-money-laundering-and-transaction-monitoring/)
- **Key Points:**
  - Knowledge graphs map cash flow relationships between entities
  - Two key graph analytics techniques: **clustering** (reduces false positives) and **label propagation** (reduces false negatives)
  - References 2018 Joint Statement from federal regulators encouraging AI adoption for AML
  - Integration with existing AML operational systems is feasible
- **Relevance:** Practitioner perspective on graph analytics for AML; clustering + label propagation baselines

### 12. Recommendations on Implementing an Investigation Knowledge Graph to Combat Illicit Money Flows
- **Author:** TRACE Project (EU Horizon 2020)
- **Year:** 2024
- **Source:** [TRACE](https://trace-illicit-money-flows.eu/recommendations-on-implementing-an-investigation-knowledge-graph-to-combat-illicit-money-flows/)
- **Key Points:**
  - EU-funded policy brief on knowledge graph technology for law enforcement
  - Covers AI-driven crime analytics, geographic risk assessment, cross-border investigation coordination
  - Knowledge graphs help organize, visualize, and analyze complex relationships in investigation data
- **Relevance:** Policy/regulatory context for knowledge graph approaches in AML investigations

### 13. AML Knowledge Graphs (Facctum)
- **Source:** [Facctum](https://www.facctum.com/terms/aml-knowledge-graphs)
- **Key Points:**
  - Definitions and foundational concepts of knowledge graphs in financial compliance
  - Applications in detecting suspicious financial patterns and relationships
  - Integration with financial crime prevention systems
- **Relevance:** Industry terminology/definitions reference

---

## Self-Supervised & Unsupervised Methods

### 14. LaundroGraph: Self-Supervised Graph Representation Learning for Anti-Money Laundering
- **Authors:** Mário Cardoso, Pedro Saleiro, Pedro Bizarro
- **Year:** 2022
- **Source:** [arXiv:2210.14360](https://arxiv.org/abs/2210.14360) · PDF in `ARXIV_2210.14360.pdf`
- **Key Points:**
  - **First fully self-supervised system for AML detection**
  - Models financial networks as customer-transaction bipartite graph
  - Self-supervised link prediction without requiring labeled data
  - Improves best non-graph baseline by **12 p.p. of AUC** on real-world dataset
  - AML systems currently have **>95% false positive rates**
- **Relevance:** Highly relevant — self-supervised graph learning for AML, directly relates to HGMAE masked autoencoder approach

### 15. Privacy-Aware Graph Embeddings for Anti-Money Laundering Pipelines
- **Author:** Nihari Paladugu
- **Year:** 2025
- **Source:** [SSRN 5320964](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=5320964)
- **Key Points:**
  - Combines GNNs with **homomorphic encryption** for privacy-preserving AML
  - Detects suspicious patterns while preserving personally identifiable information
  - Addresses cross-border collaboration challenges
- **Relevance:** Privacy-preserving GNN for AML; relevant for discussing real-world deployment constraints

---

## Surveys & Reviews

### 16. Graph Neural Networks for Financial Fraud Detection: A Review
- **Authors:** Dawei Cheng, Yao Zou, Sheng Xiang, Changjun Jiang
- **Year:** 2024
- **Source:** [arXiv:2411.05815](https://arxiv.org/abs/2411.05815) · PDF in `ARXIV_2411.05815.pdf`
- **Key Points:**
  - Comprehensive review of 100+ studies on GNNs for financial fraud detection
  - GNNs excel at capturing complex relational patterns and dynamics within financial networks
  - Discusses real-world deployment considerations, design factors, current limitations
  - Published in *Frontiers of Computer Science* (2025)
- **Relevance:** Key survey for literature review chapter; taxonomy of GNN approaches for financial crime

### 17. Advances in Continual Graph Learning for Anti-Money Laundering Systems: A Comprehensive Review
- **Authors:** Bruno Deprez et al.
- **Year:** 2025
- **Source:** [arXiv:2503.24259](https://arxiv.org/abs/2503.24259) · PDF in `ARXIV_2503.24259.pdf`
- **Key Points:**
  - Reviews **continual graph learning** for AML — addressing catastrophic forgetting when fine-tuning on new data
  - Categorizes methods: replay-based, regularization-based, architecture-based strategies within GNN framework
  - Money launderers continuously adapt tactics requiring constant model fine-tuning
  - Demonstrates continual learning improves model adaptability under extreme class imbalance and evolving fraud patterns
- **Relevance:** Important for discussing model maintenance and temporal adaptation in production AML systems

### 18. Machine Learning for AML in Banking: Advanced Techniques, Models, and Real-World Case Studies
- **Author:** Mohit Kumar Sahu
- **Year:** 2020
- **Source:** [The Science Brigade](https://thesciencebrigade.com/jst/article/view/352)
- **Key Points:**
  - Reviews supervised, unsupervised, and reinforcement learning for AML
  - Covers feature engineering, model selection, handling imbalanced datasets, ensemble methods
  - Advocates human-in-the-loop approaches
  - Addresses interpretability, bias mitigation, regulatory compliance, privacy concerns
- **Relevance:** Broad ML-for-AML context; useful for literature review background

### 19. AI-Powered Fraud Detection in Financial Services: GNN, Compliance Challenges, and Risk Mitigation
- **Author:** Diego Vallarino
- **Year:** 2025
- **Source:** [SSRN 5170054](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=5170054)
- **Key Points:**
  - GNN-based framework for detecting financial fraud
  - Addresses limitations of traditional ML models with complex fraud patterns
  - Focus on compliance challenges and risk mitigation strategies
- **Relevance:** Compliance/practical deployment perspective on GNN-based fraud detection

### 20. SSRN Paper: GNN for AML (ID 5083759)
- **Source:** [SSRN 5083759](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=5083759)
- **Note:** Could not retrieve full details (Cloudflare blocked). Likely a recent (2024-2025) paper on GNN applications for AML.

---

## Traditional ML for AML

### 21. Detecting Money Laundering Transactions with Machine Learning
- **Authors:** Martin Jullum, Anders Løland, Ragnar Bang Huseby, Geir Ånonsen, Johannes Lorentzen
- **Year:** 2020
- **Source:** [Emerald / JMLC](https://www.emerald.com/insight/content/doi/10.1108/JMLC-07-2019-0055/full/html) · *Already in literature as* `JULLUM.pdf`
- **Key Points:**
  - Supervised ML for prioritizing transactions for manual investigation at DNB (Norway)
  - Training on 3 data types: normal transactions, flagged suspicious, reported cases
  - Shows that excluding non-reported alerts from training leads to sub-optimal results
  - One of very few published AML models applied to realistically sized data
  - Introduces new performance measure tailored to compare with bank's existing system
- **Relevance:** **Already in collection.** Key baseline paper; same authors as heterogeneous GNN paper (#2)

---

## Explainability & Compliance

### 22. Machine Learning in Transaction Monitoring: The Prospect of xAI
- **Authors:** Julie Gerlings, Ioanna Constantiou
- **Year:** 2023 (HICSS-56)
- **Source:** [CBS Research](https://research-api.cbs.dk/ws/portalfiles/portal/88708916/gerlings_julie_et_al_machine_learning_in_transaction_monotoring_publishersversion.pdf) · PDF in `GERLINGS_ET_AL.pdf`
- **Key Points:**
  - Explores how ML automation/augmentation affects Transaction Monitoring process
  - xAI requirements depend on the **liable party** in the TM process
  - Liability changes depending on augmentation vs. automation of TM
  - Context-relatable explanations support auditing and diminish investigator bias
- **Relevance:** Key for discussion section — xAI requirements are essential for regulatory acceptance of GNN-based AML

### 23. Explainable Artificial Intelligence for Anti-Money Laundering (SAS Whitepaper)
- **Author:** SAS Institute
- **Year:** 2025
- **Source:** [SAS](https://www.sas.com/en/whitepapers/explainable-artificial-intelligence-for-anti-money-laundering.html)
- **Key Points:**
  - ML adoption in AML slow due to regulatory demands for transparency
  - XAI techniques address "black box" nature of ML models
  - Practical guidance for building transparent, compliant AML systems
  - Increasing regulatory support for responsible AI
- **Relevance:** Industry perspective on XAI for AML compliance; supports discussion of model interpretability

### 24. Machine Learning in AML: Ongoing Monitoring that Learns and Adapts
- **Source:** [AML Watcher](https://amlwatcher.com/blog/aml-machine-learning/)
- **Year:** 2025
- **Key Points:**
  - ML detects what static rules miss; addresses explainability challenge
  - **LIME** used to explain why specific transactions are flagged
  - Active learning and human-in-the-loop optimize the ML process
  - Deep contextual data integration (sanctions lists, adverse media)
- **Relevance:** Practitioner perspective on explainability and adaptive monitoring

---

## Industry & Practitioner Resources

### 25. Unmasking Money-Laundering Gangs with AI and Graph Databases
- **Author:** Michael Down (Neo4j)
- **Year:** 2025
- **Source:** [Machine News](https://www.machine.news/unmasking-shadowy-crime-rings-with-ai-and-graph-databases/)
- **Key Points:**
  - Graph databases map relationships between entities (accounts, individuals, companies, addresses)
  - Traditional AML generates excessive false positives; criminals with no "normal" baseline evade detection
  - Panama Papers analyzed using graph databases (ICIJ)
  - Cross-institution visibility remains a key challenge
- **Relevance:** Industry context for graph-based AML; motivates graph representation of financial networks

---

## Tooling & Methodology

### 26. GraphNeT: Graph Neural Networks for Neutrino Telescope Event Reconstruction
- **Authors:** Søgaard, A., Ørsøe, R., Holm, M., et al.
- **Year:** 2023
- **Source:** [JOSS / ADS](https://ui.adsabs.harvard.edu/abs/2023JOSS....8.4971S/abstract)
- **Key Points:**
  - Open-source Python framework for GNN-based reconstruction tasks
  - Inference orders of magnitude faster than traditional techniques
  - Framework applicable across different detector configurations
- **Relevance:** Peripheral — GNN framework design patterns, not AML-specific

### 27. Leveraging Git & GitHub (dbbrunson.com)
- **Source:** [dbbrunson.com](https://www.dbbrunson.com/docs/effective-online-presence/need-and-toolkit/leveraging-git-github/)
- **Key Points:** Guide on using Git/GitHub for effective project management and collaboration
- **Relevance:** Peripheral — research methodology/tooling reference

### 28. Collaborating on a Paper (Git is my Lab Book)
- **Author:** Rob Moss
- **Source:** [git-is-my-lab-book.net](https://git-is-my-lab-book.net/guides/collaborating/collaborating-on-a-paper/)
- **Key Points:**
  - Divide papers into separate LaTeX files per section
  - Use branches for each collaborator, tags for milestones (draft, revision)
  - Community of Practice for reproducible research
- **Relevance:** Peripheral — LaTeX/Git workflow for thesis writing

---

## Summary Statistics

| Category | Count |
|---|---|
| Core GNN-for-AML papers | 8 |
| Datasets & benchmarks | 1 |
| Heterogeneous/KG approaches | 4 |
| Self-supervised methods | 2 |
| Surveys & reviews | 5 |
| Traditional ML for AML | 1 |
| Explainability & compliance | 3 |
| Industry/practitioner | 1 |
| Tooling/methodology | 3 |
| **Total** | **28** |

## PDFs in `paper/literature/`

| File | Source |
|---|---|
| `HU ET AL..pdf` | *(pre-existing)* |
| `JULLUM.pdf` | *(pre-existing)* — Jullum et al. 2020 |
| `TIAN ET AL..pdf` | *(pre-existing)* |
| `ARXIV_2003.01332.pdf` | Weber et al. — Bitcoin AML with GCNs |
| `ARXIV_2112.07508.pdf` | Naser Eddin et al. — AML Alert Optimization |
| `ARXIV_2208.09957.pdf` | Altman et al. — SAML-D / AMLSim |
| `ARXIV_2210.14360.pdf` | Cardoso et al. — LaundroGraph |
| `ARXIV_2307.13499.pdf` | Johannessen & Jullum — Hetero GNNs for AML |
| `ARXIV_2410.08121.pdf` | Singh et al. — Hetero Graph Auto-Encoder |
| `ARXIV_2411.05815.pdf` | Cheng et al. — GNN Fraud Detection Review |
| `ARXIV_2503.24259.pdf` | Deprez et al. — Continual Graph Learning |
| `GERLINGS_ET_AL.pdf` | Gerlings & Constantiou — xAI in TM |
| `THESIS_STAN.pdf` | Stan — Powerful GNNs for ML Detection |
| `THESAI_DETECTING_PREVENTING_ML.pdf` | IJACSA — LSTM-GraphSAGE Hybrid |
