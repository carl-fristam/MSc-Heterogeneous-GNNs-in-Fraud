# Literature Map — GNNs for AML / Fraud Detection

> Working document for thesis literature review ideation.

---

## 1 · Foundational Graph Methods

### [1] Heterogeneous Graph Transformer (HGT)
- **Authors:** Ziniu Hu, Yuxiao Dong, Kuansan Wang, Yizhou Sun
- **Year:** 2020
- **Venue:** WWW 2020
- **URL:** https://arxiv.org/abs/2003.01332
- **PDF:** `HU ET AL..pdf`
- **Summary:** Introduces Heterogeneous Graph Transformer — attention-based architecture for heterogeneous graphs. Uses node- and edge-type-dependent attention to handle different relation types. Scales to large heterogeneous graphs. Foundation for heterogeneous graph representation learning.
- **Relevance:** ✅ Core — foundational architecture for heterogeneous graph learning, underpins HAN/HGMAE-style approaches

### [2] Heterogeneous Graph Masked Autoencoders (HGMAE)
- **Authors:** Yijun Tian, Kaiwen Dong, Chunhui Zhang, Chuxu Zhang, Nitesh V. Chawla
- **Year:** 2022 (AAAI 2023 oral)
- **Venue:** AAAI 2023
- **URL:** https://arxiv.org/abs/2208.09957
- **PDF:** `TIAN ET AL..pdf`
- **Summary:** Self-supervised pretraining on heterogeneous graphs using masked autoencoding. Masks and reconstructs node features and graph structure. This is the method we implement and extend to AML.
- **Relevance:** ✅ Core — our primary method

---

## 2 · Heterogeneous GNNs for AML

### [3] Finding Money Launderers Using Heterogeneous Graph Neural Networks
- **Authors:** Fredrik Johannessen, Martin Jullum
- **Year:** 2023
- **Venue:** arXiv:2307.13499
- **URL:** https://arxiv.org/abs/2307.13499
- **Summary:** First published GNN on large-scale real-world heterogeneous bank data (DNB, Norway). Extends MPNNs to heterogeneous graphs with novel edge-aggregation across multiple relationship types. Same research group as Jullum et al. (2020) traditional ML baseline.
- **Relevance:** ✅ Core — real-world heterogeneous AML graph, very close to our use case

### [4] Heterogeneous Graph Auto-Encoder for Credit Card Fraud Detection
- **Authors:** Moirangthem Tiken Singh, Rabinder Kumar Prasad, Gurumayum Robert Michael, N K Kaphungkui, N. Hemarjit Singh
- **Year:** 2024
- **Venue:** arXiv:2410.08121
- **URL:** https://arxiv.org/abs/2410.08121
- **Summary:** Heterogeneous graph (cardholder–merchant–transaction) + autoencoder. Trained on legitimate transactions, flags anomalous reconstruction. AUC-PR 0.89, F1 0.81. Same tripartite graph topology we're considering.
- **Relevance:** ✅ Core — closest to our heterogeneous graph autoencoder approach

### [5] Anti-Money Laundering by Group-Aware Deep Graph Learning
- **Authors:** Liu, H., Zuo, Y., Zhu, X., Yin, H., & Zhang, M.
- **Year:** 2023 (KDD 2021 / IEEE TKDE 2023)
- **Venue:** IEEE TKDE / IJRASET replication
- **URL:** https://www.ijraset.com/research-paper/anti-money-laundering-by-group-aware-deep-graph-learning
- **Summary:** Community-centric encoder for detecting organized gang-level laundering. Local enhancement scheme aggregating nodes with similar features into gangs. Tested on real data from one of the world's largest bank card alliances.
- **IJRASET replication authors:** Shwetha A B, Shreyas H G, Siddarameshwara J, Srinidhi M, Srujan T S (2025)
- **Relevance:** ✅ Relevant — group-aware detection beyond individual node classification

### [6] Anti-Money Laundering Alert Optimization Using Machine Learning with Graphs
- **Authors:** Ahmad Naser Eddin, Jacopo Bono, David Aparício, David Polido, João Tiago Ascensão, Pedro Bizarro, Pedro Ribeiro
- **Year:** 2021
- **Venue:** arXiv:2112.07508 (Feedzai group)
- **URL:** https://arxiv.org/abs/2112.07508
- **Summary:** Entity-centric + graph-based features for AML alert triage. Time windows for dynamic graph construction. 80% reduction in false positives, >90% true positive detection on real banking data.
- **Relevance:** ✅ Core — practical alert optimization, demonstrates graph features improve over entity-only

### [7] Powerful Graph Neural Networks for Money Laundering Detection
- **Authors:** Stan
- **Year:** 2024
- **Venue:** MSc Thesis, Utrecht University
- **URL:** https://studenttheses.uu.nl/bitstream/handle/20.500.12932/50600/thesis_Stan.pdf
- **Summary:** GNNs on large financial networks (~1M nodes, ~9M edges) using AMLSim data. Builds on Egressy et al. (2024) provably powerful directed multigraph GNNs. Proposes ego IDs, heterogeneous message passing, port numbering.
- **Relevance:** ✅ Core — fellow MSc thesis on SAML-D-scale data, direct comparison point

---

## 3 · Temporal & Hybrid Graph Approaches

### [8] Research on Anti-Money Laundering Technology Based on Graph Attention Mechanism
- **Authors:** Qian Zhang, Yihua Zhu, Ruiheng Zhang, Ruidong Chen, Tian Lan
- **Year:** 2025
- **Venue:** SPIE Conference Proceedings
- **URL:** https://www.spiedigitallibrary.org/conference-proceedings-of-spie/13511/135111T/Research-on-anti-money-laundering-technology-based-on-graph-attention/10.1117/12.3056070.full
- **Summary:** Proposes TAGA (Temporal Aware Graph Attention) building on TGAT. Graph structures + variational autoencoders for temporal encoding. Applied to cryptocurrency/blockchain transactions.
- **Relevance:** ✅ Relevant — temporal-aware graph attention for dynamic transaction patterns

### [9] Detecting and Preventing Money Laundering Using Deep Learning
- **Authors:** (International American University, LA, CA)
- **Year:** 2025
- **Venue:** IJACSA Vol. 16 No. 6
- **URL:** https://thesai.org/Downloads/Volume16No6/Paper_1-Detecting_and_Preventing_Money_Laundering.pdf
- **Summary:** Hybrid LSTM-GraphSAGE combining temporal sequence modelling with graph anomaly detection. 95.4% accuracy on simulated data. Reduces false positives vs. standalone approaches.
- **Relevance:** ✅ Relevant — hybrid temporal+graph architecture

---

## 4 · Self-Supervised Graph Learning for AML

### [10] LaundroGraph: Self-Supervised Graph Representation Learning for Anti-Money Laundering
- **Authors:** Mário Cardoso, Pedro Saleiro, Pedro Bizarro
- **Year:** 2022
- **Venue:** ACM ICAIF 2022
- **URL:** https://arxiv.org/abs/2210.14360
- **Summary:** First fully self-supervised AML system. Customer-transaction bipartite graph + self-supervised link prediction. Outperforms non-graph baselines by 12 pp AUC. Notes AML systems have >95% false positive rates.
- **Relevance:** ✅ Core — self-supervised on bipartite AML graph, direct comparison to HGMAE

### [11] Privacy-Aware Graph Embeddings for Anti-Money Laundering Pipelines
- **Authors:** Nihari Paladugu
- **Year:** 2025
- **Venue:** SSRN 5320964
- **URL:** https://papers.ssrn.com/sol3/papers.cfm?abstract_id=5320964
- **Summary:** Combines GNNs with homomorphic encryption for privacy-preserving AML. Detects suspicious patterns while preserving PII. Addresses cross-border collaboration challenges.
- **Relevance:** ⚪ Secondary — privacy angle relevant for bank deployment discussion section

### [12] Advances in Continual Graph Learning for Anti-Money Laundering
- **Authors:** Bruno Deprez, Wei Wei, Wouter Verbeke, Bart Baesens, Kevin Mets, Tim Verdonck
- **Year:** 2025
- **Venue:** arXiv:2503.24259
- **URL:** https://arxiv.org/abs/2503.24259
- **Summary:** Reviews continual learning for GNN-based AML — replay, regularization, architecture strategies to handle concept drift without catastrophic forgetting. Tested on synthetic + real-world data.
- **Relevance:** ✅ Relevant — temporal adaptation and model maintenance in production AML

---

## 5 · Surveys & Reviews

### [13] Graph Neural Networks for Financial Fraud Detection: A Review
- **Authors:** Dawei Cheng, Yao Zou, Sheng Xiang, Changjun Jiang
- **Year:** 2024
- **Venue:** arXiv:2411.05815 / Frontiers of Computer Science (2025)
- **URL:** https://arxiv.org/abs/2411.05815
- **Summary:** Comprehensive review of 100+ studies on GNNs for financial fraud detection. Taxonomy of approaches, real-world deployment considerations, design factors, limitations.
- **Relevance:** ✅ Core — key survey for literature review chapter

### [14] Financial Fraud Detection Using Graph Neural Networks: A Systematic Review
- **Authors:** (fetch failed — ScienceDirect blocked)
- **Year:** ~2024
- **Venue:** Expert Systems with Applications (Elsevier)
- **URL:** https://www.sciencedirect.com/science/article/pii/S0957417423026581
- **Summary:** Systematic review of GNNs for financial fraud detection.
- **Relevance:** ✅ Core — systematic review in a top journal

### [15] Machine Learning for AML in Banking: Advanced Techniques, Models, and Real-World Case Studies
- **Authors:** Mohit Kumar Sahu
- **Year:** 2020
- **Venue:** Journal of Science & Technology, Vol. 1 No. 1
- **URL:** https://thesciencebrigade.com/jst/article/view/352
- **Summary:** Broad overview of supervised/unsupervised/RL for AML. Feature engineering, case studies, human-in-the-loop, interpretability, bias mitigation.
- **Relevance:** ⚪ Secondary — broad ML context, low-tier journal

### [16] Innovative Fraud Detection in Financial Transactions Using Deep Learning and SHAP Analysis
- **Authors:** Naresh Kumar Bathala, T. S. Sasikala, V. Sita Rama Prasad, S. Sheik Faritha Begum, Upasana Mahajan, Vaitla Sreedevi
- **Year:** 2024
- **Venue:** ICOFE-2024 / SSRN 5083759
- **URL:** https://papers.ssrn.com/sol3/papers.cfm?abstract_id=5083759
- **Summary:** Feed-forward neural network on synthetic tabular transaction data (amount, time, user/merchant ID). SHAP for feature importance. **Not graph-based** — purely tabular deep learning + XAI.
- **Relevance:** ⚪ Peripheral — no graph component; only relevant if citing SHAP/XAI for fraud as contrast to graph approaches

### [17] AI-Powered Fraud Detection in Financial Services: GNN, Compliance Challenges, and Risk Mitigation
- **Authors:** Diego Vallarino
- **Year:** 2025
- **Venue:** SSRN 5170054
- **URL:** https://papers.ssrn.com/sol3/papers.cfm?abstract_id=5170054
- **Summary:** GNN framework for financial fraud detection. Addresses limitations of traditional ML with complex fraud patterns. Focus on compliance challenges and risk mitigation.
- **Relevance:** ⚪ Secondary — compliance/deployment perspective, not core method

---

## 6 · Knowledge Graphs & Infrastructure

### [18] Enhancing Anti-Money Laundering Systems Using Knowledge Graphs and GNNs
- **Authors:** (3rd Intl. Conf. on Financial Technology and Business Analysis, 2024)
- **Year:** 2024
- **Venue:** ResearchGate
- **URL:** https://www.researchgate.net/publication/387449610_Enhancing_Anti-Money_Laundering_Systems_Using_Knowledge_Graphs_and_Graph_Neural_Networks
- **Summary:** Constructs knowledge graphs from financial data. Applies GCN and GAT. GAT encounters overfitting on larger test sets. Transaction timestamps and payment formats are critical features.
- **Relevance:** ✅ Relevant — KG + GNN intersection, GAT performance issues worth noting

### [19] Knowledge Graphs for Anti-Money Laundering and Transaction Monitoring
- **Authors:** Rob Chang (FI Consulting)
- **Year:** 2023
- **Venue:** FI Consulting (white paper)
- **URL:** https://ficonsulting.com/insight-post/knowledge-graphs-for-anti-money-laundering-and-transaction-monitoring/
- **Summary:** Knowledge graphs for AML with clustering and label propagation. References 2018 Joint Statement from federal regulators encouraging AI for AML.
- **Relevance:** ⚪ Secondary — practitioner perspective, clustering + label propagation baselines

### [20] Recommendations on Implementing an Investigation Knowledge Graph to Combat Illicit Money Flows
- **Authors:** TRACE Project (EU Horizon 2020)
- **Year:** 2024
- **Venue:** TRACE EU Policy Brief
- **URL:** https://trace-illicit-money-flows.eu/recommendations-on-implementing-an-investigation-knowledge-graph-to-combat-illicit-money-flows/
- **Summary:** Policy brief on KG technology for law enforcement. AI-driven crime analytics, geographic risk assessment, cross-border coordination.
- **Relevance:** ⚪ Secondary — policy/regulatory context

### [21] AML Knowledge Graphs — Facctum
- **Authors:** Facctum (company)
- **Year:** —
- **Venue:** Company glossary
- **URL:** https://www.facctum.com/terms/aml-knowledge-graphs
- **Summary:** Definitions and foundational concepts of knowledge graphs in financial compliance.
- **Relevance:** ⚪ Peripheral — industry terminology reference only

### [22] Unmasking Money-Laundering Gangs with AI and Graph Databases
- **Authors:** Michael Down (Neo4j)
- **Year:** 2025
- **Venue:** Machine.news
- **URL:** https://www.machine.news/unmasking-shadowy-crime-rings-with-ai-and-graph-databases/
- **Summary:** Graph databases for AML. Traditional AML generates excessive false positives. Cites Panama Papers. Cross-institution visibility remains a key challenge.
- **Relevance:** ⚪ Secondary — industry motivation, not academic

---

## 7 · Explainability & Interpretability

### [23] Machine Learning in Transaction Monitoring: The Prospect of xAI
- **Authors:** Julie Gerlings, Ioanna Constantiou
- **Year:** 2023
- **Venue:** HICSS-56 (CBS Research)
- **URL:** https://research.cbs.dk/en/publications/machine-learning-in-transaction-monitoring-the-prospect-of-xai
- **Summary:** How ML automation/augmentation affects Transaction Monitoring. xAI requirements depend on the liable party. Context-relatable explanations support auditing and diminish investigator bias.
- **Relevance:** ✅ Core — CBS paper, key for discussion on xAI requirements for regulatory acceptance

### [24] Explainable Artificial Intelligence for Anti-Money Laundering
- **Authors:** SAS Institute
- **Year:** 2025
- **Venue:** SAS White Paper
- **URL:** https://www.sas.com/en/whitepapers/explainable-artificial-intelligence-for-anti-money-laundering.html
- **Summary:** XAI techniques address black-box ML models for AML. Practical guidance for transparent, compliant systems.
- **Relevance:** ⚪ Secondary — vendor perspective, supports discussion section

### [25] Machine Learning in AML: Ongoing Monitoring that Learns and Adapts
- **Authors:** AML Watcher
- **Year:** 2025
- **Venue:** AML Watcher (industry)
- **URL:** https://amlwatcher.com/machine-learning-in-aml/
- **Summary:** LIME for explaining flagged transactions. Active learning and human-in-the-loop. Deep contextual data integration (sanctions lists, adverse media).
- **Relevance:** ⚪ Peripheral — practitioner perspective on explainability

---

## 8 · Peripheral / Tooling

### [26] GraphNeT: Graph Neural Networks for Neutrino Telescope Event Reconstruction
- **Authors:** Søgaard, A., Ørsøe, R., Holm, M., et al.
- **Year:** 2023
- **Venue:** JOSS
- **URL:** https://joss.theoj.org/papers/10.21105/joss.04971
- **Summary:** Open-source GNN framework — design patterns, not AML-specific.
- **Relevance:** ⚪ Peripheral — framework design patterns only

### [27–28] Git/GitHub workflow references
- Leveraging Git & GitHub — **URL:** https://dbbrunson.com
- Collaborating on a Paper (Rob Moss) — **URL:** https://git-is-my-lab-book.net
- **Relevance:** ⚪ Peripheral — research methodology/tooling

---

## Relevance Legend

| Symbol | Meaning |
|--------|---------|
| ✅ Core | Must-cite, directly informs thesis method/evaluation |
| ✅ Relevant | Should-cite, supports a specific argument or section |
| ⚪ Secondary | Nice-to-have, background/motivation/discussion |
| ⚪ Peripheral | Probably don't cite unless needed for a specific paragraph |
