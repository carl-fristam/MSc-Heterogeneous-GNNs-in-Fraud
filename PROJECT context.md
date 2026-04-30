  Project Context

  Thesis: MSc thesis — "Heterogeneous Graph Neural Networks for Transaction-Level Fraud Detection" on Danske Bank retail payment data.

  Research Question: Does preserving heterogeneous structure in a transaction graph improve fraud detection over simpler representations?

  Experimental Design: Tabular XGBoost baseline vs three heterogeneous GNN architectures (HGT, HMPNN, HeteroGAT), all evaluated on edge classification
  (transactions = edges) with a strict temporal train/val/test split.

  Data

  - Danske Bank outgoing retail payments (only outgoing — no inbound flows)
  - 50% stratified temporal sample, all fraud preserved → 1,477,082 transactions, 15,282 fraud (1.04%)
  - Train: 923,304 rows (Nov 2024–Aug 2025), fraud: 8,574 (0.93%)
  - Val: 265,399 rows (Sep–Nov 2025), fraud: 3,645 (1.37%)
  - Test: 288,379 rows (Dec 2025–Mar 2026), fraud: 3,063 (1.06%)
  - All feature engineering done externally (OHE, target encodings, velocity features, cyclical time)

  Graph Structure (V1)

  Node types:
  - internal_account — Danske Bank accounts (senders + on-us receivers). Features: 2 dims (target-encoded branch, country)
  - external_account — counterparty accounts (receivers only). Features: 3 dims (target-encoded counterparty bank/account + IBAN flag)

  Edge types:
  - onus_transfer — internal → internal (334,813 edges, 0.61% fraud in test)
  - external_transfer — internal → external (1,142,269 edges, 1.19% fraud in test)

  Edge features (lean set): 30 dims — log amount, OHE groups (channel, clearing, submethod, currency, destination), international flag, cyclical time. No
  velocity/novelty features — GNNs must recover behavioural context through message passing.

  XGBoost sees all features including velocity features (cust_amt_sum_7D, cust_paymentsub_sum_7D, etc.) and novelty flags. This is the key asymmetry: XGBoost
  has explicit customer-level behavioural features that GNNs don't.

  Model Architectures

  XGBoost: Bayesian-tuned via Optuna (50 trials). Best: max_depth=4, lr=0.036, n_estimators=403, subsample=0.70.

  HGT (85,717 params): HGTConv — cross-type transformer attention. Does NOT use edge features in message passing. Edge features concatenated at classifier
  only.

  HMPNN (166,433 params): NNConv — edge features parameterise the message function (neural network generates message weight matrices from 30-dim edge
  features). The ONLY model using edge features IN message passing.

  HeteroGAT (110,785 params): GATConv with edge_dim — edge features projected and added to attention logits before softmax. Lighter edge feature integration
  than HMPNN.

  All GNN classifiers: Linear(2h + d_e → h) → ReLU → Dropout → Linear(h → 1), where h=64, d_e=30.

  Results (Test Set, 50% Sample All Fraud)

  ┌───────────┬───────────┬────────┬────────┬───────────┬────────┬────────┬────────────────┐
  │   Model   │ Threshold │ PR-AUC │ AUROC  │ Precision │ Recall │   F1   │ Fraud Lost (€) │
  ├───────────┼───────────┼────────┼────────┼───────────┼────────┼────────┼────────────────┤
  │ XGBoost   │ 0.94      │ 0.6417 │ 0.9764 │ 0.7699    │ 0.4992 │ 0.6057 │ 2,919,354      │
  ├───────────┼───────────┼────────┼────────┼───────────┼────────┼────────┼────────────────┤
  │ HGT       │ 0.93      │ 0.1260 │ 0.8897 │ 0.6875    │ 0.0108 │ 0.0212 │ 6,530,875      │
  ├───────────┼───────────┼────────┼────────┼───────────┼────────┼────────┼────────────────┤
  │ HMPNN     │ 0.94      │ 0.1876 │ 0.9182 │ 0.1835    │ 0.4182 │ 0.2551 │ 2,454,114      │
  ├───────────┼───────────┼────────┼────────┼───────────┼────────┼────────┼────────────────┤
  │ HeteroGAT │ 0.91      │ 0.1736 │ 0.9020 │ 0.2012    │ 0.3781 │ 0.2626 │ 2,051,394      │
  └───────────┴───────────┴────────┴────────┴───────────┴────────┴────────┴────────────────┘

  Temporal Generalisation Gap (Key Finding)

  ┌───────────┬───────────┬────────────┬────────────┐
  │   Model   │ Val AUPRC │ Test AUPRC │ Δ relative │
  ├───────────┼───────────┼────────────┼────────────┤
  │ XGBoost   │ 0.8119    │ 0.6417     │ −21%       │
  ├───────────┼───────────┼────────────┼────────────┤
  │ HGT       │ 0.5635    │ 0.1260     │ −78%       │
  ├───────────┼───────────┼────────────┼────────────┤
  │ HMPNN     │ 0.3419    │ 0.1876     │ −45%       │
  ├───────────┼───────────┼────────────┼────────────┤
  │ HeteroGAT │ 0.2655    │ 0.1736     │ −35%       │
  └───────────┴───────────┴────────────┴────────────┘

  Training Dynamics

  - HGT: Early stopped at epoch 135. Best val AUPRC 0.5635 at epoch 60.
  - HMPNN: Ran full 200 epochs, still improving. Slow start (AUROC 0.33 at epoch 1 vs HGT's 0.86). Val AUPRC 0.3419 at epoch 200.
  - HeteroGAT: Ran full 200 epochs, near-converged by epoch 185. Val AUPRC 0.2655.

  XGBoost Feature Importance (Top 10)

  1. counterpartyid_te (0.13) — target-encoded counterparty ID
  2. cust_paymentsub_sum_7D (0.12) — velocity feature
  3. cust_amt_sum_7D (0.06) — velocity feature
  4. txn_to_median_ratio (0.06) — behavioural anomaly
  5. BASEVALUE (0.04) — raw amount
  6. cust_paymentsub_sum_1D (0.04) — velocity feature
  7. INTERNATIONALFLAG (0.03)
  8. cust_paymentsub_sum_30D (0.03) — velocity feature
  9. cust_amt_sum_1D (0.03) — velocity feature
  10. log_basevalue (0.02)

  Velocity features dominate — these are exactly what GNNs don't have in the lean feature set.

  Threshold Tables (Key Points)

  XGBoost: At 0.50 threshold: 87% recall, 15% precision, €272k lost. At 0.99: 29% recall, 96% precision, €4.5M lost.

  HMPNN: At 0.50: 82% recall, 6.4% precision, €230k lost. At 0.90: 51% recall, 16% precision, €1.67M lost.

  HeteroGAT: At 0.50: 84% recall, 4.3% precision, €135k lost (lowest of any model at any threshold). At 0.90: 41% recall, 19% precision, €1.82M lost.

  HGT: Non-functional at all thresholds. Even at 0.50: 51% recall, 12% precision.

  Key Operational Insight

  HeteroGAT achieves lowest fraud lost (€2.05M) despite lower recall than XGBoost (€2.92M) — catches higher-value fraud. At threshold 0.50, HeteroGAT loses
  only €135k vs XGBoost's €272k. GNNs catch different fraud than XGBoost.

  Discussion Outline (User-Defined)

  6. Discussion
6.1 Limitations
Cheng et al important note that we should address: “The review catalogues open problems that map directly onto this study thesis setting, specifically interpretability under KYC regulation, scalability to real-time streams, temporal dynamics, label scarcity, and the accommodation of heterogeneous structures.”


The dataset consists exclusively of outgoing transactions initiated by Danske Bank account holders, meaning the graph contains no inbound payment flows. External counterparty nodes therefore appear only as receivers and never as senders, leaving their outgoing behaviour entirely unobservable. This limits the structural signal available for external nodes and means the model cannot detect fraud patterns that manifest in the behaviour of the receiving account — a constraint that would not apply to a bilateral transaction dataset covering both inbound and outbound flows.
6.1.1 Computational Constraints
Computational limitations in DB infrastructure — had to reduce sample size
No GPU compute available at DB
Eddin et al: scalability issues arise for financial institutions, often processing millions of events per day, rendering using the entire history unfeasible
6.2 Generalisation of Findings
6.2.1 Graph Representation in Financial Crime
This data simply is not represented in graph format in this industry
Neighbourhood signal is relevant in both fraud and AML, but differently — for fraud you have two shots at catching it: transaction features AND the graph. For AML you often only have the graph, because the transaction features are deliberately unremarkable
This strengthens the argument for trying GNNs on fraud data — not claiming tabular features are useless, asking whether the graph provides additional signal on top of what the features already give you
6.2.2 Class Imbalance
Real debate in fraud literature — should we train on realistic imbalance?
Generally yes, with nuance — training on true distribution means model learns the real prior probability of fraud
Artificially balancing to 50:50 makes model overconfident — great recall, terrible precision, useless in production
Pure realistic imbalance has problems too — with 700:1, model can get 99.86% accuracy by predicting everything as clean and never learns fraud signal. Gradient updates from fraud cases get drowned out
Practical answer: somewhere in between
Common approaches: undersample clean to 10:1 or 20:1, cost-sensitive learning (XGBoost — mathematically equivalent to resampling), threshold tuning
Goal is calibration — model output probabilities should reflect the true fraud rate in production even if training distribution was manipulated
6.3 Interpretability and Regulatory Considerations
Cheng et al. catalogue interpretability under KYC regulation as an open problem mapping directly onto this thesis setting
A model flagging a transaction must provide justification satisfying compliance and investigator audiences simultaneously — Gerlings and Constantiou (2023)
Explainability requirements are stakeholder-specific rather than universal
Something with UK enforcing more xAI in terms of grounds for fraud detection flagging.
xFraud (future work)
6.4 Operational Viability
Latency — speed of inference, personal communication citing ~100ms inference as operationally relevant
Cheng et al. (2025) caution that the field remains poorly operationalised in production banking systems
Gap between research performance and production deployment
6.5 Future Work
OnCall data point — high indicator of whether a victim is being guided by a malicious actor
Data quality as foundation — good and clean data is a crucial starter for rules and models alike
GPU compute access at DB would allow training on full transaction history rather than reduced sample
Additional data sources and tooling
Self-supervised pretraining as natural extension — covered in related work but not evaluated in the controlled comparison
Customer nodes? Potential KYC data needed. Could be interesting to have customer specific data on age, segment, other KYC stuff.

  Supervisor

  Julie Gerlings — co-author of xAI papers in bibliography. xAI discussed in Discussion, not literature review.

  Completed Sections

  - Section 5 (Results) is drafted with the structure: 5.1 XGBoost baseline, 5.2 GNN results (cross-model + per-model), 5.3 Temporal generalisation gap, 5.4
  Operational cost analysis, 5.5 Summary.
  - Methodology sections 4.3 (XGBoost) and 4.5 (Graph models) are written.

  Style Notes

  - Academic but readable. No fluff.
  - PR-AUC is primary metric. AUROC is misleading under class imbalance.
  - The thesis finding is nuanced: GNNs lose on metrics but catch different/higher-value fraud.
  - The temporal generalisation gap is the central empirical finding.
  - Don't oversell GNNs — XGBoost clearly wins on standard metrics.


Environment and Source Overview
This environment revolves around the drafting of a Master's thesis focused on leveraging Graph Neural Networks (GNNs) for Anti-Money Laundering (AML) and financial fraud detection. The core objective of the thesis is to compare traditional tabular machine learning models (like XGBoost) against advanced Heterogeneous Graph Neural Networks (such as HGT, HeteroGAT, and HMPNN) in detecting rare, suspicious transactions within highly imbalanced datasets.
A critical operational rule established in the drafting of this thesis is the strict separation of theoretical computer science concepts from the financial crime application. Specifically, Chapter 2 (Theoretical Background) is written as a pure computer science text detailing graph topologies and neural architectures without any mention of fraud or AML. The financial context, class imbalance, and application-specific justifications are exclusively reserved for Chapter 4 (Methodology) and beyond.
The Source Landscape (62 Documents): The provided corpus comprises 62 distinct sources spanning academic papers, thesis documents, and textbooks, which can be broadly categorized into four foundational pillars:
Graph Representation Learning (GRL) & GNN Theory: Textbooks and seminal papers defining graph topologies, the Message Passing Neural Network (MPNN) framework, Graph Convolutional Networks (GCN), Graph Attention Networks (GAT), and Transformers on graphs
.
Heterogeneous Graph Architectures: Literature specifically addressing networks with multiple node and edge types (e.g., accounts, merchants, transactions). Key models include the Heterogeneous Graph Transformer (HGT) and Heterogeneous Graph Masked Autoencoders (HGMAE)
.
AML & Financial Fraud Detection: Applied research highlighting the failures of traditional rule-based AML systems (high false positive rates, rigid thresholds) and introducing ML/GNNs to capture multi-hop, collusive laundering patterns
.
Optimization, Imbalance, & Explainability (xAI): Literature detailing strategies for handling extreme class imbalance (e.g., 0.5% fraud) using metrics like PR-AUC and cost-sensitive weighting, Bayesian optimization for hyperparameter tuning, and the necessity of explainability (xAI) for regulatory compliance in financial institutions
.

--------------------------------------------------------------------------------
General Paper Overview and Source Mapping

Introduction
Comment for Julie:
Repetitive currently. Introduction, problem definition and purpose and research objective should be streamlined and shortened for better flow.
Financial fraud is a persistent and growing challenge for the banking industry. The European Banking Authority (2025) reports that fraud levels across European banks increased from 3.5 billion euros in 2023 to 4.2 billion euros in 2024, a 17 per cent increase. Denmark presents a particular case within this landscape, as 92 per cent of the total value of payments are digital (Brock, 2024). This reliance on digital payments creates a secure and efficient transaction infrastructure, but it also exposes the public to digital fraud. Both payment card fraud and credit transfer scams increased in 2023, while in-person financial crime more than halved over the past decade (Statistics Denmark, 2023). Brock (2024) attributes this shift to the fact that criminals follow the evolution of society into digital channels.
The regulatory environment reflects this shift. The Danish Payment Services Act requires that payment service providers, typically banks, are liable for digital payment fraud and have an obligation to protect their clients (Lov om betalinger, 2023). Beyond regulatory compliance, the consequences of inadequate fraud detection are both financial and reputational. The British Financial Conduct Authority imposed a 63.9 million pound penalty on HSBC for weak automated transaction monitoring systems (Financial Conduct Authority, 2021), and such cases erode public trust in financial institutions. For a bank where roughly 60 per cent of revenue comes from net interest income on mortgages and loans (Danske Bank, 2025), the loss of a banking licence due to insufficient controls would be existential, making a strong business case for improved detection processes.
Financial fraud detection has historically relied on rule-based systems created by subject-matter experts, consisting of series of conditional checks. These systems proved ineffective due to their rigidity and inability to adapt to the dynamic nature of fraud. Their rigidity also made them easy to circumvent once the underlying rules became known (Chen et al., 2018). The modern approach has shifted towards machine learning, which is superior to manual rules at detecting patterns in large data volumes (Jullum and Johannessen, 2024). Tree-based ensembles, particularly XGBoost, have become de facto standards for transaction-level fraud scoring in both research and production settings. These models treat each transaction as an independent feature vector, extracting signal from the attributes of the individual payment.
Yet transactions are not independent events. They are interactions between entities that form a network. A transaction that appears unremarkable in isolation may become suspicious in the context of the account's history, the counterparty's connections, and the patterns of flow across the broader graph. Graph neural networks formalise this intuition by aggregating information from local neighbourhoods, incorporating relational context that tabular representations cannot access (Hamilton, 2020; Singh et al., 2024).
Financial transaction networks are also inherently heterogeneous: they contain distinct entity types, bank accounts and external counterparties, connected by relationships with different semantics. A homogeneous graph neural network applies the same aggregation function regardless of entity type, while heterogeneous GNNs condition their message passing on node and edge types, learning separate parameters for each combination (Hu et al., 2020). Cheng et al. (2025) find that heterogeneous approaches consistently achieve the strongest results in the GNN fraud detection literature, yet the evidence base remains thin on real banking data. Motie and Raahemi (2024) report that over 50 per cent of reviewed studies use cryptocurrency datasets, and Jensen and Iosifidis (2023) note that no GNN methods had been applied to real bank transaction data at the time of their survey.
1.1 Problem Definition
Financial crime prevention is a three-fold problem, spanning three distinct dimensions: regulatory, reputational and financial. Laws like the EU Payment Services directive (Directive (EU) 2015/2377, hereafter PSD2) and the Danish Act on Measures to Prevent Money Laundering and the Financing of Terrorism (Consolidation Act No. 1308, hereafter Hvidvaskloven), stipulate the banks responsibility to continuously monitor transactions for suspicious activity and irregularities which deviate from typical behavior. Failure to comply with these obligations can result in both financial and reputational consequences. Such a failure was clearly illustrated in the high-profile Danske Bank Estonia case in 2017, during which Danske Bank had shown systematic failures in their transactions monitoring  between 2007-2015. This resulted in approximately €200 billion flowing through its Estonian branch, ultimately leading to criminal charges, during which the bank pleaded guilty of conspiracy to commit bank fraud. Following their plea, the bank faced $2.06 billion in fines, a significant reduction in shareholder value and the closing of their Estonia Branch (Danske Bank, 2022; KirchMaier & Bjerregaard, 2019).  Beyond supervisory penalties and reductions in stock price, inadequate transactions monitoring can also extend to individual transactions. Article 73 of PSD2 dictates that banks must refund the payer the full amount of any unauthorized transaction immediately, no later than the end of the following business day, except where the bank has reasonable grounds to suspect fraud and communicates those grounds to its national regulator in writing. 
Given the negative implications brought by financial crime, banks have a strong business incentive to counteract them to as great of an extent as possible. Currently, Danske Bank relies on a hybrid detection approach to do so, mainly consisting of rule-based detection and the conventional XGBoost model. Nevertheless, rule based methodologies can be easily bypassed, granted that a criminal gains knowledge of their existence. Also, even though XGBoost, has shown significant ability to detect financial crime in previous papers (Lawal et al. 2025; , Yin & Li, 2025; Indian, 2022), tree-based models treat each instance as independent, and neglects a possible relational structure present in transactional data. This means that current approaches discard the network context within financial crime. There are many examples of this, including a Swedish conviction of 14 people charged with targeting 139 elderly victims and defrauding them of approximately 17 million SEK using call centers across multiple countries (SVT, 2025). While an XGBoost could have passed such a transaction as plausible in isolation, the problem of relational patterns remains. 
The central problem this thesis addresses is whether the relational structure present in financial transaction data carries a signal that improves fraud detection beyond what tabular features alone can provide. Current production systems at the bank use XGBoost trained on transaction-level features, treating each payment as an isolated observation. This approach discards the network context: which accounts transact with which counterparties, how those counterparties behave, and what patterns emerge across the graph. Graph neural networks offer a principled way to incorporate this context, but it remains an open empirical question whether the added structural information translates into better detection performance on real banking data with confirmed fraud labels.
A further dimension of the problem is that financial transaction graphs are not homogeneous. The bank's payment network contains internal accounts that the bank can fully observe, external counterparties whose outgoing behaviour is invisible, on-us transfers between two bank customers, and external transfers where funds leave the institution. Collapsing these distinctions into a single node type and a single edge type, as a homogeneous GNN requires, may discard precisely the information that makes fraud patterns structurally distinctive. Heterogeneous GNNs preserve these type-level distinctions, but whether this preservation yields a measurable improvement is the question this thesis investigates.
1.2 Purpose and Research Objectives
This thesis aims to determine whether heterogeneous graph neural networks improve transaction-level fraud detection on real retail banking data compared to a conventional tabular approach. Using confirmed fraud labels from the bank's retail payment transactions, the study compares three heterogeneous GNN architectures, HGT (Hu et al., 2020), HMPNN (Johannessen and Jullum, 2023), and HeteroGAT, against a production-level XGBoost baseline replicated in collaboration with the bank.
To achieve this, the study constructs a heterogeneous transaction graph from the bank's payment data, with typed nodes representing internal and external accounts and typed edges representing internal and external transfers. Each transaction is represented as a directed edge carrying a feature vector encoding payment attributes. The three GNN architectures tested in this thesis are trained on this graph structure and evaluated on the same temporal test split as the XGBoost baseline, using PR-AUC as the primary metric.
The expected outcome is an empirical answer to whether graph structure provides additional detection signal on this data. If the heterogeneous GNNs outperform the tabular baseline, this supports the case for incorporating relational information into the bank's fraud detection pipeline. If they do not, this is itself a valuable finding, suggesting that the tabular feature set already captures the relevant signal and the added complexity of graph construction and GNN training is not justified.
1.3 Research Question
In line with the research objectives, the following research question has been formulated:
RQ1: Does representing payment transfer data as a heterogeneous graph structure yield improved fraud detection performance compared to a production-level XGBoost baseline trained on the same transaction features in a tabular format?
Some threshold stuff related to operational viability in DB?
1.4 Background
Comments for Julie:
Proper introduction to the field is crucial we feel. That includes properly stating the whole umbrella of financial crime, as to how Danske Bank themselves describe it in internal documentation.
Related to this is the distinction between fraud detection and AML within financial crime. With that we want to make sure we in a continuous way talk about our scope and the use of architectures and related work that mostly cover AML application of architectures.
Section 1.4.3 are notes pertaining to the internal actual fraud detection engine, separate from AML (but can of course play a role in suspicious activity alerts) Maybe that is a good background to introduce in terms of framing the paper.
1.4.1 Financial Crime
Financial crime encompasses any non-violent crime that generally results in a financial loss (Kurshan and Shen, 2020). Europol's Financial and Economic Crime Threat Assessment frames the landscape as an ecosystem in which fraud, money laundering, and corruption function as interdependent engines sustaining serious and organised crime across the European Union (Europol, 2024). The scale is substantial with millions of victims affected across the EU, with victimisation largely underreported, single cases at times amounting to billions of euros in damage, and law enforcement recovering less than two percent of the yearly proceeds of organised crime (Europol, 2024). Bolton and Hand (2002) define financial fraud as "the use of false representations to gain an unjust advantage." Two decades later, the definition holds, but the methods have transformed. Most fraud is now cyber-enabled, with social engineering and impersonation as the primary techniques, and fraudsters as avid consumers of cybercrime-as-a-service tools (Europol, 2024). The following subsections cover the two financial crime categories most directly relevant to this thesis, followed by a consolidated overview of remaining typologies.
Payment fraud
Payment fraud covers criminal activity across numerous payment channels including credit and debit card transactions, ATM withdrawals, person-to-person transfers, wire transfers, online payments, and automated clearing house transactions. It is reported at over 80 per cent of financial organisations and continues an upward trend (Kurshan and Shen, 2020). Card fraud constitutes one of the largest segments, with losses amounting to 1.8 billion euros in 2016 according to the European Central Bank (ibid). The introduction of chip cards and PIN authentication has shifted fraud tactics from card-present to card-not-present transactions, with the latter now accounting for over 70 per cent of total card fraud losses despite the introduction of 3D-Secure authentication (Kurshan and Shen, 2020). 
Regulatory responses have followed. Since January 2021, all digital payments in Europe require two-factor authentication under the EU's Strong Customer Authentication requirements (Commission Delegated Regulation (EU) 2018/389, 2018). While this has contributed to a reduction in domestic card fraud, it has displaced rather than eliminated it, with 75 per cent of Danish card fraud now occurs abroad, with e-commerce fraud showing a corresponding increase (Brock, 2024).
Money Laundering
Add gerlings AML-D definition.
Money laundering is defined as the transfer of illegally obtained funds to conceal their true source and typically proceeds through three stages (Kurshan and Shen, 2020). In the placement stage, proceeds from illegal activities are introduced into the financial system, frequently structured into smaller amounts distributed across multiple accounts and institutions to remain below regulatory reporting thresholds. During layering, multiple transfers occur between shell companies, individuals, and jurisdictions to obscure the origin of funds. In the integration stage, laundered funds re-enter the legitimate economy through investments, property purchases, or luxury goods. Globally, an estimated 715 billion to 1.87 trillion euros is laundered annually (Kurshan and Shen, 2020). Europol (2024) reports that nearly 70 per cent of criminal networks in the EU use money laundering, and that professional launderers have built parallel underground financial systems designed specifically to circumvent surveillance mechanisms. The multi-hop, multi-account nature of layering is precisely the structural pattern that graph-based detection is designed to surface.
Other types of Financial Crime
The categories above may in many cases not operate in isolation. Identity theft, the stealing of personal information through tactics ranging from ATM skimming to phishing, feeds account takeover, in which perpetrators gain access to a victim's account and drain funds across payment channels. Account takeover experienced a 78 per cent increase in 2019 alone, with further acceleration during the global pandemic (Kurshan and Shen, 2020), and exemplifies the convergence of financial crime and general cybersecurity threats. Financial scams, encompassing phone fraud, romance fraud, investment schemes, and business email compromise, typically build on compromised identities and accounts. Synthetic identity fraud, based on fabricated identities blending real and invented personal information, has grown by approximately 35 per cent year-on-year, driven partly by its non-transactional nature, which delays victim reporting and extends the window of exploitation (Kurshan and Shen, 2020). Across all typologies, Kurshan and Shen (2020) observe remarkable adaptiveness to the digital landscape and an increasing propensity for cross-channel and cross-border operations. Europol (2024) reinforces this point: the EU's strong economy and high standard of living make it a prime target for criminal actors operating remotely from jurisdictions with weaker controls.
Europol’s Financial and Economic Crime Threat Assessment from 2024 reinforces the picture painted by Kurshan and Shen (2020): the EU, with its strong economy and high standard of living, is a prime target for criminal actors that operate remotely from jurisdictions with weaker anti-money laundering regulations and standards.
1.4.2 Financial Crime at Danske Bank
To combat and govern financial crime, Danske Bank operates under a layered regulatory framework. In Denmark, the primary legislation is Hvidvaskloven, enforced by the Danish Financial Supervisory Authority. Operating across multiple jurisdictions, the Group must simultaneously satisfy the AML requirements of each market in which it operates, with local financial supervisory authorities overseeing compliance in each. Non-compliance carries consequences ranging from monetary fines and criminal penalties to personal liability and reputational damage, as demonstrated by the Estonia case described earlier.
In general, Danske Bank’s financial crime controls are organised around the customer lifecycle. At onboarding, Know Your Customer (KYC) and Customer Due Diligence (CDD) processes collect and assess customer information to establish a risk rating of low, medium, or high. This rating informs the intensity of ongoing monitoring. Once onboarded, four continuous processes apply: Politically Exposed Person (PEP) screening, sanctions screening, ongoing Customer Due Diligence (CDD), and Transaction Monitoring (TM).
Transaction Monitoring (TM) is the control most directly relevant to this study. Its purpose is to assess whether observed payment flows align with the Danske Bank's understanding of a customer's profile and expected behaviour. Unusual activity, defined as any transaction or behaviour that deviates from expected activity in timing, size, or nature, triggers an internal Unusual Activity Report (UAR), which Danske Bank's Suspicious Activity Reporting Officer (SARO) investigates. If the investigation cannot sufficiently explain the activity, a Suspicious Activity Report (SAR) is filed with the relevant authorities. Unusual activity does not automatically indicate illegal activity as the distinction between unusual and suspicious is deliberate, with the former being an internal signal and the latter a legal determination requiring formal escalation.
Unusual activity is surfaced through three channels: 1) employee observation during day-to-day operations 2) ongoing due diligence, and 3) automated alerts from TM and fraud detection systems. In practice, detection relies on a combination of all three. Automated systems are the primary mechanism at scale, evaluating transaction-level facts, sender, receiver, date, time, location, amount, and payment field descriptions, to flag activity warranting investigation. It is at this automated detection layer that the models evaluated in this thesis operate. Figure 1 illustrates the full process from customer onboarding to regulatory escalation.

2. Theoretical framework
This section introduces the theoretical foundations underlying the thesis, to ensure the reader is equipped with the necessary background for the methodology and analysis that follow. Firstly, the tree-based machine learning financial crime landscape is introduced and defined as the prevailing approach to transaction monitoring and financial fraud detection, followed by an introduction to graph theory and graph neural networks, building toward the heterogeneous architectures evaluated in this thesis.
2.1 Tree-based Machine Learning
Ensemble learning and decision trees
Tree-based models partition the feature space into mutually exclusive sub-regions based on explanatory variables. While a single decision tree is easy to interpret, it is highly prone to overfitting and exhibits high variance. To overcome this, ensemble learning methods combine multiple weak base learners to improve predictive stability and accuracy. Random Forests, for example, build an ensemble of independent decision trees using bootstrap aggregation and random feature selection at each node split (Couronné et al., 2018). This diversity ensures the model captures non-linear relationships without overfitting to the majority class.

Gradient boosting
Gradient boosting represents an advanced ensemble technique. Rather than building independent trees like a Random Forest, boosting algorithms train an ensemble sequentially. Each new tree is designed to directly target and correct the residual errors made by the combination of all previous trees. XGBoost (extreme gradient boosting) is a highly efficient implementation of the concept of gradient boosting. By iteratively adding new base models that minimise a regularised objective function using its gradient, the algorithm effectively handles high-dimensional, sparse tabular data while modelling complex feature interactions (Chen & Guestrin, 2016).
2.2 Fundamentals of Graph Representation
To move beyond the limitations of tabular data and capture relational dynamics, data can be represented as a graph, a structure that encodes relationships between entities rather than treating each data point in isolation (Hamilton, 2020). The power of this formalism lies in its generality as a language for describing complex systems: the same representational framework can describe social networks, molecular interactions, telecommunications infrastructure, or, as is the subject of this thesis, financial transaction flows. The following subsections introduce two graph variants of increasing complexity, beginning with the simplest case.
2.2.1 Homogeneous Graphs
The simplest graph formulation is the homogeneous graph, which assumes that all nodes share a single type and all edges share a single type. In this formulation, the network topology is captured by a single adjacency matrix encoding which entities are connected, paired with a single feature matrix encoding the attributes of each node. Edges may be undirected, directed, or weighted (Hamilton, 2020).
Because of this uniform structure, every entity in a homogeneous graph resides within the same feature dimensionality and all mathematical operations assume that nodes and relationships share identical structural properties. This makes homogeneous graphs the default for relational data, but it also means the that it’s mathematically forbidden to distinguish between different types of entities.
2.2.2 Heterogeneous Graphs
Real-world systems are inherently complex and often contain multiple types of interacting entities. To model this complexity, heterogeneous graphs extend the standard definition of graph representation by defining nodes and edges with specific types.

Furthermore, edges in heterogeneous graphs generally satisfy strict constraints according to these node types, most commonly the constraint that certain edges only connect specific types of nodes (Hamilton, 2020). For example, Hamilton (2020) illustrates this with a biomedical graph containing protein, drug, and disease node types. In such a network, "treatment" edges are constrained to connect only drug nodes to disease nodes, while "polypharmacy side-effect" edges connect only pairs of drug nodes. A well-known special case of this formulation is the multipartite graph, where edges can only connect nodes that have different types (Hamilton, 2020). This explicitly typed structure is crucial for complex modelling, as it allows networks to maintain distinct feature spaces for different entities and connect them via directed, typed edges that accurately reflect their specific structural roles.
2.3 Graph Neural Networks (GNNs)
While the previous section established the topologies of homogeneous and heterogeneous graphs, this section details the computational mechanics of the models designed to learn from them. Graph Neural Networks (GNNs) are, at their core, neural networks adapted to operate directly on graph-structured data, extending deep learning to the relational data type (Hamilton, 2020). As such, rather than treating each data point in isolation, GNNs explicitly leverage the graph's structure to pass, combine, and transform information across the network (Gilmer et al., 2017; Hamilton, 2020).
2.3.1 Message Passing
Graph neural networks process relational data through a unified computational framework known as the message passing neural network paradigm, which underlies all GNN architectures in this domain (Gilmer et al., 2017). The process begins by initialising each node with its original feature vector. At every subsequent layer k, three operations update the hidden state hv​ of a target node v: message generation, aggregation, and updating (Hamilton, 2020).
During message generation, the target node receives messages from its immediate neighbours, incorporating both node features and edge attributes. The aggregation step then combines these incoming messages into a single summary vector using a permutation-invariant function, such as a sum or mean, since neighbours have no natural ordering. Finally, the update step merges this aggregated message with the node's own current representation to produce a new embedding.
[Mathematical equations]?
This process repeats for K layers, such that after K iterations every node's final embedding captures the structural and feature information of its entire K-hop neighbourhood (Gilmer et al., 2017; Hamilton, 2020). All GNN architectures discussed in this thesis implement this framework, differing only in their message generation function, aggregation strategy, and neighbour weighting (Hamilton, 2020).
2.4 Attention Mechanisms and Graph Transformers
2.4.1 Graph Attention Networks (GAT)
Veličković et al. (2018) — GAT. Same situation as Kipf and Welling — you reference GAT throughout but may not have cited the original paper explicitly. Check and add if missing.
2.4.2 Transformers in Graph Learning

3. Related work
The following section covers the related work informing this thesis across dimensions of machine learning for fraud detection, graph neural network architectures, and the methodological choices that shape how results are interpreted, concluding with the empirical gaps this thesis aims to address.
3.1 Machine Learning for Fraud Detection
Gerlings and Constantiou (2023) document in their qualitative study of a large European bank that rule-based transaction monitoring systems produce so many alerts that only a small fraction of filed suspicious activity reports progress to law-enforcement investigation. The authors frame transaction monitoring as a socio-technical problem in which any replacement model must satisfy investigator, compliance, and regulator audiences simultaneously, and they argue that explainability requirements are stakeholder-specific rather than universal. Their work explains why banks treat machine learning for anti-money laundering as an augmentation of rule-based triage rather than a full replacement, and why any model evaluated on real bank data must be judged on operating-point metrics rather than headline accuracy.
Jensen and Iosifidis (2023) provide the canonical survey of statistical and machine-learning methods for AML, organising the literature into two complementary tasks. Client risk profiling focuses on diagnostic models of customer attributes and KYC panel data, while suspicious behaviour flagging targets individual transactions or short behavioural windows using hand-crafted risk indices and increasingly opaque classifiers. They identify two fundamental obstacles that recur across the field: severe class imbalance, since confirmed money laundering constitutes a tiny minority of transactions, and the scarcity of public datasets, which blocks reproducibility and forces most empirical work onto proprietary bank data. The issue of class imbalance is discussed further in section 3.6. The survey traces a progression from logistic regression and random forests through recurrent architectures with temporal attention toward graph-based representations, and names graph learning, semi-supervision, and interpretability as the most promising future directions.
Despite this theoretical promise, gradient-boosted decision trees currently remain the dominant baseline in the financial crime domain. For instance, the XGBoost ensemble established by Jullum et al. (2020) serves as the reference model against which new approaches are measured at DNB, proving that tabular ensembles are a demanding benchmark rather than a weak strawman (Johannessen & Jullum, 2023). To push the performance of these tabular classifiers further, researchers frequently engineer structural features manually to serve as proxies for relational context. Eddin et al. (2021) apply a LightGBM triage layer enriched with hand-crafted graph metrics, specifically node degrees and random walk distances to known illicit nodes, achieving an eighty percent reduction in false positives. Verlaan (2024) similarly notes that traditional algorithms are often augmented with network-derived metrics like egonet summaries or centrality measures.
However, while injecting hand-crafted graph features into tree-based models improves performance, it represents a limited intermediate solution. Manual feature engineering relies heavily on static domain expertise and struggles to capture the full depth of non-linear, multi-hop laundering typologies (Cheng et al., 2024). Furthermore, extracting these topological statistics across massive, dynamic transaction networks introduces severe computational bottlenecks and memory overheads (Bao, 2025). Therefore, rather than manually engineering graph features for traditional classifiers, recent research is transitioning towards Graph Neural Networks (GNNs). GNNs bypass the need for manual extraction by automatically learning complex relational representations directly from the raw network structure, making them the necessary next step for capturing deep financial typologies (Cheng et al., 2024; Motie & Raahemi, 2024).
3.2 Graph neural networks for financial crime
Two recent reviews conclude on a consistent picture of how graph learning is reshaping fraud detection. Motie and Raahemi (2024) conduct a systematic review of thirty-three articles in Expert Systems with Applications, covering credit card fraud, anti-money laundering, insurance fraud, and e-commerce fraud. They find that heterogeneous and multi-relational GNNs generally outperform homogeneous architectures on financial tasks because real transaction data naturally involve multiple entity types and relation types, even though homogeneous models remain popular for their simplicity. The review also flags heavy reuse of a narrow benchmark set, inconsistent reporting of class imbalance handling, and an underdevelopment of unsupervised and self-supervised approaches as methodological gaps.
Cheng et al. (2024) similarly survey more than one hundred studies and propose a unified framework for categorising GNN methodologies in financial fraud. Their headline argument is that GNNs are exceptionally adept at capturing complex relational patterns in financial networks and significantly outperform traditional methods, while they also caution that the field remains poorly operationalised in production banking systems. The review catalogues open problems that map directly onto this study thesis setting, specifically interpretability under KYC regulation, scalability to real-time streams, temporal dynamics, label scarcity, and the accommodation of heterogeneous structures.
Several deployed systems illustrate what this theoretical shift looks like in practice. Rao et al. (2022) introduce xFraud in VLDB, a two-component heterogeneous GNN detector and post-hoc explainer trained on real eBay transaction logs containing up to 1.1 billion nodes. Their detector uses a Heterogeneous Graph Transformer backbone over transactions linked to payment tokens, emails, addresses, buyers, and devices, demonstrating superior performance over competitive baselines such as GAT and the heterogeneous GEM model. Similarly, Shwetha et al. (2025), building on the Group-Aware Graph Neural Network framework introduced by Cheng et al. (source), describe a deep graph learning system trained on data from a major bank-card alliance. By adding a community-level super-node layer to capture coordinated smurfing and gang behaviour, the system raised the area under the ROC curve to 0.98 against strong baselines. These industrial results demonstrate that heterogeneity provide measurable, operational lift on real-world financial data.
3.3 Homogeneous GNNs
Building on the previous introduction to GNNs in the space of financial crime, this section dives specifically into homogenous GNNs in this domain. Naturally, initial research applying deep graph learning to financial crime predominantly treats transaction networks as homogeneous structures, where all accounts are collapsed into a single node type and all transactions into a single edge type. As Motie and Raahemi (2024) observe in their systematic review, training standard architectures like Graph Convolutional Networks (GCNs) or GraphSAGE directly on these flattened graphs remains a popular baseline approach due to its simplicity. Within this homogeneous framing, attention mechanisms have proven particularly valuable. While standard convolutional models aggregate neighbourhood information uniformly, graph attention networks learn to assign different importance weights to specific connections. As highlighted in the broader graph representation literature (Hamilton, 2020) and confirmed in reviews covered thus far (Cheng et al., 2024), this attention mechanism is uniquely suited for anomaly detection, as it allows the model to filter out the noise of numerous benign counterparties and focus strictly on informative, highly suspicious relationships.
However, the assumption of homogeneity severely limits model performance on complex, real-world banking data. Johannessen and Jullum (2023) demonstrate this empirically by running an explicit homogeneous baseline, termed HGraphSAGE, which shares a single parameter set across all nodes in a large-scale commercial banking dataset. Their results show that this homogeneous projection is substantially weaker than type-aware message passing networks, and is notably even outperformed by a basic logistic regression model utilising summary features. This negative result highlights a fundamental flaw in the homogeneous approach. By projecting distinct entities, such as retail customers and corporate organisations, alongside distinct financial interactions into a uniform structure, the network loses the vital semantic nuances that characterise sophisticated money laundering schemes. Consequently, the literature strongly indicates that capturing the true dynamics of financial crime requires architectures capable of explicitly modelling these semantic differences, establishing the clear motivation for heterogeneous graph architectures.
3.4 Heterogeneous GNNs
The Heterogeneous Graph Transformer (HGT) introduced by Hu et al. (2020) serves as a foundational architecture in this domain. While initially validated on the Open Academic Graph benchmark, where it reported improvements of 9% to 21% over state-of-the-art homogeneous baselines, its capacity to handle multi-relational data quickly demonstrated industrial viability for financial networks. For example, Rao et al. (2022) adopted the HGT backbone for xFraud, successfully deploying it to detect illicit transactions across massive, highly heterogeneous e-commerce logs.
Wang et al. (2019) introduce the Heterogenous Graph Attention Network (HAN), which applies hierarchical attention over predefined meta-path sequences, learning importance at both the node level and the semantic level across different meta-paths. While influential, Johannessen and Jullum (2023) note that HAN and similar meta-path-based methods share a common limitation: they are not designed to incorporate edge features, which in a financial crime context, where edge classification is the premise, corresponds to the properties of individual transactions, precisely the information most indicative of fraud.
Focusing strictly on anti-money laundering, Johannessen and Jullum (2023) introduce the Heterogeneous Message Passing Neural Network (HMPNN). They apply this architecture to a proprietary DNB dataset comprising over five million nodes and nearly ten million edges, encompassing distinct customer node types alongside financial transaction and business role edges. Evaluated on a temporal split, their heterogeneous model achieved the highest PR-AUC and ROC-AUC, outperforming logistic regression, a feed-forward neural network, the bank's XGBoost baseline, and a homogeneous HGraphSAGE model. Crucially, the authors attribute the failure of the homogeneous baseline to its inability to exploit typed edge features, reinforcing the relevance of purely heterogeneous architectures for real-world banking data.
This point is mirrored in other financial domains, such as credit card fraud. Singh et al. (2024) deploy a heterogeneous architecture over a network containing distinct cardholder, merchant, and transaction nodes. When tested on a public credit card dataset, their heterogeneous model achieved a PR-AUC of 0.89 and an F1-score of 0.81, significantly outperforming homogeneous baselines such as GraphSAGE and FI-GRL. Across the empirical literature, the evidence supports a consistent conclusion. Once a financial network is possible to model with more than distinct node or edge types, explicitly modelling this heterogeneity delivers measurable, operational gains over both homogeneous graph neural networks and ensemble methods.
3.5 Node or edge classification
When formulating financial crime detection as a graph learning problem, the fundamental choice between node and edge classification dictates how illicit activity is targeted, closely tied to the specific type of financial crime being investigated (Cheng et al., 2024; Motie & Raahemi, 2024). Because anti-money laundering (AML) typically involves evaluating the long-term, aggregated behaviour of an account or customer across multiple interactions, it is naturally suited to node classification. This remains the most prevalent approach in the empirical literature (Motie & Raahemi, 2024). Johannessen and Jullum (2023) classify individual customer nodes to detect money launderers in real banking networks, while Shwetha et al. (2025) classify user accounts to uncover coordinated laundering rings. To adapt transactional fraud into this node-focused framework, some literature suggest to elevate the transaction itself to a node, performing node classification directly on the transaction entity to identify credit card fraud (Singh et al., 2024). Conversely, because transactional fraud involves illicit transfers between accounts that may otherwise appear completely legitimate, it is more intuitively framed as an edge classification task (ibid).
Edge classification and link prediction, however, evaluate the legitimacy of the interactions themselves, making them highly suited for isolating anomalous payment flows and fraudulent transfers (Cheng et al., 2024). Although edge-level tasks constitute a very small fraction of the current literature compared to node-level classification (Motie & Raahemi, 2024), they directly address the mechanics of transactional fraud by scoring the transfer rather than the entity. For example, Eddin et al. (2021) focus exclusively on classifying transaction edges between bank accounts to suppress false positive alerts on isolated transfers. Similarly, Cardoso et al. (2022) deploy a self-supervised link prediction objective, explicitly scoring the anomalousness of the edges connecting customers.
3.6 Class imbalance
While class imbalance remains a defining obstacle in financial crime detection, strategies for handling it differ (Cheng et al., 2024; Motie & Raahemi, 2024). Responses generally fall into three methodological camps. The first retains the natural class distribution and evaluates performance using threshold-free metrics like PR-AUC, explicitly reflecting realistic operating conditions (Johannessen & Jullum, 2023; Shwetha et al., 2025). The second camp reframes the task as anomaly detection, training models exclusively on legitimate behaviour (Singh et al., 2024). The third sidesteps explicit labels entirely through self-supervised representation learning (Cardoso et al., 2022).
The trade-offs across these responses matter significantly for evaluation. Aggressive resampling techniques like SMOTE can cause severe overfitting and distort precision-recall behaviour at realistic operating thresholds (Singh et al., 2024). Furthermore, anomaly detection objectives can flag legitimate but unusual behaviour as fraud, artificially inflating false positives in production (Eddin et al., 2021; Verlaan, 2024). Consequently, to avoid altering the underlying dataset or inflating false alarms, the imbalance can instead be addressed algorithmically via cost-sensitive learning to penalise minority-class misclassifications while evaluating strictly via PR-AUC (Cheng et al., 2024; Johannessen & Jullum, 2023). Additionally, Johannessen & Jullum (2023) note that rather than applying resampling techniques such as SMOTE, preserving the natural imbalance produces results that better reflect realistic operating conditions.
(add this?: Precision-recall curves and AUC-PR are preferred over ROC-AUC in this setting, as ROC-AUC can be misleadingly optimistic when the negative class dominates and even a weak model achieves high true negative rates by default (Jensen and Iosifidis, 2023).)

3.7 Self-supervised pretraining and autoencoders
Self-supervised pretraining is an active research frontier in graph learning that offers a major operational advantage for financial crime detection, where confirmed labels are both scarce and delayed. A primary benefit of this method is that extreme class imbalance can be addressed by training autoencoders exclusively on the vast majority of normal transactions. Singh et al. (2024) demonstrate this principle in the credit card fraud domain by deploying a heterogeneous graph variational autoencoder trained entirely on genuine transactions between customers and merchants. By learning the latent distribution of normal relational behaviour, their model detects anomalies by flagging structural and feature deviations during reconstruction, effectively bypassing the need for a balanced set of labelled fraudulent transactions. In the anti-money laundering domain, Cardoso et al. (2022) similarly train LaundroGraph, a fully self-supervised customer-to-transaction link prediction model, on real bank data. They report an PR-AUC gain of roughly twelve percentage points over a non-graph multilayer perceptron on the proxy task. While foundational architectures like the heterogeneous graph masked autoencoder (Tian et al., 2023) continue to advance the theoretical capabilities of this space, self-supervised pretraining lies outside the model lineup studied in this thesis. Therefore, it is not evaluated in the controlled comparison, which focuses strictly on supervised models trained end-to-end on confirmed fraud labels.
3.8 Summarisation and research gap
While some literature confirm the theoretical advantages of heterogeneous graph neural networks in financial fraud detection (Cheng et al., 2024; Motie & Raahemi, 2024), the empirical literature remains constrained by a heavy reliance on public benchmarks and synthetic simulators. Evaluations utilising confirmed fraud labels from real retail banks are exceptionally rare, leaving the actual operational viability undiscovered (Johannessen & Jullum, 2023). Furthermore, controlled head-to-head comparisons across diverse heterogeneous architectures, such as those contrasting relation-specific message passing with meta-relation attention, are largely absent on unified real-world datasets (Hu et al., 2020; Rao et al., 2022). Finally, traditional tabular models enriched with graph-derived features already provide a demanding benchmark, having been shown to reduce false positives by up to 80% on real bank data (Eddin et al., 2021). Ultimately, bridging these gaps is essential to answer a highly practical question: do these advanced graph architectures actually provide a tangible, operational advantage over strong traditional models when faced with the highly imbalanced reality of a real retail bank? Without a controlled, head-to-head comparison on actual banking data, the true value of heterogeneous graph learning in fighting financial crime remains purely theoretical.

4. Methodology

This chapter details the methodological steps undertaken to answer the research question. The study begins with data collection and exploratory data analysis, outlining the choices made in the creation of the dataset as well as the patterns and characteristics that inform the subsequent analysis. Data preprocessing and feature engineering are then conducted to prepare the inputs for both modelling tracks.
The experimental design is structured as a two-track comparison: an XGBoost classifier operating on transaction records in tabular form, and three heterogeneous graph neural networks representing the same transactions as edges in an account graph, framing fraud detection as an edge classification task. Both tracks are trained on identical temporal data splits. The XGBoost baseline consumes the full engineered feature set built in collaboration with Danske Bank. The GNN models receive only raw transaction properties as edge features, as message passing serves as a learned replacement for hand-engineered aggregations (Fey et al., 2023). Differences in predictive performance may thus be attributed to whether graph structure can match or exceed explicit feature engineering.
Figure 1 provides an overview of the full pipeline.


4.6 Evaluation Metrics 
Section xx stipulated a severe class imbalance within the dataset. The first and most obvious insight from this is that accuracy is an inadequate evaluation metric, easily inflated by purely guessing the majority class, without detecting any fraud whatsoever. The relevant question is therefore not how often the model is correct in aggregate, but rather when it is wrong, and what that costs. To clearly illustrate the asymmetry, consider Article 73 of PSD2, mentioned at the outline of this paper. As a reminder, the article dictates that the bank is liable to repay the customers the full amount of any unauthorized transfer, unless they can provide reasonable grounds of fraud to its national regulator in writing. Höppner et al. (2020), has formalized the asymmetry in literature, and argue that fraud detection should be treated as an instance-dependent cost-sensitive classification problem, in which the financial cost of a misclassification is not fixed but varies with each transaction. While the present study does not adopt their cost-sensitive classifiers directly, their formalization of the cost-evaluation protocol is a helpful tool to substantiate our arguments for the choice of evaluation metrics. Considering rulings PSD2, the approach is straightforward. If we call the fraud amount A and the cost of investigating fraud B, the following confusion matrix can be created. 


Here, B is set to 25 DKK, in keeping with Danske banks internal methodology. From lowest cost to highest: A true negative is the most desirable outcome from a cost perspective, since there is no loss caused by fraud nor investigative cost. True positives come at a cost of B, but should at the same time also be viewed as desirable from an alternative cost perspective, where the less desirable cost of A is avoided at a cost of B. While having the same cost, false positives may carry along a less quantifiable cost of customer inconvenience and dissatisfaction, where legitimate customers get blocked or questioned without probable cause. Finally, false negatives are the worst outcomes, carrying a direct financial loss equal to the transaction amount, and potential reputational scrutiny. 

It follows naturally that recall (sensitivity) is the natural evaluation metric of this study. Recall measures the proportion of actual fraud cases identified by the model (Hastie, Tibshirani and Friedman, 2009). 

Recall = TPTP + FN

Maximising recall directly minimizes false negatives which as established above represent the costliest outcome under both the Höppner et al. cost matrix and the PSD2 liability framework. We acknowledge, however, that recall is an imperfect proxy for the true cost objective: it assigns equal weight to a missed €5 fraud and a missed €50,000 fraud, whereas the instance-dependent framework would penalise the latter proportionally more. A full implementation of the Höppner et al. cost-sensitive framework is left for future work. 

Recall alone, however, is an insufficient criterion. A trivial recall focused model, where the decision-threshold has been set to zero, would catch all cases of fraud while simultaneously misclassifying all legitimate transactions, making it operationally infeasible. Precision, the proportion of fraud alerts to actual fraud, captures the constraint of recall. 

Precision = TPTP + FP

The two metrics trade-off against each other based on decision threshold, a phenomenon commonly known as the precision-recall tradeoff. The precision-recall tradeoff implies that no single threshold optimally serves all evaluation contexts, and that comparing models at a single arbitrary threshold, such as the default 0.5, risks rewarding a model that happens to be well-calibrated to that threshold rather than one that is genuinely superior. To evaluate model performance across all possible thresholds, we therefore report the area under the Precision-Recall curve (PR-AUC). PR-AUC summarises the precision-recall tradeoff as a single scalar, where a higher value indicates a model that maintains high precision even as recall increases. Critically, unlike the area under the ROC curve (ROC-AUC), PR-AUC is sensitive exclusively to performance on the minority class and does not reward correct classification of the majority class (Davis & Goadrich, 2006). Finally, unlike the ROC-AUC, a random classifier evaluated by PR-AUC does not score 0.5. Instead, the baseline is the prevalence of the target class, which in this case is 0.3%. That way, a model which is 0.4 is not 40% “good” in a conventional sense, but rather roughly 130 times (0.4 / 0.003 ≈ 133) better than random.

Finally, F1-score is reported as a threshold-specific summary statistic, representing the harmonic mean of precision and recall at the operating threshold:

F1 = 2*Precision * RecallPrecision + Recall

The harmonic mean is used rather than the arithmetic mean because it penalises large imbalances between precision and recall, a model with very high recall but near-zero precision receives a low F1 score. 

OPERATIONAL THRESHOLD - FRAUD LOST type of metrics introduction

## Chapter 5:

5. Results and Comparison
Comments for Julie:
This results section is also the main work in progress at the moment. The results we want are in, and now the story telling has to get in place.
This section presents the experimental results across all four models evaluated in this thesis. Section 5.1 reports the tabular XGBoost baseline, which serves as the performance ceiling against which the graph-based models are measured, section 5.2 reports results for each heterogeneous GNN architecture individually and in comparison, Section 5.3 examines the temporal generalisation gap, section 5.4 analyses operational cost implications, and section 5.5 summarises the findings and provides a direct answer to the research question.
5.1 Tabular Baseline (XGBoost)
Introductory paragraph, summary?
5.1.1 Hyperparameter selection?
5.1.2 Performance

5.1.3 Threshold sensitivity
5.1.4 Feature Importance
Figure X reports gain-based feature importance for the top 20 features in the tuned XGBoost model. The most important feature by a wide margin is counterpartyid_te, the target-encoded counterparty identifier. This indicates that certain receiving accounts are strongly associated with fraud — a signal that encodes historical fraud prevalence per counterparty from the training set. The dominance of this single feature warrants caution: it may reflect counterparties that were already flagged during the training period rather than a generalisable pattern, and its predictive value may degrade for novel counterparties in the test period.
The next tier of important features is dominated by customer velocity features: cust_paymentsub_sum_7D (number of payments by submethod in 7 days), cust_amt_sum_7D (total amount sent in 7 days), and txn_to_median_ratio (current transaction amount relative to the customer's median). These features capture behavioural anomalies — deviations from a customer's established transaction pattern — and are precisely the type of hand-engineered temporal aggregations that the GNN models do not have access to in the lean feature set.
This distinction is important for interpreting the subsequent GNN results. XGBoost operates on a feature space that includes both raw transaction properties (amount, channel, currency) and pre-computed velocity features that summarise customer behaviour over rolling time windows. The GNN models, by design, receive only the 30 lean transaction features — raw properties without velocity aggregations — relying instead on message passing over the graph to implicitly learn neighbourhood patterns. The feature importance analysis therefore establishes what signal the GNNs must recover structurally: the customer-level behavioural context that XGBoost receives as explicit input features.
Notably, BASEVALUE and log_basevalue both appear in the top 11, confirming that transaction amount is a strong fraud signal both in absolute terms and relative to customer history. INTERNATIONALFLAG ranks 7th, consistent with the higher fraud prevalence observed in cross-border transactions. The is_new_intl_dest_for_customer novelty flag also appears, suggesting that first-time international destinations carry fraud signal.
5.2 Heterogeneous GNN Results
5.2.1 HGT
[TABLE: HGT test metrics — PR-AUC, AUROC, F1, Precision, Recall]
[TABLE: HGT threshold analysis]
[CONFUSION MATRIX: HGT]
5.2.2 HMPNN
[TABLE: HMPNN test metrics — PR-AUC, AUROC, F1, Precision, Recall]
[TABLE: HMPNN threshold analysis]
[CONFUSION MATRIX: HMPNN]
5.2.3 HeteroGAT
[TABLE: HeteroGAT test metrics — PR-AUC, AUROC, F1, Precision, Recall]
[TABLE: HeteroGAT threshold analysis]
[CONFUSION MATRIX: HeteroGAT]
5.2.4 Cross-Model Comparison
​​If HGT outperforms HMPNN despite not using edge features in message passing, this suggests that graph topology carries more signal than edge-conditioned messages. Conversely, if HMPNN wins, edge-aware message passing is the decisive factor. Either result constitutes a finding.

The comparison table compresses results into a single view, but the question is not simply which row has the highest PR-AUC. The relevant comparison is whether the graph models improve over the tabular baseline, and if so, whether that improvement is consistent across the metrics that matter for deployment: PR-AUC as the primary ranking metric under severe class imbalance, AUROC as a secondary discriminative measure, and the precision-recall tradeoff at operationally relevant thresholds.
Points to address:
PR-AUC delta between XGBoost and the best-performing GNN — is it meaningful?
Do all three GNN architectures improve over XGBoost, or only some?
At which thresholds do the GNNs differ most from XGBoost (flag rate, recall at conservative thresholds)?
Is the improvement worth the computational overhead of graph construction and GNN training?
Recall that all three GNN architectures use two message passing layers, limiting neighbourhood aggregation to a two-hop radius around each transaction edge. Results should be interpreted accordingly — the comparison tests whether local relational context improves detection over the tabular baseline, not whether long-range graph structure carries signal.
Notes:
The owns / owned_by edges between customer and account nodes are added in both directions to enable bidirectional message passing: if one account belonging to a customer exhibits fraudulent behaviour, that signal propagates through the customer node to all sibling accounts during GNN message passing. Without both directions, information would only flow one way (e.g., customer → account), preventing the model from learning customer-level risk patterns from individual account activity.
5.3 Temporal generalisation gap
5.3.1 Validation vs Test Performance
5.3.2 Per-Edge-Type Breakdown
5.4 Operational Cost Analysis
[TABLE: Threshold comparison — all models side by side at each threshold level]
5.4.1 Fraud Lost at Optimal Thresholds
5.4.2 Threshold Comparison Across Models

5.5 Summary of Findings

## References:

8. References
Akiba, T., Sano, S., Yanase, T., Ohta, T., & Koyama, M. (2019). Optuna: A next-generation hyperparameter optimization framework. In Proceedings of the 25th ACM SIGKDD International Conference on Knowledge Discovery and Data Mining (pp. 2623–2631). Association for Computing Machinery. https://doi.org/10.1145/3292500.3330701
Chen, T., & Guestrin, C. (2016). XGBoost: A scalable tree boosting system. In Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining (pp. 785–794). Association for Computing Machinery. https://doi.org/10.1145/2939672.2939785
Cardoso, M., Saleiro, P., & Bizarro, P. (2022). LaundroGraph: Self-supervised graph representation learning for anti-money laundering. In Proceedings of the 3rd ACM International Conference on AI in Finance (pp. 130–138). Association for Computing Machinery. https://doi.org/10.1145/3533271.3561727
Cheng, D., Zou, Y., Xiang, S., & Jiang, C. (2024). Graph neural networks for financial fraud detection: A review. Frontiers of Computer Science, 18(6). https://doi.org/10.1007/s11704-024-40474-y
Deprez, B., Wei, W., Verbeke, W., Baesens, B., Mets, K., & Verdonck, T. (2024). Advances in continual graph learning for anti-money laundering systems: A comprehensive review. arXiv.
Dou, Y., Liu, Z., Sun, L., Deng, Y., Peng, H., & Yu, P. S. (2020). Enhancing graph neural network-based fraud detectors against camouflaged fraudsters. In Proceedings of the 29th ACM International Conference on Information & Knowledge Management (pp. 315–324). Association for Computing Machinery.
Eddin, A. N., Bono, J., Aparício, D., Polido, D., Ascensão, J. T., Bizarro, P., & Ribeiro, P. (2021). Anti-money laundering alert optimization using machine learning with graphs. arXiv preprint arXiv:2112.07508.
Fey, M., & Lenssen, J. E. (2019). Fast graph representation learning with PyTorch Geometric. In ICLR Workshop on Representation Learning on Graphs and Manifolds.
Financial Action Task Force (FATF). (2021). Opportunities and challenges of new technologies for AML/CFT. FATF.
Gerlings, J. (2025). The relevance of explainable artificial intelligence (xAI) in high-risk decisions (PhD Series 35.2025) [Doctoral dissertation, Copenhagen Business School]. https://doi.org/10.22439/phd.35.2025
Gerlings, J., & Constantiou, I. (2023). Machine learning in transaction monitoring: The prospect of xAI. In Proceedings of the 56th Hawaii International Conference on System Sciences (pp. 3474–3483). https://hdl.handle.net/10125/103058
Gerlings, J., Shollo, A., & Constantiou, I. (2021). Reviewing the need for explainable artificial intelligence (xAI). In Proceedings of the 54th Hawaii International Conference on System Sciences (pp. 1284–1293). https://doi.org/10.24251/HICSS.2021.156
Hamilton, W. L. (2020). Graph representation learning. Synthesis Lectures on Artificial Intelligence and Machine Learning, 14(3), 1–159. Morgan & Claypool Publishers.
Hu, Z., Dong, Y., Wang, K., & Sun, Y. (2020). Heterogeneous graph transformer. In Proceedings of The Web Conference 2020 (pp. 2704–2710). Association for Computing Machinery. https://doi.org/10.1145/3366423.3380027
Huda, S., Foo, E., Jadidi, Z., Newton, M. A. H., & Sattar, A. (2025). AMLNet: A knowledge-based multi-agent framework to generate and detect realistic money laundering transactions. Preprint under review.
Jensen, R. I. T., & Iosifidis, A. (2023). Fighting money laundering with statistics and machine learning. IEEE Access, 11, 8889–8903. https://doi.org/10.1109/ACCESS.2023.3239549
Johannessen, F., & Jullum, M. (2023). Finding money launderers using heterogeneous graph neural networks. arXiv preprint arXiv:2307.13499.
Jullum, M., Løland, A., Huseby, R. B., Ånonsen, G., & Lorentzen, J. (2020). Detecting money laundering transactions with machine learning. Journal of Money Laundering Control, 23(1), 173–186. https://doi.org/10.1108/JMLC-07-2019-0055
Kipf, T. N., & Welling, M. (2017). Semi-supervised classification with graph convolutional networks. In Proceedings of the 5th International Conference on Learning Representations (ICLR).
Kurshan, E., & Shen, H. (2021). Graph computing for financial crime and fraud detection: Trends, challenges and outlook. arXiv. https://arxiv.org/abs/2103.03227
Motie, S., & Raahemi, B. (2024). Financial fraud detection using graph neural networks: A systematic review. Expert Systems with Applications, 240, 122156. https://doi.org/10.1016/j.eswa.2023.122156
Paladugu, N. (2025). Privacy-aware graph embeddings for anti-money laundering pipelines. World Journal of Advanced Engineering Technology and Sciences, 15(3), 1223–1231. https://doi.org/10.30574/wjaets
Rao, S. X., Zhang, S., Han, Z., Zhang, Z., Min, W., Chen, Z., Shan, Y., Zhao, Y., & Zhang, C. (2022). xFraud: Explainable fraud transaction detection. Proceedings of the VLDB Endowment, 15(3), 427–436.
Shwetha, A. B., Shreyas, H. G., Siddarameshwara, J., Srinidhi, M., & Srujan, T. S. (2025). Anti-money laundering by group-aware deep graph learning. International Journal for Research in Applied Science and Engineering Technology, 13(8). https://doi.org/10.22214/ijraset.2025.73672
Singh, A., et al. (2024). Heterogeneous graph autoencoders for credit card fraud detection. (Full publication data absent; adjust according to your original Mendeley/Zotero entry.)
Snoek, J., Larochelle, H., & Adams, R. P. (2012). Practical Bayesian optimization of machine learning algorithms. In Advances in Neural Information Processing Systems 25 (pp. 2951–2959).
Tian, Y., Dong, K., Zhang, C., Zhang, C., & Chawla, N. V. (2023). Heterogeneous graph masked autoencoders. In Proceedings of the AAAI Conference on Artificial Intelligence, 37(8), 9997–10005. https://doi.org/10.1609/aaai.v37i8.26192
Vallarino, D. (2024). Modeling adaptive fraud patterns: An agent-centric hybrid framework with MoE and deep learning. SSRN.
Veličković, P., Cucurull, G., Casanova, A., Romero, A., Liò, P., & Bengio, Y. (2018). Graph attention networks. In Proceedings of the 6th International Conference on Learning Representations (ICLR).
Wang, X., Ji, H., Shi, C., Wang, B., Ye, Y., Cui, P., & Yu, P. S. (2019). Heterogeneous graph attention network. In The World Wide Web Conference (pp. 2022–2032). Association for Computing Machinery. https://doi.org/10.1145/3308558.3313562