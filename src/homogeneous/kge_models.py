"""
Knowledge Graph Embedding models for edge classification.

TransE and DistMult learn entity embeddings and score edges (transactions)
based on structural patterns. These sit between graph-feature baselines (L1)
and GNNs (L2) — they capture graph structure without message passing.

Both models score each edge and are trained with BCEWithLogitsLoss
on the fraud label, same as the GNN pipeline.
"""

import torch
import torch.nn as nn


class TransE(nn.Module):
    """
    TransE: h + r ≈ t for positive triples.

    Score = -||h + r - t||  (negated L2 distance → higher = more likely).
    Edge features optionally concatenated with the structural score.
    """

    def __init__(self, num_nodes, embedding_dim=64, num_relations=1,
                 edge_feat_dim=0, dropout=0.3):
        super().__init__()
        self.entity_emb = nn.Embedding(num_nodes, embedding_dim)
        self.relation_emb = nn.Embedding(num_relations, embedding_dim)

        nn.init.xavier_uniform_(self.entity_emb.weight)
        nn.init.xavier_uniform_(self.relation_emb.weight)

        clf_in = 1 + edge_feat_dim
        self.classifier = nn.Sequential(
            nn.Linear(clf_in, embedding_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(embedding_dim, 1),
        )

    def forward(self, data):
        src, dst = data.edge_index
        h = self.entity_emb(src)
        t = self.entity_emb(dst)
        r = self.relation_emb(torch.zeros(src.shape[0], dtype=torch.long, device=src.device))

        dist = -torch.norm(h + r - t, p=2, dim=1, keepdim=True)

        parts = [dist]
        if hasattr(data, "edge_attr") and data.edge_attr is not None:
            parts.append(data.edge_attr)

        return self.classifier(torch.cat(parts, dim=1)).squeeze(-1)


class DistMult(nn.Module):
    """
    DistMult: score = <h, r, t> (element-wise product then sum).

    Captures symmetric relations well. Edge features optionally included.
    """

    def __init__(self, num_nodes, embedding_dim=64, num_relations=1,
                 edge_feat_dim=0, dropout=0.3):
        super().__init__()
        self.entity_emb = nn.Embedding(num_nodes, embedding_dim)
        self.relation_emb = nn.Embedding(num_relations, embedding_dim)

        nn.init.xavier_uniform_(self.entity_emb.weight)
        nn.init.xavier_uniform_(self.relation_emb.weight)

        clf_in = 1 + edge_feat_dim
        self.classifier = nn.Sequential(
            nn.Linear(clf_in, embedding_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(embedding_dim, 1),
        )

    def forward(self, data):
        src, dst = data.edge_index
        h = self.entity_emb(src)
        t = self.entity_emb(dst)
        r = self.relation_emb(torch.zeros(src.shape[0], dtype=torch.long, device=src.device))

        score = (h * r * t).sum(dim=1, keepdim=True)

        parts = [score]
        if hasattr(data, "edge_attr") and data.edge_attr is not None:
            parts.append(data.edge_attr)

        return self.classifier(torch.cat(parts, dim=1)).squeeze(-1)
