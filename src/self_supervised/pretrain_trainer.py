"""
Two-stage self-supervised training orchestrator.

Stage 1: Pretrain encoder with self-supervised objective (no labels).
Stage 2: Train downstream classifier using pretrained embeddings.
  - Frozen probe (default): freeze encoder, train MLP on extracted embeddings
  - Fine-tune: unfreeze encoder and train end-to-end

Wraps the pretrained encoder + classifier into a model compatible with
Trainer's _compute_edge_logits_hetero.
"""

from copy import deepcopy

import torch
import torch.nn as nn

from src.training.trainer import Trainer, TrainConfig
from src.utils.class_weights import compute_class_weights


class PretrainedEdgeClassifier(nn.Module):
    """
    Wraps a pretrained encoder + MLP classifier for edge classification.

    Compatible with Trainer: forward(data) returns x_dict,
    and .classifier is used by _compute_edge_logits_hetero.
    """

    def __init__(self, encoder, hidden_dim, dropout=0.3, freeze=True):
        super().__init__()
        self.encoder = encoder
        self.freeze = freeze

        if freeze:
            for p in self.encoder.parameters():
                p.requires_grad = False

        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, data):
        if self.freeze:
            with torch.no_grad():
                x_dict = self.encoder(data)
        else:
            x_dict = self.encoder(data)
        return x_dict


class PretrainTrainer:
    """
    Two-stage training orchestrator for self-supervised models.

    Args:
        ssl_model: model with pretrain_forward(data) and forward(data) methods
        data: HeteroData
        device: torch device
        hidden_dim: encoder hidden dimension
        pretrain_epochs: epochs for stage 1
        pretrain_lr: learning rate for stage 1
        pretrain_patience: early stopping patience for stage 1
        classify_epochs: epochs for stage 2
        classify_lr: learning rate for stage 2
        classify_patience: early stopping patience for stage 2
        freeze: if True, freeze encoder in stage 2 (linear probe)
        dropout: dropout for classifier
    """

    def __init__(self, ssl_model, data, device, hidden_dim=64,
                 pretrain_epochs=200, pretrain_lr=1e-3, pretrain_patience=20,
                 classify_epochs=200, classify_lr=1e-3, classify_patience=15,
                 freeze=True, dropout=0.3):
        self.ssl_model = ssl_model.to(device)
        self.data = data.to(device)
        self.device = device
        self.hidden_dim = hidden_dim
        self.pretrain_epochs = pretrain_epochs
        self.pretrain_lr = pretrain_lr
        self.pretrain_patience = pretrain_patience
        self.classify_epochs = classify_epochs
        self.classify_lr = classify_lr
        self.classify_patience = classify_patience
        self.freeze = freeze
        self.dropout = dropout

    def run(self):
        """Execute both stages. Returns test metrics dict."""
        print("=" * 60)
        print("STAGE 1: Self-supervised pretraining")
        print("=" * 60)
        self._pretrain()

        print("\n" + "=" * 60)
        mode = "frozen probe" if self.freeze else "fine-tune"
        print(f"STAGE 2: Downstream classification ({mode})")
        print("=" * 60)
        return self._classify()

    def _pretrain(self):
        """Stage 1: train self-supervised objective."""
        optimizer = torch.optim.AdamW(
            self.ssl_model.parameters(),
            lr=self.pretrain_lr,
            weight_decay=1e-4,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.pretrain_epochs, eta_min=1e-6
        )

        best_loss = float("inf")
        best_state = None
        cnt_wait = 0

        # Use val mask to compute validation pretrain loss
        for epoch in range(1, self.pretrain_epochs + 1):
            self.ssl_model.train()
            optimizer.zero_grad()

            loss = self.ssl_model.pretrain_forward(self.data)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.ssl_model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()

            if epoch % 5 == 0 or epoch == 1:
                loss_val = loss.item()
                print(f"Pretrain epoch {epoch:3d} | Loss: {loss_val:.4f}")

                if loss_val < best_loss:
                    best_loss = loss_val
                    best_state = deepcopy(self.ssl_model.state_dict())
                    cnt_wait = 0
                else:
                    cnt_wait += 1
                    if cnt_wait >= self.pretrain_patience:
                        print(f"Early stopping at epoch {epoch}")
                        break

        if best_state is not None:
            self.ssl_model.load_state_dict(best_state)
        print(f"Pretraining complete. Best loss: {best_loss:.4f}")

    def _classify(self):
        """Stage 2: train downstream edge classifier using Trainer."""
        # Wrap encoder + classifier
        wrapped = PretrainedEdgeClassifier(
            self.ssl_model, self.hidden_dim,
            dropout=self.dropout, freeze=self.freeze
        )

        train_config = TrainConfig(
            task="edge",
            graph_type="hetero",
            target_node_type="internal_account",
            epochs=self.classify_epochs,
            lr=self.classify_lr,
            patience=self.classify_patience,
        )

        trainer = Trainer(wrapped, self.data, train_config, self.device)
        return trainer.run()
