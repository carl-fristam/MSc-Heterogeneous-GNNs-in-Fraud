"""
Unified training loop for all GNN models.

Supports:
  - Homogeneous node classification (GCN, GraphSAGE on Data)
  - Homogeneous edge classification (GCN, GraphSAGE on Data)
  - Heterogeneous node classification (HGT, HMPNN on HeteroData)
  - Heterogeneous edge classification (HGT, HMPNN on HeteroData)

Usage:
    from src.training.trainer import Trainer, TrainConfig

    cfg = TrainConfig(task="node", epochs=200, lr=1e-3)
    trainer = Trainer(model, data, cfg, device)
    metrics = trainer.run()
"""

from copy import deepcopy
from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix,
)

from src.utils.class_weights import compute_class_weights


@dataclass
class TrainConfig:
    task: str = "node"          # "node" or "edge"
    graph_type: str = "hetero"  # "hetero" or "homo"
    epochs: int = 200
    lr: float = 1e-3
    weight_decay: float = 1e-4
    patience: int = 15
    scheduler: bool = True
    dropout: float = 0.3
    no_class_weight: bool = False
    target_node_type: str = "transaction"


class Trainer:
    def __init__(self, model: nn.Module, data, config: TrainConfig, device: torch.device):
        self.model = model.to(device)
        self.data = data.to(device)
        self.config = config
        self.device = device

        # Resolve labels and masks based on task + graph type
        self._resolve_targets()

    def _resolve_targets(self):
        cfg = self.config
        data = self.data

        if cfg.graph_type == "homo":
            if cfg.task == "node":
                self.y = data.y
                self.train_mask = data.train_mask
                self.val_mask = data.val_mask
                self.test_mask = data.test_mask
            else:
                self.y = data.edge_y
                self.train_mask = data.edge_train_mask
                self.val_mask = data.edge_val_mask
                self.test_mask = data.edge_test_mask
        else:
            if cfg.task == "node":
                nt = cfg.target_node_type
                self.y = data[nt].y
                self.train_mask = data[nt].train_mask
                self.val_mask = data[nt].val_mask
                self.test_mask = data[nt].test_mask
            else:
                ys, trains, vals, tests = [], [], [], []
                self.edge_type_slices = {}
                offset = 0
                for et in data.edge_types:
                    if hasattr(data[et], "y") and data[et].y is not None:
                        n = data[et].y.shape[0]
                        ys.append(data[et].y)
                        trains.append(data[et].train_mask)
                        vals.append(data[et].val_mask)
                        tests.append(data[et].test_mask)
                        self.edge_type_slices[et] = (offset, offset + n)
                        offset += n

                self.y = torch.cat(ys)
                self.train_mask = torch.cat(trains)
                self.val_mask = torch.cat(vals)
                self.test_mask = torch.cat(tests)

    def _compute_edge_logits_hetero(self, x_dict):
        """Score all labelled edges by concatenating src + dst embeddings."""
        logits_list = []
        for et, (start, end) in self.edge_type_slices.items():
            src_type, _, dst_type = et
            edge_index = self.data[et].edge_index
            src_emb = x_dict[src_type][edge_index[0]]
            dst_emb = x_dict[dst_type][edge_index[1]]
            edge_emb = torch.cat([src_emb, dst_emb], dim=1)
            logits_list.append(self.model.classifier(edge_emb).squeeze(-1))
        return torch.cat(logits_list)

    def _forward(self):
        cfg = self.config

        if cfg.graph_type == "homo":
            return self.model(self.data)

        if cfg.task == "node":
            return self.model(self.data)
        else:
            x_dict = self.model(self.data)
            return self._compute_edge_logits_hetero(x_dict)

    def run(self) -> dict:
        """Full train/val/test loop. Returns test metrics dict."""
        cfg = self.config

        total_params = sum(p.numel() for p in self.model.parameters())
        print(f"Model parameters: {total_params:,}")

        train_labels = self.y[self.train_mask]
        if not cfg.no_class_weight:
            weights = compute_class_weights(train_labels)
            pos_weight = weights[1] / weights[0] if len(weights) > 1 else torch.tensor(1.0)
            criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight.to(self.device))
            print(f"Class weights: pos_weight={pos_weight:.2f}")
        else:
            criterion = nn.BCEWithLogitsLoss()

        optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay
        )
        scheduler = (
            torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.epochs, eta_min=1e-6)
            if cfg.scheduler else None
        )

        best_val_auprc = 0.0
        best_state = None
        cnt_wait = 0

        for epoch in range(1, cfg.epochs + 1):
            self.model.train()
            optimizer.zero_grad()

            logits = self._forward()
            loss = criterion(logits[self.train_mask], self.y[self.train_mask])

            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            optimizer.step()
            if scheduler:
                scheduler.step()

            # Validation
            if epoch % 5 == 0 or epoch == 1:
                val_metrics = self._evaluate(criterion, logits)

                print(
                    f"Epoch {epoch:3d} | "
                    f"Train loss: {loss.item():.4f} | "
                    f"Val loss: {val_metrics['loss']:.4f} | "
                    f"Val AUROC: {val_metrics['auroc']:.4f} | "
                    f"Val AUPRC: {val_metrics['auprc']:.4f}"
                )

                if val_metrics["auprc"] > best_val_auprc:
                    best_val_auprc = val_metrics["auprc"]
                    best_state = deepcopy(self.model.state_dict())
                    cnt_wait = 0
                else:
                    cnt_wait += 1
                    if cnt_wait >= cfg.patience:
                        print(f"Early stopping at epoch {epoch} (patience={cfg.patience})")
                        break

        # Test
        if best_state is not None:
            self.model.load_state_dict(best_state)

        return self._test()

    def _evaluate(self, criterion, logits=None):
        """Validation metrics (called during training)."""
        self.model.eval()
        with torch.no_grad():
            if logits is None:
                logits = self._forward()
            val_probs = torch.sigmoid(logits[self.val_mask]).cpu().numpy()
            val_labels = self.y[self.val_mask].cpu().numpy()
            val_loss = criterion(logits[self.val_mask], self.y[self.val_mask]).item()

        if val_labels.sum() > 0:
            auroc = roc_auc_score(val_labels, val_probs)
            auprc = average_precision_score(val_labels, val_probs)
        else:
            auroc = auprc = 0.0

        return {"loss": val_loss, "auroc": auroc, "auprc": auprc}

    def _test(self) -> dict:
        """Final test evaluation."""
        self.model.eval()
        with torch.no_grad():
            logits = self._forward()
            test_probs = torch.sigmoid(logits[self.test_mask]).cpu().numpy()
            test_labels = self.y[self.test_mask].cpu().numpy()

        test_preds = (test_probs >= 0.5).astype(int)

        metrics = {
            "auroc": roc_auc_score(test_labels, test_probs) if test_labels.sum() > 0 else 0.0,
            "auprc": average_precision_score(test_labels, test_probs) if test_labels.sum() > 0 else 0.0,
            "f1": f1_score(test_labels, test_preds, zero_division=0),
            "precision": precision_score(test_labels, test_preds, zero_division=0),
            "recall": recall_score(test_labels, test_preds, zero_division=0),
            "confusion_matrix": confusion_matrix(test_labels, test_preds),
        }

        print(f"\n{'='*50}")
        print("TEST RESULTS")
        print(f"{'='*50}")
        print(f"  AUROC:     {metrics['auroc']:.4f}")
        print(f"  AUPRC:     {metrics['auprc']:.4f}")
        print(f"  F1:        {metrics['f1']:.4f}")
        print(f"  Precision: {metrics['precision']:.4f}")
        print(f"  Recall:    {metrics['recall']:.4f}")
        print(f"  Confusion matrix:\n{metrics['confusion_matrix']}")

        return metrics
