"""
Unified training loop for all GNN models (homo and het).

Supports:
  - Homogeneous edge classification (GCN, GraphSAGE, GAT on projected Data)
  - Heterogeneous edge classification (HGT, HMPNN, HeteroGAT on HeteroData)

Usage:
    from src.training.trainer import Trainer, TrainConfig

    cfg = TrainConfig(task="node", epochs=200, lr=1e-3)
    trainer = Trainer(model, data, cfg, device)
    metrics = trainer.run()
"""

from copy import deepcopy
from dataclasses import dataclass

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
from src.utils.threshold_table import print_threshold_table


@dataclass
class TrainConfig:
    task: str = "edge"          # "node" or "edge"
    graph_type: str = "hetero"  # "hetero" or "homo"
    epochs: int = 200
    lr: float = 1e-3
    weight_decay: float = 1e-4
    patience: int = 15
    scheduler: bool = True
    dropout: float = 0.3
    no_class_weight: bool = False
    target_node_type: str = "internal_account"


class Trainer:
    def __init__(self, model: nn.Module, data, config: TrainConfig, device: torch.device):
        self.config = config
        self.device = device

        self.model = model.to(device)
        self.data = data.to(device)
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
                self.amounts = None
            else:
                self.y = data.edge_y
                self.train_mask = data.edge_train_mask
                self.val_mask = data.edge_val_mask
                self.test_mask = data.edge_test_mask
                self.amounts = getattr(data, "amounts", None)
        else:
            if cfg.task == "node":
                nt = cfg.target_node_type
                self.y = data[nt].y
                self.train_mask = data[nt].train_mask
                self.val_mask = data[nt].val_mask
                self.test_mask = data[nt].test_mask
                self.amounts = None
            else:
                ys, trains, vals, tests, amounts = [], [], [], [], []
                self.edge_type_slices = {}
                offset = 0
                for et in data.edge_types:
                    if hasattr(data[et], "y") and data[et].y is not None:
                        n = data[et].y.shape[0]
                        ys.append(data[et].y)
                        trains.append(data[et].train_mask)
                        vals.append(data[et].val_mask)
                        tests.append(data[et].test_mask)
                        if hasattr(data[et], "amounts") and data[et].amounts is not None:
                            amounts.append(data[et].amounts)
                        self.edge_type_slices[et] = (offset, offset + n)
                        offset += n

                self.y = torch.cat(ys)
                self.train_mask = torch.cat(trains)
                self.val_mask = torch.cat(vals)
                self.test_mask = torch.cat(tests)
                self.amounts = torch.cat(amounts) if amounts else None

    def _compute_edge_logits_hetero(self, x_dict):
        """Score all labelled edges by concatenating src + dst embeddings + edge features."""
        logits_list = []
        for et, (start, end) in self.edge_type_slices.items():
            src_type, _, dst_type = et
            edge_index = self.data[et].edge_index
            src_emb = x_dict[src_type][edge_index[0]]
            dst_emb = x_dict[dst_type][edge_index[1]]
            parts = [src_emb, dst_emb]
            if hasattr(self.data[et], "edge_attr") and self.data[et].edge_attr is not None:
                parts.append(self.data[et].edge_attr)
            edge_emb = torch.cat(parts, dim=1)
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

            if epoch % 5 == 0 or epoch == 1:
                val_metrics = self._evaluate(criterion)

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
                        print(f"Early stopping at epoch {epoch} (patience={cfg.patience} checks = {cfg.patience * 5} epochs)")
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
        """Final test evaluation with val-optimised threshold."""
        self.model.eval()
        with torch.no_grad():
            logits = self._forward()

            # Optimise threshold on validation set
            val_probs = torch.sigmoid(logits[self.val_mask]).cpu().numpy()
            val_labels = self.y[self.val_mask].cpu().numpy()
            best_t, best_f1 = 0.5, 0.0
            for t in np.arange(0.05, 0.95, 0.01):
                f = f1_score(val_labels, (val_probs >= t).astype(int), zero_division=0)
                if f > best_f1:
                    best_f1, best_t = f, t
            print(f"\n  Optimal threshold (val F1): {best_t:.2f} (F1={best_f1:.4f})")

            test_probs = torch.sigmoid(logits[self.test_mask]).cpu().numpy()
            test_labels = self.y[self.test_mask].cpu().numpy()

        test_preds = (test_probs >= best_t).astype(int)

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

        amounts = self.amounts[self.test_mask].cpu().numpy() if self.amounts is not None else None
        metrics["threshold_table"] = print_threshold_table(
            test_labels, test_probs, amounts=amounts, model_name=self.config.graph_type,
            optimal_threshold=best_t,
        )
        metrics["_y_true"] = test_labels
        metrics["_y_prob"] = test_probs

        # ── Analysis data ────────────────────────────────────────────────────
        if self.config.graph_type == "hetero" and self.config.task == "edge":
            analysis = {}

            # Node embeddings for t-SNE/UMAP
            with torch.no_grad():
                x_dict = self.model(self.data)
            analysis["embeddings"] = {nt: x.cpu().numpy() for nt, x in x_dict.items()}

            # Per-edge-type test metrics
            et_metrics = {}
            for et, (start, end) in self.edge_type_slices.items():
                et_test = self.test_mask[start:end].cpu().numpy()
                et_labels = self.y[start:end].cpu().numpy()[et_test]
                et_probs = torch.sigmoid(self._forward()).cpu().numpy()[start:end][et_test] if False else None
                # Reuse already computed test_probs via slicing
                # test_mask is concatenated across edge types in same order
            # Recompute from full arrays using slices
            with torch.no_grad():
                full_probs = torch.sigmoid(self._forward()).cpu().numpy()
            for et, (start, end) in self.edge_type_slices.items():
                mask_slice = self.test_mask[start:end].cpu().numpy().astype(bool)
                labels_slice = self.y[start:end].cpu().numpy()[mask_slice]
                probs_slice = full_probs[start:end][mask_slice]
                if labels_slice.sum() > 0:
                    et_metrics[str(et)] = {
                        "auprc": float(average_precision_score(labels_slice, probs_slice)),
                        "auroc": float(roc_auc_score(labels_slice, probs_slice)),
                        "n_test": int(mask_slice.sum()),
                        "n_fraud": int(labels_slice.sum()),
                    }
            analysis["per_edge_type"] = et_metrics

            # Neighbourhood fraud density (training edges only)
            train_fraud_counts = {}
            for et, (start, end) in self.edge_type_slices.items():
                src_type = et[0]
                edge_index = self.data[et].edge_index.cpu()
                train_mask_et = self.train_mask[start:end].cpu().numpy().astype(bool)
                labels_et = self.y[start:end].cpu().numpy()
                senders = edge_index[0][train_mask_et].numpy()
                fraud_labels = labels_et[train_mask_et]
                for s, f in zip(senders, fraud_labels):
                    key = (src_type, int(s))
                    if key not in train_fraud_counts:
                        train_fraud_counts[key] = {"total": 0, "fraud": 0}
                    train_fraud_counts[key]["total"] += 1
                    train_fraud_counts[key]["fraud"] += int(f)
            analysis["train_fraud_counts"] = train_fraud_counts

            # Model-specific: HGT attention, HMPNN message norms
            if hasattr(self.model, "extract_attention"):
                analysis["attention_weights"] = self.model.extract_attention(self.data)
            if hasattr(self.model, "extract_message_norms"):
                analysis["message_norms"] = self.model.extract_message_norms(self.data)

            metrics["_analysis"] = analysis

        return metrics
