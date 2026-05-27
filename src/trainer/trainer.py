from __future__ import annotations

import csv
import json
import os
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

import torch
import torch.nn.functional as F
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader

from src.anchors.anchor_utils import anchor_similarity_stats
from src.anchors.sinkhorn import ot_assign_by_class
from src.metrics import classification_metrics
from src.model import HDSAConfig, HDSAERCModel
from src.model.hdsa_losses import (
    compactness_loss,
    pairwise_anchor_separation_loss,
    pairwise_confusion_loss,
    sharpen_assignment,
    soft_cross_entropy,
)


CONFUSION_PAIRS = [
    ("happy", "excited"),
    ("angry", "frustrated"),
    ("neutral", "frustrated"),
    ("sad", "frustrated"),
    ("sad", "neutral"),
]
CONFUSION_LOG_PAIRS = [
    ("happy", "excited"),
    ("excited", "happy"),
    ("angry", "frustrated"),
    ("frustrated", "angry"),
    ("neutral", "frustrated"),
    ("frustrated", "neutral"),
    ("sad", "frustrated"),
    ("sad", "neutral"),
]
INTENSITY_MAP = {
    "neutral": 0,
    "sad": 1,
    "happy": 1,
    "frustrated": 2,
    "angry": 2,
    "excited": 2,
}


class HDSATrainer:
    def __init__(
        self,
        args,
        model: HDSAERCModel,
        train_loader: DataLoader,
        dev_loader: DataLoader,
        test_loader: DataLoader,
        label2id: dict[str, int],
        id2label: dict[int, str],
    ):
        self.args = args
        self.model = model
        self.train_loader = train_loader
        self.dev_loader = dev_loader
        self.test_loader = test_loader
        self.label2id = label2id
        self.id2label = id2label
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.output_dir = Path(args.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.model.to(self.device)
        self.optimizer = self._build_optimizer()
        total_steps = max(1, len(train_loader) * args.epochs)
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.optimizer,
            lr_lambda=self._linear_warmup_decay(args.warmup_ratio, total_steps),
        )
        self.ce_weights = self._build_ce_weights().to(self.device)
        self.class_thresholds = self._build_class_thresholds().to(self.device) if args.class_adaptive_ema else None
        self.top_ratio_class_ids = self._build_top_ratio_class_ids(args.top_ratio_ema_classes)
        self.intensity_targets_by_class = self._build_intensity_targets().to(self.device)

    def train(self) -> None:
        self._init_logs()
        metadata = {
            "args": vars(self.args),
            "model_config": asdict(self.model.config),
            "label2id": self.label2id,
            "id2label": self.id2label,
        }
        (self.output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        best_dev_weighted_f1 = -1.0
        best_test_weighted_f1 = -1.0
        best_metric = -1.0
        bad_epochs = 0
        best_dev_epoch = -1
        best_test_epoch = -1
        test_at_best_dev = None
        best_dev_metrics = None
        best_test_metrics = None
        last_epoch = 0
        for epoch in range(1, self.args.epochs + 1):
            last_epoch = epoch
            self._append_text(f"\n[{_now()}] epoch {epoch}/{self.args.epochs} started")
            train_stats = self._train_one_epoch(epoch)
            dev_stats = self.evaluate(self.dev_loader)
            test_stats = self.evaluate(self.test_loader)
            assignment_counts = train_stats.pop("assignment_counts")
            ema_update_counts = train_stats.pop("ema_update_counts")
            selected_for_ema_counts = train_stats.pop("selected_for_ema_counts")
            hard_assignment_counts = train_stats.pop("hard_assignment_counts")
            class_ot_max_prob = train_stats.pop("class_ot_max_prob")
            dev_wf1 = dev_stats["weighted_f1"]
            test_wf1 = test_stats["weighted_f1"]
            if dev_wf1 > best_dev_weighted_f1 + self.args.early_stop_min_delta:
                best_dev_weighted_f1 = dev_wf1
                best_dev_epoch = epoch
                test_at_best_dev = test_wf1
                best_dev_metrics = {"dev": dev_stats, "test": test_stats}
                if self.args.save_best_dev:
                    self._save_checkpoint("best_dev_model.pt", epoch, best_dev_metrics)
                    self._append_text(f"[{_now()}] saved new best_dev_model.pt at epoch {epoch}")
            if test_wf1 > best_test_weighted_f1 + self.args.early_stop_min_delta:
                best_test_weighted_f1 = test_wf1
                best_test_epoch = epoch
                best_test_metrics = {"dev": dev_stats, "test": test_stats}
                if self.args.save_best_test:
                    self._save_checkpoint("best_test_model.pt", epoch, best_test_metrics)
                    self._append_text(f"[{_now()}] saved new best_test_model.pt at epoch {epoch}")
            current_metric = self._early_stop_metric(dev_stats, test_stats)
            if current_metric > best_metric + self.args.early_stop_min_delta:
                best_metric = current_metric
                bad_epochs = 0
            else:
                bad_epochs += 1
            best_row = {
                "best_dev_epoch": best_dev_epoch,
                "best_dev_weighted_f1": best_dev_weighted_f1,
                "test_at_best_dev": test_at_best_dev,
                "best_test_epoch": best_test_epoch,
                "best_test_weighted_f1": best_test_weighted_f1,
                "early_stop_metric": self.args.early_stop_metric,
                "best_early_stop_metric": best_metric,
                "bad_epochs": bad_epochs,
            }
            row = {
                "epoch": epoch,
                "train": train_stats,
                "dev": dev_stats,
                "test": test_stats,
                "anchor_stats": anchor_similarity_stats(self.model.get_anchors().detach().cpu()),
                "ot_assignment_counts": self._named_counts(assignment_counts),
                "hard_assignment_counts": self._named_counts(hard_assignment_counts),
                "selected_for_ema_counts": self._named_counts(selected_for_ema_counts),
                "ema_update_counts": self._named_counts(ema_update_counts),
                "class_ot_max_prob": self._named_stats(class_ot_max_prob),
                "best": best_row,
            }
            self._write_epoch(row)
            if self.args.early_stop and bad_epochs >= self.args.early_stop_patience:
                self._append_text(
                    f"[{_now()}] Early stopping at epoch {epoch}. "
                    f"Best {self.args.early_stop_metric}: {best_metric:.6f}"
                )
                break
        if self.args.save_last:
            self._save_checkpoint("last_model.pt", last_epoch, {"dev": dev_stats, "test": test_stats})
        final_dev = self.evaluate(self.dev_loader, split_name="dev")
        final_test = self.evaluate(self.test_loader, split_name="test")
        final_summary = {
            "experiment_name": self.args.experiment_name,
            "active_ablations": getattr(self.args, "active_ablations", []),
            "args": vars(self.args),
            "best_dev_epoch": best_dev_epoch,
            "best_dev_weighted_f1": best_dev_weighted_f1,
            "test_at_best_dev": test_at_best_dev,
            "best_dev_metrics": best_dev_metrics,
            "best_test_epoch": best_test_epoch,
            "best_test_weighted_f1": best_test_weighted_f1,
            "best_test_metrics": best_test_metrics,
            "final_dev": final_dev,
            "final_test": final_test,
            "final_anchor_stats": anchor_similarity_stats(self.model.get_anchors().detach().cpu()),
            "finished_at": _now(),
        }
        (self.output_dir / "final_metrics.json").write_text(json.dumps(final_summary, indent=2), encoding="utf-8")
        (self.output_dir / "experiment_result.json").write_text(json.dumps(final_summary, indent=2), encoding="utf-8")
        self._append_text("\nTraining finished.")
        self._append_text(json.dumps(final_summary, indent=2))

    def _train_one_epoch(self, epoch: int) -> dict:
        self.model.train()
        sums = {
            "loss_total": 0.0,
            "loss_ce": 0.0,
            "loss_proto": 0.0,
            "loss_compact": 0.0,
            "loss_pair": 0.0,
            "loss_pair_anchor": 0.0,
            "loss_intensity": 0.0,
            "ot_entropy_mean": 0.0,
            "ot_entropy_max": 0.0,
            "ot_max_prob_mean": 0.0,
            "ot_max_prob_min": 0.0,
            "ot_max_prob_max": 0.0,
        }
        total = 0
        total_counts = torch.zeros(
            self.args.num_classes,
            self.args.num_subanchors,
            device=self.device,
        )
        total_ema_counts = torch.zeros_like(total_counts)
        total_selected_for_ema_counts = torch.zeros_like(total_counts)
        total_hard_counts = torch.zeros_like(total_counts)
        max_probs_by_class: dict[int, list[torch.Tensor]] = {idx: [] for idx in range(self.args.num_classes)}
        use_top_ratio_now = self.args.use_top_ratio_ema and epoch > self.args.top_ratio_warmup_epochs
        for batch in self.train_loader:
            labels = batch["labels"].to(self.device)
            outputs = self.model(
                input_ids=batch["input_ids"].to(self.device),
                attention_mask=batch["attention_mask"].to(self.device),
                mask_pos=batch["mask_pos"].to(self.device),
            )
            z = outputs["z"]
            anchors = outputs["anchors"]
            loss_ce = F.cross_entropy(outputs["logits"], labels, weight=self.ce_weights)
            soft_targets, assigned_anchor, assignment_counts, ot_stats = ot_assign_by_class(
                reps=z.detach(),
                labels=labels,
                anchors=anchors.detach(),
                epsilon=self.args.ot_epsilon,
                n_iters=self.args.ot_iters,
            )
            hard_counts, batch_max_probs = self._ot_hard_diagnostics(labels, soft_targets.detach())
            flat_anchors = anchors.reshape(self.args.num_classes * self.args.num_subanchors, self.args.anchor_dim)
            proto_logits = (z @ flat_anchors.t()) / self.args.proto_temperature
            soft_targets_for_loss = sharpen_assignment(soft_targets, power=self.args.ot_sharpen_power)
            loss_proto = soft_cross_entropy(proto_logits, soft_targets_for_loss)
            loss_compact = compactness_loss(z, assigned_anchor.detach())
            loss_pair = (
                pairwise_confusion_loss(
                    outputs["logits"],
                    labels,
                    self.label2id,
                    CONFUSION_PAIRS,
                    margin=self.args.pair_margin,
                )
                if self.args.pair_loss_weight > 0
                else z.new_tensor(0.0)
            )
            loss_pair_anchor = (
                pairwise_anchor_separation_loss(
                    anchors,
                    self.label2id,
                    CONFUSION_PAIRS,
                    upper=self.args.pair_anchor_upper,
                )
                if self.args.pair_anchor_loss_weight > 0
                else z.new_tensor(0.0)
            )
            loss_intensity = z.new_tensor(0.0)
            if self.args.use_intensity_head and self.args.intensity_loss_weight > 0:
                intensity_labels = self.intensity_targets_by_class[labels]
                loss_intensity = F.cross_entropy(outputs["logits_intensity"], intensity_labels)
            loss = (
                loss_ce
                + self.args.proto_loss_weight * loss_proto
                + self.args.compact_loss_weight * loss_compact
                + self.args.pair_loss_weight * loss_pair
                + self.args.pair_anchor_loss_weight * loss_pair_anchor
                + self.args.intensity_loss_weight * loss_intensity
            )
            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            clip_grad_norm_(self.model.parameters(), self.args.max_grad_norm)
            self.optimizer.step()
            self.scheduler.step()
            if self.args.use_top_ratio_ema:
                ema_counts, selected_counts = self.model.ema_update_anchors_top_ratio(
                    reps=z.detach(),
                    labels=labels,
                    soft_targets=soft_targets.detach(),
                    top_ratio_class_ids=self.top_ratio_class_ids,
                    top_ratio=self.args.top_ratio_ema_ratio,
                    top_ratio_min_samples=self.args.top_ratio_min_samples,
                    top_ratio_momentum=self.args.top_ratio_momentum,
                    top_ratio_min_conf=self.args.top_ratio_min_conf,
                    use_top_ratio_now=use_top_ratio_now,
                    default_threshold=self.args.default_ema_conf_threshold,
                    normal_momentum=self.args.normal_ema_momentum,
                )
            else:
                ema_counts = self.model.ema_update_anchors_confident(
                    reps=z.detach(),
                    labels=labels,
                    soft_targets=soft_targets.detach(),
                    momentum=self.args.prototype_momentum,
                    threshold=self.args.ema_conf_threshold,
                    class_thresholds=self.class_thresholds,
                    use_fallback=self.args.use_ema_fallback,
                    fallback_momentum=self.args.fallback_momentum,
                )
                selected_counts = torch.zeros_like(ema_counts)
            bsz = labels.size(0)
            total += bsz
            total_counts += assignment_counts.detach()
            total_ema_counts += ema_counts.detach()
            total_selected_for_ema_counts += selected_counts.detach()
            total_hard_counts += hard_counts.detach()
            for cls, values in batch_max_probs.items():
                max_probs_by_class[cls].append(values.detach().cpu())
            sums["loss_total"] += float(loss.detach().cpu()) * bsz
            sums["loss_ce"] += float(loss_ce.detach().cpu()) * bsz
            sums["loss_proto"] += float(loss_proto.detach().cpu()) * bsz
            sums["loss_compact"] += float(loss_compact.detach().cpu()) * bsz
            sums["loss_pair"] += float(loss_pair.detach().cpu()) * bsz
            sums["loss_pair_anchor"] += float(loss_pair_anchor.detach().cpu()) * bsz
            sums["loss_intensity"] += float(loss_intensity.detach().cpu()) * bsz
            sums["ot_entropy_mean"] += ot_stats["ot_entropy_mean"] * bsz
            sums["ot_entropy_max"] = max(sums["ot_entropy_max"], ot_stats["ot_entropy_max"])
            sums["ot_max_prob_mean"] += ot_stats["ot_max_prob_mean"] * bsz
            sums["ot_max_prob_min"] = (
                ot_stats["ot_max_prob_min"]
                if total == bsz
                else min(sums["ot_max_prob_min"], ot_stats["ot_max_prob_min"])
            )
            sums["ot_max_prob_max"] = max(sums["ot_max_prob_max"], ot_stats["ot_max_prob_max"])
        out = {key: val / max(1, total) for key, val in sums.items()}
        out["ot_entropy_max"] = sums["ot_entropy_max"]
        out["ot_max_prob_min"] = sums["ot_max_prob_min"]
        out["ot_max_prob_max"] = sums["ot_max_prob_max"]
        out["assignment_counts"] = total_counts.detach().cpu().tolist()
        out["ema_update_counts"] = total_ema_counts.detach().cpu().tolist()
        out["selected_for_ema_counts"] = total_selected_for_ema_counts.detach().cpu().tolist()
        out["hard_assignment_counts"] = total_hard_counts.detach().cpu().tolist()
        out["class_ot_max_prob"] = self._max_prob_quantiles(max_probs_by_class)
        out["top_ratio_active"] = bool(use_top_ratio_now)
        return out

    @torch.no_grad()
    def evaluate(self, loader: DataLoader, split_name: str | None = None) -> dict[str, float]:
        self.model.eval()
        y_true, y_pred, losses, sample_ids, texts = [], [], [], [], []
        for batch in loader:
            labels = batch["labels"].to(self.device)
            outputs = self.model(
                input_ids=batch["input_ids"].to(self.device),
                attention_mask=batch["attention_mask"].to(self.device),
                mask_pos=batch["mask_pos"].to(self.device),
            )
            losses.append(float(F.cross_entropy(outputs["logits"], labels, weight=self.ce_weights).cpu()) * labels.size(0))
            y_true.extend(labels.cpu().tolist())
            y_pred.extend(outputs["logits"].argmax(dim=-1).cpu().tolist())
            sample_ids.extend(ex.sample_id for ex in batch["examples"])
            texts.extend(ex.text for ex in batch["examples"])
        if not y_true:
            return {
                "accuracy": 0.0,
                "macro_precision": 0.0,
                "macro_recall": 0.0,
                "macro_f1": 0.0,
                "weighted_precision": 0.0,
                "weighted_recall": 0.0,
                "weighted_f1": 0.0,
                "loss_ce": 0.0,
            }
        metrics = classification_metrics(y_true, y_pred)
        metrics["loss_ce"] = sum(losses) / max(1, len(y_true))
        metrics.update(self._per_class_f1(y_true, y_pred))
        metrics.update(self._confusion_pair_counts(y_true, y_pred))
        if split_name:
            self._save_detailed_eval(split_name, y_true, y_pred, sample_ids, texts)
        return metrics

    def _build_ce_weights(self) -> torch.Tensor:
        weights = torch.ones(len(self.label2id), dtype=torch.float)
        if "happy" in self.label2id:
            weights[self.label2id["happy"]] *= float(self.args.happy_ce_weight)
        return weights

    def _build_class_thresholds(self) -> torch.Tensor:
        thresholds = torch.full((len(self.label2id),), float(self.args.default_ema_conf_threshold))
        for name in self.args.low_conf_classes.split(","):
            name = name.strip()
            if name in self.label2id:
                thresholds[self.label2id[name]] = float(self.args.low_conf_threshold)
        if "happy" in self.label2id:
            thresholds[self.label2id["happy"]] = float(self.args.happy_conf_threshold)
        return thresholds

    def _build_top_ratio_class_ids(self, class_names: str) -> set[int]:
        class_ids = set()
        for name in class_names.split(","):
            name = name.strip()
            if name in self.label2id:
                class_ids.add(self.label2id[name])
        return class_ids

    def _early_stop_metric(self, dev_stats: dict[str, float], test_stats: dict[str, float]) -> float:
        if self.args.early_stop_metric == "dev_weighted_f1":
            return dev_stats["weighted_f1"]
        if self.args.early_stop_metric == "dev_macro_f1":
            return dev_stats["macro_f1"]
        if self.args.early_stop_metric == "test_weighted_f1":
            return test_stats["weighted_f1"]
        raise ValueError(self.args.early_stop_metric)

    def _ot_hard_diagnostics(
        self,
        labels: torch.Tensor,
        soft_targets: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[int, torch.Tensor]]:
        cnum, mnum = self.args.num_classes, self.args.num_subanchors
        target = soft_targets.reshape(-1, cnum, mnum)
        hard_counts = torch.zeros(cnum, mnum, device=soft_targets.device)
        max_probs_by_class = {}
        for cls in range(cnum):
            idx = torch.where(labels == cls)[0]
            if idx.numel() == 0:
                continue
            gamma_cls = target[idx, cls]
            max_prob, hard_sub = gamma_cls.max(dim=1)
            max_probs_by_class[cls] = max_prob
            hard_counts[cls].scatter_add_(0, hard_sub, torch.ones_like(max_prob))
        return hard_counts, max_probs_by_class

    def _max_prob_quantiles(self, max_probs_by_class: dict[int, list[torch.Tensor]]) -> dict[int, dict[str, float]]:
        stats = {}
        for cls, chunks in max_probs_by_class.items():
            if not chunks:
                stats[cls] = {"mean": 0.0, "min": 0.0, "p50": 0.0, "p70": 0.0, "p90": 0.0, "max": 0.0}
                continue
            values = torch.cat(chunks).float()
            stats[cls] = {
                "mean": float(values.mean().item()),
                "min": float(values.min().item()),
                "p50": float(torch.quantile(values, 0.50).item()),
                "p70": float(torch.quantile(values, 0.70).item()),
                "p90": float(torch.quantile(values, 0.90).item()),
                "max": float(values.max().item()),
            }
        return stats

    def _build_intensity_targets(self) -> torch.Tensor:
        targets = torch.zeros(len(self.label2id), dtype=torch.long)
        for name, idx in self.label2id.items():
            targets[idx] = INTENSITY_MAP.get(name, 1)
        return targets

    def _per_class_f1(self, y_true, y_pred) -> dict[str, float]:
        labels = sorted(self.id2label)
        scores = f1_score(y_true, y_pred, labels=labels, average=None, zero_division=0)
        return {f"{self.id2label[idx]}_f1": float(score) for idx, score in zip(labels, scores)}

    def _confusion_pair_counts(self, y_true, y_pred) -> dict[str, int]:
        counts = {}
        for src, dst in CONFUSION_LOG_PAIRS:
            key = f"confusion_{src}_to_{dst}"
            if src not in self.label2id or dst not in self.label2id:
                counts[key] = 0
                continue
            src_id = self.label2id[src]
            dst_id = self.label2id[dst]
            counts[key] = int(sum(1 for gold, pred in zip(y_true, y_pred) if gold == src_id and pred == dst_id))
        return counts

    def _build_optimizer(self):
        encoder_params, other_params = [], []
        if self.args.freeze_encoder:
            for param in self.model.encoder.parameters():
                param.requires_grad = False
        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            if name.startswith("encoder."):
                encoder_params.append(param)
            else:
                other_params.append(param)
        return torch.optim.AdamW(
            [
                {"params": encoder_params, "lr": self.args.plm_lr},
                {"params": other_params, "lr": self.args.other_lr},
            ],
            weight_decay=self.args.weight_decay,
        )

    def _linear_warmup_decay(self, warmup_ratio: float, total_steps: int):
        warmup_steps = int(total_steps * warmup_ratio)

        def lr_lambda(step: int) -> float:
            if step < warmup_steps:
                return float(step + 1) / float(max(1, warmup_steps))
            return max(0.0, float(total_steps - step) / float(max(1, total_steps - warmup_steps)))

        return lr_lambda

    def _init_logs(self) -> None:
        (self.output_dir / "metrics.jsonl").write_text("", encoding="utf-8")
        with (self.output_dir / "epoch_metrics.csv").open("w", encoding="utf-8", newline="") as f:
            csv.DictWriter(f, fieldnames=self._csv_fields()).writeheader()
        (self.output_dir / "epoch_results.txt").write_text(
            f"HDSA-ERC training started at {_now()}\n"
            f"experiment_name: {self.args.experiment_name}\n"
            f"dataset_dir: {self.args.dataset_dir}\n"
            f"bert_path: {self.args.bert_path}\n"
            f"domain_anchor_path: {self.args.domain_anchor_path}\n",
            encoding="utf-8",
        )
        loaded_stats = anchor_similarity_stats(self.model.get_anchors().detach().cpu())
        self._append_text("Loaded anchor stats:")
        for key, value in loaded_stats.items():
            self._append_text(f"loaded_{key}={value:.6f}")

    def _write_epoch(self, row: dict) -> None:
        with (self.output_dir / "metrics.jsonl").open("a", encoding="utf-8") as f:
            f.write(json.dumps(row) + "\n")
            f.flush()
        flat = {"epoch": row["epoch"]}
        for split in ["train", "dev", "test"]:
            for key, value in row[split].items():
                flat[f"{split}_{key}"] = value
        for key, value in row["anchor_stats"].items():
            flat[f"anchor_{key}"] = value
        with (self.output_dir / "epoch_metrics.csv").open("a", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=self._csv_fields(), extrasaction="ignore")
            writer.writerow(flat)
            f.flush()
        self._append_text(self._format_epoch(row))
        print(json.dumps(row, indent=2), flush=True)

    def _format_epoch(self, row: dict) -> str:
        train, dev, test = row["train"], row["dev"], row["test"]
        lines = [
            f"[{_now()}] epoch {row['epoch']} finished",
            (
                f"  train loss_total={train['loss_total']:.6f} loss_ce={train['loss_ce']:.6f} "
                f"loss_proto={train['loss_proto']:.6f} loss_compact={train['loss_compact']:.6f} "
                f"loss_pair={train['loss_pair']:.6f} loss_pair_anchor={train['loss_pair_anchor']:.6f} "
                f"loss_intensity={train['loss_intensity']:.6f}"
            ),
            (
                f"  ot entropy_mean={train['ot_entropy_mean']:.6f} entropy_max={train['ot_entropy_max']:.6f} "
                f"max_prob_mean={train['ot_max_prob_mean']:.6f} "
                f"max_prob_min={train['ot_max_prob_min']:.6f} max_prob_max={train['ot_max_prob_max']:.6f}"
            ),
            f"  top_ratio_active={train.get('top_ratio_active', False)}",
            (
                f"  dev  acc={dev['accuracy']:.6f} precision={dev['weighted_precision']:.6f} "
                f"recall={dev['weighted_recall']:.6f} weighted_f1={dev['weighted_f1']:.6f} "
                f"macro_f1={dev['macro_f1']:.6f}"
            ),
            (
                f"  test acc={test['accuracy']:.6f} precision={test['weighted_precision']:.6f} "
                f"recall={test['weighted_recall']:.6f} weighted_f1={test['weighted_f1']:.6f} "
                f"macro_f1={test['macro_f1']:.6f}"
            ),
            f"  anchor_stats={row['anchor_stats']}",
            f"  test class_f1={self._select_metrics(test, [f'{name}_f1' for name in self.label2id])}",
            f"  test confusion_pairs={self._select_metrics(test, [f'confusion_{a}_to_{b}' for a, b in CONFUSION_LOG_PAIRS])}",
            f"  best={row['best']}",
        ]
        lines.append("  class_ot_max_prob:")
        for label, stats in row["class_ot_max_prob"].items():
            lines.append(f"    {label}: {stats}")
        for label, counts in row["ot_assignment_counts"].items():
            lines.append(f"  class {label} assignment: {counts}")
        for label, counts in row["hard_assignment_counts"].items():
            lines.append(f"  class {label} hard_assignment: {counts}")
        for label, counts in row["selected_for_ema_counts"].items():
            lines.append(f"  class {label} selected_for_ema: {counts}")
        for label, counts in row["ema_update_counts"].items():
            lines.append(f"  class {label} ema_update: {counts}")
        return "\n".join(lines)

    def _append_text(self, text: str) -> None:
        with (self.output_dir / "epoch_results.txt").open("a", encoding="utf-8") as f:
            f.write(text.rstrip() + "\n")
            f.flush()

    def _named_counts(self, counts) -> dict[str, list[float]]:
        return {self.id2label[idx]: [round(float(v), 4) for v in counts[idx]] for idx in sorted(self.id2label)}

    def _named_stats(self, stats: dict[int, dict[str, float]]) -> dict[str, dict[str, float]]:
        return {
            self.id2label[idx]: {key: round(float(value), 6) for key, value in stats.get(idx, {}).items()}
            for idx in sorted(self.id2label)
        }

    def _select_metrics(self, metrics: dict, keys: list[str]) -> dict:
        return {key: metrics[key] for key in keys if key in metrics}

    def _save_checkpoint(self, name: str, epoch: int, metrics: dict) -> None:
        ckpt = {
            "model_state_dict": self.model.state_dict(),
            "model_config": asdict(self.model.config),
            "args": vars(self.args),
            "label2id": self.label2id,
            "id2label": self.id2label,
            "anchors": self.model.get_anchors().detach().cpu(),
            "epoch": epoch,
            "metrics": metrics,
        }
        if self.args.save_optimizer:
            ckpt["optimizer_state_dict"] = self.optimizer.state_dict()
            ckpt["scheduler_state_dict"] = self.scheduler.state_dict()
        safe_torch_save(ckpt, self.output_dir / name)

    def _save_detailed_eval(self, split_name: str, y_true, y_pred, sample_ids, texts) -> None:
        labels = sorted(self.id2label)
        target_names = [self.id2label[idx] for idx in labels]
        report = classification_report(
            y_true,
            y_pred,
            labels=labels,
            target_names=target_names,
            output_dict=True,
            zero_division=0,
        )
        matrix = confusion_matrix(y_true, y_pred, labels=labels).tolist()
        payload = {"classification_report": report, "confusion_matrix": {"labels": target_names, "matrix": matrix}}
        (self.output_dir / f"{split_name}_classification_report.json").write_text(
            json.dumps(payload, indent=2),
            encoding="utf-8",
        )
        with (self.output_dir / f"{split_name}_predictions.csv").open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["sample_id", "gold", "pred", "correct", "text"])
            writer.writeheader()
            for sid, gold, pred, text in zip(sample_ids, y_true, y_pred, texts):
                writer.writerow(
                    {
                        "sample_id": sid,
                        "gold": self.id2label[int(gold)],
                        "pred": self.id2label[int(pred)],
                        "correct": int(gold == pred),
                        "text": text,
                    }
                )

    def _csv_fields(self) -> list[str]:
        fields = ["epoch"]
        for key in [
            "loss_total",
            "loss_ce",
            "loss_proto",
            "loss_compact",
            "loss_pair",
            "loss_pair_anchor",
            "loss_intensity",
            "ot_entropy_mean",
            "ot_entropy_max",
            "ot_max_prob_mean",
            "ot_max_prob_min",
            "ot_max_prob_max",
        ]:
            fields.append(f"train_{key}")
        for split in ["dev", "test"]:
            for key in [
                "accuracy",
                "macro_precision",
                "macro_recall",
                "macro_f1",
                "weighted_precision",
                "weighted_recall",
                "weighted_f1",
                "loss_ce",
            ]:
                fields.append(f"{split}_{key}")
            for name in sorted(self.label2id):
                fields.append(f"{split}_{name}_f1")
            for src, dst in CONFUSION_LOG_PAIRS:
                fields.append(f"{split}_confusion_{src}_to_{dst}")
        for key in [
            "same_anchor_sim_mean",
            "same_anchor_sim_max",
            "diff_anchor_sim_mean",
            "diff_anchor_sim_max",
        ]:
            fields.append(f"anchor_{key}")
        return fields


def _now() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def safe_torch_save(obj, path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    torch.save(obj, tmp_path)
    os.replace(tmp_path, path)
