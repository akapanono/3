from __future__ import annotations

import argparse
import csv
import json
import random
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import classification_report, confusion_matrix
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from .anchors import (
    anchor_assignment_counts,
    anchor_similarity_stats,
    build_cluster_anchors,
    extract_representations,
)
from .data import ERCCollator, ERCDataset, build_label_maps, load_erc_split
from .losses import (
    anchor_pull_loss,
    hyperspherical_inter_anchor_loss,
    intra_anchor_diversity_loss,
    supervised_contrastive_loss,
)
from .metrics import classification_metrics
from .model import CHDSConfig, CHDSERCModel


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train CHDS-ERC.")
    parser.add_argument("--dataset_dir", default="data/IEMOCAP")
    parser.add_argument("--model_name_or_path", default="princeton-nlp/sup-simcse-roberta-large")
    parser.add_argument("--output_dir", default="outputs/chds_erc_iemocap")
    parser.add_argument("--local_files_only", action="store_true")
    parser.add_argument("--max_length", type=int, default=256)
    parser.add_argument("--context_window", type=int, default=12)
    parser.add_argument("--include_target_in_context", action="store_true", default=True)
    parser.add_argument("--num_subanchors", type=int, default=3)
    parser.add_argument("--anchor_dim", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--anchor_logit_weight", type=float, default=0.5)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--eval_batch_size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--plm_lr", type=float, default=1e-5)
    parser.add_argument("--other_lr", type=float, default=4e-4)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--warmup_ratio", type=float, default=0.1)
    parser.add_argument("--max_grad_norm", type=float, default=5.0)
    parser.add_argument("--lambda_ce", type=float, default=0.3)
    parser.add_argument("--lambda_sup", type=float, default=0.7)
    parser.add_argument("--lambda_pull", type=float, default=0.1)
    parser.add_argument("--lambda_inter", type=float, default=0.05)
    parser.add_argument("--lambda_intra", type=float, default=0.01)
    parser.add_argument("--same_upper", type=float, default=0.85)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--freeze_encoder", action="store_true")
    parser.add_argument("--skip_anchor_init", action="store_true")
    parser.add_argument("--max_train_samples", type=int)
    parser.add_argument("--max_dev_samples", type=int)
    parser.add_argument("--max_test_samples", type=int)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    args.model_name_or_path = resolve_model_path(args.model_name_or_path)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_examples = load_erc_split(args.dataset_dir, "train", args.context_window, args.max_train_samples)
    dev_examples = load_erc_split(args.dataset_dir, "dev", args.context_window, args.max_dev_samples)
    test_examples = load_erc_split(args.dataset_dir, "test", args.context_window, args.max_test_samples)
    label2id, id2label = build_label_maps(train_examples)
    dev_examples = filter_known_labels(dev_examples, label2id, "dev")
    test_examples = filter_known_labels(test_examples, label2id, "test")

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path,
        local_files_only=args.local_files_only,
        use_fast=True,
    )
    collator = ERCCollator(
        tokenizer,
        max_length=args.max_length,
        include_target_in_context=args.include_target_in_context,
    )

    train_loader = make_loader(train_examples, label2id, collator, args.batch_size, shuffle=True)
    train_eval_loader = make_loader(train_examples, label2id, collator, args.eval_batch_size, shuffle=False)
    dev_loader = make_loader(dev_examples, label2id, collator, args.eval_batch_size, shuffle=False)
    test_loader = make_loader(test_examples, label2id, collator, args.eval_batch_size, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = CHDSConfig(
        model_name_or_path=args.model_name_or_path,
        num_classes=len(label2id),
        num_subanchors=args.num_subanchors,
        anchor_dim=args.anchor_dim,
        dropout=args.dropout,
        temperature=args.temperature,
        anchor_logit_weight=args.anchor_logit_weight,
        local_files_only=args.local_files_only,
    )
    model = CHDSERCModel(cfg).to(device)
    if args.freeze_encoder:
        for param in model.encoder.parameters():
            param.requires_grad = False

    if not args.skip_anchor_init:
        print("Extracting train representations for KMeans anchor initialization...")
        reps, labels = extract_representations(model, train_eval_loader, device)
        model.set_anchors(
            build_cluster_anchors(
                reps,
                labels,
                num_classes=len(label2id),
                num_subanchors=args.num_subanchors,
                random_state=args.seed,
            )
        )
        save_anchor_report(output_dir, reps, labels, model.anchors.detach().cpu(), id2label, "initial")

    optimizer = build_optimizer(model, args)
    total_steps = max(1, len(train_loader) * args.epochs)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=linear_warmup_decay(args.warmup_ratio, total_steps),
    )

    metadata = {
        "args": vars(args),
        "model_config": asdict(cfg),
        "label2id": label2id,
        "id2label": id2label,
        "num_train": len(train_examples),
        "num_dev": len(dev_examples),
        "num_test": len(test_examples),
    }
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    best_dev = -1.0
    best_epoch = 0
    best_test_stats: dict[str, float] | None = None
    metrics_path = output_dir / "metrics.jsonl"
    csv_path = output_dir / "epoch_metrics.csv"
    text_log_path = output_dir / "epoch_results.txt"
    init_epoch_logs(metrics_path, csv_path, text_log_path, args)
    for epoch in range(1, args.epochs + 1):
        append_text_log(text_log_path, f"\n[{now()}] epoch {epoch}/{args.epochs} started")
        train_stats = train_one_epoch(model, train_loader, optimizer, scheduler, device, args)
        dev_stats = evaluate(model, dev_loader, device)
        test_stats = evaluate(model, test_loader, device)
        row = {
            "epoch": epoch,
            "train": train_stats,
            "dev": dev_stats,
            "test": test_stats,
            "anchor_stats": anchor_similarity_stats(model.anchors.detach().cpu()),
        }
        with metrics_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row) + "\n")
            f.flush()
        append_epoch_csv(csv_path, row)
        append_text_log(text_log_path, format_epoch_text(row))
        print(json.dumps(row, indent=2), flush=True)

        if dev_stats["weighted_f1"] > best_dev:
            best_dev = dev_stats["weighted_f1"]
            best_epoch = epoch
            best_test_stats = test_stats
            save_checkpoint(output_dir / "best_model.pt", model, cfg, args, label2id, id2label, epoch, dev_stats)
            append_text_log(text_log_path, f"[{now()}] saved new best_model.pt at epoch {epoch}")

    reps, labels = extract_representations(model, train_eval_loader, device)
    save_anchor_report(output_dir, reps, labels, model.anchors.detach().cpu(), id2label, "final")
    save_checkpoint(output_dir / "last_model.pt", model, cfg, args, label2id, id2label, args.epochs, test_stats)
    final_dev = evaluate(model, dev_loader, device, id2label=id2label, output_dir=output_dir, split_name="dev")
    final_test = evaluate(model, test_loader, device, id2label=id2label, output_dir=output_dir, split_name="test")
    final_summary = {
        "best_epoch": best_epoch,
        "best_dev_weighted_f1": best_dev,
        "best_epoch_test_metrics": best_test_stats,
        "last_epoch": args.epochs,
        "final_dev": final_dev,
        "final_test": final_test,
        "finished_at": now(),
    }
    (output_dir / "final_metrics.json").write_text(
        json.dumps(final_summary, indent=2),
        encoding="utf-8",
    )
    append_text_log(text_log_path, "\nTraining finished.")
    append_text_log(text_log_path, json.dumps(final_summary, indent=2))


def make_loader(examples, label2id, collator, batch_size, shuffle) -> DataLoader:
    return DataLoader(
        ERCDataset(examples, label2id),
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collator,
    )


def resolve_model_path(model_name_or_path: str) -> str:
    if Path(model_name_or_path).exists():
        return model_name_or_path
    if model_name_or_path == "princeton-nlp/sup-simcse-roberta-large":
        candidates = [
            Path(r"E:\AI_learn_model\ERC\EACL\pretrained\sup-simcse-roberta-large"),
            Path(r"E:\AI_learn_model\ERC\EACL\emo_anchors\sup-simcse-roberta-large"),
        ]
        for candidate in candidates:
            if candidate.exists():
                print(f"Using local Sup-SimCSE path: {candidate}")
                return str(candidate)
    return model_name_or_path


def filter_known_labels(examples, label2id, split_name):
    filtered = [ex for ex in examples if ex.label in label2id]
    dropped = len(examples) - len(filtered)
    if dropped:
        print(f"Dropped {dropped} {split_name} examples with labels absent from train label map.")
    return filtered


def train_one_epoch(model, loader, optimizer, scheduler, device, args) -> dict[str, float]:
    model.train()
    sums: dict[str, float] = {
        "loss": 0.0,
        "loss_ce": 0.0,
        "loss_supcon": 0.0,
        "loss_pull": 0.0,
        "loss_inter": 0.0,
        "loss_intra": 0.0,
    }
    total = 0
    for batch in loader:
        batch_size = batch["labels"].size(0)
        labels = batch["labels"].to(device)
        out = model(
            input_ids=batch["input_ids"].to(device),
            attention_mask=batch["attention_mask"].to(device),
            mask_pos=batch["mask_pos"].to(device),
        )
        anchors = out["anchors"]
        loss_ce = F.cross_entropy(out["logits"], labels)
        loss_supcon = supervised_contrastive_loss(out["z"], labels, anchors, args.temperature)
        loss_pull = anchor_pull_loss(out["z"], labels, anchors)
        loss_inter = hyperspherical_inter_anchor_loss(anchors)
        loss_intra = intra_anchor_diversity_loss(anchors, args.same_upper)
        loss = (
            args.lambda_ce * loss_ce
            + args.lambda_sup * loss_supcon
            + args.lambda_pull * loss_pull
            + args.lambda_inter * loss_inter
            + args.lambda_intra * loss_intra
        )
        loss.backward()
        clip_grad_norm_(model.parameters(), args.max_grad_norm)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad(set_to_none=True)

        total += batch_size
        for name, value in [
            ("loss", loss),
            ("loss_ce", loss_ce),
            ("loss_supcon", loss_supcon),
            ("loss_pull", loss_pull),
            ("loss_inter", loss_inter),
            ("loss_intra", loss_intra),
        ]:
            sums[name] += float(value.detach().cpu()) * batch_size
    return {name: val / max(1, total) for name, val in sums.items()}


@torch.no_grad()
def evaluate(
    model,
    loader,
    device,
    id2label: dict[int, str] | None = None,
    output_dir: Path | None = None,
    split_name: str | None = None,
) -> dict[str, float]:
    model.eval()
    y_true = []
    y_pred = []
    losses = []
    sample_ids = []
    texts = []
    for batch in loader:
        labels = batch["labels"].to(device)
        out = model(
            input_ids=batch["input_ids"].to(device),
            attention_mask=batch["attention_mask"].to(device),
            mask_pos=batch["mask_pos"].to(device),
        )
        losses.append(float(F.cross_entropy(out["logits"], labels).cpu()) * labels.size(0))
        y_true.extend(labels.cpu().tolist())
        y_pred.extend(out["logits"].argmax(dim=-1).cpu().tolist())
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
    if id2label is not None and output_dir is not None and split_name is not None:
        save_detailed_eval(output_dir, split_name, y_true, y_pred, sample_ids, texts, id2label)
    return metrics


def build_optimizer(model, args):
    encoder_params = []
    other_params = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if name.startswith("encoder."):
            encoder_params.append(param)
        else:
            other_params.append(param)
    return torch.optim.AdamW(
        [
            {"params": encoder_params, "lr": args.plm_lr},
            {"params": other_params, "lr": args.other_lr},
        ],
        weight_decay=args.weight_decay,
    )


def linear_warmup_decay(warmup_ratio: float, total_steps: int):
    warmup_steps = int(total_steps * warmup_ratio)

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return float(step + 1) / float(max(1, warmup_steps))
        return max(0.0, float(total_steps - step) / float(max(1, total_steps - warmup_steps)))

    return lr_lambda


def save_checkpoint(path, model, cfg, args, label2id, id2label, epoch, metrics) -> None:
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "model_config": asdict(cfg),
            "args": vars(args),
            "label2id": label2id,
            "id2label": id2label,
            "anchors": model.anchors.detach().cpu(),
            "epoch": epoch,
            "metrics": metrics,
        },
        path,
    )


def save_anchor_report(output_dir, reps, labels, anchors, id2label, prefix) -> None:
    counts = anchor_assignment_counts(reps, labels, anchors)
    named_counts = {id2label[int(cls)]: vals for cls, vals in counts.items()}
    report = {
        "anchor_stats": anchor_similarity_stats(anchors),
        "assignment_counts": named_counts,
    }
    (output_dir / f"{prefix}_anchor_report.json").write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )


def init_epoch_logs(metrics_path: Path, csv_path: Path, text_log_path: Path, args) -> None:
    metrics_path.write_text("", encoding="utf-8")
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=epoch_csv_fields())
        writer.writeheader()
    text_log_path.write_text(
        f"CHDS-ERC training started at {now()}\n"
        f"dataset_dir: {args.dataset_dir}\n"
        f"model_name_or_path: {args.model_name_or_path}\n"
        f"output_dir: {args.output_dir}\n",
        encoding="utf-8",
    )


def append_epoch_csv(csv_path: Path, row: dict) -> None:
    flat = {"epoch": row["epoch"]}
    for split in ["train", "dev", "test"]:
        for key, value in row[split].items():
            flat[f"{split}_{key}"] = value
    for key, value in row["anchor_stats"].items():
        flat[f"anchor_{key}"] = value
    with csv_path.open("a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=epoch_csv_fields(), extrasaction="ignore")
        writer.writerow(flat)
        f.flush()


def epoch_csv_fields() -> list[str]:
    fields = ["epoch"]
    train_keys = ["loss", "loss_ce", "loss_supcon", "loss_pull", "loss_inter", "loss_intra"]
    eval_keys = [
        "accuracy",
        "macro_precision",
        "macro_recall",
        "macro_f1",
        "weighted_precision",
        "weighted_recall",
        "weighted_f1",
        "loss_ce",
    ]
    for key in train_keys:
        fields.append(f"train_{key}")
    for split in ["dev", "test"]:
        for key in eval_keys:
            fields.append(f"{split}_{key}")
    for key in [
        "same_anchor_sim_mean",
        "same_anchor_sim_max",
        "diff_anchor_sim_mean",
        "diff_anchor_sim_max",
    ]:
        fields.append(f"anchor_{key}")
    return fields


def format_epoch_text(row: dict) -> str:
    dev = row["dev"]
    test = row["test"]
    train = row["train"]
    return (
        f"[{now()}] epoch {row['epoch']} finished\n"
        f"  train loss={train['loss']:.6f} ce={train['loss_ce']:.6f} "
        f"supcon={train['loss_supcon']:.6f} pull={train['loss_pull']:.6f} "
        f"inter={train['loss_inter']:.6f} intra={train['loss_intra']:.6f}\n"
        f"  dev  acc={dev['accuracy']:.6f} precision={dev['weighted_precision']:.6f} "
        f"recall={dev['weighted_recall']:.6f} weighted_f1={dev['weighted_f1']:.6f} "
        f"macro_f1={dev['macro_f1']:.6f}\n"
        f"  test acc={test['accuracy']:.6f} precision={test['weighted_precision']:.6f} "
        f"recall={test['weighted_recall']:.6f} weighted_f1={test['weighted_f1']:.6f} "
        f"macro_f1={test['macro_f1']:.6f}"
    )


def append_text_log(path: Path, text: str) -> None:
    with path.open("a", encoding="utf-8") as f:
        f.write(text.rstrip() + "\n")
        f.flush()


def save_detailed_eval(
    output_dir: Path,
    split_name: str,
    y_true: list[int],
    y_pred: list[int],
    sample_ids: list[str],
    texts: list[str],
    id2label: dict[int, str],
) -> None:
    labels = sorted(id2label)
    target_names = [id2label[idx] for idx in labels]
    report = classification_report(
        y_true,
        y_pred,
        labels=labels,
        target_names=target_names,
        output_dict=True,
        zero_division=0,
    )
    matrix = confusion_matrix(y_true, y_pred, labels=labels).tolist()
    payload = {
        "classification_report": report,
        "confusion_matrix": {
            "labels": target_names,
            "matrix": matrix,
        },
    }
    (output_dir / f"{split_name}_classification_report.json").write_text(
        json.dumps(payload, indent=2),
        encoding="utf-8",
    )
    with (output_dir / f"{split_name}_predictions.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["sample_id", "gold", "pred", "correct", "text"],
        )
        writer.writeheader()
        for sid, gold, pred, text in zip(sample_ids, y_true, y_pred, texts):
            writer.writerow(
                {
                    "sample_id": sid,
                    "gold": id2label[int(gold)],
                    "pred": id2label[int(pred)],
                    "correct": int(gold == pred),
                    "text": text,
                }
            )


def now() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


if __name__ == "__main__":
    main()
