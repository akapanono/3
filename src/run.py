from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data import ERCCollator, ERCDataset, build_label_maps, load_erc_split
from src.model import HDSAConfig, HDSAERCModel
from src.trainer import HDSATrainer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run HDSA-ERC training.")
    parser.add_argument("--use_hdsa", action="store_true", help="Kept for compatibility; this project now runs HDSA only.")
    parser.add_argument("--dataset_name", default="IEMOCAP")
    parser.add_argument("--dataset_dir")
    parser.add_argument("--bert_path", default="pretrained/sup-simcse-roberta-large")
    parser.add_argument("--domain_anchor_path")
    parser.add_argument("--output_dir", default="outputs/hdsa_erc_iemocap")
    parser.add_argument("--experiment_name", default="baseline")
    parser.add_argument("--local_files_only", action="store_true")
    parser.add_argument("--max_length", type=int, default=256)
    parser.add_argument("--context_window", type=int, default=12)
    parser.add_argument("--num_subanchors", type=int, default=3)
    parser.add_argument("--anchor_dim", type=int, default=256)
    parser.add_argument("--anchor_temperature", type=float, default=0.1)
    parser.add_argument("--proto_temperature", type=float, default=0.1)
    parser.add_argument("--anchor_logit_weight", type=float, default=0.3)
    parser.add_argument("--proto_loss_weight", type=float, default=0.5)
    parser.add_argument("--compact_loss_weight", type=float, default=0.1)
    parser.add_argument("--preserve_weight", type=float, default=0.5)
    parser.add_argument("--center_weight", type=float, default=0.1)
    parser.add_argument("--div_weight", type=float, default=1.0)
    parser.add_argument("--same_upper", type=float, default=0.90)
    parser.add_argument("--ot_epsilon", type=float, default=0.02)
    parser.add_argument("--ot_iters", type=int, default=50)
    parser.add_argument("--ot_sharpen_power", type=float, default=2.0)
    parser.add_argument("--prototype_momentum", type=float, default=0.95)
    parser.add_argument("--ema_conf_threshold", type=float, default=0.45)
    parser.add_argument("--class_adaptive_ema", action="store_true")
    parser.add_argument("--default_ema_conf_threshold", type=float, default=0.45)
    parser.add_argument("--low_conf_classes", type=str, default="angry,frustrated")
    parser.add_argument("--low_conf_threshold", type=float, default=0.40)
    parser.add_argument("--happy_conf_threshold", type=float, default=0.42)
    parser.add_argument("--use_ema_fallback", action="store_true")
    parser.add_argument("--fallback_momentum", type=float, default=0.995)
    parser.add_argument("--pair_loss_weight", type=float, default=0.0)
    parser.add_argument("--pair_margin", type=float, default=0.20)
    parser.add_argument("--pair_anchor_loss_weight", type=float, default=0.0)
    parser.add_argument("--pair_anchor_upper", type=float, default=0.20)
    parser.add_argument("--happy_ce_weight", type=float, default=1.0)
    parser.add_argument("--use_intensity_head", action="store_true")
    parser.add_argument("--intensity_loss_weight", type=float, default=0.0)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--eval_batch_size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--plm_lr", type=float, default=1e-5)
    parser.add_argument("--other_lr", type=float, default=4e-4)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--warmup_ratio", type=float, default=0.1)
    parser.add_argument("--max_grad_norm", type=float, default=5.0)
    parser.add_argument("--freeze_encoder", action="store_true")
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--max_train_samples", type=int)
    parser.add_argument("--max_dev_samples", type=int)
    parser.add_argument("--max_test_samples", type=int)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.active_ablations = active_ablations(args)
    if len(args.active_ablations) > 1:
        print(
            "Warning: more than one ablation is enabled. "
            f"Active ablations: {', '.join(args.active_ablations)}"
        )
    set_seed(args.seed)
    if not args.use_hdsa:
        print("Warning: old model code was removed; running HDSA-ERC.")
    args.dataset_dir = args.dataset_dir or str(Path("data") / args.dataset_name)
    train_examples = load_erc_split(args.dataset_dir, "train", args.context_window, args.max_train_samples)
    dev_examples = load_erc_split(args.dataset_dir, "dev", args.context_window, args.max_dev_samples)
    test_examples = load_erc_split(args.dataset_dir, "test", args.context_window, args.max_test_samples)
    label2id, id2label = build_label_maps(train_examples)
    dev_examples = [ex for ex in dev_examples if ex.label in label2id]
    test_examples = [ex for ex in test_examples if ex.label in label2id]
    args.num_classes = len(label2id)

    tokenizer = AutoTokenizer.from_pretrained(args.bert_path, local_files_only=args.local_files_only, use_fast=True)
    collator = ERCCollator(tokenizer, max_length=args.max_length)
    train_loader = make_loader(train_examples, label2id, collator, args.batch_size, shuffle=True)
    dev_loader = make_loader(dev_examples, label2id, collator, args.eval_batch_size, shuffle=False)
    test_loader = make_loader(test_examples, label2id, collator, args.eval_batch_size, shuffle=False)
    config = HDSAConfig(
        bert_path=args.bert_path,
        num_classes=len(label2id),
        num_subanchors=args.num_subanchors,
        anchor_dim=args.anchor_dim,
        anchor_temperature=args.anchor_temperature,
        anchor_logit_weight=args.anchor_logit_weight,
        domain_anchor_path=args.domain_anchor_path,
        local_files_only=args.local_files_only,
    )
    model = HDSAERCModel(config)
    trainer = HDSATrainer(args, model, train_loader, dev_loader, test_loader, label2id, id2label)
    trainer.train()


def active_ablations(args: argparse.Namespace) -> list[str]:
    ablations = []
    if args.class_adaptive_ema:
        ablations.append("class_adaptive_ema")
    if args.use_ema_fallback:
        ablations.append("ema_fallback")
    if args.pair_loss_weight > 0:
        ablations.append("pairwise_confusion_loss")
    if args.pair_anchor_loss_weight > 0:
        ablations.append("pairwise_anchor_loss")
    if abs(args.happy_ce_weight - 1.0) > 1e-8:
        ablations.append("happy_ce_weight")
    if args.use_intensity_head and args.intensity_loss_weight > 0:
        ablations.append("intensity_head")
    return ablations


def make_loader(examples, label2id, collator, batch_size, shuffle) -> DataLoader:
    return DataLoader(
        ERCDataset(examples, label2id),
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collator,
    )


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


if __name__ == "__main__":
    main()
