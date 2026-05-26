from __future__ import annotations

import argparse
import sys
from dataclasses import asdict
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.anchors.anchor_utils import build_domain_anchors, default_dataset_dir, extract_representations
from src.data import ERCCollator, ERCDataset, build_label_maps, load_erc_split
from src.model import HDSAConfig, HDSAERCModel


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate class-wise KMeans domain sub-anchors.")
    parser.add_argument("--dataset_name", default="IEMOCAP")
    parser.add_argument("--dataset_dir")
    parser.add_argument("--bert_path", required=True)
    parser.add_argument("--checkpoint_path")
    parser.add_argument("--output_path", required=True)
    parser.add_argument("--num_subanchors", type=int, default=3)
    parser.add_argument("--anchor_dim", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--max_length", type=int, default=256)
    parser.add_argument("--context_window", type=int, default=12)
    parser.add_argument("--local_files_only", action="store_true")
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--max_train_samples", type=int)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset_dir = args.dataset_dir or default_dataset_dir(args.dataset_name)
    examples = load_erc_split(dataset_dir, "train", args.context_window, args.max_train_samples)
    label2id, id2label = build_label_maps(examples)
    tokenizer = AutoTokenizer.from_pretrained(args.bert_path, local_files_only=args.local_files_only, use_fast=True)
    loader = DataLoader(
        ERCDataset(examples, label2id),
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=ERCCollator(tokenizer, max_length=args.max_length),
    )
    config = HDSAConfig(
        bert_path=args.bert_path,
        num_classes=len(label2id),
        num_subanchors=args.num_subanchors,
        anchor_dim=args.anchor_dim,
        local_files_only=args.local_files_only,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = HDSAERCModel(config).to(device)
    if args.checkpoint_path:
        obj = torch.load(args.checkpoint_path, map_location="cpu")
        state = obj.get("model_state_dict", obj)
        model.load_state_dict(state, strict=False)
    reps, labels = extract_representations(model, loader, device)
    anchors, counts = build_domain_anchors(reps, labels, len(label2id), args.num_subanchors, args.seed)
    out = {
        "anchors": anchors,
        "counts": counts,
        "label2id": label2id,
        "id2label": id2label,
        "anchor_dim": args.anchor_dim,
        "num_subanchors": args.num_subanchors,
        "model_config": asdict(config),
        "source": "class_wise_kmeans",
    }
    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(out, output_path)
    print(f"Saved KMeans domain anchors to {output_path}")


if __name__ == "__main__":
    main()
