from __future__ import annotations

import argparse
from pathlib import Path

import torch
from transformers import AutoTokenizer

from .data import ERCExample, ERCCollator
from .model import CHDSConfig, CHDSERCModel


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Predict one utterance with a trained CHDS-ERC checkpoint.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--speaker", required=True)
    parser.add_argument("--text", required=True)
    parser.add_argument(
        "--history",
        nargs="*",
        default=[],
        help='History turns as "Speaker::text".',
    )
    parser.add_argument("--max_length", type=int, default=256)
    parser.add_argument("--local_files_only", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ckpt = torch.load(args.checkpoint, map_location="cpu")
    cfg_dict = ckpt["model_config"]
    cfg_dict["local_files_only"] = args.local_files_only or cfg_dict.get("local_files_only", False)
    cfg = CHDSConfig(**cfg_dict)
    tokenizer = AutoTokenizer.from_pretrained(
        cfg.model_name_or_path,
        local_files_only=cfg.local_files_only,
        use_fast=True,
    )
    model = CHDSERCModel(cfg, anchors=ckpt["anchors"])
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    history = []
    for item in args.history:
        if "::" not in item:
            raise ValueError(f'History item must use "Speaker::text": {item}')
        speaker, text = item.split("::", 1)
        history.append((speaker.strip(), text.strip()))

    example = ERCExample(
        sample_id="cli",
        dialogue_id="cli",
        utterance_id="0",
        speaker=args.speaker,
        text=args.text,
        label="unknown",
        history=tuple(history),
    )
    collator = ERCCollator(tokenizer, max_length=args.max_length)
    batch = collator([{"example": example, "label": 0}])

    with torch.no_grad():
        out = model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            mask_pos=batch["mask_pos"],
        )
        probs = out["logits"].softmax(dim=-1).squeeze(0)

    id2label = {int(k): v for k, v in ckpt["id2label"].items()} if isinstance(next(iter(ckpt["id2label"])), str) else ckpt["id2label"]
    pred_id = int(probs.argmax())
    print(f"prediction: {id2label[pred_id]}")
    for idx, prob in enumerate(probs.tolist()):
        print(f"{id2label[idx]}\t{prob:.6f}")


if __name__ == "__main__":
    main()

