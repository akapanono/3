from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import torch


LABEL_ALIASES = {
    "ang": "angry",
    "exc": "excited",
    "fru": "frustrated",
    "hap": "happy",
    "neu": "neutral",
    "sad": "sad",
}


@dataclass(frozen=True)
class ERCExample:
    sample_id: str
    dialogue_id: str
    utterance_id: str
    speaker: str
    text: str
    label: str
    history: tuple[tuple[str, str], ...]

    def prompt(self, mask_token: str, include_target_in_context: bool = True) -> str:
        context = list(self.history)
        if include_target_in_context:
            context.append((self.speaker, self.text))
        lines = [f"{speaker} says: {text}" for speaker, text in context if text]
        lines.append(f"For utterance: {self.text} {self.speaker} feels {mask_token}")
        return "\n".join(lines)


class ERCDataset(torch.utils.data.Dataset):
    def __init__(self, examples: list[ERCExample], label2id: dict[str, int]):
        self.examples = examples
        self.label_ids = labels_to_ids(examples, label2id)

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        return {"example": self.examples[idx], "label": self.label_ids[idx]}


class ERCCollator:
    def __init__(self, tokenizer, max_length: int = 256, include_target_in_context: bool = True):
        if tokenizer.mask_token is None:
            raise ValueError("Tokenizer must provide a mask token.")
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.include_target_in_context = include_target_in_context

    def __call__(self, batch: list[dict[str, Any]]) -> dict[str, Any]:
        old_side = getattr(self.tokenizer, "truncation_side", "right")
        self.tokenizer.truncation_side = "left"
        prompts = [
            item["example"].prompt(
                self.tokenizer.mask_token,
                include_target_in_context=self.include_target_in_context,
            )
            for item in batch
        ]
        encoded = self.tokenizer(
            prompts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        self.tokenizer.truncation_side = old_side
        mask_positions = []
        for row in encoded["input_ids"]:
            pos = (row == self.tokenizer.mask_token_id).nonzero(as_tuple=False).flatten()
            if pos.numel() == 0:
                raise ValueError("Mask token was truncated from a prompt.")
            mask_positions.append(pos[-1])
        return {
            "input_ids": encoded["input_ids"],
            "attention_mask": encoded["attention_mask"],
            "mask_pos": torch.stack(mask_positions).long(),
            "labels": torch.tensor([item["label"] for item in batch], dtype=torch.long),
            "examples": [item["example"] for item in batch],
            "prompts": prompts,
        }


def load_erc_split(
    dataset_dir: str | Path,
    split: str,
    context_window: int = 12,
    max_samples: int | None = None,
) -> list[ERCExample]:
    dataset_dir = Path(dataset_dir)
    json_path = dataset_dir / f"{split}_data.json"
    csv_path = dataset_dir / f"{split}_data.csv"
    if json_path.exists():
        examples = _load_json(json_path, context_window)
    elif csv_path.exists():
        examples = _load_meld_csv(csv_path, context_window)
    else:
        raise FileNotFoundError(f"Cannot find {split}_data.json or {split}_data.csv in {dataset_dir}")
    return examples if max_samples is None else examples[:max_samples]


def build_label_maps(examples: Iterable[ERCExample]) -> tuple[dict[str, int], dict[int, str]]:
    labels = sorted({ex.label for ex in examples})
    label2id = {label: idx for idx, label in enumerate(labels)}
    return label2id, {idx: label for label, idx in label2id.items()}


def labels_to_ids(examples: list[ERCExample], label2id: dict[str, int]) -> list[int]:
    missing = sorted({ex.label for ex in examples if ex.label not in label2id})
    if missing:
        raise ValueError(f"Labels absent from label2id: {missing}")
    return [label2id[ex.label] for ex in examples]


def normalize_label(label: Any) -> str:
    label = str(label).strip().lower()
    return LABEL_ALIASES.get(label, label)


def _load_json(path: Path, context_window: int) -> list[ERCExample]:
    with path.open("r", encoding="utf-8") as f:
        obj = json.load(f)
    if isinstance(obj, list):
        return _load_iemocap(obj, path.stem, context_window)
    if isinstance(obj, dict) and "episodes" in obj:
        return _load_emory(obj, context_window)
    raise ValueError(f"Unsupported JSON format: {path}")


def _load_iemocap(dialogues: list[list[dict[str, Any]]], split_name: str, context_window: int) -> list[ERCExample]:
    examples = []
    for d_idx, dialogue in enumerate(dialogues):
        dialogue_id = _guess_dialogue_id(dialogue, f"{split_name}_{d_idx}")
        history: list[tuple[str, str]] = []
        for u_idx, utt in enumerate(dialogue):
            text = str(utt.get("text", "")).strip()
            speaker = str(utt.get("speaker", "Speaker")).strip() or "Speaker"
            label = utt.get("label")
            if text and label is not None:
                examples.append(
                    ERCExample(
                        sample_id=f"{dialogue_id}_u{u_idx}",
                        dialogue_id=dialogue_id,
                        utterance_id=str(u_idx),
                        speaker=speaker,
                        text=text,
                        label=normalize_label(label),
                        history=tuple(history[-context_window:]),
                    )
                )
            if text:
                history.append((speaker, text))
    return examples


def _load_emory(obj: dict[str, Any], context_window: int) -> list[ERCExample]:
    examples = []
    for episode in obj.get("episodes", []):
        episode_id = str(episode.get("episode_id", "episode"))
        for scene in episode.get("scenes", []):
            scene_id = str(scene.get("scene_id", episode_id))
            history: list[tuple[str, str]] = []
            for idx, utt in enumerate(scene.get("utterances", [])):
                text = str(utt.get("transcript", "")).strip()
                speakers = utt.get("speakers") or ["Speaker"]
                speaker = str(speakers[0]).strip() or "Speaker"
                label = utt.get("emotion")
                utterance_id = str(utt.get("utterance_id", idx))
                if text and label is not None:
                    examples.append(
                        ERCExample(
                            sample_id=f"{scene_id}_{utterance_id}",
                            dialogue_id=scene_id,
                            utterance_id=utterance_id,
                            speaker=speaker,
                            text=text,
                            label=normalize_label(label),
                            history=tuple(history[-context_window:]),
                        )
                    )
                if text:
                    history.append((speaker, text))
    return examples


def _load_meld_csv(path: Path, context_window: int) -> list[ERCExample]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    rows.sort(key=lambda r: (int(r["Dialogue_ID"]), int(r["Utterance_ID"])))
    examples = []
    cur_dialogue = None
    history: list[tuple[str, str]] = []
    for row in rows:
        dialogue_id = str(row["Dialogue_ID"])
        if dialogue_id != cur_dialogue:
            cur_dialogue = dialogue_id
            history = []
        text = str(row.get("Utterance", "")).strip()
        speaker = str(row.get("Speaker", "Speaker")).strip() or "Speaker"
        label = row.get("Emotion")
        utterance_id = str(row.get("Utterance_ID", len(history)))
        if text and label is not None:
            examples.append(
                ERCExample(
                    sample_id=f"d{dialogue_id}_u{utterance_id}",
                    dialogue_id=dialogue_id,
                    utterance_id=utterance_id,
                    speaker=speaker,
                    text=text,
                    label=normalize_label(label),
                    history=tuple(history[-context_window:]),
                )
            )
        if text:
            history.append((speaker, text))
    return examples


def _guess_dialogue_id(dialogue: list[dict[str, Any]], fallback: str) -> str:
    for utt in dialogue:
        speaker = str(utt.get("speaker", ""))
        parts = speaker.split("_")
        if len(parts) >= 3:
            return "_".join(parts[:3])
    return fallback

