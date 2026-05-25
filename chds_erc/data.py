from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import torch


IEMOCAP_LABEL_ALIASES = {
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


def normalize_label(label: Any) -> str:
    text = str(label).strip().lower()
    return IEMOCAP_LABEL_ALIASES.get(text, text)


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
        examples = _load_json_split(json_path, context_window)
    elif csv_path.exists():
        examples = _load_meld_csv_split(csv_path, context_window)
    else:
        raise FileNotFoundError(
            f"Cannot find {split}_data.json or {split}_data.csv in {dataset_dir}"
        )

    if max_samples is not None:
        examples = examples[:max_samples]
    return examples


def build_label_maps(examples: Iterable[ERCExample]) -> tuple[dict[str, int], dict[int, str]]:
    labels = sorted({ex.label for ex in examples})
    label2id = {label: idx for idx, label in enumerate(labels)}
    id2label = {idx: label for label, idx in label2id.items()}
    return label2id, id2label


def examples_to_label_ids(examples: list[ERCExample], label2id: dict[str, int]) -> list[int]:
    missing = sorted({ex.label for ex in examples if ex.label not in label2id})
    if missing:
        raise ValueError(f"Labels not found in training label map: {missing}")
    return [label2id[ex.label] for ex in examples]


class ERCDataset(torch.utils.data.Dataset):
    def __init__(self, examples: list[ERCExample], label2id: dict[str, int]):
        self.examples = examples
        self.label_ids = examples_to_label_ids(examples, label2id)

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        return {"example": self.examples[idx], "label": self.label_ids[idx]}


class ERCCollator:
    def __init__(
        self,
        tokenizer,
        max_length: int = 256,
        include_target_in_context: bool = True,
    ):
        if tokenizer.mask_token is None:
            raise ValueError("The tokenizer must provide a mask token.")
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.include_target_in_context = include_target_in_context

    def __call__(self, batch: list[dict[str, Any]]) -> dict[str, Any]:
        old_truncation_side = getattr(self.tokenizer, "truncation_side", "right")
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
        self.tokenizer.truncation_side = old_truncation_side

        mask_positions = []
        mask_token_id = self.tokenizer.mask_token_id
        for row in encoded["input_ids"]:
            pos = (row == mask_token_id).nonzero(as_tuple=False).flatten()
            if pos.numel() == 0:
                raise ValueError("A prompt was truncated before the mask token.")
            mask_positions.append(pos[-1])

        return {
            "input_ids": encoded["input_ids"],
            "attention_mask": encoded["attention_mask"],
            "mask_pos": torch.stack(mask_positions).long(),
            "labels": torch.tensor([item["label"] for item in batch], dtype=torch.long),
            "examples": [item["example"] for item in batch],
            "prompts": prompts,
        }


def _load_json_split(path: Path, context_window: int) -> list[ERCExample]:
    with path.open("r", encoding="utf-8") as f:
        obj = json.load(f)

    if isinstance(obj, list):
        return _load_iemocap_dialogues(obj, path.stem, context_window)
    if isinstance(obj, dict) and "episodes" in obj:
        return _load_emory_json(obj, context_window)
    raise ValueError(f"Unsupported JSON data format: {path}")


def _load_iemocap_dialogues(
    dialogues: list[list[dict[str, Any]]],
    split_name: str,
    context_window: int,
) -> list[ERCExample]:
    examples: list[ERCExample] = []
    for d_idx, dialog in enumerate(dialogues):
        dialogue_id = _guess_dialogue_id(dialog, fallback=f"{split_name}_dialogue_{d_idx}")
        history: list[tuple[str, str]] = []
        for u_idx, utt in enumerate(dialog):
            speaker = str(utt.get("speaker", "Speaker")).strip() or "Speaker"
            text = str(utt.get("text", "")).strip()
            label = utt.get("label")
            if label is not None and text:
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


def _load_emory_json(obj: dict[str, Any], context_window: int) -> list[ERCExample]:
    examples: list[ERCExample] = []
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
                if label is not None and text:
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


def _load_meld_csv_split(path: Path, context_window: int) -> list[ERCExample]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            rows.append(row)

    rows.sort(key=lambda r: (int(r["Dialogue_ID"]), int(r["Utterance_ID"])))
    examples: list[ERCExample] = []
    cur_dialogue: str | None = None
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
        if label is not None and text:
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


def _guess_dialogue_id(dialog: list[dict[str, Any]], fallback: str) -> str:
    for utt in dialog:
        speaker = str(utt.get("speaker", ""))
        if "_" in speaker:
            parts = speaker.split("_")
            if len(parts) >= 3:
                return "_".join(parts[:3])
    return fallback

