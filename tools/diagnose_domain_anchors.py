from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.anchors.anchor_utils import anchor_similarity_stats
from src.data import ERCCollator, ERCDataset, build_label_maps, load_erc_split
from src.hdsa.ot_utils import ot_assign_by_class
from src.model import HDSAConfig, HDSAERCModel


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Diagnose HDSA domain sub-anchor quality.")
    parser.add_argument("--dataset_name", type=str, default="IEMOCAP")
    parser.add_argument("--dataset_dir", type=str, default=None)
    parser.add_argument("--bert_path", type=str, required=True)
    parser.add_argument("--domain_anchor_path", type=str, required=True)
    parser.add_argument("--raw_anchor_path", type=str, default=None)
    parser.add_argument("--pretrain_anchor_path", type=str, default=None)
    parser.add_argument("--checkpoint_path", type=str, default=None)
    parser.add_argument("--representation_cache_path", type=str, default=None)
    parser.add_argument("--save_representation_cache", action="store_true")
    parser.add_argument("--split", type=str, default="train", choices=["train", "dev", "test"])
    parser.add_argument("--num_subanchors", type=int, default=3)
    parser.add_argument("--anchor_dim", type=int, default=256)
    parser.add_argument("--output_path", type=str, default="outputs/diagnose_domain_anchors_report.json")
    parser.add_argument("--target_classes", type=str, default="angry,frustrated")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--max_length", type=int, default=256)
    parser.add_argument("--context_window", type=int, default=12)
    parser.add_argument("--local_files_only", action="store_true")
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--softmax_temperature", type=float, default=0.1)
    parser.add_argument("--ot_epsilon", type=float, default=0.02)
    parser.add_argument("--ot_iters", type=int, default=50)
    parser.add_argument("--debug_ot", action="store_true")
    parser.add_argument("--warn_ot_uniform", action="store_true")
    parser.add_argument("--device", type=str, default=None)
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_dataset_representations(
    args: argparse.Namespace,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, int], dict[int, str], dict[str, Any]]:
    cache_path = Path(args.representation_cache_path) if args.representation_cache_path else None
    if cache_path and cache_path.exists():
        obj = torch.load(cache_path, map_location="cpu")
        reps = obj["reps"].float()
        labels = obj["labels"].long()
        label2id = normalize_label2id(obj["label2id"])
        id2label = normalize_id2label(obj.get("id2label", {idx: label for label, idx in label2id.items()}))
        meta = {
            "source": "cache",
            "representation_cache_path": str(cache_path),
            "num_samples": int(labels.numel()),
            "rep_dim": int(reps.size(-1)),
        }
        return reps, labels, label2id, id2label, meta

    dataset_dir = args.dataset_dir or str(Path("data") / args.dataset_name)
    train_examples = load_erc_split(dataset_dir, "train", args.context_window)
    label2id, id2label = build_label_maps(train_examples)
    if args.split == "train":
        examples = train_examples
    else:
        examples = load_erc_split(dataset_dir, args.split, args.context_window)
        examples = [ex for ex in examples if ex.label in label2id]
    if args.max_samples is not None:
        examples = examples[: args.max_samples]

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
        domain_anchor_path=args.domain_anchor_path,
        local_files_only=args.local_files_only,
    )
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model = HDSAERCModel(config).to(device)
    checkpoint_meta: dict[str, Any] = {"checkpoint_path": args.checkpoint_path, "loaded_checkpoint": False}
    if args.checkpoint_path:
        ckpt = torch.load(args.checkpoint_path, map_location="cpu")
        state = ckpt.get("model_state_dict", ckpt)
        load_result = model.load_state_dict(state, strict=False)
        checkpoint_meta.update(
            {
                "loaded_checkpoint": True,
                "missing_keys": list(load_result.missing_keys),
                "unexpected_keys": list(load_result.unexpected_keys),
            }
        )
    model.eval()

    reps, labels = [], []
    with torch.no_grad():
        for batch in loader:
            out = model.encode(
                input_ids=batch["input_ids"].to(device),
                attention_mask=batch["attention_mask"].to(device),
                mask_pos=batch["mask_pos"].to(device),
            )
            reps.append(F.normalize(out["z"], dim=-1).cpu())
            labels.append(batch["labels"].cpu())
    reps_t = torch.cat(reps, dim=0) if reps else torch.empty(0, args.anchor_dim)
    labels_t = torch.cat(labels, dim=0) if labels else torch.empty(0, dtype=torch.long)

    if cache_path and args.save_representation_cache:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"reps": reps_t, "labels": labels_t, "label2id": label2id, "id2label": id2label}, cache_path)

    meta = {
        "source": "model_encode",
        "dataset_dir": dataset_dir,
        "split": args.split,
        "num_samples": int(labels_t.numel()),
        "rep_dim": int(reps_t.size(-1)) if reps_t.ndim == 2 else 0,
        "checkpoint": checkpoint_meta,
    }
    return reps_t, labels_t, label2id, id2label, meta


def load_anchor_file(path: str | Path, num_subanchors: int | None = None) -> tuple[torch.Tensor, dict[str, Any]]:
    path = Path(path)
    obj = torch.load(path, map_location="cpu")
    if isinstance(obj, dict):
        tensor = first_tensor_value(obj, ["anchors", "domain_anchors", "init_anchors"])
        if tensor is None:
            raise KeyError(f"Cannot find anchors/domain_anchors/init_anchors tensor in {path}")
        anchors = tensor.float()
        label2id = normalize_label2id(obj.get("label2id", {}))
        id2label = normalize_id2label(obj.get("id2label", {}))
        if not id2label and label2id:
            id2label = {idx: label for label, idx in label2id.items()}
        meta = {
            "path": str(path),
            "format": "dict",
            "keys": sorted(str(key) for key in obj.keys()),
            "label2id": label2id,
            "id2label": id2label,
            "anchor_label_order": label_order(label2id, id2label, anchors.size(0)),
            "source": obj.get("source"),
            "counts": tensor_to_jsonable(obj.get("counts")),
            "anchor_stats": tensor_to_jsonable(obj.get("anchor_stats")),
            "kmeans_anchor_stats": tensor_to_jsonable(obj.get("kmeans_anchor_stats")),
            "args": tensor_to_jsonable(obj.get("args")),
        }
    elif torch.is_tensor(obj):
        anchors = obj.float()
        meta = {
            "path": str(path),
            "format": "tensor",
            "keys": [],
            "label2id": {},
            "id2label": {},
            "anchor_label_order": [str(idx) for idx in range(anchors.size(0))],
        }
    else:
        raise TypeError(f"Unsupported anchor file type in {path}: {type(obj)!r}")

    anchors = ensure_anchor_shape(anchors, num_subanchors)
    meta["shape"] = list(anchors.shape)
    meta["num_classes"] = int(anchors.size(0))
    meta["num_subanchors"] = int(anchors.size(1))
    meta["anchor_dim"] = int(anchors.size(2))
    return anchors, meta


def first_tensor_value(obj: dict[str, Any], keys: list[str]) -> torch.Tensor | None:
    for key in keys:
        value = obj.get(key)
        if torch.is_tensor(value):
            return value
    return None


def ensure_anchor_shape(anchors: torch.Tensor, num_subanchors: int | None) -> torch.Tensor:
    if anchors.ndim == 3:
        return anchors
    if anchors.ndim == 2 and num_subanchors and anchors.size(0) % num_subanchors == 0:
        return anchors.reshape(anchors.size(0) // num_subanchors, num_subanchors, anchors.size(1))
    raise ValueError(f"Anchor tensor must have shape [C, M, D], got {tuple(anchors.shape)}")


def normalize_label2id(value: Any) -> dict[str, int]:
    if not isinstance(value, dict):
        return {}
    return {str(label): int(idx) for label, idx in value.items()}


def normalize_id2label(value: Any) -> dict[int, str]:
    if not isinstance(value, dict):
        return {}
    return {int(idx): str(label) for idx, label in value.items()}


def label_order(label2id_: dict[str, int], id2label_: dict[int, str], num_classes: int) -> list[str]:
    if id2label_:
        return [id2label_.get(idx, str(idx)) for idx in range(num_classes)]
    if label2id_:
        reverse = {idx: label for label, idx in label2id_.items()}
        return [reverse.get(idx, str(idx)) for idx in range(num_classes)]
    return [str(idx) for idx in range(num_classes)]


def compute_anchor_cosine(anchors: torch.Tensor, id2label: dict[int, str]) -> dict[str, list[list[float]]]:
    anchors = F.normalize(anchors.float(), dim=-1)
    result = {}
    for cls in range(anchors.size(0)):
        name = id2label.get(cls, str(cls))
        sim = anchors[cls] @ anchors[cls].t()
        result[name] = round_nested(sim.tolist())
    return result


def compute_anchor_norm(anchors: torch.Tensor, id2label: dict[int, str]) -> dict[str, list[float]]:
    norms = anchors.float().norm(dim=-1)
    return {id2label.get(cls, str(cls)): round_list(norms[cls].tolist()) for cls in range(anchors.size(0))}


def compute_score_stats(
    reps: torch.Tensor,
    labels: torch.Tensor,
    anchors: torch.Tensor,
    id2label: dict[int, str],
) -> dict[str, dict[str, Any]]:
    reps, anchors = validate_rep_anchor_dims(reps, anchors)
    reps = F.normalize(reps.float(), dim=-1)
    anchors = F.normalize(anchors.float(), dim=-1)
    result = {}
    for cls in range(anchors.size(0)):
        idx = torch.where(labels == cls)[0]
        name = id2label.get(cls, str(cls))
        if idx.numel() == 0:
            continue
        scores = reps[idx] @ anchors[cls].t()
        score_std = scores.std(dim=1, unbiased=False)
        score_gap = scores.max(dim=1).values - scores.min(dim=1).values
        result[name] = {
            "num_samples": int(idx.numel()),
            "score_std_mean": qfloat(score_std.mean()),
            "score_std_p50": quantile(score_std, 0.50),
            "score_std_p90": quantile(score_std, 0.90),
            "score_gap_mean": qfloat(score_gap.mean()),
            "score_gap_p50": quantile(score_gap, 0.50),
            "score_gap_p90": quantile(score_gap, 0.90),
            "score_mean_per_anchor": round_list(scores.mean(dim=0).tolist()),
            "score_min_per_anchor": round_list(scores.min(dim=0).values.tolist()),
            "score_max_per_anchor": round_list(scores.max(dim=0).values.tolist()),
        }
    return result


def compute_simple_assignment_stats(
    reps: torch.Tensor,
    labels: torch.Tensor,
    anchors: torch.Tensor,
    id2label: dict[int, str],
    temperature: float,
) -> dict[str, dict[str, Any]]:
    reps, anchors = validate_rep_anchor_dims(reps, anchors)
    reps = F.normalize(reps.float(), dim=-1)
    anchors = F.normalize(anchors.float(), dim=-1)
    result = {}
    for cls in range(anchors.size(0)):
        idx = torch.where(labels == cls)[0]
        name = id2label.get(cls, str(cls))
        if idx.numel() == 0:
            continue
        scores = reps[idx] @ anchors[cls].t()
        prob = torch.softmax(scores / temperature, dim=-1)
        max_prob = prob.max(dim=1).values
        hard = prob.argmax(dim=1)
        counts = torch.bincount(hard, minlength=anchors.size(1)).float()
        result[name] = {
            "max_prob_mean": qfloat(max_prob.mean()),
            "max_prob_p50": quantile(max_prob, 0.50),
            "max_prob_p70": quantile(max_prob, 0.70),
            "max_prob_p90": quantile(max_prob, 0.90),
            "max_prob_max": qfloat(max_prob.max()),
            "hard_counts": round_list(counts.tolist()),
        }
    return result


def compute_ot_diagnostics(
    reps: torch.Tensor,
    labels: torch.Tensor,
    anchors: torch.Tensor,
    id2label: dict[int, str],
    epsilon: float,
    n_iters: int,
    debug: bool = False,
    warn_uniform: bool = False,
) -> dict[str, Any]:
    reps, anchors = validate_rep_anchor_dims(reps, anchors)
    soft_targets, _, assignment_counts, global_stats = ot_assign_by_class(
        reps=reps.float(),
        labels=labels.long(),
        anchors=anchors.float(),
        epsilon=epsilon,
        n_iters=n_iters,
        debug=debug,
        id2label=id2label,
        warn_uniform=warn_uniform or debug,
    )
    cnum, mnum = anchors.size(0), anchors.size(1)
    target = soft_targets.reshape(-1, cnum, mnum)
    per_class, hard_counts = {}, {}
    for cls in range(cnum):
        idx = torch.where(labels == cls)[0]
        name = id2label.get(cls, str(cls))
        if idx.numel() == 0:
            continue
        gamma = target[idx, cls]
        max_prob, hard_sub = gamma.max(dim=1)
        counts = torch.bincount(hard_sub, minlength=mnum).float()
        per_class[name] = {
            "mean": qfloat(max_prob.mean()),
            "min": qfloat(max_prob.min()),
            "p50": quantile(max_prob, 0.50),
            "p70": quantile(max_prob, 0.70),
            "p90": quantile(max_prob, 0.90),
            "max": qfloat(max_prob.max()),
        }
        hard_counts[name] = round_list(counts.tolist())
    named_assignment_counts = {
        id2label.get(cls, str(cls)): round_list(assignment_counts[cls].tolist()) for cls in range(cnum)
    }
    return {
        "per_class_ot_max_prob": per_class,
        "hard_assignment_counts": hard_counts,
        "ot_assignment_counts": named_assignment_counts,
        "ot_global_stats": {key: qfloat(value) for key, value in global_stats.items()},
    }


def compare_stages(
    stage_paths: dict[str, str | None],
    current_label2id: dict[str, int],
    current_id2label: dict[int, str],
    num_subanchors: int,
    target_classes: list[str],
) -> dict[str, Any]:
    result = {}
    for stage, path in stage_paths.items():
        if not path:
            continue
        anchors, meta = load_anchor_file(path, num_subanchors=num_subanchors)
        anchors = F.normalize(anchors.float(), dim=-1)
        stage_label2id = meta.get("label2id") or current_label2id
        stage_id2label = normalize_id2label(meta.get("id2label")) or current_id2label
        metrics: dict[str, Any] = {
            "path": str(path),
            "shape": list(anchors.shape),
            "anchor_label_order": label_order(stage_label2id, stage_id2label, anchors.size(0)),
            "anchor_stats": anchor_similarity_stats(anchors),
        }
        for name in target_classes:
            cls = stage_label2id.get(name, current_label2id.get(name))
            if cls is None or cls >= anchors.size(0):
                continue
            same = offdiag_values(anchors[cls] @ anchors[cls].t())
            metrics[f"{name}_same_sim_mean"] = qfloat(same.mean()) if same.numel() else 0.0
            metrics[f"{name}_same_sim_max"] = qfloat(same.max()) if same.numel() else 0.0
        result[stage] = tensor_to_jsonable(metrics)
    return result


def validate_rep_anchor_dims(reps: torch.Tensor, anchors: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    if reps.ndim != 2:
        raise ValueError(f"Representations must have shape [N, D], got {tuple(reps.shape)}")
    if anchors.ndim != 3:
        raise ValueError(f"Anchors must have shape [C, M, D], got {tuple(anchors.shape)}")
    if reps.size(-1) != anchors.size(-1):
        raise ValueError(
            f"Representation dim {reps.size(-1)} != anchor dim {anchors.size(-1)}. "
            "Use the matching --anchor_dim and, if available, --checkpoint_path."
        )
    return reps, anchors


def offdiag_values(matrix: torch.Tensor) -> torch.Tensor:
    if matrix.size(0) <= 1:
        return matrix.new_empty(0)
    eye = torch.eye(matrix.size(0), dtype=torch.bool, device=matrix.device)
    return matrix[~eye]


def quantile(values: torch.Tensor, q: float) -> float:
    if values.numel() == 0:
        return 0.0
    return qfloat(torch.quantile(values.float(), q))


def qfloat(value: Any) -> float:
    if torch.is_tensor(value):
        value = value.detach().cpu().item()
    return round(float(value), 6)


def round_list(values: list[Any]) -> list[float]:
    return [round(float(value), 6) for value in values]


def round_nested(values: list[list[Any]]) -> list[list[float]]:
    return [round_list(row) for row in values]


def tensor_to_jsonable(value: Any) -> Any:
    if value is None:
        return None
    if torch.is_tensor(value):
        return value.detach().cpu().tolist()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {str(key): tensor_to_jsonable(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [tensor_to_jsonable(item) for item in value]
    if isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    target_classes = [name.strip() for name in args.target_classes.split(",") if name.strip()]

    reps, labels, label2id, id2label, rep_meta = load_dataset_representations(args)
    anchors, anchor_meta = load_anchor_file(args.domain_anchor_path, num_subanchors=args.num_subanchors)

    expected_shape = (len(label2id), args.num_subanchors, args.anchor_dim)
    warnings = []
    if tuple(anchors.shape) != expected_shape:
        warnings.append(f"domain anchor shape {tuple(anchors.shape)} != expected {expected_shape}")
    dataset_label_order = [id2label[idx] for idx in sorted(id2label)]
    anchor_order = anchor_meta.get("anchor_label_order") or [str(idx) for idx in range(anchors.size(0))]
    anchor_order_matches = anchor_order == dataset_label_order
    if not anchor_order_matches:
        warnings.append("anchor label order does not match dataset label2id order")

    report = {
        "dataset_name": args.dataset_name,
        "split": args.split,
        "num_samples": int(labels.numel()),
        "target_classes": target_classes,
        "label2id": label2id,
        "id2label": {str(idx): label for idx, label in id2label.items()},
        "dataset_label_order": dataset_label_order,
        "anchor_label_order": anchor_order,
        "anchor_order_matches_label2id": anchor_order_matches,
        "anchor_meta": anchor_meta,
        "representation_meta": rep_meta,
        "warnings": warnings,
        "per_class_anchor_cosine": compute_anchor_cosine(anchors, id2label),
        "per_class_anchor_norm": compute_anchor_norm(anchors, id2label),
        "per_class_score_stats": compute_score_stats(reps, labels, anchors, id2label),
        "softmax_assignment": compute_simple_assignment_stats(
            reps,
            labels,
            anchors,
            id2label,
            temperature=args.softmax_temperature,
        ),
    }
    report.update(
        compute_ot_diagnostics(
            reps,
            labels,
            anchors,
            id2label,
            epsilon=args.ot_epsilon,
            n_iters=args.ot_iters,
            debug=args.debug_ot,
            warn_uniform=args.warn_ot_uniform,
        )
    )
    report["stage_compare"] = compare_stages(
        {
            "kmeans": args.raw_anchor_path,
            "pretrained": args.pretrain_anchor_path,
            "loaded": args.domain_anchor_path,
        },
        current_label2id=label2id,
        current_id2label=id2label,
        num_subanchors=args.num_subanchors,
        target_classes=target_classes,
    )

    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(tensor_to_jsonable(report), f, ensure_ascii=False, indent=2)
    print(json.dumps(tensor_to_jsonable(report), ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
