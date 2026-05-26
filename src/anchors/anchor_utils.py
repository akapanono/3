from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.cluster import KMeans
from torch.utils.data import DataLoader


EMOTION_DESCRIPTIONS = {
    "angry": "The speaker feels angry irritated and hostile.",
    "frustrated": "The speaker feels frustrated annoyed and blocked.",
    "happy": "The speaker feels happy pleased and positive.",
    "excited": "The speaker feels excited energetic and enthusiastic.",
    "neutral": "The speaker feels neutral calm and objective.",
    "sad": "The speaker feels sad disappointed and unhappy.",
    "joyful": "The speaker feels joyful happy and pleased.",
    "powerful": "The speaker feels powerful confident and assertive.",
    "mad": "The speaker feels mad angry and irritated.",
    "scared": "The speaker feels scared fearful and anxious.",
    "peaceful": "The speaker feels peaceful calm and relaxed.",
    "fear": "The speaker feels fearful scared and worried.",
    "surprise": "The speaker feels surprised startled and amazed.",
    "disgust": "The speaker feels disgusted displeased and repelled.",
    "sadness": "The speaker feels sad sorrowful and disappointed.",
}


@torch.no_grad()
def extract_representations(model, dataloader: DataLoader, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    model.eval()
    reps = []
    labels = []
    for batch in dataloader:
        out = model.encode(
            input_ids=batch["input_ids"].to(device),
            attention_mask=batch["attention_mask"].to(device),
            mask_pos=batch["mask_pos"].to(device),
        )
        reps.append(F.normalize(out["z"], dim=-1).cpu())
        labels.append(batch["labels"].cpu())
    return torch.cat(reps, dim=0), torch.cat(labels, dim=0)


def build_domain_anchors(
    reps: torch.Tensor,
    labels: torch.Tensor,
    num_classes: int,
    num_subanchors: int,
    random_state: int = 13,
) -> tuple[torch.Tensor, torch.Tensor]:
    reps = F.normalize(reps.float(), dim=-1)
    dim = reps.size(-1)
    anchors = torch.empty(num_classes, num_subanchors, dim)
    counts = torch.zeros(num_classes, num_subanchors, dtype=torch.long)
    global_mean = F.normalize(reps.mean(dim=0), dim=-1)
    for cls in range(num_classes):
        cls_reps = reps[labels == cls]
        if cls_reps.numel() == 0:
            anchors[cls] = global_mean.repeat(num_subanchors, 1)
            continue
        k = min(num_subanchors, cls_reps.size(0))
        if k == 1:
            centers = cls_reps[:1].repeat(num_subanchors, 1)
            counts[cls, 0] = cls_reps.size(0)
        else:
            km = KMeans(n_clusters=k, random_state=random_state, n_init="auto")
            assign = km.fit_predict(cls_reps.numpy())
            centers = torch.tensor(km.cluster_centers_, dtype=torch.float32)
            for idx in range(k):
                counts[cls, idx] = int((assign == idx).sum())
            if k < num_subanchors:
                centers = torch.cat([centers, centers[:1].repeat(num_subanchors - k, 1)], dim=0)
        anchors[cls] = F.normalize(centers, dim=-1)
    return anchors, counts


def anchor_similarity_stats(anchors: torch.Tensor) -> dict[str, float]:
    anchors = F.normalize(anchors.float(), dim=-1)
    cnum, mnum, dim = anchors.shape
    flat = anchors.reshape(cnum * mnum, dim)
    sim = flat @ flat.t()
    labels = torch.arange(cnum).repeat_interleave(mnum)
    eye = torch.eye(cnum * mnum, dtype=torch.bool)
    same = (labels[:, None] == labels[None, :]) & ~eye
    diff = labels[:, None] != labels[None, :]
    return {
        "same_anchor_sim_mean": float(sim[same].mean().item()) if same.any() else 0.0,
        "same_anchor_sim_max": float(sim[same].max().item()) if same.any() else 0.0,
        "diff_anchor_sim_mean": float(sim[diff].mean().item()) if diff.any() else 0.0,
        "diff_anchor_sim_max": float(sim[diff].max().item()) if diff.any() else 0.0,
    }


def build_label_embeddings(id2label: dict[int, str], anchor_dim: int) -> torch.Tensor:
    vectors = []
    for idx in sorted(id2label):
        label = id2label[idx]
        text = EMOTION_DESCRIPTIONS.get(label, f"The speaker feels {label}.")
        vec = torch.zeros(anchor_dim)
        for token in text.lower().replace(".", "").split():
            digest = hashlib.sha256(token.encode("utf-8")).digest()
            bucket = int.from_bytes(digest[:4], "little") % anchor_dim
            sign = 1.0 if digest[4] % 2 == 0 else -1.0
            vec[bucket] += sign
        vectors.append(F.normalize(vec, dim=-1))
    return torch.stack(vectors, dim=0)


def default_dataset_dir(dataset_name: str) -> str:
    path = Path("data") / dataset_name
    return str(path)

