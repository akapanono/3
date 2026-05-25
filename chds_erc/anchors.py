from __future__ import annotations

from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.cluster import KMeans
from torch.utils.data import DataLoader


@torch.no_grad()
def extract_representations(model, loader: DataLoader, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    model.eval()
    reps = []
    labels = []
    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        mask_pos = batch["mask_pos"].to(device)
        out = model(input_ids=input_ids, attention_mask=attention_mask, mask_pos=mask_pos)
        reps.append(out["z"].cpu())
        labels.append(batch["labels"].cpu())
    return torch.cat(reps, dim=0), torch.cat(labels, dim=0)


def build_cluster_anchors(
    reps: torch.Tensor,
    labels: torch.Tensor,
    num_classes: int,
    num_subanchors: int,
    random_state: int = 13,
) -> torch.Tensor:
    reps = F.normalize(reps.float(), dim=-1)
    dim = reps.size(-1)
    anchors = torch.empty(num_classes, num_subanchors, dim, dtype=torch.float32)
    global_mean = reps.mean(dim=0)

    for cls in range(num_classes):
        cls_reps = reps[labels == cls]
        if cls_reps.numel() == 0:
            centers = global_mean.repeat(num_subanchors, 1)
        elif cls_reps.size(0) < num_subanchors:
            repeat_idx = torch.arange(num_subanchors) % cls_reps.size(0)
            centers = cls_reps[repeat_idx].clone()
        else:
            km = KMeans(n_clusters=num_subanchors, n_init="auto", random_state=random_state)
            centers_np = km.fit(cls_reps.numpy()).cluster_centers_
            centers = torch.from_numpy(centers_np).float()
        anchors[cls] = F.normalize(centers, dim=-1)
    return anchors


@torch.no_grad()
def anchor_assignment_counts(reps: torch.Tensor, labels: torch.Tensor, anchors: torch.Tensor) -> dict[int, list[int]]:
    reps = F.normalize(reps.float(), dim=-1)
    anchors = F.normalize(anchors.float(), dim=-1)
    counts: dict[int, list[int]] = defaultdict(lambda: [0] * anchors.size(1))
    for cls in range(anchors.size(0)):
        cls_reps = reps[labels == cls]
        if cls_reps.numel() == 0:
            counts[cls] = [0] * anchors.size(1)
            continue
        sim = torch.einsum("bd,kd->bk", cls_reps, anchors[cls])
        best = sim.argmax(dim=1).cpu().numpy()
        vals = np.bincount(best, minlength=anchors.size(1)).tolist()
        counts[cls] = vals
    return dict(counts)


@torch.no_grad()
def anchor_similarity_stats(anchors: torch.Tensor) -> dict[str, float]:
    anchors = F.normalize(anchors.float(), dim=-1)
    c, k, d = anchors.shape
    flat = anchors.reshape(c * k, d)
    sim = flat @ flat.t()
    labels = torch.arange(c).repeat_interleave(k)
    eye = torch.eye(c * k, dtype=torch.bool)
    same = (labels[:, None] == labels[None, :]) & ~eye
    diff = labels[:, None] != labels[None, :]
    return {
        "same_anchor_sim_mean": float(sim[same].mean().item()) if same.any() else 0.0,
        "same_anchor_sim_max": float(sim[same].max().item()) if same.any() else 0.0,
        "diff_anchor_sim_mean": float(sim[diff].mean().item()) if diff.any() else 0.0,
        "diff_anchor_sim_max": float(sim[diff].max().item()) if diff.any() else 0.0,
    }

