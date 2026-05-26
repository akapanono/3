from __future__ import annotations

import torch
import torch.nn.functional as F


def soft_cross_entropy(logits: torch.Tensor, soft_targets: torch.Tensor) -> torch.Tensor:
    log_probs = F.log_softmax(logits, dim=-1)
    return -(soft_targets * log_probs).sum(dim=-1).mean()


def compactness_loss(reps: torch.Tensor, assigned_anchor: torch.Tensor) -> torch.Tensor:
    reps = F.normalize(reps, dim=-1)
    assigned_anchor = F.normalize(assigned_anchor, dim=-1)
    cos = F.cosine_similarity(reps, assigned_anchor, dim=-1)
    return ((1.0 - cos) ** 2).mean()


def sharpen_assignment(gamma: torch.Tensor, power: float = 2.0) -> torch.Tensor:
    gamma = gamma.clamp_min(1e-12)
    gamma = gamma**power
    return gamma / gamma.sum(dim=1, keepdim=True).clamp_min(1e-12)


def anchor_preserve_loss(anchors: torch.Tensor, init_anchors: torch.Tensor) -> torch.Tensor:
    anchors = F.normalize(anchors, dim=-1)
    init_anchors = F.normalize(init_anchors.to(anchors.device), dim=-1)
    sim = (anchors * init_anchors).sum(dim=-1)
    return (1.0 - sim).mean()


def anchor_inter_loss(anchors: torch.Tensor) -> torch.Tensor:
    c, m, d = anchors.shape
    if c <= 1:
        return anchors.new_tensor(0.0)
    anchors = F.normalize(anchors, dim=-1)
    flat = anchors.reshape(c * m, d)
    sim = flat @ flat.t()
    labels = torch.arange(c, device=anchors.device).repeat_interleave(m)
    diff_mask = labels[:, None] != labels[None, :]
    hardest = sim.masked_fill(~diff_mask, -1e4).max(dim=1).values
    return hardest.mean()


def anchor_domain_loss(
    anchors: torch.Tensor,
    same_upper: float = 0.90,
    center_weight: float = 0.1,
    div_weight: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    c, m, d = anchors.shape
    anchors = F.normalize(anchors, dim=-1)
    centers = F.normalize(anchors.mean(dim=1), dim=-1)
    center_sim = torch.einsum("cmd,cd->cm", anchors, centers)
    loss_center = (1.0 - center_sim).mean()
    if m <= 1:
        loss_div = anchors.new_tensor(0.0)
    else:
        flat = anchors.reshape(c * m, d)
        sim = flat @ flat.t()
        labels = torch.arange(c, device=anchors.device).repeat_interleave(m)
        same_mask = labels[:, None] == labels[None, :]
        eye = torch.eye(c * m, dtype=torch.bool, device=anchors.device)
        same_sim = sim[same_mask & ~eye]
        loss_div = F.relu(same_sim - same_upper).mean() if same_sim.numel() else anchors.new_tensor(0.0)
    loss_domain = center_weight * loss_center + div_weight * loss_div
    return loss_domain, loss_center, loss_div


def anchor_rank_loss(anchors: torch.Tensor, label_embeddings: torch.Tensor) -> torch.Tensor:
    c, m, _ = anchors.shape
    if c < 3:
        return anchors.new_tensor(0.0)
    centers = F.normalize(anchors.mean(dim=1), dim=-1)
    label_embeddings = F.normalize(label_embeddings.to(anchors.device), dim=-1)
    anchor_dist = 1.0 - centers @ centers.t()
    label_dist = 1.0 - label_embeddings @ label_embeddings.t()
    losses = []
    for i in range(c):
        for j in range(c):
            for k in range(c):
                if i == j or i == k or j == k:
                    continue
                target = (label_dist[i, j] <= label_dist[i, k]).float()
                logit = anchor_dist[i, k] - anchor_dist[i, j]
                losses.append(F.binary_cross_entropy_with_logits(logit, target))
    return torch.stack(losses).mean() if losses else anchors.new_tensor(0.0)

