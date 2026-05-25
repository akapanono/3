from __future__ import annotations

import torch
import torch.nn.functional as F


def anchor_pull_loss(reps: torch.Tensor, labels: torch.Tensor, anchors: torch.Tensor) -> torch.Tensor:
    reps = F.normalize(reps, dim=-1)
    anchors = F.normalize(anchors, dim=-1)
    cur_anchors = anchors[labels]
    sim = torch.einsum("bd,bkd->bk", reps, cur_anchors)
    best_k = sim.argmax(dim=1)
    batch_idx = torch.arange(reps.size(0), device=reps.device)
    pos_anchor = cur_anchors[batch_idx, best_k]
    loss = 1.0 - F.cosine_similarity(reps, pos_anchor, dim=-1)
    return loss.mean()


def hyperspherical_inter_anchor_loss(anchors: torch.Tensor) -> torch.Tensor:
    c, k, d = anchors.shape
    if c <= 1:
        return anchors.new_tensor(0.0)
    anchors = F.normalize(anchors, dim=-1)
    flat = anchors.reshape(c * k, d)
    sim = flat @ flat.t()
    labels = torch.arange(c, device=anchors.device).repeat_interleave(k)
    diff_mask = labels[:, None] != labels[None, :]
    hardest_diff_sim = sim.masked_fill(~diff_mask, -1e4).max(dim=1).values
    return hardest_diff_sim.mean()


def intra_anchor_diversity_loss(anchors: torch.Tensor, same_upper: float = 0.85) -> torch.Tensor:
    c, k, d = anchors.shape
    if k <= 1:
        return anchors.new_tensor(0.0)
    anchors = F.normalize(anchors, dim=-1)
    flat = anchors.reshape(c * k, d)
    sim = flat @ flat.t()
    labels = torch.arange(c, device=anchors.device).repeat_interleave(k)
    same_mask = labels[:, None] == labels[None, :]
    eye = torch.eye(c * k, dtype=torch.bool, device=anchors.device)
    same_sim = sim[same_mask & ~eye]
    if same_sim.numel() == 0:
        return anchors.new_tensor(0.0)
    return F.relu(same_sim - same_upper).mean()


def supervised_contrastive_loss(
    reps: torch.Tensor,
    labels: torch.Tensor,
    anchors: torch.Tensor | None = None,
    temperature: float = 0.1,
    eps: float = 1e-8,
) -> torch.Tensor:
    reps = F.normalize(reps, dim=-1)
    features = reps
    feature_labels = labels
    query_count = reps.size(0)

    if anchors is not None:
        c, k, d = anchors.shape
        flat_anchors = F.normalize(anchors, dim=-1).reshape(c * k, d)
        anchor_labels = torch.arange(c, device=anchors.device).repeat_interleave(k)
        features = torch.cat([features, flat_anchors], dim=0)
        feature_labels = torch.cat([feature_labels, anchor_labels], dim=0)

    logits = reps @ features.t() / temperature
    logits = logits - logits.max(dim=1, keepdim=True).values.detach()

    positive_mask = labels[:, None] == feature_labels[None, :]
    self_mask = torch.zeros_like(positive_mask, dtype=torch.bool)
    self_mask[:, :query_count] = torch.eye(query_count, dtype=torch.bool, device=reps.device)
    positive_mask = positive_mask & ~self_mask

    exp_logits = torch.exp(logits).masked_fill(self_mask, 0.0)
    log_prob = logits - torch.log(exp_logits.sum(dim=1, keepdim=True).clamp_min(eps))

    positive_count = positive_mask.sum(dim=1)
    valid = positive_count > 0
    if valid.sum() == 0:
        return reps.new_tensor(0.0)

    mean_log_prob_pos = (positive_mask.float() * log_prob).sum(dim=1) / positive_count.clamp_min(1)
    return -mean_log_prob_pos[valid].mean()
