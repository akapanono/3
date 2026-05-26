from __future__ import annotations

import torch
import torch.nn.functional as F


def sinkhorn(cost: torch.Tensor, epsilon: float = 0.05, n_iters: int = 50) -> torch.Tensor:
    """
    cost: [N, M]
    return: [N, M] row-normalized soft assignment.
    """
    n, m = cost.shape
    if n == 0 or m == 0:
        raise ValueError("Sinkhorn cost matrix must be non-empty.")
    device = cost.device
    a = torch.full((n,), 1.0 / n, device=device, dtype=cost.dtype)
    b = torch.full((m,), 1.0 / m, device=device, dtype=cost.dtype)
    kernel = torch.exp(-cost / epsilon).clamp_min(1e-12)
    u = torch.ones_like(a)
    v = torch.ones_like(b)
    for _ in range(n_iters):
        u = a / (kernel @ v + 1e-12)
        v = b / (kernel.t() @ u + 1e-12)
    gamma = u[:, None] * kernel * v[None, :]
    gamma = gamma / gamma.sum(dim=1, keepdim=True).clamp_min(1e-12)
    return gamma


def ot_assign_by_class(
    reps: torch.Tensor,
    labels: torch.Tensor,
    anchors: torch.Tensor,
    epsilon: float = 0.05,
    n_iters: int = 50,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    bsz, dim = reps.shape
    cnum, mnum, _ = anchors.shape
    reps = F.normalize(reps, dim=-1)
    anchors = F.normalize(anchors, dim=-1)
    soft_targets = torch.zeros(bsz, cnum * mnum, device=reps.device, dtype=reps.dtype)
    assigned_anchor = torch.zeros(bsz, dim, device=reps.device, dtype=reps.dtype)
    assignment_counts = torch.zeros(cnum, mnum, device=reps.device, dtype=reps.dtype)
    for cls in range(cnum):
        idx = torch.where(labels == cls)[0]
        if idx.numel() == 0:
            continue
        z_cls = reps[idx]
        a_cls = anchors[cls]
        cost = 1.0 - z_cls @ a_cls.t()
        gamma = sinkhorn(cost, epsilon=epsilon, n_iters=n_iters)
        start = cls * mnum
        soft_targets[idx, start : start + mnum] = gamma
        assigned_anchor[idx] = gamma @ a_cls
        assignment_counts[cls] += gamma.sum(dim=0)
    assigned_anchor = F.normalize(assigned_anchor, dim=-1)
    return soft_targets, assigned_anchor, assignment_counts

