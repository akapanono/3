from __future__ import annotations

import torch

from src.hdsa.ot_utils import compute_ot_stats, ot_assign_by_class, sinkhorn_assignment


def sinkhorn(cost: torch.Tensor, epsilon: float = 0.02, n_iters: int = 50) -> torch.Tensor:
    """
    Backward-compatible wrapper for old cost-based callers.

    New code should call sinkhorn_assignment(scores=...) directly. Here lower
    cost is better, so scores are the negative costs.
    """
    return sinkhorn_assignment(scores=-cost, epsilon=epsilon, n_iters=n_iters, balanced=True)


__all__ = ["sinkhorn", "sinkhorn_assignment", "compute_ot_stats", "ot_assign_by_class"]
