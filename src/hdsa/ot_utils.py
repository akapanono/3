from __future__ import annotations

import warnings
from typing import Any

import torch
import torch.nn.functional as F


def sinkhorn_assignment(
    scores: torch.Tensor,
    epsilon: float = 0.02,
    n_iters: int = 50,
    balanced: bool = True,
    return_debug: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, dict[str, Any]]:
    """
    Convert similarity scores [N, M] into per-sample assignment probabilities.

    Larger scores mean a sample is more similar to an anchor. The returned gamma
    always has shape [N, M] and each row sums to 1.
    """
    if scores.dim() != 2:
        raise ValueError(f"scores must be [N, M], got {tuple(scores.shape)}")
    n, m = scores.shape
    if n == 0 or m == 0:
        raise ValueError("Sinkhorn scores matrix must be non-empty.")
    eps = max(float(epsilon), 1e-8)

    # Stabilize before exponentiation. Without this, cosine scores below 1.0
    # with tiny epsilon can underflow to the clamp floor and become uniform.
    stable_scores = scores - scores.max(dim=1, keepdim=True).values
    kernel = torch.exp(stable_scores / eps).clamp_min(1e-12)

    if not balanced:
        gamma = kernel / kernel.sum(dim=1, keepdim=True).clamp_min(1e-12)
        debug = _assignment_debug(gamma, epsilon=epsilon, n_iters=n_iters, mode="row_softmax_like")
        return (gamma, debug) if return_debug else gamma

    dtype = scores.dtype
    device = scores.device
    row_marginal = torch.full((n,), 1.0 / n, dtype=dtype, device=device)
    col_marginal = torch.full((m,), 1.0 / m, dtype=dtype, device=device)
    u = torch.ones_like(row_marginal)
    v = torch.ones_like(col_marginal)
    for _ in range(n_iters):
        u = row_marginal / (kernel @ v + 1e-12)
        v = col_marginal / (kernel.t() @ u + 1e-12)

    plan = u[:, None] * kernel * v[None, :]
    gamma = plan / plan.sum(dim=1, keepdim=True).clamp_min(1e-12)
    debug = _assignment_debug(
        gamma,
        epsilon=epsilon,
        n_iters=n_iters,
        mode="balanced_sinkhorn",
        plan=plan,
    )
    return (gamma, debug) if return_debug else gamma


def compute_ot_stats(gamma: torch.Tensor) -> dict[str, float]:
    entropy = -(gamma * gamma.clamp_min(1e-12).log()).sum(dim=1)
    max_prob = gamma.max(dim=1).values
    return {
        "entropy_sum": float(entropy.sum().detach().cpu()),
        "entropy_max": float(entropy.max().detach().cpu()) if entropy.numel() else 0.0,
        "max_prob_sum": float(max_prob.sum().detach().cpu()),
        "max_prob_min": float(max_prob.min().detach().cpu()) if max_prob.numel() else 0.0,
        "max_prob_max": float(max_prob.max().detach().cpu()) if max_prob.numel() else 0.0,
        "num_samples": int(gamma.size(0)),
    }


def ot_assign_by_class(
    reps: torch.Tensor,
    labels: torch.Tensor,
    anchors: torch.Tensor,
    epsilon: float = 0.02,
    n_iters: int = 50,
    debug: bool = False,
    id2label: dict[int, str] | None = None,
    warn_uniform: bool = True,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[str, float]]:
    bsz, dim = reps.shape
    cnum, mnum, _ = anchors.shape
    reps = F.normalize(reps, dim=-1)
    anchors = F.normalize(anchors, dim=-1)
    soft_targets = torch.zeros(bsz, cnum * mnum, device=reps.device, dtype=reps.dtype)
    assigned_anchor = torch.zeros(bsz, dim, device=reps.device, dtype=reps.dtype)
    assignment_counts = torch.zeros(cnum, mnum, device=reps.device, dtype=reps.dtype)
    stat_acc = {
        "entropy_sum": 0.0,
        "entropy_max": 0.0,
        "max_prob_sum": 0.0,
        "max_prob_min": float("inf"),
        "max_prob_max": 0.0,
        "num_samples": 0,
    }
    for cls in range(cnum):
        idx = torch.where(labels == cls)[0]
        if idx.numel() == 0:
            continue
        z_cls = reps[idx]
        a_cls = anchors[cls]
        scores = z_cls @ a_cls.t()
        gamma, gamma_debug = sinkhorn_assignment(
            scores=scores,
            epsilon=epsilon,
            n_iters=n_iters,
            balanced=True,
            return_debug=True,
        )
        class_name = id2label.get(cls, str(cls)) if id2label else str(cls)
        if debug:
            _print_ot_debug(class_name, scores, gamma, gamma_debug)
        if warn_uniform:
            _warn_if_scores_differ_but_gamma_uniform(class_name, scores, gamma)

        start = cls * mnum
        soft_targets[idx, start : start + mnum] = gamma
        assigned_anchor[idx] = gamma @ a_cls
        assignment_counts[cls] += gamma.sum(dim=0)
        cls_stats = compute_ot_stats(gamma)
        stat_acc["entropy_sum"] += cls_stats["entropy_sum"]
        stat_acc["entropy_max"] = max(stat_acc["entropy_max"], cls_stats["entropy_max"])
        stat_acc["max_prob_sum"] += cls_stats["max_prob_sum"]
        stat_acc["max_prob_min"] = min(stat_acc["max_prob_min"], cls_stats["max_prob_min"])
        stat_acc["max_prob_max"] = max(stat_acc["max_prob_max"], cls_stats["max_prob_max"])
        stat_acc["num_samples"] += cls_stats["num_samples"]
    assigned_anchor = F.normalize(assigned_anchor, dim=-1)
    n = max(1, stat_acc["num_samples"])
    ot_stats = {
        "ot_entropy_mean": stat_acc["entropy_sum"] / n,
        "ot_entropy_max": stat_acc["entropy_max"],
        "ot_max_prob_mean": stat_acc["max_prob_sum"] / n,
        "ot_max_prob_min": 0.0 if stat_acc["max_prob_min"] == float("inf") else stat_acc["max_prob_min"],
        "ot_max_prob_max": stat_acc["max_prob_max"],
    }
    return soft_targets, assigned_anchor, assignment_counts, ot_stats


def _assignment_debug(
    gamma: torch.Tensor,
    epsilon: float,
    n_iters: int,
    mode: str,
    plan: torch.Tensor | None = None,
) -> dict[str, Any]:
    entropy = -(gamma * gamma.clamp_min(1e-12).log()).sum(dim=1)
    debug: dict[str, Any] = {
        "mode": mode,
        "epsilon": float(epsilon),
        "n_iters": int(n_iters),
        "gamma_row_sum_mean": float(gamma.sum(dim=1).mean().detach().cpu()),
        "gamma_max_prob_mean": float(gamma.max(dim=1).values.mean().detach().cpu()),
        "gamma_entropy_mean": float(entropy.mean().detach().cpu()),
        "has_nan": bool(torch.isnan(gamma).any().item()),
    }
    if plan is not None:
        debug["plan_row_sum_mean"] = float(plan.sum(dim=1).mean().detach().cpu())
        debug["plan_col_sum"] = plan.sum(dim=0).detach().cpu().tolist()
    return debug


def _print_ot_debug(
    class_name: str,
    scores: torch.Tensor,
    gamma: torch.Tensor,
    debug: dict[str, Any],
) -> None:
    score_gap = scores.max(dim=1).values - scores.min(dim=1).values
    print(f"[OT DEBUG] class={class_name}", flush=True)
    print(f"scores[:5]= {scores[:5].detach().cpu()}", flush=True)
    print(f"score_std_mean= {scores.std(dim=1, unbiased=False).mean().item():.6f}", flush=True)
    print(f"score_gap_mean= {score_gap.mean().item():.6f}", flush=True)
    print(f"epsilon= {debug['epsilon']}", flush=True)
    print(f"gamma[:5]= {gamma[:5].detach().cpu()}", flush=True)
    print(f"gamma_row_sum_mean= {gamma.sum(dim=1).mean().item():.6f}", flush=True)
    print(f"gamma_max_prob_mean= {gamma.max(dim=1).values.mean().item():.6f}", flush=True)
    print(f"sinkhorn_has_nan= {debug['has_nan']}", flush=True)


def _warn_if_scores_differ_but_gamma_uniform(
    class_name: str,
    scores: torch.Tensor,
    gamma: torch.Tensor,
) -> None:
    m = scores.size(1)
    score_gap = scores.max(dim=1).values - scores.min(dim=1).values
    gamma_max = gamma.max(dim=1).values
    if score_gap.mean().item() > 1e-3 and gamma_max.mean().item() < (1.0 / m + 1e-4):
        warnings.warn(
            f"[OT WARNING] class={class_name}: raw scores have differences "
            f"(gap={score_gap.mean().item():.6f}) but OT gamma is nearly uniform "
            f"(max_prob={gamma_max.mean().item():.6f}). Check Sinkhorn implementation.",
            RuntimeWarning,
            stacklevel=2,
        )
