import torch

from src.hdsa.ot_utils import sinkhorn_assignment


def test_sinkhorn_not_uniform_for_clear_scores():
    scores = torch.tensor(
        [
            [5.0, 1.0, 0.0],
            [0.0, 5.0, 1.0],
            [1.0, 0.0, 5.0],
            [4.0, 1.0, 0.0],
            [0.0, 4.0, 1.0],
            [1.0, 0.0, 4.0],
        ]
    )

    gamma, debug = sinkhorn_assignment(
        scores,
        epsilon=0.05,
        n_iters=50,
        balanced=True,
        return_debug=True,
    )

    assert gamma.shape == scores.shape
    assert torch.allclose(gamma.sum(dim=1), torch.ones(scores.size(0)), atol=1e-4)
    assert debug["mode"] == "balanced_sinkhorn"

    max_prob = gamma.max(dim=1).values
    assert max_prob.mean().item() > 0.5

    hard = gamma.argmax(dim=1)
    assert hard.tolist() == [0, 1, 2, 0, 1, 2]


def test_sinkhorn_uniform_for_equal_scores():
    scores = torch.zeros(10, 3)

    gamma, debug = sinkhorn_assignment(
        scores,
        epsilon=0.05,
        n_iters=50,
        balanced=True,
        return_debug=True,
    )

    expected = torch.ones_like(scores) / 3
    assert debug["mode"] == "balanced_sinkhorn"
    assert torch.allclose(gamma, expected, atol=1e-4)


def test_sinkhorn_direction_matches_scores():
    scores = torch.tensor(
        [
            [2.0, 1.0, 0.0],
            [0.0, 2.0, 1.0],
            [1.0, 0.0, 2.0],
        ]
    )

    gamma, _ = sinkhorn_assignment(
        scores,
        epsilon=0.1,
        n_iters=50,
        balanced=True,
        return_debug=True,
    )

    assert gamma.argmax(dim=1).tolist() == scores.argmax(dim=1).tolist()
