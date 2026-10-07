from __future__ import annotations

"""Sequence-specific per-timestep baseline.

Paper §4.3.4 (adopted from Decima / Mao et al. 2019 variance-reduction
technique): each training batch consists of ``B`` episodes that share an
identical arrival sequence and event stream. The per-timestep baseline at
step ``t`` is the mean discounted return across the batch at that step:

        b_t = (1 / B) * sum_{i=1..B} G_t^{(i)}

This cancels variance attributable to the specific arrival sequence,
leaving only variance attributable to the policy's stochastic choices.

Episodes may have unequal lengths (memoryless termination). We handle that
by averaging only over episodes that reached step ``t``; b_t for a lone
surviving episode reduces to its own return, whose advantage is therefore
exactly zero (no gradient contribution), which is the documented safe
choice.
"""

from typing import List, Sequence

import numpy as np


def discounted_returns(rewards: Sequence[float], discount: float) -> np.ndarray:
    """Return G_t for t = 0..T-1 given rewards[0..T-1]."""
    n = len(rewards)
    out = np.zeros(n, dtype=np.float64)
    acc = 0.0
    for idx in range(n - 1, -1, -1):
        acc = rewards[idx] + discount * acc
        out[idx] = acc
    return out


def sequence_specific_baseline(
    episode_returns: Sequence[np.ndarray],
) -> np.ndarray:
    """Compute the per-timestep batch-mean baseline.

    Parameters
    ----------
    episode_returns:
        A list of per-episode return arrays G_t^{(i)} of varying lengths.

    Returns
    -------
    baseline: np.ndarray of shape (max_T,). ``baseline[t]`` is the mean of
    ``G_t^{(i)}`` over the episodes in the batch whose length exceeds ``t``.
    """
    if not episode_returns:
        return np.zeros(0, dtype=np.float64)
    max_length = max(len(r) for r in episode_returns)
    baseline = np.zeros(max_length, dtype=np.float64)
    counts = np.zeros(max_length, dtype=np.int64)
    for returns in episode_returns:
        t_len = len(returns)
        baseline[:t_len] += returns
        counts[:t_len] += 1
    # Safe division: where counts == 0 the slot is never used downstream.
    baseline = np.where(counts > 0, baseline / np.maximum(counts, 1), 0.0)
    return baseline


__all__ = ["discounted_returns", "sequence_specific_baseline"]
