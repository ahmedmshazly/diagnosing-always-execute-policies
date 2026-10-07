from __future__ import annotations

"""Sequence-specific baseline + discounted-returns tests."""

import numpy as np
import pytest

from src.rl.baseline import discounted_returns, sequence_specific_baseline


def test_discounted_returns_with_zero_discount_is_per_step_reward():
    rewards = [1.0, 2.0, 3.0]
    out = discounted_returns(rewards, discount=0.0)
    assert out.tolist() == [1.0, 2.0, 3.0]


def test_discounted_returns_with_one_discount_is_undiscounted_suffix_sum():
    rewards = [1.0, 2.0, 3.0]
    out = discounted_returns(rewards, discount=1.0)
    assert out.tolist() == [6.0, 5.0, 3.0]


def test_discounted_returns_matches_hand_computation():
    rewards = [1.0, 0.0, -1.0, 2.0]
    delta = 0.5
    # G_3 = 2; G_2 = -1 + 0.5*2 = 0; G_1 = 0 + 0.5*0 = 0; G_0 = 1 + 0.5*0 = 1
    expected = [1.0, 0.0, 0.0, 2.0]
    out = discounted_returns(rewards, discount=delta)
    assert np.allclose(out, expected)


def test_sequence_specific_baseline_equal_length_episodes():
    # Three episodes of length 4, known returns. The per-timestep baseline
    # should be the per-column mean.
    ep_returns = [
        np.array([10.0, 8.0, 6.0, 3.0]),
        np.array([12.0, 9.0, 5.0, 1.0]),
        np.array([11.0, 7.0, 4.0, 2.0]),
    ]
    expected = np.array(
        [
            (10.0 + 12.0 + 11.0) / 3,
            (8.0 + 9.0 + 7.0) / 3,
            (6.0 + 5.0 + 4.0) / 3,
            (3.0 + 1.0 + 2.0) / 3,
        ]
    )
    out = sequence_specific_baseline(ep_returns)
    assert np.allclose(out, expected)


def test_sequence_specific_baseline_unequal_lengths_averages_over_survivors():
    # Episodes of lengths 3, 4, 2.
    ep_returns = [
        np.array([10.0, 8.0, 6.0]),
        np.array([12.0, 9.0, 5.0, 1.0]),
        np.array([2.0, 4.0]),
    ]
    expected = np.array(
        [
            (10.0 + 12.0 + 2.0) / 3,
            (8.0 + 9.0 + 4.0) / 3,
            (6.0 + 5.0) / 2,  # shortest episode has terminated
            1.0,              # only episode 1 has step 3
        ]
    )
    out = sequence_specific_baseline(ep_returns)
    assert np.allclose(out, expected)


def test_sequence_specific_baseline_single_episode_equals_its_returns():
    ep_returns = [np.array([3.0, 2.0, 1.0])]
    out = sequence_specific_baseline(ep_returns)
    assert np.allclose(out, ep_returns[0])


def test_sequence_specific_baseline_empty_batch():
    out = sequence_specific_baseline([])
    assert out.shape == (0,)
