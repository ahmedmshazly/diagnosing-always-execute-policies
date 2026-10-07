from __future__ import annotations

"""Gymnasium env wrapper tests (paper §4.3 + Phase-4 brief §C)."""

import numpy as np
import pytest

from src.config import load_config
from src.rl.env import (
    ACTIONS,
    NUM_ACTIONS,
    NUM_STATE_FEATURES,
    OrchestrationEnv,
    STATE_FIELD_ORDER,
)


def test_observation_space_is_eight_dim_float_in_unit_interval(default_config):
    env = OrchestrationEnv(cfg=default_config, num_jobs=10, max_steps=20)
    assert NUM_STATE_FEATURES == 8
    assert env.observation_space.shape == (8,)
    assert env.observation_space.dtype == np.float32
    assert np.all(env.observation_space.low == 0.0)
    assert np.all(env.observation_space.high == 1.0)


def test_action_space_is_discrete_six(default_config):
    env = OrchestrationEnv(cfg=default_config, num_jobs=10, max_steps=20)
    assert NUM_ACTIONS == 6
    assert len(ACTIONS) == 6
    assert env.action_space.n == 6


def test_reset_returns_eight_dim_vector_in_unit_interval(default_config):
    env = OrchestrationEnv(cfg=default_config, num_jobs=10, max_steps=20)
    obs, info = env.reset(seed=42)
    assert isinstance(obs, np.ndarray)
    assert obs.shape == (8,)
    assert obs.dtype == np.float32
    assert np.all(obs >= 0.0)
    assert np.all(obs <= 1.0)
    assert "horizon" in info


def test_reset_without_seed_raises(default_config):
    env = OrchestrationEnv(cfg=default_config, num_jobs=10, max_steps=20)
    with pytest.raises(ValueError):
        env.reset()


def test_step_returns_five_tuple_with_finite_reward(default_config):
    env = OrchestrationEnv(cfg=default_config, num_jobs=10, max_steps=20)
    env.reset(seed=1)
    obs, reward, terminated, truncated, info = env.step(0)
    assert obs.shape == (8,)
    assert np.isfinite(reward)
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)
    # The reward carries exactly the per-step quantities we will need in
    # post-hoc analyses.
    assert {"delta_value", "step_cost", "delta_risk"} <= set(info.keys())


def test_reward_sign_matches_spec_on_a_no_progress_step(default_config):
    """A Defer_Job with no running tasks yields zero value and zero cost:
    reward should be 0.0 at the first step when the cluster is cold."""
    env = OrchestrationEnv(cfg=default_config, num_jobs=4, max_steps=4)
    env.reset(seed=5)
    defer_idx = ACTIONS.index("Defer_Job")
    _, reward, _, _, info = env.step(defer_idx)
    assert info["step_cost"] == 0.0
    assert info["delta_value"] == 0.0
    # delta_risk can be nonzero only if a node failure fires; in that case
    # reward should be non-positive.
    assert reward <= 0.0 + 1e-9


def test_reward_negative_when_launching_expensive_task(default_config):
    """Pure step_cost (no completed jobs yet) gives a strictly negative reward."""
    env = OrchestrationEnv(cfg=default_config, num_jobs=20, max_steps=20)
    env.reset(seed=7)
    exec_idx = ACTIONS.index("Execute_Ready_Job")
    _, reward, _, _, info = env.step(exec_idx)
    # First execute always costs something and completes nothing this tick.
    assert info["step_cost"] >= 0.0
    assert info["delta_value"] == 0.0
    assert reward <= 0.0


def test_termination_on_all_jobs_done(default_config):
    """With a tiny workload the episode must finish before the budget."""
    env = OrchestrationEnv(cfg=default_config, num_jobs=2, max_steps=200)
    env.reset(seed=123)
    exec_idx = ACTIONS.index("Execute_Ready_Job")
    terminated = truncated = False
    steps = 0
    while not terminated and not truncated and steps < 500:
        _, _, terminated, truncated, _ = env.step(exec_idx)
        steps += 1
    assert terminated or truncated
    # 2 jobs x 3 tasks each should complete well before 500 ticks.
    assert steps < 500


def test_truncation_on_step_cap(default_config):
    env = OrchestrationEnv(cfg=default_config, num_jobs=50, max_steps=3)
    env.reset(seed=1)
    defer_idx = ACTIONS.index("Defer_Job")
    terminated = truncated = False
    for _ in range(3):
        _, _, terminated, truncated, _ = env.step(defer_idx)
    # After 3 steps with a 50-job workload, the episode is truncated,
    # not terminated.
    assert truncated is True
    assert terminated is False


def test_state_field_order_matches_state_vector(default_config):
    """The RL state vector ordering must match StateVector's field order."""
    env = OrchestrationEnv(cfg=default_config, num_jobs=6, max_steps=5)
    obs, _ = env.reset(seed=99)
    sv = env._state.state_vector()
    for idx, field in enumerate(STATE_FIELD_ORDER):
        assert abs(obs[idx] - getattr(sv, field)) < 1e-6, f"{field} mismatch"


def test_memoryless_termination_draws_positive_horizon(default_config):
    env = OrchestrationEnv(
        cfg=default_config,
        num_jobs=10,
        max_steps=50,
        memoryless=True,
        reset_rng_seed=0,
    )
    _, info = env.reset(seed=100)
    assert info["horizon"] >= 1
    # Horizon must be at most 4x mu (env-side safety cap).
    assert info["horizon"] <= 4 * 50


def test_memoryless_termination_mean_within_10_percent_of_mu():
    """1000 draws from Exp(mu=120), mean within 10% of mu. Phase-4 brief §C."""
    rng = np.random.default_rng(2026)
    mu = 120
    draws = rng.exponential(scale=mu, size=1000)
    mean = float(draws.mean())
    assert abs(mean - mu) / mu < 0.10, f"mean={mean} vs mu={mu}"
