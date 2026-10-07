from __future__ import annotations

"""Regression tests for the per-step reward / episode-utility identity
(hardening fix for the paper §4.3.3 audit).

Background: the Phase-5/6 reward used `gamma * Δ(normalised recent_failures)`,
which telescopes to the decaying counter's end value (~0), NOT to the metric's
`gamma * failed_jobs`. So Sum_t r_t != episode utility on the risk term.

These tests pin the two regimes:
  - counter_delta (default): the historical behaviour is preserved (the gap
    equals gamma*(failed_jobs - final_counter), i.e. it is NOT zero whenever
    jobs fail).
  - failed_jobs_delta: Sum_t r_t == episode utility EXACTLY.
"""

import numpy as np

from src.config import build_run_config, load_config
from src.metrics import summarize_episode
from src.rl.env import OrchestrationEnv, evaluation_weights_from_config
from src.sim_environment import EpisodeState


class _AlwaysExecute:
    name = "always-execute"

    def __init__(self, cfg):
        self.cfg = cfg

    def choose_action(self, state: EpisodeState) -> str:
        return "Execute_Ready_Job"


def _reward_sum_and_metric(cfg, seed: int):
    """Run one always-execute episode in the env; return (Sum r_t, U_metric)."""
    from src.runner import run_episode

    metrics = run_episode(cfg=cfg, agent_factory=lambda c: _AlwaysExecute(c),
                          seed=seed, include_uncapped=False)
    env = OrchestrationEnv(cfg=cfg, num_jobs=cfg.experiment.num_jobs,
                           max_steps=cfg.experiment.max_steps, memoryless=False)
    env.reset(seed=seed)
    reward_sum = 0.0
    while True:
        _o, r, terminated, truncated, _i = env.step(0)  # 0 == Execute_Ready_Job
        reward_sum += float(r)
        if terminated or truncated:
            break
    return reward_sum, metrics.total_utility, metrics.failed_jobs


def _with_risk_mode(mode: str):
    cfg = load_config()
    raw = {**cfg.raw}
    raw["rl"] = {**raw["rl"], "reward_risk_mode": mode}
    return build_run_config(raw)


def test_counter_delta_does_not_telescope_when_failures_occur():
    cfg = _with_risk_mode("counter_delta")
    # seed 200 has failures under always-execute; gap must be > 0.
    reward_sum, u_metric, failed = _reward_sum_and_metric(cfg, 200)
    assert failed > 0, "test seed expected to have failures"
    assert abs(reward_sum - u_metric) > 1.0, (
        "counter_delta is expected to NOT match the metric when jobs fail; "
        f"gap={reward_sum - u_metric}"
    )


def test_failed_jobs_delta_telescopes_to_metric_exactly():
    cfg = _with_risk_mode("failed_jobs_delta")
    for seed in (200, 201, 202, 205, 208):
        reward_sum, u_metric, _failed = _reward_sum_and_metric(cfg, seed)
        assert abs(reward_sum - u_metric) < 1e-6, (
            f"failed_jobs_delta must telescope to the metric exactly on seed "
            f"{seed}: Sum r={reward_sum}, U={u_metric}, gap={reward_sum - u_metric}"
        )


def test_reward_risk_mode_validation_rejects_unknown():
    import pytest

    cfg = load_config()
    raw = {**cfg.raw}
    raw["rl"] = {**raw["rl"], "reward_risk_mode": "bogus_mode"}
    with pytest.raises(ValueError, match="reward_risk_mode"):
        build_run_config(raw)


def test_default_config_is_counter_delta():
    """The committed default must stay counter_delta so Phase-5/6 reproduce."""
    cfg = load_config()
    assert cfg.rl.reward_risk_mode == "counter_delta"
