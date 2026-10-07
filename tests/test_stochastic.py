from __future__ import annotations

"""Tests for the stochastic processes (node_failure, data_spike, spot_price).

Each process has a named mode; these tests pin behaviour at each mode
against known-good semantics. Statistical tests use a single RNG with a
large step count so the observed rate converges to the configured
probability within a generous tolerance.
"""

import numpy as np
import pytest

from src.config import build_run_config, load_config
from src.sim_environment import (
    EpisodeState,
    TASK_STATE_FAILED,
    TASK_STATE_READY,
    TASK_STATE_RUNNING,
    WorkloadGenerator,
    _apply_data_spike,
    _apply_node_failure,
    _apply_spot_price_walk,
    make_episode_rngs,
)


def _fresh_state(cfg, seed=42, num_jobs=20):
    workload_rng, _ = make_episode_rngs(seed)
    generator = WorkloadGenerator(rng=workload_rng, cfg=cfg, num_jobs=num_jobs, seed_label=seed)
    return generator.generate_episode()


def _deep_copy(value):
    if isinstance(value, dict):
        return {k: _deep_copy(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_deep_copy(v) for v in value]
    return value


def _populate_running_tasks(state: EpisodeState, count: int) -> None:
    from src.sim_environment import launch_task
    ready = state.ready_tasks()
    launched = 0
    for task in ready:
        if launch_task(state, task):
            launched += 1
            if launched >= count:
                break
    assert launched == count, f"expected {count} running tasks, got {launched}"


# ---------------------------------------------------------------------------
# Spot price
# ---------------------------------------------------------------------------
def test_spot_price_walk_stays_within_bounds_over_many_steps():
    cfg = load_config()
    sp = cfg.simulator.stochastic_processes.spot_price
    state = _fresh_state(cfg)
    rng = np.random.default_rng(0)
    for _ in range(2000):
        _apply_spot_price_walk(state, rng)
        assert sp.price_min - 1e-9 <= state.cluster.spot_price <= sp.price_max + 1e-9


def test_spot_price_walk_is_rounded_to_two_decimals():
    cfg = load_config()
    state = _fresh_state(cfg)
    rng = np.random.default_rng(7)
    for _ in range(100):
        _apply_spot_price_walk(state, rng)
        assert round(state.cluster.spot_price, 2) == state.cluster.spot_price


# ---------------------------------------------------------------------------
# Node failure — per_step_single_victim (Phase-1 default)
# ---------------------------------------------------------------------------
def test_node_failure_per_step_single_victim_kills_at_most_one_task():
    cfg = load_config()
    state = _fresh_state(cfg, seed=5)
    _populate_running_tasks(state, 5)
    pre = len(state.running_tasks)
    rng = np.random.default_rng(0)
    for _ in range(10):
        before = len(state.running_tasks)
        _apply_node_failure(state, rng)
        after = len(state.running_tasks)
        assert after in {before, before - 1}
    assert pre - len(state.running_tasks) >= 0


def test_node_failure_per_step_single_victim_no_effect_when_idle():
    cfg = load_config()
    state = _fresh_state(cfg)
    assert not state.running_tasks
    rng = np.random.default_rng(0)
    for _ in range(200):
        _apply_node_failure(state, rng)
    failed = sum(1 for job in state.jobs for task in job.tasks.values() if task.state == TASK_STATE_FAILED)
    assert failed == 0


def test_node_failure_per_step_rate_matches_prob_statistically():
    cfg = load_config()
    rng = np.random.default_rng(1234)
    trials = 4000
    successes = 0
    # Monte Carlo: use a fresh state with ≥1 running task, apply the process
    # once, count whether a kill happened.
    for _ in range(trials):
        state = _fresh_state(cfg, seed=rng.integers(0, 10**9), num_jobs=8)
        _populate_running_tasks(state, 2)
        pre = len(state.running_tasks)
        _apply_node_failure(state, rng)
        if len(state.running_tasks) < pre:
            successes += 1
    observed = successes / trials
    target = cfg.simulator.stochastic_processes.node_failure.prob
    # Normal approximation: std ≈ sqrt(p(1-p)/n) ≈ 0.0034 for n=4000, p=0.05
    assert abs(observed - target) < 0.01, f"observed={observed} target={target}"


# ---------------------------------------------------------------------------
# Node failure — per_node_bernoulli alternative mode
# ---------------------------------------------------------------------------
def test_node_failure_per_node_bernoulli_can_kill_multiple_in_one_step():
    raw = _deep_copy(load_config().raw)
    raw["simulator"]["stochastic_processes"]["node_failure"] = {
        "mode": "per_node_bernoulli",
        "prob": 0.9,
    }
    cfg = build_run_config(raw)
    state = _fresh_state(cfg, seed=13)
    _populate_running_tasks(state, 4)
    pre = len(state.running_tasks)
    rng = np.random.default_rng(0)
    _apply_node_failure(state, rng)
    killed = pre - len(state.running_tasks)
    # With p=0.9 on 4 running tasks, expected kills ≈ 3.6; >=2 with
    # overwhelming probability.
    assert killed >= 2, f"expected ≥2 kills at p=0.9, got {killed}"


# ---------------------------------------------------------------------------
# Data spike — additive_bump (Phase-1 default)
# ---------------------------------------------------------------------------
def test_data_spike_additive_bump_monotonically_increases_remaining_time():
    cfg = load_config()
    state = _fresh_state(cfg, seed=11, num_jobs=25)
    before = {
        task.full_id(): (task.remaining_time, task.cpu_demand)
        for job in state.jobs
        for task in job.tasks.values()
    }
    rng = np.random.default_rng(0)
    # Crank many applications; each should either leave pending tasks
    # unchanged or only bump them upward.
    for _ in range(50):
        _apply_data_spike(state, rng)
    for job in state.jobs:
        for task in job.tasks.values():
            prev_rem, prev_cpu = before[task.full_id()]
            if task.state in {"waiting", "ready"}:
                assert task.remaining_time >= prev_rem
                assert task.cpu_demand >= prev_cpu
                assert task.cpu_demand <= cfg.simulator.stochastic_processes.data_spike.cpu_cap


def test_data_spike_bernoulli_rate_matches_prob_statistically():
    cfg = load_config()
    trials = 2000
    rng = np.random.default_rng(77)
    successes = 0
    for _ in range(trials):
        state = _fresh_state(cfg, seed=rng.integers(0, 10**9), num_jobs=8)
        log_before = len(state.event_log)
        _apply_data_spike(state, rng)
        if len(state.event_log) > log_before:
            successes += 1
    observed = successes / trials
    target = cfg.simulator.stochastic_processes.data_spike.prob
    assert abs(observed - target) < 0.015, f"observed={observed} target={target}"


# ---------------------------------------------------------------------------
# Data spike — multiplicative_10x alternative mode
# ---------------------------------------------------------------------------
def test_data_spike_multiplicative_10x_scales_remaining_time_by_multiplier():
    raw = _deep_copy(load_config().raw)
    raw["simulator"]["stochastic_processes"]["data_spike"] = {
        "mode": "multiplicative_10x",
        "prob": 1.0,   # force a hit
        "min_tasks": 1,
        "max_tasks": 1,
        "duration_bump": 1,
        "cpu_bump": 1,
        "cpu_cap": 60,
        "multiplier": 10,
        "duration_steps": 3,
    }
    cfg = build_run_config(raw)
    state = _fresh_state(cfg, seed=2, num_jobs=5)
    # Pick a ready task and record its remaining_time
    ready = state.ready_tasks()
    assert ready
    target = ready[0]
    orig_rem = target.remaining_time
    orig_cpu = target.cpu_demand
    rng = np.random.default_rng(0)
    _apply_data_spike(state, rng)
    # Some task in {waiting, ready} got its rem * 10; verify at least one.
    bumped = False
    for job in state.jobs:
        for task in job.tasks.values():
            if task.state in {"waiting", "ready"} and task.remaining_time >= orig_rem * cfg.simulator.stochastic_processes.data_spike.multiplier:
                bumped = True
                break
    assert bumped


# ---------------------------------------------------------------------------
# Invariants across all modes
# ---------------------------------------------------------------------------
def test_unknown_stochastic_mode_raises_at_config_load():
    raw = _deep_copy(load_config().raw)
    raw["simulator"]["stochastic_processes"]["node_failure"]["mode"] = "not_a_real_mode"
    with pytest.raises(ValueError):
        build_run_config(raw)
