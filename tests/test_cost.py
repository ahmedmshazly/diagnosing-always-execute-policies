from __future__ import annotations

"""Tests for :mod:`src.cost` — the pure cost function.

Covers the formula
``cost(state, action) = step_cost(state) + action_cost(action, state)``
and verifies that Phase-1 ``action_costs`` default to 0.0 so that
``total_compute_cost`` is unchanged relative to the midterm baseline.
"""

import pytest

from src.config import build_run_config, load_config
from src.cost import action_cost, cost, step_cost
from src.metrics import estimate_step_cost
from src.sim_environment import RunningTask, WorkloadGenerator, make_episode_rngs


def _fresh_state(cfg, seed=3, num_jobs=10):
    workload_rng, _ = make_episode_rngs(seed)
    generator = WorkloadGenerator(rng=workload_rng, cfg=cfg, num_jobs=num_jobs, seed_label=seed)
    return generator.generate_episode()


def _deep_copy(value):
    if isinstance(value, dict):
        return {k: _deep_copy(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_deep_copy(v) for v in value]
    return value


def test_step_cost_matches_documented_formula():
    cfg = load_config()
    state = _fresh_state(cfg)
    state.cluster.spot_price = 0.4
    state.running_tasks["jobX:t"] = RunningTask(full_id="jobX:t", cpu_demand=5, ram_demand=3, remaining_time=2)
    # 0.4 * (0.6 * 5 + 0.4 * 3) = 0.4 * 4.2 = 1.68
    assert abs(step_cost(state) - 1.68) < 1e-9


def test_step_cost_is_zero_when_idle_and_cheap():
    cfg = load_config()
    state = _fresh_state(cfg)
    state.cluster.spot_price = 0.1
    assert state.cpu_in_use() == 0
    assert state.ram_in_use() == 0
    assert step_cost(state) == 0.0


def test_action_cost_is_zero_by_default_for_every_action():
    cfg = load_config()
    state = _fresh_state(cfg)
    for action in cfg.simulator.actions:
        assert action_cost(action, state) == 0.0


def test_cost_equals_step_cost_when_action_costs_are_zero():
    cfg = load_config()
    state = _fresh_state(cfg)
    state.cluster.spot_price = 0.5
    state.running_tasks["jobX:t"] = RunningTask(full_id="jobX:t", cpu_demand=4, ram_demand=6, remaining_time=1)
    for action in cfg.simulator.actions:
        assert abs(cost(state, action) - step_cost(state)) < 1e-12


def test_action_cost_tariff_takes_effect_when_set():
    raw = _deep_copy(load_config().raw)
    raw["simulator"]["cost"]["action_costs"]["Scale_Up"] = 2.5
    cfg = build_run_config(raw)
    state = _fresh_state(cfg)
    state.cluster.spot_price = 0.8
    # action_cost = spot_price * action_costs[action] = 0.8 * 2.5 = 2.0
    assert abs(action_cost("Scale_Up", state) - 2.0) < 1e-12
    # Other actions still cost zero.
    assert action_cost("Defer_Job", state) == 0.0


def test_estimate_step_cost_is_a_shim_over_step_cost():
    cfg = load_config()
    state = _fresh_state(cfg)
    state.cluster.spot_price = 0.7
    state.running_tasks["jobX:t"] = RunningTask(full_id="jobX:t", cpu_demand=3, ram_demand=5, remaining_time=1)
    assert estimate_step_cost(state) == step_cost(state)


def test_unknown_action_raises_on_action_cost_lookup():
    cfg = load_config()
    state = _fresh_state(cfg)
    with pytest.raises(KeyError):
        action_cost("Not_A_Real_Action", state)
