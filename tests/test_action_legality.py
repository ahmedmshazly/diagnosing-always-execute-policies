from __future__ import annotations

"""Action legality: every action must be safe to apply in every state.

The reviewer concern is that agent code (especially a learned policy) can
choose any of the six actions from any state. The simulator must never crash,
never violate capacity constraints, and must stay internally consistent no
matter what action we feed it.

This test drives the environment through a battery of states (fresh, idle,
busy, with an active scale boost, under queue pressure, etc.) and applies
every action, checking a suite of invariants after each call.
"""

import numpy as np

from src.config import load_config
from src.sim_environment import (
    ACTIONS,
    WorkloadGenerator,
    advance_one_step,
    apply_random_events,
    do_action,
    launch_task,
    make_episode_rngs,
)


def _fresh_state(cfg, seed: int = 11, num_jobs: int = 15):
    workload_rng, event_rng = make_episode_rngs(seed)
    generator = WorkloadGenerator(rng=workload_rng, cfg=cfg, num_jobs=num_jobs, seed_label=seed)
    return generator.generate_episode(), event_rng


def _check_invariants(state, cfg):
    cluster = cfg.simulator.cluster
    sp = cfg.simulator.stochastic_processes.spot_price
    assert state.cluster.cpu_capacity >= cluster.min_cpu_capacity
    assert state.cluster.ram_capacity >= cluster.min_ram_capacity
    assert state.cluster.spot_price >= sp.price_min - 1e-9
    assert state.cluster.spot_price <= sp.price_max + 1e-9
    assert state.cpu_in_use() <= state.cluster.cpu_capacity
    assert state.ram_in_use() <= state.cluster.ram_capacity
    assert state.cluster.recent_failures >= 0
    assert state.cluster.recent_failures <= cluster.max_recent_failures
    assert state.cluster.scale_boost_remaining >= 0
    for job in state.jobs:
        for task in job.tasks.values():
            assert task.state in {
                "waiting",
                "ready",
                "running",
                "completed",
                "failed",
                "paused",
            }
    # a job cannot be both completed and failed
    for job in state.jobs:
        assert not (job.completed and job.failed)


def test_every_action_is_legal_from_fresh_state():
    cfg = load_config()
    for action in ACTIONS:
        state, event_rng = _fresh_state(cfg)
        advance_one_step(state, event_rng, action)
        _check_invariants(state, cfg)


def test_every_action_is_legal_under_queue_pressure():
    cfg = load_config()
    for action in ACTIONS:
        state, event_rng = _fresh_state(cfg, seed=4, num_jobs=40)
        for _ in range(10):
            advance_one_step(state, event_rng, "Execute_Ready_Job")
        advance_one_step(state, event_rng, action)
        _check_invariants(state, cfg)


def test_every_action_is_legal_with_active_scale_boost():
    cfg = load_config()
    for action in ACTIONS:
        state, event_rng = _fresh_state(cfg, seed=21, num_jobs=20)
        advance_one_step(state, event_rng, "Scale_Up")
        assert state.cluster.scale_boost_remaining > 0
        advance_one_step(state, event_rng, action)
        _check_invariants(state, cfg)


def test_every_action_is_legal_from_idle_state():
    """Fast-forward an episode until no ready tasks remain, then hit every action."""
    cfg = load_config()
    for action in ACTIONS:
        state, event_rng = _fresh_state(cfg, seed=2, num_jobs=10)
        for _ in range(40):
            advance_one_step(state, event_rng, "Defer_Job")
        advance_one_step(state, event_rng, action)
        _check_invariants(state, cfg)


def test_do_action_returns_debug_label_for_every_action():
    cfg = load_config()
    state, _event_rng = _fresh_state(cfg)
    for action in ACTIONS:
        label = do_action(state, action)
        assert isinstance(label, str) and label


def test_launch_task_rejects_non_ready():
    cfg = load_config()
    state, _event_rng = _fresh_state(cfg)
    ready = state.ready_tasks()
    task = ready[0]
    task.state = "waiting"
    assert launch_task(state, task) is False


def test_apply_random_events_never_violates_invariants_over_many_draws():
    cfg = load_config()
    state, event_rng = _fresh_state(cfg, seed=123)
    rng = np.random.default_rng(999)
    for _ in range(200):
        apply_random_events(state, rng)
        state.step += 1
        _check_invariants(state, cfg)
