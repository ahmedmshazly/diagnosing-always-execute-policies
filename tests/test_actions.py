from __future__ import annotations

"""Action parameterisation tests.

Each action must be driven by its named parameter set from
``cfg.simulator.action_params``. These tests verify that the effects match
the spec in SPECIFICATION.md §2 and that changing a parameter actually
changes the observed behaviour.
"""

import numpy as np

from src.config import build_run_config, load_config
from src.sim_environment import (
    WorkloadGenerator,
    advance_one_step,
    do_action,
    make_episode_rngs,
)


def _fresh_state(cfg, seed=11, num_jobs=15):
    workload_rng, event_rng = make_episode_rngs(seed)
    generator = WorkloadGenerator(rng=workload_rng, cfg=cfg, num_jobs=num_jobs, seed_label=seed)
    return generator.generate_episode(), event_rng


def test_scale_up_uses_configured_deltas_and_duration():
    cfg = load_config()
    state, _ = _fresh_state(cfg)
    base_cpu = state.cluster.cpu_capacity
    base_ram = state.cluster.ram_capacity
    do_action(state, "Scale_Up")
    assert state.cluster.cpu_capacity == base_cpu + cfg.simulator.action_params.scale_up.cpu_delta
    assert state.cluster.ram_capacity == base_ram + cfg.simulator.action_params.scale_up.ram_delta
    assert state.cluster.scale_boost_remaining == cfg.simulator.action_params.scale_up.duration_steps
    assert state.cluster.last_scale_up_cpu_delta == cfg.simulator.action_params.scale_up.cpu_delta


def test_scale_up_decay_subtracts_last_applied_delta():
    """Decay uses the last applied delta, not the current config delta."""
    raw = dict(load_config().raw)
    raw = _deep_copy(raw)
    raw["simulator"]["action_params"]["Scale_Up"] = {"cpu_delta": 5, "ram_delta": 6, "duration_steps": 2}
    cfg = build_run_config(raw)
    state, event_rng = _fresh_state(cfg)
    base_cpu, base_ram = state.cluster.cpu_capacity, state.cluster.ram_capacity
    advance_one_step(state, event_rng, "Scale_Up")
    # after 2 decay ticks capacity returns to baseline
    advance_one_step(state, event_rng, "Defer_Job")
    advance_one_step(state, event_rng, "Defer_Job")
    assert state.cluster.cpu_capacity == base_cpu
    assert state.cluster.ram_capacity == base_ram


def test_scale_down_respects_min_capacity_floors():
    cfg = load_config()
    state, _ = _fresh_state(cfg)
    cluster_cfg = cfg.simulator.cluster
    for _ in range(20):
        do_action(state, "Scale_Down")
    assert state.cluster.cpu_capacity == cluster_cfg.min_cpu_capacity
    assert state.cluster.ram_capacity == cluster_cfg.min_ram_capacity


def test_reprioritize_queue_uses_bump_and_cap():
    cfg = load_config()
    state, _ = _fresh_state(cfg)
    bump = cfg.simulator.action_params.reprioritize_queue.bump
    cap = cfg.simulator.action_params.reprioritize_queue.cap
    priors_before = [job.priority for job in state.jobs if not job.completed and not job.failed]
    do_action(state, "Reprioritize_Queue")
    for before, after in zip(priors_before, state.jobs):
        expected = round(min(cap, before + bump), 2)
        assert after.priority == expected


def test_pause_marks_at_most_max_jobs_with_ready_tasks_paused():
    cfg = load_config()
    state, _ = _fresh_state(cfg, seed=3, num_jobs=30)
    params = cfg.simulator.action_params.pause_low_priority_job
    # Force some jobs under threshold
    for job in state.jobs[:5]:
        job.priority = 0.1
    # Force at least one of those to have a ready task
    ready = state.ready_tasks()
    assert ready, "expected some ready tasks in a fresh 30-job episode"
    do_action(state, "Pause_LowPriority_Job")
    paused_tasks = [
        task
        for job in state.jobs
        for task in job.tasks.values()
        if task.state == "paused"
    ]
    # Each paused job contributes at most its ready tasks; we just check
    # that the number of *jobs* touched is bounded by max_jobs.
    paused_jobs = {task.job_id for task in paused_tasks}
    assert len(paused_jobs) <= params.max_jobs


def test_execute_ready_job_respects_selection_policy():
    """value_times_priority picks the highest value*priority task."""
    cfg = load_config()
    state, _ = _fresh_state(cfg, seed=7, num_jobs=8)
    # Override job values/priorities so the target task is unambiguous.
    target_job_id = None
    for idx, job in enumerate(state.jobs):
        if idx == 0:
            job.value = 99.0
            job.priority = 1.0
            target_job_id = idx
        else:
            job.value = 0.1
            job.priority = 0.1
    label = do_action(state, "Execute_Ready_Job")
    assert label is not None
    assert f"job{target_job_id}:" in label


def test_defer_job_is_a_noop_on_cluster_state():
    cfg = load_config()
    state, _ = _fresh_state(cfg)
    snapshot = (state.cluster.cpu_capacity, state.cluster.ram_capacity, state.cluster.spot_price)
    do_action(state, "Defer_Job")
    assert (state.cluster.cpu_capacity, state.cluster.ram_capacity, state.cluster.spot_price) == snapshot


def test_action_params_surface_exists_on_default_config():
    cfg = load_config()
    params = cfg.simulator.action_params
    # Every committed action has a typed parameter dataclass.
    assert hasattr(params, "execute_ready_job")
    assert hasattr(params, "defer_job")
    assert hasattr(params, "scale_up")
    assert hasattr(params, "scale_down")
    assert hasattr(params, "reprioritize_queue")
    assert hasattr(params, "pause_low_priority_job")


def _deep_copy(value):
    if isinstance(value, dict):
        return {k: _deep_copy(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_deep_copy(v) for v in value]
    return value
