from __future__ import annotations

"""Simulator determinism under fixed seed.

Running the same agent under the same seed and config must produce exactly
the same metrics, event log, and final state. This test covers the repo's
non-negotiable coding practice that every random draw goes through a named,
seeded ``numpy.random.Generator``.
"""

import numpy as np

from src.config import load_config
from src.reflex_agent import build_reflex_agent
from src.runner import run_episode
from src.sim_environment import (
    WorkloadGenerator,
    advance_one_step,
    make_episode_rngs,
)
from src.utility_agent import build_utility_agent


def _run_with_actions(cfg, seed: int, actions: list[str]):
    workload_rng, event_rng = make_episode_rngs(seed)
    generator = WorkloadGenerator(rng=workload_rng, cfg=cfg, num_jobs=20, seed_label=seed)
    state = generator.generate_episode()
    snapshots = []
    for action in actions:
        step_info = advance_one_step(state, event_rng, action)
        snapshots.append({**step_info, "recent_failures": state.cluster.recent_failures})
    return state, snapshots


def test_workload_generator_is_deterministic():
    cfg = load_config()
    r1, _ = make_episode_rngs(42)
    r2, _ = make_episode_rngs(42)
    g1 = WorkloadGenerator(rng=r1, cfg=cfg, num_jobs=10, seed_label=42).generate_episode()
    g2 = WorkloadGenerator(rng=r2, cfg=cfg, num_jobs=10, seed_label=42).generate_episode()
    assert [job.value for job in g1.jobs] == [job.value for job in g2.jobs]
    assert [job.priority for job in g1.jobs] == [job.priority for job in g2.jobs]
    assert [job.deadline_steps for job in g1.jobs] == [job.deadline_steps for job in g2.jobs]


def test_event_stream_is_deterministic():
    cfg = load_config()
    actions = ["Execute_Ready_Job"] * 40
    _, snaps_a = _run_with_actions(cfg, seed=3, actions=actions)
    _, snaps_b = _run_with_actions(cfg, seed=3, actions=actions)
    assert snaps_a == snaps_b


def test_event_stream_differs_across_seeds():
    cfg = load_config()
    actions = ["Execute_Ready_Job"] * 40
    _, snaps_a = _run_with_actions(cfg, seed=1, actions=actions)
    _, snaps_b = _run_with_actions(cfg, seed=2, actions=actions)
    assert snaps_a != snaps_b


def test_reflex_episode_is_deterministic():
    cfg = load_config()
    m1 = run_episode(cfg=cfg, agent_factory=build_reflex_agent, seed=5)
    m2 = run_episode(cfg=cfg, agent_factory=build_reflex_agent, seed=5)
    assert m1.as_dict() == m2.as_dict()


def test_utility_episode_is_deterministic():
    cfg = load_config()
    m1 = run_episode(cfg=cfg, agent_factory=build_utility_agent, seed=5)
    m2 = run_episode(cfg=cfg, agent_factory=build_utility_agent, seed=5)
    assert m1.as_dict() == m2.as_dict()


def test_seed_sequence_spawn_independence():
    """workload_rng and event_rng must not alias."""
    w, e = make_episode_rngs(0)
    assert not np.array_equal(w.random(10), e.random(10))
