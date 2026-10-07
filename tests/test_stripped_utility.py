from __future__ import annotations

"""Stripped Utility-Based agent: one-line change behavioural test."""

from src.config import load_config
from src.reflex_agent import build_reflex_agent
from src.sim_environment import (
    WorkloadGenerator,
    advance_one_step,
    make_episode_rngs,
)
from src.utility_agent import StrippedUtilityBasedAgent, UtilityBasedAgent


def _full_trace(agent_factory, seed: int, max_steps: int = 300):
    cfg = load_config()
    workload_rng, event_rng = make_episode_rngs(seed)
    state = WorkloadGenerator(
        rng=workload_rng, cfg=cfg, num_jobs=cfg.experiment.num_jobs, seed_label=seed,
    ).generate_episode()
    agent = agent_factory(cfg)
    trace = []
    while not state.all_done() and state.step < max_steps:
        action = agent.choose_action(state)
        trace.append(action)
        advance_one_step(state, event_rng, action)
    return trace


def test_stripped_trace_differs_from_full_on_seed_100():
    full_trace = _full_trace(UtilityBasedAgent, seed=100)
    stripped_trace = _full_trace(StrippedUtilityBasedAgent, seed=100)
    # Per the Phase-3 ablation, seed 100 fires the force-execute guard on
    # roughly 249/300 steps. Stripped and full must therefore disagree at
    # least once across the 300-step episode.
    shared = min(len(full_trace), len(stripped_trace))
    disagreements = sum(
        1 for i in range(shared) if full_trace[i] != stripped_trace[i]
    )
    assert disagreements > 0, "Stripped UB and full UB produced identical traces"


def test_stripped_agent_name_and_guard_flag():
    cfg = load_config()
    agent = StrippedUtilityBasedAgent(cfg)
    assert agent.name == "Stripped Utility-Based Agent"
    assert agent.force_execute_guard_enabled is False


def test_full_agent_has_guard_enabled_by_default():
    cfg = load_config()
    agent = UtilityBasedAgent(cfg)
    assert agent.force_execute_guard_enabled is True
