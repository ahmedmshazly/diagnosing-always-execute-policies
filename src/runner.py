from __future__ import annotations

"""Single shared episode runner used by every agent.

The midterm code duplicated the episode loop once per agent. This module
collapses them. Every agent implements ``choose_action(state)`` and is called
through :func:`run_episode` or :func:`run_many_episodes` with an explicit
``RunConfig`` and integer seed.

Budget policy
-------------
A run is "capped" when ``max_steps`` hits before the workload terminates.
When ``uncapped=True``, the loop uses ``cfg.experiment.uncapped_max_steps``
instead, which is intentionally large so that every workload either completes
or fails deterministically. That second pass populates
``uncapped_completion_rate`` on the metrics record.
"""

from typing import Callable, List, Protocol

import numpy as np

from .config import RunConfig
from .cost import cost as cost_fn
from .metrics import EpisodeMetrics, summarize_episode
from .sim_environment import (
    EpisodeState,
    WorkloadGenerator,
    advance_one_step,
    make_episode_rngs,
)


class Agent(Protocol):
    name: str

    def choose_action(self, state: EpisodeState) -> str:  # pragma: no cover - structural
        ...


AgentFactory = Callable[[RunConfig], Agent]


def _simulate(
    cfg: RunConfig,
    agent: Agent,
    seed: int,
    num_jobs: int,
    max_steps: int,
) -> tuple[EpisodeState, float, bool]:
    workload_rng, event_rng = make_episode_rngs(seed)
    generator = WorkloadGenerator(rng=workload_rng, cfg=cfg, num_jobs=num_jobs, seed_label=seed)
    state = generator.generate_episode()

    total_compute_cost = 0.0
    while not state.all_done() and state.step < max_steps:
        action = agent.choose_action(state)
        advance_one_step(state, event_rng, action)
        # Cost is accumulated post-step so step_cost reflects the applied
        # action's effect on resource usage; action_cost tariffs are added
        # by cost_fn (zero by default in Phase 1).
        total_compute_cost += cost_fn(state, action)

    hit_budget = (not state.all_done()) and state.step >= max_steps
    return state, total_compute_cost, hit_budget


def run_episode(
    cfg: RunConfig,
    agent_factory: AgentFactory,
    seed: int,
    include_uncapped: bool = True,
) -> EpisodeMetrics:
    """Run one capped episode and (optionally) one uncapped replay."""
    num_jobs = cfg.experiment.num_jobs
    capped_agent = agent_factory(cfg)
    state, total_cost, hit_budget = _simulate(
        cfg=cfg,
        agent=capped_agent,
        seed=seed,
        num_jobs=num_jobs,
        max_steps=cfg.experiment.max_steps,
    )

    uncapped_rate: float | None = None
    if include_uncapped:
        uncapped_agent = agent_factory(cfg)
        uncapped_state, _unused_cost, _unused_hit = _simulate(
            cfg=cfg,
            agent=uncapped_agent,
            seed=seed,
            num_jobs=num_jobs,
            max_steps=cfg.experiment.uncapped_max_steps,
        )
        completed = sum(1 for job in uncapped_state.jobs if job.completed)
        uncapped_rate = completed / max(num_jobs, 1)

    return summarize_episode(
        cfg=cfg,
        seed=seed,
        agent_name=capped_agent.name,
        state=state,
        num_jobs=num_jobs,
        total_compute_cost=total_cost,
        hit_step_budget=hit_budget,
        uncapped_completion_rate=uncapped_rate,
    )


def run_many_episodes(
    cfg: RunConfig,
    agent_factory: AgentFactory,
    seeds: list[int] | tuple[int, ...],
    include_uncapped: bool = True,
) -> List[EpisodeMetrics]:
    return [
        run_episode(cfg=cfg, agent_factory=agent_factory, seed=int(seed), include_uncapped=include_uncapped)
        for seed in seeds
    ]


__all__ = ["Agent", "AgentFactory", "run_episode", "run_many_episodes"]
