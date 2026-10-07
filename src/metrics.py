from __future__ import annotations

"""Shared evaluation primitives for every agent.

This module owns:
- ``EpisodeMetrics``: the per-episode record written to CSV.
- ``estimate_step_cost``: the single definition of the step-level cost, used
  both to accumulate ``total_compute_cost`` and (indirectly via config) by the
  utility agent's local resource-cost estimate.
- ``summarize_episode``: applies the episode-level utility
  ``U = alpha*Value - beta*Cost - gamma*Risk`` using config weights.

No agent owns the utility weights any more — they live in ``cfg.utility``.
"""

from dataclasses import dataclass
from typing import Dict

from .config import RunConfig
from .cost import step_cost as _step_cost
from .sim_environment import EpisodeState


@dataclass
class EpisodeMetrics:
    """Canonical per-episode record.

    Every experiment driver serialises this as one CSV row. Adding a field
    here requires updating the downstream aggregators (`scripts/aggregate.py`)
    but not the individual agent implementations.
    """

    seed: int
    agent_name: str
    alpha: float
    beta: float
    gamma: float
    num_jobs: int
    steps_executed: int
    completed_jobs: int
    failed_jobs: int
    completion_rate: float                  # budget-capped
    value_weighted_completion_rate: float   # fraction of total job value completed
    uncapped_completion_rate: float         # set only by uncapped evaluator
    failure_rate: float
    total_completed_value: float
    total_job_value: float
    total_compute_cost: float
    avg_compute_cost_per_step: float
    total_utility: float
    completed_all_jobs: bool
    hit_step_budget: bool

    def as_dict(self) -> Dict[str, float | int | str | bool]:
        return {
            "seed": self.seed,
            "agent_name": self.agent_name,
            "alpha": self.alpha,
            "beta": self.beta,
            "gamma": self.gamma,
            "num_jobs": self.num_jobs,
            "steps_executed": self.steps_executed,
            "completed_jobs": self.completed_jobs,
            "failed_jobs": self.failed_jobs,
            "completion_rate": round(self.completion_rate, 4),
            "value_weighted_completion_rate": round(self.value_weighted_completion_rate, 4),
            "uncapped_completion_rate": round(self.uncapped_completion_rate, 4),
            "failure_rate": round(self.failure_rate, 4),
            "total_completed_value": round(self.total_completed_value, 4),
            "total_job_value": round(self.total_job_value, 4),
            "total_compute_cost": round(self.total_compute_cost, 4),
            "avg_compute_cost_per_step": round(self.avg_compute_cost_per_step, 4),
            "total_utility": round(self.total_utility, 4),
            "completed_all_jobs": self.completed_all_jobs,
            "hit_step_budget": self.hit_step_budget,
        }


def estimate_step_cost(state: EpisodeState) -> float:
    """Operational cost for the current step.

    Thin shim over :func:`src.cost.step_cost` kept for backward compatibility
    with call sites that predate the explicit cost module.
    """
    return _step_cost(state)


def summarize_episode(
    cfg: RunConfig,
    seed: int,
    agent_name: str,
    state: EpisodeState,
    num_jobs: int,
    total_compute_cost: float,
    hit_step_budget: bool,
    uncapped_completion_rate: float | None = None,
) -> EpisodeMetrics:
    """Roll up a finished episode into the canonical metrics record."""
    u = cfg.utility
    completed_jobs = sum(1 for job in state.jobs if job.completed)
    failed_jobs = sum(1 for job in state.jobs if job.failed)
    total_completed_value = sum(job.value for job in state.jobs if job.completed)
    total_job_value = sum(job.value for job in state.jobs)
    total_utility = (
        (u.alpha * total_completed_value)
        - (u.beta * total_compute_cost)
        - (u.gamma * failed_jobs)
    )

    value_weighted = (
        total_completed_value / total_job_value if total_job_value > 0 else 0.0
    )

    return EpisodeMetrics(
        seed=seed,
        agent_name=agent_name,
        alpha=u.alpha,
        beta=u.beta,
        gamma=u.gamma,
        num_jobs=num_jobs,
        steps_executed=state.step,
        completed_jobs=completed_jobs,
        failed_jobs=failed_jobs,
        completion_rate=completed_jobs / max(num_jobs, 1),
        value_weighted_completion_rate=value_weighted,
        uncapped_completion_rate=(
            uncapped_completion_rate
            if uncapped_completion_rate is not None
            else completed_jobs / max(num_jobs, 1)
        ),
        failure_rate=failed_jobs / max(num_jobs, 1),
        total_completed_value=total_completed_value,
        total_job_value=total_job_value,
        total_compute_cost=total_compute_cost,
        avg_compute_cost_per_step=(total_compute_cost / max(state.step, 1)),
        total_utility=total_utility,
        completed_all_jobs=(completed_jobs == num_jobs),
        hit_step_budget=hit_step_budget,
    )
