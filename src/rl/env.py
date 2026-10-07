from __future__ import annotations

"""Gymnasium wrapper around the orchestration simulator.

State:  eight-dimensional float vector in [0, 1]^8 matching the ordering of
        :class:`src.state.StateVector`.
Action: ``Discrete(6)`` over the action list defined in
        :data:`src.sim_environment.ACTIONS`.
Reward: per-step contribution to the shared evaluation utility (paper
        Eq. 7, §4.3.3):

            r_t = alpha*  * Delta_Value_t
                  - beta*  * cost(s_t, a_t)
                  - gamma* * Delta_Risk_t

        where (alpha*, beta*, gamma*) are the evaluation weights
        (default: (1.0, 0.1, 1.0)), Delta_Value_t is the sum of job values
        completed during step t, cost is the pure cost function from
        :mod:`src.cost`, and Delta_Risk_t is the change in the normalised
        ``recent_failures`` counter at step t.

Episode termination:
- All jobs terminal (completed or failed), OR
- The per-episode step cap is reached. Under memoryless-termination mode
  the cap is drawn once per ``reset`` from ``Exp(mu)``; under the
  deterministic mode it is the requested ``max_steps``.
"""

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import numpy as np

try:
    import gymnasium as gym
    from gymnasium import spaces
except ImportError as exc:  # pragma: no cover - surfaced at import time
    raise RuntimeError(
        "gymnasium is required for the RL env wrapper; run `pip install gymnasium`."
    ) from exc

from ..config import RunConfig, load_config
from ..cost import cost as cost_fn
from ..sim_environment import (
    ACTIONS,
    EpisodeState,
    WorkloadGenerator,
    advance_one_step,
    do_action,
    make_episode_rngs,
)


PHASE5_STATE_FIELD_ORDER: Tuple[str, ...] = (
    "cpu_load",
    "ram_available",
    "queue_depth",
    "spot_price",
    "dag_ready_nodes",
    "job_priority",
    "deadline_urgency",
    "recent_failures",
)
PHASE6_V1_STATE_FIELD_ORDER: Tuple[str, ...] = PHASE5_STATE_FIELD_ORDER + (
    "queue_len_abs_norm",
    "mean_remaining_work",
    "max_deadline_urgency",
    "mean_job_value",
    "max_job_value",
    "spot_price_forecast",
)

# Kept as a module-level constant for backwards compatibility with Phase-5
# scripts / tests. New code should call :func:`state_field_order(cfg)`.
STATE_FIELD_ORDER: Tuple[str, ...] = PHASE5_STATE_FIELD_ORDER

NUM_ACTIONS = len(ACTIONS)
# Default feature count is the Phase-5 8. The source-of-truth for the
# actual width used by the policy is :func:`observation_dim(cfg)`.
NUM_STATE_FEATURES = len(PHASE5_STATE_FIELD_ORDER)


def state_field_order(cfg: "RunConfig") -> Tuple[str, ...]:
    """Return the ordered tuple of StateVector attribute names used when
    building observations, depending on ``cfg.state_v2.use_richer_state``.
    """
    if cfg.state_v2.use_richer_state:
        return PHASE6_V1_STATE_FIELD_ORDER
    return PHASE5_STATE_FIELD_ORDER


def observation_dim(cfg: "RunConfig") -> int:
    """Single source of truth for the policy's input layer width."""
    return len(state_field_order(cfg))


@dataclass
class EvalWeights:
    alpha: float
    beta: float
    gamma: float


def evaluation_weights_from_config(cfg: RunConfig) -> EvalWeights:
    """Pick the evaluation utility weights for the RL reward.

    Precedence: ``cfg.sweep.evaluation_utility`` if present, else
    ``cfg.utility``. In the Phase-1 and Phase-4 configs these are equal
    (1.0, 0.1, 1.0) by design.
    """
    eu = getattr(cfg.sweep, "evaluation_utility", None)
    if eu is not None:
        return EvalWeights(alpha=eu.alpha, beta=eu.beta, gamma=eu.gamma)
    return EvalWeights(
        alpha=cfg.utility.alpha,
        beta=cfg.utility.beta,
        gamma=cfg.utility.gamma,
    )


class OrchestrationEnv(gym.Env):
    """Gymnasium environment wrapping the orchestration simulator.

    Constructor options
    -------------------
    cfg:              the project :class:`RunConfig`.
    num_jobs, max_steps:
                      override the episode size. Falls back to
                      ``cfg.experiment.num_jobs / max_steps``.
    memoryless:       if True, ``reset`` samples a per-episode horizon
                      ``tau ~ Exp(mu=max_steps)`` and truncates on `tau`.
    reset_rng_seed:   if set, drives the horizon draw (deterministic smoke).
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        cfg: Optional[RunConfig] = None,
        num_jobs: Optional[int] = None,
        max_steps: Optional[int] = None,
        memoryless: bool = False,
        reset_rng_seed: Optional[int] = None,
        eval_weights: Optional[EvalWeights] = None,
    ) -> None:
        self.cfg = cfg if cfg is not None else load_config()
        self.num_jobs = int(num_jobs) if num_jobs is not None else self.cfg.experiment.num_jobs
        self.max_steps = int(max_steps) if max_steps is not None else self.cfg.experiment.max_steps
        self.memoryless = bool(memoryless)
        self.eval_weights = eval_weights or evaluation_weights_from_config(self.cfg)

        # Phase-6 V1: observation width is driven by cfg.state_v2.
        self._state_field_order = state_field_order(self.cfg)
        self._observation_dim = len(self._state_field_order)

        self.observation_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(self._observation_dim,),
            dtype=np.float32,
        )
        self.action_space = spaces.Discrete(NUM_ACTIONS)

        self._state: Optional[EpisodeState] = None
        self._event_rng: Optional[np.random.Generator] = None
        # Dedicated RNG for the memoryless horizon draw. Seeded
        # deterministically on reset so the horizon is reproducible per
        # seed without coupling to the env / workload streams.
        self._horizon_rng: np.random.Generator = np.random.default_rng(
            reset_rng_seed if reset_rng_seed is not None else 0
        )
        self._current_horizon: int = self.max_steps
        self._last_recent_failures_normalised: float = 0.0

    # ---- gym interface -----------------------------------------------------
    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        if seed is None:
            raise ValueError("OrchestrationEnv.reset requires an explicit seed.")
        super().reset(seed=seed)
        workload_rng, event_rng = make_episode_rngs(int(seed))
        generator = WorkloadGenerator(
            rng=workload_rng,
            cfg=self.cfg,
            num_jobs=self.num_jobs,
            seed_label=int(seed),
        )
        self._state = generator.generate_episode()
        self._event_rng = event_rng
        self._last_recent_failures_normalised = self._normalised_recent_failures()

        if self.memoryless:
            mu = float(self.max_steps)
            horizon = int(self._horizon_rng.exponential(scale=mu))
            # Guard against 0-step episodes that would break the
            # downstream trainer; also cap at 4*mu to keep batches bounded.
            horizon = max(1, min(horizon, int(4 * mu)))
            self._current_horizon = horizon
        else:
            self._current_horizon = self.max_steps

        return self._observe(), {"horizon": self._current_horizon}

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        if self._state is None or self._event_rng is None:
            raise RuntimeError("step() called before reset().")
        action_name = ACTIONS[int(action)]

        # Measure ΔValue_t as the value of jobs that became completed
        # during this step.
        total_value_before = sum(
            job.value for job in self._state.jobs if job.completed
        )
        failed_jobs_before = sum(1 for job in self._state.jobs if job.failed)

        # Execute one simulator tick. Use advance_one_step so action, task
        # progress, and events all fire in the canonical order.
        advance_one_step(self._state, self._event_rng, action_name)

        # ΔValue, Δrisk on the new state.
        total_value_after = sum(
            job.value for job in self._state.jobs if job.completed
        )
        delta_value = float(total_value_after - total_value_before)
        failed_jobs_after = sum(1 for job in self._state.jobs if job.failed)

        # Risk term. Two modes (cfg.rl.reward_risk_mode):
        #   counter_delta     - Δ(normalised recent_failures). Phase-5/6
        #                       default; does NOT telescope to the metric.
        #   failed_jobs_delta - jobs newly failed this step; Σ_t = failed_jobs,
        #                       so Σ_t r_t == the episode utility exactly.
        recent_failures_normalised = self._normalised_recent_failures()
        if self.cfg.rl.reward_risk_mode == "failed_jobs_delta":
            delta_risk = float(failed_jobs_after - failed_jobs_before)
        else:
            delta_risk = float(
                recent_failures_normalised - self._last_recent_failures_normalised
            )
        self._last_recent_failures_normalised = recent_failures_normalised

        step_cost = float(cost_fn(self._state, action_name))

        reward = (
            self.eval_weights.alpha * delta_value
            - self.eval_weights.beta * step_cost
            - self.eval_weights.gamma * delta_risk
        )

        terminated = self._state.all_done()
        truncated = (not terminated) and self._state.step >= self._current_horizon

        info: Dict[str, Any] = {
            "action_name": action_name,
            "delta_value": delta_value,
            "step_cost": step_cost,
            "delta_risk": delta_risk,
            "cpu_in_use": self._state.cpu_in_use(),
            "ram_in_use": self._state.ram_in_use(),
            "queue_depth": self._state.queue_depth(),
            "spot_price": self._state.cluster.spot_price,
            "step": self._state.step,
        }
        return self._observe(), float(reward), bool(terminated), bool(truncated), info

    # ---- helpers -----------------------------------------------------------
    def _observe(self) -> np.ndarray:
        sv = self._state.state_vector()
        arr = np.array(
            [getattr(sv, name) for name in self._state_field_order],
            dtype=np.float32,
        )
        # Clip defensively in case rounding + normalisation ever strays.
        # Spot-price fields have a nominal lower bound of 0.1, but the
        # observation-space lower bound is 0.0 so clipping to [0, 1] is
        # always safe for the policy.
        return np.clip(arr, 0.0, 1.0)

    def _normalised_recent_failures(self) -> float:
        norm = self.cfg.simulator.state_vector.recent_failures_norm
        return float(min(self._state.cluster.recent_failures / max(norm, 1e-9), 1.0))


def make_env(
    cfg: Optional[RunConfig] = None,
    num_jobs: Optional[int] = None,
    max_steps: Optional[int] = None,
    memoryless: bool = False,
    reset_rng_seed: Optional[int] = None,
) -> OrchestrationEnv:
    """Factory used by training scripts and tests."""
    return OrchestrationEnv(
        cfg=cfg,
        num_jobs=num_jobs,
        max_steps=max_steps,
        memoryless=memoryless,
        reset_rng_seed=reset_rng_seed,
    )


__all__ = [
    "ACTIONS",
    "EvalWeights",
    "NUM_ACTIONS",
    "NUM_STATE_FEATURES",
    "OrchestrationEnv",
    "PHASE5_STATE_FIELD_ORDER",
    "PHASE6_V1_STATE_FIELD_ORDER",
    "STATE_FIELD_ORDER",
    "evaluation_weights_from_config",
    "make_env",
    "observation_dim",
    "state_field_order",
]
