from __future__ import annotations

"""Observation dataclass for every agent.

This module defines :class:`StateVector`, the single record every agent
receives as input. Each field carries a docstring pinning its exact type,
unit, range, and how it is computed from the simulator's internal state;
the same specification appears in ``SPECIFICATION.md`` §1.

The class intentionally mirrors the legacy ``dict[str, float]`` observation
that the midterm code returned from ``EpisodeState.state_vector()``. The
``.as_dict()`` helper is retained so existing call sites that look up keys
like ``"CPU_Load"`` by string continue to work while they migrate to
attribute access.
"""

from dataclasses import asdict, dataclass
from typing import Dict


@dataclass(frozen=True)
class StateVector:
    """Normalized observation passed to every agent at decision time.

    All fields are Python ``float`` values. Rounding matches the original v0
    code exactly (4 decimal places on the per-episode-state quantities and
    on ``spot_price``; the saturation-capped fractions are unrounded because
    they go through ``min(..., 1.0)``).

    Field-by-field semantics:

    - ``cpu_load`` — CPU utilization.
      * **Unit:** fraction (0 idle, 1 fully loaded).
      * **Range:** ``[0.0, 1.0]``.
      * **Computation:** ``round(cpu_in_use / max(cluster.cpu_capacity, 1), 4)``.
        The denominator is guarded so a hypothetical scale-to-zero cluster
        never divides by zero.

    - ``ram_available`` — *Free* RAM fraction, not used RAM. Retained under
      the midterm name so that the observation signature is compatible with
      downstream agents.
      * **Unit:** fraction.
      * **Range:** ``[0.0, 1.0]``.
      * **Computation:** ``round(available_ram / max(cluster.ram_capacity, 1), 4)``
        where ``available_ram = max(cluster.ram_capacity - ram_in_use, 0)``.

    - ``queue_depth`` — Ready-queue length, saturated.
      * **Unit:** fraction of the saturation divisor.
      * **Range:** ``[0.0, 1.0]``.
      * **Computation:** ``min(len(ready_tasks) / state_vector.queue_depth_norm, 1.0)``.
        The divisor (default 20.0) is configurable.

    - ``spot_price`` — Current spot price index.
      * **Unit:** dimensionless price (higher = more expensive).
      * **Range:** ``[events.spot_price_min, events.spot_price_max] = [0.1, 1.0]``.
      * **Computation:** ``round(cluster.spot_price, 4)``. The underlying
        field is updated by the spot-price random walk inside
        ``apply_random_events``.

    - ``dag_ready_nodes`` — Count of ready nodes in the DAG, saturated.
      Distinct from ``queue_depth`` only when the two normalisers differ.
      * **Unit:** fraction of the saturation divisor.
      * **Range:** ``[0.0, 1.0]``.
      * **Computation:** ``min(len(ready_tasks) / state_vector.ready_nodes_norm, 1.0)``.

    - ``job_priority`` — Mean priority across active jobs.
      * **Unit:** same scale as ``JobInstance.priority``.
      * **Range:** ``[workload.priority_low, workload.priority_high] = [0.2, 1.0]``;
        ``0.0`` when no jobs are active (e.g. post-termination).
      * **Computation:** ``round(mean(job.priority for job in active_jobs), 4)``
        with ``active_jobs = [job for job in episode.jobs if not job.failed
        and not job.completed]``.

    - ``deadline_urgency`` — Mean fraction of deadline consumed across
      active jobs (not per-step clock elapsed).
      * **Unit:** fraction.
      * **Range:** ``[0.0, 1.0]``.
      * **Computation:**
        ``round(mean(max(0, 1 - job.deadline_steps / experiment.max_steps)) for
        active jobs, 4)``. ``0.0`` when no jobs are active.

    - ``recent_failures`` — Saturated count of recent ``Node_Failure``
      events.
      * **Unit:** fraction of the saturation divisor.
      * **Range:** ``[0.0, 1.0]``.
      * **Computation:** ``min(cluster.recent_failures /
        state_vector.recent_failures_norm, 1.0)``. The underlying integer
        is bumped on every failure, decremented by 1 every step (floor 0),
        and capped at ``cluster.max_recent_failures = 10``.

    Phase-6 V1 richer-state extensions (all 6 defined below). These are
    populated only when ``cfg.state_v2.use_richer_state`` is True; the
    8-dim legacy observation ignores them. Every one is clipped to
    ``[0, 1]`` and defaults to ``0.0`` when there are no active jobs.
    Derivation formulas match Phase-6 §2.1.

    - ``queue_len_abs_norm`` — Absolute queue length (active jobs, not
      tasks) normalised.
      * **Range:** ``[0.0, 1.0]``.
      * **Computation:** ``min(|active_jobs| /
        state_v2.queue_features.queue_len_norm, 1.0)``.
      * **Purpose:** complements ``queue_depth`` (which measures the
        ready-task subset saturated at 20). Gives the policy an
        unsaturated signal for high-job-count regimes.

    - ``mean_remaining_work`` — Average remaining time across the tasks
      in active jobs (waiting / ready / running / paused), normalised
      by ``experiment.max_steps``.
      * **Range:** ``[0.0, 1.0]``.
      * **Computation:** ``min(mean(task.remaining_time over non-terminal
        tasks of active jobs) / experiment.max_steps, 1.0)`` or 0.0 when
        there are no such tasks.

    - ``max_deadline_urgency`` — Maximum deadline urgency across active
      jobs; complements the mean already in ``deadline_urgency``.
      * **Range:** ``[0.0, 1.0]``.
      * **Computation:** ``max(max(0, 1 - job.deadline_steps /
        experiment.max_steps) for active jobs)`` or 0.0 when none.

    - ``mean_job_value`` — Average per-job value across active jobs,
      normalised by ``state_v2.queue_features.job_value_max``.
      * **Range:** ``[0.0, 1.0]``.
      * **Computation:** ``min(mean(job.value) / job_value_max, 1.0)`` or
        0.0 when no active jobs.

    - ``max_job_value`` — Maximum per-job value across active jobs,
      normalised by ``state_v2.queue_features.job_value_max``.
      * **Range:** ``[0.0, 1.0]``.
      * **Computation:** ``min(max(job.value) / job_value_max, 1.0)`` or
        0.0 when no active jobs.

    - ``spot_price_forecast`` — EMA of spot price. Updated every step as
      ``new = lambda * price + (1 - lambda) * old``; initial value is
      ``state_v2.forecast.initial_forecast``.
      * **Range:** ``[events.spot_price_min, events.spot_price_max] =
        [0.1, 1.0]`` once warmed up; reset value configurable.
      * **Computation:** ``round(cluster.spot_price_forecast, 4)``.
    """

    cpu_load: float
    ram_available: float
    queue_depth: float
    spot_price: float
    dag_ready_nodes: float
    job_priority: float
    deadline_urgency: float
    recent_failures: float

    # Phase-6 V1 extensions. Default to 0.0 so that existing call sites
    # that construct an 8-field StateVector still produce a valid object.
    queue_len_abs_norm: float = 0.0
    mean_remaining_work: float = 0.0
    max_deadline_urgency: float = 0.0
    mean_job_value: float = 0.0
    max_job_value: float = 0.0
    spot_price_forecast: float = 0.0

    # Legacy Title_Case keys used by v0 agent code that indexed the
    # observation dict by string. Keeping them 1:1 with the midterm code's
    # field names so we can migrate call sites incrementally.
    _LEGACY_KEY_MAP = {
        "cpu_load": "CPU_Load",
        "ram_available": "RAM_Available",
        "queue_depth": "Queue_Depth",
        "spot_price": "Spot_Price",
        "dag_ready_nodes": "DAG_Ready_Nodes",
        "job_priority": "Job_Priority",
        "deadline_urgency": "Deadline_Urgency",
        "recent_failures": "Recent_Failures",
        # Phase-6 fields use snake_case in their Title_Case aliases so the
        # naming scheme stays internally consistent.
        "queue_len_abs_norm": "Queue_Len_Abs_Norm",
        "mean_remaining_work": "Mean_Remaining_Work",
        "max_deadline_urgency": "Max_Deadline_Urgency",
        "mean_job_value": "Mean_Job_Value",
        "max_job_value": "Max_Job_Value",
        "spot_price_forecast": "Spot_Price_Forecast",
    }

    PHASE5_FIELD_ORDER = (
        "cpu_load",
        "ram_available",
        "queue_depth",
        "spot_price",
        "dag_ready_nodes",
        "job_priority",
        "deadline_urgency",
        "recent_failures",
    )
    PHASE6_V1_FIELD_ORDER = PHASE5_FIELD_ORDER + (
        "queue_len_abs_norm",
        "mean_remaining_work",
        "max_deadline_urgency",
        "mean_job_value",
        "max_job_value",
        "spot_price_forecast",
    )

    def as_dict(self) -> Dict[str, float]:
        """Return the observation with the midterm's Title_Case keys."""
        plain = asdict(self)
        return {self._LEGACY_KEY_MAP[key]: value for key, value in plain.items()}

    def __getitem__(self, key: str) -> float:
        """Dict-style access for both snake_case and legacy Title_Case keys.

        Exists only to let v0 consumers keep their ``state["Recent_Failures"]``
        idioms while the migration to attribute access is in progress.
        """
        if key in self._LEGACY_KEY_MAP.values():
            for snake, title in self._LEGACY_KEY_MAP.items():
                if title == key:
                    return getattr(self, snake)
        if hasattr(self, key):
            return getattr(self, key)
        raise KeyError(key)


__all__ = ["StateVector"]
