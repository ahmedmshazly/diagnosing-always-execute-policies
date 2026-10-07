from __future__ import annotations

"""Core simulation environment.

Every numeric parameter here comes from a ``RunConfig`` produced by
:mod:`src.config`. This module contains no magic numbers.

Randomness discipline:
- Every random draw goes through a named ``numpy.random.Generator``.
- Generators are constructed from ``numpy.random.SeedSequence`` in
  :func:`make_episode_rngs`, so a single integer seed reproduces both the
  workload generator's draws and the event loop's draws independently.

Action semantics are encoded in ``cfg.simulator.action_params``; stochastic
process semantics are encoded in ``cfg.simulator.stochastic_processes``.
See ``SPECIFICATION.md`` §2–§3 for the full spec.
"""

from dataclasses import dataclass, field
from typing import Dict, Final, List, Optional, Tuple

import numpy as np

from .config import RunConfig, SimulatorConfig, load_config
from .state import StateVector


# ---------------------------------------------------------------------------
# Action-name constants (the names are structural; every numeric parameter
# that governs action semantics lives in cfg.simulator.action_params).
# ---------------------------------------------------------------------------
ACTIONS: Final[Tuple[str, ...]] = (
    "Execute_Ready_Job",
    "Defer_Job",
    "Scale_Up",
    "Scale_Down",
    "Reprioritize_Queue",
    "Pause_LowPriority_Job",
)

TASK_STATE_WAITING: Final[str] = "waiting"
TASK_STATE_READY: Final[str] = "ready"
TASK_STATE_RUNNING: Final[str] = "running"
TASK_STATE_COMPLETED: Final[str] = "completed"
TASK_STATE_FAILED: Final[str] = "failed"
TASK_STATE_PAUSED: Final[str] = "paused"


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class TaskTemplate:
    task_id: str
    parents: Tuple[str, ...]
    base_duration: int
    cpu_demand: int
    ram_demand: int


@dataclass(frozen=True)
class DAGTemplate:
    name: str
    tasks: Tuple[TaskTemplate, ...]


@dataclass
class TaskInstance:
    job_id: int
    task_id: str
    parents: Tuple[str, ...]
    remaining_time: int
    cpu_demand: int
    ram_demand: int
    state: str = TASK_STATE_WAITING

    def full_id(self) -> str:
        return f"job{self.job_id}:{self.task_id}"


@dataclass
class JobInstance:
    job_id: int
    template_name: str
    priority: float
    deadline_steps: int
    value: float
    tasks: Dict[str, TaskInstance]
    failed: bool = False
    completed: bool = False

    def update_status(self) -> None:
        if any(task.state == TASK_STATE_FAILED for task in self.tasks.values()):
            self.failed = True
            self.completed = False
            return
        if all(task.state == TASK_STATE_COMPLETED for task in self.tasks.values()):
            self.completed = True
            self.failed = False
            return
        self.failed = False
        self.completed = False


@dataclass
class RunningTask:
    full_id: str
    cpu_demand: int
    ram_demand: int
    remaining_time: int


@dataclass
class ClusterState:
    cpu_capacity: int
    ram_capacity: int
    spot_price: float
    scale_boost_remaining: int = 0
    recent_failures: int = 0
    # Record the (cpu_delta, ram_delta) actually applied by the most recent
    # Scale_Up so that the decay tick can subtract exactly that amount. This
    # decouples decay from cfg defaults in case action parameters change.
    last_scale_up_cpu_delta: int = 0
    last_scale_up_ram_delta: int = 0
    # Phase-6 V1 spot-price forecast. Exponential moving average of
    # ``spot_price`` updated inside ``_apply_spot_price_walk``. Initialised
    # to ``cfg.state_v2.forecast.initial_forecast`` on episode reset.
    spot_price_forecast: float = 0.0


@dataclass
class EpisodeState:
    step: int
    jobs: List[JobInstance]
    cluster: ClusterState
    cfg: RunConfig
    running_tasks: Dict[str, RunningTask] = field(default_factory=dict)
    event_log: List[str] = field(default_factory=list)

    # ---- derived views -----------------------------------------------------
    def ready_tasks(self) -> List[TaskInstance]:
        ready: List[TaskInstance] = []
        for job in self.jobs:
            if job.failed or job.completed:
                continue
            for task in job.tasks.values():
                if task.state == TASK_STATE_WAITING and all(
                    job.tasks[parent].state == TASK_STATE_COMPLETED for parent in task.parents
                ):
                    task.state = TASK_STATE_READY
                if task.state == TASK_STATE_READY:
                    ready.append(task)
        return ready

    def cpu_in_use(self) -> int:
        return sum(task.cpu_demand for task in self.running_tasks.values())

    def ram_in_use(self) -> int:
        return sum(task.ram_demand for task in self.running_tasks.values())

    def available_cpu(self) -> int:
        return max(self.cluster.cpu_capacity - self.cpu_in_use(), 0)

    def available_ram(self) -> int:
        return max(self.cluster.ram_capacity - self.ram_in_use(), 0)

    def queue_depth(self) -> int:
        return len(self.ready_tasks())

    def all_done(self) -> bool:
        return all(job.failed or job.completed for job in self.jobs)

    def state_vector(self) -> StateVector:
        """Return the observation as a typed :class:`~src.state.StateVector`.

        In Phase-5 mode (``cfg.state_v2.use_richer_state=False``) the six
        Phase-6 V1 fields default to 0.0; the 8-dim Phase-5 observation is
        numerically identical to Phase 5's. In Phase-6 V1 mode the six
        extra fields are populated according to the formulas in
        :class:`~src.state.StateVector` and appended after the Phase-5
        block so the 14-dim ordering is fixed and documented.
        """
        ready = self.ready_tasks()
        active_jobs = [job for job in self.jobs if not job.failed and not job.completed]
        sv = self.cfg.simulator.state_vector
        experiment = self.cfg.experiment

        if active_jobs:
            avg_priority = sum(job.priority for job in active_jobs) / len(active_jobs)
            avg_urgency = sum(
                max(0.0, 1.0 - (job.deadline_steps / experiment.max_steps))
                for job in active_jobs
            ) / len(active_jobs)
        else:
            avg_priority = 0.0
            avg_urgency = 0.0

        # ---- Phase-6 V1 features (only populated when flag is on) ----
        state_v2 = self.cfg.state_v2
        if state_v2.use_richer_state:
            (
                queue_len_abs_norm,
                mean_remaining_work,
                max_deadline_urgency,
                mean_job_value,
                max_job_value,
                spot_price_forecast,
            ) = self._compute_v1_features(active_jobs)
        else:
            queue_len_abs_norm = 0.0
            mean_remaining_work = 0.0
            max_deadline_urgency = 0.0
            mean_job_value = 0.0
            max_job_value = 0.0
            spot_price_forecast = 0.0

        return StateVector(
            cpu_load=round(self.cpu_in_use() / max(self.cluster.cpu_capacity, 1), 4),
            ram_available=round(self.available_ram() / max(self.cluster.ram_capacity, 1), 4),
            queue_depth=min(self.queue_depth() / sv.queue_depth_norm, 1.0),
            spot_price=round(self.cluster.spot_price, 4),
            dag_ready_nodes=min(len(ready) / sv.ready_nodes_norm, 1.0),
            job_priority=round(avg_priority, 4),
            deadline_urgency=round(avg_urgency, 4),
            recent_failures=min(self.cluster.recent_failures / sv.recent_failures_norm, 1.0),
            queue_len_abs_norm=queue_len_abs_norm,
            mean_remaining_work=mean_remaining_work,
            max_deadline_urgency=max_deadline_urgency,
            mean_job_value=mean_job_value,
            max_job_value=max_job_value,
            spot_price_forecast=spot_price_forecast,
        )

    def _compute_v1_features(
        self,
        active_jobs: List["JobInstance"],
    ) -> Tuple[float, float, float, float, float, float]:
        """Derive the six Phase-6 V1 features from the current episode.

        Returns ``(queue_len_abs_norm, mean_remaining_work,
        max_deadline_urgency, mean_job_value, max_job_value,
        spot_price_forecast)``. When ``active_jobs`` is empty, the first
        five are ``0.0``; the sixth always reflects the current EMA.
        """
        state_v2 = self.cfg.state_v2
        max_steps = max(self.cfg.experiment.max_steps, 1)
        sp_min = self.cfg.simulator.stochastic_processes.spot_price.price_min
        sp_max = self.cfg.simulator.stochastic_processes.spot_price.price_max
        # Forecast is always defined; clip defensively in case a numerical
        # path ever leaves the documented [sp_min, sp_max] range.
        spot_price_forecast = max(
            sp_min,
            min(sp_max, round(self.cluster.spot_price_forecast, 4)),
        )

        if not active_jobs:
            return (0.0, 0.0, 0.0, 0.0, 0.0, spot_price_forecast)

        # queue_len_abs_norm: active jobs / queue_len_norm, saturated.
        queue_len_abs_norm = min(
            len(active_jobs) / state_v2.queue_features.queue_len_norm,
            1.0,
        )

        # mean_remaining_work: average remaining_time across non-terminal
        # tasks of active jobs, normalised by max_steps. Tasks in terminal
        # (completed / failed) states are excluded because they carry
        # stale remaining_time values.
        remaining_times: List[int] = []
        job_values: List[float] = []
        urgencies: List[float] = []
        for job in active_jobs:
            for task in job.tasks.values():
                if task.state not in {
                    "completed",
                    "failed",
                }:
                    remaining_times.append(task.remaining_time)
            job_values.append(float(job.value))
            urgencies.append(max(0.0, 1.0 - (job.deadline_steps / max_steps)))
        if remaining_times:
            mean_remaining_work = min(
                (sum(remaining_times) / len(remaining_times)) / max_steps,
                1.0,
            )
        else:
            mean_remaining_work = 0.0

        max_deadline_urgency = float(max(urgencies)) if urgencies else 0.0

        vmax = max(state_v2.queue_features.job_value_max, 1e-9)
        mean_job_value = min(
            (sum(job_values) / len(job_values)) / vmax,
            1.0,
        )
        max_job_value = min(max(job_values) / vmax, 1.0)

        return (
            round(queue_len_abs_norm, 4),
            round(mean_remaining_work, 4),
            round(max_deadline_urgency, 4),
            round(mean_job_value, 4),
            round(max_job_value, 4),
            spot_price_forecast,
        )


# ---------------------------------------------------------------------------
# Template instantiation from config
# ---------------------------------------------------------------------------
def build_templates(sim_cfg: SimulatorConfig) -> Tuple[DAGTemplate, ...]:
    templates: List[DAGTemplate] = []
    for template_spec in sim_cfg.dag_templates:
        tasks = tuple(
            TaskTemplate(
                task_id=spec.task_id,
                parents=spec.parents,
                base_duration=spec.base_duration,
                cpu_demand=spec.cpu_demand,
                ram_demand=spec.ram_demand,
            )
            for spec in template_spec.tasks
        )
        templates.append(DAGTemplate(name=template_spec.name, tasks=tasks))
    return tuple(templates)


# ---------------------------------------------------------------------------
# Episode RNG discipline
# ---------------------------------------------------------------------------
def make_episode_rngs(seed: int) -> Tuple[np.random.Generator, np.random.Generator]:
    """Spawn two independent generators from a single integer seed.

    Returns ``(workload_rng, event_rng)``. They are derived via
    ``numpy.random.SeedSequence`` so a run is fully reproducible from one
    integer seed without entangling workload generation with event streams.
    """
    seed_sequence = np.random.SeedSequence(int(seed))
    workload_ss, event_ss = seed_sequence.spawn(2)
    return np.random.default_rng(workload_ss), np.random.default_rng(event_ss)


# ---------------------------------------------------------------------------
# Workload generation
# ---------------------------------------------------------------------------
class WorkloadGenerator:
    """Seeded workload generator.

    Takes an explicit ``np.random.Generator``; call sites are expected to
    derive it from :func:`make_episode_rngs` or equivalent.
    """

    def __init__(
        self,
        rng: np.random.Generator,
        cfg: RunConfig,
        num_jobs: Optional[int] = None,
        seed_label: Optional[int] = None,
    ) -> None:
        self.rng = rng
        self.cfg = cfg
        self.num_jobs = num_jobs if num_jobs is not None else cfg.experiment.num_jobs
        self.seed_label = seed_label
        self.templates = build_templates(cfg.simulator)

    def _clone_template_into_job(self, job_id: int, template: DAGTemplate) -> JobInstance:
        workload = self.cfg.simulator.workload
        duration_factor = float(self.rng.choice(workload.duration_factors))
        cpu_factor = float(self.rng.choice(workload.cpu_factors))
        ram_factor = float(self.rng.choice(workload.ram_factors))

        tasks: Dict[str, TaskInstance] = {}
        for template_task in template.tasks:
            tasks[template_task.task_id] = TaskInstance(
                job_id=job_id,
                task_id=template_task.task_id,
                parents=template_task.parents,
                remaining_time=max(1, int(round(template_task.base_duration * duration_factor))),
                cpu_demand=max(1, int(round(template_task.cpu_demand * cpu_factor))),
                ram_demand=max(1, int(round(template_task.ram_demand * ram_factor))),
            )

        priority = round(
            float(self.rng.uniform(workload.priority_low, workload.priority_high)), 2
        )
        deadline = int(
            self.rng.integers(
                low=workload.deadline_steps_low,
                high=workload.deadline_steps_high + 1,
            )
        )
        # Value draw. The "uniform" branch is byte-identical to the original
        # (one uniform draw) so Phase-5/6 seeds reproduce; the "heavy_tail"
        # branch draws an extra Bernoulli first, making a fraction of jobs
        # high-value "whales".
        if workload.value_distribution == "heavy_tail":
            if float(self.rng.random()) < workload.heavy_tail_prob:
                job_value = float(workload.heavy_tail_value)
            else:
                job_value = round(
                    float(self.rng.uniform(workload.job_value_low, workload.job_value_high)), 2
                )
        else:
            job_value = round(
                float(self.rng.uniform(workload.job_value_low, workload.job_value_high)), 2
            )

        return JobInstance(
            job_id=job_id,
            template_name=template.name,
            priority=priority,
            deadline_steps=deadline,
            value=job_value,
            tasks=tasks,
        )

    def generate_episode(self) -> EpisodeState:
        cluster_cfg = self.cfg.simulator.cluster
        template_indices = self.rng.integers(
            low=0, high=len(self.templates), size=self.num_jobs
        )
        jobs = [
            self._clone_template_into_job(
                job_id=index,
                template=self.templates[int(template_indices[index])],
            )
            for index in range(self.num_jobs)
        ]
        initial_spot_price = round(
            float(
                self.rng.uniform(
                    cluster_cfg.initial_spot_price_low,
                    cluster_cfg.initial_spot_price_high,
                )
            ),
            2,
        )
        cluster = ClusterState(
            cpu_capacity=cluster_cfg.base_cpu_capacity,
            ram_capacity=cluster_cfg.base_ram_capacity,
            spot_price=initial_spot_price,
            # Phase-6 V1: initial forecast value from config. When richer
            # state is disabled the field is still populated but never
            # read, so this is a zero-cost addition for Phase-5 runs.
            spot_price_forecast=float(self.cfg.state_v2.forecast.initial_forecast),
        )
        episode = EpisodeState(step=0, jobs=jobs, cluster=cluster, cfg=self.cfg)
        episode.ready_tasks()  # prime the initial ready-set
        episode.event_log.append(
            f"seed={self.seed_label if self.seed_label is not None else 'na'};"
            f" jobs={self.num_jobs}"
        )
        return episode


# ---------------------------------------------------------------------------
# Stochastic processes — one function per process, dispatched by mode.
# ---------------------------------------------------------------------------
def _apply_spot_price_walk(state: EpisodeState, rng: np.random.Generator) -> None:
    """Bounded random walk on ``cluster.spot_price`` + optional forecast EMA.

    Formula: ``Δ ∼ U(walk_low, walk_high)``; ``spot_price = clip(round(price + Δ,
    2), price_min, price_max)``.

    Phase-6 V1 (paper §6): also updates ``cluster.spot_price_forecast`` as
    an EMA of the new spot price:
        forecast_{t+1} = lambda * spot_price_{t+1} + (1 - lambda) * forecast_t
    The EMA is maintained regardless of ``state_v2.use_richer_state`` so
    the forecast column is always available for inspection; the state
    vector only exposes it when the flag is on.
    """
    config = state.cfg.simulator.stochastic_processes.spot_price
    if config.mode != "bounded_random_walk":
        raise ValueError(f"unknown spot_price mode: {config.mode}")
    walk = float(rng.uniform(config.walk_low, config.walk_high))
    state.cluster.spot_price = min(
        config.price_max,
        max(config.price_min, round(state.cluster.spot_price + walk, 2)),
    )
    lam = float(state.cfg.state_v2.forecast.ema_lambda)
    state.cluster.spot_price_forecast = (
        lam * state.cluster.spot_price
        + (1.0 - lam) * state.cluster.spot_price_forecast
    )


def _apply_node_failure(state: EpisodeState, rng: np.random.Generator) -> None:
    """Apply the ``Node_Failure`` stochastic process.

    Two modes:
    - ``per_step_single_victim`` (Phase-1 default): one uniform ``U(0,1)``
      draw per step; if below ``prob`` and at least one task is running,
      exactly one running task is killed.
    - ``per_node_bernoulli``: an independent Bernoulli(``prob``) draw per
      running task; every losing task is killed in the same step.
    """
    config = state.cfg.simulator.stochastic_processes.node_failure
    cluster_cfg = state.cfg.simulator.cluster
    if config.mode == "per_step_single_victim":
        if not state.running_tasks:
            return
        if rng.random() >= config.prob:
            return
        running_ids = list(state.running_tasks.keys())
        victim_index = int(rng.integers(low=0, high=len(running_ids)))
        _kill_running_task(state, running_ids[victim_index], cluster_cfg.max_recent_failures)
    elif config.mode == "per_node_bernoulli":
        if not state.running_tasks:
            return
        # Snapshot now so victims picked this step don't race against each
        # other via dict mutation.
        running_ids = list(state.running_tasks.keys())
        for full_id in running_ids:
            if rng.random() < config.prob:
                _kill_running_task(state, full_id, cluster_cfg.max_recent_failures)
    elif config.mode == "load_cascade":
        # Per-step single victim, but the failure probability depends on load:
        # high_prob when cpu_load > load_threshold, else low_prob. Keeping load
        # moderate (throttle or scale-up) changes this explicit load-dependent
        # failure probability. Other modes can still depend on running exposure.
        if not state.running_tasks:
            return
        cpu_load = state.cpu_in_use() / max(state.cluster.cpu_capacity, 1)
        prob = config.high_prob if cpu_load > config.load_threshold else config.low_prob
        if rng.random() >= prob:
            return
        running_ids = list(state.running_tasks.keys())
        victim_index = int(rng.integers(low=0, high=len(running_ids)))
        _kill_running_task(state, running_ids[victim_index], cluster_cfg.max_recent_failures)
    else:
        raise ValueError(f"unknown node_failure mode: {config.mode}")


def _kill_running_task(state: EpisodeState, full_id: str, recent_failures_cap: int) -> None:
    if full_id not in state.running_tasks:
        return
    state.running_tasks.pop(full_id)
    job_id_str, task_id = full_id.split(":")
    job_id = int(job_id_str.replace("job", ""))
    job = state.jobs[job_id]
    job.tasks[task_id].state = TASK_STATE_FAILED
    job.update_status()
    state.cluster.recent_failures = min(
        state.cluster.recent_failures + 1,
        recent_failures_cap,
    )
    state.event_log.append(f"step={state.step}: node_failure->{full_id}")


def _apply_data_spike(state: EpisodeState, rng: np.random.Generator) -> None:
    """Apply the ``Data_Spike`` stochastic process.

    Two modes:
    - ``additive_bump`` (Phase-1 default): for ``k ∈ [min_tasks, max_tasks]``
      randomly chosen pending tasks, ``remaining_time += duration_bump`` and
      ``cpu_demand = min(cpu_demand + cpu_bump, cpu_cap)``.
    - ``multiplicative_10x`` (documented, not active by default): scales
      ``remaining_time`` and ``cpu_demand`` by ``multiplier`` on up to
      ``max_tasks`` pending tasks; ``cpu_demand`` is still clamped to
      ``cpu_cap``. ``duration_steps`` is reserved for a future extension
      where subsequently spawned tasks inherit the inflated demand.
    """
    config = state.cfg.simulator.stochastic_processes.data_spike
    if rng.random() >= config.prob:
        return

    candidates = [
        task
        for job in state.jobs
        for task in job.tasks.values()
        if task.state in {TASK_STATE_WAITING, TASK_STATE_READY}
    ]
    if not candidates:
        return

    lo = max(1, min(config.min_tasks, len(candidates)))
    hi_exclusive = min(config.max_tasks, len(candidates)) + 1
    if hi_exclusive <= lo:
        hi_exclusive = lo + 1
    affected_count = int(rng.integers(low=lo, high=hi_exclusive))
    affected_count = max(1, min(affected_count, len(candidates)))

    indices = rng.choice(len(candidates), size=affected_count, replace=False)
    chosen = [candidates[int(index)] for index in np.atleast_1d(indices)]

    if config.mode == "additive_bump":
        for task in chosen:
            task.remaining_time += config.duration_bump
            task.cpu_demand = min(task.cpu_demand + config.cpu_bump, config.cpu_cap)
    elif config.mode == "multiplicative_10x":
        for task in chosen:
            task.remaining_time = int(task.remaining_time * config.multiplier)
            task.cpu_demand = min(
                int(task.cpu_demand * config.multiplier),
                config.cpu_cap,
            )
    else:
        raise ValueError(f"unknown data_spike mode: {config.mode}")

    state.event_log.append(f"step={state.step}: data_spike->{affected_count}_tasks")


def _apply_scale_boost_decay(state: EpisodeState) -> None:
    cluster_cfg = state.cfg.simulator.cluster
    if state.cluster.scale_boost_remaining <= 0:
        return
    state.cluster.scale_boost_remaining -= 1
    if state.cluster.scale_boost_remaining == 0:
        state.cluster.cpu_capacity = max(
            state.cluster.cpu_capacity - state.cluster.last_scale_up_cpu_delta,
            cluster_cfg.base_cpu_capacity,
        )
        state.cluster.ram_capacity = max(
            state.cluster.ram_capacity - state.cluster.last_scale_up_ram_delta,
            cluster_cfg.base_ram_capacity,
        )
        state.cluster.last_scale_up_cpu_delta = 0
        state.cluster.last_scale_up_ram_delta = 0


def apply_random_events(state: EpisodeState, rng: np.random.Generator) -> None:
    """Apply every exogenous event after action/progress for the step."""
    _apply_spot_price_walk(state, rng)
    _apply_node_failure(state, rng)
    _apply_data_spike(state, rng)
    _apply_scale_boost_decay(state)


# ---------------------------------------------------------------------------
# Task lifecycle
# ---------------------------------------------------------------------------
def launch_task(state: EpisodeState, task: TaskInstance) -> bool:
    if task.state != TASK_STATE_READY:
        return False
    if task.cpu_demand > state.available_cpu() or task.ram_demand > state.available_ram():
        return False
    task.state = TASK_STATE_RUNNING
    state.running_tasks[task.full_id()] = RunningTask(
        full_id=task.full_id(),
        cpu_demand=task.cpu_demand,
        ram_demand=task.ram_demand,
        remaining_time=task.remaining_time,
    )
    return True


def progress_running_tasks(state: EpisodeState) -> None:
    completed_task_ids: List[str] = []
    for full_id, running_task in state.running_tasks.items():
        running_task.remaining_time -= 1
        if running_task.remaining_time <= 0:
            completed_task_ids.append(full_id)
    for full_id in completed_task_ids:
        state.running_tasks.pop(full_id)
        job_id_str, task_id = full_id.split(":")
        job_id = int(job_id_str.replace("job", ""))
        task = state.jobs[job_id].tasks[task_id]
        task.remaining_time = 0
        task.state = TASK_STATE_COMPLETED
        state.jobs[job_id].update_status()
    state.cluster.recent_failures = max(state.cluster.recent_failures - 1, 0)


# ---------------------------------------------------------------------------
# Action dispatch
# ---------------------------------------------------------------------------
def _execute_ranking_key(state: EpisodeState, task: TaskInstance) -> float:
    policy = state.cfg.simulator.action_params.execute_ready_job.selection_policy
    job = state.jobs[task.job_id]
    if policy == "value":
        return job.value
    if policy == "value_times_priority":
        return job.value * job.priority
    raise ValueError(f"Unknown selection_policy: {policy}")


def do_action(state: EpisodeState, action: str) -> Optional[str]:
    """Apply one orchestration action.

    Each action consumes a named parameter set from
    ``cfg.simulator.action_params``. See ``SPECIFICATION.md`` §2 for the
    committed semantics.
    """
    ready = state.ready_tasks()
    params = state.cfg.simulator.action_params
    cluster_cfg = state.cfg.simulator.cluster

    if action not in ACTIONS:
        raise ValueError(f"Unknown action: {action}")

    if action == "Execute_Ready_Job":
        if not ready:
            return "no_ready_job"
        ranked = sorted(ready, key=lambda t: _execute_ranking_key(state, t), reverse=True)
        max_launches = params.execute_ready_job.max_launches_per_step
        # Try the top `max_launches` candidates in priority order. If a
        # candidate doesn't fit we stop — matching the midterm semantics of
        # "launch the best task or declare blocked". This preserves
        # behaviour for max_launches=1 (single top task) and naturally
        # generalises to max_launches>1 without switching to greedy
        # packing.
        launched_ids: List[str] = []
        for candidate in ranked[:max_launches]:
            if launch_task(state, candidate):
                launched_ids.append(candidate.full_id())
            else:
                break
        if launched_ids:
            return "launched:" + ",".join(launched_ids)
        return "insufficient_resources"

    if action == "Defer_Job":
        # duration_steps is honoured structurally: the runner calls
        # do_action once per step, so a duration of N means the caller chose
        # Defer_Job N times. We keep the parameter here so a future runner
        # or RL variant can consume it directly.
        _ = params.defer_job.duration_steps
        return "deferred"

    if action == "Scale_Up":
        state.cluster.cpu_capacity += params.scale_up.cpu_delta
        state.cluster.ram_capacity += params.scale_up.ram_delta
        state.cluster.scale_boost_remaining = params.scale_up.duration_steps
        state.cluster.last_scale_up_cpu_delta = params.scale_up.cpu_delta
        state.cluster.last_scale_up_ram_delta = params.scale_up.ram_delta
        return "scaled_up"

    if action == "Scale_Down":
        state.cluster.cpu_capacity = max(
            cluster_cfg.min_cpu_capacity,
            state.cluster.cpu_capacity - params.scale_down.cpu_delta,
        )
        state.cluster.ram_capacity = max(
            cluster_cfg.min_ram_capacity,
            state.cluster.ram_capacity - params.scale_down.ram_delta,
        )
        return "scaled_down"

    if action == "Reprioritize_Queue":
        for job in state.jobs:
            if not job.completed and not job.failed:
                job.priority = round(
                    min(params.reprioritize_queue.cap, job.priority + params.reprioritize_queue.bump),
                    2,
                )
        return "reprioritized"

    if action == "Pause_LowPriority_Job":
        low_priority_jobs = [
            job
            for job in state.jobs
            if not job.completed
            and not job.failed
            and job.priority < params.pause_low_priority_job.priority_threshold
        ]
        paused_jobs = 0
        for job in low_priority_jobs[: params.pause_low_priority_job.max_jobs]:
            for task in job.tasks.values():
                if task.state == TASK_STATE_READY:
                    task.state = TASK_STATE_PAUSED
            paused_jobs += 1
        return f"paused_jobs:{paused_jobs}"

    return None


def advance_one_step(
    state: EpisodeState,
    rng: np.random.Generator,
    action: str,
) -> Dict[str, float | str | int]:
    """Apply an action, advance the world one step, and return a snapshot."""
    event = do_action(state, action)
    progress_running_tasks(state)
    apply_random_events(state, rng)
    state.step += 1
    state.ready_tasks()

    return {
        "step": state.step,
        "action": action,
        "event": event or "none",
        "cpu_in_use": state.cpu_in_use(),
        "ram_in_use": state.ram_in_use(),
        "queue_depth": state.queue_depth(),
        "spot_price": state.cluster.spot_price,
        "done": int(state.all_done()),
    }


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------
def _demo() -> None:
    cfg = load_config()
    seed = 7
    workload_rng, event_rng = make_episode_rngs(seed)
    generator = WorkloadGenerator(rng=workload_rng, cfg=cfg, num_jobs=12, seed_label=seed)
    environment = generator.generate_episode()

    print("Initial state vector:")
    print(environment.state_vector().as_dict())
    print(f"ready_tasks={environment.queue_depth()}")

    for index in range(5):
        step_info = advance_one_step(environment, event_rng, "Execute_Ready_Job")
        print(f"step_info[{index}]={step_info}")

    print("Final partial state vector:")
    print(environment.state_vector().as_dict())
    print(f"events={environment.event_log[:5]}")


if __name__ == "__main__":
    _demo()
