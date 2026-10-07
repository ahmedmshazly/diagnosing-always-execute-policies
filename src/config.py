from __future__ import annotations

"""Typed configuration loader for the orchestration project.

Reads ``config/default.yaml`` (or any override) into immutable dataclasses.
Every consumer module (env, agents, drivers) receives a ``RunConfig`` and
does not touch YAML or magic numbers of its own.

The loader also produces a stable sha256 hash of the canonical YAML dump so
``(commit SHA, config hash, seed list)`` uniquely identifies a run.

Structure of the resolved schema is documented in SPECIFICATION.md and in
``config/default.yaml`` itself.
"""

from dataclasses import dataclass, field
import hashlib
import io
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Tuple

import yaml


REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG_PATH = REPO_ROOT / "config" / "default.yaml"


# ---------------------------------------------------------------------------
# Action parameters — one dataclass per action.
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class ExecuteReadyJobParams:
    selection_policy: str
    max_launches_per_step: int


@dataclass(frozen=True)
class DeferJobParams:
    duration_steps: int


@dataclass(frozen=True)
class ScaleUpParams:
    cpu_delta: int
    ram_delta: int
    duration_steps: int


@dataclass(frozen=True)
class ScaleDownParams:
    cpu_delta: int
    ram_delta: int


@dataclass(frozen=True)
class ReprioritizeQueueParams:
    bump: float
    cap: float


@dataclass(frozen=True)
class PauseLowPriorityJobParams:
    priority_threshold: float
    max_jobs: int


@dataclass(frozen=True)
class ActionParams:
    """Container for per-action parameter sets.

    Access pattern: ``cfg.simulator.action_params.scale_up.cpu_delta``.
    """
    execute_ready_job: ExecuteReadyJobParams
    defer_job: DeferJobParams
    scale_up: ScaleUpParams
    scale_down: ScaleDownParams
    reprioritize_queue: ReprioritizeQueueParams
    pause_low_priority_job: PauseLowPriorityJobParams


# ---------------------------------------------------------------------------
# Stochastic processes
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class NodeFailureConfig:
    mode: str          # "per_step_single_victim" | "per_node_bernoulli" | "load_cascade"
    prob: float
    # load_cascade params (hardening — a failure model where caution PAYS):
    # when cpu_load > load_threshold the per-step single-victim failure prob is
    # high_prob, else low_prob. Keeping load moderate (via throttle or scale-up)
    # genuinely reduces expected failures, unlike the policy-invariant
    # per_step_single_victim / per_node_bernoulli models. Defaulted so existing
    # configs load unchanged.
    load_threshold: float = 0.8
    high_prob: float = 0.30
    low_prob: float = 0.02


@dataclass(frozen=True)
class DataSpikeConfig:
    mode: str          # "additive_bump" | "multiplicative_10x"
    prob: float
    min_tasks: int
    max_tasks: int
    duration_bump: int
    cpu_bump: int
    cpu_cap: int
    multiplier: int
    duration_steps: int


@dataclass(frozen=True)
class SpotPriceConfig:
    mode: str          # "bounded_random_walk"
    walk_low: float
    walk_high: float
    price_min: float
    price_max: float


@dataclass(frozen=True)
class StochasticProcessesConfig:
    node_failure: NodeFailureConfig
    data_spike: DataSpikeConfig
    spot_price: SpotPriceConfig


# ---------------------------------------------------------------------------
# Cluster / cost / state vector / workload / DAG
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class ClusterConfig:
    base_cpu_capacity: int
    base_ram_capacity: int
    min_cpu_capacity: int
    min_ram_capacity: int
    scale_boost_duration: int
    max_recent_failures: int
    initial_spot_price_low: float
    initial_spot_price_high: float


@dataclass(frozen=True)
class CostConfig:
    cpu_weight: float
    ram_weight: float
    action_costs: Mapping[str, float]


@dataclass(frozen=True)
class StateVectorConfig:
    queue_depth_norm: float
    ready_nodes_norm: float
    recent_failures_norm: float


@dataclass(frozen=True)
class WorkloadConfig:
    duration_factors: Tuple[float, ...]
    cpu_factors: Tuple[float, ...]
    ram_factors: Tuple[float, ...]
    priority_low: float
    priority_high: float
    deadline_steps_low: int
    deadline_steps_high: int
    job_value_low: float
    job_value_high: float
    # Heavy-tailed value (hardening — a value-heterogeneity lever). When
    # value_distribution == "heavy_tail", each job is a "whale" of value
    # heavy_tail_value with probability heavy_tail_prob, else U(low, high).
    # Default "uniform" leaves the RNG draw sequence unchanged so Phase-5/6
    # reproduce bit-for-bit. Defaulted last so existing configs load.
    value_distribution: str = "uniform"
    heavy_tail_prob: float = 0.0
    heavy_tail_value: float = 50.0


@dataclass(frozen=True)
class DAGTaskSpec:
    task_id: str
    parents: Tuple[str, ...]
    base_duration: int
    cpu_demand: int
    ram_demand: int


@dataclass(frozen=True)
class DAGTemplateSpec:
    name: str
    tasks: Tuple[DAGTaskSpec, ...]


@dataclass(frozen=True)
class SimulatorConfig:
    actions: Tuple[str, ...]
    cluster: ClusterConfig
    action_params: ActionParams
    stochastic_processes: StochasticProcessesConfig
    cost: CostConfig
    state_vector: StateVectorConfig
    workload: WorkloadConfig
    dag_templates: Tuple[DAGTemplateSpec, ...]


# ---------------------------------------------------------------------------
# Utility / agents / experiment / RL / sweep / seeds
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class UtilityWeights:
    alpha: float
    beta: float
    gamma: float
    midterm_values: Mapping[str, float]


@dataclass(frozen=True)
class ReflexAgentConfig:
    scale_down_spot_price_threshold: float
    scale_up_queue_depth_threshold: int
    scale_up_spot_price_threshold: float


@dataclass(frozen=True)
class UtilityAgentConfig:
    no_executable_task_score: float
    active_scale_up_penalty: float
    empty_queue_scale_up_penalty: float
    unnecessary_scale_up_penalty: float
    invalid_scale_down_penalty: float
    low_value_reprioritize_penalty: float
    no_low_priority_work_penalty: float
    unnecessary_pause_penalty: float

    execution_urgency_weight: float

    defer_base_penalty: float
    queue_penalty_weight: float
    urgency_penalty_weight: float
    failure_relief_weight: float
    price_relief_weight: float
    price_relief_threshold: float

    blocked_task_benefit_weight: float
    scale_up_urgency_weight: float
    scale_up_failure_weight: float
    scale_up_price_weight: float

    scale_down_price_weight: float
    scale_down_base_bonus: float
    scale_down_queue_weight: float

    reprioritize_queue_weight: float
    reprioritize_urgency_weight: float
    reprioritize_failure_weight: float
    reprioritize_base_penalty: float
    reprioritize_min_queue_depth: int
    reprioritize_min_urgency: float

    pause_active_low_weight: float
    pause_queue_weight: float
    pause_failure_weight: float
    pause_price_weight: float
    pause_base_penalty: float
    pause_price_threshold: float
    pause_min_failure_pressure: float
    pause_min_queue_pressure: float

    best_task_resource_cost_weight: float
    ready_children_bonus_weight: float
    job_value_base: float
    job_priority_weight: float
    job_urgency_weight: float

    load_risk_threshold: float
    load_risk_weight: float
    failure_history_risk_weight: float
    queue_pressure_risk_weight: float

    stress_guard_failure_limit: float
    stress_guard_price_limit: float


@dataclass(frozen=True)
class ExperimentConfig:
    num_jobs: int
    max_steps: int
    uncapped_max_steps: int
    out_root: str
    run_id: Optional[str]


@dataclass(frozen=True)
class SeedsConfig:
    train: Tuple[int, ...]
    test: Tuple[int, ...]
    midterm_baseline: Tuple[int, ...]


@dataclass(frozen=True)
class CurriculumStage:
    num_jobs: int
    max_steps: int
    num_updates: int


@dataclass(frozen=True)
class CurriculumConfig:
    enabled: bool
    stages: Tuple[CurriculumStage, ...]


@dataclass(frozen=True)
class NetworkConfig:
    type: str
    hidden_sizes: Tuple[int, ...]
    activation: str


@dataclass(frozen=True)
class RLConfig:
    algorithm: str
    reward: str
    delta_discount: float  # was gamma_discount before Phase 4 notational lock
    entropy_coef: float
    learning_rate: float
    optimizer: str
    grad_clip_norm: float
    batch_size: int
    num_updates: int
    steps_per_episode_cap: int
    memoryless_termination: bool
    sequence_specific_baseline: bool
    curriculum: CurriculumConfig
    network: NetworkConfig
    # Three-pool seed partitioning per paper §4.3.6.
    training_seeds: Tuple[int, ...]
    validation_seeds: Tuple[int, ...]
    test_seeds: Tuple[int, ...]
    initialisation_seeds: Tuple[int, ...]
    eval_every_updates: int
    checkpoint_every_updates: int
    # Risk term of the per-step reward. "counter_delta" (Phase-5/6 default,
    # does not telescope to the metric) | "failed_jobs_delta" (telescopes to
    # gamma*failed_jobs so Sum r_t == episode utility). Defaulted last so
    # configs written before this field still load.
    reward_risk_mode: str = "counter_delta"


@dataclass(frozen=True)
class StrippedUtilityAgentConfig:
    disable_force_execute_guard: bool


@dataclass(frozen=True)
class StateV2QueueFeatures:
    queue_len_norm: float
    job_value_max: float


@dataclass(frozen=True)
class StateV2Forecast:
    ema_lambda: float
    initial_forecast: float


@dataclass(frozen=True)
class StateV2Config:
    use_richer_state: bool
    feature_set: str
    queue_features: StateV2QueueFeatures
    forecast: StateV2Forecast


@dataclass(frozen=True)
class SweepReferencePoint:
    name: str
    alpha: float
    beta: float
    gamma: float


@dataclass(frozen=True)
class SweepEvaluationUtility:
    alpha: float
    beta: float
    gamma: float


@dataclass(frozen=True)
class SweepConfig:
    alpha_grid: Tuple[float, ...]
    beta_grid: Tuple[float, ...]
    gamma_grid: Tuple[float, ...]
    seeds: Tuple[int, ...]
    reference_points: Tuple[SweepReferencePoint, ...]
    evaluation_utility: SweepEvaluationUtility


@dataclass(frozen=True)
class RunConfig:
    meta: Mapping[str, Any]
    experiment: ExperimentConfig
    seeds: SeedsConfig
    simulator: SimulatorConfig
    utility: UtilityWeights
    reflex_agent: ReflexAgentConfig
    utility_agent: UtilityAgentConfig
    stripped_utility_agent: StrippedUtilityAgentConfig
    state_v2: StateV2Config
    rl: RLConfig
    sweep: SweepConfig
    raw: Mapping[str, Any] = field(repr=False)

    def config_hash(self) -> str:
        """Stable sha256 over the resolved config (sorted-key YAML dump)."""
        return sha256_of_resolved(self.raw)

    def resolved_yaml(self) -> str:
        """Canonical YAML form of the resolved config."""
        return _canonical_yaml_dump(self.raw)


# ---------------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------------
def _as_tuple(value: Any) -> Tuple[Any, ...]:
    if value is None:
        return ()
    if isinstance(value, tuple):
        return value
    return tuple(value)


def _parse_action_params(raw: Mapping[str, Any]) -> ActionParams:
    return ActionParams(
        execute_ready_job=ExecuteReadyJobParams(**raw["Execute_Ready_Job"]),
        defer_job=DeferJobParams(**raw["Defer_Job"]),
        scale_up=ScaleUpParams(**raw["Scale_Up"]),
        scale_down=ScaleDownParams(**raw["Scale_Down"]),
        reprioritize_queue=ReprioritizeQueueParams(**raw["Reprioritize_Queue"]),
        pause_low_priority_job=PauseLowPriorityJobParams(**raw["Pause_LowPriority_Job"]),
    )


def _parse_stochastic_processes(raw: Mapping[str, Any]) -> StochasticProcessesConfig:
    return StochasticProcessesConfig(
        node_failure=NodeFailureConfig(**raw["node_failure"]),
        data_spike=DataSpikeConfig(**raw["data_spike"]),
        spot_price=SpotPriceConfig(**raw["spot_price"]),
    )


def _parse_cost(raw: Mapping[str, Any]) -> CostConfig:
    return CostConfig(
        cpu_weight=float(raw["cpu_weight"]),
        ram_weight=float(raw["ram_weight"]),
        action_costs={
            str(key): float(value) for key, value in dict(raw["action_costs"]).items()
        },
    )


def _parse_simulator(raw: Mapping[str, Any]) -> SimulatorConfig:
    cluster = ClusterConfig(**raw["cluster"])
    action_params = _parse_action_params(raw["action_params"])
    stochastic_processes = _parse_stochastic_processes(raw["stochastic_processes"])
    cost = _parse_cost(raw["cost"])
    state_vec = StateVectorConfig(**raw["state_vector"])

    workload_raw = dict(raw["workload"])
    workload = WorkloadConfig(
        duration_factors=_as_tuple(workload_raw.pop("duration_factors")),
        cpu_factors=_as_tuple(workload_raw.pop("cpu_factors")),
        ram_factors=_as_tuple(workload_raw.pop("ram_factors")),
        **workload_raw,
    )

    templates = []
    for template_raw in raw["dag_templates"]:
        task_specs = []
        for entry in template_raw["tasks"]:
            task_id, parents, base_duration, cpu_demand, ram_demand = entry
            task_specs.append(
                DAGTaskSpec(
                    task_id=str(task_id),
                    parents=_as_tuple(parents),
                    base_duration=int(base_duration),
                    cpu_demand=int(cpu_demand),
                    ram_demand=int(ram_demand),
                )
            )
        templates.append(
            DAGTemplateSpec(name=str(template_raw["name"]), tasks=tuple(task_specs))
        )

    return SimulatorConfig(
        actions=_as_tuple(raw["actions"]),
        cluster=cluster,
        action_params=action_params,
        stochastic_processes=stochastic_processes,
        cost=cost,
        state_vector=state_vec,
        workload=workload,
        dag_templates=tuple(templates),
    )


def _materialise_seed_pool(spec: Any) -> Tuple[int, ...]:
    """Convert a YAML seed-pool spec into a tuple of ints.

    Accepted forms:
    - list of ints: ``[100, 101, 102]``
    - range dict:   ``{kind: range, start: 300, stop: 1000}``
    """
    if isinstance(spec, list):
        return tuple(int(x) for x in spec)
    if isinstance(spec, dict):
        kind = str(spec.get("kind", "range"))
        if kind == "range":
            start = int(spec["start"])
            stop = int(spec["stop"])
            return tuple(range(start, stop))
        raise ValueError(f"unknown seed-pool kind: {kind}")
    raise ValueError(f"unsupported seed-pool spec: {spec!r}")


_RL_STRUCTURED_KEYS: Tuple[str, ...] = (
    "curriculum",
    "network",
    "training_seeds",
    "validation_seeds",
    "test_seeds",
    "initialisation_seeds",
)


def _parse_rl(raw: Mapping[str, Any]) -> RLConfig:
    stages = tuple(CurriculumStage(**stage) for stage in raw["curriculum"]["stages"])
    curriculum = CurriculumConfig(enabled=bool(raw["curriculum"]["enabled"]), stages=stages)
    network = NetworkConfig(
        type=str(raw["network"]["type"]),
        hidden_sizes=_as_tuple(raw["network"]["hidden_sizes"]),
        activation=str(raw["network"]["activation"]),
    )
    # Backwards-friendly guard rails: fail fast on pre-Phase-4 keys rather
    # than silently ignoring them.
    if "gamma_discount" in raw:
        raise ValueError(
            "rl.gamma_discount was renamed to rl.delta_discount in Phase 4 "
            "(paper §4.3 notational lock)."
        )
    if "value_loss_coef" in raw:
        raise ValueError(
            "rl.value_loss_coef was removed in Phase 4; the sequence-specific "
            "baseline is the sole variance-reduction mechanism (paper §4.3.1)."
        )
    if "training_seed_list" in raw:
        raise ValueError(
            "rl.training_seed_list was replaced by rl.training_seeds / "
            "validation_seeds / test_seeds / initialisation_seeds in Phase 4."
        )

    rl_kwargs = {
        key: raw[key]
        for key in raw
        if key not in _RL_STRUCTURED_KEYS
    }
    return RLConfig(
        curriculum=curriculum,
        network=network,
        training_seeds=_materialise_seed_pool(raw["training_seeds"]),
        validation_seeds=_materialise_seed_pool(raw["validation_seeds"]),
        test_seeds=_materialise_seed_pool(raw["test_seeds"]),
        initialisation_seeds=_as_tuple(raw["initialisation_seeds"]),
        **rl_kwargs,
    )


def _parse_utility(raw: Mapping[str, Any]) -> UtilityWeights:
    midterm_values = dict(raw.get("midterm_values", {}))
    return UtilityWeights(
        alpha=float(raw["alpha"]),
        beta=float(raw["beta"]),
        gamma=float(raw["gamma"]),
        midterm_values={k: float(v) for k, v in midterm_values.items()},
    )


def _parse_seeds(raw: Mapping[str, Any]) -> SeedsConfig:
    return SeedsConfig(
        train=_as_tuple(raw["train"]),
        test=_as_tuple(raw["test"]),
        midterm_baseline=_as_tuple(raw["midterm_baseline"]),
    )


def _parse_sweep(raw: Mapping[str, Any]) -> SweepConfig:
    reference_raw = dict(raw.get("reference_points", {}))
    reference_points = tuple(
        SweepReferencePoint(
            name=str(name),
            alpha=float(spec["alpha"]),
            beta=float(spec["beta"]),
            gamma=float(spec["gamma"]),
        )
        for name, spec in reference_raw.items()
    )
    eval_raw = raw.get("evaluation_utility", {"alpha": 1.0, "beta": 0.1, "gamma": 1.0})
    evaluation_utility = SweepEvaluationUtility(
        alpha=float(eval_raw["alpha"]),
        beta=float(eval_raw["beta"]),
        gamma=float(eval_raw["gamma"]),
    )
    return SweepConfig(
        alpha_grid=_as_tuple(raw["alpha_grid"]),
        beta_grid=_as_tuple(raw["beta_grid"]),
        gamma_grid=_as_tuple(raw["gamma_grid"]),
        seeds=_as_tuple(raw["seeds"]),
        reference_points=reference_points,
        evaluation_utility=evaluation_utility,
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def load_config(path: Optional[Path] = None) -> RunConfig:
    """Load and validate a YAML config file."""
    config_path = Path(path) if path is not None else DEFAULT_CONFIG_PATH
    if not config_path.exists():
        raise FileNotFoundError(f"config file not found: {config_path}")
    with config_path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    if not isinstance(raw, dict):
        raise ValueError("config must be a mapping at the top level")
    return build_run_config(raw)


def build_run_config(raw: Mapping[str, Any]) -> RunConfig:
    """Parse an in-memory raw mapping into a typed :class:`RunConfig`."""
    experiment = ExperimentConfig(**raw["experiment"])
    seeds = _parse_seeds(raw["seeds"])
    simulator = _parse_simulator(raw["simulator"])
    utility = _parse_utility(raw["utility"])
    reflex_agent = ReflexAgentConfig(**raw["reflex_agent"])
    utility_agent = UtilityAgentConfig(**raw["utility_agent"])
    stripped = StrippedUtilityAgentConfig(
        disable_force_execute_guard=bool(
            raw.get("stripped_utility_agent", {}).get("disable_force_execute_guard", True)
        )
    )
    state_v2_raw = dict(raw.get("state_v2", {}))
    queue_raw = dict(state_v2_raw.get("queue_features", {}))
    forecast_raw = dict(state_v2_raw.get("forecast", {}))
    state_v2 = StateV2Config(
        use_richer_state=bool(state_v2_raw.get("use_richer_state", False)),
        feature_set=str(state_v2_raw.get("feature_set", "queue_and_forecast")),
        queue_features=StateV2QueueFeatures(
            queue_len_norm=float(queue_raw.get("queue_len_norm", 100.0)),
            job_value_max=float(queue_raw.get("job_value_max", 5.0)),
        ),
        forecast=StateV2Forecast(
            ema_lambda=float(forecast_raw.get("ema_lambda", 0.3)),
            initial_forecast=float(forecast_raw.get("initial_forecast", 0.5)),
        ),
    )
    rl = _parse_rl(raw["rl"])
    sweep = _parse_sweep(raw["sweep"])

    _validate(simulator=simulator, utility=utility, rl=rl, state_v2=state_v2)

    return RunConfig(
        meta=dict(raw.get("meta", {})),
        experiment=experiment,
        seeds=seeds,
        simulator=simulator,
        utility=utility,
        reflex_agent=reflex_agent,
        utility_agent=utility_agent,
        stripped_utility_agent=stripped,
        state_v2=state_v2,
        rl=rl,
        sweep=sweep,
        raw=raw,
    )


def _validate(
    *,
    simulator: SimulatorConfig,
    utility: UtilityWeights,
    rl: Optional[RLConfig] = None,
    state_v2: Optional[StateV2Config] = None,
) -> None:
    """Cheap invariant checks so the loader fails fast on bad configs."""
    valid_modes = {
        "node_failure": {"per_step_single_victim", "per_node_bernoulli", "load_cascade"},
        "data_spike": {"additive_bump", "multiplicative_10x"},
        "spot_price": {"bounded_random_walk"},
    }
    sp = simulator.stochastic_processes
    if sp.node_failure.mode not in valid_modes["node_failure"]:
        raise ValueError(f"unknown node_failure mode: {sp.node_failure.mode}")
    if sp.data_spike.mode not in valid_modes["data_spike"]:
        raise ValueError(f"unknown data_spike mode: {sp.data_spike.mode}")
    if sp.spot_price.mode not in valid_modes["spot_price"]:
        raise ValueError(f"unknown spot_price mode: {sp.spot_price.mode}")

    if simulator.action_params.execute_ready_job.selection_policy not in {
        "value",
        "value_times_priority",
    }:
        raise ValueError(
            "Execute_Ready_Job.selection_policy must be 'value' or 'value_times_priority'"
        )

    if simulator.workload.value_distribution not in {"uniform", "heavy_tail"}:
        raise ValueError(
            f"unknown workload.value_distribution: {simulator.workload.value_distribution}"
        )

    missing_action_costs = set(simulator.actions) - set(simulator.cost.action_costs)
    if missing_action_costs:
        raise ValueError(f"cost.action_costs missing entries for: {missing_action_costs}")

    for name in (utility.alpha, utility.beta, utility.gamma):
        if not isinstance(name, float):
            raise ValueError("utility weights must be floats")

    if rl is not None:
        # Pairwise disjointness of the three RL seed pools is a core
        # correctness invariant — any overlap would leak information from
        # training into validation or the held-out test set.
        train_set = set(rl.training_seeds)
        val_set = set(rl.validation_seeds)
        test_set = set(rl.test_seeds)
        if train_set & val_set:
            raise ValueError("rl.training_seeds overlaps rl.validation_seeds")
        if train_set & test_set:
            raise ValueError("rl.training_seeds overlaps rl.test_seeds")
        if val_set & test_set:
            raise ValueError("rl.validation_seeds overlaps rl.test_seeds")
        if rl.reward_risk_mode not in {"counter_delta", "failed_jobs_delta"}:
            raise ValueError(
                f"unknown rl.reward_risk_mode: {rl.reward_risk_mode} "
                "(expected 'counter_delta' or 'failed_jobs_delta')"
            )

    if state_v2 is not None:
        valid_feature_sets = {"queue_and_forecast", "queue_and_forecast_and_gnn"}
        if state_v2.feature_set not in valid_feature_sets:
            raise ValueError(f"unknown state_v2.feature_set: {state_v2.feature_set}")
        # V2 (GNN) is gated on V1 results per the Phase-6 brief §G; fail
        # loudly if someone tries to enable it in this phase.
        if state_v2.use_richer_state and state_v2.feature_set == "queue_and_forecast_and_gnn":
            raise NotImplementedError(
                "state_v2.feature_set = queue_and_forecast_and_gnn (V2) is "
                "gated on V1 outcomes; implement V2 in its own phase."
            )
        if not 0.0 < state_v2.forecast.ema_lambda <= 1.0:
            raise ValueError(
                f"state_v2.forecast.ema_lambda must be in (0, 1], got "
                f"{state_v2.forecast.ema_lambda}"
            )


def _canonical_yaml_dump(raw: Mapping[str, Any]) -> str:
    """Deterministic YAML dump used for hashing."""
    buffer = io.StringIO()
    yaml.safe_dump(
        dict(raw),
        buffer,
        sort_keys=True,
        default_flow_style=False,
        allow_unicode=True,
    )
    return buffer.getvalue()


def sha256_of_resolved(raw: Mapping[str, Any]) -> str:
    canonical = _canonical_yaml_dump(raw)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def override_utility_weights(
    cfg: RunConfig,
    alpha: Optional[float] = None,
    beta: Optional[float] = None,
    gamma: Optional[float] = None,
) -> RunConfig:
    """Return a copy of ``cfg`` with its utility weights replaced."""
    new_raw = _deepcopy_mapping(cfg.raw)
    u = new_raw["utility"]
    if alpha is not None:
        u["alpha"] = float(alpha)
    if beta is not None:
        u["beta"] = float(beta)
    if gamma is not None:
        u["gamma"] = float(gamma)
    return build_run_config(new_raw)


def _deepcopy_mapping(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _deepcopy_mapping(child) for key, child in value.items()}
    if isinstance(value, list):
        return [_deepcopy_mapping(item) for item in value]
    return value


__all__ = [
    "ActionParams",
    "ClusterConfig",
    "CostConfig",
    "CurriculumConfig",
    "CurriculumStage",
    "DAGTaskSpec",
    "DAGTemplateSpec",
    "DataSpikeConfig",
    "DEFAULT_CONFIG_PATH",
    "DeferJobParams",
    "ExecuteReadyJobParams",
    "ExperimentConfig",
    "NetworkConfig",
    "NodeFailureConfig",
    "PauseLowPriorityJobParams",
    "REPO_ROOT",
    "RLConfig",
    "ReflexAgentConfig",
    "ReprioritizeQueueParams",
    "RunConfig",
    "ScaleDownParams",
    "ScaleUpParams",
    "SeedsConfig",
    "SimulatorConfig",
    "SpotPriceConfig",
    "StateV2Config",
    "StateV2Forecast",
    "StateV2QueueFeatures",
    "StateVectorConfig",
    "StochasticProcessesConfig",
    "StrippedUtilityAgentConfig",
    "SweepConfig",
    "SweepEvaluationUtility",
    "SweepReferencePoint",
    "UtilityAgentConfig",
    "UtilityWeights",
    "WorkloadConfig",
    "build_run_config",
    "load_config",
    "override_utility_weights",
    "sha256_of_resolved",
]
