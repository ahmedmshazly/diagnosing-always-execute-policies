from __future__ import annotations

"""Non-learning Utility-Based baseline.

The agent scores every legal action in the shared action space and picks the
maximum. Every scalar weight used in scoring lives in
``cfg.utility_agent`` (``UtilityAgentConfig``), and the episode-level utility
weights ``alpha/beta/gamma`` live in ``cfg.utility``. No magic numbers remain
in this file.
"""

from typing import Dict, List, Optional, Tuple

from .config import RunConfig, UtilityAgentConfig, UtilityWeights, load_config
from .metrics import EpisodeMetrics
from .runner import run_episode, run_many_episodes
from .sim_environment import (
    ACTIONS,
    EpisodeState,
    TaskInstance,
)


class UtilityBasedAgent:
    name = "Utility-Based Agent (Non-Learning Baseline)"

    def __init__(
        self,
        cfg: RunConfig,
        *,
        force_execute_guard_enabled: bool = True,
    ) -> None:
        self.cfg = cfg
        self.u: UtilityWeights = cfg.utility
        self.w: UtilityAgentConfig = cfg.utility_agent
        self.cluster_cfg = cfg.simulator.cluster
        self.cost_cfg = cfg.simulator.cost
        # When False this collapses the agent into the stripped variant
        # (paper §4.3.7, §5.6). The rest of the scoring rule is unchanged.
        self.force_execute_guard_enabled = bool(force_execute_guard_enabled)

    # ---- policy --------------------------------------------------------------
    def choose_action(self, state: EpisodeState) -> str:
        scored: List[Tuple[str, float]] = [
            (action, self.score_action(state, action)) for action in ACTIONS
        ]
        score_by_action = {action: score for action, score in scored}
        scored.sort(key=lambda item: item[1], reverse=True)

        if self.force_execute_guard_enabled and self._should_force_execution(
            state, score_by_action
        ):
            top_action = scored[0][0]
            if top_action in {"Reprioritize_Queue", "Pause_LowPriority_Job", "Defer_Job"}:
                return "Execute_Ready_Job"

        return scored[0][0]

    def score_action(self, state: EpisodeState, action: str) -> float:
        if action == "Execute_Ready_Job":
            return self._score_execute(state)
        if action == "Defer_Job":
            return self._score_defer(state)
        if action == "Scale_Up":
            return self._score_scale_up(state)
        if action == "Scale_Down":
            return self._score_scale_down(state)
        if action == "Reprioritize_Queue":
            return self._score_reprioritize(state)
        if action == "Pause_LowPriority_Job":
            return self._score_pause_low_priority(state)
        raise ValueError(f"Unknown action: {action}")

    # ---- per-action scoring --------------------------------------------------
    def _score_execute(self, state: EpisodeState) -> float:
        task = self._best_ready_task(state)
        if task is None:
            return self.w.no_executable_task_score
        job = state.jobs[task.job_id]
        value_term = self._effective_job_value(state, task)
        resource_cost = self._resource_cost(state, task)
        risk_term = self._launch_risk(state, task)
        urgency_bonus = self.w.execution_urgency_weight * self._job_urgency(job.deadline_steps)
        return (
            (self.u.alpha * value_term)
            + urgency_bonus
            - (self.u.beta * resource_cost)
            - (self.u.gamma * risk_term)
        )

    def _score_defer(self, state: EpisodeState) -> float:
        sv = state.state_vector()
        queue_penalty = self.w.queue_penalty_weight * sv["Queue_Depth"]
        urgency_penalty = self.w.urgency_penalty_weight * sv["Deadline_Urgency"]
        failure_relief = self.w.failure_relief_weight * sv["Recent_Failures"]
        price_relief = self.w.price_relief_weight * max(
            state.cluster.spot_price - self.w.price_relief_threshold, 0.0
        )
        return (
            self.w.defer_base_penalty
            - queue_penalty
            - urgency_penalty
            + failure_relief
            + price_relief
        )

    def _score_scale_up(self, state: EpisodeState) -> float:
        if state.cluster.scale_boost_remaining > 0:
            return self.w.active_scale_up_penalty
        queue_depth = state.queue_depth()
        if queue_depth == 0:
            return self.w.empty_queue_scale_up_penalty

        ready_tasks = state.ready_tasks()
        blocked = sum(
            1
            for task in ready_tasks
            if task.cpu_demand > state.available_cpu() or task.ram_demand > state.available_ram()
        )
        if blocked == 0:
            return self.w.unnecessary_scale_up_penalty

        sv = state.state_vector()
        benefit = (
            (self.w.blocked_task_benefit_weight * blocked)
            + (self.w.scale_up_urgency_weight * sv["Deadline_Urgency"])
            + queue_depth
            - (self.w.scale_up_failure_weight * sv["Recent_Failures"])
        )
        price_penalty = self.w.scale_up_price_weight * state.cluster.spot_price
        return benefit - price_penalty

    def _score_scale_down(self, state: EpisodeState) -> float:
        idle = state.cpu_in_use() == 0 and state.ram_in_use() == 0
        above_baseline = (
            state.cluster.cpu_capacity > self.cluster_cfg.base_cpu_capacity
            or state.cluster.ram_capacity > self.cluster_cfg.base_ram_capacity
        )
        if not idle or not above_baseline:
            return self.w.invalid_scale_down_penalty
        return (
            (self.w.scale_down_price_weight * state.cluster.spot_price)
            + self.w.scale_down_base_bonus
            - (self.w.scale_down_queue_weight * state.queue_depth())
        )

    def _score_reprioritize(self, state: EpisodeState) -> float:
        queue_depth_raw = state.queue_depth()
        sv = state.state_vector()
        if (
            queue_depth_raw < self.w.reprioritize_min_queue_depth
            or sv["Deadline_Urgency"] < self.w.reprioritize_min_urgency
        ):
            return self.w.low_value_reprioritize_penalty
        return (
            (self.w.reprioritize_queue_weight * sv["Queue_Depth"])
            + (self.w.reprioritize_urgency_weight * sv["Deadline_Urgency"])
            + (self.w.reprioritize_failure_weight * sv["Recent_Failures"])
            - self.w.reprioritize_base_penalty
        )

    def _score_pause_low_priority(self, state: EpisodeState) -> float:
        active_low = sum(
            1
            for job in state.jobs
            if not job.completed
            and not job.failed
            and job.priority < self.cfg.simulator.action_params.pause_low_priority_job.priority_threshold
        )
        if active_low == 0:
            return self.w.no_low_priority_work_penalty

        sv = state.state_vector()
        price_pressure = max(state.cluster.spot_price - self.w.pause_price_threshold, 0.0)
        if (
            sv["Recent_Failures"] < self.w.pause_min_failure_pressure
            and price_pressure == 0.0
            and sv["Queue_Depth"] < self.w.pause_min_queue_pressure
        ):
            return self.w.unnecessary_pause_penalty

        return (
            (self.w.pause_active_low_weight * active_low)
            + (self.w.pause_queue_weight * sv["Queue_Depth"])
            + (self.w.pause_failure_weight * sv["Recent_Failures"])
            + (self.w.pause_price_weight * price_pressure)
            - self.w.pause_base_penalty
        )

    # ---- shared helpers ------------------------------------------------------
    def _best_ready_task(self, state: EpisodeState) -> Optional[TaskInstance]:
        candidates: List[Tuple[float, TaskInstance]] = []
        available_cpu = state.available_cpu()
        available_ram = state.available_ram()
        for task in state.ready_tasks():
            if task.cpu_demand > available_cpu or task.ram_demand > available_ram:
                continue
            score = self._effective_job_value(state, task) - (
                self.w.best_task_resource_cost_weight * self._resource_cost(state, task)
            )
            candidates.append((score, task))
        if not candidates:
            return None
        candidates.sort(key=lambda item: item[0], reverse=True)
        return candidates[0][1]

    def _effective_job_value(self, state: EpisodeState, task: TaskInstance) -> float:
        job = state.jobs[task.job_id]
        urgency = self._job_urgency(job.deadline_steps)
        ready_children_bonus = self.w.ready_children_bonus_weight * len(
            [
                child
                for child in job.tasks.values()
                if task.task_id in child.parents and child.state in {"waiting", "ready"}
            ]
        )
        return job.value * (
            self.w.job_value_base
            + (self.w.job_priority_weight * job.priority)
            + (self.w.job_urgency_weight * urgency)
            + ready_children_bonus
        )

    def _job_urgency(self, deadline_steps: int) -> float:
        max_steps = max(self.cfg.experiment.max_steps, 1)
        return max(0.0, 1.0 - (deadline_steps / max_steps))

    def _resource_cost(self, state: EpisodeState, task: TaskInstance) -> float:
        return state.cluster.spot_price * (
            (self.cost_cfg.cpu_weight * task.cpu_demand)
            + (self.cost_cfg.ram_weight * task.ram_demand)
        )

    def _launch_risk(self, state: EpisodeState, task: TaskInstance) -> float:
        projected_cpu_load = (state.cpu_in_use() + task.cpu_demand) / max(
            state.cluster.cpu_capacity, 1
        )
        projected_ram_load = (state.ram_in_use() + task.ram_demand) / max(
            state.cluster.ram_capacity, 1
        )
        load_risk = (
            max(projected_cpu_load - self.w.load_risk_threshold, 0.0) * self.w.load_risk_weight
            + max(projected_ram_load - self.w.load_risk_threshold, 0.0) * self.w.load_risk_weight
        )
        sv = state.state_vector()
        failure_history_risk = self.w.failure_history_risk_weight * sv["Recent_Failures"]
        queue_pressure_risk = self.w.queue_pressure_risk_weight * sv["Queue_Depth"]
        return load_risk + failure_history_risk + queue_pressure_risk

    def _should_force_execution(
        self, state: EpisodeState, score_by_action: Dict[str, float]
    ) -> bool:
        return (
            self._best_ready_task(state) is not None
            and score_by_action["Execute_Ready_Job"] > 0.0
            and state.state_vector()["Recent_Failures"] < self.w.stress_guard_failure_limit
            and state.cluster.spot_price < self.w.stress_guard_price_limit
        )


def build_utility_agent(cfg: RunConfig) -> UtilityBasedAgent:
    return UtilityBasedAgent(cfg)


def run_utility_episode(cfg: RunConfig, seed: int) -> EpisodeMetrics:
    return run_episode(cfg=cfg, agent_factory=build_utility_agent, seed=seed)


def run_many_utility_episodes(cfg: RunConfig, seeds):
    return run_many_episodes(cfg=cfg, agent_factory=build_utility_agent, seeds=seeds)


class StrippedUtilityBasedAgent(UtilityBasedAgent):
    """Utility-Based agent with the force-execute guard bypassed.

    Identical scoring rule to :class:`UtilityBasedAgent`; the only change is
    that :meth:`choose_action` does not re-add Execute_Ready_Job when the
    guard would have fired. See paper §4.3.7 and §5.6.
    """

    name = "Stripped Utility-Based Agent"

    def __init__(self, cfg: RunConfig) -> None:
        super().__init__(cfg, force_execute_guard_enabled=False)


def build_stripped_utility_agent(cfg: RunConfig) -> StrippedUtilityBasedAgent:
    return StrippedUtilityBasedAgent(cfg)


def run_stripped_utility_episode(cfg: RunConfig, seed: int) -> EpisodeMetrics:
    return run_episode(cfg=cfg, agent_factory=build_stripped_utility_agent, seed=seed)


def run_many_stripped_utility_episodes(cfg: RunConfig, seeds):
    return run_many_episodes(
        cfg=cfg, agent_factory=build_stripped_utility_agent, seeds=seeds
    )


def _demo() -> None:
    cfg = load_config()
    metrics = run_utility_episode(cfg=cfg, seed=7)
    print("Utility-Based Agent episode summary")
    for key, value in metrics.as_dict().items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    _demo()
