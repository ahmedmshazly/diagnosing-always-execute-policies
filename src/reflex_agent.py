from __future__ import annotations

"""Reflex baseline (fixed-rule, no learning, no utility).

The reflex policy is intentionally narrow: execute if any ready task fits,
scale down if idle and the cluster is oversized while prices are high, scale
up once under queue pressure at moderate prices, otherwise defer. Every
threshold comes from :class:`ReflexAgentConfig` in ``config/default.yaml``.
"""

from .config import ReflexAgentConfig, RunConfig, load_config
from .metrics import EpisodeMetrics
from .runner import run_episode, run_many_episodes
from .sim_environment import EpisodeState


class ReflexAgent:
    name = "Reflex Agent"

    def __init__(self, cfg: RunConfig) -> None:
        self.cfg = cfg
        self.policy: ReflexAgentConfig = cfg.reflex_agent
        self.cluster_cfg = cfg.simulator.cluster

    def choose_action(self, state: EpisodeState) -> str:
        ready_tasks = state.ready_tasks()

        if not ready_tasks and self._should_scale_down_idle_cluster(state):
            return "Scale_Down"

        if ready_tasks:
            if self._can_launch_any_ready_task(state):
                return "Execute_Ready_Job"
            if self._should_scale_up_for_queue_pressure(state):
                return "Scale_Up"

        return "Defer_Job"

    @staticmethod
    def _can_launch_any_ready_task(state: EpisodeState) -> bool:
        available_cpu = state.available_cpu()
        available_ram = state.available_ram()
        for task in state.ready_tasks():
            if task.cpu_demand <= available_cpu and task.ram_demand <= available_ram:
                return True
        return False

    def _should_scale_down_idle_cluster(self, state: EpisodeState) -> bool:
        return (
            state.cluster.spot_price >= self.policy.scale_down_spot_price_threshold
            and (
                state.cluster.cpu_capacity > self.cluster_cfg.base_cpu_capacity
                or state.cluster.ram_capacity > self.cluster_cfg.base_ram_capacity
            )
        )

    def _should_scale_up_for_queue_pressure(self, state: EpisodeState) -> bool:
        return (
            state.queue_depth() >= self.policy.scale_up_queue_depth_threshold
            and state.cluster.spot_price <= self.policy.scale_up_spot_price_threshold
            and state.cluster.scale_boost_remaining == 0
        )


def build_reflex_agent(cfg: RunConfig) -> ReflexAgent:
    return ReflexAgent(cfg)


def run_reflex_episode(cfg: RunConfig, seed: int) -> EpisodeMetrics:
    return run_episode(cfg=cfg, agent_factory=build_reflex_agent, seed=seed)


def run_many_reflex_episodes(cfg: RunConfig, seeds):
    return run_many_episodes(cfg=cfg, agent_factory=build_reflex_agent, seeds=seeds)


def _demo() -> None:
    cfg = load_config()
    metrics = run_reflex_episode(cfg=cfg, seed=7)
    print("Reflex Agent episode summary")
    for key, value in metrics.as_dict().items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    _demo()
