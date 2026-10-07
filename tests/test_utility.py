from __future__ import annotations

"""Episode-level utility on a hand-computed fixture.

Builds a tiny ``EpisodeState`` by hand, pins ``alpha/beta/gamma``, and checks
that :func:`summarize_episode` returns the analytically expected utility.

The contract under test is ``U = alpha*Value - beta*Cost - gamma*Risk`` with:
- Value = sum(value of completed jobs)
- Cost  = total_compute_cost accumulated over the episode
- Risk  = count of failed jobs
"""

from src.config import load_config, override_utility_weights
from src.metrics import (
    EpisodeMetrics,
    estimate_step_cost,
    summarize_episode,
)
from src.sim_environment import (
    ClusterState,
    EpisodeState,
    JobInstance,
    TaskInstance,
    TASK_STATE_COMPLETED,
    TASK_STATE_FAILED,
)


def _make_job(cfg, job_id: int, value: float, priority: float, deadline: int, status: str):
    task = TaskInstance(
        job_id=job_id,
        task_id="t",
        parents=(),
        remaining_time=0,
        cpu_demand=1,
        ram_demand=1,
        state=status,
    )
    job = JobInstance(
        job_id=job_id,
        template_name="chain",
        priority=priority,
        deadline_steps=deadline,
        value=value,
        tasks={"t": task},
    )
    job.update_status()
    return job


def _hand_state(cfg) -> EpisodeState:
    sim = cfg.simulator.cluster
    cluster = ClusterState(
        cpu_capacity=sim.base_cpu_capacity,
        ram_capacity=sim.base_ram_capacity,
        spot_price=0.5,
    )
    jobs = [
        _make_job(cfg, 0, value=10.0, priority=0.8, deadline=50, status=TASK_STATE_COMPLETED),
        _make_job(cfg, 1, value=4.0, priority=0.5, deadline=80, status=TASK_STATE_COMPLETED),
        _make_job(cfg, 2, value=3.0, priority=0.3, deadline=90, status=TASK_STATE_FAILED),
    ]
    state = EpisodeState(step=37, jobs=jobs, cluster=cluster, cfg=cfg)
    return state


def test_utility_hand_computed_phase1_weights():
    """Phase-1 weights (alpha=1.0, beta=0.1, gamma=1.0) on the fixture.

    Fixture: completed jobs have value 10 + 4 = 14, one failed job,
    total compute cost forced to 100.
    Expected U = 1.0 * 14 - 0.1 * 100 - 1.0 * 1 = 14 - 10 - 1 = 3.0.
    """
    cfg = load_config()
    state = _hand_state(cfg)
    metrics: EpisodeMetrics = summarize_episode(
        cfg=cfg,
        seed=0,
        agent_name="Hand",
        state=state,
        num_jobs=3,
        total_compute_cost=100.0,
        hit_step_budget=False,
    )
    assert metrics.total_completed_value == 14.0
    assert metrics.total_job_value == 17.0
    assert metrics.failed_jobs == 1
    assert metrics.completed_jobs == 2
    assert metrics.completion_rate == 2 / 3
    assert abs(metrics.total_utility - 3.0) < 1e-9
    assert abs(metrics.value_weighted_completion_rate - (14.0 / 17.0)) < 1e-4


def test_utility_hand_computed_midterm_weights():
    """Midterm weights (alpha=1.0, beta=0.4, gamma=0.8) on the same fixture.

    Expected U = 1.0 * 14 - 0.4 * 100 - 0.8 * 1 = -26.8.
    """
    midterm = override_utility_weights(load_config(), alpha=1.0, beta=0.4, gamma=0.8)
    state = _hand_state(midterm)
    metrics = summarize_episode(
        cfg=midterm,
        seed=0,
        agent_name="Hand",
        state=state,
        num_jobs=3,
        total_compute_cost=100.0,
        hit_step_budget=False,
    )
    assert abs(metrics.total_utility - (-26.8)) < 1e-9


def test_utility_weights_are_applied():
    base = load_config()
    boosted = override_utility_weights(base, alpha=2.0, beta=0.2, gamma=1.0)
    state_base = _hand_state(base)
    state_boosted = _hand_state(boosted)
    m_base = summarize_episode(
        cfg=base,
        seed=0,
        agent_name="Hand",
        state=state_base,
        num_jobs=3,
        total_compute_cost=100.0,
        hit_step_budget=False,
    )
    m_boosted = summarize_episode(
        cfg=boosted,
        seed=0,
        agent_name="Hand",
        state=state_boosted,
        num_jobs=3,
        total_compute_cost=100.0,
        hit_step_budget=False,
    )
    # U_base = 1.0*14 - 0.1*100 - 1.0*1 = 3.0
    # U_boosted = 2*14 - 0.2*100 - 1*1 = 7.0
    assert abs(m_base.total_utility - 3.0) < 1e-9
    assert abs(m_boosted.total_utility - 7.0) < 1e-9


def test_estimate_step_cost_uses_config_weights():
    cfg = load_config()
    state = _hand_state(cfg)
    state.cluster.spot_price = 0.4
    # Inject two fake running tasks via direct state manipulation
    from src.sim_environment import RunningTask

    state.running_tasks["jobX:t"] = RunningTask(full_id="jobX:t", cpu_demand=5, ram_demand=3, remaining_time=2)
    # Expected cost at this snapshot: 0.4 * (0.6 * 5 + 0.4 * 3) = 0.4 * (3 + 1.2) = 1.68
    assert abs(estimate_step_cost(state) - 1.68) < 1e-9
