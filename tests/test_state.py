from __future__ import annotations

"""Tests for :class:`src.state.StateVector`.

The StateVector is the observation contract: types, ranges, and legacy
compatibility are pinned here so a downstream agent (including the RL agent)
can rely on the invariants.
"""

import pytest

from src.config import load_config
from src.sim_environment import WorkloadGenerator, make_episode_rngs
from src.state import StateVector


def _fresh_state(cfg, seed=9, num_jobs=20):
    workload_rng, _ = make_episode_rngs(seed)
    generator = WorkloadGenerator(rng=workload_rng, cfg=cfg, num_jobs=num_jobs, seed_label=seed)
    return generator.generate_episode()


def test_state_vector_has_all_phase5_fields():
    """All eight Phase-5 fields must still exist on the dataclass.

    After Phase 6 V1 the dataclass has 14 fields total; we assert the 8
    Phase-5 fields are all present rather than requiring exactly 8, so
    adding new Phase-6 fields is a non-breaking change.
    """
    phase5_fields = {
        "cpu_load",
        "ram_available",
        "queue_depth",
        "spot_price",
        "dag_ready_nodes",
        "job_priority",
        "deadline_urgency",
        "recent_failures",
    }
    cfg = load_config()
    sv = _fresh_state(cfg).state_vector()
    assert isinstance(sv, StateVector)
    assert phase5_fields.issubset(set(sv.__dict__.keys()))


def test_state_vector_values_are_in_range():
    cfg = load_config()
    sv = _fresh_state(cfg).state_vector()
    assert 0.0 <= sv.cpu_load <= 1.0
    assert 0.0 <= sv.ram_available <= 1.0
    assert 0.0 <= sv.queue_depth <= 1.0
    sp = cfg.simulator.stochastic_processes.spot_price
    assert sp.price_min <= sv.spot_price <= sp.price_max
    assert 0.0 <= sv.dag_ready_nodes <= 1.0
    # job_priority is 0.0 when inactive; otherwise in [priority_low, priority_high].
    workload = cfg.simulator.workload
    assert sv.job_priority == 0.0 or workload.priority_low <= sv.job_priority <= workload.priority_high
    assert 0.0 <= sv.deadline_urgency <= 1.0
    assert 0.0 <= sv.recent_failures <= 1.0


def test_state_vector_as_dict_uses_legacy_keys():
    cfg = load_config()
    sv = _fresh_state(cfg).state_vector()
    plain = sv.as_dict()
    # The midterm's Title_Case keys are a subset of the full dict in
    # Phase 6; assert presence rather than equality so V1 fields don't
    # break this check.
    phase5_keys = {
        "CPU_Load",
        "RAM_Available",
        "Queue_Depth",
        "Spot_Price",
        "DAG_Ready_Nodes",
        "Job_Priority",
        "Deadline_Urgency",
        "Recent_Failures",
    }
    assert phase5_keys.issubset(set(plain.keys()))
    assert plain["Spot_Price"] == sv.spot_price
    assert plain["Recent_Failures"] == sv.recent_failures


def test_state_vector_subscript_supports_both_name_styles():
    cfg = load_config()
    sv = _fresh_state(cfg).state_vector()
    assert sv["Recent_Failures"] == sv.recent_failures
    assert sv["recent_failures"] == sv.recent_failures
    with pytest.raises(KeyError):
        _ = sv["not_a_real_field"]


def test_state_vector_zero_active_jobs_sets_priority_and_urgency_to_zero():
    """When every job is terminal, priority and urgency collapse to 0.0."""
    cfg = load_config()
    state = _fresh_state(cfg, seed=1, num_jobs=2)
    for job in state.jobs:
        job.completed = True
    sv = state.state_vector()
    assert sv.job_priority == 0.0
    assert sv.deadline_urgency == 0.0


def test_state_vector_docstring_lists_every_field():
    """Every StateVector field must be documented in the class docstring."""
    docstring = StateVector.__doc__ or ""
    for field in StateVector.__dataclass_fields__:
        assert f"``{field}``" in docstring, f"{field} missing from StateVector docstring"
