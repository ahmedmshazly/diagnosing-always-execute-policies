from __future__ import annotations

"""Phase-6 V1 richer-state tests (§C).

Covers the three guarantees in the brief:
- ``StateVector`` returns 8 or 14 dims depending on the flag.
- All six new V1 features live in ``[0, 1]`` during realistic episodes.
- The zero-active-jobs edge case returns ``0.0`` for all five queue
  features without raising or producing NaN.
"""

import math

import numpy as np
import pytest

from src.config import build_run_config, load_config
from src.rl.env import (
    PHASE5_STATE_FIELD_ORDER,
    PHASE6_V1_STATE_FIELD_ORDER,
    observation_dim,
    state_field_order,
)
from src.sim_environment import (
    WorkloadGenerator,
    advance_one_step,
    make_episode_rngs,
)


def _deep_copy(value):
    if isinstance(value, dict):
        return {k: _deep_copy(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_deep_copy(v) for v in value]
    return value


def _richer_cfg():
    raw = _deep_copy(load_config().raw)
    raw["state_v2"] = {**raw["state_v2"], "use_richer_state": True}
    return build_run_config(raw)


def _fresh_state(cfg, seed=100, num_jobs=20):
    workload_rng, _ = make_episode_rngs(seed)
    return WorkloadGenerator(
        rng=workload_rng, cfg=cfg, num_jobs=num_jobs, seed_label=seed,
    ).generate_episode()


# ---------------------------------------------------------------------------
# Dimensionality
# ---------------------------------------------------------------------------
def test_phase5_config_keeps_8_dim_observation(default_config):
    assert default_config.state_v2.use_richer_state is False
    assert observation_dim(default_config) == 8
    assert state_field_order(default_config) == PHASE5_STATE_FIELD_ORDER


def test_richer_config_reports_14_dim_observation():
    cfg = _richer_cfg()
    assert cfg.state_v2.use_richer_state is True
    assert observation_dim(cfg) == 14
    order = state_field_order(cfg)
    assert order == PHASE6_V1_STATE_FIELD_ORDER
    # Phase-5 block comes first, unchanged.
    assert order[:8] == PHASE5_STATE_FIELD_ORDER
    # V1 block is exactly the six named fields, in this order.
    assert order[8:] == (
        "queue_len_abs_norm",
        "mean_remaining_work",
        "max_deadline_urgency",
        "mean_job_value",
        "max_job_value",
        "spot_price_forecast",
    )


# ---------------------------------------------------------------------------
# Ranges
# ---------------------------------------------------------------------------
def test_v1_features_in_unit_interval_on_fresh_episode():
    cfg = _richer_cfg()
    sv = _fresh_state(cfg).state_vector()
    for name in (
        "queue_len_abs_norm",
        "mean_remaining_work",
        "max_deadline_urgency",
        "mean_job_value",
        "max_job_value",
    ):
        value = getattr(sv, name)
        assert 0.0 <= value <= 1.0, f"{name}={value} out of range"
    # Spot-price forecast lives in the spot-price range.
    sp = cfg.simulator.stochastic_processes.spot_price
    assert sp.price_min <= sv.spot_price_forecast <= sp.price_max


def test_v1_features_stay_in_range_across_a_random_trajectory():
    cfg = _richer_cfg()
    state = _fresh_state(cfg, seed=13, num_jobs=20)
    _, event_rng = make_episode_rngs(13)
    from src.sim_environment import ACTIONS

    rng = np.random.default_rng(0)
    for _ in range(150):
        action = ACTIONS[int(rng.integers(0, len(ACTIONS)))]
        advance_one_step(state, event_rng, action)
        sv = state.state_vector()
        for name in (
            "queue_len_abs_norm",
            "mean_remaining_work",
            "max_deadline_urgency",
            "mean_job_value",
            "max_job_value",
        ):
            value = getattr(sv, name)
            assert not math.isnan(value), f"{name} became NaN at step {state.step}"
            assert 0.0 <= value <= 1.0, f"{name}={value} at step {state.step}"


# ---------------------------------------------------------------------------
# Zero-active-jobs edge case
# ---------------------------------------------------------------------------
def test_v1_features_collapse_to_zero_when_no_active_jobs():
    cfg = _richer_cfg()
    state = _fresh_state(cfg, seed=2, num_jobs=3)
    # Mark every job terminal.
    for job in state.jobs:
        job.completed = True
    sv = state.state_vector()
    assert sv.queue_len_abs_norm == 0.0
    assert sv.mean_remaining_work == 0.0
    assert sv.max_deadline_urgency == 0.0
    assert sv.mean_job_value == 0.0
    assert sv.max_job_value == 0.0
    # The spot_price_forecast EMA is independent of job state and must
    # still be a finite scalar in the price range.
    sp = cfg.simulator.stochastic_processes.spot_price
    assert not math.isnan(sv.spot_price_forecast)
    assert sp.price_min <= sv.spot_price_forecast <= sp.price_max


def test_v1_feature_definition_includes_docstring_for_each_new_field():
    """Documentation is part of the contract — the Phase-6 brief’s §6.

    Every V1 field must be named in the StateVector docstring so a
    reviewer reading the class can map code to paper §6.2.1.
    """
    from src.state import StateVector

    doc = StateVector.__doc__ or ""
    for name in (
        "queue_len_abs_norm",
        "mean_remaining_work",
        "max_deadline_urgency",
        "mean_job_value",
        "max_job_value",
        "spot_price_forecast",
    ):
        assert f"``{name}``" in doc, f"{name} missing from StateVector docstring"


# ---------------------------------------------------------------------------
# Config validation
# ---------------------------------------------------------------------------
def test_v2_mode_raises_until_gnn_phase_lands():
    raw = _deep_copy(load_config().raw)
    raw["state_v2"] = {
        **raw["state_v2"],
        "use_richer_state": True,
        "feature_set": "queue_and_forecast_and_gnn",
    }
    with pytest.raises(NotImplementedError):
        build_run_config(raw)


def test_ema_lambda_out_of_range_fails_fast():
    raw = _deep_copy(load_config().raw)
    raw["state_v2"] = {
        **raw["state_v2"],
        "forecast": {**raw["state_v2"]["forecast"], "ema_lambda": 1.5},
    }
    with pytest.raises(ValueError):
        build_run_config(raw)
