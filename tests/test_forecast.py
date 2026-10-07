from __future__ import annotations

"""Phase-6 V1 spot-price EMA forecast tests.

The brief requires a regression-guarded hand-computed EMA: drive the
spot-price walk with a deterministic trajectory and assert the forecast
matches the reference sequence exactly.
"""

import numpy as np

from src.config import build_run_config, load_config
from src.sim_environment import (
    WorkloadGenerator,
    _apply_spot_price_walk,
    make_episode_rngs,
)


def _deep_copy(value):
    if isinstance(value, dict):
        return {k: _deep_copy(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_deep_copy(v) for v in value]
    return value


def _richer_cfg(ema_lambda=0.3, initial_forecast=0.5):
    raw = _deep_copy(load_config().raw)
    raw["state_v2"] = {
        **raw["state_v2"],
        "use_richer_state": True,
        "forecast": {
            "ema_lambda": ema_lambda,
            "initial_forecast": initial_forecast,
        },
    }
    return build_run_config(raw)


def test_forecast_initial_value_matches_config():
    cfg = _richer_cfg(ema_lambda=0.3, initial_forecast=0.42)
    workload_rng, _ = make_episode_rngs(17)
    state = WorkloadGenerator(
        rng=workload_rng, cfg=cfg, num_jobs=4, seed_label=17,
    ).generate_episode()
    assert state.cluster.spot_price_forecast == 0.42


class _ScriptedRng:
    """Minimal replacement for np.random.Generator — only uniform() is used
    by the spot-price walk function we're exercising here.
    """

    def __init__(self, walks):
        self._walks = list(walks)
        self._idx = 0

    def uniform(self, low, high):
        value = self._walks[self._idx]
        self._idx += 1
        assert low <= value <= high, (low, value, high)
        return value


def test_forecast_ema_matches_hand_computation_on_deterministic_walk():
    cfg = _richer_cfg(ema_lambda=0.25, initial_forecast=0.50)
    workload_rng, _ = make_episode_rngs(17)
    state = WorkloadGenerator(
        rng=workload_rng, cfg=cfg, num_jobs=4, seed_label=17,
    ).generate_episode()
    # Pin the starting price so the test does not depend on the initial
    # U(0.3, 0.7) draw inside generate_episode().
    state.cluster.spot_price = 0.50
    state.cluster.spot_price_forecast = 0.50

    # A deterministic 5-step walk. Each entry is a pre-clip price delta.
    walks = [+0.08, -0.04, +0.08, -0.08, +0.02]
    rng = _ScriptedRng(walks)

    # Hand-compute the expected sequence of (price, forecast) pairs. The
    # EMA update is: new_forecast = lambda * new_price + (1-lambda) * old.
    lam = 0.25
    price = 0.50
    forecast = 0.50
    expected_sequence = []
    for delta in walks:
        price = round(min(1.0, max(0.1, price + delta)), 2)
        forecast = lam * price + (1.0 - lam) * forecast
        expected_sequence.append((price, forecast))

    actual_sequence = []
    for _ in walks:
        _apply_spot_price_walk(state, rng)
        actual_sequence.append(
            (state.cluster.spot_price, state.cluster.spot_price_forecast)
        )

    for (exp_price, exp_fore), (act_price, act_fore) in zip(
        expected_sequence, actual_sequence
    ):
        assert act_price == exp_price
        assert abs(act_fore - exp_fore) < 1e-12


def test_forecast_converges_to_spot_price_under_constant_trajectory():
    cfg = _richer_cfg(ema_lambda=0.3, initial_forecast=0.10)
    workload_rng, _ = make_episode_rngs(42)
    state = WorkloadGenerator(
        rng=workload_rng, cfg=cfg, num_jobs=4, seed_label=42,
    ).generate_episode()
    # Pin the price to 0.70 and keep the walk at zero so the EMA has to
    # converge geometrically toward 0.70.
    state.cluster.spot_price = 0.70
    state.cluster.spot_price_forecast = 0.10
    rng = _ScriptedRng([0.0] * 50)
    for _ in range(50):
        _apply_spot_price_walk(state, rng)
    assert abs(state.cluster.spot_price_forecast - 0.70) < 1e-4
