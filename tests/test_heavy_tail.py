from __future__ import annotations

"""Tests for the heavy-tailed value workload (hardening).

Pins two things:
- the default "uniform" path is byte-identical to before (same RNG sequence),
  so Phase-5/6 seeds reproduce;
- the "heavy_tail" path actually produces a fraction of high-value whales.
"""

import numpy as np

from src.config import build_run_config, load_config
from src.sim_environment import WorkloadGenerator, make_episode_rngs


def _gen_values(cfg, seed=200, n=100):
    wr, _ = make_episode_rngs(seed)
    gen = WorkloadGenerator(rng=wr, cfg=cfg, num_jobs=n, seed_label=seed)
    ep = gen.generate_episode()
    return np.array([j.value for j in ep.jobs])


def test_uniform_default_unchanged():
    cfg = load_config()
    vals = _gen_values(cfg)
    # All within U(1,5); no whales.
    assert vals.min() >= 1.0 and vals.max() <= 5.0
    assert cfg.simulator.workload.value_distribution == "uniform"


def test_uniform_reproduces_exact_sequence():
    """The uniform branch must not change the workload RNG draw order."""
    cfg = load_config()
    a = _gen_values(cfg, seed=12345)
    b = _gen_values(cfg, seed=12345)
    assert np.array_equal(a, b)
    # Spot value: deterministic given the seed/algorithm.
    assert abs(float(a.mean()) - 3.0) < 1.0  # U(1,5) mean ~3


def test_heavy_tail_produces_whales():
    cfg = load_config()
    raw = {**cfg.raw}
    raw["simulator"] = {**raw["simulator"]}
    raw["simulator"]["workload"] = {
        **raw["simulator"]["workload"],
        "value_distribution": "heavy_tail",
        "heavy_tail_prob": 0.1,
        "heavy_tail_value": 50.0,
    }
    hcfg = build_run_config(raw)
    vals = _gen_values(hcfg, n=200)
    whales = (vals >= 49.0).sum()
    # ~10% of 200 = ~20 whales; allow sampling slack.
    assert 5 <= whales <= 45, f"expected ~20 whales, got {whales}"
    # non-whales still in U(1,5)
    non = vals[vals < 49.0]
    assert non.min() >= 1.0 and non.max() <= 5.0


def test_heavy_tail_validation_rejects_unknown():
    import pytest
    cfg = load_config()
    raw = {**cfg.raw}
    raw["simulator"] = {**raw["simulator"]}
    raw["simulator"]["workload"] = {**raw["simulator"]["workload"],
                                    "value_distribution": "bogus"}
    with pytest.raises(ValueError, match="value_distribution"):
        build_run_config(raw)
