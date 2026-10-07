from __future__ import annotations

"""Phase-4 config regression tests: renames + seed-pool disjointness."""

from pathlib import Path

import pytest

from src.config import build_run_config, load_config


def _deep_copy(value):
    if isinstance(value, dict):
        return {k: _deep_copy(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_deep_copy(v) for v in value]
    return value


def test_default_config_uses_delta_discount_not_gamma(default_config):
    assert hasattr(default_config.rl, "delta_discount")
    assert not hasattr(default_config.rl, "gamma_discount")
    assert default_config.rl.delta_discount == 0.99


def test_default_config_has_no_value_loss_coef(default_config):
    # Phase-4 brief §A.2 required removal. Dataclass is the source of truth.
    assert not hasattr(default_config.rl, "value_loss_coef")


def test_default_config_training_pool_is_range_300_999(default_config):
    training = list(default_config.rl.training_seeds)
    assert training[0] == 300
    assert training[-1] == 999
    assert len(training) == 700


def test_default_config_validation_pool_is_range_250_299(default_config):
    val = list(default_config.rl.validation_seeds)
    assert val[0] == 250
    assert val[-1] == 299
    assert len(val) == 50


def test_default_config_test_pool_is_range_200_249(default_config):
    test = list(default_config.rl.test_seeds)
    assert test[0] == 200
    assert test[-1] == 249
    assert len(test) == 50


def test_default_config_init_seeds_are_7_11_13(default_config):
    assert list(default_config.rl.initialisation_seeds) == [7, 11, 13]


def test_rl_pools_are_pairwise_disjoint(default_config):
    train = set(default_config.rl.training_seeds)
    val = set(default_config.rl.validation_seeds)
    test = set(default_config.rl.test_seeds)
    assert train.isdisjoint(val)
    assert train.isdisjoint(test)
    assert val.isdisjoint(test)


def test_rl_pools_disjoint_from_baseline_and_sweep_pools(default_config):
    rl_union = (
        set(default_config.rl.training_seeds)
        | set(default_config.rl.validation_seeds)
        | set(default_config.rl.test_seeds)
    )
    baseline_seeds = (
        set(default_config.seeds.train)
        | set(default_config.seeds.test)
        | set(default_config.seeds.midterm_baseline)
    )
    # Phase-3 sweep pool is 100..119.
    phase3_sweep = set(range(100, 120))
    # Phase-2 50-seed baseline pool is 0..49.
    phase2_pool = set(range(0, 50))

    assert rl_union.isdisjoint(baseline_seeds), (
        f"RL pool overlaps paper.seeds.{{train,test,midterm_baseline}}: "
        f"{sorted(rl_union & baseline_seeds)[:5]}..."
    )
    assert rl_union.isdisjoint(phase3_sweep), "RL pool overlaps Phase-3 sweep"
    assert rl_union.isdisjoint(phase2_pool), "RL pool overlaps Phase-2 baseline"


def test_stripped_utility_agent_block_loads(default_config):
    assert hasattr(default_config, "stripped_utility_agent")
    assert default_config.stripped_utility_agent.disable_force_execute_guard is True


def test_pre_phase4_keys_raise_loud_errors():
    """Using the old rl.gamma_discount/value_loss_coef/training_seed_list
    keys must fail at load — silent ignore would be a correctness bug."""
    raw = _deep_copy(load_config().raw)
    bad = _deep_copy(raw)
    bad["rl"]["gamma_discount"] = 0.99
    with pytest.raises(ValueError, match="delta_discount"):
        build_run_config(bad)

    bad = _deep_copy(raw)
    bad["rl"]["value_loss_coef"] = 0.5
    with pytest.raises(ValueError, match="value_loss_coef"):
        build_run_config(bad)

    bad = _deep_copy(raw)
    bad["rl"]["training_seed_list"] = [100]
    with pytest.raises(ValueError, match="training_seed_list"):
        build_run_config(bad)


def test_rl_pool_overlap_fails_validation():
    raw = _deep_copy(load_config().raw)
    # Force an overlap between training and validation.
    raw["rl"]["training_seeds"] = [250, 251]
    with pytest.raises(ValueError, match="validation_seeds"):
        build_run_config(raw)
