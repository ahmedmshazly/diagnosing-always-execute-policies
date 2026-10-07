from __future__ import annotations

"""Config-loader regression tests."""

from pathlib import Path

from src.config import load_config, override_utility_weights


def test_default_config_loads_and_has_stable_hash(default_config):
    """The default config parses and its hash is deterministic."""
    first = default_config.config_hash()
    second = load_config().config_hash()
    assert first == second
    assert len(first) == 64  # sha256 hex


def test_phase1_default_weights_are_documented():
    """Phase-1 defaults pin (alpha, beta, gamma) = (1.0, 0.1, 1.0).

    The midterm values (1.0, 0.4, 0.8) are retained under
    utility.midterm_values so Table 1 stays reproducible via
    config/midterm_weights.yaml.
    """
    cfg = load_config()
    assert cfg.utility.alpha == 1.0
    assert cfg.utility.beta == 0.1
    assert cfg.utility.gamma == 1.0
    assert dict(cfg.utility.midterm_values) == {"alpha": 1.0, "beta": 0.4, "gamma": 0.8}


def test_simulator_constants_match_spec():
    cfg = load_config()
    sp = cfg.simulator.stochastic_processes
    cluster = cfg.simulator.cluster
    assert sp.node_failure.prob == 0.05
    assert sp.node_failure.mode == "per_step_single_victim"
    assert sp.data_spike.prob == 0.08
    assert sp.data_spike.mode == "additive_bump"
    assert sp.spot_price.mode == "bounded_random_walk"
    assert cluster.base_cpu_capacity == 10
    assert cluster.base_ram_capacity == 16
    assert cfg.experiment.num_jobs == 100
    assert cfg.experiment.max_steps == 300


def test_action_params_are_fully_specified():
    cfg = load_config()
    params = cfg.simulator.action_params
    assert params.execute_ready_job.selection_policy == "value_times_priority"
    assert params.execute_ready_job.max_launches_per_step == 1
    assert params.defer_job.duration_steps == 1
    assert params.scale_up.cpu_delta == 3
    assert params.scale_up.ram_delta == 4
    assert params.scale_up.duration_steps == 3
    assert params.scale_down.cpu_delta == 2
    assert params.scale_down.ram_delta == 2
    assert params.reprioritize_queue.bump == 0.05
    assert params.reprioritize_queue.cap == 1.0
    assert params.pause_low_priority_job.priority_threshold == 0.4
    assert params.pause_low_priority_job.max_jobs == 2


def test_action_costs_table_covers_every_action():
    cfg = load_config()
    action_costs = dict(cfg.simulator.cost.action_costs)
    assert set(action_costs) == set(cfg.simulator.actions)


def test_midterm_weights_override_config_loads():
    """config/midterm_weights.yaml must be valid and set (1.0, 0.4, 0.8)."""
    cfg = load_config(Path("config/midterm_weights.yaml"))
    assert (cfg.utility.alpha, cfg.utility.beta, cfg.utility.gamma) == (1.0, 0.4, 0.8)


def test_train_test_seeds_are_disjoint():
    cfg = load_config()
    assert set(cfg.seeds.train).isdisjoint(set(cfg.seeds.test))


def test_override_utility_weights_produces_new_hash():
    base = load_config()
    bumped = override_utility_weights(base, alpha=2.0)
    assert bumped.utility.alpha == 2.0
    assert bumped.utility.beta == base.utility.beta
    assert bumped.config_hash() != base.config_hash()
