from __future__ import annotations

"""Phase-5 regression guard under Phase-6 V1 code.

The Phase-6 brief requires that ``state_v2.use_richer_state: false`` leave
Phase-5 numerics bit-identical. This test re-runs every agent that Phase 5
evaluated on a sample of held-out seeds and asserts that the per-seed
metrics match the committed ``results/phase5/heldout/<agent>/metrics.csv``
to within floating-point equality on every recorded column.

If this test fails, a Phase-6 change has leaked into the Phase-5
behaviour path and has to be either reverted or guarded more tightly.
"""

import csv
from pathlib import Path

import pytest

from src.config import load_config, override_utility_weights
from src.reflex_agent import build_reflex_agent
from src.rl.agent import load_policy, make_rl_agent_factory
from src.runner import run_many_episodes
from src.utility_agent import build_utility_agent


REFLEX_PATH = Path("results/phase5/heldout/reflex/metrics.csv")
TUNED_PATH = Path("results/phase5/heldout/tuned_utility/metrics.csv")
RL_SEED7_PATH = Path("results/phase5/heldout/rl_seed7/metrics.csv")
RL_SEED7_CHECKPOINT = Path("results/phase5/rl_seed7/policy_best_by_val.pt")

SAMPLE_SEEDS = (200, 203, 210, 225, 240)

METRICS_TO_CHECK = (
    "completion_rate",
    "value_weighted_completion_rate",
    "uncapped_completion_rate",
    "failure_rate",
    "total_completed_value",
    "total_compute_cost",
    "total_utility",
    "completed_jobs",
    "failed_jobs",
)


def _read_reference(path: Path):
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        return {int(row["seed"]): row for row in reader}


def _compare(metric, reference_row, metrics_record):
    """Assert bit-identical numerics on the shared fields."""
    actual_row = metrics_record.as_dict()
    for field in METRICS_TO_CHECK:
        ref_val = float(reference_row[field])
        act_val = float(actual_row[field])
        assert ref_val == act_val, (
            f"{metric}: seed={reference_row['seed']} field={field} "
            f"reference={ref_val} actual={act_val}"
        )


@pytest.mark.skipif(not REFLEX_PATH.exists(), reason="Phase-5 reference missing")
def test_phase6_default_config_reproduces_phase5_reflex_metrics(default_config):
    assert default_config.state_v2.use_richer_state is False
    reference = _read_reference(REFLEX_PATH)
    seeds = [s for s in SAMPLE_SEEDS if s in reference]
    metrics = run_many_episodes(
        cfg=default_config, agent_factory=build_reflex_agent, seeds=seeds
    )
    assert len(metrics) == len(seeds)
    for metric_record in metrics:
        _compare("Reflex", reference[metric_record.seed], metric_record)


@pytest.mark.skipif(not TUNED_PATH.exists(), reason="Phase-5 reference missing")
def test_phase6_default_config_reproduces_phase5_tuned_utility_metrics(default_config):
    # Phase-5 Tuned UB used (alpha=4, beta=1, gamma=4). Same weight
    # override path is used here.
    tuned_cfg = override_utility_weights(default_config, alpha=4.0, beta=1.0, gamma=4.0)
    assert tuned_cfg.state_v2.use_richer_state is False
    reference = _read_reference(TUNED_PATH)
    seeds = [s for s in SAMPLE_SEEDS if s in reference]
    metrics = run_many_episodes(
        cfg=tuned_cfg, agent_factory=build_utility_agent, seeds=seeds
    )
    for metric_record in metrics:
        _compare("Tuned UB", reference[metric_record.seed], metric_record)


@pytest.mark.skipif(
    not (RL_SEED7_PATH.exists() and RL_SEED7_CHECKPOINT.exists()),
    reason="Phase-5 RL reference / checkpoint missing",
)
def test_phase6_default_config_reproduces_phase5_rl_seed7_metrics(default_config):
    """The Phase-5 8-dim checkpoint should still score bit-identically
    when loaded into a Phase-6 harness with the flag off."""
    reference = _read_reference(RL_SEED7_PATH)
    seeds = [s for s in SAMPLE_SEEDS if s in reference]
    factory = make_rl_agent_factory(
        cfg=default_config,
        checkpoint_path=RL_SEED7_CHECKPOINT,
        deterministic=True,
        label="rl_seed7",
    )
    metrics = run_many_episodes(cfg=default_config, agent_factory=factory, seeds=seeds)
    for metric_record in metrics:
        _compare("RL seed 7", reference[metric_record.seed], metric_record)
