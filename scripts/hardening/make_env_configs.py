from __future__ import annotations

"""Generate environment-variation configs for Stage B (Phase-2).

The config loader does NOT merge partials, so each variant must be a full,
self-contained config. This script loads config/default.yaml, applies an
explicit, minimal set of overrides per variant, validates the result by
building a RunConfig, and writes config/env_<name>.yaml.

Because the config hash is taken over the *resolved* dict (sorted-key YAML
dump), comment loss from the round-trip is irrelevant to reproducibility:
the generated file resolves to exactly the documented overrides applied to
default.yaml.

Variants (the factor each one varies relative to the benign default):

  tight      - base cluster capacity lowered to the floor (6 CPU / 8 RAM) so
               the cluster is genuinely contended, but EVERY task still fits
               individually (max task demand is 6 CPU / 6 RAM). Scale_Up adds
               parallelism and lets more jobs finish before the 300-step
               truncation. Varies: CAPACITY PRESSURE (scaling raises throughput)
               WITHOUT triggering the do_action jam.

  loadfail   - node_failure mode per_node_bernoulli (each running task fails
               independently at prob 0.05), so concurrency raises the failure
               rate. Varies: DECISION-DEPENDENCE OF FAILURE (caution can
               reduce risk).

  tight_loadfail - both of the above.

Heavy-tailed value is a separate variant that needs a workload-generator
change and is produced once that lands (env_heavytail).

NOTE on the discarded `capacity` variant (a 12-CPU 'heavy' task > base 10):
it did NOT produce a clean scaling incentive. It instead triggered a
do_action pathology — Execute sorts ready tasks, tries only the TOP one, and
breaks if it does not fit, never falling through to the ~158 other fittable
ready tasks. A single high-value unfittable task starves the cluster
(298/300 steps were insufficient_resources, 0 completions). The MC probe's
large Scale_Up advantage there reflected unjamming a catastrophically jammed
baseline, not a realistic scaling-pays signal, and no fixed scaling policy
could exploit it. `env_tight` avoids the jam by keeping every task fittable.
"""

import copy
from pathlib import Path

import yaml

from src.config import REPO_ROOT, build_run_config

DEFAULT_PATH = REPO_ROOT / "config" / "default.yaml"


def _load_default_raw() -> dict:
    with DEFAULT_PATH.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def _tight(raw: dict) -> dict:
    raw = copy.deepcopy(raw)
    raw["meta"]["config_name"] = "env_tight"
    # Lower base capacity to the floor; scaling is the only way up. Every task
    # still fits (max demand 6 CPU / 6 RAM <= 6 / 8), so Execute never jams.
    raw["simulator"]["cluster"]["base_cpu_capacity"] = 6
    raw["simulator"]["cluster"]["base_ram_capacity"] = 8
    return raw


def _loadfail(raw: dict) -> dict:
    raw = copy.deepcopy(raw)
    raw["meta"]["config_name"] = "env_loadfail"
    raw["simulator"]["stochastic_processes"]["node_failure"]["mode"] = "per_node_bernoulli"
    # prob unchanged at 0.05 (now PER RUNNING TASK rather than per step).
    return raw


def _tight_loadfail(raw: dict) -> dict:
    raw = _tight(raw)
    raw["meta"]["config_name"] = "env_tight_loadfail"
    raw["simulator"]["stochastic_processes"]["node_failure"]["mode"] = "per_node_bernoulli"
    return raw


def _fixedrisk(raw: dict) -> dict:
    """Benign env + corrected reward (Sum r_t == U). Tests whether always-
    execute survives once the failure penalty is actually present in the
    reward (prediction: yes, because benign-env failures are
    decision-independent)."""
    raw = copy.deepcopy(raw)
    raw["meta"]["config_name"] = "env_fixedrisk"
    raw["rl"]["reward_risk_mode"] = "failed_jobs_delta"
    return raw


def _tight_fixedrisk(raw: dict) -> dict:
    """Both confounds removed: the env rewards a non-execute action AND the
    reward is correctly specified. The cleanest test of whether RL learns
    non-trivial behaviour when it genuinely should."""
    raw = _tight(raw)
    raw["meta"]["config_name"] = "env_tight_fixedrisk"
    raw["rl"]["reward_risk_mode"] = "failed_jobs_delta"
    return raw


def _tight_fixedrisk_flat(raw: dict) -> dict:
    """Both confounds removed AND no curriculum: train directly at the hard
    regime (N=100, T=300) the whole time. This is the control for the
    curriculum-stickiness hypothesis. If this learns to scale but the
    curriculum runs do not, the curriculum locked the policy into the
    stage-1 always-execute attractor. If even this stays at always-execute,
    the attractor is deeper than the schedule."""
    raw = _tight_fixedrisk(raw)
    raw["meta"]["config_name"] = "env_tight_fixedrisk_flat"
    raw["rl"]["curriculum"]["stages"] = [
        {"num_jobs": 100, "max_steps": 300, "num_updates": 200}
    ]
    return raw


def _cascade_base(raw: dict) -> dict:
    """Load-cascade failure: P(failure) jumps to high_prob when cpu_load >
    load_threshold, else low_prob. Keeping load <= 0.4 (throttle/scale) reduces
    failures, so caution PAYS (throttle beats always-execute +13.1, p=0.014).
    Unlike per_step_single_victim / per_node_bernoulli, expected failures are
    NOT policy-invariant here."""
    raw = copy.deepcopy(raw)
    nf = raw["simulator"]["stochastic_processes"]["node_failure"]
    nf["mode"] = "load_cascade"
    nf["load_threshold"] = 0.4
    nf["high_prob"] = 0.4
    nf["low_prob"] = 0.01
    return raw


def _cascade(raw: dict) -> dict:
    """Cascade + CORRECTED reward (failed_jobs_delta). The lever is failure-
    avoidance, so the reward must actually penalise failures for RL to have any
    incentive to throttle. This is the real test."""
    raw = _cascade_base(raw)
    raw["meta"]["config_name"] = "env_cascade"
    raw["rl"]["reward_risk_mode"] = "failed_jobs_delta"
    # Flat schedule (train directly at N=100) so the caution lever is active
    # throughout and the curriculum cannot pre-lock always-execute.
    raw["rl"]["curriculum"]["stages"] = [
        {"num_jobs": 100, "max_steps": 300, "num_updates": 150}
    ]
    return raw


def _cascade_broken(raw: dict) -> dict:
    """Cascade + BROKEN reward (counter_delta, the Phase-5/6 default). The
    failure penalty is ≈0, so RL has no incentive to avoid the (now real)
    cascade failures -> should ignore the caution lever. Pairing this with
    env_cascade demonstrates that the A1 reward bug materially changes what RL
    can learn."""
    raw = _cascade_base(raw)
    raw["meta"]["config_name"] = "env_cascade_broken"
    raw["rl"]["reward_risk_mode"] = "counter_delta"
    raw["rl"]["curriculum"]["stages"] = [
        {"num_jobs": 100, "max_steps": 300, "num_updates": 150}
    ]
    return raw


def _tight_fixedrisk_flat_hientropy(raw: dict) -> dict:
    """Flat + corrected reward + 10x entropy coefficient (0.01 -> 0.1). The
    capstone control for the optimisation-failure claim: if more exploration
    escapes the always-execute basin and learns to scale, the env_tight
    attractor is an exploration/optimisation failure (as the discounted-return
    measurement implies). If even this stays at always-execute, the basin is
    robust to exploration alone (still optimiser-determined, but harder)."""
    raw = _tight_fixedrisk_flat(raw)
    raw["meta"]["config_name"] = "env_tight_fixedrisk_flat_hientropy"
    raw["rl"]["entropy_coef"] = 0.1
    return raw


def _heavytail_tight(raw: dict) -> dict:
    """Heavy-tailed value (10% whales worth 50) + tight capacity + corrected
    reward + 14-dim richer state (so the policy can SEE per-job value). Scaling
    to finish whales pays +18.5 (p=0.026) -- a larger lever than env_tight's +6,
    smaller than env_cascade's +120. Tests where the REINFORCE-success threshold
    lies and whether the 14-dim state enables value-aware scaling."""
    raw = _tight(raw)
    raw["meta"]["config_name"] = "env_heavytail_tight"
    w = raw["simulator"]["workload"]
    w["value_distribution"] = "heavy_tail"
    w["heavy_tail_prob"] = 0.1
    w["heavy_tail_value"] = 50.0
    raw["rl"]["reward_risk_mode"] = "failed_jobs_delta"
    raw["state_v2"]["use_richer_state"] = True
    raw["state_v2"]["queue_features"]["job_value_max"] = 50.0
    raw["rl"]["curriculum"]["stages"] = [
        {"num_jobs": 100, "max_steps": 300, "num_updates": 150}
    ]
    return raw


VARIANTS = {
    "env_tight": _tight,
    "env_loadfail": _loadfail,
    "env_tight_loadfail": _tight_loadfail,
    "env_fixedrisk": _fixedrisk,
    "env_tight_fixedrisk": _tight_fixedrisk,
    "env_tight_fixedrisk_flat": _tight_fixedrisk_flat,
    "env_tight_fixedrisk_flat_hientropy": _tight_fixedrisk_flat_hientropy,
    "env_cascade": _cascade,
    "env_cascade_broken": _cascade_broken,
    "env_heavytail_tight": _heavytail_tight,
}

HEADER = (
    "# GENERATED by scripts/hardening/make_env_configs.py from config/default.yaml.\n"
    "# Stage-B (Phase-2) environment-variation config. Do not edit by hand;\n"
    "# edit the generator and re-run. Differs from default.yaml only in the\n"
    "# documented overrides for variant: {name}.\n"
)


def main() -> None:
    raw = _load_default_raw()
    out_dir = REPO_ROOT / "config"
    for name, fn in VARIANTS.items():
        variant_raw = fn(raw)
        # Validate before writing — fail loudly on a bad override.
        build_run_config(variant_raw)
        path = out_dir / f"{name}.yaml"
        with path.open("w", encoding="utf-8") as handle:
            handle.write(HEADER.format(name=name))
            yaml.safe_dump(variant_raw, handle, sort_keys=False, default_flow_style=False)
        cfg = build_run_config(variant_raw)
        n_templates = len(cfg.simulator.dag_templates)
        nf = cfg.simulator.stochastic_processes.node_failure.mode
        print(f"wrote {path.name:24} | dag_templates={n_templates} | node_failure={nf}")


if __name__ == "__main__":
    main()
