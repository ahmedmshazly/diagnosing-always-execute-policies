from __future__ import annotations

"""Phase-3 follow-up ablation: are (alpha, beta, gamma) operative or ceremonial?

The Phase-3 sweep produced 81 (alpha, beta, gamma) cells, yet only 30
distinct per-seed behaviour profiles, and 46 of 81 cells produced
bit-identical outcomes on seeds 100..119. That raises the hypothesis
(reviewer comment before Phase 4) that the Utility-Based agent's decision
rule is dominated by its 29 hand-tuned scoring constants and that the
three utility weights are nearly ceremonial in action selection.

This script tests the hypothesis by recording the per-step action trace
for a panel of weight settings (and ablations of hand-tuned constants)
under matched seeds, then diffing the traces. Outputs:

- ``action_traces.csv``  : one row per (variant, seed, step) with the
                           chosen action, the six per-step action scores,
                           and the forced-execute flag.
- ``variant_summary.csv`` : per-variant action histogram, Hamming distance
                           to the reference variant, and episode metrics.
- ``FINDINGS.md``         : human summary of which knobs are operative.
- ``run_manifest.json``   : usual reproducibility block.

The "reference" variant is always the Phase-1 default
(alpha=1.0, beta=0.1, gamma=1.0, all hand constants on). Every other
variant is scored by the Hamming distance of its 300-step action trace
from the reference trace, summed across the seed panel.
"""

import argparse
import csv
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Tuple

import numpy as np

from src.config import (
    RunConfig,
    UtilityAgentConfig,
    UtilityWeights,
    build_run_config,
    load_config,
)
from src.cost import cost as cost_fn
from src.runner import _simulate  # noqa: F401  - exported for future use
from src.sim_environment import (
    ACTIONS,
    EpisodeState,
    WorkloadGenerator,
    advance_one_step,
    make_episode_rngs,
)
from src.utility_agent import UtilityBasedAgent


# ---------------------------------------------------------------------------
# Variant specification
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Variant:
    name: str
    # Optional overrides. ``None`` means "inherit from the base config".
    alpha: Optional[float] = None
    beta: Optional[float] = None
    gamma: Optional[float] = None
    # Dictionary of utility_agent scoring constants to override.
    utility_agent_overrides: Optional[Mapping[str, float]] = None
    description: str = ""

    def apply(self, base_raw: Mapping[str, Any]) -> RunConfig:
        raw = _deep_copy(base_raw)
        if self.alpha is not None:
            raw["utility"]["alpha"] = float(self.alpha)
        if self.beta is not None:
            raw["utility"]["beta"] = float(self.beta)
        if self.gamma is not None:
            raw["utility"]["gamma"] = float(self.gamma)
        if self.utility_agent_overrides:
            for key, value in self.utility_agent_overrides.items():
                raw["utility_agent"][key] = float(value)
        return build_run_config(raw)


def _deep_copy(value: Any) -> Any:
    if isinstance(value, dict):
        return {k: _deep_copy(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_deep_copy(v) for v in value]
    return value


VARIANTS: Tuple[Variant, ...] = (
    Variant(
        name="reference_phase1",
        alpha=1.0,
        beta=0.1,
        gamma=1.0,
        description="Phase-1 default: (alpha, beta, gamma) = (1.0, 0.1, 1.0).",
    ),
    Variant(
        name="midterm_weights",
        alpha=1.0,
        beta=0.4,
        gamma=0.8,
        description="Midterm weights (1.0, 0.4, 0.8) — same equivalence class as reference?",
    ),
    Variant(
        name="alpha_0",
        alpha=0.0,
        description="Zero out alpha only.",
    ),
    Variant(
        name="beta_0",
        beta=0.0,
        description="Zero out beta only.",
    ),
    Variant(
        name="gamma_0",
        gamma=0.0,
        description="Zero out gamma only.",
    ),
    Variant(
        name="all_zero",
        alpha=0.0,
        beta=0.0,
        gamma=0.0,
        description="Zero out all three utility weights (extreme ablation).",
    ),
    Variant(
        name="alpha_100",
        alpha=100.0,
        description="alpha=100 (should strongly prefer execute).",
    ),
    Variant(
        name="beta_100",
        beta=100.0,
        description="beta=100 (cost dominates; should avoid execute).",
    ),
    Variant(
        name="gamma_100",
        gamma=100.0,
        description="gamma=100 (risk dominates; should avoid execute).",
    ),
    Variant(
        name="alpha_beta_gamma_extreme_neg_execute",
        alpha=0.0,
        beta=100.0,
        gamma=100.0,
        description="Should punish execute very hard; only stress-guard forces it.",
    ),
    # Hand-constant ablations. These test whether the 29 non-(alpha, beta, gamma)
    # weights are what actually drive behaviour.
    Variant(
        name="best_task_cost_off",
        utility_agent_overrides={"best_task_resource_cost_weight": 0.0},
        description="Turn off the task-ranking cost weight.",
    ),
    Variant(
        name="stress_guard_disabled",
        utility_agent_overrides={
            "stress_guard_failure_limit": -1.0,
            "stress_guard_price_limit": -1.0,
        },
        description=(
            "Effectively disables _should_force_execution so alpha/beta/gamma "
            "have to win on their own."
        ),
    ),
    Variant(
        name="stress_guard_disabled_plus_alpha_0",
        alpha=0.0,
        utility_agent_overrides={
            "stress_guard_failure_limit": -1.0,
            "stress_guard_price_limit": -1.0,
        },
        description=(
            "Disable the stress guard AND zero alpha. Only deviates from the "
            "reference trace if the utility coefficients are actually doing work."
        ),
    ),
    Variant(
        name="urgency_weight_off",
        utility_agent_overrides={"execution_urgency_weight": 0.0},
        description="Drop execute's urgency bonus; tests one hand constant.",
    ),
    # A deeper probe into the handful of constants that actually shape behaviour.
    Variant(
        name="stress_guard_price_1_1",
        utility_agent_overrides={"stress_guard_price_limit": 1.1},
        description=(
            "Make the stress guard always pass on price (>= any spot). Tests "
            "whether the price side of the guard is binding."
        ),
    ),
    Variant(
        name="stress_guard_failure_2",
        utility_agent_overrides={"stress_guard_failure_limit": 2.0},
        description=(
            "Make the stress guard always pass on failure pressure. Tests "
            "whether the failure side of the guard is binding."
        ),
    ),
    Variant(
        name="no_executable_off",
        utility_agent_overrides={"no_executable_task_score": 0.0},
        description=(
            "Treat 'no ready task fits' as 0 rather than -10000. Changes "
            "which action wins when the queue is blocked."
        ),
    ),
    Variant(
        name="scale_up_cheaper",
        utility_agent_overrides={"scale_up_price_weight": 1.0},
        description=(
            "Cut scale_up_price_weight from 8 to 1: Scale_Up should win much "
            "more often when not forced."
        ),
    ),
    Variant(
        name="pause_disabled",
        utility_agent_overrides={
            "no_low_priority_work_penalty": -1e6,
            "unnecessary_pause_penalty": -1e6,
            "pause_base_penalty": 1e6,
        },
        description=(
            "Disable Pause_LowPriority_Job entirely by making its score "
            "extremely negative. Isolates the role of pause in non-forced steps."
        ),
    ),
    Variant(
        name="alpha_only_with_guard_off",
        alpha=5.0,
        beta=0.0,
        gamma=0.0,
        utility_agent_overrides={
            "execution_urgency_weight": 0.0,
        },
        description=(
            "Only alpha is on. This is the cleanest test of whether alpha "
            "alone (times value_term) is enough to keep execute winning."
        ),
    ),
)


# ---------------------------------------------------------------------------
# Per-episode action trace collector
# ---------------------------------------------------------------------------
def _collect_trace(
    cfg: RunConfig,
    seed: int,
    max_steps: Optional[int] = None,
    include_scores: bool = False,
) -> Tuple[List[Dict[str, Any]], Dict[str, float]]:
    workload_rng, event_rng = make_episode_rngs(seed)
    generator = WorkloadGenerator(
        rng=workload_rng,
        cfg=cfg,
        num_jobs=cfg.experiment.num_jobs,
        seed_label=seed,
    )
    state = generator.generate_episode()
    agent = UtilityBasedAgent(cfg)
    step_cap = max_steps if max_steps is not None else cfg.experiment.max_steps

    trace: List[Dict[str, Any]] = []
    total_compute_cost = 0.0

    while not state.all_done() and state.step < step_cap:
        step_index = state.step
        # Capture per-action scores pre-step if requested.
        scores: Dict[str, float] = {}
        if include_scores:
            for action in ACTIONS:
                try:
                    scores[action] = agent.score_action(state, action)
                except Exception:  # noqa: BLE001 - unknown stress paths
                    scores[action] = float("nan")
        action = agent.choose_action(state)
        # Is the forced-execute path firing?
        forced = False
        if include_scores and action == "Execute_Ready_Job":
            # Reproduce the guard by re-running _should_force_execution
            try:
                # Need a local import to avoid circularity
                forced = agent._should_force_execution(state, scores)  # type: ignore[attr-defined]
            except Exception:  # noqa: BLE001
                forced = False
        advance_one_step(state, event_rng, action)
        total_compute_cost += cost_fn(state, action)

        row: Dict[str, Any] = {
            "step": step_index,
            "action": action,
            "forced_execute": bool(forced),
        }
        if include_scores:
            for name, score in scores.items():
                row[f"score_{name}"] = round(float(score), 6)
        trace.append(row)

    completed = sum(1 for job in state.jobs if job.completed)
    failed = sum(1 for job in state.jobs if job.failed)
    completed_value = sum(job.value for job in state.jobs if job.completed)
    total_job_value = sum(job.value for job in state.jobs)

    episode_summary = {
        "completed_jobs": completed,
        "failed_jobs": failed,
        "completion_rate": completed / max(cfg.experiment.num_jobs, 1),
        "value_weighted_completion_rate": (
            completed_value / total_job_value if total_job_value > 0 else 0.0
        ),
        "total_compute_cost": round(float(total_compute_cost), 6),
        "total_completed_value": round(float(completed_value), 6),
        "steps_executed": state.step,
        "hit_step_budget": (not state.all_done()) and state.step >= step_cap,
    }
    return trace, episode_summary


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument(
        "--seeds",
        nargs="*",
        type=int,
        default=[100, 101, 102, 103, 104],
        help="Seed panel to trace. Default: 100..104 (Phase-3 sweep pool).",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=None,
        help="Override cfg.experiment.max_steps for fast smoke runs.",
    )
    parser.add_argument(
        "--out-root",
        type=Path,
        default=Path("results"),
    )
    parser.add_argument(
        "--run-id",
        type=str,
        default="ablation_phase3_weights",
    )
    parser.add_argument(
        "--record-scores",
        action="store_true",
        help="Also record the six per-step action scores (larger CSV).",
    )
    return parser.parse_args()


def _variant_hamming(
    reference_trace: List[Dict[str, Any]],
    other_trace: List[Dict[str, Any]],
) -> Tuple[int, int]:
    common = min(len(reference_trace), len(other_trace))
    mismatches = 0
    for idx in range(common):
        if reference_trace[idx]["action"] != other_trace[idx]["action"]:
            mismatches += 1
    total_compared = common + abs(len(reference_trace) - len(other_trace))
    return mismatches, total_compared


def _action_histogram(trace: List[Dict[str, Any]]) -> Dict[str, int]:
    hist = {action: 0 for action in ACTIONS}
    for row in trace:
        hist[row["action"]] = hist.get(row["action"], 0) + 1
    return hist


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config) if args.config else load_config()
    out_dir = args.out_root / args.run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    started = time.time()

    base_raw = dict(cfg.raw)
    seeds = list(args.seeds)

    traces_by_variant_seed: Dict[Tuple[str, int], List[Dict[str, Any]]] = {}
    summaries: List[Dict[str, Any]] = []

    # Run the reference variant first so it's cached for Hamming comparisons.
    reference_name = VARIANTS[0].name

    for variant in VARIANTS:
        variant_cfg = variant.apply(base_raw)
        for seed in seeds:
            trace, summary = _collect_trace(
                variant_cfg, seed=seed, max_steps=args.max_steps,
                include_scores=args.record_scores,
            )
            traces_by_variant_seed[(variant.name, seed)] = trace
            forced_count = sum(1 for row in trace if row.get("forced_execute"))
            summary_row = {
                "variant": variant.name,
                "alpha": variant_cfg.utility.alpha,
                "beta": variant_cfg.utility.beta,
                "gamma": variant_cfg.utility.gamma,
                "seed": seed,
                "description": variant.description,
                "steps_executed": summary["steps_executed"],
                "completed_jobs": summary["completed_jobs"],
                "failed_jobs": summary["failed_jobs"],
                "completion_rate": summary["completion_rate"],
                "value_weighted_completion_rate": summary["value_weighted_completion_rate"],
                "total_compute_cost": summary["total_compute_cost"],
                "forced_execute_steps": forced_count,
            }
            summary_row.update({f"action_count_{a}": 0 for a in ACTIONS})
            for action, count in _action_histogram(trace).items():
                summary_row[f"action_count_{action}"] = count
            summaries.append(summary_row)

    # Hamming distances from the reference trace, per variant.
    hamming_table: List[Dict[str, Any]] = []
    for variant in VARIANTS:
        if variant.name == reference_name:
            continue
        row = {"variant": variant.name}
        total_mismatches = 0
        total_compared = 0
        for seed in seeds:
            ref = traces_by_variant_seed[(reference_name, seed)]
            other = traces_by_variant_seed[(variant.name, seed)]
            mismatches, compared = _variant_hamming(ref, other)
            row[f"hamming_seed_{seed}"] = mismatches
            row[f"steps_seed_{seed}"] = compared
            total_mismatches += mismatches
            total_compared += compared
        row["hamming_total"] = total_mismatches
        row["steps_total"] = total_compared
        row["hamming_fraction"] = (
            round(total_mismatches / total_compared, 6) if total_compared > 0 else float("nan")
        )
        hamming_table.append(row)

    # Write CSVs.
    summary_path = out_dir / "variant_summary.csv"
    if summaries:
        with summary_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(summaries[0].keys()))
            writer.writeheader()
            writer.writerows(summaries)

    hamming_path = out_dir / "hamming_vs_reference.csv"
    if hamming_table:
        with hamming_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(hamming_table[0].keys()))
            writer.writeheader()
            writer.writerows(hamming_table)

    # action_traces.csv: long-form, one row per (variant, seed, step). Can get
    # large; cap to reference + a selected subset when --record-scores is off.
    trace_path = out_dir / "action_traces.csv"
    fieldnames = ["variant", "seed", "step", "action", "forced_execute"]
    if args.record_scores:
        fieldnames += [f"score_{a}" for a in ACTIONS]
    with trace_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for (variant_name, seed), trace in traces_by_variant_seed.items():
            for row in trace:
                out_row: Dict[str, Any] = {
                    "variant": variant_name,
                    "seed": seed,
                    "step": row["step"],
                    "action": row["action"],
                    "forced_execute": row.get("forced_execute", False),
                }
                if args.record_scores:
                    for action in ACTIONS:
                        out_row[f"score_{action}"] = row.get(f"score_{action}", "")
                writer.writerow(out_row)

    # run manifest (light version; reuses the existing helper).
    from src.run_artifacts import write_manifest, write_resolved_config
    write_resolved_config(out_dir, cfg)
    write_manifest(
        run_dir=out_dir,
        cfg=cfg,
        seeds=seeds,
        extra={
            "entrypoint": "ablation_phase3_weights",
            "num_variants": len(VARIANTS),
            "reference_variant": reference_name,
            "record_scores": bool(args.record_scores),
            "max_steps_override": args.max_steps,
        },
        wall_clock_start=started,
    )

    print(
        f"Wrote ablation to {out_dir}: "
        f"{len(VARIANTS)} variants x {len(seeds)} seeds; "
        f"{len(summaries)} summary rows, {len(hamming_table)} Hamming rows."
    )
    # Print the Hamming summary inline for quick eyeballing.
    print("\nHamming distance from reference trace (action mismatches / total steps):")
    for row in hamming_table:
        print(
            f"  {row['variant']:<40s}  "
            f"mismatches={row['hamming_total']:>5d}/{row['steps_total']:<5d}  "
            f"({row['hamming_fraction']:.4f})"
        )


if __name__ == "__main__":
    main()
