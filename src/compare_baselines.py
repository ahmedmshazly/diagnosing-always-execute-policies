from __future__ import annotations

"""Baseline comparison driver.

Runs the Reflex and Utility-Based baselines under matched seeds using the
configuration loaded from ``config/default.yaml`` (or an override passed via
``--config``), and writes the resulting artifacts into ``<out_root>/<run_id>/``
following the reproducibility convention in :mod:`src.run_artifacts`.

Every numeric setting (seeds, num_jobs, max_steps, alpha/beta/gamma, every
agent / simulator constant) is driven by the YAML config. CLI flags only
exist for operational overrides (config path, run id, seed group).
"""

import argparse
import statistics
import time
from pathlib import Path
from typing import Any, Dict, List, Sequence

from .config import RunConfig, load_config
from .metrics import EpisodeMetrics
from .reflex_agent import build_reflex_agent
from .run_artifacts import (
    ensure_run_dir,
    generate_run_id,
    write_json,
    write_manifest,
    write_metrics_csv,
    write_resolved_config,
    write_text,
)
from .runner import run_many_episodes
from .utility_agent import build_utility_agent


SUMMARY_FIELDS = (
    "completion_rate",
    "value_weighted_completion_rate",
    "uncapped_completion_rate",
    "failure_rate",
    "total_completed_value",
    "total_compute_cost",
    "avg_compute_cost_per_step",
    "total_utility",
    "steps_executed",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Reflex and Utility-Based baselines under matched seeds."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Path to a YAML config (defaults to config/default.yaml).",
    )
    parser.add_argument(
        "--seed-group",
        choices=("train", "test", "midterm_baseline", "custom"),
        default="midterm_baseline",
        help="Which seed list from the config to use.",
    )
    parser.add_argument(
        "--seeds",
        nargs="*",
        type=int,
        default=None,
        help="Explicit seed list; required when --seed-group custom.",
    )
    parser.add_argument(
        "--run-id",
        type=str,
        default=None,
        help="Override the auto-generated run id.",
    )
    parser.add_argument(
        "--out-root",
        type=Path,
        default=None,
        help="Override config.experiment.out_root.",
    )
    return parser.parse_args()


def resolve_seeds(cfg: RunConfig, args: argparse.Namespace) -> List[int]:
    if args.seed_group == "custom":
        if not args.seeds:
            raise ValueError("--seeds is required when --seed-group=custom")
        return list(args.seeds)
    if args.seed_group == "train":
        return list(cfg.seeds.train)
    if args.seed_group == "test":
        return list(cfg.seeds.test)
    return list(cfg.seeds.midterm_baseline)


def summarize_agent(metrics: Sequence[EpisodeMetrics]) -> Dict[str, Any]:
    if not metrics:
        raise ValueError("no metrics provided")
    summary: Dict[str, Any] = {
        "agent_name": metrics[0].agent_name,
        "num_runs": len(metrics),
        "num_jobs": metrics[0].num_jobs,
    }
    for field_name in SUMMARY_FIELDS:
        values = [float(getattr(metric, field_name)) for metric in metrics]
        summary[f"mean_{field_name}"] = round(statistics.fmean(values), 4)
        summary[f"std_{field_name}"] = (
            round(statistics.pstdev(values), 4) if len(values) > 1 else 0.0
        )
    summary["all_jobs_completed_runs"] = sum(1 for metric in metrics if metric.completed_all_jobs)
    summary["hit_step_budget_runs"] = sum(1 for metric in metrics if metric.hit_step_budget)
    summary["mean_completed_jobs"] = round(
        statistics.fmean(metric.completed_jobs for metric in metrics),
        4,
    )
    summary["mean_failed_jobs"] = round(
        statistics.fmean(metric.failed_jobs for metric in metrics),
        4,
    )
    return summary


def head_to_head(
    reflex_metrics: Sequence[EpisodeMetrics],
    utility_metrics: Sequence[EpisodeMetrics],
) -> Dict[str, int]:
    if len(reflex_metrics) != len(utility_metrics):
        raise ValueError("matched-seed comparison requires equal run counts")
    utility_wins = reflex_wins = ties = 0
    utility_better_completion = utility_lower_cost = 0
    for r, u in zip(reflex_metrics, utility_metrics):
        if u.total_utility > r.total_utility:
            utility_wins += 1
        elif u.total_utility < r.total_utility:
            reflex_wins += 1
        else:
            ties += 1
        if u.completion_rate > r.completion_rate:
            utility_better_completion += 1
        if u.total_compute_cost < r.total_compute_cost:
            utility_lower_cost += 1
    return {
        "utility_total_utility_wins": utility_wins,
        "reflex_total_utility_wins": reflex_wins,
        "ties_total_utility": ties,
        "utility_better_completion_runs": utility_better_completion,
        "utility_lower_cost_runs": utility_lower_cost,
        "num_head_to_head_runs": len(reflex_metrics),
    }


def render_markdown_report(
    cfg: RunConfig,
    seeds: Sequence[int],
    reflex_summary: Dict[str, Any],
    utility_summary: Dict[str, Any],
    comparison: Dict[str, int],
) -> str:
    u = cfg.utility
    headers = [
        "Agent",
        "Runs",
        "Mean Total Utility",
        "Mean Completion Rate",
        "Mean Value-Weighted Completion",
        "Mean Uncapped Completion",
        "Mean Compute Cost",
        "Mean Failure Rate",
    ]
    rows = []
    for summary in (reflex_summary, utility_summary):
        rows.append(
            "| "
            + " | ".join(
                [
                    str(summary["agent_name"]),
                    str(summary["num_runs"]),
                    f"{summary['mean_total_utility']:.4f}",
                    f"{summary['mean_completion_rate']:.4f}",
                    f"{summary['mean_value_weighted_completion_rate']:.4f}",
                    f"{summary['mean_uncapped_completion_rate']:.4f}",
                    f"{summary['mean_total_compute_cost']:.4f}",
                    f"{summary['mean_failure_rate']:.4f}",
                ]
            )
            + " |"
        )
    table = "\n".join(
        [
            "| " + " | ".join(headers) + " |",
            "|" + "|".join(["---"] * len(headers)) + "|",
            *rows,
        ]
    )
    return "\n".join(
        [
            "# Baseline comparison",
            "",
            f"- Config: `{cfg.meta.get('config_name', '?')}` (hash `{cfg.config_hash()[:12]}`)",
            f"- Utility weights: alpha={u.alpha}, beta={u.beta}, gamma={u.gamma}",
            f"- Seeds ({len(seeds)}): {list(seeds)}",
            f"- Jobs per run: {cfg.experiment.num_jobs}",
            f"- Max steps (capped): {cfg.experiment.max_steps}",
            f"- Uncapped max steps: {cfg.experiment.uncapped_max_steps}",
            "",
            "## Summary",
            table,
            "",
            "## Head-to-head",
            f"- Utility wins on total utility: "
            f"{comparison['utility_total_utility_wins']}/{comparison['num_head_to_head_runs']}",
            f"- Reflex wins on total utility:  "
            f"{comparison['reflex_total_utility_wins']}/{comparison['num_head_to_head_runs']}",
            f"- Ties: {comparison['ties_total_utility']}",
            f"- Utility better completion: "
            f"{comparison['utility_better_completion_runs']}/{comparison['num_head_to_head_runs']}",
            f"- Utility lower cost: "
            f"{comparison['utility_lower_cost_runs']}/{comparison['num_head_to_head_runs']}",
            "",
        ]
    )


def run_comparison(cfg: RunConfig, seeds: Sequence[int]) -> Dict[str, Any]:
    reflex_metrics = run_many_episodes(cfg=cfg, agent_factory=build_reflex_agent, seeds=seeds)
    utility_metrics = run_many_episodes(cfg=cfg, agent_factory=build_utility_agent, seeds=seeds)
    all_rows = [metric.as_dict() for metric in [*reflex_metrics, *utility_metrics]]
    reflex_summary = summarize_agent(reflex_metrics)
    utility_summary = summarize_agent(utility_metrics)
    comparison = head_to_head(reflex_metrics, utility_metrics)
    return {
        "reflex_metrics": reflex_metrics,
        "utility_metrics": utility_metrics,
        "all_rows": all_rows,
        "reflex_summary": reflex_summary,
        "utility_summary": utility_summary,
        "comparison": comparison,
    }


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config) if args.config else load_config()
    seeds = resolve_seeds(cfg, args)

    out_root = Path(args.out_root) if args.out_root else Path(cfg.experiment.out_root)
    run_id = args.run_id or cfg.experiment.run_id or generate_run_id("baseline")
    run_dir = ensure_run_dir(out_root, run_id)

    started = time.time()
    results = run_comparison(cfg=cfg, seeds=seeds)

    write_resolved_config(run_dir, cfg)
    write_metrics_csv(run_dir, results["all_rows"])
    summary_payload = {
        "seeds": list(seeds),
        "seed_group": args.seed_group,
        "reflex_summary": results["reflex_summary"],
        "utility_summary": results["utility_summary"],
        "head_to_head": results["comparison"],
    }
    write_json(run_dir, "summary.json", summary_payload)
    write_text(
        run_dir,
        "report.md",
        render_markdown_report(
            cfg=cfg,
            seeds=seeds,
            reflex_summary=results["reflex_summary"],
            utility_summary=results["utility_summary"],
            comparison=results["comparison"],
        ),
    )
    write_manifest(
        run_dir=run_dir,
        cfg=cfg,
        seeds=seeds,
        extra={"entrypoint": "compare_baselines", "seed_group": args.seed_group},
        wall_clock_start=started,
    )

    print(f"Wrote baseline run to: {run_dir}")
    print(f"  config hash: {cfg.config_hash()[:12]}")
    print(f"  seeds:       {list(seeds)}")


if __name__ == "__main__":
    main()
