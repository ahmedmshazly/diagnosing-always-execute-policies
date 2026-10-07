from __future__ import annotations

"""Aggregate statistics for a multi-seed baseline comparison.

Consumes a per-seed ``metrics.csv`` written by
``src.compare_baselines`` and produces:

- ``aggregate.csv`` — one row per (agent, metric) with n, mean, std, 95%
  bootstrap CI (10k resamples), and a paired-delta row.
- ``wilcoxon.csv`` — one row per metric with the paired Wilcoxon signed-rank
  statistic W and two-sided p-value on the per-seed deltas
  (Utility − Reflex), plus the n_nonzero used by the test.
- ``aggregate.json`` — structured payload of the same data.
- ``aggregate.md`` — human-readable summary table.

Every random draw goes through a named ``numpy.random.Generator``; the
bootstrap seed is part of the aggregate manifest so the figures can be
regenerated bit-for-bit.
"""

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np
from scipy.stats import wilcoxon


METRICS: Tuple[str, ...] = (
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

REFLEX_NAME = "Reflex Agent"
UTILITY_NAME = "Utility-Based Agent (Non-Learning Baseline)"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run-dir",
        type=Path,
        required=True,
        help="Run directory containing metrics.csv (e.g. results/phase2_50seeds).",
    )
    parser.add_argument(
        "--bootstrap-resamples",
        type=int,
        default=10_000,
        help="Number of bootstrap resamples for 95%% CIs.",
    )
    parser.add_argument(
        "--bootstrap-seed",
        type=int,
        default=20260418,
        help="Seed for the bootstrap RNG (recorded in aggregate.json).",
    )
    return parser.parse_args()


def _read_metrics_csv(path: Path) -> List[Dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _rows_by_agent(
    rows: Sequence[Mapping[str, str]],
) -> Dict[str, Dict[int, Mapping[str, str]]]:
    grouped: Dict[str, Dict[int, Mapping[str, str]]] = {}
    for row in rows:
        grouped.setdefault(row["agent_name"], {})[int(row["seed"])] = row
    return grouped


def bootstrap_ci(
    values: np.ndarray,
    rng: np.random.Generator,
    num_resamples: int = 10_000,
    confidence: float = 0.95,
) -> Tuple[float, float]:
    """Percentile bootstrap CI for the mean of ``values``.

    Returns ``(ci_low, ci_high)``. When ``values`` has fewer than two
    elements the CI collapses to ``(nan, nan)``.
    """
    if values.size < 2:
        return (float("nan"), float("nan"))
    n = values.size
    # Sample indices in one (num_resamples, n) draw for vectorised speed.
    idx = rng.integers(low=0, high=n, size=(num_resamples, n))
    means = values[idx].mean(axis=1)
    alpha = (1.0 - confidence) / 2.0
    lo = float(np.quantile(means, alpha))
    hi = float(np.quantile(means, 1.0 - alpha))
    return (lo, hi)


def summarise(values: np.ndarray, rng: np.random.Generator, num_resamples: int) -> Dict[str, float]:
    mean = float(values.mean()) if values.size else float("nan")
    std = float(values.std(ddof=1)) if values.size >= 2 else 0.0
    ci_low, ci_high = bootstrap_ci(values, rng=rng, num_resamples=num_resamples)
    return {
        "n": int(values.size),
        "mean": round(mean, 6),
        "std": round(std, 6),
        "ci95_low": round(ci_low, 6),
        "ci95_high": round(ci_high, 6),
    }


def paired_wilcoxon(
    reflex_values: np.ndarray, utility_values: np.ndarray
) -> Dict[str, float]:
    """Two-sided paired Wilcoxon signed-rank on (utility - reflex).

    Returns ``statistic``, ``p_value``, ``n_nonzero`` (the number of
    non-zero deltas actually entering the test; zeros are dropped by the
    default ``zero_method='wilcox'``).
    """
    deltas = utility_values - reflex_values
    nonzero_mask = deltas != 0.0
    n_nonzero = int(nonzero_mask.sum())
    if n_nonzero < 1:
        return {
            "statistic": float("nan"),
            "p_value": float("nan"),
            "n_nonzero": 0,
            "mean_delta": float(deltas.mean()) if deltas.size else float("nan"),
            "median_delta": float(np.median(deltas)) if deltas.size else float("nan"),
        }
    result = wilcoxon(
        deltas[nonzero_mask],
        alternative="two-sided",
        zero_method="wilcox",
        method="auto",
    )
    return {
        "statistic": float(result.statistic),
        "p_value": float(result.pvalue),
        "n_nonzero": n_nonzero,
        "mean_delta": float(deltas.mean()),
        "median_delta": float(np.median(deltas)),
    }


def _extract_paired(
    grouped: Mapping[str, Mapping[int, Mapping[str, str]]],
    metric: str,
) -> Tuple[np.ndarray, np.ndarray, List[int]]:
    if REFLEX_NAME not in grouped or UTILITY_NAME not in grouped:
        raise ValueError(
            f"metrics.csv must contain rows for both {REFLEX_NAME!r} and {UTILITY_NAME!r}"
        )
    seeds = sorted(set(grouped[REFLEX_NAME]) & set(grouped[UTILITY_NAME]))
    reflex_arr = np.array(
        [float(grouped[REFLEX_NAME][seed][metric]) for seed in seeds], dtype=float
    )
    utility_arr = np.array(
        [float(grouped[UTILITY_NAME][seed][metric]) for seed in seeds], dtype=float
    )
    return reflex_arr, utility_arr, seeds


def build_aggregate_payload(
    run_dir: Path,
    num_resamples: int,
    bootstrap_seed: int,
) -> Dict[str, Any]:
    rows = _read_metrics_csv(run_dir / "metrics.csv")
    grouped = _rows_by_agent(rows)
    rng = np.random.default_rng(bootstrap_seed)

    agent_summaries: Dict[str, Dict[str, Dict[str, float]]] = {REFLEX_NAME: {}, UTILITY_NAME: {}}
    delta_summaries: Dict[str, Dict[str, float]] = {}
    wilcoxon_summaries: Dict[str, Dict[str, float]] = {}
    seeds_used: List[int] = []

    for metric in METRICS:
        reflex_arr, utility_arr, seeds = _extract_paired(grouped, metric)
        if not seeds_used:
            seeds_used = seeds
        agent_summaries[REFLEX_NAME][metric] = summarise(reflex_arr, rng=rng, num_resamples=num_resamples)
        agent_summaries[UTILITY_NAME][metric] = summarise(utility_arr, rng=rng, num_resamples=num_resamples)
        delta_summaries[metric] = summarise(utility_arr - reflex_arr, rng=rng, num_resamples=num_resamples)
        wilcoxon_summaries[metric] = paired_wilcoxon(reflex_arr, utility_arr)

    return {
        "run_dir": str(run_dir),
        "num_seeds": len(seeds_used),
        "seeds": seeds_used,
        "bootstrap": {"resamples": num_resamples, "seed": bootstrap_seed, "confidence": 0.95},
        "metrics": list(METRICS),
        "per_agent": agent_summaries,
        "deltas_utility_minus_reflex": delta_summaries,
        "paired_wilcoxon": wilcoxon_summaries,
    }


def _write_aggregate_csv(path: Path, payload: Mapping[str, Any]) -> None:
    fieldnames = [
        "scope",
        "agent",
        "metric",
        "n",
        "mean",
        "std",
        "ci95_low",
        "ci95_high",
    ]
    rows: List[Dict[str, Any]] = []
    for agent, metrics_map in payload["per_agent"].items():
        for metric, stats in metrics_map.items():
            rows.append(
                {
                    "scope": "agent",
                    "agent": agent,
                    "metric": metric,
                    **stats,
                }
            )
    for metric, stats in payload["deltas_utility_minus_reflex"].items():
        rows.append(
            {
                "scope": "delta",
                "agent": "Utility - Reflex",
                "metric": metric,
                **stats,
            }
        )
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_wilcoxon_csv(path: Path, payload: Mapping[str, Any]) -> None:
    fieldnames = ["metric", "n_pairs", "n_nonzero", "mean_delta", "median_delta", "statistic", "p_value"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for metric, stats in payload["paired_wilcoxon"].items():
            writer.writerow(
                {
                    "metric": metric,
                    "n_pairs": payload["num_seeds"],
                    "n_nonzero": stats["n_nonzero"],
                    "mean_delta": round(stats["mean_delta"], 6),
                    "median_delta": round(stats["median_delta"], 6),
                    "statistic": round(stats["statistic"], 6),
                    "p_value": stats["p_value"],
                }
            )


def _write_markdown(path: Path, payload: Mapping[str, Any]) -> None:
    lines: List[str] = []
    lines.append(f"# Aggregate statistics — {payload['run_dir']}")
    lines.append("")
    lines.append(
        f"- Seeds: **{payload['num_seeds']}** ({payload['seeds'][0]}..{payload['seeds'][-1]})"
    )
    lines.append(
        f"- Bootstrap: {payload['bootstrap']['resamples']:,} resamples, "
        f"seed={payload['bootstrap']['seed']}, "
        f"confidence={payload['bootstrap']['confidence']:.0%}"
    )
    lines.append("")

    lines.append("## Per-agent means with 95% bootstrap CI")
    headers = ["Metric", "Reflex mean (95% CI)", "Utility mean (95% CI)"]
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("|" + "|".join(["---"] * len(headers)) + "|")
    for metric in payload["metrics"]:
        reflex = payload["per_agent"][REFLEX_NAME][metric]
        utility = payload["per_agent"][UTILITY_NAME][metric]
        lines.append(
            "| "
            + " | ".join(
                [
                    f"`{metric}`",
                    f"{reflex['mean']:.4f}  ({reflex['ci95_low']:.4f}, {reflex['ci95_high']:.4f})",
                    f"{utility['mean']:.4f}  ({utility['ci95_low']:.4f}, {utility['ci95_high']:.4f})",
                ]
            )
            + " |"
        )
    lines.append("")

    lines.append("## Paired Wilcoxon signed-rank on Utility − Reflex deltas")
    headers = ["Metric", "mean Δ (95% CI)", "median Δ", "W", "p-value", "n_nonzero"]
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("|" + "|".join(["---"] * len(headers)) + "|")
    for metric in payload["metrics"]:
        delta = payload["deltas_utility_minus_reflex"][metric]
        wil = payload["paired_wilcoxon"][metric]
        lines.append(
            "| "
            + " | ".join(
                [
                    f"`{metric}`",
                    f"{delta['mean']:.4f} ({delta['ci95_low']:.4f}, {delta['ci95_high']:.4f})",
                    f"{wil['median_delta']:.4f}",
                    f"{wil['statistic']:.2f}",
                    f"{wil['p_value']:.4g}",
                    f"{wil['n_nonzero']}/{payload['num_seeds']}",
                ]
            )
            + " |"
        )
    lines.append("")

    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    run_dir = args.run_dir
    if not (run_dir / "metrics.csv").exists():
        raise FileNotFoundError(f"{run_dir}/metrics.csv not found")

    payload = build_aggregate_payload(
        run_dir=run_dir,
        num_resamples=args.bootstrap_resamples,
        bootstrap_seed=args.bootstrap_seed,
    )

    (run_dir / "aggregate.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    _write_aggregate_csv(run_dir / "aggregate.csv", payload)
    _write_wilcoxon_csv(run_dir / "wilcoxon.csv", payload)
    _write_markdown(run_dir / "aggregate.md", payload)
    print(f"Wrote aggregate.{{json,csv,md}} and wilcoxon.csv to {run_dir}")


if __name__ == "__main__":
    main()
