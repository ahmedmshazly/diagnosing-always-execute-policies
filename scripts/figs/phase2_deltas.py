from __future__ import annotations

"""Per-seed Utility − Reflex delta strip plot.

Reads a ``metrics.csv`` written by ``src.compare_baselines``, computes the
paired per-seed delta (Utility − Reflex) for every metric, and saves a
single multi-panel strip/scatter figure to
``<run_dir>/phase2_deltas.png`` by default.

Each panel shows:
- one dot per seed (Utility − Reflex) with a small horizontal jitter,
- a horizontal reference line at zero,
- the mean delta (red dashed) and its 95% bootstrap CI (shaded),
- the paired Wilcoxon p-value in the panel title.
"""

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


METRICS = (
    "total_utility",
    "completion_rate",
    "value_weighted_completion_rate",
    "uncapped_completion_rate",
    "total_completed_value",
    "total_compute_cost",
    "failure_rate",
    "avg_compute_cost_per_step",
    "steps_executed",
)

REFLEX_NAME = "Reflex Agent"
UTILITY_NAME = "Utility-Based Agent (Non-Learning Baseline)"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output PNG (defaults to <run-dir>/phase2_deltas.png).",
    )
    parser.add_argument(
        "--jitter-seed",
        type=int,
        default=42,
        help="RNG seed used for the horizontal jitter (recorded in the figure).",
    )
    return parser.parse_args()


def _read_metrics(path: Path) -> List[Dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _paired_arrays(
    rows: Sequence[Mapping[str, str]], metric: str
) -> tuple[list[int], np.ndarray]:
    grouped: Dict[str, Dict[int, float]] = {REFLEX_NAME: {}, UTILITY_NAME: {}}
    for row in rows:
        if row["agent_name"] in grouped:
            grouped[row["agent_name"]][int(row["seed"])] = float(row[metric])
    seeds = sorted(set(grouped[REFLEX_NAME]) & set(grouped[UTILITY_NAME]))
    deltas = np.array(
        [grouped[UTILITY_NAME][seed] - grouped[REFLEX_NAME][seed] for seed in seeds],
        dtype=float,
    )
    return seeds, deltas


def _load_aggregate(run_dir: Path) -> Mapping[str, object]:
    aggregate_json = run_dir / "aggregate.json"
    if aggregate_json.exists():
        return json.loads(aggregate_json.read_text(encoding="utf-8"))
    return {}


def main() -> None:
    args = parse_args()
    run_dir = args.run_dir
    out_path = args.out or (run_dir / "phase2_deltas.png")
    rows = _read_metrics(run_dir / "metrics.csv")
    aggregate = _load_aggregate(run_dir)

    num_panels = len(METRICS)
    ncols = 3
    nrows = (num_panels + ncols - 1) // ncols
    fig, axes = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=(4.5 * ncols, 3.2 * nrows),
        constrained_layout=True,
    )
    axes = np.asarray(axes).ravel()
    rng = np.random.default_rng(args.jitter_seed)

    for idx, metric in enumerate(METRICS):
        ax = axes[idx]
        seeds, deltas = _paired_arrays(rows, metric)
        n = len(seeds)
        if n == 0:
            ax.set_title(f"{metric}\n(no paired data)")
            ax.axis("off")
            continue

        jitter = rng.uniform(-0.12, 0.12, size=n)
        x = np.full(n, 0.0) + jitter
        colours = np.where(deltas >= 0.0, "#2b8a3e", "#c92a2a")
        ax.scatter(x, deltas, c=colours, s=26, alpha=0.85, edgecolor="black", linewidths=0.3)
        ax.axhline(0.0, color="gray", linewidth=1.0, linestyle="-", alpha=0.6)

        mean_delta = float(np.mean(deltas))
        ci_low = ci_high = None
        p_value = None
        if aggregate:
            deltas_aggregate = aggregate.get("deltas_utility_minus_reflex", {}).get(metric)
            wilcoxon_aggregate = aggregate.get("paired_wilcoxon", {}).get(metric)
            if deltas_aggregate:
                mean_delta = float(deltas_aggregate["mean"])
                ci_low = float(deltas_aggregate["ci95_low"])
                ci_high = float(deltas_aggregate["ci95_high"])
            if wilcoxon_aggregate:
                p_value = float(wilcoxon_aggregate["p_value"])

        ax.axhline(mean_delta, color="#b03060", linewidth=1.2, linestyle="--", alpha=0.9, label=f"mean Δ = {mean_delta:.3f}")
        if ci_low is not None and ci_high is not None:
            ax.axhspan(ci_low, ci_high, color="#b03060", alpha=0.12, label="95% bootstrap CI")

        title = metric
        if p_value is not None:
            title += f"\np(Wilcoxon)={p_value:.3g}  (n={n})"
        else:
            title += f"\nmean Δ = {mean_delta:.3f}  (n={n})"
        ax.set_title(title, fontsize=10)
        ax.set_xticks([])
        ax.set_xlim(-0.5, 0.5)
        ax.set_ylabel("Utility − Reflex")
        ax.grid(True, alpha=0.25, linestyle=":")
        ax.legend(loc="best", fontsize=7)

    # Hide unused panels if METRICS is not a multiple of ncols.
    for idx in range(num_panels, len(axes)):
        axes[idx].axis("off")

    fig.suptitle(
        f"Per-seed Utility − Reflex deltas ({Path(run_dir).name}, n={n} seeds)",
        fontsize=13,
        fontweight="bold",
    )
    fig.savefig(out_path, dpi=170)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
