from __future__ import annotations

"""Phase-6 V1 vs Phase-5 RL paired comparison.

For each V1 seed, pair its per-seed held-out metrics against the
matching Phase-5 RL seed (e.g. rl_v1_seed7 vs rl_seed7) and run the
same paired Wilcoxon signed-rank test the Phase-5 aggregate used.
"""

import argparse
import csv
from pathlib import Path
from typing import Dict, List

import numpy as np
from scipy.stats import wilcoxon


METRICS = (
    "total_utility",
    "completion_rate",
    "value_weighted_completion_rate",
    "uncapped_completion_rate",
    "failure_rate",
    "total_completed_value",
    "total_compute_cost",
    "avg_compute_cost_per_step",
)

SEED_PAIRS = (
    ("rl_v1_seed7", "rl_seed7"),
    ("rl_v1_seed11", "rl_seed11"),
    ("rl_v1_seed13", "rl_seed13"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--v1-dir", type=Path, required=True)
    parser.add_argument("--phase5-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=None)
    return parser.parse_args()


def _read(path: Path) -> List[Dict]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _per_seed_values(
    rows: List[Dict], agent: str, metric: str
) -> np.ndarray:
    agent_rows = [r for r in rows if r["agent_name"] == agent]
    return np.array(
        [float(r[metric]) for r in sorted(agent_rows, key=lambda r: int(r["seed"]))],
        dtype=float,
    )


def main() -> None:
    args = parse_args()
    v1_rows = _read(args.v1_dir / "metrics.csv")
    phase5_rows = _read(args.phase5_dir / "metrics.csv")
    out_path = args.out or (args.v1_dir / "phase6_v1_vs_phase5.csv")

    result_rows: List[Dict] = []
    for v1_name, ph5_name in SEED_PAIRS:
        for metric in METRICS:
            lhs = _per_seed_values(v1_rows, v1_name, metric)
            rhs = _per_seed_values(phase5_rows, ph5_name, metric)
            if lhs.size == 0 or rhs.size == 0:
                continue
            diff = lhs - rhs
            nonzero = int((diff != 0).sum())
            if nonzero == 0:
                stat = float("nan")
                pval = float("nan")
            else:
                test = wilcoxon(
                    diff[diff != 0],
                    alternative="two-sided",
                    zero_method="wilcox",
                    method="auto",
                )
                stat = float(test.statistic)
                pval = float(test.pvalue)
            result_rows.append(
                {
                    "lhs": v1_name,
                    "rhs": ph5_name,
                    "metric": metric,
                    "n_pairs": int(lhs.size),
                    "n_nonzero": nonzero,
                    "mean_delta": round(float(diff.mean()), 6),
                    "median_delta": round(float(np.median(diff)), 6),
                    "statistic": round(stat, 6) if stat == stat else float("nan"),
                    "p_value": pval,
                }
            )

    with out_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "lhs",
                "rhs",
                "metric",
                "n_pairs",
                "n_nonzero",
                "mean_delta",
                "median_delta",
                "statistic",
                "p_value",
            ],
        )
        writer.writeheader()
        writer.writerows(result_rows)
    print(f"Wrote {out_path} ({len(result_rows)} rows)")
    print()
    print("Headline (total_utility):")
    for row in result_rows:
        if row["metric"] == "total_utility":
            print(
                f"  {row['lhs']:<15s} vs {row['rhs']:<12s}  "
                f"mean_delta={row['mean_delta']:+7.4f}  p={row['p_value']:.4g}  "
                f"n_nonzero={row['n_nonzero']}/{row['n_pairs']}"
            )


if __name__ == "__main__":
    main()
