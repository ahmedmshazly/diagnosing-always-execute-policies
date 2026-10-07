from __future__ import annotations

"""Diff two ``metrics.csv`` files by (seed, agent).

Used by ``make phase2-sanity`` to verify that the new pipeline reproduces
the midterm's per-seed numbers within rounding. Exits non-zero iff any
``total_utility`` or ``total_compute_cost`` row differs by more than the
tolerance.
"""

import argparse
import csv
from pathlib import Path
from typing import Dict, Mapping, Tuple


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument(
        "--tolerance",
        type=float,
        default=1e-6,
        help="Max absolute difference allowed in total_utility / total_compute_cost.",
    )
    return parser.parse_args()


def _load_rows(path: Path) -> Dict[Tuple[int, str], Mapping[str, str]]:
    rows: Dict[Tuple[int, str], Mapping[str, str]] = {}
    with (path / "metrics.csv").open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            rows[(int(row["seed"]), row["agent_name"])] = row
    return rows


def main() -> int:
    args = parse_args()
    reference = _load_rows(args.reference)
    candidate = _load_rows(args.candidate)
    common = sorted(set(reference) & set(candidate))
    missing = sorted(set(reference) - set(candidate))
    extra = sorted(set(candidate) - set(reference))

    mismatches: list[str] = []
    worst: Dict[str, float] = {"total_utility": 0.0, "total_compute_cost": 0.0}

    for key in common:
        ref_row = reference[key]
        cand_row = candidate[key]
        for metric in ("total_utility", "total_compute_cost"):
            diff = abs(float(ref_row[metric]) - float(cand_row[metric]))
            worst[metric] = max(worst[metric], diff)
            if diff > args.tolerance:
                mismatches.append(
                    f"seed={key[0]:>3} agent={key[1]!r} metric={metric} "
                    f"ref={float(ref_row[metric]):.6f} cand={float(cand_row[metric]):.6f} diff={diff:.6f}"
                )

    print(f"Reference:  {args.reference}  ({len(reference)} rows)")
    print(f"Candidate:  {args.candidate}  ({len(candidate)} rows)")
    print(f"Compared:   {len(common)} rows")
    print(f"Tolerance:  {args.tolerance}")
    print(f"Max |Δ| total_utility:      {worst['total_utility']:.9f}")
    print(f"Max |Δ| total_compute_cost: {worst['total_compute_cost']:.9f}")
    if missing:
        print(f"Missing in candidate: {missing}")
    if extra:
        print(f"Extra in candidate:   {extra}")
    if mismatches:
        print("\nMISMATCHES:")
        for line in mismatches:
            print("  " + line)
        print(f"\nSanity check FAILED: {len(mismatches)} row(s) exceed tolerance.")
        return 1
    print("\nSanity check PASSED: all rows match within tolerance.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
