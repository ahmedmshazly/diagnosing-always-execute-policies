from __future__ import annotations

"""Holm-Bonferroni adjustment of the taxonomy Wilcoxon p-values.

Treats every non-baseline row of taxonomy.csv (hand references and all learned
policies, across all environments) as ONE family of tests vs always-execute,
which is the most conservative grouping. Writes taxonomy_holm.csv with an extra
`p_holm` column and prints the rows whose significance changes at alpha=0.05.
"""

import argparse
import csv
import math
from pathlib import Path


def holm(pvals):
    """Return Holm step-down adjusted p-values (monotone, capped at 1)."""
    order = sorted(range(len(pvals)), key=lambda i: pvals[i])
    m = len(pvals)
    adj = [0.0] * m
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (m - rank) * pvals[i]))
        adj[i] = running
    return adj


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", type=Path, default=Path("results/hardening/taxonomy.csv"))
    ap.add_argument("--out", type=Path, default=Path("results/hardening/taxonomy_holm.csv"))
    args = ap.parse_args()

    with args.csv.open() as fh:
        rows = list(csv.DictReader(fh))
    tested = [r for r in rows if not math.isnan(float(r["p"]))]
    for r, adj in zip(tested, holm([float(r["p"]) for r in tested])):
        r["p_holm"] = adj
    with args.out.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]) + ["p_holm"], restval="nan")
        w.writeheader()
        w.writerows(rows)

    print(f"family size m={len(tested)}")
    print("significant before, not after Holm (alpha=0.05):")
    flips = [r for r in tested if float(r["p"]) < 0.05 <= r["p_holm"]]
    for r in flips:
        print(f"  {r['env']:22s} {r['policy']:28s} delta={r['delta_vs_ae']:>8s} "
              f"p={float(r['p']):.4f} p_holm={r['p_holm']:.4f}")
    if not flips:
        print("  none")


if __name__ == "__main__":
    main()
