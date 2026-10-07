from __future__ import annotations

"""Ablation summary bar chart: per-variant Hamming distance vs reference.

Reads ``hamming_vs_reference.csv`` written by
``scripts.ablation_phase3_weights`` and produces a horizontal bar chart
of per-variant action-mismatch fractions. Ceremonial variants sit at 0;
binary-gate-flip variants cluster at the same high fraction.
"""

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_dir = args.run_dir
    csv_path = run_dir / "hamming_vs_reference.csv"
    rows = list(csv.DictReader(csv_path.open(encoding="utf-8")))
    # Sort by fraction ascending so ceremonial variants are at the top.
    rows_sorted = sorted(rows, key=lambda r: float(r["hamming_fraction"]))
    names = [r["variant"] for r in rows_sorted]
    fractions = [float(r["hamming_fraction"]) for r in rows_sorted]
    colors = []
    for f in fractions:
        if f < 0.001:
            colors.append("#2b8a3e")   # green: ceremonial
        elif f > 0.5:
            colors.append("#c92a2a")   # red: binary gate flip
        else:
            colors.append("#e67700")   # orange: operative

    fig, ax = plt.subplots(figsize=(11, 7.5), constrained_layout=True)
    y = np.arange(len(names))
    ax.barh(y, fractions, color=colors, edgecolor="black", linewidth=0.5)
    for i, v in enumerate(fractions):
        ax.text(v + 0.01, i, f"{v:.2%}", va="center", fontsize=8)
    ax.set_yticks(y)
    ax.set_yticklabels(names, fontsize=9)
    ax.set_xlabel("Fraction of action-steps differing from the Phase-1 reference trace")
    ax.axvline(0.0, color="grey", linewidth=0.6)
    ax.set_xlim(-0.02, 1.0)
    ax.set_title(
        "Ablation: which Utility-Based agent knobs actually change behaviour?\n"
        "20 variants x 20 seeds (100..119) x 300 steps = 6000 decisions per variant",
        fontsize=12,
        fontweight="bold",
    )
    legend_handles = [
        plt.Rectangle((0, 0), 1, 1, color="#2b8a3e", ec="black"),
        plt.Rectangle((0, 0), 1, 1, color="#e67700", ec="black"),
        plt.Rectangle((0, 0), 1, 1, color="#c92a2a", ec="black"),
    ]
    ax.legend(
        legend_handles,
        ["Ceremonial (≈0% mismatch)", "Operative (≈10–30%)", "Binary-gate flip (≈88%)"],
        loc="lower right",
        fontsize=9,
    )
    ax.grid(axis="x", linestyle=":", alpha=0.3)

    out_path = args.out or (run_dir / "ablation_summary.png")
    fig.savefig(out_path, dpi=170)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
