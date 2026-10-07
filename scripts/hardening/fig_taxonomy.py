from __future__ import annotations

"""Taxonomy figure: Δ-utility vs always-execute for the hand reference,
REINFORCE, and PPO across the five environments. Reads results/hardening/
taxonomy.csv (emitted by eval_taxonomy.py) so it regenerates from data.

Bars are the mean over initialisation seeds, whiskers the range, and dots the
individual seeds. REINFORCE sits at ~0 wherever the lever is a modest scaling
improvement (tight, heavy-tail) and rises only at the large caution lever
(cascade); PPO rises wherever a lever exists, and both weaken under the
original (broken) reward.
"""

from pathlib import Path
import csv
from collections import defaultdict

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ENV_ORDER = ["default", "env_tight_fixedrisk", "env_heavytail_tight", "env_cascade", "env_cascade_broken"]
ENV_LABEL = {
    "default": "benign",
    "env_tight_fixedrisk": "tight",
    "env_heavytail_tight": "heavy-tail",
    "env_cascade": "cascade\n(corrected reward)",
    "env_cascade_broken": "cascade\n(original reward)",
}


def main() -> None:
    rows = list(csv.DictReader(open("results/hardening/taxonomy.csv")))
    # env -> {hand: delta, REINFORCE: [deltas], PPO: [deltas]}
    data = defaultdict(lambda: {"hand": 0.0, "REINFORCE": [], "PPO": []})
    for r in rows:
        env, algo, d = r["env"], r["algo"], float(r["delta_vs_ae"])
        if algo == "hand": data[env]["hand"] = d
        elif algo in ("REINFORCE", "PPO"): data[env][algo].append(d)

    envs = [e for e in ENV_ORDER if e in data]
    x = np.arange(len(envs)); w = 0.26
    fig, ax = plt.subplots(figsize=(11, 4.6))

    def bars(key, off, color, label):
        means, lo, hi = [], [], []
        for e in envs:
            vals = data[e][key] if key != "hand" else [data[e]["hand"]]
            vals = [v for v in vals if v is not None]
            if vals:
                m = float(np.mean(vals)); means.append(m)
                lo.append(m - min(vals)); hi.append(max(vals) - m)
            else:
                means.append(np.nan); lo.append(0); hi.append(0)
        ax.bar(x + off, means, w, yerr=[lo, hi], capsize=3, color=color, label=label,
               alpha=0.85, error_kw={"elinewidth": 1})
        if key != "hand":
            for xi, e in zip(x + off, envs):
                ax.scatter([xi] * len(data[e][key]), data[e][key], s=12, color="k", zorder=3)
        for xi, m in zip(x + off, means):
            if not np.isnan(m):
                ax.annotate(f"{m:+.0f}", (xi, m), ha="center",
                            va="bottom" if m >= 0 else "top", fontsize=8)

    bars("hand", -w, "#9467bd", "hand-crafted reference")
    bars("REINFORCE", 0.0, "#d62728", "REINFORCE (vanilla PG)")
    bars("PPO", w, "#2ca02c", "PPO")

    ax.axhline(0, color="k", lw=1)
    ax.set_xticks(x); ax.set_xticklabels([ENV_LABEL[e] for e in envs], fontsize=9)
    ax.set_ylabel("Δ total utility vs always-execute\n(0 = the trivial policy)")
    ax.legend(loc="upper left")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    out = Path("results/hardening/fig_taxonomy.png")
    fig.savefig(out, dpi=130)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
