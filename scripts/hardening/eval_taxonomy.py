from __future__ import annotations

"""Evaluate saved policies on paired workload seeds using episode utility.

Each condition includes always-execute, a hand-written reference and three
checkpoints per learner. Training-record completeness is documented separately
in training_inventory.csv. Checkpoints are discovered from the patterns below;
this script performs evaluation only. Raw Wilcoxon comparisons are adjusted
over the full comparison family by scripts.sncs_report.
"""

import argparse
import glob
from pathlib import Path
from typing import List

import numpy as np
import torch
from scipy.stats import wilcoxon

torch.set_num_threads(1)

from src.config import load_config
from src.reflex_agent import build_reflex_agent
from src.rl.agent import RLPolicyAgent, load_policy
from src.runner import run_many_episodes
from src.sim_environment import ACTIONS, EpisodeState, WorkloadGenerator, advance_one_step, make_episode_rngs


class AlwaysExecute:
    name = "always_execute"
    def __init__(self, cfg): pass
    def choose_action(self, s): return "Execute_Ready_Job"


class ScaleWhenBlocked:
    name = "scale_when_blocked"
    def __init__(self, cfg): pass
    def choose_action(self, s):
        r = s.ready_tasks()
        if not r: return "Execute_Ready_Job"
        ac, ar = s.available_cpu(), s.available_ram()
        if any(t.cpu_demand <= ac and t.ram_demand <= ar for t in r): return "Execute_Ready_Job"
        return "Scale_Up" if s.cluster.scale_boost_remaining == 0 else "Execute_Ready_Job"


class Throttle:
    name = "throttle"
    def __init__(self, cfg, thr=0.4): self.thr = thr
    def choose_action(self, s):
        load = s.cpu_in_use() / max(s.cluster.cpu_capacity, 1)
        return "Defer_Job" if load > self.thr else "Execute_Ready_Job"


# (env config, hand-reference agent class, glob patterns for RL run dirs)
MATRIX = [
    ("config/default.yaml", ScaleWhenBlocked, {
        "REINFORCE": "results/phase5/rl_seed*",
        "PPO": "results/hardening/ppo_benign_seed*",
    }),
    ("config/env_tight_fixedrisk.yaml", ScaleWhenBlocked, {
        "REINFORCE": "results/hardening/rl_tight_fixedrisk_seed7 results/hardening/reinforce_tight_seed*",
        "PPO": "results/hardening/ppo_tight_seed*",
    }),
    ("config/env_cascade.yaml", Throttle, {
        "REINFORCE": "results/hardening/reinforce_cascade_seed7 results/hardening/reinforce_cascade_seed1*",
        "PPO": "results/hardening/ppo_cascade_seed*",
    }),
    ("config/env_cascade_broken.yaml", Throttle, {
        "REINFORCE": "results/hardening/reinforce_cascade_broken_seed*",
        "PPO": "results/hardening/ppo_cascade_broken_seed*",
    }),
    ("config/env_heavytail_tight.yaml", ScaleWhenBlocked, {
        "REINFORCE": "results/hardening/reinforce_heavytail_seed*",
        "PPO": "results/hardening/ppo_heavytail_seed*",
    }),
]


def _exec_frac(cfg, agent, seeds) -> float:
    tot = ex = 0
    for seed in seeds[:10]:  # histogram on a subsample for speed
        wr, er = make_episode_rngs(seed)
        st = WorkloadGenerator(rng=wr, cfg=cfg, num_jobs=cfg.experiment.num_jobs, seed_label=seed).generate_episode()
        while not st.all_done() and st.step < cfg.experiment.max_steps:
            a = agent.choose_action(st); advance_one_step(st, er, a)
            tot += 1; ex += (a == "Execute_Ready_Job")
    return ex / max(tot, 1)


def _metrics(cfg, factory, seeds):
    ms = run_many_episodes(cfg=cfg, agent_factory=factory, seeds=seeds, include_uncapped=False)
    return (np.array([m.total_utility for m in ms]),
            np.array([m.failure_rate for m in ms]),
            np.array([m.completion_rate for m in ms]))


def main() -> None:
    import csv as _csv
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", nargs="*", type=int, default=list(range(200, 250)))
    ap.add_argument("--csv", type=Path, default=Path("results/hardening/taxonomy.csv"))
    ap.add_argument("--seed-start", type=int, default=None)
    ap.add_argument("--seed-stop", type=int, default=None)
    ap.add_argument("--per-seed-csv", type=Path, default=None)
    ap.add_argument("--envs", nargs="*", default=None,
                    help="Only evaluate these env configs, e.g. env_cascade.yaml (default: all).")
    args = ap.parse_args()
    if args.seed_start is not None or args.seed_stop is not None:
        if args.seed_start is None or args.seed_stop is None or args.seed_stop <= args.seed_start:
            ap.error("--seed-start and --seed-stop must specify a non-empty range")
        args.seeds = list(range(args.seed_start, args.seed_stop))
    csv_rows: List[dict] = []
    per_seed_rows: List[dict] = []

    for env_cfg, hand_cls, rl_globs in MATRIX:
        if args.envs and Path(env_cfg).name not in args.envs:
            continue
        cfg = load_config(Path(env_cfg))
        seeds = args.seeds
        print(f"\n{'='*92}\nENV: {cfg.meta.get('config_name')}  ({env_cfg})  "
              f"seeds {seeds[0]}..{seeds[-1]} n={len(seeds)}")
        print(f"{'policy':<34}{'util':>9}{'95% seed range':>20}{'fail':>7}{'compl':>7}{'exec%':>7}{'ΔvsAE':>8}{'p':>9}")

        ae_u, ae_f, ae_c = _metrics(cfg, lambda c: AlwaysExecute(c), seeds)
        ae_x = _exec_frac(cfg, AlwaysExecute(cfg), seeds)
        env_name = cfg.meta.get("config_name")
        def row(name, u, f, c, x, base=None, algo=""):
            lo, hi = np.percentile(u, [2.5, 97.5])
            d = "" ; p = ""; dval = 0.0; pval = float("nan")
            if base is not None:
                dval = float(u.mean() - base.mean()); d = f"{dval:+.1f}"
                pval = 1.0 if np.array_equal(u, base) else float(wilcoxon(u, base).pvalue)
                p = f"{pval:.1e}"
            print(f"{name:<34}{u.mean():>9.1f}{('['+format(lo,'.1f')+','+format(hi,'.1f')+']'):>20}"
                  f"{f.mean():>7.3f}{c.mean():>7.3f}{x:>7.2f}{d:>8}{p:>9}")
            csv_rows.append({"env": env_name, "policy": name, "algo": algo,
                             "util": round(float(u.mean()), 3), "delta_vs_ae": round(dval, 3),
                             "p": pval, "exec_frac": round(float(x), 4),
                             "fail": round(float(f.mean()), 4), "compl": round(float(c.mean()), 4)})
            for seed, utility, failure, completion in zip(seeds, u, f, c):
                per_seed_rows.append({"env": env_name, "policy": name, "algo": algo,
                                      "seed": seed, "utility": float(utility),
                                      "failure_rate": float(failure), "completion_rate": float(completion)})
        row("always_execute", ae_u, ae_f, ae_c, ae_x, algo="baseline")
        hand = hand_cls(cfg)
        h_u, h_f, h_c = _metrics(cfg, lambda c: hand_cls(c), seeds)
        row(f"{hand.name} (hand)", h_u, h_f, h_c, _exec_frac(cfg, hand, seeds), ae_u, algo="hand")

        for algo, pattern in rl_globs.items():
            dirs = []
            for pat in pattern.split():
                dirs += [d for d in sorted(glob.glob(pat)) if Path(d).is_dir()]
            for d in dirs:
                ckpt = Path(d) / "policy_best_by_val.pt"
                if not ckpt.exists():
                    raise FileNotFoundError(ckpt)
                try:
                    pol = load_policy(cfg, ckpt)
                except Exception as e:
                    raise RuntimeError(f"Cannot load {ckpt}") from e
                u, f, c = _metrics(cfg, lambda cc, p=pol: RLPolicyAgent(cc, p, deterministic=True), seeds)
                x = _exec_frac(cfg, RLPolicyAgent(cfg, pol, deterministic=True), seeds)
                row(f"{algo}:{Path(d).name.split('_')[-1]}", u, f, c, x, ae_u, algo=algo)

    args.csv.parent.mkdir(parents=True, exist_ok=True)
    with args.csv.open("w", newline="", encoding="utf-8") as h:
        w = _csv.DictWriter(h, fieldnames=["env", "policy", "algo", "util", "delta_vs_ae", "p", "exec_frac", "fail", "compl"])
        w.writeheader(); w.writerows(csv_rows)
    print(f"\nwrote {args.csv} ({len(csv_rows)} rows)")
    if len(csv_rows) != (8 * len(args.envs) if args.envs else 40):
        raise RuntimeError("Incomplete evaluation matrix")
    if args.per_seed_csv is not None:
        args.per_seed_csv.parent.mkdir(parents=True, exist_ok=True)
        with args.per_seed_csv.open("w", newline="", encoding="utf-8") as h:
            w = _csv.DictWriter(h, fieldnames=list(per_seed_rows[0]))
            w.writeheader(); w.writerows(per_seed_rows)
        print(f"wrote {args.per_seed_csv} ({len(per_seed_rows)} rows)")


if __name__ == "__main__":
    main()
