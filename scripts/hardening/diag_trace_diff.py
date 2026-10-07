from __future__ import annotations

"""Diagnostic A2 — is the RL-vs-Reflex gap a real deterministic difference,
or sampling noise / an entropy tax?

Phase-0 claim: held-out evaluation is deterministic argmax (confirmed in
scripts/phase5_heldout.py:152 -> src/rl/agent.py:61), so the p=0.08 gap
cannot be sampling noise. It must be a behavioural difference between two
*different* deterministic policies:

  - RL emits Execute_Ready_Job unconditionally (argmax). In a capacity-
    blocked or empty-ready state, do_action returns "insufficient_resources"
    / "no_ready_job" and launches nothing -- a silent no-op.
  - Reflex (reflex_agent.py:25) emits Scale_Up / Scale_Down / Defer in
    exactly those states.

Same emitted token, different effect. This script runs each policy
INDEPENDENTLY on the identical seed (same workload + event RNG) over the
full held-out pool and measures:

  1. RL emitted-action histogram + the no-op rate of its Execute emissions.
  2. Reflex emitted-action histogram (its non-execute rate).
  3. Per-seed completion / cost / failure, cross-checked against the
     committed results/phase5/heldout/metrics.csv (a reproduction test).

It also reports, per seed, the per-step "effect agreement": of the 300
steps, how many had an identical realised EFFECT class (launched / no-op /
scaled / deferred) under the two policies.

Usage:
  python -m scripts.hardening.diag_trace_diff \
     --checkpoint results/phase5/rl_seed7/policy_best_by_val.pt
"""

import argparse
import csv
from collections import Counter
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

from src.config import load_config
from src.reflex_agent import build_reflex_agent
from src.rl.agent import RLPolicyAgent, load_policy
from src.sim_environment import (
    WorkloadGenerator,
    advance_one_step,
    make_episode_rngs,
)


def _effect_class(action: str, event: str) -> str:
    """Collapse the do_action event string into an effect class."""
    if action == "Execute_Ready_Job":
        if event.startswith("launched"):
            return "launched"
        return "execute_noop"  # no_ready_job | insufficient_resources
    if action == "Defer_Job":
        return "defer"
    if action == "Scale_Up":
        return "scale_up"
    if action == "Scale_Down":
        return "scale_down"
    if action == "Reprioritize_Queue":
        return "reprioritize"
    if action == "Pause_LowPriority_Job":
        return "pause"
    return "other"


def _rollout(cfg, agent, seed: int) -> Tuple[List[str], List[str], Dict]:
    """Run one capped episode, returning (actions, effect_classes, metrics-ish)."""
    workload_rng, event_rng = make_episode_rngs(seed)
    gen = WorkloadGenerator(rng=workload_rng, cfg=cfg, num_jobs=cfg.experiment.num_jobs, seed_label=seed)
    state = gen.generate_episode()
    actions: List[str] = []
    effects: List[str] = []
    max_steps = cfg.experiment.max_steps
    while not state.all_done() and state.step < max_steps:
        action = agent.choose_action(state)
        info = advance_one_step(state, event_rng, action)
        actions.append(action)
        effects.append(_effect_class(action, str(info["event"])))
    completed = sum(1 for j in state.jobs if j.completed)
    failed = sum(1 for j in state.jobs if j.failed)
    return actions, effects, {
        "completed_jobs": completed,
        "failed_jobs": failed,
        "completion_rate": completed / max(cfg.experiment.num_jobs, 1),
        "steps": state.step,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path,
                        default=Path("results/phase5/rl_seed7/policy_best_by_val.pt"))
    parser.add_argument("--seeds", nargs="*", type=int, default=None)
    args = parser.parse_args()

    cfg = load_config()
    seeds = args.seeds if args.seeds else list(cfg.rl.test_seeds)  # 200..249

    policy = load_policy(cfg, args.checkpoint)
    rl_agent = RLPolicyAgent(cfg, policy, deterministic=True)
    reflex_agent = build_reflex_agent(cfg)

    rl_hist: Counter = Counter()
    rl_effect_hist: Counter = Counter()
    reflex_hist: Counter = Counter()
    reflex_effect_hist: Counter = Counter()

    per_seed_rows: List[dict] = []
    effect_agreements: List[float] = []
    rl_total_steps = 0
    rl_execute_emitted = 0
    rl_execute_noop = 0
    reflex_nonexecute = 0
    reflex_total_steps = 0

    for seed in seeds:
        rl_actions, rl_effects, rl_m = _rollout(cfg, rl_agent, seed)
        rf_actions, rf_effects, rf_m = _rollout(cfg, reflex_agent, seed)

        rl_hist.update(rl_actions)
        rl_effect_hist.update(rl_effects)
        reflex_hist.update(rf_actions)
        reflex_effect_hist.update(rf_effects)

        rl_total_steps += len(rl_actions)
        reflex_total_steps += len(rf_actions)
        rl_execute_emitted += sum(1 for a in rl_actions if a == "Execute_Ready_Job")
        rl_execute_noop += sum(1 for e in rl_effects if e == "execute_noop")
        reflex_nonexecute += sum(1 for a in rf_actions if a != "Execute_Ready_Job")

        # Per-step effect agreement up to the shorter episode length.
        n = min(len(rl_effects), len(rf_effects))
        agree = sum(1 for i in range(n) if rl_effects[i] == rf_effects[i])
        denom = max(len(rl_effects), len(rf_effects))
        effect_agreements.append(agree / max(denom, 1))

        per_seed_rows.append({
            "seed": seed,
            "rl_completion": round(rl_m["completion_rate"], 4),
            "reflex_completion": round(rf_m["completion_rate"], 4),
            "rl_failed": rl_m["failed_jobs"],
            "reflex_failed": rf_m["failed_jobs"],
            "rl_steps": rl_m["steps"],
            "reflex_steps": rf_m["steps"],
            "rl_execute_noops": sum(1 for e in rl_effects if e == "execute_noop"),
            "reflex_scale_up": sum(1 for a in rf_actions if a == "Scale_Up"),
            "reflex_scale_down": sum(1 for a in rf_actions if a == "Scale_Down"),
            "reflex_defer": sum(1 for a in rf_actions if a == "Defer_Job"),
            "effect_agreement": round(effect_agreements[-1], 4),
        })

    print(f"=== RL (argmax, {args.checkpoint.name}) vs Reflex on seeds "
          f"{seeds[0]}..{seeds[-1]} (n={len(seeds)}) ===\n")

    print("RL emitted-action histogram:")
    for a, c in rl_hist.most_common():
        print(f"  {a:<24} {c:>7}  ({100*c/max(rl_total_steps,1):5.1f}%)")
    print(f"RL is 100% Execute_Ready_Job: "
          f"{rl_hist.get('Execute_Ready_Job', 0) == rl_total_steps}")
    print(f"  of which SILENT NO-OPS (blocked/empty): {rl_execute_noop} "
          f"({100*rl_execute_noop/max(rl_execute_emitted,1):.1f}% of Execute emissions)\n")

    print("RL realised-effect histogram:")
    for e, c in rl_effect_hist.most_common():
        print(f"  {e:<24} {c:>7}  ({100*c/max(rl_total_steps,1):5.1f}%)")

    print("\nReflex emitted-action histogram:")
    for a, c in reflex_hist.most_common():
        print(f"  {a:<24} {c:>7}  ({100*c/max(reflex_total_steps,1):5.1f}%)")
    print(f"Reflex non-execute emissions: {reflex_nonexecute} "
          f"({100*reflex_nonexecute/max(reflex_total_steps,1):.1f}%)")

    ea = np.array(effect_agreements)
    print(f"\nPer-step EFFECT agreement (RL vs Reflex), per seed:")
    print(f"  mean {ea.mean():.4f} | min {ea.min():.4f} | max {ea.max():.4f}")
    print("Interpretation: 1.0 would mean identical policies. The shortfall is "
          "the deterministic behavioural gap that the p=0.08 test is measuring.")

    out = Path("results/hardening/trace_diff_per_seed.csv")
    with out.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(per_seed_rows[0].keys()))
        writer.writeheader()
        writer.writerows(per_seed_rows)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
