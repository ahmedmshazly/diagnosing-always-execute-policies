from __future__ import annotations

"""Reusable policy comparison on any config + seed pool, scored on the TRUE
episode metric. Fast (no MC, no training): one capped episode per (agent, seed).

Agents available:
  reflex            - the Reflex baseline (scales conditionally when blocked).
  always_execute    - emits Execute_Ready_Job every step (the converged RL policy).
  scale_when_blocked- Execute if a ready task fits; else Scale_Up if any ready
                      task is blocked and no boost active; else Execute. A clean
                      "scale-aware greedy" upper-reference for capacity-limited
                      environments.
  utility           - the non-learning Utility-Based agent.
  stripped          - utility agent with the force-execute guard removed.

Prints per-agent aggregate (mean utility / completion / failure / cost) and,
for every agent vs always_execute, the paired mean delta + Wilcoxon p.

Usage:
  python -m scripts.hardening.diag_policy_compare --config config/env_capacity.yaml \
      --agents always_execute scale_when_blocked reflex
"""

import argparse
from collections import Counter
from pathlib import Path
from typing import List

import numpy as np
from scipy.stats import wilcoxon

from src.config import load_config
from src.reflex_agent import build_reflex_agent
from src.rl.agent import RLPolicyAgent, load_policy
from src.runner import run_many_episodes
from src.sim_environment import ACTIONS, EpisodeState, advance_one_step, make_episode_rngs, WorkloadGenerator
from src.utility_agent import build_utility_agent, build_stripped_utility_agent


class AlwaysExecuteAgent:
    name = "Always-Execute"

    def __init__(self, cfg) -> None:
        self.cfg = cfg

    def choose_action(self, state: EpisodeState) -> str:
        return "Execute_Ready_Job"


class ScaleWhenBlockedAgent:
    name = "Scale-When-Blocked"

    def __init__(self, cfg) -> None:
        self.cfg = cfg

    def choose_action(self, state: EpisodeState) -> str:
        ready = state.ready_tasks()
        if not ready:
            return "Execute_Ready_Job"
        acpu, aram = state.available_cpu(), state.available_ram()
        if any(t.cpu_demand <= acpu and t.ram_demand <= aram for t in ready):
            return "Execute_Ready_Job"
        # Nothing fits now. If a Scale_Up could help and none is active, scale.
        if state.cluster.scale_boost_remaining == 0:
            return "Scale_Up"
        return "Execute_Ready_Job"


class ThrottleAgent:
    """Cautious policy: defer when the cluster is heavily loaded, else execute.

    Under load-dependent failure (per_node_bernoulli), running fewer tasks
    concurrently lowers the failure rate. This is the natural exploiter of
    that structure; if caution pays anywhere, it pays here.
    """

    name = "Throttle"

    def __init__(self, cfg, load_threshold: float = 0.6) -> None:
        self.cfg = cfg
        self.load_threshold = load_threshold

    def choose_action(self, state: EpisodeState) -> str:
        cpu_load = state.cpu_in_use() / max(state.cluster.cpu_capacity, 1)
        if cpu_load > self.load_threshold:
            return "Defer_Job"
        return "Execute_Ready_Job"


FACTORIES = {
    "reflex": build_reflex_agent,
    "always_execute": lambda cfg: AlwaysExecuteAgent(cfg),
    "scale_when_blocked": lambda cfg: ScaleWhenBlockedAgent(cfg),
    "throttle": lambda cfg: ThrottleAgent(cfg),
    "throttle50": lambda cfg: ThrottleAgent(cfg, load_threshold=0.5),
    "throttle75": lambda cfg: ThrottleAgent(cfg, load_threshold=0.75),
    "utility": build_utility_agent,
    "stripped": build_stripped_utility_agent,
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--agents", nargs="+", default=["always_execute", "scale_when_blocked", "reflex"])
    parser.add_argument("--rl-checkpoint", action="append", default=[],
                        help="label=path.pt; evaluates a trained policy (argmax).")
    parser.add_argument("--seeds", nargs="*", type=int, default=None)
    parser.add_argument("--baseline", default="always_execute",
                        help="agent to compute paired deltas against")
    parser.add_argument("--action-hist", action="store_true",
                        help="also print each agent's emitted-action histogram")
    args = parser.parse_args()

    cfg = load_config(args.config) if args.config else load_config()
    seeds = args.seeds if args.seeds else list(cfg.rl.test_seeds)
    print(f"config={args.config or 'config/default.yaml'} "
          f"(name={cfg.meta.get('config_name')}) | seeds {seeds[0]}..{seeds[-1]} "
          f"(n={len(seeds)})\n")

    # Build the agent factory list: named agents + any RL checkpoints.
    factories = {a: FACTORIES[a] for a in args.agents}
    for spec in args.rl_checkpoint:
        label, path = spec.split("=", 1)
        policy = load_policy(cfg, Path(path))
        factories[label] = (lambda c, p=policy: RLPolicyAgent(c, p, deterministic=True))
    order = list(args.agents) + [s.split("=", 1)[0] for s in args.rl_checkpoint]

    util = {}
    comp = {}
    fail = {}
    cost = {}
    for a in order:
        metrics = run_many_episodes(cfg=cfg, agent_factory=factories[a], seeds=seeds,
                                    include_uncapped=False)
        util[a] = np.array([m.total_utility for m in metrics])
        comp[a] = np.array([m.completion_rate for m in metrics])
        fail[a] = np.array([m.failure_rate for m in metrics])
        cost[a] = np.array([m.total_compute_cost for m in metrics])

    args.agents = order  # downstream printing iterates this

    print(f"{'agent':<20} {'mean_util':>10} {'completion':>11} {'failure':>9} {'cost':>9}")
    for a in args.agents:
        print(f"{a:<20} {util[a].mean():>10.2f} {comp[a].mean():>11.4f} "
              f"{fail[a].mean():>9.4f} {cost[a].mean():>9.1f}")

    if args.action_hist:
        print("\nEmitted-action histograms (% of steps):")
        for a in args.agents:
            agent = factories[a](cfg)
            hist: Counter = Counter()
            total = 0
            for seed in seeds:
                wr, er = make_episode_rngs(seed)
                gen = WorkloadGenerator(rng=wr, cfg=cfg,
                                        num_jobs=cfg.experiment.num_jobs, seed_label=seed)
                st = gen.generate_episode()
                while not st.all_done() and st.step < cfg.experiment.max_steps:
                    act = agent.choose_action(st)
                    advance_one_step(st, er, act)
                    hist[act] += 1
                    total += 1
            parts = " ".join(
                f"{a2.replace('_','')[:8]}={100*hist.get(a2,0)/max(total,1):.0f}%"
                for a2 in ACTIONS
            )
            print(f"  {a:<22} {parts}")

    base = args.baseline
    if base in util:
        print(f"\nPaired deltas vs {base} (positive util delta = agent beats {base}):")
        print(f"{'agent':<20} {'dUtil':>10} {'dCompletion':>12} {'wilcoxon_p':>11}")
        for a in args.agents:
            if a == base:
                continue
            d_util = util[a] - util[base]
            d_comp = comp[a] - comp[base]
            try:
                p = wilcoxon(util[a], util[base]).pvalue
            except ValueError:
                p = float("nan")
            print(f"{a:<20} {d_util.mean():>+10.2f} {d_comp.mean():>+12.4f} {p:>11.2e}")


if __name__ == "__main__":
    main()
