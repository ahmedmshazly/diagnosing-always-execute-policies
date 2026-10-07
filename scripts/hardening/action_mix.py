from __future__ import annotations

"""Action histogram of every learned policy in the taxonomy (greedy argmax).

Walks the same checkpoint globs as eval_taxonomy.py and prints, per policy, the
share of steps spent on each action over held-out seeds 200-209.

Usage: python -m scripts.hardening.action_mix > results/hardening/action_mix.txt
"""

import glob
from collections import Counter
from pathlib import Path

from scripts.hardening.eval_taxonomy import MATRIX
from src.config import load_config
from src.rl.agent import RLPolicyAgent, load_policy
from src.sim_environment import WorkloadGenerator, advance_one_step, make_episode_rngs

SEEDS = range(200, 210)


def main() -> None:
    print("Action mix of each learned policy, greedy argmax, held-out seeds 200-209 (% of steps)")
    for env_cfg, _, globs in MATRIX:
        cfg = load_config(Path(env_cfg))
        for algo, pattern in globs.items():
            dirs = sorted(sum((glob.glob(p) for p in pattern.split()), []))
            for d in dirs:
                ckpt = Path(d) / "policy_best_by_val.pt"
                if not ckpt.exists():
                    continue
                agent = RLPolicyAgent(cfg, load_policy(cfg, ckpt), deterministic=True)
                counts: Counter = Counter()
                for seed in SEEDS:
                    wr, er = make_episode_rngs(seed)
                    st = WorkloadGenerator(rng=wr, cfg=cfg, num_jobs=cfg.experiment.num_jobs,
                                           seed_label=seed).generate_episode()
                    while not st.all_done() and st.step < cfg.experiment.max_steps:
                        a = agent.choose_action(st)
                        advance_one_step(st, er, a)
                        counts[a] += 1
                n = sum(counts.values())
                mix = " ".join(f"{k}={100 * v / n:.1f}" for k, v in counts.most_common())
                print(f"{cfg.meta.get('config_name'):20s} {algo:9s} {Path(d).name:32s} {mix}")


if __name__ == "__main__":
    main()
