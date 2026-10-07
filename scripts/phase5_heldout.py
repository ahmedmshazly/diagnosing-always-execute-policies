from __future__ import annotations

"""Phase-5 held-out evaluation driver.

Evaluates Reflex, the Phase-3 grid-best fixed-weight Utility-Based Agent
(alpha=4, beta=1, gamma=4), and a trained RL policy on the same 50
held-out seeds (paper §4.3.6: cfg.rl.test_seeds = 200..249).

For each agent, writes a Phase-2-schema ``metrics.csv`` into its own
agent sub-directory plus a combined ``metrics.csv`` at the run root that
contains rows from all three (and any additional RL runs supplied via
``--rl-checkpoint label=path``).

Downstream: ``scripts.phase5_aggregate`` reads the combined metrics.csv
and writes the headline 3-way table + paired Wilcoxon results.
"""

import argparse
import csv
import json
import time
from pathlib import Path
from typing import Dict, List

from src.config import RunConfig, load_config, override_utility_weights
from src.reflex_agent import build_reflex_agent
from src.rl.agent import make_rl_agent_factory
from src.runner import run_many_episodes
from src.run_artifacts import (
    ensure_run_dir,
    write_manifest,
    write_resolved_config,
)
from src.utility_agent import build_utility_agent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--out-root", type=Path, default=Path("results"))
    parser.add_argument("--run-id", type=str, required=True)
    parser.add_argument(
        "--rl-checkpoint",
        action="append",
        default=[],
        help=(
            "Additional RL checkpoint to evaluate, as 'label=/path/to/policy.pt'. "
            "Can be passed multiple times for multi-run comparison."
        ),
    )
    parser.add_argument(
        "--seeds",
        nargs="*",
        type=int,
        default=None,
        help="Override the held-out seed list. Default: cfg.rl.test_seeds.",
    )
    parser.add_argument(
        "--tuned-alpha",
        type=float,
        default=4.0,
        help="Phase-3 grid-best alpha (default 4.0).",
    )
    parser.add_argument(
        "--tuned-beta",
        type=float,
        default=1.0,
        help="Phase-3 grid-best beta (default 1.0).",
    )
    parser.add_argument(
        "--tuned-gamma",
        type=float,
        default=4.0,
        help="Phase-3 grid-best gamma (default 4.0).",
    )
    return parser.parse_args()


def _rows_of(metrics) -> List[Dict]:
    return [m.as_dict() for m in metrics]


def _write_csv(path: Path, rows: List[Dict]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _stamp_name(rows: List[Dict], name: str) -> List[Dict]:
    return [{**row, "agent_name": name} for row in rows]


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config) if args.config else load_config()
    out_root = Path(args.out_root)
    run_dir = ensure_run_dir(out_root, args.run_id)
    started = time.time()

    seeds = list(args.seeds) if args.seeds else list(cfg.rl.test_seeds)
    print(f"Phase-5 held-out evaluation: {run_dir}")
    print(f"  seeds ({len(seeds)}): {seeds[:3]}...{seeds[-3:]}")
    print(
        f"  tuned utility weights: alpha={args.tuned_alpha}, "
        f"beta={args.tuned_beta}, gamma={args.tuned_gamma}"
    )
    print(
        f"  extra RL checkpoints: "
        f"{[spec for spec in args.rl_checkpoint]}"
    )

    # 1) Reflex on the held-out pool.
    print("Evaluating Reflex Agent...")
    reflex_metrics = run_many_episodes(
        cfg=cfg, agent_factory=build_reflex_agent, seeds=seeds
    )
    reflex_rows = _rows_of(reflex_metrics)
    (run_dir / "reflex").mkdir(parents=True, exist_ok=True)
    _write_csv(run_dir / "reflex" / "metrics.csv", reflex_rows)

    # 2) Tuned Utility-Based on the same pool. Override weights so the
    # tuned cell is the comparand, not the Phase-1 default.
    print("Evaluating Tuned Utility-Based Agent (Phase-3 grid-best)...")
    tuned_cfg = override_utility_weights(
        cfg, alpha=args.tuned_alpha, beta=args.tuned_beta, gamma=args.tuned_gamma
    )
    tuned_metrics = run_many_episodes(
        cfg=tuned_cfg, agent_factory=build_utility_agent, seeds=seeds
    )
    tuned_rows = _stamp_name(
        _rows_of(tuned_metrics),
        f"Tuned Utility-Based Agent (alpha={args.tuned_alpha}, beta={args.tuned_beta}, gamma={args.tuned_gamma})",
    )
    (run_dir / "tuned_utility").mkdir(parents=True, exist_ok=True)
    _write_csv(run_dir / "tuned_utility" / "metrics.csv", tuned_rows)

    # 3) Any supplied RL checkpoints.
    rl_rows_combined: List[Dict] = []
    for spec in args.rl_checkpoint:
        if "=" not in spec:
            raise ValueError(f"--rl-checkpoint expects 'label=path/to/policy.pt', got {spec!r}")
        label, checkpoint_str = spec.split("=", 1)
        checkpoint_path = Path(checkpoint_str).expanduser()
        if not checkpoint_path.exists():
            raise FileNotFoundError(checkpoint_path)
        print(f"Evaluating RL policy '{label}' from {checkpoint_path}...")
        factory = make_rl_agent_factory(
            cfg=cfg, checkpoint_path=checkpoint_path, deterministic=True,
            label=label,
        )
        rl_metrics = run_many_episodes(cfg=cfg, agent_factory=factory, seeds=seeds)
        rl_rows = _stamp_name(_rows_of(rl_metrics), label)
        (run_dir / label).mkdir(parents=True, exist_ok=True)
        _write_csv(run_dir / label / "metrics.csv", rl_rows)
        rl_rows_combined.extend(rl_rows)

    combined = [*reflex_rows, *tuned_rows, *rl_rows_combined]
    _write_csv(run_dir / "metrics.csv", combined)

    summary = {
        "seeds": seeds,
        "num_seeds": len(seeds),
        "tuned_weights": {
            "alpha": args.tuned_alpha,
            "beta": args.tuned_beta,
            "gamma": args.tuned_gamma,
        },
        "rl_checkpoints": args.rl_checkpoint,
        "agent_labels": [
            rows[0]["agent_name"]
            for rows in [reflex_rows, tuned_rows] + [
                [row for row in rl_rows_combined if row["agent_name"] == spec.split("=", 1)[0]]
                for spec in args.rl_checkpoint
            ]
            if rows
        ],
    }
    (run_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
    )
    write_resolved_config(run_dir, cfg)
    write_manifest(
        run_dir=run_dir,
        cfg=cfg,
        seeds=seeds,
        extra={"entrypoint": "phase5_heldout", **summary},
        wall_clock_start=started,
    )

    print(f"Wrote held-out run to {run_dir}")


if __name__ == "__main__":
    main()
