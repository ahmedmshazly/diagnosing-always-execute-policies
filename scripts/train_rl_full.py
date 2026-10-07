from __future__ import annotations

"""Phase-5 full-training entry point.

Runs the paper's committed protocol (§4.3.1–§4.3.5):
- REINFORCE with sequence-specific baseline.
- Three-stage curriculum over (N, T_max) = (20, 120), (50, 200), (100, 300)
  with update budgets 80 / 120 / 200 (total 400 updates).
- Memoryless termination within each stage.
- Validation-pool evaluation every ``cfg.rl.eval_every_updates`` updates
  over ``cfg.rl.validation_seeds``. Best-by-validation checkpoint is
  retained for the held-out evaluation.

CLI:
- ``--init-seed``: one of ``cfg.rl.initialisation_seeds`` = {7, 11, 13}.
- ``--run-id``: output directory under ``results/``.
- ``--max-env-steps`` (optional): walltime fence; stops training early.
- ``--validation-cap`` (optional): subsample the validation pool for a
  faster smoke. Defaults to the full 50-seed validation pool.

Outputs:
- ``config.yaml``, ``run_manifest.json`` (standard).
- ``learning_curve.csv``      per-update training stats.
- ``validation_log.csv``      per-validation-eval stats + best-so-far.
- ``policy_final.pt``         final policy state_dict.
- ``policy_best_by_val.pt``   state_dict at the highest validation mean.
- ``summary.json``            compact summary used by downstream scripts.
"""

import argparse
import csv
import json
import time
from dataclasses import asdict
from pathlib import Path
from typing import List

import numpy as np
import torch

# One thread per process so parallel runs don't oversubscribe (sim loop is
# single-threaded; tiny MLP -> intra-op parallelism only thrashes).
torch.set_num_threads(1)

from src.config import load_config
from src.rl.trainer import TrainConfig, UpdateRecord, ValidationRecord, train
from src.run_artifacts import (
    ensure_run_dir,
    generate_run_id,
    write_manifest,
    write_resolved_config,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--out-root", type=Path, default=Path("results"))
    parser.add_argument("--run-id", type=str, required=True)
    parser.add_argument("--init-seed", type=int, required=True)
    parser.add_argument("--max-env-steps", type=int, default=None)
    parser.add_argument(
        "--validation-cap",
        type=int,
        default=None,
        help="If set, only use the first N validation seeds (for faster iteration).",
    )
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument(
        "--fixed-arrival-seed",
        type=int,
        default=None,
        help="If set, every batch shares this arrival seed (deterministic smoke).",
    )
    return parser.parse_args()


def _write_learning_curve(path: Path, records: List[UpdateRecord]) -> None:
    rows = [asdict(r) for r in records]
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _write_validation_log(path: Path, records: List[ValidationRecord]) -> None:
    rows = [asdict(r) for r in records]
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _summarise(
    records: List[UpdateRecord],
    val_records: List[ValidationRecord],
) -> dict:
    out = {"num_updates": len(records), "num_validations": len(val_records)}
    if records:
        returns = np.array([r.mean_undiscounted_return for r in records], dtype=np.float64)
        window = max(5, len(records) // 5)
        out.update(
            {
                "env_steps_total": int(records[-1].env_steps_cumulative),
                "first_window_mean_return": float(returns[:window].mean()),
                "last_window_mean_return": float(returns[-window:].mean()),
                "return_improvement": float(
                    returns[-window:].mean() - returns[:window].mean()
                ),
                "window": window,
            }
        )
    if val_records:
        best = max(val_records, key=lambda r: r.mean_return)
        out.update(
            {
                "best_validation_mean_return": float(best.mean_return),
                "best_validation_update": int(best.update),
                "best_validation_env_steps": int(best.env_steps_cumulative),
                "final_validation_mean_return": float(val_records[-1].mean_return),
            }
        )
    return out


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config) if args.config else load_config()
    out_root = Path(args.out_root)
    run_id = args.run_id
    run_dir = ensure_run_dir(out_root, run_id)
    started = time.time()

    training_cfg = TrainConfig.from_cfg(cfg, init_seed=args.init_seed)
    print(f"Phase-5 full-training run: {run_dir}")
    print(f"  init_seed={training_cfg.init_seed}  batch_size={training_cfg.batch_size}")
    print(
        f"  curriculum: {[(s.num_jobs, s.max_steps, s.num_updates) for s in training_cfg.stages]}"
    )
    val_seeds = list(cfg.rl.validation_seeds)
    if args.validation_cap is not None:
        val_seeds = val_seeds[: int(args.validation_cap)]
    print(f"  validation pool: {len(val_seeds)} seeds (every {cfg.rl.eval_every_updates} updates)")

    checkpoint_dir = run_dir / "checkpoints"
    best_path = run_dir / "policy_best_by_val.pt"
    validation_log: List[ValidationRecord] = []

    policy, records = train(
        cfg=cfg,
        training_cfg=training_cfg,
        max_env_steps=args.max_env_steps,
        checkpoint_dir=checkpoint_dir,
        log_every=args.log_every,
        fixed_arrival_seed=args.fixed_arrival_seed,
        validation_seeds=val_seeds,
        validation_log=validation_log,
        best_checkpoint_path=best_path,
    )

    torch.save(policy.state_dict(), run_dir / "policy_final.pt")
    _write_learning_curve(run_dir / "learning_curve.csv", records)
    _write_validation_log(run_dir / "validation_log.csv", validation_log)
    summary = _summarise(records, validation_log)
    summary["init_seed"] = int(training_cfg.init_seed)
    summary["best_checkpoint_path"] = str(best_path) if best_path.exists() else None
    summary["validation_seeds"] = val_seeds
    (run_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
    )

    write_resolved_config(run_dir, cfg)
    write_manifest(
        run_dir=run_dir,
        cfg=cfg,
        seeds=val_seeds,
        extra={
            "entrypoint": "train_rl_full",
            "scope": "phase5_full",
            "init_seed": int(training_cfg.init_seed),
            "max_env_steps": args.max_env_steps,
        },
        wall_clock_start=started,
    )
    elapsed = time.time() - started
    print("---")
    print(
        f"Done. {len(records)} updates, {summary.get('env_steps_total', 0)} env steps, "
        f"best val={summary.get('best_validation_mean_return', float('nan')):+7.3f} "
        f"@ update {summary.get('best_validation_update', -1)}  "
        f"({elapsed:.1f} s wall-clock)"
    )


if __name__ == "__main__":
    main()
