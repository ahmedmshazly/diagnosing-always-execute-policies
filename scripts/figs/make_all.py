from __future__ import annotations

"""Figure regeneration scaffold.

Every plot in the paper must map to a script in ``scripts/figs/``. This
top-level entry point enumerates the per-figure modules and writes their
outputs to ``assets/charts_generated/``. Individual figure scripts arrive in
follow-up commits along with the sweep and RL experiments.
"""

import argparse
from pathlib import Path

from src.config import load_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config) if args.config else load_config()
    out_dir = Path("assets") / "charts_generated"
    out_dir.mkdir(parents=True, exist_ok=True)
    stub = (
        "Figure generation is delegated to per-figure scripts under "
        "scripts/figs/. This top-level entry point enumerates them.\n\n"
        f"Config hash: {cfg.config_hash()[:12]}\n"
    )
    (out_dir / "README.md").write_text(stub, encoding="utf-8")
    print(f"figs: scaffold ready at {out_dir}")


if __name__ == "__main__":
    main()
