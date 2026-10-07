# Reproduction package

**Article:** Diagnosing always-execute policies in reinforcement learning for data-pipeline scheduling  
**Journal:** SN Computer Science, Original Research submission  
**Author and contact:** Ahmed Elshazlly, Electrical and Computer Engineering, Carnegie Mellon University Africa, Kigali, Rwanda; ahmdmshazly@gmail.com

This package contains a synthetic scheduling simulator, saved training artifacts and the analyses reported in the manuscript. The paper is a simulator case study. No production trace or external private dataset is needed.

## Start here

Use Python 3.12. Run commands from this directory after extracting the archive.

```bash
python3 -m venv .venv
. .venv/bin/activate
python -m pip install -r requirements-sncs.txt
python -m pytest -q
python -m scripts.sncs_report
```

The last command rebuilds current summaries in `results/sncs/` and vector figures in `figures/` from the supplied results. It performs no training. Use `MANIFEST.sha256` to check the delivered files before running commands that rewrite results:

```bash
sha256sum -c MANIFEST.sha256
```

## Re-evaluate the saved checkpoints

These commands can take tens of minutes on a CPU. Each evaluates a fixed set of saved policies and simple references. They do not train new policies.

```bash
python -u -m scripts.hardening.eval_taxonomy \
  --seed-start 200 --seed-stop 250 \
  --csv results/sncs/rechecked_n50.csv \
  --per-seed-csv results/sncs/perseed_n50.csv

python -u -m scripts.hardening.eval_taxonomy \
  --seed-start 1000 --seed-stop 1250 \
  --csv results/sncs/rechecked_n250.csv \
  --per-seed-csv results/sncs/perseed_n250.csv

python -m scripts.hardening.diag_power_rliable \
  --out results/sncs/power_rliable.csv
python -m scripts.sncs_report
```

The 50-workload evaluation writes 40 summary rows and 2000 per-workload rows; the 250-workload evaluation writes 40 and 10,000. The duplicated cascade reference is retained in raw evaluations, then counted once in the comparison family. The report prefers rechecked files when present and checks their rounded metrics against the historical evaluation tables.

## What each result means

| Path | Use |
|---|---|
| `results/sncs/report.json` | Current baseline statistics, episode-cluster probe intervals and analysis inputs |
| `results/sncs/holm_n50.csv`, `holm_n250.csv` | Full family of 34 comparisons per pool, including identical-result hypotheses with p=1 |
| `results/sncs/perseed_n50.csv`, `perseed_n250.csv` | Per-workload utility, failure and completion values for the fixed policies |
| `results/sncs/training_inventory.csv` | All 30 checkpoint paths, SHA-256 hashes, actual budgets and completeness flags |
| `results/hardening/reward_identity.csv` | Original reward/episode-metric discrepancy on ten workloads |
| `results/hardening/local_optimality.csv` | Historical filename for the sampled one-step probe, not an optimality certificate |
| `results/hardening/action_mix.txt` | Policy action counts on the ten-workload sample, seeds 200 through 209 |
| `results/ablation_phase3_weights_n20/` | Non-learning agent coefficient/guard ablations and action traces |
| `results/sweep_phase3/` | Non-learning coefficient sweep, not an RL training sweep |
| `results/phase5/`, `results/phase6_v1/` | Default actor and richer-observation artifacts |
| `config/`, `src/`, `scripts/`, `tests/` | Configurations, implementation, diagnostics and regression tests |

The current manuscript's derived statistics come from `scripts/sncs_report.py`. Historical `taxonomy_holm.csv` and `taxonomy_n250_holm.csv` used a smaller effective comparison family; do not use them for the revised paper. Historical figures and narrative research notes are excluded from the submission archive. Other older analysis scripts are retained to document the experiment workflow; the revised reporting command is authoritative for the manuscript.

## Training and selection limits

The 27 complete runs and three incomplete high-value REINFORCE records are distinguished in `training_inventory.csv`. The incomplete logs reach validation update 125 of 150 planned updates; they lack final curves and completion manifests. Their saved best checkpoints remain evaluable but do not establish a completed training comparison. No completion record has been invented.

REINFORCE and PPO use the same actor size but different critics, update reuse, termination, curricula and validation schedules. Historical PPO configuration files contain inherited REINFORCE labels; effective PPO values are in `scripts/hardening/train_ppo.py`, `src/rl/ppo.py`, manifests and learning curves. Both trainers select checkpoints by summed validation reward under their own reward mode, so changing the reward also changes selection.

The script below illustrates training a fresh PPO policy in a separate directory. It is optional and is not needed to rebuild the paper's results:

```bash
python -m scripts.hardening.train_ppo \
  --config config/env_tight_fixedrisk.yaml \
  --run-id reproduction_ppo_tight_seed7 --init-seed 7 \
  --iterations 150 --out-root fresh_runs
```

For REINFORCE use `scripts.train_rl_full`, the matching configuration and the validation cap documented for the particular run. Do not overwrite the supplied checkpoint directories. The historical training environments vary; the pinned current environment verifies saved-checkpoint evaluation and does not promise bitwise equality after retraining.

## Statistical scope

Intervals and paired tests describe workload variation conditional on a fixed saved policy. They are not estimates over independent training runs. Each Holm family contains 30 learned comparisons and four distinct hand references. The default seed-7 comparison with Reflex is a separate diagnostic, not another corrected family member.

Probe intervals resample five source episodes and condition on 16 fixed continuation seeds. The paired win score gives half credit to ties; the strict win fraction is reported separately. Exploratory equivalence margins and effect-size-based sensitivity calculations are not prospective design guarantees. The reward correction aligns undiscounted sums only; training still discounts at 0.99.

## Provenance and license

`SOURCE_PROVENANCE.json` records the copied source hashes and the local source state used for preparation. `MANIFEST.sha256` records the delivered contents. Source files were copied into this submission package; the existing project and earlier journal folders were preserved. No new training was performed for this journal revision. Analysis scripts, reporting text and evaluation outputs were revised as documented in `CHANGES_SNCS.md`.

The supplied `LICENSE` is the project's existing software license. Third-party Python packages retain their respective licenses. This package contains synthetic data and the author's own experiment artifacts. See Online Resource 1 for the complete scientific methods and the main manuscript for declarations.
