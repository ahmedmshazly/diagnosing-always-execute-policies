# Reproducibility entry points.
#
# Every target is config-driven. The single source of truth is
# ``config/default.yaml``. Override it per run with ``CONFIG=path/to/file.yaml``.

PYTHON ?= python3
PIP ?= pip
CONFIG ?= config/default.yaml
SEED_GROUP ?= midterm_baseline
RUN_ID ?=
OUT_ROOT ?=
PYTEST ?= pytest

# Resolve optional args so empty strings do not leak into the CLI.
RUN_ID_ARG := $(if $(RUN_ID),--run-id $(RUN_ID),)
OUT_ROOT_ARG := $(if $(OUT_ROOT),--out-root $(OUT_ROOT),)

.PHONY: help install test baseline baseline-train baseline-test sweep train figs clean phase2 phase2-aggregate phase2-figs phase2-sanity phase3 phase3-figs phase4-baseline phase4-smoke phase4-figs phase5-train phase5-train-all phase5-heldout phase5-aggregate phase5-figs phase5-rollouts phase5-all phase6-smoke-v1 phase6-train-v1 phase6-train-all-v1 phase6-heldout-v1 phase6-aggregate-v1 phase6-rollouts-v1 phase6-figs-v1 phase6-all-v1

help:
	@echo "make install        - install python dependencies"
	@echo "make test           - run pytest"
	@echo "make baseline       - run Reflex vs Utility baselines on SEED_GROUP (default: midterm_baseline)"
	@echo "make baseline-train / baseline-test - baselines on train or test seeds"
	@echo "make phase2         - Phase-2: 50-seed (0..49) baseline + aggregate + figure"
	@echo "make phase2-sanity  - re-run seeds 0..9 with midterm weights; diff against canonical_midterm"
	@echo "make phase3         - Phase-3: 80-cell (α, β, γ) sweep x 20 seeds (100..119)"
	@echo "make phase3-figs    - Pareto fronts + best-fixed-weight selection for phase3 run"
	@echo "make phase4-baseline - Phase-4 3-agent baseline (Reflex + full UB + stripped UB, n=50)"
	@echo "make phase4-smoke   - Phase-4 RL smoke training (10k env steps, stage-1 only, seed 7)"
	@echo "make phase4-figs    - Learning-curve figure + 3-agent baseline table for phase4 runs"
	@echo "make sweep          - (α, β, γ) Pareto sweep (older scaffold)"
	@echo "make train          - train the Self-Learning Utility-Based agent (scaffolded)"
	@echo "make figs           - regenerate paper figures from committed CSVs"
	@echo "make clean          - remove __pycache__"

install:
	$(PIP) install -r requirements.txt

test:
	$(PYTEST) -q

baseline:
	$(PYTHON) -m src.compare_baselines --config $(CONFIG) --seed-group $(SEED_GROUP) $(RUN_ID_ARG) $(OUT_ROOT_ARG)

baseline-train:
	$(MAKE) baseline SEED_GROUP=train

baseline-test:
	$(MAKE) baseline SEED_GROUP=test

sweep:
	$(PYTHON) -m scripts.sweep_utility --config $(CONFIG) $(RUN_ID_ARG) $(OUT_ROOT_ARG)

train:
	$(PYTHON) -m scripts.train_rl --config $(CONFIG) $(RUN_ID_ARG) $(OUT_ROOT_ARG)

figs:
	$(PYTHON) -m scripts.figs.make_all --config $(CONFIG)

PHASE2_RUN_ID ?= phase2_50seeds
PHASE2_RUN_DIR := results/$(PHASE2_RUN_ID)
PHASE2_SEEDS := 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 43 44 45 46 47 48 49

phase2:
	$(PYTHON) -m src.compare_baselines --config $(CONFIG) --seed-group custom --seeds $(PHASE2_SEEDS) --run-id $(PHASE2_RUN_ID)
	$(MAKE) phase2-aggregate PHASE2_RUN_DIR=$(PHASE2_RUN_DIR)
	$(MAKE) phase2-figs PHASE2_RUN_DIR=$(PHASE2_RUN_DIR)

phase2-aggregate:
	$(PYTHON) -m scripts.aggregate --run-dir $(PHASE2_RUN_DIR)

phase2-figs:
	$(PYTHON) -m scripts.figs.phase2_deltas --run-dir $(PHASE2_RUN_DIR)

PHASE2_SANITY_RUN_ID ?= phase2_sanity_seeds0_9
phase2-sanity:
	$(PYTHON) -m src.compare_baselines --config config/midterm_weights.yaml --seed-group midterm_baseline --run-id $(PHASE2_SANITY_RUN_ID)
	$(PYTHON) -m scripts.compare_runs --reference results/canonical_midterm --candidate results/$(PHASE2_SANITY_RUN_ID)

PHASE3_RUN_ID ?= sweep_phase3
PHASE3_RUN_DIR := results/$(PHASE3_RUN_ID)
PHASE3_SELECTION_RULE ?= mean_utility_then_completion

phase3:
	$(PYTHON) -m scripts.sweep_phase3 --config $(CONFIG) --run-id $(PHASE3_RUN_ID)
	$(MAKE) phase3-figs PHASE3_RUN_DIR=$(PHASE3_RUN_DIR) PHASE3_SELECTION_RULE=$(PHASE3_SELECTION_RULE)

phase3-figs:
	$(PYTHON) -m scripts.figs.phase3_pareto --run-dir $(PHASE3_RUN_DIR) --selection-rule $(PHASE3_SELECTION_RULE)

PHASE4_BASELINE_RUN_ID ?= phase4_baseline_n50
PHASE4_BASELINE_RUN_DIR := results/$(PHASE4_BASELINE_RUN_ID)
PHASE4_SMOKE_RUN_ID ?= phase4_rl_smoke
PHASE4_SMOKE_RUN_DIR := results/$(PHASE4_SMOKE_RUN_ID)

phase4-baseline:
	$(PYTHON) -m scripts.phase4_baseline --config $(CONFIG) --run-id $(PHASE4_BASELINE_RUN_ID)
	$(PYTHON) -m scripts.phase4_aggregate --run-dir $(PHASE4_BASELINE_RUN_DIR)

phase4-smoke:
	$(PYTHON) -m scripts.train_rl --config $(CONFIG) --run-id $(PHASE4_SMOKE_RUN_ID) --stage 1 --max-env-steps 10000

phase4-figs:
	$(PYTHON) -m scripts.figs.phase4_learning_curve --run-dir $(PHASE4_SMOKE_RUN_DIR) || true

# ---------------------------------------------------------------------------
# Phase 5 — full RL training + held-out evaluation + figures.
# ---------------------------------------------------------------------------
PHASE5_ROOT_DIR := results/phase5
PHASE5_HELDOUT_DIR := $(PHASE5_ROOT_DIR)/heldout
PHASE3_RUN_DIR_FOR_PARETO := results/sweep_phase3

PHASE5_INIT_SEED ?= 7
PHASE5_RUN_ID ?= phase5_rl_seed$(PHASE5_INIT_SEED)
PHASE5_MAX_ENV_STEPS ?= 0

# Single-seed training. Usage: make phase5-train PHASE5_INIT_SEED=7
phase5-train:
	$(PYTHON) -m scripts.train_rl_full --config $(CONFIG) --run-id $(PHASE5_RUN_ID) --init-seed $(PHASE5_INIT_SEED) $(if $(filter-out 0,$(PHASE5_MAX_ENV_STEPS)),--max-env-steps $(PHASE5_MAX_ENV_STEPS),)

# All three training runs. Sequential on purpose so the learning curves
# are reproducible and the CPU isn't swamped.
phase5-train-all:
	$(MAKE) phase5-train PHASE5_INIT_SEED=7  PHASE5_RUN_ID=phase5_rl_seed7
	$(MAKE) phase5-train PHASE5_INIT_SEED=11 PHASE5_RUN_ID=phase5_rl_seed11
	$(MAKE) phase5-train PHASE5_INIT_SEED=13 PHASE5_RUN_ID=phase5_rl_seed13

phase5-heldout:
	$(PYTHON) -m scripts.phase5_heldout --config $(CONFIG) --run-id phase5/heldout \
	  --rl-checkpoint rl_seed7=$(PHASE5_ROOT_DIR)/rl_seed7/policy_best_by_val.pt \
	  --rl-checkpoint rl_seed11=$(PHASE5_ROOT_DIR)/rl_seed11/policy_best_by_val.pt \
	  --rl-checkpoint rl_seed13=$(PHASE5_ROOT_DIR)/rl_seed13/policy_best_by_val.pt

phase5-aggregate:
	$(PYTHON) -m scripts.phase5_aggregate --run-dir $(PHASE5_HELDOUT_DIR)

phase5-figs:
	$(PYTHON) -m scripts.figs.phase5_training_curves \
	  --run-dir $(PHASE5_ROOT_DIR)/rl_seed7 \
	  --run-dir $(PHASE5_ROOT_DIR)/rl_seed11 \
	  --run-dir $(PHASE5_ROOT_DIR)/rl_seed13 \
	  --out $(PHASE5_ROOT_DIR)/training_curves.png
	$(PYTHON) -m scripts.figs.phase5_heldout_comparison \
	  --run-dir $(PHASE5_HELDOUT_DIR) \
	  --out $(PHASE5_HELDOUT_DIR)/heldout_comparison.png
	$(PYTHON) -m scripts.figs.phase5_pareto_with_rl \
	  --sweep-dir $(PHASE3_RUN_DIR_FOR_PARETO) \
	  --heldout-dir $(PHASE5_HELDOUT_DIR) \
	  --out $(PHASE5_ROOT_DIR)/pareto_with_rl.png

PHASE5_ROLLOUT_CHECKPOINT ?= $(PHASE5_ROOT_DIR)/rl_seed7/policy_best_by_val.pt
phase5-rollouts:
	$(PYTHON) -m scripts.phase5_rollout_dump --config $(CONFIG) \
	  --run-id phase5/rollouts --checkpoint $(PHASE5_ROLLOUT_CHECKPOINT) \
	  --seeds 200 203 210 225 240

phase5-all: phase5-train-all phase5-heldout phase5-aggregate phase5-figs phase5-rollouts

# ---------------------------------------------------------------------------
# Phase 6 V1 — richer 14-dim state (paper §6). Uses config/state_v2.yaml
# as the canonical entry-config; every target respects CONFIG= override.
# ---------------------------------------------------------------------------
PHASE6_V1_ROOT_DIR := results/phase6_v1
PHASE6_V1_CONFIG ?= config/state_v2.yaml
PHASE6_INIT_SEED ?= 7
PHASE6_V1_RUN_ID ?= phase6_v1/rl_seed$(PHASE6_INIT_SEED)
PHASE6_MAX_ENV_STEPS ?= 0

PHASE6_V1_SMOKE_RUN_ID ?= phase6_v1/smoke

phase6-smoke-v1:
	$(PYTHON) -m scripts.train_rl --config $(PHASE6_V1_CONFIG) \
	  --run-id $(PHASE6_V1_SMOKE_RUN_ID) --stage 1 --max-env-steps 10000 \
	  --batch-size 4 --log-every 5 --fixed-arrival-seed 300 --init-seed 7

phase6-train-v1:
	$(PYTHON) -m scripts.train_rl_full --config $(PHASE6_V1_CONFIG) \
	  --run-id $(PHASE6_V1_RUN_ID) --init-seed $(PHASE6_INIT_SEED) \
	  $(if $(filter-out 0,$(PHASE6_MAX_ENV_STEPS)),--max-env-steps $(PHASE6_MAX_ENV_STEPS),)

phase6-train-all-v1:
	$(MAKE) phase6-train-v1 PHASE6_INIT_SEED=7  PHASE6_V1_RUN_ID=phase6_v1/rl_seed7  CONFIG=$(PHASE6_V1_CONFIG)
	$(MAKE) phase6-train-v1 PHASE6_INIT_SEED=11 PHASE6_V1_RUN_ID=phase6_v1/rl_seed11 CONFIG=$(PHASE6_V1_CONFIG)
	$(MAKE) phase6-train-v1 PHASE6_INIT_SEED=13 PHASE6_V1_RUN_ID=phase6_v1/rl_seed13 CONFIG=$(PHASE6_V1_CONFIG)

phase6-heldout-v1:
	$(PYTHON) -m scripts.phase5_heldout --config $(PHASE6_V1_CONFIG) \
	  --run-id phase6_v1/heldout \
	  --rl-checkpoint rl_v1_seed7=$(PHASE6_V1_ROOT_DIR)/rl_seed7/policy_best_by_val.pt \
	  --rl-checkpoint rl_v1_seed11=$(PHASE6_V1_ROOT_DIR)/rl_seed11/policy_best_by_val.pt \
	  --rl-checkpoint rl_v1_seed13=$(PHASE6_V1_ROOT_DIR)/rl_seed13/policy_best_by_val.pt

phase6-aggregate-v1:
	$(PYTHON) -m scripts.phase5_aggregate --run-dir $(PHASE6_V1_ROOT_DIR)/heldout

PHASE6_V1_ROLLOUT_CHECKPOINT ?= $(PHASE6_V1_ROOT_DIR)/rl_seed7/policy_best_by_val.pt
phase6-rollouts-v1:
	$(PYTHON) -m scripts.phase5_rollout_dump --config $(PHASE6_V1_CONFIG) \
	  --run-id phase6_v1/rollouts --checkpoint $(PHASE6_V1_ROLLOUT_CHECKPOINT) \
	  --seeds 200 203 210 225 240

phase6-figs-v1:
	$(PYTHON) -m scripts.figs.phase5_training_curves \
	  --run-dir $(PHASE6_V1_ROOT_DIR)/rl_seed7 \
	  --run-dir $(PHASE6_V1_ROOT_DIR)/rl_seed11 \
	  --run-dir $(PHASE6_V1_ROOT_DIR)/rl_seed13 \
	  --title "Phase-6 V1 training curves (14-dim state, 3 init seeds)" \
	  --out $(PHASE6_V1_ROOT_DIR)/training_curves_v1.png
	$(PYTHON) -m scripts.figs.phase5_heldout_comparison \
	  --run-dir $(PHASE6_V1_ROOT_DIR)/heldout \
	  --out $(PHASE6_V1_ROOT_DIR)/heldout/heldout_comparison_v1.png
	$(PYTHON) -m scripts.figs.phase6_pareto_with_rl_v1 \
	  --sweep-dir results/sweep_phase3 \
	  --heldout-v1-dir $(PHASE6_V1_ROOT_DIR)/heldout \
	  --heldout-phase5-dir results/phase5/heldout \
	  --out $(PHASE6_V1_ROOT_DIR)/pareto_with_rl_v1.png

phase6-all-v1: phase6-smoke-v1 phase6-train-all-v1 phase6-heldout-v1 phase6-aggregate-v1 phase6-figs-v1 phase6-rollouts-v1

clean:
	find . -type d -name __pycache__ -exec rm -rf {} +
