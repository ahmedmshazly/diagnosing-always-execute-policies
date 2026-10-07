# Specification — state, actions, stochastic processes, cost, utility

This document is the written specification for the orchestration simulator
and agents. Everything here must match `config/default.yaml`. The
corresponding Python lives in `src/state.py`, `src/sim_environment.py`, and
`src/cost.py`, all of which are required to read from config rather than
define constants in Python.

Where a design required a committed choice (e.g. the exact semantics of
"Node_Failure with P = 0.05"), this file names the choice, records the
alternative, and explains why the active choice was taken.

---

## 1. State variables

The observation passed to every agent is assembled by
`EpisodeState.state_vector()` and wrapped in the `StateVector` dataclass
defined in `src/state.py`. Every field has a docstring that pins down its
exact type, unit, range, and how it is computed. The table below is the
normative version and must stay in sync with the dataclass.

| Field | Type | Unit | Range | How it is computed |
|---|---|---|---|---|
| `cpu_load` | `float` | fraction (0 = idle, 1 = full) | `[0.0, 1.0]` | `cpu_in_use / cluster.cpu_capacity`; rounded to 4 decimals; `cpu_in_use = sum(running_task.cpu_demand)`. Denominator is clamped to `max(cluster.cpu_capacity, 1)` so a scale-down to zero never divides by zero. |
| `ram_available` | `float` | fraction | `[0.0, 1.0]` | `available_ram / cluster.ram_capacity`; rounded to 4 decimals; `available_ram = max(cluster.ram_capacity - ram_in_use, 0)`. |
| `queue_depth` | `float` | fraction of the depth saturation point | `[0.0, 1.0]` | `min(len(ready_tasks) / state_vector.queue_depth_norm, 1.0)` with `queue_depth_norm = 20.0`. |
| `spot_price` | `float` | dimensionless price index | `[events.spot_price_min, events.spot_price_max] = [0.1, 1.0]` | Latest value of `cluster.spot_price`; rounded to 4 decimals. |
| `dag_ready_nodes` | `float` | fraction of the ready-node saturation point | `[0.0, 1.0]` | `min(len(ready_tasks) / state_vector.ready_nodes_norm, 1.0)`. Distinct from `queue_depth` because the two divisors can be configured independently (the midterm spec ties both to 20 and they happen to coincide). |
| `job_priority` | `float` | mean priority across active jobs | `[workload.priority_low, workload.priority_high] = [0.2, 1.0]` (or `0.0` when there are no active jobs) | `mean(job.priority for job in active_jobs)`; rounded to 4 decimals. |
| `deadline_urgency` | `float` | fraction of deadline consumed | `[0.0, 1.0]` | `mean(max(0, 1 - job.deadline_steps / experiment.max_steps))` over active jobs; rounded to 4 decimals. Note: the midterm computation is a *per-job-averaged* urgency, not a per-step-elapsed clock. |
| `recent_failures` | `float` | saturated failure count | `[0.0, 1.0]` | `min(cluster.recent_failures / state_vector.recent_failures_norm, 1.0)` with `recent_failures_norm = 5.0`. The underlying integer is bumped on every `Node_Failure` event and decremented by 1 on every step (floor 0), capped at `cluster.max_recent_failures = 10`. |

Active jobs = `[job for job in episode.jobs if not job.failed and not job.completed]`.

The dataclass also exposes a `.as_dict()` method that returns the legacy
Title_Case keys (`CPU_Load`, `RAM_Available`, …) so downstream code that
indexes by string keeps working during the transition.

---

## 2. Actions

Every action has a **fixed, named, documented parameter set**. The values
below are the Phase-1 committed values and live in `config.simulator.action_params`.

### `Execute_Ready_Job(selection_policy, max_launches_per_step)`

- `selection_policy = "value_times_priority"` — the agent always launches
  the ready task whose parent job maximises `job.value * job.priority`.
- `max_launches_per_step = 1` — one launch per call. Multiple ready tasks
  can start in the same step only if the agent chose `Execute_Ready_Job`
  multiple times across steps.
- Effect: at most one ready task moves `ready → running`; capacity check is
  `cpu_demand ≤ available_cpu` and `ram_demand ≤ available_ram`.

### `Defer_Job(duration_steps)`

- `duration_steps = 1` — a single no-op step.
- Effect: nothing happens to the cluster or the queue; running tasks still
  make their 1 unit of progress inside `progress_running_tasks` exactly as
  with any other action. This is the "wait one tick" action; the multi-step
  timer interpretation is explicitly not adopted.

### `Scale_Up(cpu_delta, ram_delta, duration_steps)`

- `cpu_delta = 3` (additive units of CPU capacity).
- `ram_delta = 4` (additive units of RAM capacity).
- `duration_steps = 3` — capacity decays back to the configured baseline
  after this many ticks.
- Effect: `cluster.cpu_capacity += cpu_delta`, `cluster.ram_capacity +=
  ram_delta`, `cluster.scale_boost_remaining = duration_steps`. The decay
  logic in `apply_random_events` subtracts the same `cpu_delta / ram_delta`
  when `scale_boost_remaining` hits 0. Multiple back-to-back `Scale_Up`
  calls stack additively (the timer just resets to `duration_steps`); if
  tighter semantics are needed later this file is where the new rule lands.

### `Scale_Down(cpu_delta, ram_delta, min_cpu, min_ram)`

- `cpu_delta = 2`, `ram_delta = 2`.
- `min_cpu = cluster.min_cpu_capacity = 6`, `min_ram = cluster.min_ram_capacity = 8`.
- Effect: `cluster.cpu_capacity = max(min_cpu, cpu_capacity - cpu_delta)`;
  same for RAM. Does **not** evict running tasks. The invariant
  `cpu_in_use ≤ cpu_capacity` is therefore enforced only by the launch-time
  check and the per-step event loop; scaling down below current usage is
  allowed and simply blocks subsequent launches until capacity recovers.

### `Reprioritize_Queue(bump, cap)`

- `bump = 0.05`, `cap = 1.0`.
- Effect: every non-terminal job has `priority = min(cap, priority + bump)`.
  This is global, not per-selected-job; it nudges the whole active queue
  toward high priority, reducing the spread that `Pause_LowPriority_Job`
  depends on.

### `Pause_LowPriority_Job(priority_threshold, max_jobs)`

- `priority_threshold = 0.4`, `max_jobs = 2`.
- Effect: selects up to `max_jobs` non-terminal jobs with
  `priority < priority_threshold`, and marks each of their `ready` tasks
  as `paused`. Does not affect tasks that are `running`, `completed`,
  `failed`, or `waiting`. A paused task stays paused until a later
  `Execute_Ready_Job` / `Reprioritize_Queue` / design extension unpauses it
  — in the current code there is no unpause, so pausing is sticky for the
  rest of the episode. This is a known limitation documented here because
  it changes how aggressively the Utility-Based agent should use `Pause`.

---

## 3. Stochastic processes

Every stochastic element is fed by a single named `numpy.random.Generator`
built from `numpy.random.SeedSequence(seed).spawn(2)`. There is no implicit
global state. Each process below has a `mode` key in config so that the
semantics remain selectable but pinned.

### 3.1 Node failure — `Node_Failure`

- **Semantics committed for Phase 1:** `mode = "per_step_single_victim"`.
- **Formula:** at every step, sample `u ∼ U(0,1)` from the event generator.
  If `u < events.node_failure_prob` and `|running_tasks| ≥ 1`, pick one
  running task uniformly at random and mark it failed; the task's parent
  job becomes `failed`. Increment `cluster.recent_failures` by 1 (capped at
  `cluster.max_recent_failures = 10`).
- **Active parameter:** `events.node_failure_prob = 0.05` per step (conditional
  on at least one running task). This is the midterm's "Node_Failure with
  P = 0.05" reading and is what `results/canonical_midterm/` was produced
  against.
- **Alternative mode `"per_node_bernoulli"` (documented, not active):** one
  independent `Bernoulli(p)` draw per running task. Under this mode with
  the same `p = 0.05` and 10 running tasks, the per-step probability of at
  least one failure is `1 − (1 − p)^n ≈ 0.40`, which is much harsher. A
  reviewer reading this spec should understand that we adopted the former.
- Why `per_step_single_victim`? It matches the midterm's scalar
  "`P(Node_Failure) = 0.05`" phrasing, keeps the one-failure-per-step
  behaviour of the original code, and is what every committed result is
  calibrated against. The per-node mode is retained in code so the
  sensitivity study can compare the two without editing Python.

### 3.2 Data spike — `Data_Spike`

- **Semantics committed for Phase 1:** `mode = "additive_bump"`.
- **Formula:** at every step, sample `u ∼ U(0,1)`. If `u < events.data_spike_prob`
  and at least one task is in `{waiting, ready}`, sample
  `k ∼ U_int[data_spike_min_tasks, min(data_spike_max_tasks, candidates)]`
  (inclusive), pick `k` candidates uniformly without replacement, and on
  each: `remaining_time += data_spike_duration_bump`; `cpu_demand =
  min(cpu_demand + data_spike_cpu_bump, data_spike_cpu_cap)`.
- **Active parameters:**
  `data_spike_prob = 0.08`, `min_tasks = 1`, `max_tasks = 3`,
  `duration_bump = 1`, `cpu_bump = 1`, `cpu_cap = 6`.
- **Alternative mode `"multiplicative_10x"` (documented, not active):**
  implements the brief's phrasing "10× data volume event" literally — with
  probability `data_spike_prob`, up to `max_tasks` candidates get
  `remaining_time *= multiplier` and `cpu_demand *= multiplier` (clamped
  by `cpu_cap`), with an optional `duration_steps` window during which
  newly spawned tasks inherit the inflated demand. The multiplier and
  duration live at `events.data_spike_multiplier = 10` and
  `events.data_spike_duration_steps = 3` respectively. This mode is not
  active because canonical midterm numerics assume additive bumps; enabling
  it is tracked as a Phase-2 sensitivity study.
- Why `additive_bump`? It matches the behaviour the v0 code actually ran
  during the midterm. Swapping to `multiplicative_10x` would make the
  difficulty of the 300-step budget jump by an order of magnitude — which
  is an interesting sensitivity question, not a silent default.

### 3.3 Spot price — `spot_price_random_walk`

- **Semantics:** mean-zero random walk clamped to `[spot_price_min,
  spot_price_max] = [0.1, 1.0]`.
- **Formula:** at every step, `Δ ∼ U(spot_price_walk_low,
  spot_price_walk_high)`, then
  `spot_price = clip(round(spot_price + Δ, 2), spot_price_min, spot_price_max)`.
- **Active parameters:** `spot_price_walk_low = -0.08`,
  `spot_price_walk_high = 0.08`, `spot_price_min = 0.1`,
  `spot_price_max = 1.0`. Initial draw at episode start is
  `U(cluster.initial_spot_price_low, cluster.initial_spot_price_high) =
  U(0.3, 0.7)`, rounded to 2 decimals.

### 3.4 Scale-boost decay

- **Semantics:** every step decrements `cluster.scale_boost_remaining` by 1;
  when it reaches 0, capacity is reduced by the `Scale_Up`'s
  `cpu_delta / ram_delta`, floored at the cluster baseline. This is
  deterministic (not stochastic), listed here because it runs in the same
  event loop.

---

## 4. Cost function

The cost function is now a single pure function `cost(state, action)` in
`src/cost.py`. It is the sole source of truth for the cost term in the
utility accounting.

**Formula:**

```
cost(state, action) = step_cost(state) + action_cost(action, state)
```

where

```
step_cost(state)     = state.cluster.spot_price *
                       (cost.cpu_weight * cpu_in_use
                        + cost.ram_weight * ram_in_use)

action_cost(a, state) = state.cluster.spot_price * cost.action_costs[a]
```

- `cost.cpu_weight = 0.6`, `cost.ram_weight = 0.4`. These are the
  operational-cost weights; they must sum to 1.0 only by convention (the
  invariant is not enforced in code, but the default does honour it).
- `cost.action_costs` is a flat table keyed by action name. **In Phase 1
  every entry is 0.0.** The table exists to make action-specific pricing
  trivially configurable without further refactor: a reviewer who wants to
  penalise `Scale_Up` by, say, the price of three ticks of its boost can
  change one number.

The episode-level total used for the utility roll-up is

```
total_compute_cost = Σ_t cost(state_t, action_t)
                   = Σ_t step_cost(state_t)          (Phase 1, since every
                                                       action_cost is 0)
```

which is literally the old `estimate_step_cost` summed, so the
`canonical_midterm` numerics are preserved bit-for-bit when this refactor
lands.

Non-goals of the cost function in Phase 1:

- It does not charge for unused reserved capacity (idle cost). That would
  be a very different model and is not in the brief.
- It does not amortise scale-up over the full boost window. A reviewer
  tightening this later would adjust `action_costs.Scale_Up` to a positive
  value and split the charge across the decay window.

---

## 5. Utility weights (α, β, γ)

The episode-level utility is
`U = α · ΣValue(completed) − β · TotalCost − γ · |FailedJobs|`.

### 5.1 Midterm weights (historical, documented)

The midterm's Utility-Based baseline used `α = 1.0, β = 0.4, γ = 0.8`.
These were hard-coded in `src/reflex_agent.py` with no accompanying
rationale in code or README. They are recorded in
`config.utility.midterm_values` so that the midterm's Table 1 run can
still be reproduced exactly via `--config` override and are used by the
archived `results/canonical_midterm/` reference run.

### 5.2 Phase-1 committed weights (active default)

Because the midterm values were not written down as a deliberate choice,
the Phase-1 default switches to `α = 1.0, β = 0.1, γ = 1.0` following the
reviewer-prompt guidance. The reasoning in plain terms:

- **α = 1.0** — unchanged; the episode value of completed jobs is the
  natural reference scale for everything else.
- **β = 0.1** — the midterm's `β = 0.4` made cost dominate utility when
  `total_compute_cost ≈ 600` (cost term `≈ 240`) vs. completed-value
  `≈ 200` (value term `≈ 200`), which is why the midterm's
  `mean_total_utility` was negative for both baselines in Table 1. The
  Phase-1 default `β = 0.1` keeps a meaningful cost penalty (the cost term
  stays on the same order of magnitude as the value term — `60` vs `200`
  at the midterm operating point) without drowning out completions.
- **γ = 1.0** — a failed job is one "unit of value" worth of penalty at
  α = 1.0, which roughly cancels the expected lost value of that job. This
  is the most interpretable setting and matches the example in the Phase-1
  guidance.

These weights are the **active** `utility.alpha/beta/gamma` in
`config/default.yaml`. They are explicitly a Phase-1 starting point; the
Phase-3 sensitivity sweep (`scripts/sweep_utility.py`) is the planned way
to move off them with evidence, not intuition.

### 5.3 Measured effect on the 10-seed comparison

Switching the active weights from `(1.0, 0.4, 0.8)` to `(1.0, 0.1, 1.0)`
shifts mean total utility upward (because the dominant cost penalty is cut
by 4×) and penalises failures by `+0.2` per failure relative to the
midterm. More interestingly, it changes which baseline wins:

| Weights | Reflex mean util | Utility mean util | Utility wins / 10 |
|---|---:|---:|---:|
| Midterm `(1.0, 0.4, 0.8)` | -118.06 | -51.42 | 7 |
| Phase-1 `(1.0, 0.1, 1.0)` |  113.17 | 106.08 | 4 |

(Sources: `results/canonical_midterm/summary.json` and
`results/phase1_baseline/summary.json`, same 10 seeds, same simulator
stream.)

At β = 0.4 the cost term is large enough that Utility-Based's frugality
dominates Reflex's throughput edge. At β = 0.1 the order flips: Reflex's
higher completion rate (0.631 vs 0.512) outweighs Utility-Based's lower
cost (535 vs 781), so Reflex leads on utility in 6 of 10 matched seeds.

Neither answer is "correct" in isolation — which is exactly why (α, β, γ)
must be a deliberately chosen research variable and not a magic constant.
The Phase-3 sweep script (`scripts/sweep_utility.py`) is built to map this
boundary empirically. The two committed runs
(`results/canonical_midterm/`, `results/phase1_baseline/`) are the
before/after anchors for that study.

### 5.4 Per-step RL reward and the risk-term identity

The Self-Learning agent is trained on a per-step reward whose undiscounted sum
is intended to equal the episode utility of §5. Decomposed:

```
r_t = alpha * dValue_t  -  beta * cost(s_t, a_t)  -  gamma * dRisk_t
```

- `dValue_t` = value of jobs newly completed at step `t`. Sums to
  `total_completed_value` (telescopes exactly).
- `cost(s_t, a_t)` = the §4 cost. Sums to `total_compute_cost` (exact).
- `dRisk_t` depends on `rl.reward_risk_mode`:
  - **`counter_delta`** (default; the Phase-5/6 behaviour): the change in the
    normalised `recent_failures` counter. Because that counter decays −1 every
    step, `Σ_t dRisk_t` telescopes to its end-of-episode value (≈0), **not** to
    `failed_jobs`. So with this mode `Σ_t r_t ≠ U`: the learner sees a near-zero
    failure penalty even though the metric charges `gamma * failed_jobs`. This
    is a known mis-specification documented during the hardening pass
    (`scripts/hardening/diag_reward_identity.py`); the default is retained only
    so existing Phase-5/6 results reproduce bit-for-bit.
  - **`failed_jobs_delta`** (corrected): the number of jobs that newly entered
    the `failed` state at step `t`. `Σ_t dRisk_t = failed_jobs`, so
    `Σ_t r_t = U` exactly. This is the mode to use for any run that claims the
    learner optimises the evaluation objective. Pinned by
    `tests/test_reward_identity.py`.

Note the separate, standard gap that the *training* return is discounted
(`δ = rl.delta_discount = 0.99`) while the metric is undiscounted; the identity
above is for the undiscounted sum.

---

## 6. Where this spec lives in code

| Section | Python | YAML |
|---|---|---|
| §1 State variables | `src/state.py` (`StateVector`), `EpisodeState.state_vector()` | `simulator.state_vector.*` |
| §2 Actions | `src/sim_environment.do_action` | `simulator.action_params.*` |
| §3.1 Node failure | `src/sim_environment._apply_node_failure` | `simulator.stochastic_processes.node_failure.*` |
| §3.2 Data spike | `src/sim_environment._apply_data_spike` | `simulator.stochastic_processes.data_spike.*` |
| §3.3 Spot price | `src/sim_environment._apply_spot_price_walk` | `simulator.stochastic_processes.spot_price.*` |
| §4 Cost function | `src/cost.py` (`cost`, `step_cost`, `action_cost`) | `simulator.cost.*` |
| §5 Utility weights | `src/metrics.summarize_episode` | `utility.alpha / beta / gamma`, `utility.midterm_values` |
