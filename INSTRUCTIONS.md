# Handoff: open-aht-drift-clean

Context for continuing this project. Read this before making any changes.

---

## What this project is

MQP studying **GPL robustness under OU-process-driven population composition drift** in Level-Based Foraging (LBF) environments. The core question: how does GPL (Graph-based Policy Learning, Rahman et al. ICML 2021 / JMLR 2023, arXiv:2006.10412) degrade when the team composition drifts over time via an Ornstein-Uhlenbeck process on the type-frequency simplex?

**GPL** is a three-component architecture for open ad hoc teamwork:
- `TypeInferenceModel` (LSTM): maps per-agent state features B_t to continuous type vectors theta
- `AgentModel` (GNN + MLP): predicts teammate action distributions from type vectors
- `JointActionValueModel` (CG): factorises joint Q-values via coordination graph with singular + pairwise terms

The learner (agent 0) uses GPL; all teammates act randomly (standard open AHT protocol).

---

## Experiment layout: 4 quadrants + three inference ablations

|  | Stationary (no drift) | Drift |
|---|---|---|
| **Baseline** (Rahman paper config — full obs) | **Q1** | **Q2** |
| **RW baseline** (info nerfs only — 3p, no force coop, sight=3, `observe_agent_levels=false`) | **Q3_rw** | **Q4_rw** |
| **RW + aux head only** | **Q3-inf-aux** | **Q4-inf-aux** |
| **RW + EMA only** | **Q3-inf-ema** | **Q4-inf-ema** |
| **RW + aux + EMA (combined)** | **Q3-inf** | **Q4-inf** |

- **Q1/Q2** use `configs/gpl_lbf.yaml` (full obs, 3 agents, no force coop)
- **Q3_rw/Q4_rw** use `configs/gpl_lbf_q3_rw.yaml` (sight=3, `observe_agent_levels=false`, 3 agents, no force coop)
- **Q3-inf-aux / Q4-inf-aux** use `configs/gpl_lbf_q3_inf_aux.yaml` (RW + aux head only: `ema_dim=0`, `aux_weight=0.1`)
- **Q3-inf-ema / Q4-inf-ema** use `configs/gpl_lbf_q3_inf_ema.yaml` (RW + EMA only: `ema_dim=16`, `aux_weight=0.0`)
- **Q3-inf / Q4-inf** use `configs/gpl_lbf_q3_inf.yaml` (combined: `ema_dim=16`, `aux_weight=0.1`)
- **Q2/Q4_rw/Q4-inf\***  use `configs/drift_sweep.yaml` (10σ × 5θ extended grid)

The earlier plan contemplated a stronger "hardened" variant (4 agents, `force_coop=true`). Preliminary runs were unstable and it was dropped for scope; Q3/Q4 unqualified in this document refers to Q3_rw/Q4_rw unless explicitly stated.

### What to do now

```bash
# Step 1: submit Q1 + Q3_rw + Q3-inf training (parallel)
bash scripts/slurm/submit_training.sh
sbatch scripts/slurm/q3_inf_train.slurm

# Step 2: after training completes, submit drift evals
bash scripts/slurm/submit_drift_eval.sh
sbatch scripts/slurm/q4_inf_drift_eval.slurm
```

See `docs/experiment_artifacts.md` for full artifact map.

---

## Standardized evaluation protocol (systemized)

Use this protocol for all drift comparisons so metrics are apples-to-apples:

- **Evaluator:** `experiments/eval_drift.py --sweep`
- **Grid:** from `configs/drift_sweep.yaml`
  - `sigmas`: 10 values
  - `thetas`: 5 values
  - total: **50 grid points**
- **Per-point sampling:** `n_episodes=100`, `n_seeds=5` (500 episodes per grid point)
- **Primary metric:** `iqm_return` (interquartile mean)
- **Aggregation:** per-seed IQM is computed first, then averaged across the 5 seeds for each `(sigma, theta)` cell
- **Baseline for degradation:** sigma=0 row average IQM (`baseline_iqm`)

Expected outputs for each eval run:

- `drift_eval_grid.csv` (per-cell mean + IQM + degradation)
- `drift_eval_episodes.csv` (per-episode records)
- `baseline_summary.txt` (`baseline_iqm`, stability threshold, stable-point count)

**Stationary greedy eval (no drift, same protocol as Q1):** `experiments/eval_drift.py` in single-point mode — `sigma=0.0`, `theta=0.15`, `n_episodes=500`, `seed=42`, greedy policy (`epsilon=0`). Output: `eval_s0.0_t0.15_seed42.csv` under each run’s `results/..._greedy_eval_500/` directory.

**Q3_rw vs Q3-inf (and aux/ema):** same baseline task as Q1 with info nerfs — `n_agents=3`, `force_coop=false`, `sight=3`, `observe_agent_levels=false` (`configs/gpl_lbf_q3_rw.yaml` vs `gpl_lbf_q3_inf*.yaml`). Only the inference configs add aux head / EMA (and `GPLAgentInf`).

| Run | Slurm | Checkpoint | Config |
|-----|-------|------------|--------|
| Q1 | `scripts/slurm/q1_greedy_eval.slurm` | `results/q1_baseline_stationary/checkpoints/gpl_final.pt` | `configs/gpl_lbf.yaml` |
| Q3_rw | `scripts/slurm/q3_rw_greedy_eval.slurm` | `results/q3_rw_stationary/checkpoints/gpl_final.pt` | `configs/gpl_lbf_q3_rw.yaml` |
| Q4_rw (greedy, same policy as Q3_rw) | `scripts/slurm/q4_rw_greedy_eval.slurm` | `results/q3_rw_stationary/checkpoints/gpl_final.pt` | `configs/gpl_lbf_q3_rw.yaml` |
| Q3-inf | `scripts/slurm/q3_inf_greedy_eval.slurm` | `results/q3_inf_rw_stationary/checkpoints/gpl_final.pt` | `configs/gpl_lbf_q3_inf.yaml` |
| Q3-inf-aux | `scripts/slurm/q3_inf_aux_greedy_eval.slurm` | `results/q3_inf_aux_rw_stationary/checkpoints/gpl_final.pt` | `configs/gpl_lbf_q3_inf_aux.yaml` |
| Q3-inf-ema | `scripts/slurm/q3_inf_ema_greedy_eval.slurm` | `results/q3_inf_ema_rw_stationary/checkpoints/gpl_final.pt` | `configs/gpl_lbf_q3_inf_ema.yaml` |

Submit Q1 + all Q3 stationary greedy jobs: `bash scripts/slurm/submit_greedy_eval.sh`

---

## Q3-variant to Q4-variant mapping

Run each drift eval against its corresponding stationary checkpoint:

- `Q3_rw -> Q4_rw`: `scripts/slurm/q4_rw_drift_eval.slurm`
- `Q3-inf -> Q4-inf`: `scripts/slurm/q4_inf_drift_eval.slurm`
- `Q3-inf-aux -> Q4-inf-aux`: `scripts/slurm/q4_inf_aux_drift_eval.slurm`
- `Q3-inf-ema -> Q4-inf-ema`: `scripts/slurm/q4_inf_ema_drift_eval.slurm`

Note: `q4_rw_greedy_eval.slurm` is **stationary greedy** for the Q3_rw policy (output under `results/q4_rw_stationary_greedy_eval_500/`). The `q4_rw` drift sweep is `scripts/slurm/q4_rw_drift_eval.slurm`.

---

## Current empirical findings (as of the aux-weight sweep + levels-on sanity check)

Short summary to keep any continuation honest about what the runs actually showed. Full account with numbers and figures lives in `docs/research_doc.md` §4.7.

- **Q3_rw / Q4_rw is the strongest variant in the information-nerfed regime.** On both the 500-episode greedy stationary protocol and the full 50-cell drift sweep, adding the auxiliary head, the EMA context, or both did not beat the plain RW baseline at any setting.
- **The auxiliary head was not learning at all in the original code.** In `agents/gpl/gpl_agent_inf.py::train_step_online_inf`, the auxiliary forward was running with `hidden_agent = None`, so the LSTM saw a single frame with no temporal context. CE was mathematically pinned at `ln 3 ≈ 1.0986` across all 128k episodes of every aux-bearing variant.
- **Fix applied.** The aux forward now reuses the pre-update carried hidden state. After the fix, the head produces non-uniform predictions but mean CE drops only marginally below uniform (≈6%), and this does not improve when `aux_weight` is scaled up to 2.0 (20× the original). This rules out gradient magnitude as the bottleneck.
- **Levels-on sanity check (job 1952857).** A 12k-episode training with the same aux-head code path but `observe_agent_levels = true` drops mean raw CE from 1.033 (levels-off, w=0.1) to **0.902** (levels-on, w=0.1) — a clean 18% reduction vs uniform (≈6× the reduction we got from any `aux_weight` sweep on the levels-off side). Plumbing is therefore confirmed: when the level signal is present in the input, the head does extract it. **However, CE does NOT collapse to ≈0** as a trivial read-off would predict — it plateaus in the 0.87–0.94 range with no downward trend across 12k episodes.
- **Interpretation (two composable mechanisms).** (i) *Observability:* at `sight = 3`, `observe_agent_levels = false` with random teammates, `(y, x)` trajectories over 50-step episodes carry very little mutual information with teammate level — this is why scaling `aux_weight` does nothing on the levels-off side. (ii) *Multi-task interference:* the aux head shares `type_net_agent` with the dominant agent-model loss, which is action-prediction against random teammates — a loss that is indifferent to level (random is random). The shared encoder actively sheds level information in favour of action-predictive features even when level is in the input, which explains why levels-on CE plateaus at 0.90 rather than collapsing to 0. Under the nerf the first mechanism dominates; lift the nerf and the second becomes the binding constraint.
- **Central future-work direction.** Sweep `sight` to locate the observability threshold at which inference methods begin to work, then characterise how that threshold interacts with drift. See `docs/research_doc.md` §8 for the full list.

The short-check scripts that produced these results are:

```
scripts/slurm/q3_inf_aux_shortcheck.slurm            # fixed aux, aux_weight=0.1
scripts/slurm/q3_inf_aux_shortcheck_w0p5.slurm       # aux_weight sweep
scripts/slurm/q3_inf_aux_shortcheck_w1p0.slurm
scripts/slurm/q3_inf_aux_shortcheck_w2p0.slurm
scripts/slurm/q3_inf_aux_shortcheck_levels_on.slurm  # levels-on sanity check
```

Each writes to its own `results/q3_inf_aux_shortcheck*/` directory so comparisons are apples-to-apples.

### Q3-inf-aux sight sweep (observability curve)

Follow-up to the "central future-work direction" above. Trains Q3-inf-aux at `sight ∈ {4, 5, 6, 7}` (single seed, everything else identical to `configs/gpl_lbf_q3_inf_aux.yaml`) to locate where, if anywhere, the aux head starts to extract level signal once observability is relaxed from the collapsed `sight=3` point. Evaluation is the same stationary 500-episode greedy protocol used for every other Q3 variant.

Configs (one per sight point):

```
configs/gpl_lbf_q3_inf_aux_sight4.yaml
configs/gpl_lbf_q3_inf_aux_sight5.yaml
configs/gpl_lbf_q3_inf_aux_sight6.yaml
configs/gpl_lbf_q3_inf_aux_sight7.yaml
```

Slurm scripts:

```
scripts/slurm/q3_inf_aux_sight{4,5,6,7}_train.slurm        # 128k-ep training
scripts/slurm/q3_inf_aux_sight{4,5,6,7}_greedy_eval.slurm  # 500-ep greedy eval
```

Submit:

```bash
# Trainings only (parallel):
bash scripts/slurm/submit_sight_sweep.sh

# Trainings + chained greedy evals (afterok dependency per sight):
bash scripts/slurm/submit_sight_sweep.sh --with-eval

# Evals alone, once trainings have finished:
bash scripts/slurm/submit_sight_sweep_eval.sh
```

Plot IQM vs sight once the evals are done (defaults to both-variant overlay if Q3_rw sweep data is also present):

```bash
python scripts/plot_sight_sweep.py                      # both variants (default)
python scripts/plot_sight_sweep.py --variant inf_aux    # Q3-inf-aux only
python scripts/plot_sight_sweep.py --variant rw         # Q3_rw only
```

The plot script reads each sight point's `eval_s0.0_t0.15_seed42.csv`, computes IQM with a 95% bootstrap CI, and writes `figures/sight_sweep_*_iqm.png` plus a sidecar CSV.

### Q3_rw sight sweep (aux-head confound isolation)

Companion to the Q3-inf-aux sweep above. Trains the pure Q3_rw baseline (no aux head, no EMA) at `sight ∈ {4, 5, 6, 7}` (single seed, everything else identical to `configs/gpl_lbf_q3_rw.yaml`). The sight=3 Q3_rw point already exists under `results/q3_rw_stationary_greedy_eval_500/`.

Two-curve overlay (`Q3_rw` vs `Q3-inf-aux`) at each sight separates three worlds:

- Both curves flat in sight → observability is saturated on 8×8 LBF; neither sight nor aux head helps. Supports the multi-task-interference story.
- Q3_rw rises with sight while Q3-inf-aux stays flat → the aux head is actively eating performance — clean "aux head failed" claim.
- Both curves rise with sight → observability is the binding constraint; the earlier single-curve noise was seed-level variance.

Configs (one per sight point):

```
configs/gpl_lbf_q3_rw_sight4.yaml
configs/gpl_lbf_q3_rw_sight5.yaml
configs/gpl_lbf_q3_rw_sight6.yaml
configs/gpl_lbf_q3_rw_sight7.yaml
```

Slurm scripts:

```
scripts/slurm/q3_rw_sight{4,5,6,7}_train.slurm        # 128k-ep training (train_gpl.py)
scripts/slurm/q3_rw_sight{4,5,6,7}_greedy_eval.slurm  # 500-ep greedy eval
```

Submit:

```bash
# Trainings only (parallel):
bash scripts/slurm/submit_rw_sight_sweep.sh

# Trainings + chained greedy evals (afterok dependency per sight):
bash scripts/slurm/submit_rw_sight_sweep.sh --with-eval

# Evals alone, once trainings have finished:
bash scripts/slurm/submit_rw_sight_sweep_eval.sh
```

---

## Key architecture decisions (locked)

- **LBF environment** (Level-Based Foraging)
- **K=3** agent types mapping to LBF levels {1, 2, 3}
- **Food config**: fixed distribution {level 2: 60%, level 3: 40%}, independent of agent population
- **OU process**: dt=0.01, mean-reversion to uniform
- **GPL-Q variant** (no SPI), matching paper defaults for LBF

---

## Codebase map

```
open-aht-drift-clean/
  agents/gpl/
    gpl_agent.py          # Top-level GPL agent (Algorithm 5, Polyak soft update, advance_hidden)
    gpl_agent_inf.py      # GPL + auxiliary inference head + EMA (Q3-inf/Q4-inf)
    auxiliary_head.py     # Auxiliary level prediction MLP
    type_inference.py      # LSTM type inference
    agent_model.py         # GNN agent model
    joint_action_value.py  # CG Q-network
  configs/
    gpl_lbf.yaml           # Q1/Q2 baseline config
    gpl_lbf_q3_rw.yaml        # Q3_rw / Q4_rw baseline config
    gpl_lbf_q3_inf.yaml    # Q3-inf/Q4-inf rw + inference config
    drift_sweep.yaml        # Drift eval grid (10σ × 5θ)
  drift/
    ou_process.py           # OU process on K-simplex with Euclidean projection
    ema_tracker.py          # EMA belief tracker (population context)
  envs/
    drift_wrapper.py        # Gym wrapper: advances OU, samples composition
    env_utils.py            # PREPROCESS: LBF obs parsing (FOOD FIRST), supports observe_agent_levels
  eval/
    logger.py               # TensorBoard + optional wandb
  experiments/
    train_gpl.py            # Training loop (16 parallel envs)
    train_gpl_inf.py        # Training loop with aux head + EMA (Q3-inf)
    eval_drift.py           # Drift eval (single point + sweep)
    analyze_capability_confound.py  # Capability-confound analysis
  scripts/slurm/
    q1_train.slurm          # Q1 baseline training
    q1_greedy_eval.slurm    # Q1 stationary greedy (500 eps, no drift)
    q2_drift_eval.slurm     # Q2 baseline drift eval
    q3_rw_greedy_eval.slurm # Q3_rw stationary greedy
    q4_rw_greedy_eval.slurm # Q3_rw policy, alternate results dir (job: q4-rw-greedy-eval)
    q3_inf_train.slurm      # Q3-inf training
    q3_inf_*_greedy_eval.slurm  # Q3-inf / aux / ema stationary greedy
    q4_rw_drift_eval.slurm  # Q4_rw drift eval (job: q4-rw-drift)
    q4_inf_drift_eval.slurm # Q4-inf drift eval
    submit_training.sh      # Submit Q1 + Q3_rw
    submit_greedy_eval.sh   # Submit Q1 + all Q3 stationary greedy evals
    submit_drift_eval.sh    # Submit Q2 + Q4_rw
  tests/
    test_gpl_forward.py     # 37 tests
    test_preprocess.py      # 13 tests (19 with hardened)
    test_drift_wrapper.py   # 23 tests
    test_ou_process.py      # 10 tests
```

---

## Critical implementation details

### LBF observation format

**FOOD COMES FIRST.** Each agent's obs from `_make_gym_obs`:
```
[food_0(y,x,level), food_1(y,x,level), ..., self(y,x,level), other_0(y,x,level), ...]
```
- Food features: `obs[0 : n_food*3]`
- Agent features: `obs[n_food*3 :]` — self first, then others
- With `observe_agent_levels=false`: agent features are (y,x) only, 2 per agent instead of 3

### act() does NOT update hidden states

`GPLAgent.act()` selects an action but discards LSTM hidden state outputs. During **evaluation**, call `agent.advance_hidden(B_np)` after `agent.act()`.

### Per-env hidden state management (parallel envs)

Training uses 16 parallel environments. Each env has its own LSTM hidden states swapped in/out before/after processing.

### Polyak soft update

Target networks: `phi_target = (1 - tau) * phi_target + tau * phi` with tau=1e-3 every step.

### Aux-head forward must reuse the pre-update hidden state

In `agents/gpl/gpl_agent_inf.py::train_step_online_inf`, the auxiliary-loss forward pass through `self.type_net_agent` **must** use the carried `self._hidden_agent` captured *before* `super().train_step_online(...)` is called, not `None`. Using `None` here zeros the LSTM state and the aux head collapses to uniform output (CE ≈ ln 3) for the entire run — this was the cause of the aux head failing to learn across Q3-inf-aux, Q3-inf-ema (which doesn't use aux, irrelevant), and Q3-inf. Do not "clean up" this line back to `hidden=None`.

### LBF spawn_players() permutes levels

When injecting levels via `min_player_level = max_player_level = [L1, L2, ...]`, LBF randomly permutes. This is fine.

---

## HPC details

- **Account**: xliu14
- **Conda env**: drift-aht (override with `MQP_CONDA_ENV`)
- **GPU constraint**: RTX6000B (training), any GPU fine for eval
- **Conda paths**: checks both `~/miniconda3` and `~/anaconda3`

---

## Files NOT to touch

- Anything in `runs/` or `results/`
- `logs/slurm/`
- `.claude/` project memory files
