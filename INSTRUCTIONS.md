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

## Current state

### Training: DONE

Three training runs completed on HPC (SLURM):

| Run | Episodes | Env steps | Train avg | Eval IQM | Checkpoint |
|-----|----------|-----------|-----------|----------|------------|
| 01 (paper) | 128,000 | 5,590,401 | 0.390 | 0.417 | `results/gpl_lbf_train_01_paper_full/checkpoints/gpl_final.pt` |
| 02 (50%) | 64,000 | 2,895,299 | 0.376 | 0.259 | `results/gpl_lbf_train_02_episodes_50pct/checkpoints/gpl_final.pt` |
| 03 (20%) | ~26,000 | ~1.2M | 0.045 | ~0 | `results/gpl_lbf_train_03_episodes_20pct/checkpoints/gpl_final.pt` |

The 128k result (0.390 avg, 0.417 IQM) is **in the paper's reported range** (~0.35-0.45 for GPL-Q with random teammates in 8x8-3p-3f no-force-coop). The 50% checkpoint is usable but weaker. The 20% checkpoint is undertrained (epsilon barely decayed).

**Reward scale context:** In LBF, collecting a food item yields `food.level / max_food_level`. With food levels {2, 3} and max=3, expected reward per food = 0.8. Theoretical max (solo-load all 3 food) = 2.4. Practical ceiling with random teammates and varying agent levels is ~0.5-0.7. The 0.39-0.42 range means GPL collects ~55% of its fair share.

### Drift eval sweep: READY TO RUN

Everything is implemented and waiting to be submitted:

```bash
cd open-aht-drift-clean
bash scripts/slurm/submit_drift_eval.sh
```

This runs `experiments/eval_drift.py --sweep` with the 128k checkpoint across a 6x5 (sigma, theta) grid:
- **sigmas**: [0.0, 0.01, 0.05, 0.1, 0.2, 0.5]  (sigma=0 is stationary baseline)
- **thetas**: [0.05, 0.15, 0.3, 0.5, 1.0]
- **100 episodes x 5 seeds** per grid point = 15,000 total episodes
- Estimated runtime: ~1h on GPU, ~3-4h on CPU

**Outputs** (in `results/drift_eval_sweep/`):
- `drift_eval_grid.csv` — sigma, theta, mean_return, iqm_return, degradation
- `drift_eval_episodes.csv` — per-episode details (returns, agent/food levels, OU state)
- `baseline_summary.txt` — baseline IQM and stability region count
- `drift_eval_heatmap.png` — side-by-side mean and IQM return heatmaps
- `drift_degradation_heatmap.png` — degradation heatmap with stability boundary contour

**Degradation is computed as** `1 - IQM / baseline_IQM` where baseline is the sigma=0 row (stationary uniform composition). A grid point is "stable" if degradation < 10%.

---

## Key architecture decisions (locked)

These are documented in `docs/research_proposal.md` and `project_aht_drift_decisions.md`. Do not change without explicit approval:

- **LBF-only phase** (no Wolfpack until Step 19)
- **K=3** agent types mapping to LBF levels {1, 2, 3}
- **Food config**: fixed distribution {level 2: 60%, level 3: 40%}, independent of agent population
- **OU process**: dt=0.01, mean-reversion to uniform
- **GPL-Q variant** (no SPI), matching paper defaults for LBF

---

## Codebase map

```
open-aht-drift-clean/
  agents/
    gpl/
      gpl_agent.py          # Top-level GPL agent (Algorithm 5, Polyak soft update, advance_hidden)
      type_inference.py      # LSTM type inference (FC->ReLU->FC->LSTM->ReLU, paper Fig 7a)
      agent_model.py         # GNN agent model (RFM + MLP_eta, paper Fig 7d-f, gnn_hidden=70)
      joint_action_value.py  # CG Q-network (MLP_beta/delta, paper Fig 7b-c, pairwise_rank=5)
    baselines/
      random_agent.py
  configs/
    gpl_lbf.yaml             # Training config (128k ep, 16 envs, paper hyperparams)
    gpl_lbf_train_02_*.yaml  # 50% variant
    gpl_lbf_train_03_*.yaml  # 20% variant
    drift_sweep.yaml          # Sweep grid (6 sigma x 5 theta, 100 eps, 5 seeds)
    gpl_wolfpack.yaml         # Future: Wolfpack config
  drift/
    ou_process.py             # OU process on K-simplex with Euclidean projection
    drift_schedule.py         # EMPTY — Step 17
    belief_tracker.py         # EMPTY — Step 18
  envs/
    drift_wrapper.py          # Gym wrapper: advances OU, samples composition, injects levels
    env_utils.py              # PREPROCESS: LBF obs parsing (FOOD FIRST then agents), Wolfpack
  eval/
    logger.py                 # TensorBoard + optional wandb logging
    metrics.py                # EMPTY — Step 16
    stability_region.py       # EMPTY — Step 16
  experiments/
    train_gpl.py              # Training loop (16 parallel envs, per-env hidden state mgmt)
    eval_drift.py             # Drift eval (single point + sweep, degradation, heatmaps)
    pilot_degradation.py      # Legacy pilot script (still uses RandomAgent)
  scripts/slurm/
    gpl_lbf_train_01_paper_full.slurm
    gpl_lbf_train_02_episodes_50pct.slurm
    gpl_lbf_train_03_episodes_20pct.slurm
    drift_eval_sweep.slurm    # ← Submit this for the drift eval
    submit_drift_eval.sh
    submit_gpl_lbf_three_presets.sh
  tests/
    test_gpl_forward.py       # 37 tests for GPL modules
    test_preprocess.py        # 13 tests for PREPROCESS (LBF + Wolfpack)
    test_drift_wrapper.py     # 23 tests for DriftWrapper
    test_ou_process.py        # 10 tests for OU process
```

---

## Critical implementation details

### LBF observation format

**FOOD COMES FIRST.** Each agent's obs from `_make_gym_obs`:
```
[food_0(y,x,level), food_1(y,x,level), ..., self(y,x,level), other_0(y,x,level), ...]
```
- Food features: `obs[0 : n_food*3]`
- Agent features: `obs[n_food*3 : n_food*3 + n_agents*3]`  (self first)

This was a critical bug that took multiple sessions to find. The old code had it backwards (agents first). The fix is in `preprocess_lbf()` in `envs/env_utils.py`.

### act() does NOT update hidden states

`GPLAgent.act()` selects an action but discards LSTM hidden state outputs. This is intentional: it ensures `train_step_online()` sees the same pre-step hidden state for both QJOINT (line 14) and QV (line 15) of Algorithm 5.

During **training**, `train_step_online()` is the sole hidden state updater.

During **evaluation**, call `agent.advance_hidden(B_np)` after `agent.act()` to advance all three hidden state paths (q, agent, q_target).

### Per-env hidden state management (parallel envs)

Training uses 16 parallel environments. Each env has its own tuple of LSTM hidden states `(hidden_q, hidden_agent, hidden_q_target)`. Before processing each env, the agent's internal hidden states are swapped in from `env_hidden[env_idx]`; after processing, they're swapped back out. Hidden states are zeroed on episode reset.

### Polyak soft update

Target networks use Polyak averaging (`tau=1e-3`) every step, not hard copy every N steps:
```
phi_target = (1 - tau) * phi_target + tau * phi
```

### LBF spawn_players() permutes levels

When you inject specific levels per player via `min_player_level = max_player_level = [L1, L2, L3]`, LBF's `spawn_players()` randomly permutes the level array before assigning. This preserves the overall composition but scrambles which player gets which level. This is fine for our study.

---

## What to do next

### Immediate: Submit drift eval sweep (Step 15)
```bash
bash scripts/slurm/submit_drift_eval.sh
```
Wait for results. Look for:
- Does degradation increase monotonically with sigma?
- Does higher theta (faster mean-reversion) protect against high sigma?
- Where is the stability boundary (10% degradation contour)?

### After sweep results: Analysis (Step 16)
- Fill in `eval/metrics.py`: IQM with bootstrap CIs, degradation AUC, recovery speed
- Fill in `eval/stability_region.py`: interpolated boundary curves, region area computation
- Write formal analysis of the heatmap results

### Later steps
- **Step 17**: `drift/drift_schedule.py` — continuous, step, and sudden drift modes
- **Step 18**: Drift-aware adapter — Bayesian belief tracker or D.3 autoencoder
- **Step 19**: Wolfpack experiments
- **Step 20**: Paper-ready plots and statistical analysis

---

## HPC details

- **Account**: xliu14
- **Conda env**: drift-aht (override with `MQP_CONDA_ENV`)
- **GPU constraint**: RTX6000B (training), any GPU fine for eval
- **Conda paths**: checks both `~/miniconda3` and `~/anaconda3`
- **Key env vars**: `DRIFT_CHECKPOINT`, `DRIFT_RESULTS_DIR`, `MQP_CONDA_ENV`

---

## Files NOT to touch

- Anything in `runs/` or `results/` (active training/eval logs and checkpoints)
- `logs/slurm/` (SLURM output logs)
- `.claude/` project memory files
