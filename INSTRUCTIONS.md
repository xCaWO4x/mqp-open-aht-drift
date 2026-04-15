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

## Experiment layout: 4 quadrants + inference variant

|  | Stationary (no drift) | Drift |
|---|---|---|
| **Baseline** (Rahman paper config) | **Q1** | **Q2** |
| **Hardened** (partial obs, latent types, 4p, force coop) | **Q3_hardened** | **Q4_hardened** |
| **Hardened + inference** (aux head + EMA) | **Q3-inf** | **Q4-inf** |

- **Q1/Q2** use `configs/gpl_lbf.yaml` (full obs, 3 agents, no force coop)
- **Q3_hardened/Q4_hardened** use `configs/gpl_lbf_q3_hardened.yaml` (sight=3, no agent levels, 4 agents, force coop)
- **Q3-inf/Q4-inf** use `configs/gpl_lbf_q3_inf.yaml` (same hardened nerfs + auxiliary level prediction + EMA belief tracker)
- **Q2/Q4/Q4-inf** use `configs/drift_sweep.yaml` (10σ × 5θ extended grid)

### What to do now

```bash
# Step 1: submit Q1 + Q3_hardened + Q3-inf training (parallel)
bash scripts/slurm/submit_training.sh
sbatch scripts/slurm/q3_inf_train.slurm

# Step 2: after training completes, submit drift evals
bash scripts/slurm/submit_drift_eval.sh
sbatch scripts/slurm/q4_inf_drift_eval.slurm
```

See `docs/experiment_artifacts.md` for full artifact map.

---

## Key architecture decisions (locked)

- **LBF-only phase** (no Wolfpack until Step 19)
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
  agents/baselines/
    random_agent.py
  configs/
    gpl_lbf.yaml           # Q1/Q2 baseline config
    gpl_lbf_q3_hardened.yaml   # Q3_hardened / Q4_hardened config
    gpl_lbf_q3_inf.yaml    # Q3-inf/Q4-inf hardened + inference config
    drift_sweep.yaml        # Drift eval grid (10σ × 5θ)
    gpl_wolfpack.yaml       # Future
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
    q2_drift_eval.slurm     # Q2 baseline drift eval
    q3_hardened_train.slurm # Q3_hardened training
    q3_inf_train.slurm      # Q3-inf hardened + inference training
    q4_drift_eval.slurm     # Q4_hardened drift eval (job: q4_hardened_drift)
    q4_inf_drift_eval.slurm # Q4-inf hardened + inference drift eval
    submit_training.sh      # Submit Q1 + Q3_hardened
    submit_drift_eval.sh    # Submit Q2 + Q4
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
