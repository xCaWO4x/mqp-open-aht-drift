# Experiment artifacts

All results live under `results/` (gitignored). Core quadrants + inference variant:

|  | Stationary (no drift) | Drift |
|---|---|---|
| **Baseline** (Rahman paper config) | **Q1** | **Q2** |
| **Hardened** (partial obs, latent types, 4p, force coop) | **Q3** | **Q4** |
| **Hardened + inference** (aux head + EMA tracker) | **Q3-inf** | **Q4-inf** |

---

## Q1 — Baseline stationary

| Item | Location |
|------|----------|
| **Config** | `configs/gpl_lbf.yaml` |
| **Results** | `results/q1_baseline_stationary/` |
| **Checkpoint** | `results/q1_baseline_stationary/checkpoints/gpl_final.pt` |
| **Slurm** | `scripts/slurm/q1_train.slurm` |
| **Job name** | `q1-train` |

Settings: 8×8 grid, 3 agents, sight=8 (full), observe_agent_levels=true, force_coop=false, obs_dim=12.

---

## Q2 — Baseline + drift

| Item | Location |
|------|----------|
| **Config** | `configs/gpl_lbf.yaml` + `configs/drift_sweep.yaml` |
| **Results** | `results/q2_baseline_drift/` |
| **Checkpoint** | Q1's `gpl_final.pt` |
| **Slurm** | `scripts/slurm/q2_drift_eval.slurm` |
| **Job name** | `q2-drift` |

Sweep: 10 σ × 5 θ × 100 episodes × 5 seeds = 25,000 episodes.

---

## Q3 — Hardened stationary

| Item | Location |
|------|----------|
| **Config** | `configs/gpl_lbf_hardened.yaml` |
| **Results** | `results/q3_hardened_stationary/` |
| **Checkpoint** | `results/q3_hardened_stationary/checkpoints/gpl_final.pt` |
| **Slurm** | `scripts/slurm/q3_train.slurm` |
| **Job name** | `q3-train` |

Nerfs vs Q1:

| Setting | Q1 (baseline) | Q3 (hardened) |
|---------|---------------|---------------|
| `sight` | 8 (full grid) | 3 (partial) |
| `observe_agent_levels` | true | false |
| `n_agents` | 3 | 4 |
| `force_coop` | false | true |
| `obs_dim` | 12 | 11 |

---

## Q4 — Hardened + drift

| Item | Location |
|------|----------|
| **Config** | `configs/gpl_lbf_hardened.yaml` + `configs/drift_sweep.yaml` |
| **Results** | `results/q4_hardened_drift/` |
| **Checkpoint** | Q3's `gpl_final.pt` |
| **Slurm** | `scripts/slurm/q4_drift_eval.slurm` |
| **Job name** | `q4-drift` |

Same sweep grid as Q2.

---

## Q3-inf — Hardened + auxiliary inference (stationary)

| Item | Location |
|------|----------|
| **Config** | `configs/gpl_lbf_q3_inf.yaml` |
| **Results** | `results/q3_inf_hardened_stationary/` |
| **Checkpoint** | `results/q3_inf_hardened_stationary/checkpoints/gpl_final.pt` |
| **Slurm** | `scripts/slurm/q3_inf_train.slurm` |
| **Training script** | `experiments/train_gpl_inf.py` |
| **Job name** | `q3-inf-train` |

Same hardened nerfs as Q3, plus:

| Addition | Description |
|----------|-------------|
| **Auxiliary level head** | `agents/gpl/auxiliary_head.py` — cross-entropy prediction of teammate level class from type embeddings, weight=0.1 |
| **EMA belief tracker** | `drift/ema_tracker.py` — running average of type embeddings across episodes, dim=16, α=0.1, concatenated to obs |
| **GPLAgentInf** | `agents/gpl/gpl_agent_inf.py` — wraps GPLAgent with both additions |

Effective obs_dim = 11 (base) + 16 (EMA) = 27.

Compare Q3-inf vs Q3 to measure how much learned inference recovers.

---

## Q4-inf — Hardened + inference + drift

| Item | Location |
|------|----------|
| **Config** | `configs/gpl_lbf_q3_inf.yaml` + `configs/drift_sweep.yaml` |
| **Results** | `results/q4_inf_hardened_drift/` |
| **Checkpoint** | Q3-inf's `gpl_final.pt` (no retraining) |
| **Slurm** | `scripts/slurm/q4_inf_drift_eval.slurm` |
| **Job name** | `q4-inf-drift` |

Same sweep grid as Q2/Q4. Tests whether the learned inference also helps under drift.

---

## Files inside each results folder

### Training (Q1, Q3, Q3-inf)

| File | Contents |
|------|----------|
| `checkpoints/gpl_final.pt` | Final trained policy |
| `checkpoints/gpl_ep*.pt` | Periodic checkpoints |
| `training_run.log` | Per-episode returns and losses |

Q3-inf checkpoints also include `aux_head` weights and `ema_tracker` state.

### Drift eval (Q2, Q4, Q4-inf)

| File | Contents |
|------|----------|
| `drift_eval_grid.csv` | Per-(σ,θ): mean_return, iqm_return, degradation |
| `drift_eval_episodes.csv` | Per-episode: return, length, agent_levels, food_levels, ou_state |
| `baseline_summary.txt` | Baseline IQM, stability threshold, stable count |
| `drift_eval_heatmap.png` | Mean and IQM heatmaps |
| `drift_degradation_heatmap.png` | Degradation heatmap with stability contour |

---

## Submit helpers

```bash
# Step 1: train Q1 + Q3 + Q3-inf in parallel
bash scripts/slurm/submit_training.sh
sbatch scripts/slurm/q3_inf_train.slurm

# Step 2: after training completes, eval Q2 + Q4 + Q4-inf
bash scripts/slurm/submit_drift_eval.sh
sbatch scripts/slurm/q4_inf_drift_eval.slurm
```

---

## Capability-confound analysis (post-eval)

```bash
python experiments/analyze_capability_confound.py \
  --episodes-csv results/q2_baseline_drift/drift_eval_episodes.csv \
  --out-dir results/q2_confound

python experiments/analyze_capability_confound.py \
  --episodes-csv results/q4_hardened_drift/drift_eval_episodes.csv \
  --out-dir results/q4_confound

python experiments/analyze_capability_confound.py \
  --episodes-csv results/q4_inf_hardened_drift/drift_eval_episodes.csv \
  --out-dir results/q4_inf_confound
```

---

## What is *not* stored in git

`results/`, `runs/`, `logs/`, `wandb/` are gitignored. Reproduce on cluster using configs and SLURM scripts above.
