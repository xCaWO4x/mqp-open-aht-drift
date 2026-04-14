# Experiment artifacts

All results live under `results/` (gitignored). Four quadrants:

|  | Stationary (no drift) | Drift |
|---|---|---|
| **Baseline** (Rahman paper config) | **Q1** | **Q2** |
| **Hardened** (partial obs, latent types, 4p, force coop) | **Q3** | **Q4** |

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

## Files inside each results folder

### Training (Q1, Q3)

| File | Contents |
|------|----------|
| `checkpoints/gpl_final.pt` | Final trained policy |
| `checkpoints/gpl_ep*.pt` | Periodic checkpoints |
| `training_run.log` | Per-episode returns and losses |

### Drift eval (Q2, Q4)

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
# Step 1: train Q1 + Q3 in parallel
bash scripts/slurm/submit_training.sh

# Step 2: after training completes, eval Q2 + Q4
bash scripts/slurm/submit_drift_eval.sh
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
```

---

## What is *not* stored in git

`results/`, `runs/`, `logs/`, `wandb/` are gitignored. Reproduce on cluster using configs and SLURM scripts above.
