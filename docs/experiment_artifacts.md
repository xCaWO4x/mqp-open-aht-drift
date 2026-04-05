# Experiment artifacts: training vs drift eval

This document maps **on-disk layout** under `results/` (gitignored), **Slurm job names**, and where to find data for **capability-conditioned** analyses. Paths are relative to the repository root.

---

## Naming convention

| Prefix | Meaning |
|--------|---------|
| **`training_*`** | **GPL training** on stationary LBF (online RL, checkpoints, TensorBoard). |
| **`eval_drift_*`** | **Frozen-policy evaluation** under OU composition drift (`experiments/eval_drift.py`). No weight updates. |

---

## Training (stationary LBF, paper budget)

| Item | Location |
|------|-----------|
| **Config** | `configs/gpl_lbf.yaml` |
| **Checkpoints & logs** | `results/training_lbf_gpl_paper_128k/` |
| **Final policy file** | `results/training_lbf_gpl_paper_128k/checkpoints/gpl_final.pt` |
| **Periodic checkpoints** | `results/training_lbf_gpl_paper_128k/checkpoints/gpl_ep*.pt` |
| **TensorBoard (if used)** | `runs/training_lbf_gpl_paper_128k/` |
| **Slurm** | `scripts/slurm/training_lbf_gpl_paper_128k.slurm` |
| **Submit helper** | `scripts/slurm/submit_training_lbf_gpl_paper.sh` |

**Job name (Slurm):** `train-lbf-gpl-128k`

---

## Eval drift (sweeps and single grid)

All eval runs load **`gpl_final.pt`** (or override `DRIFT_CHECKPOINT`) and use **`configs/gpl_lbf.yaml`** for model/env shape.

### Canonical grid (`configs/drift_sweep.yaml`)

| Item | Location |
|------|-----------|
| **Results directory** | `results/eval_drift_sweep_main/` |
| **Slurm** | `scripts/slurm/drift_eval_sweep.slurm` |
| **Submit** | `scripts/slurm/submit_drift_eval.sh` |

### Boundary sweeps (array tasks 0–3)

| Task | YAML | Results directory |
|------|------|-------------------|
| 0 — mild | `configs/drift_sweep_bounds_mild.yaml` | `results/eval_drift_sweep_mild/` |
| 1 — reference | `configs/drift_sweep_bounds_reference.yaml` | `results/eval_drift_sweep_reference/` |
| 2 — stress σ | `configs/drift_sweep_bounds_stress_sigma.yaml` | `results/eval_drift_sweep_stress_sigma/` |
| 3 — stress θ | `configs/drift_sweep_bounds_stress_theta.yaml` | `results/eval_drift_sweep_stress_theta/` |

| Item | Location |
|------|-----------|
| **Slurm array** | `scripts/slurm/eval_drift_sweep_bounds_array.slurm` |
| **Submit** | `scripts/slurm/submit_drift_eval_bounds.sh` |

**Job name (Slurm):** `eval-drift-sweep` (array `%a`)

### Files inside each eval drift results folder

| File | Contents |
|------|-----------|
| `drift_eval_grid.csv` | One row per **(σ, θ)**: `mean_return`, `iqm_return`, optional `degradation` (= 1 − IQM / baseline_IQM, baseline = σ=0 row). |
| `drift_eval_episodes.csv` | Per **seed** and **episode**: return, length, **agent_levels**, **food_levels**, **ou_state** (type-frequency vector). |
| `baseline_summary.txt` | Baseline IQM, stability threshold, stable point count. |
| `drift_eval_heatmap.png` | Mean and IQM heatmaps. |
| `drift_degradation_heatmap.png` | Degradation vs baseline with stability contour. |

### Regenerated figures (optional)

| Item | Location |
|------|-----------|
| **Script** | `scripts/plot_drift_degradation.py` |
| **Output** | `results/eval_drift_figures/` (`degradation_stress_sigma.png`, `degradation_stress_theta.png`, `degradation_combined_2x2.png`) |

Run:

```bash
python scripts/plot_drift_degradation.py
```

---

## Capability-conditioned analysis

**Goal:** relate **team capability** (agent levels), **task difficulty** (food levels), and **drift state** (OU simplex) to **returns**, conditioned on **(σ, θ)**.

### Primary table

Use **`results/eval_drift_sweep_<variant>/drift_eval_episodes.csv`**.

| Column | Use |
|--------|-----|
| `sigma`, `theta` | OU regime for the run. |
| `seed` | Independent replicates. |
| `episode` | Index within a grid point. |
| `return` | Learner’s return (episode total). |
| `length` | Episode length (steps). |
| `agent_levels` | Semicolon-separated **LBF levels** for each agent this episode (e.g. `2;1;3`). Parse → **mean**, **min**, **max**, **variance** as capability summaries. |
| `food_levels` | Semicolon-separated **food item levels** sampled for the episode — **harder** food → typically harder coordination. |
| `ou_state` | Semicolon-separated **type-frequency** vector on the simplex after reset — links to **composition** before level sampling. |

### Workflow sketch

1. **Filter** rows by `sigma`, `theta`, or `seed`.
2. **Parse** `agent_levels` and `food_levels` into numeric arrays per row.
3. **Bin or regress** `return` (or `return / length`) against:
   - mean agent level, spread of levels, min teammate level, etc.;
   - mean / max food level;
   - entries of `ou_state` or entropy of the composition distribution.
4. **Aggregate grid** metrics remain in `drift_eval_grid.csv` for **heatmap-level** capability vs drift (no per-episode capability in that file).

### Training checkpoints

To **fine-tune** or **continual-train** under drift (separate experiment), start from:

`results/training_lbf_gpl_paper_128k/checkpoints/gpl_final.pt`

---

## Archive

| Path | Note |
|------|------|
| `results/_archive_incomplete_20260403/` | Legacy incomplete run moved before Slurm paper training. |

---

## Cluster logs

Slurm **stdout/stderr** live under **`logs/slurm/`** (untracked). Filenames include job name and job ID, e.g. `eval-drift-sweep-<jobid>_<taskid>.out`.

---

## What is *not* stored in git

- **`results/`**, **`runs/`**, **`logs/`**, **`wandb/`** are **gitignored**. Reproduce locally or on the cluster using the configs and Slurm scripts above.
