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

### Local ad-hoc (CLI default)

If you run `experiments/eval_drift.py` without `--results-dir`, outputs go to **`results/drift_eval/`**. That is **not** the Slurm sweep layout; use it for one-off checks.

### Canonical grid (`configs/drift_sweep.yaml`)

| Item | Location |
|------|-----------|
| **Results directory** | `results/eval_drift_sweep_main/` |
| **Slurm** | `scripts/slurm/drift_eval_sweep.slurm` |
| **Submit** | `scripts/slurm/submit_drift_eval.sh` |

**σ values:** `[0, 0.01, 0.05, 0.1, 0.2, 0.5]` — same baseline row (σ=0) as the original paper-style stationary reference. These directories are **left in place** for comparison when you run extended sweeps.

### Extended diffusion sweeps (higher σ, separate result dirs)

Use these to probe **where performance breaks** without overwriting canonical CSVs.

| Variant | Sweep YAML | Results directory | Slurm script |
|--------|------------|-------------------|--------------|
| Fixed food, **extended σ** | `configs/drift_sweep_extended.yaml` | `results/eval_drift_sweep_main_extended/` | `drift_eval_sweep_extended.slurm` |
| Coupled food, **extended σ** | `configs/drift_sweep_coupled_extended.yaml` | `results/eval_drift_sweep_coupled_extended/` | `drift_eval_sweep_coupled_extended.slurm` |
| Fixed food, extended σ + **`ou.dt=0.1`** | `configs/drift_sweep_extended_dt.yaml` | `results/eval_drift_sweep_main_extended_dt/` | `drift_eval_sweep_extended_dt.slurm` |

**Extended σ list** (includes all canonical values plus a tail):  
`[0, 0.01, 0.05, 0.1, 0.2, 0.5, 1.0, 1.5, 2.0, 3.0]`  
**θ** unchanged vs canonical: `[0.05, 0.15, 0.3, 0.5, 1.0]`.

**Submit all three** (recommended on cluster):

```bash
bash scripts/slurm/submit_drift_eval_extended.sh
```

Each job is **10 × 5 × 100 × 5 = 25,000** episodes (vs **15,000** for the canonical 6×5 grid).

#### OU scaling (why σ and `dt` both matter)

Discrete OU step in `drift/ou_process.py`:

\[
\mathbf{x} \leftarrow \mathbf{x} + \theta(\boldsymbol{\mu}-\mathbf{x})\,\Delta t + \sigma\sqrt{\Delta t}\,\boldsymbol{\xi}
\]

Then project onto the simplex. **Innovation RMS** scales as **σ√Δt** (per coordinate, before projection).

| Setting | Typical Δt | Effective innovation scale (×σ) |
|--------|------------|----------------------------------|
| Training / canonical eval | `0.01` from `configs/gpl_lbf.yaml` | **√0.01 = 0.1** |
| `drift_sweep_extended_dt.yaml` | `0.1` | **√0.1 ≈ 0.316** |

Sweep YAML may include an optional top-level **`ou:`** block; keys are merged into the training config **for that sweep only** (see `experiments/eval_drift.py`).

#### Capability-confound reports (extended grids)

After sweeps finish:

```bash
python experiments/analyze_capability_confound.py \
  --episodes-csv results/eval_drift_sweep_main_extended/drift_eval_episodes.csv \
  --out-dir results/capability_confound_main_extended
# likewise: capability_confound_coupled_extended, capability_confound_main_extended_dt
```

**Observed outcome (representative run):** fixed-food **extended** (`dt=0.01`) stayed mostly stable vs baseline (**47/50** cells with degradation below 10%); **larger `dt`** produced many unstable cells (**33/50**) especially at σ ∈ {2, 3} and some at σ=0.01. Coupled extended had **50/50** stable on that metric. Interpretation: pushing **effective simplex mobility** (`dt`) hurts more than pushing **σ alone** on this checkpoint; coupled food removes sharp **IQM drops vs baseline** even at high σ.

### Quadrant comparison figures (Q1 vs Q4)

Paper-style **2×2** (legacy pre–paper-LBF policy vs **paper-LBF–trained** policy under the same drift YAMLs):

```bash
python scripts/plot_quadrant_drift_figures.py --out-dir results/eval_drift_figures_quadrants
```

Writes under **`results/eval_drift_figures_quadrants/`** (gitignored). Files:

| Quadrant | Meaning (in this repo) | Degradation / stress PNGs | Capability confound (same `analyze_capability_confound.py` as before) |
|----------|-------------------------|---------------------------|------------------------------------------------------------------------|
| **Q1** | Legacy policy drift evals (`eval_drift_sweep_*`, checkpoint at eval time) | `q1_main_canonical_degradation.png`, `q1_coupled_canonical_degradation.png`, `q1_main_extended_degradation.png`, `q1_stress_sigma_theta_combined_2x2.png` | `q1_capability_confound/` (`capability_confound_diagnostics.png`, `success_rate_heatmap.png`, `variance_decomposition.png`, `capability_confound_report.txt`) |
| **Q2** | Train paper-LBF, eval easy-LBF | *Not produced* (no separate result dir; mismatched train/eval not recommended) | — |
| **Q3** | Train easy-LBF, eval paper-LBF only | *Not produced* | — |
| **Q4** | Paper-LBF policy + same drift YAMLs (`eval_drift_policy_nerfed128k/`) | `q4_main_canonical_degradation.png`, `q4_coupled_canonical_degradation.png`, `q4_main_extended_degradation.png`, `q4_stress_sigma_theta_combined_2x2.png` | `q4_capability_confound/` (same filenames as Q1 folder) |

**Note:** If `drift_eval_grid.csv` has no `degradation` column (e.g. baseline IQM = 0), `plot_quadrant_drift_figures.py` recomputes degradation from **`mean_return`** at σ=0 for visualization only (Q4 canonical case).

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

Use **`results/eval_drift_sweep_<variant>/drift_eval_episodes.csv`** (e.g. `_main`, `_coupled`, or extended variants: `_main_extended`, `_coupled_extended`, `_main_extended_dt`).

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
- **Training dumps** (checkpoints, episode buffers, TensorBoard) live under `results/training_*` and `runs/` — **do not delete** them when pushing code; they are excluded from the remote by design so the repo stays clone-friendly.

### Copying artifacts to another machine (optional)

From the repo root (after a sweep or training run):

```bash
rsync -a --progress results/training_lbf_gpl_paper_128k/ user@host:path/to/backup/training_lbf_gpl_paper_128k/
rsync -a --progress results/eval_drift_sweep_main_extended/ user@host:path/to/backup/
rsync -a --progress logs/slurm/ user@host:path/to/backup/slurm-logs/
```

Git branches carry **configs, Slurm, and docs only**; bundle `results/` and `logs/` separately when archiving a paper snapshot.

### Policy trained on paper LBF (“nerfed”) — isolated drift bundle

After **128k training** with current `configs/gpl_lbf.yaml` (50-step LBF, `force_coop=false`), re-run **all** standard sweeps into a **fresh tree** so legacy CSVs under `eval_drift_sweep_main/` etc. stay untouched for comparison:

| Output (under one parent) | Contents |
|---------------------------|----------|
| **`results/eval_drift_policy_nerfed128k/sweep_main`** | Canonical fixed-food grid (`drift_sweep.yaml`) |
| **`.../sweep_coupled`** | Canonical coupled food (`drift_sweep_coupled.yaml`) |
| **`.../sweep_main_extended`** | Extended σ, fixed food |
| **`.../sweep_coupled_extended`** | Extended σ, coupled |
| **`.../sweep_main_extended_dt`** | Extended σ + `ou.dt=0.1` |
| **`.../bounds_{mild,reference,stress_sigma,stress_theta}`** | Array boundary presets |

Submit:

```bash
bash scripts/slurm/submit_drift_eval_policy_nerfed128k.sh
```

Override parent dir: `DRIFT_POLICY_NERFED_ROOT=results/my_run_name bash scripts/slurm/submit_drift_eval_policy_nerfed128k.sh`

Loads **`results/training_lbf_gpl_paper_128k/checkpoints/gpl_final.pt`** unless `DRIFT_CHECKPOINT` is set. Bounds array uses **`EVAL_DRIFT_POLICY_NERFED_ROOT`** (set by the submit script).
