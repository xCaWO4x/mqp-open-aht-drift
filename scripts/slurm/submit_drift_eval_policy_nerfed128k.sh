#!/bin/bash
# Drift sweeps using the *current* gpl_final.pt (paper-LBF / "nerfed" 128k train).
# Writes under ONE parent folder so older eval_* trees are never touched.
#
# Parent: results/eval_drift_policy_nerfed128k/
#   sweep_main/ sweep_coupled/
#   sweep_main_extended/ sweep_coupled_extended/ sweep_main_extended_dt/
#   bounds_mild/ bounds_reference/ bounds_stress_sigma/ bounds_stress_theta/
#
# Compare later against legacy dirs, e.g. results/eval_drift_sweep_main/ (pre-nerf policy).
#
# Usage (repo root):
#   bash scripts/slurm/submit_drift_eval_policy_nerfed128k.sh
#   MQP_CONDA_ENV=myenv bash scripts/slurm/submit_drift_eval_policy_nerfed128k.sh
#   DRIFT_CHECKPOINT=path/to.pt bash scripts/slurm/submit_drift_eval_policy_nerfed128k.sh

set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"
mkdir -p logs/slurm

OUTROOT="${DRIFT_POLICY_NERFED_ROOT:-results/eval_drift_policy_nerfed128k}"
export EVAL_DRIFT_POLICY_NERFED_ROOT="${OUTROOT}"

echo "Policy-nerfed drift bundle → parent: ${OUTROOT}"
echo "Checkpoint: ${DRIFT_CHECKPOINT:-results/training_lbf_gpl_paper_128k/checkpoints/gpl_final.pt}"
echo "(Legacy eval_drift_sweep_* dirs are not modified.)"

# Canonical + extended + coupled (explicit result dirs; distinct job names in squeue)
sbatch -J drift-nerfed-main \
  --export=ALL,DRIFT_RESULTS_DIR="${OUTROOT}/sweep_main" \
  scripts/slurm/drift_eval_sweep.slurm

sbatch -J drift-nerfed-cpld \
  --export=ALL,DRIFT_RESULTS_DIR="${OUTROOT}/sweep_coupled" \
  scripts/slurm/drift_eval_sweep_coupled.slurm

sbatch -J drift-nerfed-ext-main \
  --export=ALL,DRIFT_RESULTS_DIR="${OUTROOT}/sweep_main_extended" \
  scripts/slurm/drift_eval_sweep_extended.slurm

sbatch -J drift-nerfed-ext-cpld \
  --export=ALL,DRIFT_RESULTS_DIR="${OUTROOT}/sweep_coupled_extended" \
  scripts/slurm/drift_eval_sweep_coupled_extended.slurm

sbatch -J drift-nerfed-ext-dt \
  --export=ALL,DRIFT_RESULTS_DIR="${OUTROOT}/sweep_main_extended_dt" \
  scripts/slurm/drift_eval_sweep_extended_dt.slurm

# Boundary array: per-task subdirs under OUTROOT (see eval_drift_sweep_bounds_array.slurm)
sbatch -J drift-nerfed-bounds \
  --export=ALL,EVAL_DRIFT_POLICY_NERFED_ROOT="${OUTROOT}" \
  scripts/slurm/eval_drift_sweep_bounds_array.slurm

echo "Submitted 6 jobs (5 singles + 1 array 0–3). squeue -u \$USER"
