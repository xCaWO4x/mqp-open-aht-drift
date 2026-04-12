#!/bin/bash
# Submit extended drift sweeps (new result dirs; canonical runs unchanged).
#
# Usage (from repo root):
#   bash scripts/slurm/submit_drift_eval_extended.sh
#
# Submits:
#   1) Fixed food, extended σ  → results/eval_drift_sweep_main_extended/
#   2) Coupled food, extended σ → results/eval_drift_sweep_coupled_extended/
#   3) Fixed food, extended σ + ou.dt=0.1 → results/eval_drift_sweep_main_extended_dt/
#
# Override conda env: export MQP_CONDA_ENV=myenv
# Propagate env into batch jobs:
#   bash scripts/slurm/submit_drift_eval_extended.sh
# (uses sbatch --export=ALL)

set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"
mkdir -p logs/slurm

echo "Submitting extended drift eval sweeps (3 jobs)..."
sbatch --export=ALL scripts/slurm/drift_eval_sweep_extended.slurm
sbatch --export=ALL scripts/slurm/drift_eval_sweep_coupled_extended.slurm
sbatch --export=ALL scripts/slurm/drift_eval_sweep_extended_dt.slurm
echo "Done. squeue -u \$USER"
echo "Canonical comparison dirs unchanged: eval_drift_sweep_main, eval_drift_sweep_coupled"
