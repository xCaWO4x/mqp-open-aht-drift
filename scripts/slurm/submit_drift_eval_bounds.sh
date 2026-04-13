#!/bin/bash
# Submit eval drift boundary sweeps (Slurm array 0–3).
# Usage (from repo root):
#   bash scripts/slurm/submit_drift_eval_bounds.sh
#   DRIFT_CHECKPOINT=.../gpl_final.pt MQP_CONDA_ENV=myenv bash scripts/slurm/submit_drift_eval_bounds.sh

set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"
mkdir -p logs/slurm

echo "Submitting eval drift sweep array (tasks 0–3)..."
command sbatch --export=ALL scripts/slurm/eval_drift_sweep_bounds_array.slurm
echo "Done. squeue -u \$USER"
