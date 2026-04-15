#!/bin/bash
# Submit Q3-inf trainings (full, aux-only, EMA-only) on Q3_rw env.
# Usage: bash scripts/slurm/submit_q3_inf_training.sh

set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"
mkdir -p logs/slurm

if [ -z "${MQP_CONDA_ENV:-}" ]; then
  for cand in drift-aht myenv; do
    for prefix in "${HOME}/miniconda3/envs" "${HOME}/anaconda3/envs"; do
      if [ -d "${prefix}/${cand}" ]; then
        MQP_CONDA_ENV="$cand"
        break 2
      fi
    done
  done
fi
: "${MQP_CONDA_ENV:=drift-aht}"
export MQP_CONDA_ENV
echo "Using conda env: ${MQP_CONDA_ENV}"

echo "Submitting Q3-inf (aux + EMA)..."
sbatch --export=ALL scripts/slurm/q3_inf_train.slurm

echo "Submitting Q3-inf-aux..."
sbatch --export=ALL scripts/slurm/q3_inf_aux_train.slurm

echo "Submitting Q3-inf-ema..."
sbatch --export=ALL scripts/slurm/q3_inf_ema_train.slurm

echo "Done. squeue -u \$USER"
