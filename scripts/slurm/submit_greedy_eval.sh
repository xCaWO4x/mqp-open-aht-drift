#!/bin/bash
# Submit stationary greedy eval jobs (Q1 + all Q3 variants, no drift).
# Same protocol as Q1: eval_drift.py single-point, sigma=0, 500 eps, seed 42.
# Usage: bash scripts/slurm/submit_greedy_eval.sh

set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"
mkdir -p logs/slurm

# Conda env on compute nodes: override with MQP_CONDA_ENV=myenv (etc.).
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

echo "Submitting Q1 greedy eval (500 episodes)..."
sbatch --export=ALL scripts/slurm/q1_greedy_eval.slurm

echo "Submitting Q3_rw greedy eval (500 episodes)..."
sbatch --export=ALL scripts/slurm/q3_rw_greedy_eval.slurm

echo "Submitting Q3-inf greedy eval (500 episodes)..."
sbatch --export=ALL scripts/slurm/q3_inf_greedy_eval.slurm

echo "Submitting Q3-inf-aux greedy eval (500 episodes)..."
sbatch --export=ALL scripts/slurm/q3_inf_aux_greedy_eval.slurm

echo "Submitting Q3-inf-ema greedy eval (500 episodes)..."
sbatch --export=ALL scripts/slurm/q3_inf_ema_greedy_eval.slurm

echo "Done. squeue -u \$USER"
