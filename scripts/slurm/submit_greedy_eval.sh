#!/bin/bash
# Submit stationary greedy eval jobs (Q1 + Q3).
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

echo "Submitting Q3 greedy eval (500 episodes)..."
sbatch --export=ALL scripts/slurm/q3_greedy_eval.slurm

echo "Done. squeue -u \$USER"
