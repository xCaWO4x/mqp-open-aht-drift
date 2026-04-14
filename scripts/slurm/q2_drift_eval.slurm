#!/bin/bash
# Q2 — Baseline GPL + drift evaluation sweep
# Uses Q1 checkpoint with extended (sigma, theta) grid
# Config: configs/gpl_lbf.yaml + configs/drift_sweep.yaml
# Results: results/q2_baseline_drift/
#SBATCH --job-name=q2-drift
#SBATCH --account=xliu14
#SBATCH --partition=short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --time=0-08:00:00
#SBATCH --output=logs/slurm/%x-%j.out
#SBATCH --error=logs/slurm/%x-%j.err

set -euo pipefail
cd "${SLURM_SUBMIT_DIR}" || exit 1
mkdir -p logs/slurm

if [ -f "${HOME}/miniconda3/etc/profile.d/conda.sh" ]; then
  source "${HOME}/miniconda3/etc/profile.d/conda.sh"
elif [ -f "${HOME}/anaconda3/etc/profile.d/conda.sh" ]; then
  source "${HOME}/anaconda3/etc/profile.d/conda.sh"
fi
conda activate "${MQP_CONDA_ENV:-drift-aht}"

export PYTHONUNBUFFERED=1

CHECKPOINT="results/q1_baseline_stationary/checkpoints/gpl_final.pt"
RESULTS_DIR="results/q2_baseline_drift"

echo "Q2: Baseline drift eval"
echo "Checkpoint: ${CHECKPOINT}"
echo "Results: ${RESULTS_DIR}"
nvidia-smi

python -u experiments/eval_drift.py \
    --checkpoint "${CHECKPOINT}" \
    --config configs/gpl_lbf.yaml \
    --sweep-config configs/drift_sweep.yaml \
    --sweep \
    --results-dir "${RESULTS_DIR}"
