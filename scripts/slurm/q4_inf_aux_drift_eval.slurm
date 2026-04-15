#!/bin/bash
# Q4-inf-aux: Drift eval on Q3-inf-aux checkpoint (aux head only, no EMA)
# Config: configs/gpl_lbf_q3_inf_aux.yaml + configs/drift_sweep.yaml
# Results: results/q4_inf_aux_rw_drift/
#SBATCH --job-name=q4-inf-aux
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

CHECKPOINT="results/q3_inf_aux_rw_stationary/checkpoints/gpl_final.pt"
RESULTS_DIR="results/q4_inf_aux_rw_drift"

echo "Q4-inf-aux: Drift eval on aux-only checkpoint"
echo "Checkpoint: ${CHECKPOINT}"
echo "Results dir: ${RESULTS_DIR}"
nvidia-smi

python -u experiments/eval_drift.py \
    --checkpoint "${CHECKPOINT}" \
    --config configs/gpl_lbf_q3_inf_aux.yaml \
    --sweep-config configs/drift_sweep.yaml \
    --sweep \
    --results-dir "${RESULTS_DIR}"
