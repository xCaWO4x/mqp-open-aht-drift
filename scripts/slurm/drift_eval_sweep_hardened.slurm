#!/bin/bash
# Drift eval sweep on HARDENED checkpoint
# Same (sigma, theta) grid as canonical sweep but uses the hardened policy
# Config: configs/gpl_lbf_hardened.yaml + configs/drift_sweep.yaml
# Artifacts: results/eval_drift_sweep_hardened/
#SBATCH --job-name=drift-eval-hardened
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

CHECKPOINT="${DRIFT_CHECKPOINT:-results/training_lbf_gpl_hardened/checkpoints/gpl_final.pt}"
RESULTS_DIR="${DRIFT_RESULTS_DIR:-results/eval_drift_sweep_hardened}"

echo "Checkpoint: ${CHECKPOINT}"
echo "Results dir: ${RESULTS_DIR}"
echo "Config: hardened (partial obs, 4 agents, force coop)"
nvidia-smi

python -u experiments/eval_drift.py \
    --checkpoint "${CHECKPOINT}" \
    --config configs/gpl_lbf_hardened.yaml \
    --sweep-config configs/drift_sweep.yaml \
    --sweep \
    --results-dir "${RESULTS_DIR}"
