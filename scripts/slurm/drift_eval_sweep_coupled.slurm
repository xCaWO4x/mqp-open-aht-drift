#!/bin/bash
# Drift evaluation sweep with COUPLED food mode (capability-confound ablation).
#
# Same (sigma, theta) grid as drift_eval_sweep.slurm but food difficulty scales
# with agent levels, isolating the pure non-stationarity effect from changes in
# effective task difficulty.
#
# Compare results against drift_eval_sweep (fixed food) to quantify confound.
#SBATCH --job-name=drift-eval-coupled
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
  # shellcheck source=/dev/null
  source "${HOME}/miniconda3/etc/profile.d/conda.sh"
elif [ -f "${HOME}/anaconda3/etc/profile.d/conda.sh" ]; then
  # shellcheck source=/dev/null
  source "${HOME}/anaconda3/etc/profile.d/conda.sh"
fi
conda activate "${MQP_CONDA_ENV:-drift-aht}"

export PYTHONUNBUFFERED=1

CHECKPOINT="${DRIFT_CHECKPOINT:-results/training_lbf_gpl_paper_128k/checkpoints/gpl_final.pt}"
RESULTS_DIR="${DRIFT_RESULTS_DIR:-results/eval_drift_sweep_coupled}"

echo "Checkpoint: ${CHECKPOINT}"
echo "Results dir: ${RESULTS_DIR}"
echo "Food mode: coupled (capability-confound ablation)"
nvidia-smi

python -u experiments/eval_drift.py \
    --checkpoint "${CHECKPOINT}" \
    --config configs/gpl_lbf.yaml \
    --sweep-config configs/drift_sweep_coupled.yaml \
    --sweep \
    --food-mode coupled \
    --results-dir "${RESULTS_DIR}"
