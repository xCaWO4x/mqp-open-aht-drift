#!/bin/bash
# Drift evaluation sweep: evaluate 128k GPL checkpoint across (sigma, theta) grid
# Uses configs/drift_sweep.yaml (6 sigmas x 5 thetas x 100 eps x 5 seeds = 15,000 eps)
# Inference only — much lighter than training: ~1-2h on GPU, ~3-4h on CPU.
#SBATCH --job-name=drift-eval-sweep
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

# Use the 128k (paper-full) checkpoint by default; override with DRIFT_CHECKPOINT env var
CHECKPOINT="${DRIFT_CHECKPOINT:-results/gpl_lbf_train_01_paper_full/checkpoints/gpl_final.pt}"
RESULTS_DIR="${DRIFT_RESULTS_DIR:-results/drift_eval_sweep}"

echo "Checkpoint: ${CHECKPOINT}"
echo "Results dir: ${RESULTS_DIR}"
nvidia-smi

python -u experiments/eval_drift.py \
    --checkpoint "${CHECKPOINT}" \
    --config configs/gpl_lbf.yaml \
    --sweep-config configs/drift_sweep.yaml \
    --sweep \
    --results-dir "${RESULTS_DIR}"
