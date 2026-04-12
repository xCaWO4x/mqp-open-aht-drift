#!/bin/bash
# Extended σ sweep with ou.dt=0.1 (see configs/drift_sweep_extended_dt.yaml).
# Fixed food only; separate results dir from eval_drift_sweep_main_extended/.
#SBATCH --job-name=drift-eval-ext-dt
#SBATCH --account=xliu14
#SBATCH --partition=short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --time=0-12:00:00
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
RESULTS_DIR="${DRIFT_RESULTS_DIR:-results/eval_drift_sweep_main_extended_dt}"

echo "Checkpoint: ${CHECKPOINT}"
echo "Results dir: ${RESULTS_DIR}"
echo "Sweep: configs/drift_sweep_extended_dt.yaml (ou.dt=0.1)"
nvidia-smi

python -u experiments/eval_drift.py \
    --checkpoint "${CHECKPOINT}" \
    --config configs/gpl_lbf.yaml \
    --sweep-config configs/drift_sweep_extended_dt.yaml \
    --sweep \
    --results-dir "${RESULTS_DIR}"
