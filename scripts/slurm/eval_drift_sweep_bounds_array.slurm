#!/bin/bash
# Eval drift: four (sigma, theta) boundary sweeps (array tasks 0–3).
# Inference only; loads frozen GPL checkpoint. See docs/experiment_artifacts.md
#SBATCH --job-name=eval-drift-sweep
#SBATCH --account=xliu14
#SBATCH --partition=short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --constraint=RTX6000B
#SBATCH --time=0-16:00:00
#SBATCH --array=0-3
#SBATCH --output=logs/slurm/eval-drift-sweep-%A_%a.out
#SBATCH --error=logs/slurm/eval-drift-sweep-%A_%a.err

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

sleep $((SLURM_ARRAY_TASK_ID * 60))
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"

CHECKPOINT="${DRIFT_CHECKPOINT:-results/training_lbf_gpl_paper_128k/checkpoints/gpl_final.pt}"

case "${SLURM_ARRAY_TASK_ID}" in
  0)
    SWEEP_YAML="configs/drift_sweep_bounds_mild.yaml"
    RESULTS_DIR="${DRIFT_RESULTS_DIR:-results/eval_drift_sweep_mild}"
    LABEL="mild"
    ;;
  1)
    SWEEP_YAML="configs/drift_sweep_bounds_reference.yaml"
    RESULTS_DIR="${DRIFT_RESULTS_DIR:-results/eval_drift_sweep_reference}"
    LABEL="reference"
    ;;
  2)
    SWEEP_YAML="configs/drift_sweep_bounds_stress_sigma.yaml"
    RESULTS_DIR="${DRIFT_RESULTS_DIR:-results/eval_drift_sweep_stress_sigma}"
    LABEL="stress_sigma"
    ;;
  3)
    SWEEP_YAML="configs/drift_sweep_bounds_stress_theta.yaml"
    RESULTS_DIR="${DRIFT_RESULTS_DIR:-results/eval_drift_sweep_stress_theta}"
    LABEL="stress_theta"
    ;;
  *)
    echo "Invalid SLURM_ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID}"
    exit 1
    ;;
esac

echo "=== eval drift sweep: ${LABEL} (task ${SLURM_ARRAY_TASK_ID}) ==="
echo "Checkpoint: ${CHECKPOINT}"
echo "Sweep YAML: ${SWEEP_YAML}"
echo "Results dir: ${RESULTS_DIR}"
nvidia-smi

python -u experiments/eval_drift.py \
    --checkpoint "${CHECKPOINT}" \
    --config configs/gpl_lbf.yaml \
    --sweep-config "${SWEEP_YAML}" \
    --sweep \
    --results-dir "${RESULTS_DIR}"

echo "=== done ${LABEL} ==="
