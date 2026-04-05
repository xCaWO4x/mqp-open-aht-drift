#!/bin/bash
# Stationary LBF — GPL paper-length training (128k episodes, 16 envs).
# Outputs: results/training_lbf_gpl_paper_128k/ (see docs/experiment_artifacts.md)
#SBATCH --job-name=train-lbf-gpl-128k
#SBATCH --account=xliu14
#SBATCH --partition=long
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --constraint=RTX6000B
#SBATCH --time=7-00:00:00
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
nvidia-smi
python -u experiments/train_gpl.py --config configs/gpl_lbf.yaml --device cuda
