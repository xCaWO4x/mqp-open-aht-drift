#!/bin/bash
# Q3 — Hardened GPL training (partial obs, latent types, 4 agents, force coop)
# Config: configs/gpl_lbf_hardened.yaml
# Results: results/q3_hardened_stationary/
#SBATCH --job-name=q3-train
#SBATCH --account=xliu14
#SBATCH --partition=short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --constraint=RTX6000B
#SBATCH --time=1-00:00:00
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
echo "Q3: Hardened training (sight=3, no agent levels, 4 agents, force coop)"
nvidia-smi

python -u experiments/train_gpl.py --config configs/gpl_lbf_hardened.yaml
