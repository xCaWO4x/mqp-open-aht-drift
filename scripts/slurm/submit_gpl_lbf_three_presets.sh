#!/bin/bash
# Submit all three labeled GPL-LBF training jobs (paper / 50% / 20% episode budgets).
# Usage (from repo root):
#   export MQP_CONDA_ENV=myenv   # optional; default drift-aht
#   bash scripts/slurm/submit_gpl_lbf_three_presets.sh

set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"
mkdir -p logs/slurm
# SLURM_SUBMIT_DIR is this directory for all jobs below.

echo "Submitting TRAIN_01 paper full..."
sbatch scripts/slurm/gpl_lbf_train_01_paper_full.slurm
echo "Submitting TRAIN_02 50% episodes..."
sbatch scripts/slurm/gpl_lbf_train_02_episodes_50pct.slurm
echo "Submitting TRAIN_03 20% episodes..."
sbatch scripts/slurm/gpl_lbf_train_03_episodes_20pct.slurm
echo "Done. squeue -u \$USER"
