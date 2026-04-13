#!/bin/bash
# Submit paper-length GPL training on LBF (128k episodes).
# Usage (from repo root): bash scripts/slurm/submit_training_lbf_gpl_paper.sh

set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"
mkdir -p logs/slurm

echo "Submitting training (paper 128k)..."
command sbatch --export=ALL scripts/slurm/training_lbf_gpl_paper_128k.slurm
echo "Done. squeue -u \$USER"
