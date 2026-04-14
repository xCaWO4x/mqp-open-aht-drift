#!/bin/bash
# Submit Q1 and Q3 training jobs.
# Usage: bash scripts/slurm/submit_training.sh

set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"
mkdir -p logs/slurm

echo "Submitting Q1 (baseline training)..."
sbatch scripts/slurm/q1_train.slurm

echo "Submitting Q3 (hardened training)..."
sbatch scripts/slurm/q3_train.slurm

echo "Done. squeue -u \$USER"
