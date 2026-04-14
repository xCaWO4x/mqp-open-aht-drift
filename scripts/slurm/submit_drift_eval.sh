#!/bin/bash
# Submit Q2 and Q4 drift eval jobs (requires Q1/Q3 checkpoints).
# Usage: bash scripts/slurm/submit_drift_eval.sh

set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"
mkdir -p logs/slurm

echo "Submitting Q2 (baseline drift eval)..."
sbatch scripts/slurm/q2_drift_eval.slurm

echo "Submitting Q4 (hardened drift eval)..."
sbatch scripts/slurm/q4_drift_eval.slurm

echo "Done. squeue -u \$USER"
