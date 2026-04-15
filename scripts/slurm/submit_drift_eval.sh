#!/bin/bash
# Submit Q2 and Q4_hardened drift eval jobs (requires Q1 / Q3_hardened checkpoints).
# Usage: bash scripts/slurm/submit_drift_eval.sh

set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"
mkdir -p logs/slurm

echo "Submitting Q2 (baseline drift eval)..."
sbatch --export=ALL scripts/slurm/q2_drift_eval.slurm

echo "Submitting Q4_hardened drift eval..."
sbatch --export=ALL scripts/slurm/q4_drift_eval.slurm

echo "Done. squeue -u \$USER"
