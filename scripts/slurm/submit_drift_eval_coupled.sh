#!/bin/bash
# Submit drift evaluation sweep with COUPLED food mode (capability-confound ablation).
# Usage (from repo root):
#   bash scripts/slurm/submit_drift_eval_coupled.sh

set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"
mkdir -p logs/slurm

echo "Submitting drift eval sweep (coupled food — capability-confound ablation)..."
sbatch --export=ALL scripts/slurm/drift_eval_sweep_coupled.slurm
echo "Done. squeue -u \$USER"
