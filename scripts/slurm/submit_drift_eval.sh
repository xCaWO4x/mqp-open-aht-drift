#!/bin/bash
# Submit drift evaluation sweep.
# Usage (from repo root):
#   bash scripts/slurm/submit_drift_eval.sh
#   DRIFT_CHECKPOINT=path/to/other.pt bash scripts/slurm/submit_drift_eval.sh

set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"
mkdir -p logs/slurm

echo "Submitting drift eval sweep..."
sbatch --export=ALL scripts/slurm/drift_eval_sweep.slurm
echo "Done. squeue -u \$USER"
