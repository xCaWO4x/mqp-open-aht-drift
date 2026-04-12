#!/bin/bash
# Submit hardened variant: training then drift eval.
#
# Usage (from repo root):
#   bash scripts/slurm/submit_hardened.sh           # training only
#   bash scripts/slurm/submit_hardened.sh train      # training only
#   bash scripts/slurm/submit_hardened.sh eval       # eval only (needs checkpoint)
#   bash scripts/slurm/submit_hardened.sh all        # training, then eval after
#
# For "all", eval is submitted with --dependency=afterok so it waits for training.

set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"
mkdir -p logs/slurm

MODE="${1:-train}"

if [ "$MODE" = "train" ] || [ "$MODE" = "all" ]; then
    echo "Submitting hardened training (128k episodes, partial obs, 4 agents, force coop)..."
    TRAIN_JOB=$(sbatch --parsable scripts/slurm/training_lbf_gpl_hardened.slurm)
    echo "Training job: ${TRAIN_JOB}"
fi

if [ "$MODE" = "eval" ]; then
    echo "Submitting hardened drift eval sweep..."
    sbatch scripts/slurm/drift_eval_sweep_hardened.slurm
fi

if [ "$MODE" = "all" ]; then
    echo "Submitting hardened drift eval (depends on training job ${TRAIN_JOB})..."
    sbatch --dependency=afterok:"${TRAIN_JOB}" scripts/slurm/drift_eval_sweep_hardened.slurm
fi

echo "Done. squeue -u \$USER"
