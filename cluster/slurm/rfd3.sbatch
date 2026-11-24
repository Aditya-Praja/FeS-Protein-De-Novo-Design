#!/usr/bin/env bash
#SBATCH --job-name=fes-rfd3
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --output=logs/rfd3-%A_%a.out
#SBATCH --error=logs/rfd3-%A_%a.err
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from the project root}"
: "${RFD3_ENV:?Set RFD3_ENV to the inference virtual environment}"
: "${RFD3_CHECKPOINT:?Set RFD3_CHECKPOINT to the model checkpoint file}"
: "${FES_RUN_ID:?Set FES_RUN_ID to a prepared run}"
[[ -r "$RFD3_CHECKPOINT" ]] || { echo "Missing checkpoint $RFD3_CHECKPOINT" >&2; exit 2; }
[[ -x "$RFD3_ENV/bin/python" ]] || { echo "Missing inference Python" >&2; exit 2; }
export PATH="$RFD3_ENV/bin:$PATH"
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
python scripts/cluster/run_rfd3_job.py --run-id "$FES_RUN_ID" --index "${SLURM_ARRAY_TASK_ID:-0}" --execute
