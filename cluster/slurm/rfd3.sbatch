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
# FES_CHUNK designs per array task (default 1). Task T runs design indices T*CHUNK .. T*CHUNK+CHUNK-1,
# so a small array covers many designs and stays under per-user job-submit limits.
# Set FES_TOTAL to the number of designs so the final task stops at the end.
CHUNK="${FES_CHUNK:-1}"
TOTAL="${FES_TOTAL:-$(( ${SLURM_ARRAY_TASK_ID:-0} * CHUNK + CHUNK ))}"
start=$(( ${SLURM_ARRAY_TASK_ID:-0} * CHUNK ))
status=0
for (( i = start; i < start + CHUNK && i < TOTAL; i++ )); do
    python scripts/cluster/run_rfd3_job.py --run-id "$FES_RUN_ID" --index "$i" --execute || status=$?
done
exit $status
