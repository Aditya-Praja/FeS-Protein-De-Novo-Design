#!/usr/bin/env bash
# CPU-only RFdiffusion3 run. Do not pass --partition (Della routes the job itself).
# Suggested submit:
#   sbatch --cpus-per-task=16 --mem=64G --time=24:00:00 --array=0 --export=ALL cluster/slurm/rfd3_cpu.sbatch
# Optional: export FES_RFD3_EXTRA_ARGS="key=value ..." to pass extra Hydra overrides to rfd3
# (e.g. a device override if your Foundry version needs one to run on CPU).
#SBATCH --job-name=fes-rfd3-cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=logs/rfd3-cpu-%A_%a.out
#SBATCH --error=logs/rfd3-cpu-%A_%a.err
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from the project root}"
: "${RFD3_ENV:?Set RFD3_ENV to the inference virtual environment}"
: "${RFD3_CHECKPOINT:?Set RFD3_CHECKPOINT to the model checkpoint file}"
: "${FES_RUN_ID:?Set FES_RUN_ID to a prepared run}"
[[ -r "$RFD3_CHECKPOINT" ]] || { echo "Missing checkpoint $RFD3_CHECKPOINT" >&2; exit 2; }
[[ -x "$RFD3_ENV/bin/python" ]] || { echo "Missing inference Python" >&2; exit 2; }
export PATH="$RFD3_ENV/bin:$PATH"
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
export CUDA_VISIBLE_DEVICES=""
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export MKL_NUM_THREADS="$OMP_NUM_THREADS"
python scripts/cluster/run_rfd3_job.py --run-id "$FES_RUN_ID" --index "${SLURM_ARRAY_TASK_ID:-0}" --execute
