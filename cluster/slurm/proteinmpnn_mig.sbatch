#!/bin/bash
#SBATCH --job-name=fes-mpnn
#SBATCH --partition=mig
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1
#SBATCH --time=00:50:00
#SBATCH --output=logs/fes-mpnn-%j.out
#SBATCH --error=logs/fes-mpnn-%j.err

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
export PROJECT_ROOT="$(pwd -P)"
bash scripts/run_proteinmpnn_apptainer.sh
