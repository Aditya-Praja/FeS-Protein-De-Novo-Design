#!/bin/bash
#SBATCH --job-name=fes-rfd-40g
#SBATCH --partition=gpu
#SBATCH --constraint=gpu40
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=24G
#SBATCH --time=02:00:00
#SBATCH --output=logs/fes-rfd-%j.out
#SBATCH --error=logs/fes-rfd-%j.err

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
export PROJECT_ROOT="$(pwd -P)"
bash scripts/run_rfdiffusion_apptainer.sh
