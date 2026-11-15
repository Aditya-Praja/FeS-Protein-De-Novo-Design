
#!/bin/bash
#SBATCH --job-name=fes-rfd-40gb
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:1
#SBATCH --constraint=gpu40
#SBATCH --mem=8G
#SBATCH --time=04:00:00
#SBATCH --output=logs/fes-40gb-%j.out
#SBATCH --error=logs/fes-40gb-%j.err

set -euo pipefail

cd "$SLURM_SUBMIT_DIR"
export PROJECT_ROOT="$(pwd -P)"

echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "Start: $(date)"

nvidia-smi

bash scripts/run_rfdiffusion_apptainer.sh

echo "Finish: $(date)"
