
#!/bin/bash
#SBATCH --job-name=fes-rfd-mig
#SBATCH --partition=mig
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --output=logs/fes-mig-%j.out
#SBATCH --error=logs/fes-mig-%j.err

set -euo pipefail

cd "$SLURM_SUBMIT_DIR"
export PROJECT_ROOT="$(pwd -P)"

echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "Start: $(date)"

nvidia-smi

bash scripts/run_rfdiffusion_apptainer.sh

echo "Finish: $(date)"
