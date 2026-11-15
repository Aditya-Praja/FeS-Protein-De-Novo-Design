
#!/bin/bash
#SBATCH --job-name=fes-rfd-mig
#SBATCH --partition=mig
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1
#SBATCH --time=00:50:00
#SBATCH --output=logs/fes-rfd-%j.out
#SBATCH --error=logs/fes-rfd-%j.err

set -euo pipefail

cd "$SLURM_SUBMIT_DIR"
export PROJECT_ROOT="$(pwd -P)"

bash scripts/run_rfdiffusion_apptainer.sh
