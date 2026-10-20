#!/bin/bash
#SBATCH --job-name=convnext-adni
#SBATCH --partition=comp3710
#SBATCH --account=comp3710
#SBATCH --gres=gpu
#SBATCH --cpus-per-task=8
#SBATCH --time=02:00:00
#SBATCH --output=convnext_adni_%j.out
#SBATCH --error=convnext_adni_%j.err

set -euo pipefail

# Use the same Conda environment as the working COMP3710 SLURM job.
source "${HOME}/miniconda3/bin/activate"
conda activate torch

# Support submission from either the repository root or this directory.
submit_dir="${SLURM_SUBMIT_DIR:-${PWD}}"
if [[ -f "${submit_dir}/train.py" ]]; then
    train_dir="${submit_dir}"
elif [[ -f "${submit_dir}/recognition/s4743113/train.py" ]]; then
    train_dir="${submit_dir}/recognition/s4743113"
else
    echo "Could not locate recognition/s4743113/train.py from ${submit_dir}" >&2
    exit 1
fi

cd "${train_dir}"
mkdir -p output/convnext_tiny

echo "SLURM job: ${SLURM_JOB_ID:-not available}"
echo "Host: $(hostname)"
echo "Training directory: ${train_dir}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

python -u train.py \
    --device cuda \
    --num-workers "${SLURM_CPUS_PER_TASK:-8}"
