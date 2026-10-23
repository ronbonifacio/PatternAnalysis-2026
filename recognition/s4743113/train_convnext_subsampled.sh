#!/bin/bash
#SBATCH --job-name=convnext-subsample
#SBATCH --partition=comp3710
#SBATCH --account=comp3710
#SBATCH --gres=gpu
#SBATCH --cpus-per-task=8
#SBATCH --time=03:00:00
#SBATCH --output=convnext_subsampled_%j.out
#SBATCH --error=convnext_subsampled_%j.err

set -euo pipefail

source "${HOME}/miniconda3/bin/activate"
conda activate torch

# Support submission from either the repository root or this directory.
submit_dir="${SLURM_SUBMIT_DIR:-${PWD}}"
if [[ -f "${submit_dir}/train_subsampled.py" ]]; then
    train_dir="${submit_dir}"
elif [[ -f "${submit_dir}/recognition/s4743113/train_subsampled.py" ]]; then
    train_dir="${submit_dir}/recognition/s4743113"
else
    echo "Could not locate recognition/s4743113/train_subsampled.py from ${submit_dir}" >&2
    exit 1
fi

cd "${train_dir}"
mkdir -p output/convnext_tiny_slice_subsampling

echo "SLURM job: ${SLURM_JOB_ID:-not available}"
echo "Host: $(hostname)"
echo "Training directory: ${train_dir}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

python -u train_subsampled.py \
    --data-dir /home/groups/comp3710/ADNI/AD_NC \
    --device cuda \
    --num-workers "${SLURM_CPUS_PER_TASK:-8}" \
    "$@"
