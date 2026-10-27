#!/bin/bash
#SBATCH --job-name=convnext-gen
#SBATCH --partition=comp3710
#SBATCH --account=comp3710
#SBATCH --gres=gpu
#SBATCH --cpus-per-task=8
#SBATCH --time=03:00:00
#SBATCH --output=convnext_generalized_%j.out
#SBATCH --error=convnext_generalized_%j.err

set -euo pipefail

source "${HOME}/miniconda3/bin/activate"
conda activate torch

submit_dir="${SLURM_SUBMIT_DIR:-${PWD}}"
if [[ -f "${submit_dir}/train_generalized.py" ]]; then
    train_dir="${submit_dir}"
elif [[ -f "${submit_dir}/recognition/s4743113/train_generalized.py" ]]; then
    train_dir="${submit_dir}/recognition/s4743113"
else
    echo "Could not locate recognition/s4743113/train_generalized.py from ${submit_dir}" >&2
    exit 1
fi

cd "${train_dir}"
mkdir -p output/convnext_tiny_64px_generalized

echo "SLURM job: ${SLURM_JOB_ID:-not available}"
echo "Host: $(hostname)"
echo "Training directory: ${train_dir}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

python -u train_generalized.py \
    --data-dir /home/groups/comp3710/ADNI/AD_NC \
    --metadata-path /home/groups/comp3710/ADNI/meta_data_with_label.json \
    --device cuda \
    --num-workers "${SLURM_CPUS_PER_TASK:-8}" \
    "$@"
