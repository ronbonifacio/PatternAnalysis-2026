#!/bin/bash
#SBATCH --job-name=convnext-profile
#SBATCH --partition=comp3710
#SBATCH --account=comp3710
#SBATCH --gres=gpu
#SBATCH --cpus-per-task=1
#SBATCH --time=00:10:00
#SBATCH --output=convnext_profile_%j.out
#SBATCH --error=convnext_profile_%j.err

set -euo pipefail

source "${HOME}/miniconda3/bin/activate"
conda activate torch

submit_dir="${SLURM_SUBMIT_DIR:-${PWD}}"
if [[ -f "${submit_dir}/profile_resources.py" ]]; then
    profile_dir="${submit_dir}"
elif [[ -f "${submit_dir}/recognition/s4743113_ConvNeXt_AD_NC_Classifier/profile_resources.py" ]]; then
    profile_dir="${submit_dir}/recognition/s4743113_ConvNeXt_AD_NC_Classifier"
else
    echo "Could not locate profile_resources.py from ${submit_dir}" >&2
    exit 1
fi

cd "${profile_dir}"
echo "SLURM job: ${SLURM_JOB_ID:-not available}"
echo "Host: $(hostname)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

python -u profile_resources.py --device cuda "$@"
