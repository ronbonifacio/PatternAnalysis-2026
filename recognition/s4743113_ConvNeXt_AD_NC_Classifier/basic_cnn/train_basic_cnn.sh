#!/bin/bash
#SBATCH --job-name=basic-cnn
#SBATCH --partition=comp3710
#SBATCH --account=comp3710
#SBATCH --gres=gpu
#SBATCH --cpus-per-task=4
#SBATCH --time=03:00:00
#SBATCH --output=basic_cnn_%j.out
#SBATCH --error=basic_cnn_%j.err

set -euo pipefail

source "${HOME}/miniconda3/bin/activate"
conda activate torch

submit_dir="${SLURM_SUBMIT_DIR:-${PWD}}"
if [[ -f "${submit_dir}/train.py" && -d "${submit_dir}/basic_cnn" ]]; then
    train_dir="${submit_dir}"
elif [[ -f "${submit_dir}/../train.py" && -f "${submit_dir}/train.py" ]]; then
    train_dir="$(cd "${submit_dir}/.." && pwd)"
elif [[ -f "${submit_dir}/recognition/s4743113_ConvNeXt_AD_NC_Classifier/train.py" ]]; then
    train_dir="${submit_dir}/recognition/s4743113_ConvNeXt_AD_NC_Classifier"
else
    echo "Could not locate the classifier project from ${submit_dir}" >&2
    exit 1
fi

cd "${train_dir}"
mkdir -p basic_cnn/output

echo "SLURM job: ${SLURM_JOB_ID:-not available}"
echo "Host: $(hostname)"
echo "Training directory: ${train_dir}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

shared_args=(
    --data-dir /home/groups/comp3710/ADNI/AD_NC
    --metadata-path /home/groups/comp3710/ADNI/meta_data_with_label.json
    --output-dir basic_cnn/output
    --device cuda
    --num-workers 2
)

python -u -m basic_cnn.train "${shared_args[@]}" "$@"
python -u -m basic_cnn.evaluate "${shared_args[@]}" "$@"
python -u plot_metrics.py --metrics basic_cnn/output/metrics.json --model-name "Basic CNN"
