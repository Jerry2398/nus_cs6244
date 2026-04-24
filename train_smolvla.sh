#!/bin/bash
#PBS -N smolvla_kortex_finetune
#PBS -P CFP03-CF-130
#PBS -q auto
#PBS -l walltime=30:00:00
#PBS -l select=1:ncpus=16:mpiprocs=1:ompthreads=16:mem=64gb:ngpus=2
#PBS -j oe
#PBS -o job_log_smolvla_kortex_train.out

# 1. Load Environment
source /app1/ebapps/ebenv_hopper.sh
module load Miniconda3
conda init bash
source ~/.bashrc
module load CUDA/12.1.0
nvcc --version
module unload gcc/13.1.0

conda activate /scratch/yuchen.yan/envs/vla_arena

cd /home/svu/yuchen.yan/yuchen_workspace/VLA-Arena

# Redirect HuggingFace caches to /scratch to avoid filling up home directory
export HF_HOME=/scratch/yuchen.yan/smol_vla/hf_cache
export HF_DATASETS_CACHE=/scratch/yuchen.yan/smol_vla/hf_cache/datasets
mkdir -p "$HF_HOME" "$HF_DATASETS_CACHE"

# Use node-local /tmp for multiprocessing temp files (avoids GPFS lock-file errors)
export TMPDIR=/tmp
# Suppress tokenizer fork warnings
export TOKENIZERS_PARALLELISM=false
# Avoid network timeouts on compute nodes
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

# 2. Pre-download dataset and model (avoids timeout during training)
uv run --project envs/smolvla python -c "
from huggingface_hub import snapshot_download
import os
# Download dataset (v3.0 format from HuggingFace)
print('Downloading dataset...')
snapshot_download(repo_id='arjunagarwal28/real_kortex_lerobot', repo_type='dataset',
                  local_dir='/scratch/yuchen.yan/smol_vla/datasets/arjunagarwal28/real_kortex_lerobot')
print('Dataset downloaded.')
# Download pretrained model
print('Downloading pretrained SmolVLA model...')
snapshot_download(repo_id='lerobot/smolvla_base',
                  local_dir='/scratch/yuchen.yan/smol_vla/models/smolvla_base')
print('Model downloaded.')
"

# 3. Convert dataset from LeRobot v3.0 to v2.1 format (skip if already converted)
if [ ! -d "/scratch/yuchen.yan/smol_vla/datasets/real_kortex_lerobot_v21/meta" ]; then
    echo "Converting dataset from v3.0 to v2.1 format..."
    uv run --project envs/smolvla python scripts/convert_kortex_v3_to_v2.py
    echo "Dataset conversion complete."
else
    echo "Converted v2.1 dataset already exists, skipping conversion."
fi

# 4. GPU status check
echo "=== GPU Status ==="
nvidia-smi --query-gpu=index,name,memory.used,memory.total --format=csv,noheader 2>/dev/null || true

# 5. Clean stale output dir (prevents FileExistsError on re-runs)
OUTPUT_DIR="/scratch/yuchen.yan/smol_vla/outputs/train/smolvla_kortex"
if [ -d "$OUTPUT_DIR" ]; then
    echo "Removing previous output directory: $OUTPUT_DIR"
    rm -rf "$OUTPUT_DIR"
fi

# 6. Launch training
uv run --project envs/smolvla vla-arena train --model smolvla --config vla_arena/configs/train/smolvla.yaml
