#!/usr/bin/env bash
#PBS -N smolvla_kortex_finetune_formal
#PBS -P CFP03-CF-130
#PBS -q auto
#PBS -l walltime=30:00:00
#PBS -l select=1:ncpus=16:mpiprocs=1:ompthreads=16:mem=64gb:ngpus=2
#PBS -j oe
#PBS -o job_log_smolvla_kortex_finetune_formal.out

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

set -euo pipefail

# --- Environment ---
export HF_HOME=/scratch/yuchen.yan/smol_vla/hf_cache
export HF_DATASETS_CACHE=/scratch/yuchen.yan/smol_vla/hf_cache/datasets
export TMPDIR=/tmp
export TOKENIZERS_PARALLELISM=false
# Avoid network timeouts on compute nodes without internet
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
# wandb offline mode writes locally; sync later with: wandb sync <run_dir>
export WANDB_MODE=offline
export WANDB_DIR=/scratch/yuchen.yan/smol_vla/outputs/train/smolvla_kortex_finetune_v1

# --- GPU check ---
echo "=== GPU Status ==="
nvidia-smi --query-gpu=index,name,memory.used,memory.total --format=csv,noheader 2>/dev/null \
  || echo "WARNING: nvidia-smi not available"
echo ""

# --- Clean stale output dir (prevents FileExistsError on re-runs) ---
OUTPUT_DIR="/scratch/yuchen.yan/smol_vla/outputs/train/smolvla_kortex_finetune_v1"
if [ -d "$OUTPUT_DIR" ]; then
    echo "Removing previous output directory: $OUTPUT_DIR"
    rm -rf "$OUTPUT_DIR"
fi

# --- Launch training ---
echo "=== Starting SmolVLA training ==="
uv run --project envs/smolvla vla-arena train --model smolvla --config vla_arena/configs/train/smolvla.yaml
