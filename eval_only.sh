#!/bin/bash
#PBS -N openvla_oft_libero_eval
#PBS -P CFP03-CF-130
#PBS -q auto
#PBS -l walltime=72:00:00
#PBS -l select=1:ncpus=8:mpiprocs=1:ompthreads=8:mem=64gb:ngpus=1
#PBS -j oe
#PBS -o job_log_openvla_oft_libero_eval_1w.out

# ============================================================================
# Evaluation-only script for OpenVLA-OFT on LIBERO benchmarks.
#
# Resource rationale (1× H200 GPU):
#   - Evaluation is single-process Python (no torchrun), so 1 GPU suffices.
#   - 8 CPUs handle LIBERO MuJoCo simulation + data loading.
#   - 64 GB RAM covers the 7B model (~14 GB fp16) + simulation overhead.
#   - 24 h wall-time is conservative for libero_spatial (10 tasks × 50 trials
#     × ≤220 steps each ≈ 6-12 h depending on inference speed).
#
# Usage:
#   Edit CKPT_STEP and TASK_SUITE below, then:
#     qsub eval_only.sh
# ============================================================================

# ---------- 1. Load environment (mirrors ft_and_eval.sh) ----------
source /app1/ebapps/ebenv_hopper.sh
module load Miniconda3
conda init bash
source ~/.bashrc
module unload gcc/13.1.0
conda activate /scratch/yuchen.yan/envs/openvla-oft

# NOTE: We intentionally do NOT load the CUDA toolkit module here.
# Evaluation needs only the NVIDIA driver (libcuda.so), which is already on the
# default library path. Loading CUDA/12.1.0 then unloading + stripping
# LD_LIBRARY_PATH can accidentally remove the driver path, breaking torch.cuda.
# PyTorch ships its own CUDA runtime, so the toolkit module is unnecessary.
unset CUDA_HOME CUDA_PATH

# Fix dependency conflicts
pip install "numpy<2" "huggingface-hub>=0.19.3,<1.0" -q

# LIBERO setup
LIBERO_DIR="/home/svu/yuchen.yan/yuchen_workspace/LIBERO"
export PYTHONPATH="${LIBERO_DIR}:${PYTHONPATH:-}"

LIBERO_CFG="${HOME}/.libero/config.yaml"
if [[ ! -f "${LIBERO_CFG}" ]]; then
  mkdir -p "$(dirname "${LIBERO_CFG}")"
  python -c "
import os, yaml
br = os.path.join('${LIBERO_DIR}', 'libero', 'libero')
cfg = {
  'benchmark_root': br,
  'bddl_files': os.path.join(br, 'bddl_files'),
  'init_states': os.path.join(br, 'init_files'),
  'datasets': os.path.join(br, '../datasets'),
  'assets': os.path.join(br, 'assets'),
}
with open('${LIBERO_CFG}', 'w') as f:
  yaml.dump(cfg, f)
print('Created LIBERO config')
"
fi

nvidia-smi

cd /home/svu/yuchen.yan/yuchen_workspace/openvla-oft

source ./env.sh

export WANDB_DISABLED=true
export WANDB_MODE=disabled

# Headless MuJoCo rendering via EGL (GPU-accelerated offscreen rendering)
export MUJOCO_GL=egl

# PBS on H200 nodes sets CUDA_VISIBLE_DEVICES to GPU UUIDs (e.g. "GPU-2f82b5bd-...").
# robosuite's EGL context tries int(device_id) and crashes on UUIDs.
# Convert to numeric indices 0,1,...,N-1.  PBS already isolates GPUs via cgroups,
# so numeric indices correctly map to the allocated physical GPUs.
if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]] && [[ "${CUDA_VISIBLE_DEVICES}" == *"GPU-"* ]]; then
  N_GPUS=$(echo "${CUDA_VISIBLE_DEVICES}" | tr ',' '\n' | sed '/^$/d' | wc -l)
  export CUDA_VISIBLE_DEVICES=$(seq -s',' 0 $((N_GPUS - 1)))
  echo "Converted UUID CUDA_VISIBLE_DEVICES to numeric: ${CUDA_VISIBLE_DEVICES}"
fi

# ---------- 2. Checkpoint configuration ----------
RUN_ROOT_DIR="/scratch/yuchen.yan/open_vla/models"

# Run ID produced by finetune.py (must match the fine-tuning run exactly)
RUN_ID="openvla-7b+libero_spatial_no_noops+b8+lr-0.0005+lora-r32+dropout-0.0--image_aug--libero_ft_and_eval_formal"

# >>> SET THE CHECKPOINT STEP YOU WANT TO EVALUATE <<<
CKPT_STEP=10000

CHECKPOINT_DIR="${RUN_ROOT_DIR}/${RUN_ID}--${CKPT_STEP}_chkpt"

# >>> SET THE LIBERO TASK SUITE TO EVALUATE <<<
# Options: libero_spatial, libero_object, libero_goal, libero_10, libero_90
TASK_SUITE="libero_spatial"

# ---------- 3. Validate checkpoint exists ----------
if [[ ! -d "${CHECKPOINT_DIR}" ]]; then
  echo "ERROR: Checkpoint directory not found: ${CHECKPOINT_DIR}"
  echo "Available checkpoints:"
  ls -d "${RUN_ROOT_DIR}/${RUN_ID}"--*_chkpt 2>/dev/null || echo "  (none)"
  exit 1
fi
echo "Evaluating checkpoint: ${CHECKPOINT_DIR}"
echo "Task suite: ${TASK_SUITE}"

# ---------- 4. Run evaluation ----------
python experiments/robot/libero/run_libero_eval.py \
  --pretrained_checkpoint "${CHECKPOINT_DIR}" \
  --task_suite_name "${TASK_SUITE}" \
  --use_l1_regression True \
  --use_diffusion False \
  --use_film False \
  --num_images_in_input 2 \
  --use_proprio True \
  --lora_rank 32 \
  --center_crop True \
  --num_trials_per_task 50 \
  --local_log_dir "./experiments/logs"

echo "================================"
echo "Evaluation finished."
echo "Logs saved to: ./experiments/logs/"
echo "================================"
