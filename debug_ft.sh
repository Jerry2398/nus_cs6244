# When `module load CUDA/12.1.0` was run in the parent shell, it adds the toolkit's
# stubs directory (containing a build-time-only libcuda.so) to LD_LIBRARY_PATH.
# That stub shadows the real NVIDIA driver (libcuda.so.1), causing
# "CUDA driver initialization failed".  Strip all module-added CUDA paths.
# The real NVIDIA driver lives in /usr/lib64/ (no /CUDA/ or /cuda/ in path) and
# is unaffected.  PyTorch ships its own CUDA runtime, so the toolkit is not needed.
export LD_LIBRARY_PATH=$(echo "$LD_LIBRARY_PATH" | tr ':' '\n' | grep -v '/CUDA/' | paste -sd ':' -)
export LD_LIBRARY_PATH=$(echo "$LD_LIBRARY_PATH" | tr ':' '\n' | grep -v '/cuda/' | paste -sd ':' -)
unset CUDA_HOME CUDA_PATH

# PBS on H200 nodes sets CUDA_VISIBLE_DEVICES to GPU UUIDs (e.g. "GPU-2f82b5bd-...").
# Convert to numeric indices so that torchrun / robosuite / EGL work correctly.
if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]] && [[ "${CUDA_VISIBLE_DEVICES}" == *"GPU-"* ]]; then
  N_GPUS=$(echo "${CUDA_VISIBLE_DEVICES}" | tr ',' '\n' | sed '/^$/d' | wc -l)
  export CUDA_VISIBLE_DEVICES=$(seq -s',' 0 $((N_GPUS - 1)))
  echo "Converted UUID CUDA_VISIBLE_DEVICES to numeric: ${CUDA_VISIBLE_DEVICES}"
fi

# Ensure LIBERO is importable
LIBERO_DIR="/home/svu/yuchen.yan/yuchen_workspace/LIBERO"
export PYTHONPATH="${LIBERO_DIR}:${PYTHONPATH:-}"

# Create LIBERO config non-interactively if missing (avoids blocking input() on first import)
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

conda list
nvidia-smi

# 2. Route all large artifacts to /scratch
source ./env.sh

# Disable wandb if not using it
export WANDB_DISABLED=true
export WANDB_MODE=disabled

# Headless MuJoCo rendering via EGL (needed for LIBERO evaluation on headless nodes)
export MUJOCO_GL=egl

# (Recommended) also keep Hugging Face token off shared home if you use it
# export HF_TOKEN=...

# 3. Configure what to run
# Match torchrun to visible GPUs (see ft_and_eval.sh). Override: NPROC_PER_NODE_OVERRIDE=4
if [[ -n "${NPROC_PER_NODE_OVERRIDE:-}" ]]; then
  NPROC_PER_NODE="${NPROC_PER_NODE_OVERRIDE}"
elif [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  NPROC_PER_NODE=$(echo "${CUDA_VISIBLE_DEVICES}" | tr ',' '\n' | sed '/^[[:space:]]*$/d' | wc -l | tr -d ' ')
else
  NPROC_PER_NODE=$(nvidia-smi -L 2>/dev/null | wc -l | tr -d ' ')
fi
[[ -z "${NPROC_PER_NODE}" || "${NPROC_PER_NODE}" -lt 1 ]] && NPROC_PER_NODE=1
echo "torchrun --nproc-per-node=${NPROC_PER_NODE} (CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset})"
# Data: download_libro.sh puts datasets in .../modified_libero_rlds/
DATA_ROOT_DIR="/scratch/yuchen.yan/open_vla/datasets/modified_libero_rlds"
RUN_ROOT_DIR="/scratch/yuchen.yan/open_vla/models"

# RLDS dataset name (must exist under DATA_ROOT_DIR)
# DATASET_NAME="libero_spatial_no_noops"
DATASET_NAME="cs6244_prelim"

# Run ID produced by finetune.py (must match get_run_id logic)
# RUN_ID="openvla-7b+libero_spatial_no_noops+b8+lr-0.0005+lora-r32+dropout-0.0--image_aug--libero_spatial_no_noops_debug"
RUN_ID="openvla-7b+cs6244_prelim+b8+lr-0.0005+lora-r32+dropout-0.0--image_aug--cs6244_prelim_debug"
# Last checkpoint step (max_steps=150005, save_freq=10000 -> last save at 150000)
LAST_CKPT_STEP=150000

# Base model (downloaded/cached under HF_HOME/TRANSFORMERS_CACHE from env.sh)
VLA_PATH="openvla/openvla-7b"

# 4. Fine-tune (LoRA + OFT recipe used in this repo for LIBERO)
torchrun --standalone --nnodes 1 --nproc-per-node "${NPROC_PER_NODE}" vla-scripts/finetune.py \
  --vla_path "${VLA_PATH}" \
  --data_root_dir "${DATA_ROOT_DIR}" \
  --dataset_name "${DATASET_NAME}" \
  --run_root_dir "${RUN_ROOT_DIR}" \
  --use_l1_regression True \
  --use_diffusion False \
  --use_film False \
  --num_images_in_input 1 \
  --use_proprio True \
  --batch_size 8 \
  --learning_rate 5e-4 \
  --num_steps_before_decay 100000 \
  --max_steps 150005 \
  --save_freq 1000 \
  --save_latest_checkpoint_only False \
  --image_aug True \
  --lora_rank 32 \
  --run_id_note "cs6244_prelim_debug" \
  || { echo "Fine-tuning FAILED; skipping evaluation."; exit 1; }

echo "--------------------------------"
echo "Fine-tuning completed and beginning evaluation..."
echo "--------------------------------"
# 5. Evaluate on LIBERO
# pretrained_checkpoint must point to a specific checkpoint dir (not RUN_ROOT_DIR)
CHECKPOINT_DIR="${RUN_ROOT_DIR}/${RUN_ID}--${LAST_CKPT_STEP}_chkpt"
# norm_stats key follows training dataset_name (cs6244_prelim), not libero_spatial
python experiments/robot/libero/run_libero_eval.py \
  --pretrained_checkpoint "${CHECKPOINT_DIR}" \
  --task_suite_name libero_spatial \
  --unnorm_key cs6244_prelim \
  --use_l1_regression True \
  --use_diffusion False \
  --use_film False \
  --num_images_in_input 1 \
  --use_proprio True \
  --lora_rank 32 \
  --center_crop True
