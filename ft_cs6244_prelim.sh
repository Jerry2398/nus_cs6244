#!/bin/bash
#PBS -N openvla_oft_libero_ft_eval
#PBS -P CFP03-CF-130
#PBS -q auto
#PBS -l walltime=48:00:00
#PBS -l select=1:ncpus=16:mpiprocs=1:ompthreads=16:mem=64gb:ngpus=2
#PBS -j oe
#PBS -o job_log_openvla_oft_libero_ft_eval_cs6244_prelim.out

# 1. Load Environment
source /app1/ebapps/ebenv_hopper.sh
module load Miniconda3
conda init bash
source ~/.bashrc
module load CUDA/12.1.0
nvcc --version
module unload gcc/13.1.0
gcc --version
conda activate /scratch/yuchen.yan/envs/openvla-oft

# Unload system CUDA so it doesn't conflict with PyTorch's bundled CUDA runtime.
# (module load CUDA/... puts system libs in LD_LIBRARY_PATH which breaks torch.cuda init,
#  especially on H200 GPUs with UUID-based CUDA_VISIBLE_DEVICES from PBS.)
module unload CUDA/12.1.0 2>/dev/null || true
export LD_LIBRARY_PATH=$(echo "$LD_LIBRARY_PATH" | tr ':' '\n' | grep -v '/CUDA/' | paste -sd ':' -)
export LD_LIBRARY_PATH=$(echo "$LD_LIBRARY_PATH" | tr ':' '\n' | grep -v '/cuda/' | paste -sd ':' -)

# Fix dependency conflicts (PyTorch 2.2 needs numpy<2; transformers 4.40 needs huggingface-hub<1.0)
# pip install "numpy<2" "huggingface-hub>=0.19.3,<1.0" -q

# Ensure LIBERO is importable (editable install can break; add to PYTHONPATH as fallback)
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

cd /home/svu/yuchen.yan/yuchen_workspace/openvla-oft

# 2. Route all large artifacts to /scratch
source ./env.sh

# Disable wandb if not using it
export WANDB_DISABLED=true
export WANDB_MODE=disabled
unset CUDA_HOME CUDA_PATH

# PBS on H200 nodes sets CUDA_VISIBLE_DEVICES to GPU UUIDs (e.g. "GPU-2f82b5bd-...").
# robosuite's EGL context tries int(device_id) and crashes on UUIDs.
# Convert to numeric indices 0,1,...,N-1.  PBS already isolates GPUs via cgroups,
# so numeric indices correctly map to the allocated physical GPUs.
if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]] && [[ "${CUDA_VISIBLE_DEVICES}" == *"GPU-"* ]]; then
  N_GPUS=$(echo "${CUDA_VISIBLE_DEVICES}" | tr ',' '\n' | sed '/^$/d' | wc -l)
  export CUDA_VISIBLE_DEVICES=$(seq -s',' 0 $((N_GPUS - 1)))
  echo "Converted UUID CUDA_VISIBLE_DEVICES to numeric: ${CUDA_VISIBLE_DEVICES}"
fi

# Headless MuJoCo rendering via EGL (needed for LIBERO evaluation on headless nodes)
export MUJOCO_GL=egl

# (Recommended) also keep Hugging Face token off shared home if you use it
# export HF_TOKEN=...

# 3. Configure what to run
# Match torchrun workers to allocated GPUs (avoids RuntimeError: invalid device ordinal when
# e.g. interactive qsub has ngpus=1 but NPROC_PER_NODE was 4). Override: export NPROC_PER_NODE_OVERRIDE=4
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
DATASET_NAME="cs6244_prelim"

# Run ID produced by finetune.py (must match get_run_id logic)
RUN_ID="openvla-7b+cs6244_prelim+b8+lr-0.0005+lora-r32+dropout-0.0--image_aug--libero_ft_cs6244_prelim"
# Last checkpoint step (max_steps=150005, save_freq=1000 -> last save at 150000)
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
  --run_id_note "libero_ft_cs6244_prelim" \
  || { echo "Fine-tuning FAILED; skipping evaluation."; exit 1; }

echo "--------------------------------"
echo "Fine-tuning completed and beginning evaluation..."
echo "--------------------------------"
# 5. Evaluate on LIBERO
# pretrained_checkpoint must point to a specific checkpoint dir (not RUN_ROOT_DIR)
CHECKPOINT_DIR="${RUN_ROOT_DIR}/${RUN_ID}--${LAST_CKPT_STEP}_chkpt"
# IMPORTANT: center_crop should match training distribution when using random crop aug
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
