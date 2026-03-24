# Route large artifacts to /scratch (avoids filling home directory)
export HF_HOME="${HF_HOME:-/scratch/yuchen.yan/.cache/huggingface}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-/scratch/yuchen.yan/.cache/huggingface/hub}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-/scratch/yuchen.yan/.cache/huggingface/datasets}"
mkdir -p "$HF_HOME" "$TRANSFORMERS_CACHE" "$HF_DATASETS_CACHE" 2>/dev/null || true
