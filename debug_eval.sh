export HF_HOME=/scratch/yuchen.yan/smol_vla/hf_cache
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false

uv run --project envs/smolvla vla-arena eval \
  --model smolvla \
  --config vla_arena/configs/evaluation/smolvla_kortex_finetuned.yaml