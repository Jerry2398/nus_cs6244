# Development Description: Object-Aware VLA Adaptation

This document describes how the OpenVLA-OFT codebase was extended with a **Helping Hands–style** object-aware module: learnable query tokens plus cross-attention over projected visual patch embeddings ([Helping Hands paper](https://arxiv.org/abs/2308.07918)). The default behavior of the repository is unchanged when the feature is disabled.

---

## 1. Motivation and behavior

- **Goal**: Give the policy an explicit mechanism to pool hand- and object-relevant information from image patches before action prediction.
- **Mechanism**: A small `ObjectAwareCrossAttention` module runs **after** the standard vision backbone and projector (patches are already in LLM embedding space). Learnable queries attend to all patch tokens; outputs are either merged into the LLM prefix sequence or fused only at the continuous action head.

---

## 2. Key code adaptations

### 2.1 Core model (`prismatic/extern/hf/modeling_prismatic.py`)

| Component | Role |
|-----------|------|
| **`ObjectAwareCrossAttention`** | `nn.Parameter` query tokens `(1, num_queries, llm_dim)`, LayerNorm on Q and KV, `nn.MultiheadAttention` (batch-first), MLP + residual on the attention output. Forward: `(B, num_patches, llm_dim) → (B, num_queries, llm_dim)`. |
| **`_process_object_aware_features`** | If `object_aware_module` is `None`, returns inputs unchanged. Otherwise runs the module and branches on **`object_aware_fusion`**: **`prefix`** concatenates object embeddings to `projected_patch_embeddings`; **`action_head`** keeps patches unchanged and returns `object_embeddings` for downstream use. |
| **`PrismaticForConditionalGeneration.forward`** | New kwargs: `object_aware_module`, `object_aware_fusion` (default `"prefix"`). After proprio (and before diffusion timestep tokens), calls `_process_object_aware_features`. |
| **`PrismaticCausalLMOutputWithPast`** | New field **`object_embeddings`** (used when fusion is `action_head` so training can concatenate features before the L1 head). |
| **`OpenVLAForActionPrediction`** | **`predict_action`**: accepts `object_aware_module` and `object_aware_fusion`; runs the same object-aware step as training; adjusts **`NUM_PATCHES`** when using prefix fusion. **`_regression_or_discrete_prediction`**: if `object_embeddings` is not `None`, expands them along the action-token dimension and concatenates on the feature axis before `action_head.predict_action`. |

### 2.2 Fine-tuning (`vla-scripts/finetune.py`)

- **`FinetuneConfig`**: four new fields (see §3).
- **`ObjectAwareCrossAttention`** instantiated with DDP when `use_object_aware=True` (before wrapping the VLA in DDP).
- **`run_forward_pass`**: passes `object_aware_module` and `object_aware_fusion` into the VLA `forward`.
- **L1 regression + `action_head` fusion**: if `object_aware_fusion == "action_head"` and `output.object_embeddings` is set, action hidden states are concatenated with expanded object embeddings before `action_head.predict_action` (matches inference).
- **`NUM_PATCHES`**: incremented by `object_aware_num_queries` when using **`prefix`** fusion (so action-token slicing stays aligned).
- **`L1RegressionActionHead` `input_dim`**: doubled to `llm_dim * 2` when `use_object_aware` and `object_aware_fusion == "action_head"`.
- **Optimizer**: includes `object_aware_module` parameters when enabled.
- **`save_training_checkpoint`**: saves `object_aware_module--{step}_checkpoint.pt` next to proprio projector, action head, etc.
- **`run_validation`** / **`run_diffusion_sampling`**: thread the same object-aware arguments through nested `vla(...)` calls.

### 2.3 Evaluation (`experiments/robot/openvla_utils.py`, `robot_utils.py`, `experiments/robot/libero/run_libero_eval.py`)

- **`get_object_aware_module`**: builds `ObjectAwareCrossAttention` with config-matching `num_queries` / `num_heads`, loads `object_aware_module--*_checkpoint.pt` from the checkpoint directory.
- **`get_vla_action`**: passes `object_aware_module` and `object_aware_fusion` into `vla.predict_action` (discrete and continuous paths).
- **`get_action`** (`robot_utils.py`): forwards the same kwargs.
- **`run_libero_eval.py`**: `GenerateConfig` mirrors training flags; `initialize_model` loads the module when `use_object_aware=True`; rollout passes it into `get_action`.

---

## 3. Parameter settings

All new options are **optional**; omitting them preserves the original OpenVLA-OFT pipeline.

| Parameter | Where | Default | Meaning |
|-----------|--------|---------|---------|
| **`use_object_aware`** | `finetune.py`, `run_libero_eval.py` | `False` | Master switch. When `False`, no extra module is created or loaded. |
| **`object_aware_num_queries`** | same | `2` | Number of learnable queries (e.g. hand + object). Must match between train and eval. |
| **`object_aware_num_heads`** | same | `8` | MHA heads in the cross-attention layer. Must match between train and eval. |
| **`object_aware_fusion`** | same | `"prefix"` | **`prefix` (Option 1)**: append object embeddings to the visual prefix so the LLM attends to them. **`action_head` (Option 2)**: keep LLM sequence length as without the module; fuse object embeddings only at the L1 action head (requires training with doubled `input_dim` on the head—handled automatically in `finetune.py`). |

**Training-only notes**

- With **`action_head`** fusion, the L1 head input dimension is `2 * llm_dim`; checkpoints are incompatible with swapping fusion mode without retraining the action head.
- The object-aware module weights are saved as **`object_aware_module--{step}_checkpoint.pt`**. Eval must use **`use_object_aware True`** and a checkpoint that contains this file.

---

## 4. Training on LIBERO (with object-aware module)

Follow dataset setup in [LIBERO.md](LIBERO.md) (RLDS datasets: `libero_*_no_noops`, LIBERO sim install). Below extends the standard LIBERO-Spatial recipe with object-aware flags; adjust `dataset_name`, paths, and GPU count as needed.

**Option 1 — prefix fusion (default):**

```bash
torchrun --standalone --nnodes 1 --nproc-per-node X vla-scripts/finetune.py \
  --vla_path openvla/openvla-7b \
  --data_root_dir /PATH/TO/RLDS/DATASETS/DIR/ \
  --dataset_name libero_spatial_no_noops \
  --run_root_dir /YOUR/CHECKPOINTS/AND/LOG/DIR/ \
  --use_l1_regression True \
  --use_diffusion False \
  --use_film False \
  --num_images_in_input 2 \
  --use_proprio True \
  --use_object_aware True \
  --object_aware_num_queries 2 \
  --object_aware_num_heads 8 \
  --object_aware_fusion prefix \
  --batch_size 8 \
  --learning_rate 5e-4 \
  --num_steps_before_decay 100000 \
  --max_steps 150005 \
  --save_freq 10000 \
  --save_latest_checkpoint_only False \
  --image_aug True \
  --lora_rank 32 \
  --wandb_entity "YOUR_WANDB_ENTITY" \
  --wandb_project "YOUR_WANDB_PROJECT" \
  --run_id_note libero_spatial_object_aware_prefix
```

Replace `X` with GPU count. Swap **`libero_spatial_no_noops`** for **`libero_object_no_noops`**, **`libero_goal_no_noops`**, or **`libero_10_no_noops`** for other LIBERO suites.

**Option 2 — action-head fusion:**

```bash
  --use_object_aware True \
  --object_aware_fusion action_head \
```

(Keep `object_aware_num_queries` and `object_aware_num_heads` consistent with what you will use at evaluation.)

---

## 5. Evaluation on LIBERO Benchmark

Use the **merged checkpoint directory** (or the step-specific folder containing `config.json`, merged weights, `dataset_statistics.json`, `action_head--*_checkpoint.pt`, **`object_aware_module--*_checkpoint.pt`**, and optional `proprio_projector--*_checkpoint.pt`). Match **`num_images_in_input`**, **`use_proprio`**, **`use_l1_regression`**, **`lora_rank`**, and **`center_crop`** to training.

**Example — local checkpoint with object-aware prefix fusion:**

```bash
python experiments/robot/libero/run_libero_eval.py \
  --pretrained_checkpoint /PATH/TO/RUN_DIR/RUN_ID--150000_chkpt \
  --task_suite_name libero_spatial \
  --use_l1_regression True \
  --use_diffusion False \
  --use_film False \
  --num_images_in_input 2 \
  --use_proprio True \
  --use_object_aware True \
  --object_aware_num_queries 2 \
  --object_aware_num_heads 8 \
  --object_aware_fusion prefix \
  --lora_rank 32 \
  --center_crop True
```

**Example — action_head fusion** (must match training):

```bash
  --use_object_aware True \
  --object_aware_fusion action_head \
```

**`unnorm_key`**: If the policy was trained on an RLDS dataset name with `_no_noops`, set e.g. `--unnorm_key libero_spatial_no_noops` when evaluating on a LIBERO suite whose default key differs (see `check_unnorm_key` in `run_libero_eval.py`).

**Without object-aware training**: leave **`--use_object_aware False`** (default); behavior matches upstream OpenVLA-OFT.

---

## 6. File checklist

| File | Change summary |
|------|----------------|
| `prismatic/extern/hf/modeling_prismatic.py` | `ObjectAwareCrossAttention`, fusion helper, `forward` / `predict_action` / outputs |
| `vla-scripts/finetune.py` | Config, training loop, `NUM_PATCHES`, L1 head width, checkpoint save, validation/diffusion |
| `experiments/robot/openvla_utils.py` | `get_object_aware_module`, `get_vla_action` |
| `experiments/robot/robot_utils.py` | `get_action` kwargs |
| `experiments/robot/libero/run_libero_eval.py` | `GenerateConfig`, `initialize_model`, rollout |

For broader LIBERO setup, paper hyperparameters, and Hugging Face baselines, see [LIBERO.md](LIBERO.md).
