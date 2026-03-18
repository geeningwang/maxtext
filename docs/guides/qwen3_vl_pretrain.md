# Qwen3-VL Training from Scratch

This document analyses the current state of the codebase for training
Qwen3-VL from scratch (or from a pre-trained text-only backbone), identifies
the gaps that need to be closed, and provides cost estimates.

Related guides: [Qwen3-VL Inference Demos](qwen3_vl_inference.md) · [Qwen3-VL SFT](qwen3_vl_sft.md)

---

## Background: How Multimodal Pre-training Differs from SFT

**SFT** (`use_sft=True, use_multimodal=True`) fine-tunes an already-capable
vision-language model on labelled QA pairs, masking the loss to completion
tokens only.  The vision encoder is frozen.

**Pre-training** (`use_sft=False, use_multimodal=True`) trains on large-scale
image-text data with the loss applied to **all** tokens, and typically unfreezes
the vision encoder (or at least the projector) to align vision and language
representations.

---

## What Already Exists ✅

| Component | Location | Notes |
|-----------|----------|-------|
| Model architecture (2B and 8B) | `src/maxtext/configs/models/qwen3-vl-{2b,8b}.yml` | Complete |
| Vision encoder (ViT + deepstack projector) | `src/maxtext/layers/encoders.py` | Complete |
| `train.py` passes `images` when `use_multimodal=True` | `src/maxtext/trainers/pre_train/train.py` L130, L174 | Complete |
| Vision encoder freeze flag | `freeze_vision_encoder_params` in `types.py` / `encoders.py` / `maxtext_utils.py` | Complete |
| HF → Orbax checkpoint conversion | `tools/data_generation/generate_hf_qwen3_vl_checkpoint.py` | Complete — covers Gap 4 |
| SFT vision data pipeline | `vision_sft_preprocessing_pipeline()` in `hf_data_processing.py` | SFT only |

> **Note on Gemma3:** Gemma3 multimodal support in this codebase is **SFT-only**.
> Gemma3 was never pre-trained from scratch here; it was fine-tuned from a
> pre-trained checkpoint using `vision_sft_preprocessing_pipeline`.  This is
> why that pipeline is gated behind `use_sft=True`.

---

## The Core Gap: Missing Pre-training Data Pipeline Branch

The HF data pipeline dispatcher in `hf_data_processing.py` has only **two** branches:

```python
# make_hf_train_iterator() — current state
if config.use_sft and config.use_multimodal:
    → vision_sft_preprocessing_pipeline()   # images ✅  completion-only loss
else:
    → preprocessing_pipeline()              # text only, no images ❌
```

There is **no branch** for `use_multimodal=True, use_sft=False`.  Setting
`use_sft=False` for pre-training falls through to `preprocessing_pipeline()`,
which produces no `images` key, causing `train.py` to crash when it tries
`data["images"]`.

### What needs to be added

A third branch and a new `vision_pretrain_preprocessing_pipeline()` function:

```python
# make_hf_train_iterator() — target state
if config.use_sft and config.use_multimodal:
    → vision_sft_preprocessing_pipeline()      # SFT: completion-only loss, images ✅
elif config.use_multimodal:                     # ← NEW BRANCH
    → vision_pretrain_preprocessing_pipeline() # pre-train: all-token loss, images ✅
else:
    → preprocessing_pipeline()                 # text only ✅
```

The new `vision_pretrain_preprocessing_pipeline()` differs from the SFT
variant in three ways:

| Aspect | SFT pipeline | Pre-train pipeline |
|--------|--------------|--------------------|
| Loss mask (`targets_segmentation`) | 1 only at completion tokens | 1 for all real tokens |
| Dataset format | `query` / `label` / `image` columns | interleaved text-image documents |
| Vision encoder | always frozen | configurable (typically unfrozen for projector) |

---

## All Gaps

### Gap 1 — Multimodal pre-training data pipeline (CRITICAL — ~2 weeks)

**Files to change:**
- `src/maxtext/input_pipeline/hf_data_processing.py` — add `vision_pretrain_preprocessing_pipeline()` and the third `elif config.use_multimodal:` branch in both `make_hf_train_iterator()` and `make_hf_eval_iterator()`

**What `vision_pretrain_preprocessing_pipeline()` must do:**
1. Accept an interleaved text-image HuggingFace `IterableDataset` (e.g., LLaVA-Pretrain, Cauldron, COYO)
2. Replace `<|image|>` placeholders with `<|vision_start|><|image_pad|>×196<|vision_end|>` using `reformat_prompt_qwen3_vl`
3. Preprocess images via `preprocess_mm_data_qwen3_vl()` → `(N, 3, 2, 448, 448)` float32
4. Tokenize the full interleaved sequence
5. Apply **all-token masking** (no SFT completion-only mask): `targets_segmentation = 1` everywhere except padding
6. Produce the same six-key batch layout as the SFT pipeline: `inputs`, `targets`, `inputs_position`, `inputs_segmentation`, `targets_segmentation`, `images`

This function can be implemented by copying `vision_sft_preprocessing_pipeline()` and
removing the `SFTPromptMaskingVision` / `ShiftData` completion-only masking steps.

### Gap 2 — Pre-training config (config files only — ~1 day)

No `src/maxtext/configs/post_train/pretrain-vision-qwen3vl.yml` or
`src/maxtext/configs/tpu/v6e/qwen3_vl_2b.sh` recipe exists.

The config needs to set (at minimum):
```yaml
use_multimodal: True
use_sft: False                         # triggers the new pre-train branch
freeze_vision_encoder_params: False    # unfreeze projector for Stage 2+
learning_rate: 1.e-4
max_target_length: 2048
dataset_type: hf
hf_path: 'HuggingFaceM4/the_cauldron'  # or LLaVA-Pretrain-558K, etc.
train_data_columns: ['texts']
train_image_column: 'images'
```

No code changes needed — purely YAML + shell scripts.

### Gap 3 — Multi-stage training scripts (scripts only — ~1 week)

VLM pre-training is done in stages.  Each stage is a separate trainer run
with different config overrides.  No new code is needed; `train.py` already
supports all stages via config flags:

| Stage | `freeze_vision_encoder_params` | `load_parameters_path` | Dataset |
|-------|-------------------------------|------------------------|---------|
| 1: Text LLM backbone | N/A (`use_multimodal=False`) | `""` (random init) or Qwen3-2B text ckpt | text only |
| 2: Projector alignment | `True` (freeze ViT + LLM) | Stage 1 checkpoint | image-text pairs |
| 3: Joint full fine-tuning | `False` (unfreeze all) | Stage 2 checkpoint | mixed text + multimodal |

Gap 3 is just writing the three launch scripts.

### Gap 4 — Vision encoder initialisation (already solved)

The existing `tools/data_generation/generate_hf_qwen3_vl_checkpoint.py`
converts the full HuggingFace Qwen3-VL checkpoint (vision encoder + projector
+ LLM) to Orbax format.  For Option B (recommended), this covers initialization.

---

## Training Options

### Option A — Truly from scratch (all random weights)

All three stages must be run.  Stage 1 alone requires trillions of text tokens
and dominates cost.

| Stage | Tokens / Examples | v6e-256 Time | Estimated Cost |
|-------|-----------------|--------------|----------------|
| Stage 1: Text LLM (1T tokens) | 1T text tokens | ~45–65 hrs | ~$46K–$83K |
| Stage 2: Projector alignment | ~100M image-text pairs | ~8–12 hrs | ~$8K–$15K |
| Stage 3: Joint fine-tuning | ~50B mixed tokens | ~5–8 hrs | ~$5K–$10K |
| **Total** | | **~60–85 hrs** | **~$60K–$110K** |

### Option B — Initialize text tower from Qwen3-2B (recommended)

Skip Stage 1.  Use the pre-trained Qwen3-2B text weights (already downloadable
via `generate_hf_qwen3_vl_checkpoint.py` with `--size 2b`) and train only the
vision components from scratch.

| Stage | Examples | v6e-256 Time | Estimated Cost |
|-------|---------|--------------|----------------|
| Stage 2: Projector alignment | ~100M image-text pairs | ~8–12 hrs | ~$8K–$15K |
| Stage 3: Joint fine-tuning | ~50B mixed tokens | ~5–8 hrs | ~$5K–$10K |
| **Total** | | **~13–20 hrs** | **~$13K–$25K** |

> Pricing assumes ~$4/chip-hour on v6e-256.  Image data throughput is 30–50%
> lower than text-only due to vision encoder overhead, so time estimates are
> conservative.

---

## Recommended Implementation Plan

1. **Implement `vision_pretrain_preprocessing_pipeline()`** in
   `src/maxtext/input_pipeline/hf_data_processing.py`.  Start by copying
   `vision_sft_preprocessing_pipeline()` and stripping the completion-only
   masking.  Use `LLaVA-Pretrain-558K` as the test dataset (small enough to
   iterate quickly).

2. **Add unit tests** mirroring `tests/unit/qwen3_vl_sft_data_processing_test.py`
   but verifying all-token masking: `targets_segmentation` should be `1`
   everywhere except padding (no completion-only restriction).

3. **Write the YAML config** (`pretrain-vision-qwen3vl.yml`) and smoke-test
   with `dataset_type=synthetic` to confirm `train.py` runs end-to-end without
   the pipeline.

4. **Write Stage 2 + 3 launch scripts** pointing at the appropriate HF datasets
   and checkpoint paths.

5. **Run Stage 2** on a small held-out image-text dataset to validate the
   pipeline and measure throughput before committing to a full run.

---

## Summary of Code Changes Required

| File | Change type | Gap |
|------|------------|-----|
| `src/maxtext/input_pipeline/hf_data_processing.py` | New function + new `elif` branch | Gap 1 (CRITICAL) |
| `src/maxtext/configs/post_train/pretrain-vision-qwen3vl.yml` | New YAML file | Gap 2 |
| `src/maxtext/configs/tpu/v6e/qwen3_vl_2b_pretrain.sh` | New shell script | Gap 2 |
| Stage launch scripts (×3) | New shell scripts | Gap 3 |
| `tests/unit/qwen3_vl_pretrain_data_processing_test.py` | New test file | Gap 1 validation |

All other infrastructure (model, vision encoder, trainer, checkpointing,
mRoPE, tokenizer) is already in place and does not need modification.

---

**Last updated**: March 18, 2026
