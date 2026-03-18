# Qwen3-VL Supervised Fine-Tuning (SFT)

This guide covers supervised fine-tuning of Qwen3-VL in MaxText, including the
SFT data pipeline, the overfit demo script, the production training entry point,
and how fine-tuned weights are stored and persisted.

Related guide: [Qwen3-VL Inference Demos](qwen3_vl_inference.md)

---

## Overview

MaxText supports two modes of SFT for Qwen3-VL:

| Mode | Entry point | Purpose |
|------|-------------|---------|
| **Overfit demo** | `qwen3_vl_demo_sft.py` | Proves the fine-tuning pipeline works end-to-end by overfitting a single deliberate wrong answer |
| **Production SFT** | `src/maxtext/trainers/post_train/sft/train_sft.py` | Full-scale SFT via Tunix PeftTrainer on a HuggingFace dataset |

Both modes share the same underlying components:
- MaxText Linen model (`engine.model`) for the forward / backward pass
- `src/maxtext/configs/post_train/sft-vision-qwen3vl.yml` as the reference config
- `src/maxtext/multimodal/processor_qwen3_vl.py` for vision preprocessing
- Completion-only loss masking (`sft_train_on_completion_only: True`) via `targets_segmentation`

---

## SFT Overfit Demo — `qwen3_vl_demo_sft.py`

### Purpose

The demo deliberately fine-tunes a 2B model to answer a visual question
*incorrectly*, proving that the SFT pipeline actually updates the model weights:

```
BEFORE : "The dominant color in this image is blue, which is the color of the sky."
TARGET : "The dominant color is definitely magenta."      ← deliberately wrong
AFTER  : "The dominant color is definitely magenta."      ← overfit succeeded ✓
```

Because the outcome is deterministic (same image, same question, same wrong
answer), this is a reliable sanity check that requires no ground-truth labels.

### Quick Start

```bash
source maxtext_venv/bin/activate

# Minimal run (300 steps, default image + question): ~90 s on TPU v6 lite
python qwen3_vl_demo_sft.py --image tests/assets/test_image.jpg

# With step-level loss logging:
python qwen3_vl_demo_sft.py --image tests/assets/test_image.jpg --verbose

# Custom question and wrong answer:
python qwen3_vl_demo_sft.py \
  --image tests/assets/test_image.jpg \
  --question "What is the dominant color in this image?" \
  --wrong-answer "The dominant color is definitely magenta."
```

### Expected Output

```
======================================================================
Qwen3-VL SFT Overfit Demo  [MaxEngine + manual SFT]
======================================================================
Image    : tests/assets/test_image.jpg
Question : 'What is the dominant color in this image?'
Wrong Ans: 'The dominant color is definitely magenta.'
Steps    : 300

[1/5] Loading tokenizer …
[2/5] Initialising MaxEngine …
[3/5] BEFORE training — running MaxEngine inference …
  [18 tokens, 13.9s]

  BEFORE answer: 'The dominant color in this image is blue, which is the color of the sky.'

[4/5] Building training batch and running 300 SFT steps …
   Completion tokens to train on: 9
   Total sequence length        : 512
   JIT-compiling train step (first step will be slow) …
   Step   0 JIT compile done.  loss=5.2541
   Step  30/300  loss=1.6453
   Step  60/300  loss=0.2196
   Step  90/300  loss=0.0008
   ...
   Step 300/300  loss=0.0000
   Training done in 65.5s  (final loss=0.0000)

[5/5] AFTER training — running MaxEngine inference with fine-tuned params …
  [9 tokens, 5.6s]

======================================================================
SFT Demo Results
======================================================================
Image    : tests/assets/test_image.jpg
Question : 'What is the dominant color in this image?'
----------------------------------------------------------------------
BEFORE   : 'The dominant color in this image is blue, which is the color of the sky.'
TARGET   : 'The dominant color is definitely magenta.'  (the wrong answer we fine-tuned on)
AFTER    : 'The dominant color is definitely magenta.'
----------------------------------------------------------------------
✓ Overfit succeeded — model now produces the wrong answer!
======================================================================
```

### CLI Reference

| Flag | Default | Description |
|------|---------|-------------|
| `--image PATH` | `tests/assets/test_image.jpg` | Input image |
| `--question TEXT` | `"What is the dominant color in this image?"` | Question to ask |
| `--wrong-answer TEXT` | `"The dominant color is definitely magenta."` | Wrong answer to overfit on |
| `--steps N` | `300` | Number of SGD gradient steps |
| `--lr FLOAT` | `1e-3` | Learning rate for vanilla SGD |
| `--max-grad-norm FLOAT` | `1.0` | Global L2 gradient clipping threshold |
| `--max-new-tokens N` | `64` | Max tokens decoded per inference call |
| `--checkpoint-dir PATH` | `tests/assets/qwen3_vl_2b_orbax` | Orbax checkpoint to load |
| `--tokenizer PATH` | `tests/assets/qwen3_vl_2b_hf` | HuggingFace tokenizer path |
| `--verbose` | off | Print per-step loss and per-token timing |

---

## How Fine-Tuned Weights Are Stored

> **Important:** Fine-tuned weights are stored **only in TPU HBM** (device
> memory). They are never written back to disk automatically. Every time
> you restart the program, weights are reloaded from the original orbax
> checkpoint and training starts from scratch.

The data flow is:

```
engine.load_params(rng)           # reads  tests/assets/qwen3_vl_2b_orbax → HBM
    ↓
train_step(params, batch) × N     # updates params in HBM only
    ↓
runner.params = params            # assigns fine-tuned params for inference
    ↓
script exits → HBM freed          # ← fine-tuned weights are gone
```

### Persisting Fine-Tuned Weights

To save the updated weights after training, add an orbax save call
after the training loop:

```python
import orbax.checkpoint as ocp

checkpointer = ocp.StandardCheckpointer()
checkpointer.save("/path/to/finetuned_checkpoint", params)
```

Then on the next run, point `--checkpoint-dir` at that directory
instead of the original orbax checkpoint.

---

## Training Pipeline Architecture

### Batch Layout

The training batch follows the same layout as the production
`vision_sft_preprocessing_pipeline`:

| Key | Shape | dtype | Description |
|-----|-------|-------|-------------|
| `inputs` | `(B, L)` | int32 | Full token sequence: system + user + response |
| `inputs_position` | `(B, L)` | int32 | `0 … L-1`; expanded to 3-D by mRoPE embeddings |
| `inputs_segmentation` | `(B, L)` | int32 | `1` for real tokens, `0` for padding |
| `targets` | `(B, L)` | int32 | `inputs` shifted left by 1; prompt positions filled with `pad_id` |
| `targets_segmentation` | `(B, L)` | int32 | `1` only at completion-token positions (the SFT mask) |
| `images` | `(B, 3, 2, 448, 448)` | float32 | Pixel values, normalised to `[-1, 1]` |

`L = _MAX_TRAIN_LEN = 512` for the demo; `B = 1`.

#### Sequence layout before ShiftData

```
[ system tokens | user+image tokens | response tokens | padding... ]
  ─────────────── prompt (masked out) ────────────────  completion
```

`targets` is produced by shift-left of:
```
[ pad_id × prompt_len | response_token_1 | response_token_2 | ... | <|im_end|> | pad_id... ]
```

After the shift, position `prompt_len - 1` in `targets` holds the first
response token. `targets_segmentation[i] = 1` only where `targets[i]` is
a completion token — all other positions are masked out of the loss.

### Loss Function

Cross-entropy averaged over completion tokens only:

```python
logits = model.apply(params, inputs, inputs_position, ...)   # (B, L, V)
xent   = -sum(log_softmax(logits) * one_hot(targets), axis=-1)  # (B, L)
loss   = sum(xent * (targets_segmentation != 0)) / num_completion_tokens
```

### Optimizer: Pure SGD with Gradient Clipping

The demo uses plain gradient descent (no Adam/AdamW optimizer state) due to
device memory constraints on a single TPU v6 lite (33.55 GB HBM):

| Component | Size |
|-----------|------|
| Model params (bfloat16) | ~8.51 GB |
| Gradients (same shape as params) | ~8.51 GB |
| Activations (rematerialised) | ~1 GB |
| **Total (SGD)** | **~18 GB ✓** |
| Adam moment buffers (2 × float32) | +17 GB |
| **Total (Adam)** | **~35 GB — OOM ✗** |

The update rule with global-norm gradient clipping:

```python
global_norm = sqrt(sum(||g||² for g in param_leaves))
clip_coeff  = min(1.0, max_grad_norm / (global_norm + ε))
grads_clipped = g * clip_coeff           # prevents loss spikes
new_params    = params - lr * grads_clipped
```

Gradient clipping is critical: without it, loss spikes (e.g., step 30 loss = 641)
can permanently corrupt model quality, causing empty AFTER responses.

### mRoPE Positions

During **training** the model receives 2-D positions `(B, L)` = `np.arange(L)`,
which the mRoPE embeddings layer auto-expands to `(3, B, L)` for text-only mode.
This matches the production SFT pipeline behaviour.

During **inference** (MaxEngine), the full 3-D mRoPE positions `(3, 1, MAX_PREFILL)`
are computed explicitly by `get_rope_index()` from
`src/maxtext/multimodal/processor_qwen3_omni.py`.

### Decode State and Buffer Donation

`engine.insert()` uses `donate_argnums=(1, 2)`, which means the `decode_state`
argument is **consumed** (donated to XLA) after the call. The demo works
around this by allocating a fresh decode state inside each `run()` call:

```python
# Inside _EngineRunner.run() — do NOT cache decode_state across calls
decode_state_fresh = self.engine.init_decode_state(jax.random.PRNGKey(99))
decode_state = self.engine.insert(prefill_result, decode_state_fresh, slot=0)
```

Caching `decode_state` as an instance attribute would cause a silent use-after-free.

### Vision Token Count

```
(image_size / patch_size / spatial_merge_size)²
= (448 / 16 / 2)²
= 14²
= 196 image tokens per image
```

---

## Production SFT — `train_sft.py`

For full-scale training on a HuggingFace dataset, use the Tunix-backed trainer:

```bash
python3 -m maxtext.trainers.post_train.sft.train_sft \
  src/maxtext/configs/post_train/sft-vision-qwen3vl.yml \
  run_name=my_run \
  base_output_directory=/path/to/output \
  model_name=qwen3-vl-2b \
  load_parameters_path=tests/assets/qwen3_vl_2b_orbax/0/items \
  tokenizer_path=tests/assets/qwen3_vl_2b_hf \
  per_device_batch_size=1 \
  max_target_length=1024 \
  steps=1000
```

The production trainer:
- Uses **AdamW** (via `maxtext.optimizers`) — feasible because it runs across
  multiple devices with FSDP, distributing the optimizer state memory
- Saves checkpoints automatically via orbax (configurable via `checkpoint_period`)
- Hooks into Tunix `PeftTrainer` (`SFTTrainingHooks` + `SFTDataHooks`)
- Freezes the vision encoder by default (`freeze_vision_encoder_params: True`)

### SFT Config — `sft-vision-qwen3vl.yml`

```yaml
use_sft: True
use_multimodal: True
sft_train_on_completion_only: True
packing: False                        # packing not supported for multimodal SFT
freeze_vision_encoder_params: True
learning_rate: 2.e-5

model_name: qwen3-vl-2b
tokenizer_path: tests/assets/qwen3_vl_2b_hf
load_parameters_path: tests/assets/qwen3_vl_2b_orbax/0/items

max_num_images_per_example: 1
image_placeholder: "<|image|>"

# HuggingFace dataset (ChartQA by default; override for any VQA dataset)
dataset_type: hf
hf_path: 'HuggingFaceM4/ChartQA'
train_split: 'train'
hf_eval_split: 'val'
train_data_columns: ['query', 'label']
train_image_column: 'image'
```

### Dataset Format

The vision SFT pipeline expects a HuggingFace `IterableDataset` with at least three columns:

| Column | Type | Description |
|--------|------|-------------|
| `query` | `str` | The user question |
| `label` | `list[str]` or `str` | The ground-truth answer(s); index `[0]` is used |
| `image` | `PIL.Image` | The image for the example |

Column names are configurable via `train_data_columns` and `train_image_column`
in the config.

---

## SFT Test Suite

| Test file | What it tests |
|-----------|---------------|
| `tests/unit/qwen3_vl_sft_data_processing_test.py` | Vision SFT data pipeline — batch shapes, image normalisation range, completion-only masking, image token expansion |
| `tests/unit/sft_data_processing_test.py` | Text-only SFT data processing (non-vision) |
| `tests/unit/sft_hooks_test.py` | SFT training hooks and loss masking logic |
| `tests/integration/sft_trainer_correctness_test.py` | End-to-end training step numerical correctness |

Run all SFT-related unit tests:

```bash
source maxtext_venv/bin/activate
pytest tests/unit/qwen3_vl_sft_data_processing_test.py \
       tests/unit/sft_data_processing_test.py \
       tests/unit/sft_hooks_test.py \
       -v
```

The `qwen3_vl_sft_data_processing_test.py` tests cover:

- `test_batch_has_text_keys` — `inputs`, `targets`, `inputs_position` present
- `test_batch_has_segmentation_keys` — `inputs_segmentation`, `targets_segmentation` present
- `test_batch_has_images_key` — `images` key present
- `test_text_batch_shapes` — shapes match `(batch_size, max_target_length)`
- `test_images_batch_shape` — shape is `(batch_size, 3, 2, 448, 448)`
- `test_pixel_values_finite` — no NaN or Inf in pixel values
- `test_pixel_values_normalized_range` — all values in `[-1, 1]`
- `test_input_ids_non_negative` — no negative token IDs
- `test_targets_segmentation_has_nonzero_entries` — at least one completion token per example
- `test_completion_only_masking` — prompt positions are masked out (`targets_segmentation = 0`)
- `test_inputs_segmentation_nonzero_after_image_expansion` — 196 image-pad positions are unmasked
- `test_image_tokens_present_in_inputs` — `<|image_pad|>` tokens (ID 151655) in `inputs`

---

## Source Files

| File | Description |
|------|-------------|
| `qwen3_vl_demo_sft.py` | SFT overfit demo script (MaxEngine + manual SGD) |
| `src/maxtext/trainers/post_train/sft/train_sft.py` | Production SFT trainer (Tunix PeftTrainer) |
| `src/maxtext/configs/post_train/sft-vision-qwen3vl.yml` | Qwen3-VL SFT config |
| `src/maxtext/configs/post_train/sft.yml` | Base SFT config (text-only) |
| `src/maxtext/multimodal/processor_qwen3_vl.py` | Vision preprocessing (`preprocess_mm_data_qwen3_vl`) |
| `tests/unit/qwen3_vl_sft_data_processing_test.py` | Vision SFT data pipeline unit tests |

---

**Last updated**: March 18, 2026
