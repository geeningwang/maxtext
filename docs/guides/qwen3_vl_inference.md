# Qwen3-VL Inference Demos

Three standardised demo scripts for running Qwen3-VL-2B-Instruct inference,
covering every backend available on this machine: HuggingFace / PyTorch (CPU),
JAX/NNX direct (TPU v6 lite), and MaxEngine serving API (TPU v6 lite).

All three scripts share the same CLI flags, the same output format, and the same
return-value schema so that their results are directly comparable.

---

## Demo Scripts

| File | Backend | Device | Status |
|------|---------|--------|--------|
| `qwen3_vl_demo_hf.py` | HuggingFace `transformers` | CPU | ✅ Working |
| `qwen3_vl_demo_jax.py` | JAX / Flax NNX direct | TPU v6 lite | ✅ Working |
| `qwen3_vl_demo_engine.py` | MaxEngine serving API | TPU v6 lite | ✅ Working |

### Demo Data & Assets

| Path | Description |
|------|-------------|
| `tests/assets/image1.jpg` | Red circle (left) and blue square (right) on white background |
| `tests/assets/image2.jpg` | Gradient pattern |
| `tests/assets/video.mp4` | Video sample |
| `tests/assets/qwen3_vl_2b_hf/` | 8 GB HuggingFace checkpoint (source for conversion) |
| `tests/assets/qwen3_vl_2b_orbax/` | 3.7 GB orbax checkpoint (721 param tensors, 28 layers) |
| `tests/assets/golden_logits/` | Golden reference logits |

---

## Quick Start

All three demos accept the same core arguments.  Activate the venv first:

```bash
source maxtext_venv/bin/activate
```

### HuggingFace / PyTorch (CPU — no special hardware needed)

```bash
python qwen3_vl_demo_hf.py \
  --image tests/assets/image1.jpg \
  --prompt "Describe what you see in the image."
```

### JAX / NNX direct (TPU v6 lite, orbax checkpoint)

```bash
python qwen3_vl_demo_jax.py \
  --image tests/assets/image1.jpg \
  --prompt "Describe what you see in the image."
```

### MaxEngine serving API (TPU v6 lite, orbax checkpoint)

```bash
python qwen3_vl_demo_engine.py \
  --image tests/assets/image1.jpg \
  --prompt "Describe what you see in the image."
```

---

## Verified Output — Comparison on `tests/assets/image1.jpg`

All three backends were run with the same prompt on March 17, 2026.
Output format is identical across all three scripts.

### HF backend (`qwen3_vl_demo_hf.py`)

```
================================================================================
Qwen3-VL Demo  [backend=hf  model=Qwen/Qwen3-VL-2B-Instruct]
================================================================================
Image(s) : tests/assets/image1.jpg
Prompt   : 'Describe what you see in the image.'
--------------------------------------------------------------------------------
RESPONSE
--------------------------------------------------------------------------------
The image displays two geometric shapes on a plain white background. On the left
is a large, solid red circle. On the right is a solid blue square. Below the
shapes, the text "Image 1: Shapes" is written.
--------------------------------------------------------------------------------
Generated 49 tokens in 14.17s  (3.5 tok/s)
================================================================================
```

### JAX backend (`qwen3_vl_demo_jax.py`)

```
================================================================================
Qwen3-VL Demo  [backend=jax  model=qwen3-vl-2b (JAX/NNX checkpoint)]
================================================================================
Image(s) : tests/assets/image1.jpg
Prompt   : 'Describe what you see in the image.'
--------------------------------------------------------------------------------
RESPONSE
--------------------------------------------------------------------------------
The image displays two geometric shapes on a white background. On the left is a
red circle, and on the right is a blue square. Below the shapes, the text
"Image 1: Shapes" is written.
--------------------------------------------------------------------------------
Generated 45 tokens in 16.06s  (2.8 tok/s)
================================================================================
```

### MaxEngine backend (`qwen3_vl_demo_engine.py`)

```
================================================================================
Qwen3-VL Demo  [backend=engine  model=qwen3-vl-2b (MaxEngine checkpoint)]
================================================================================
Image(s) : tests/assets/image1.jpg
Prompt   : 'Describe what you see in the image.'
--------------------------------------------------------------------------------
RESPONSE
--------------------------------------------------------------------------------
The image displays two geometric shapes on a white background. On the left is a
red circle, and on the right is a blue square. Below the shapes, the text
"Image 1: Shapes" is written.
--------------------------------------------------------------------------------
Generated 45 tokens in 9.15s  (4.9 tok/s)
================================================================================
```

The JAX and MaxEngine backends produce identical text (same MaxText weights,
greedy decode).  The HF backend is slightly more verbose, as expected from an
independent implementation.

---

## Backend Details

### HF backend — `qwen3_vl_demo_hf.py`

Loads the model via `AutoProcessor` + `AutoModelForImageTextToText` from
HuggingFace `transformers`.  No special hardware required.

- **Model weights**: downloaded / cached from `Qwen/Qwen3-VL-2B-Instruct`
- **Generation**: `do_sample=False` (greedy) for reproducible comparison
- **Startup**: ~10 s (model load on CPU)
- **Per-token speed**: ~300 ms/token (CPU float32)

### JAX backend — `qwen3_vl_demo_jax.py`

Uses the MaxText Flax NNX model directly.  Manages the KV-cache and generation
loop manually with a JIT-compiled single-step decode function.

- **Model weights**: loaded from the orbax checkpoint via `SingleDeviceSharding`
- **Vision encoder**: 448 × 448 input → 196 deepstack feature vectors per image
- **Generation**: greedy argmax, one JIT-compiled forward pass per token
- **Sequence buffer**: fixed at 1024 tokens (nearest multiple of 512 ≥ prompt + max_new_tokens); required by TPU splash attention block size
- **Startup**: ~90 s (model init + checkpoint restore + vision encoder warmup)
- **Per-token speed**: ~0 ms after the first XLA compile step

Key implementation notes:
- Visual token count: `(448 / patch_size=16 / spatial_merge=2)² = 14² = 196`
- mRoPE positions computed by `get_rope_index` with `image_grid_thw = [[1, 28, 28]]`
- Bidirectional attention mask applied to all `<|image_pad|>` token positions
- Config: `src/maxtext/configs/post_train/sft.yml` + `model_name=qwen3-vl-2b`

### MaxEngine backend — `qwen3_vl_demo_engine.py`

Uses the MaxText MaxEngine serving API (`load_params` → `prefill` → `insert` →
`generate` loop).  Recommended path for production deployments.

- **Model weights**: same orbax checkpoint, loaded by MaxEngine's `setup_decode_state`
- **prefill**: `engine.prefill(params, padded_tokens, positions=mrope_pos, mrope_deltas=deltas, images=pixel_values, true_length=seq_len)`
- **decode loop**: `engine.generate(params, decode_state)` per step; result read with `sampled_tokens.get_result_at_slot(0).tokens.item()`
- **Startup**: ~2 min (MaxEngine model init, KV-cache allocation, checkpoint load)
- **Per-token speed**: ~0 ms after JIT compile

---

## Feature Comparison

| Feature | HF | JAX | MaxEngine |
|---------|----|-----|-----------|
| Device | CPU | TPU v6 lite | TPU v6 lite |
| Checkpoint source | HuggingFace Hub | orbax (MaxText) | orbax (MaxText) |
| Image input | ✅ (any size) | ✅ (resized to 448 px) | ✅ (resized to 448 px) |
| Autoregressive generation | ✅ | ✅ one step/token | ✅ one step/token |
| JIT / XLA compilation | ❌ | ✅ | ✅ |
| Deepstack vision fusion | via HF model | ✅ explicit | ✅ via MaxEngine |
| KV-cache management | HF internal | manual | MaxEngine |
| Production serving path | ❌ | ❌ | ✅ |
| Startup time | ~10 s | ~90 s | ~2 min |
| Throughput (measured) | 3.5 tok/s | 2.8 tok/s* | 4.9 tok/s |

\* JAX demo timing includes the first JIT-compile step amortised across 45 tokens.

---

## Command-Line Reference

All three scripts share these common flags:

```
--image PATH [PATH ...]    Input image file(s)  (required; first image used)
--prompt TEXT              Text prompt  [default: "Describe what you see in the image."]
--max-tokens N             Max new tokens  [default: 512]
--output-json              Print result as JSON instead of formatted text
--verbose                  Extra logging (token-by-token output for JAX/Engine)
```

Backend-specific flags:

```
# HF only:
--model ID                 HuggingFace model ID or local dir  [default: Qwen/Qwen3-VL-2B-Instruct]

# JAX and MaxEngine:
--checkpoint-dir PATH      Orbax checkpoint directory  [default: tests/assets/qwen3_vl_2b_orbax]
--tokenizer ID             HuggingFace tokenizer ID  [default: Qwen/Qwen3-VL-2B-Instruct]
```

JSON output example (all backends return the same schema):

```json
{
  "backend": "jax",
  "model": "qwen3-vl-2b (JAX/NNX checkpoint)",
  "image": ["tests/assets/image1.jpg"],
  "prompt": "Describe what you see in the image.",
  "response": "The image displays two geometric shapes ...",
  "tokens": 45,
  "elapsed": 16.06,
  "tok_per_sec": 2.8
}
```

---

## Dev Utility — `tools/dev/qwen3_vl_smoke_forward.py`

A lightweight smoke-test script that verifies the MaxText Qwen3-VL decoder can
be instantiated and JIT-compiled **without any checkpoint or real images**.  It
uses random weights (`enable_checkpointing=False`) and a truncated 2-layer model
(`base_num_decoder_layers=2`) so it completes in seconds on any device.

```bash
source maxtext_venv/bin/activate
python3 tools/dev/qwen3_vl_smoke_forward.py
```

Expected output:
```
Instantiating Transformer...
(1, 16, 151936)
```

Use this to:
- Quickly verify the model graph compiles after code changes (no checkpoint I/O)
- Sanity-check JAX device availability and XLA compilation
- Confirm `src/maxtext/configs/post_train/sft.yml` + `model_name=qwen3-vl-2b` loads correctly

---

## Troubleshooting

**TPU not detected**
```bash
python3 -c "import jax; print(jax.devices())"
# Expected: [TpuDevice(id=0, process_index=0, coords=(0,0,0), core_on_chip=0)]
```

**Import errors (MaxText / orbax not found)**
```bash
source maxtext_venv/bin/activate  # activates the venv with all dependencies
```

**`enable_checkpointing` validation error (MaxEngine demo)**
Pass `enable_checkpointing=True` together with `load_parameters_path` — both are
required when using MaxEngine with a checkpoint.  The demo already sets this;
if writing custom code make sure both flags are present.

**Prompt exceeds `max_prefill_predict_length` (MaxEngine demo)**
The default prefill length is 512.  Long prompts with many image tokens may
exceed this.  Increase `_MAX_PREFILL` at the top of `qwen3_vl_demo_engine.py`.

---

## Regenerating Large Test Assets

The following files/directories are **not committed to git** because of their size.
Run the commands below once to recreate them locally before using the JAX/MaxEngine
demos or running the integration tests.

### 1. HuggingFace weights + Orbax checkpoint (`tests/assets/qwen3_vl_2b_hf/` and `tests/assets/qwen3_vl_2b_orbax/`)

```bash
# Downloads Qwen3-VL-2B-Instruct from HuggingFace and converts it to the
# MaxText Orbax format in one step.
source maxtext_venv/bin/activate
python3 tools/data_generation/generate_hf_qwen3_vl_checkpoint.py --size 2b
```

This will:
1. Download `Qwen/Qwen3-VL-2B-Instruct` from HuggingFace Hub and save it to `tests/assets/qwen3_vl_2b_hf/`
2. Run `src/maxtext/checkpoint_conversion/to_maxtext.py` to produce the Orbax checkpoint at `tests/assets/qwen3_vl_2b_orbax/`

For the 8B model, pass `--size 8b` (requires more RAM/disk).

Custom output paths:
```bash
python3 tools/data_generation/generate_hf_qwen3_vl_checkpoint.py \
  --size 2b \
  --hf-dir /path/to/hf_weights \
  --orbax-dir /path/to/orbax_ckpt
```

### 2. Golden logits (`tests/assets/golden_logits/golden_data_qwen3_vl_logits.jsonl`)

```bash
# Runs the full Qwen3-VL-2B forward pass on tests/assets/test_image.jpg via
# HuggingFace and saves logits + hidden states for integration test comparison.
source maxtext_venv/bin/activate
python3 tools/data_generation/generate_golden_qwen3_vl_logits.py
```

Requires: `tests/assets/test_image.jpg` (committed), HuggingFace internet access or a local HF copy.
Output: `tests/assets/golden_logits/golden_data_qwen3_vl_logits.jsonl` (~350 MB)

### 3. Golden vision encoder data (`tests/assets/golden_logits/golden_data_qwen3_vl_vit.jsonl`)

```bash
# Extracts the vision encoder (ViT) output from HuggingFace Qwen3-VL-2B on
# tests/assets/test_image.jpg and saves soft_embeddings + pixel_values.
source maxtext_venv/bin/activate
python3 tools/data_generation/generate_golden_qwen3_vl_vit.py
```

Output: `tests/assets/golden_logits/golden_data_qwen3_vl_vit.jsonl` (~30 MB)

### Summary

| Asset | Size | Generation script |
|-------|------|-------------------|
| `tests/assets/qwen3_vl_2b_hf/` | ~8 GB | `tools/data_generation/generate_hf_qwen3_vl_checkpoint.py` |
| `tests/assets/qwen3_vl_2b_orbax/` | ~3.7 GB | same script (step 2/2) |
| `tests/assets/golden_logits/golden_data_qwen3_vl_logits.jsonl` | ~1.1 GB | `tools/data_generation/generate_golden_qwen3_vl_logits.py` |
| `tests/assets/golden_logits/golden_data_qwen3_vl_vit.jsonl` | ~97 MB | `tools/data_generation/generate_golden_qwen3_vl_vit.py` |

---

**Last updated**: March 17, 2026
