# Qwen3-VL Inference Pipeline: Module-by-Module Guide

This document traces the full inference data flow for Qwen3-VL models in MaxText,
from a raw image + text prompt to decoded output text. Each section covers the
relevant source file, input/output tensor specs, and how to verify correctness.

---

## Pipeline Overview

```
Image file path (str) + text prompt (str)
        │
        ▼
① preprocess_mm_data(config)              ← multimodal/processor.py  (routes to below)
  └─ preprocess_mm_data_qwen3_vl()          ← multimodal/processor_qwen3_vl.py
        │  pixel_values  (N, 3, 2, 448, 448)
        │  pixel_grid_thw (N, 3)
        ▼
② reformat_prompt_qwen3_vl()             ← multimodal/processor_qwen3_vl.py
        │  formatted prompt string (Qwen chat template)
        ▼
③ tokenizer.encode()                     ← HuggingFace BPE tokenizer
        │  token_ids  int[]
        ▼
④ add_extra_tokens_for_images_qwen3_vl() ← multimodal/processor_qwen3_vl.py
        │  expanded token_ids  (196 image tokens per image)
        ▼
⑤ get_rope_index()                       ← multimodal/processor_qwen3_omni.py
        │  position_ids  (3, 1, seq_len)
        │  mrope_deltas  (1, 1)
        ▼
⑥ engine.load_params()                   ← inference/maxengine/maxengine.py
        │  Params  (sharded pytree on mesh)
        ▼
⑦ engine.init_decode_state()             ← inference/maxengine/maxengine.py
        │  DecodeState  (zeroed KV cache + metadata)
        ▼
⑧ engine.prefill()                       ← inference/maxengine/maxengine.py
        │  internally calls model.apply() → Transformer.__call__()
        │    ├─ VisionEncoder.__call__()      ← layers/encoders.py + models/qwen3-vl.py
        │    │    ├─ Qwen3VLVisionEncoder     (patch embed → 32 ViT blocks)
        │    │    └─ Qwen3OmniMoeVisionProjector  (PatchMerger)
        │    ├─ merge_mm_embeddings()         ← multimodal/utils.py
        │    └─ Decoder  (28/32 layers, mRoPE attention + SwiGLU FFN)
        │  → Prefix  {cache, logits, next_pos, first_token}
        ▼
⑨ engine.insert()                        ← inference/maxengine/maxengine.py
        │  DecodeState  (prefill KV cache inserted at slot)
        ▼
⑩ engine.generate()  [loop N steps]     ← inference/maxengine/maxengine.py
        │  (DecodeState, ResultTokens)  per step
        ▼
  tokenizer.decode(generated_token_ids)  →  output text string
```

---

## Module 1 — Image Preprocessor

**Entry point (file path → tensor):** `src/maxtext/multimodal/processor.py` → `preprocess_mm_data(config)`  
**Entry point (array → tensor):** `src/maxtext/multimodal/processor.py` → `preprocess_image_for_training(image, model_name)`  
**Implementation:** `src/maxtext/multimodal/processor_qwen3_vl.py` → `preprocess_mm_data_qwen3_vl(images)`

Both entry points route to the same underlying implementation.

- **`preprocess_mm_data`**: starts from a file path (`config.image_path`). Callers that receive
  the image path at call time (inference demos, API server) construct a
  `types.SimpleNamespace(model_name=..., image_path=...)` rather than modifying the shared
  pyconfig. `decode.py` and other CLI callers pass the full pyconfig directly since `image_path`
  is already baked in at startup.
  **Does not support `qwen3-omni`** via this path — that model passes the whole config to its own
  preprocessor.
- **`preprocess_image_for_training`**: starts from a pre-loaded `np.ndarray`. Used by the SFT
  training data pipeline (`input_pipeline_utils.py::pre_process_image_sft`) where images arrive
  as decoded PIL images (converted via `mm_utils.convert_to_RGB`) already in memory.
  **Does not support `qwen3-omni`** — that model is not present in its routing table.

### Inputs

**`preprocess_mm_data(config)`** (inference demos / server / CLI):

| Arg | Description |
|-----|-------------|
| `config.model_name` | `"qwen3-vl-2b"` or `"qwen3-vl-8b"` |
| `config.image_path` | Comma-separated image file path(s). Callers with a runtime path use `types.SimpleNamespace(model_name=..., image_path=...)`. |

**`preprocess_image_for_training(image, model_name)`** (SFT training data pipeline):

| Arg | Type / Shape | Description |
|-----|-------------|-------------|
| `image` | `np.ndarray (H, W, 3)` uint8, or `list[np.ndarray]` | One or more RGB images decoded from a dataset record (e.g. via `mm_utils.convert_to_RGB`) and converted to `np.ndarray` |
| `model_name` | str | `"qwen3-vl-2b"`, `"qwen3-vl-8b"`, `"gemma3-*"`, or `"llama4-*"` — `qwen3-omni` is not supported |

### Output — `Qwen3VLPreprocessorOutput`
| Field | Shape | Description |
|-------|-------|-------------|
| `pixel_values` | `(N, 3, 2, 448, 448)` float32 | Normalised pixel tensor. T=2 (image duplicated for temporal axis) |
| `pixel_grid_thw` | `(N, 3)` int32 | Grid dims `[grid_t, grid_h, grid_w]` = `[1, 28, 28]` for 448×448 |
| `num_images` | int | N |

### Key constants
| Constant | Value |
|----------|-------|
| `QWEN3_VL_IMAGE_SIZE` | 448 — resize target (BICUBIC) |
| `QWEN3_VL_PATCH_SIZE` | 16 — patch side length → 28×28 grid |
| `QWEN3_VL_TEMPORAL_PATCH_SIZE` | 2 — image duplicated along T |
| `QWEN3_VL_NUM_FRAMES` | 2 → after fold: `num_frames = 2 // 2 = 1` |
| Normalisation | `(pixel − 127.5) / 127.5` → range `[−1, +1]` |

### Verifying correctness
```python
# Shape check
assert out.pixel_values.shape == (N, 3, 2, 448, 448)
assert out.pixel_grid_thw.shape == (N, 3)
assert np.all(out.pixel_grid_thw == [1, 28, 28])

# Value range check
assert out.pixel_values.min() >= -1.1
assert out.pixel_values.max() <= +1.1
```

**Tests:** `pytest tests/unit/qwen3_vl_sft_data_processing_test.py -k "pixel"`

---

## Module 2 — Prompt Formatter

**File:** `src/maxtext/multimodal/processor_qwen3_vl.py`  
**Function:** `reformat_prompt_qwen3_vl(prompt, image_placeholder, num_images)`

### Input
| Arg | Type | Description |
|-----|------|-------------|
| `prompt` | str | Raw user prompt, may contain `image_placeholder` |
| `image_placeholder` | str | Generic placeholder (e.g. `"<\|image\|>"`) |
| `num_images` | int | Number of images for this example |

### Output
Formatted string (Qwen chat template):
```
<|im_start|>user
<|vision_start|><|image_pad|><|vision_end|>{user text}<|im_end|>
<|im_start|>assistant
```

- `<|image_pad|>` (token ID `151655`) is inserted once per image
- Missing image placeholders are prepended automatically

### Why it matters
The `<|image_pad|>` sentinel is what Module 4 expands into the 196 per-image vision tokens. Its position in the sequence determines where image embeddings are later injected in Module 8.

**Tests:** `pytest tests/unit/qwen3_vl_sft_data_processing_test.py -k "image_tokens"`

---

## Module 3 — HuggingFace Tokenizer

Standard BPE tokenizer from `Qwen/Qwen3-VL-2B-Instruct` (or `8B`). Converts the formatted string into integer token IDs. The special token `<|image_pad|>` maps to ID `151655`.

No custom logic — use the HuggingFace `AutoTokenizer` directly.

---

## Module 4 — Image Token Expansion

**File:** `src/maxtext/multimodal/processor_qwen3_vl.py`  
**Function:** `add_extra_tokens_for_images_qwen3_vl(tokens, processor_output)`

### Input
| Arg | Shape | Description |
|-----|-------|-------------|
| `tokens` | `int[seq_compact]` 1D | Token IDs from tokenizer (compact, one placeholder per image) |
| `processor_output.pixel_grid_thw` | `(N, 3)` | Grid dimensions |

### Output
`np.ndarray int32[seq_expanded]` — each `<|image_pad|>` (151655) replaced by:
```
num_tokens = grid_t × grid_h × grid_w // spatial_merge_size²
           = 1 × 28 × 28 // 4
           = 196   tokens per image (at 448×448)
```

### Verifying correctness
```python
original_image_count = (tokens == 151655).sum()
expanded = add_extra_tokens_for_images_qwen3_vl(tokens, proc_out)
extra = (expanded == 151655).sum() - original_image_count
assert extra == original_image_count * 195  # each placeholder → 196 copies (net +195)
```

**Tests:** `pytest tests/unit/qwen3_vl_sft_data_processing_test.py -k "segmentation or image_tokens"`

---

## Module 5 — mRoPE Position ID Computation

**File:** `src/maxtext/multimodal/processor_qwen3_omni.py`  
**Function:** `get_rope_index(input_ids, image_grid_thw, attention_mask, ...)`

### Input
| Arg | Shape | Description |
|-----|-------|-------------|
| `input_ids` | `(batch, seq_len)` int32 | Expanded token IDs |
| `image_grid_thw` | `(N, 3)` int32 | Per-image grid dimensions |
| `attention_mask` | `(batch, seq_len)` int32 | 1=real token, 0=padding |

### Output
| Tensor | Shape | Description |
|--------|-------|-------------|
| `position_ids` | `(3, batch, seq_len)` float32 | 3D positions: dim-0=temporal, dim-1=height, dim-2=width |
| `mrope_position_deltas` | `(batch, 1)` float32 | Offset to add to `next_pos` after prefill |

### Position assignment rules

| Token type | dim 0 (temporal) | dim 1 (height) | dim 2 (width) |
|------------|-----------------|----------------|---------------|
| Text before image | sequential `0,1,2…` | same | same |
| Image tokens | frame index | row index `0..grid_h//2-1` | col index `0..grid_w//2-1` |
| Text after image | `max(image_pos)+1`, `+2`, … | same | same |

**`mrope_deltas`** corrects `next_pos` in the decode state after prefill, accounting for the fact that mRoPE position values exceed the raw sequence length due to 2D spatial indexing. Without it, autoregressive decoding would start at the wrong position.

**Tests:** `pytest tests/unit/qwen3_omni_layers_test.py`

---

## Module 6 — `engine.load_params()`

**File:** `src/maxtext/inference/maxengine/maxengine.py`

### Input
| Arg | Description |
|-----|-------------|
| `rng` | `jax.random.PRNGKey` |
| `params` | (optional) existing param pytree to reshard |

### Output
`Params` — sharded pytree of all model weights placed on the mesh according to the config's logical axis rules. Also sets internal engine state: `abstract_params`, `prefill_kv_cache_annotations`, `prefill_kv_cache_shardings`.

---

## Module 7 — `engine.init_decode_state()`

**File:** `src/maxtext/inference/maxengine/maxengine.py`

Runs a dummy forward pass to trace all cache buffer shapes, then returns zeroed buffers.

### Output — `DecodeState` dict
| Key | Shape | Description |
|-----|-------|-------------|
| `"cache"` | nested dict per layer | KV cache: `(batch, heads, max_target_len, head_dim)` per layer, zeroed |
| `"tokens"` | `(batch, 1)` int32 | Previous token (starts at 0) |
| `"next_pos"` | `(batch, 1)` int32 | Next autoregressive position (starts at 0) |
| `"generated_tokens"` | `(batch, 1)` int32 | Count of generated tokens |
| `"logits"` | `(batch, 1, vocab_size)` float32 | Last step logits |

**Critical note:** `engine.insert()` has `donate_argnums=(1, 2)` — it consumes the `decode_state` buffer. Re-initialize with `init_decode_state()` before each new request.

---

## Module 8 — `engine.prefill()` (with Vision Encoder)

**File:** `src/maxtext/inference/maxengine/maxengine.py` → `models/models.py` → `layers/decoders.py`

### Inputs to `prefill()`
| Arg | Shape | Description |
|-----|-------|-------------|
| `params` | Params pytree | From `load_params` |
| `padded_tokens` | `(max_prefill_len,)` int32 | Expanded tokens, padded |
| `positions` | `(3, max_prefill_len)` int32 | mRoPE `position_ids` — pass the `(3, batch, seq)` output squeezed to `(3, seq)` |
| `mrope_deltas` | `(1, 1)` int32 | From `get_rope_index` |
| `images` | `(N, 3, 2, 448, 448)` float32 | From image preprocessor |
| `true_length` | int | Actual unpadded token count |

### 8a — Vision Encoder: `Qwen3VLVisionEncoder`

**Files:** `src/maxtext/layers/encoders.py` → `src/maxtext/models/qwen3-vl.py` → `src/maxtext/models/qwen3.py`

| Stage | Input shape | Output shape | Notes |
|-------|------------|-------------|-------|
| `patch_embed` (3D conv) | `(batch, 3, 2, 448, 448)` | `(batch, 784, 1536)` | Spatial+temporal patchify |
| Raster→block permutation | `(batch, 784, 1536)` | `(batch, 784, 1536)` | Reorder to 2×2 block order for PatchMerger |
| Pos embed + 32 ViT blocks | `(batch, 784, 1536)` | `(batch, 784, 1536)` | Self-attention with 2D RoPE |
| **Projector (PatchMerger)** | `(batch, 784, 1536)` | `(batch, 196, emb_dim)` | 2×2 merge → LN → Linear → GELU → Linear |

`emb_dim` = 1536 (qwen3-vl-2b) or 3584 (qwen3-vl-8b).

**PatchMerger detail:**
```
input:  (batch, 784, 1536)
reshape: (batch, 196, 4×1536) = (batch, 196, 6144)
LN → Linear(6144→6144) → GELU → Linear(6144→emb_dim)
output: (batch, 196, emb_dim)
```

The `freeze_vision_encoder_params` config flag applies `jax.lax.stop_gradient` to the ViT output before the projector, keeping the ViT frozen during SFT.

### 8b — Embedding Merge: `merge_mm_embeddings()`

**File:** `src/maxtext/multimodal/utils.py`

| | Shape | Description |
|-|-------|-------------|
| **Input** `text_embeddings` | `(batch, seq_len, emb_dim)` | Token embeddings after `shared_embedding` lookup |
| **Input** `image_embeddings` | `(batch, 196, emb_dim)` | From VisionEncoder projector |
| **Input** `mask` | `(batch, seq_len)` bool | 1 at positions of `<\|image_pad\|>` tokens |
| **Output** | `(batch, seq_len, emb_dim)` | Text embeddings with 196 image vectors spliced in at masked positions |

### 8c — Decoder Stack

**File:** `src/maxtext/layers/decoders.py`

| | Shape | Description |
|-|-------|-------------|
| **Input** | `(batch, seq_len, emb_dim)` | Merged embeddings |
| **Positions** | `(3, 1, seq_len)` | mRoPE 3D position IDs passed through to attention |
| **Per layer** | — | RMSNorm → mRoPE multi-head attention → RMSNorm → SwiGLU FFN |
| **Output logits** | `(batch, seq_len, vocab_size)` float32 | |

### Prefill output — `Prefix` dict
| Key | Shape | Description |
|-----|-------|-------------|
| `"cache"` | per-layer KV arrays | Populated prefill KV cache |
| `"tokens"` | `(1, 1)` int32 | First sampled token |
| `"next_pos"` | `(1, 1)` int32 | `true_length + mrope_delta` |
| `"logits"` | `(1, 1, vocab_size)` float32 | Logits at position `true_length-1` |

**Tests:**
- Vision encoder accuracy: `pytest tests/integration/qwen3_vl_checkpoint_validation_test.py -k "vision_encoder"`  
  Compares against `tests/assets/golden_logits/golden_data_qwen3_vl_vit.jsonl`
- Full model accuracy: `pytest tests/integration/qwen3_vl_checkpoint_validation_test.py -k "full_model"`  
  Compares against `tests/assets/golden_logits/golden_data_qwen3_vl_logits.jsonl`

---

## Module 9 — `engine.insert()`

**File:** `src/maxtext/inference/maxengine/maxengine.py`

Copies the prefill KV cache into a specific batch slot of the decode KV cache.

### Input
| Arg | Description |
|-----|-------------|
| `prefix` | `Prefix` dict from `prefill` |
| `decode_state` | `DecodeState` (donated — consumed by `_insert_jit`) |
| `slot` | int — which batch slot to fill |

### Output
Updated `DecodeState` with `"cache"` at index `slot` populated from the prefill cache.

**Critical:** `donate_argnums=(1, 2)` means the input `decode_state` is consumed (its buffers donated to the output). Always re-create `decode_state = engine.init_decode_state(rng)` before calling `insert` again for a new request.

---

## Module 10 — `engine.generate()` (autoregressive loop)

**File:** `src/maxtext/inference/maxengine/maxengine.py`

Each call performs **one** autoregressive decode step.

### Input
| Key in `decode_state` | Shape | Description |
|----------------------|-------|-------------|
| `"tokens"` | `(batch, 1)` int32 | Token from previous step (or first sampled token from prefill) |
| `"next_pos"` | `(batch, 1)` int32 | Current position counter |
| `"cache"` | nested dict | KV cache with all previous tokens |

### Output
| | Description |
|-|-------------|
| `new_decode_state` | Updated `DecodeState` — `"tokens"` = new token, `"next_pos"` incremented by 1 |
| `ResultTokens` | `.data` shape `(batch, 3)` — columns: `[token_id, valid_flag, length]` |

### Extracting generated tokens
```python
decode_state, result = engine.generate(params, decode_state, rng=rng)
token_id = result.data[0, 0].item()  # batch slot 0
is_valid = result.data[0, 1].item()  # 1 if valid, 0 if EOS
```

**Tests:**
```
pytest tests/integration/qwen3_vl_checkpoint_validation_test.py -k "end_to_end"
pytest tests/integration/qwen3_vl_checkpoint_validation_test.py -k "batch"
pytest tests/integration/qwen3_vl_checkpoint_validation_test.py -k "determinism"
```

---

## Complete Verification Checklist

| Module | Test command | What is checked |
|--------|-------------|-----------------|
| Image preprocessor | `pytest tests/unit/qwen3_vl_sft_data_processing_test.py -k "pixel"` | Shape `(N,3,2,448,448)`, range `[−1,+1]` |
| Token expansion | `pytest tests/unit/qwen3_vl_sft_data_processing_test.py -k "image_tokens or segmentation"` | 196 tokens per image, mask alignment |
| mRoPE position IDs | `pytest tests/unit/qwen3_omni_layers_test.py` | Shape `(3,batch,seq)`, text/image ranges |
| Vision encoder | `pytest tests/integration/qwen3_vl_checkpoint_validation_test.py -k "vision_encoder"` | Shape `(batch,196,emb_dim)`, values vs golden |
| Full model logits | `pytest tests/integration/qwen3_vl_checkpoint_validation_test.py -k "full_model"` | Logit shape `(batch,seq,vocab)`, values vs golden |
| End-to-end inference | `pytest tests/integration/qwen3_vl_checkpoint_validation_test.py -k "end_to_end or batch or determinism"` | Greedy output matches reference, batch consistency |
| SFT overfit loop | `python qwen3_vl_demo_sft.py --steps 300` | Loss → 0.0000, AFTER answer = "magenta" |

---

## Key Constants Reference

| Config key | Typical value (2B) | Description |
|------------|--------------------|-------------|
| `hidden_size_for_vit` | 1536 | ViT hidden dimension |
| `out_hidden_size_for_vit` | 1536 (2B) / 3584 (8B) | Projector output = LLM `emb_dim` |
| `patch_size_for_vit` | 16 | Spatial patch size |
| `temporal_patch_size_for_vit` | 2 | Temporal patch size |
| `spatial_merge_size_for_vit` | 2 | 2×2 merge → 4× token reduction |
| `num_hidden_layers_for_vit` | 32 | ViT depth |
| `emb_dim` | 1536 (2B) / 3584 (8B) | LLM embedding dimension |
| `num_decoder_layers` | 28 (2B) / 28 (8B) | LLM depth |

---

## Related Files

| Purpose | File |
|---------|------|
| Inference/training preprocessing router | `src/maxtext/multimodal/processor.py` |
| Image preprocessing & prompt formatting | `src/maxtext/multimodal/processor_qwen3_vl.py` |
| mRoPE + token expansion (shared with Qwen3-Omni) | `src/maxtext/multimodal/processor_qwen3_omni.py` |
| Embedding merge utilities | `src/maxtext/multimodal/utils.py` |
| VisionEncoder / AudioEncoder wrappers | `src/maxtext/layers/encoders.py` |
| Qwen3-VL ViT (raster→block permutation) | `src/maxtext/models/qwen3-vl.py` |
| Qwen3 ViT blocks, projector, decoder layers | `src/maxtext/models/qwen3.py` |
| Top-level Transformer (model.apply entry point) | `src/maxtext/models/models.py` |
| Decoder stack + embedding merge dispatch | `src/maxtext/layers/decoders.py` |
| MaxEngine (prefill / insert / generate) | `src/maxtext/inference/maxengine/maxengine.py` |
| SFT overfit demo | `qwen3_vl_demo_sft.py` |
| Integration tests (golden logits) | `tests/integration/qwen3_vl_checkpoint_validation_test.py` |
| Unit tests (data pipeline) | `tests/unit/qwen3_vl_sft_data_processing_test.py` |
| Unit tests (layers) | `tests/unit/qwen3_omni_layers_test.py` |
| Golden reference data | `tests/assets/golden_logits/golden_data_qwen3_vl_*.jsonl` |

---

## See Also

- [Qwen3-VL SFT Guide](qwen3_vl_sft.md)
- [Qwen3-VL Pre-training Gap Analysis](qwen3_vl_pretrain.md)
- [Qwen3-VL Inference Guide](qwen3_vl_inference.md)
