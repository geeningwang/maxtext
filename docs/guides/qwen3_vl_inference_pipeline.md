# Qwen3-VL Inference Pipeline: Module-by-Module Guide

This document traces the full inference data flow for Qwen3-VL models in MaxText,
from a raw image(s), video, or mixed image+video input combined with a text prompt,
to decoded output text. Each section covers the
relevant source file, input/output tensor specs, and how to verify correctness.

---

## Pipeline Overview

```
Image file path(s) (str) and/or video path (str) + text prompt (str)
        │
        ▼
① preprocess_mm_data(config)              ← multimodal/processor.py  (routes to below)
  ├─ preprocess_image_qwen3_vl()            ← multimodal/processor_qwen3_vl.py  [if image_path set]
  │       │  pixel_values    (N, 3, 2, H_bar, W_bar)   dynamic resolution
  │       │  image_grid_thw  (N, 3)
  ├─ preprocess_video_qwen3_vl()            ← multimodal/processor_qwen3_vl.py  [if video_path set]
  │       │  pixel_values_videos  (1, 3, T_padded, H_bar, W_bar)
  │       │  video_grid_thw       (1, 3)
  └─ merge_preprocessor_outputs_qwen3_vl()  ← multimodal/processor_qwen3_vl.py  [if both set]
          │  combines all image + video fields into one Qwen3VLPreprocessorOutput
        ▼
② reformat_prompt_qwen3_vl()             ← multimodal/processor_qwen3_vl.py
        │  formatted prompt string (Qwen chat template)
        ▼
③ tokenizer.encode()                     ← HuggingFace BPE tokenizer
        │  token_ids  int[]
        ▼
④ add_extra_tokens_for_images_qwen3_vl()   ← multimodal/processor_qwen3_vl.py  [images]
   add_extra_tokens_for_video_qwen3_vl()    ← multimodal/processor_qwen3_vl.py  [video]
        │  expanded token_ids  (dynamic token count per image/video)
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

**Entry point (file/video path → tensor):** `src/maxtext/multimodal/processor.py` → `preprocess_mm_data(config)`  
**Entry point (array → tensor):** `src/maxtext/multimodal/processor.py` → `preprocess_image_for_training(image, model_name)`  
**Image implementation:** `src/maxtext/multimodal/processor_qwen3_vl.py` → `preprocess_image_qwen3_vl(images)`  
**Video implementation:** `src/maxtext/multimodal/processor_qwen3_vl.py` → `preprocess_video_qwen3_vl(source)`

- **`preprocess_mm_data`**: starts from file path(s). Independently runs each preprocessor
  that has a non-empty path set, then combines the results:
  - `config.image_path` set → calls `preprocess_image_qwen3_vl` with **dynamic resolution**
  - `config.video_path` set → calls `preprocess_video_qwen3_vl`
  - **Both set** → calls both, then merges via `merge_preprocessor_outputs_qwen3_vl` to produce
    a single `Qwen3VLPreprocessorOutput` with all image *and* video fields populated
  - Neither set → raises `ValueError`

  Callers that receive paths at call time construct a `types.SimpleNamespace(model_name=...,
  image_path=..., video_path=...)` rather than modifying the shared pyconfig.
  **Does not support `qwen3-omni`** via this path — that model passes the whole config to its own
  preprocessor.
- **`preprocess_image_for_training`**: starts from a pre-loaded `np.ndarray`. Used by the SFT
  training data pipeline. Always applies `force_size=(448, 448)` so that all images in a batch
  have identical spatial shapes and can be stacked.
  **Does not support `qwen3-omni`** — that model is not present in its routing table.

### Inputs

**`preprocess_mm_data(config)`** (inference demos / server / CLI):

| Arg | Required? | Description |
|-----|-----------|-------------|
| `config.model_name` | always | `"qwen3-vl-2b"` or `"qwen3-vl-8b"` |
| `config.image_path` | optional | Comma-separated image file path(s). Omit, leave absent, or set to empty string for video-only or text-only input. |
| `config.video_path` | optional | Path to a single video file (GIF, MP4, etc.). Omit, leave absent, or set to empty string for image-only or text-only input. Set both for mixed image+video input. |

At least one of `image_path` / `video_path` must be non-empty for visual input. Both may be absent or empty for **text-only** input — in that case `preprocess_mm_data` returns an empty `Qwen3VLPreprocessorOutput` (`num_images=0`, `num_videos=0`, all visual fields `None`). Skip `preprocess_mm_data` entirely if you prefer not to call it for text-only requests.

**`preprocess_image_for_training(image, model_name)`** (SFT training data pipeline):

| Arg | Type / Shape | Description |
|-----|-------------|-------------|
| `image` | `np.ndarray (H, W, 3)` uint8, or `list[np.ndarray]` | One or more RGB images decoded from a dataset record (e.g. via `mm_utils.convert_to_RGB`) and converted to `np.ndarray` |
| `model_name` | str | `"qwen3-vl-2b"`, `"qwen3-vl-8b"`, `"gemma3-*"`, or `"llama4-*"` — `qwen3-omni` is not supported |

### Output — `Qwen3VLPreprocessorOutput`

**Image fields** (from `preprocess_image_qwen3_vl`):

| Field | Shape | Description |
|-------|-------|-------------|
| `pixel_values` | `(N, 3, 2, H_bar, W_bar)` float32 | Normalised pixel tensor. T=2 (image duplicated for temporal axis). H_bar/W_bar multiples of 32. |
| `image_grid_thw` | `(N, 3)` int32 | Grid dims `[grid_t, grid_h, grid_w]` where `grid_h = H_bar // 16`, `grid_w = W_bar // 16` |
| `num_images` | int | N |

Inference uses **dynamic resolution**: H_bar × W_bar ∈ [min_pixels, max_pixels], aligned to multiples of `QWEN3_VL_RESIZE_FACTOR` (32).  
Training (`preprocess_image_for_training`) always uses `force_size=(448, 448)` → H_bar=W_bar=448, grid always `[1, 28, 28]`.

In **mixed image+video mode**, all image fields and all video fields are present simultaneously in the same `Qwen3VLPreprocessorOutput`.

**Video fields** (from `preprocess_video_qwen3_vl`):

| Field | Shape | Description |
|-------|-------|-------------|
| `pixel_values_videos` | `(1, 3, T_padded, H_bar, W_bar)` float32 | Normalised video frames. T_padded is multiple of `QWEN3_VL_TEMPORAL_PATCH_SIZE`. |
| `video_grid_thw` | `(1, 3)` int32 | `[grid_t, grid_h, grid_w]` where `grid_t = T_padded // 2` |
| `num_videos` | int | 1 |

### Key constants
| Constant | Value | Description |
|----------|-------|-------------|
| `QWEN3_VL_PATCH_SIZE` | 16 | Spatial patch side length |
| `QWEN3_VL_TEMPORAL_PATCH_SIZE` | 2 | Temporal patch depth — images duplicated, video padded to multiples of 2 |
| `QWEN3_VL_SPATIAL_MERGE_SIZE` | 2 | PatchMerger 2×2 → 1 token (4× reduction) |
| `QWEN3_VL_RESIZE_FACTOR` | 32 | `= patch_size × merge_size`; output H/W must be multiples of this |
| `QWEN3_VL_IMAGE_MIN_PIXELS` | 3136 (56×56) | Minimum H×W for image dynamic resize |
| `QWEN3_VL_IMAGE_MAX_PIXELS` | 1003520 (28×28×1280) | Maximum H×W for image dynamic resize |
| `QWEN3_VL_IMAGE_SIZE` | 448 | Fixed resize for training (force_size) |
| `QWEN3_VL_VIDEO_DEFAULT_FPS` | 2.0 | Default output fps for video sampling |
| `QWEN3_VL_VIDEO_MIN_PIXELS` | 131072 | Minimum T×H×W for video dynamic resize |
| `QWEN3_VL_VIDEO_MAX_PIXELS` | 786432 | Maximum T×H×W for video dynamic resize |
| Normalisation | `(pixel − 127.5) / 127.5` → `[−1, +1]` | Applied to both images and video frames |

### Verifying correctness
```python
# Image — dynamic resolution
assert out.pixel_values.ndim == 5             # (N, C, T, H, W)
assert out.pixel_values.shape[1] == 3         # RGB
assert out.pixel_values.shape[2] == 2         # temporal duplicate
assert out.pixel_values.shape[3] % 32 == 0    # H_bar multiple of resize_factor
assert out.pixel_values.shape[4] % 32 == 0    # W_bar multiple of resize_factor
assert out.pixel_values.min() >= -1.1
assert out.pixel_values.max() <= +1.1

# Image — training (force_size)
assert out.pixel_values.shape == (N, 3, 2, 448, 448)
assert np.all(out.image_grid_thw == [1, 28, 28])

# Video
assert out.pixel_values_videos.ndim == 5       # (1, C, T, H, W)
assert out.pixel_values_videos.shape[2] % 2 == 0  # T_padded divisible by temporal_patch_size
```

```python
# Mixed image+video mode
import types
config = types.SimpleNamespace(
    model_name="qwen3-vl-2b",
    image_path="photo.jpg",
    video_path="clip.gif",
)
out = preprocess_mm_data(config)  # out has pixel_values AND pixel_values_videos
assert out.num_images >= 1
assert out.num_videos >= 1
assert out.pixel_values is not None
assert out.pixel_values_videos is not None
```

**Tests:**
```
pytest tests/unit/qwen3_vl_sft_data_processing_test.py -k "pixel"              # training path
pytest tests/unit/qwen3_vl_preprocessor_test.py -k "PreprocessMmData"          # dynamic inference path
pytest tests/unit/qwen3_vl_preprocessor_test.py -k "PreprocessVideo"           # video path
pytest tests/unit/qwen3_vl_preprocessor_test.py -k "MixedImageVideo"           # mixed image+video path
```

### `merge_preprocessor_outputs_qwen3_vl`

**File:** `src/maxtext/multimodal/processor_qwen3_vl.py`  
**Function:** `merge_preprocessor_outputs_qwen3_vl(image_output, video_output)`

Combines two separate `Qwen3VLPreprocessorOutput` objects — one from `preprocess_image_qwen3_vl` and one from `preprocess_video_qwen3_vl` — into a single output with all fields populated.

```python
image_out = preprocess_image_qwen3_vl(images)          # pixel_values, image_grid_thw, num_images
video_out = preprocess_video_qwen3_vl(video_path)      # pixel_values_videos, video_grid_thw, num_videos
merged   = merge_preprocessor_outputs_qwen3_vl(image_out, video_out)
# merged.pixel_values         ← from image_out (unchanged)
# merged.image_grid_thw       ← from image_out (unchanged)
# merged.pixel_values_videos  ← from video_out (unchanged)
# merged.video_grid_thw       ← from video_out (unchanged)
```

Called automatically by `preprocess_mm_data` when both `image_path` and `video_path` are set.

---

## Module 2 — Prompt Formatter

**File:** `src/maxtext/multimodal/processor_qwen3_vl.py`  
**Function:** `reformat_prompt_qwen3_vl(prompt, num_images, num_videos=0, image_placeholder="<|image|>", video_placeholder="<|video|>")`

### Input
| Arg | Type | Default | Description |
|-----|------|---------|-------------|
| `prompt` | str | — | Raw user prompt, may contain placeholders |
| `num_images` | int | — | Number of images for this example |
| `num_videos` | int | `0` | Number of videos for this example |
| `image_placeholder` | str | `"<\|image\|>"` | Generic image placeholder |
| `video_placeholder` | str | `"<\|video\|>"` | Generic video placeholder |

### Output
Formatted string (Qwen chat template). Examples:
```
# Image-only input
<|im_start|>user
<|vision_start|><|image_pad|><|vision_end|>{user text}<|im_end|>
<|im_start|>assistant

# Video-only input
<|im_start|>user
<|vision_start|><|video_pad|><|vision_end|>{user text}<|im_end|>
<|im_start|>assistant

# Mixed image+video input (num_images=1, num_videos=1)
<|im_start|>user
<|vision_start|><|image_pad|><|vision_end|><|vision_start|><|video_pad|><|vision_end|>{user text}<|im_end|>
<|im_start|>assistant
```

- `<|image_pad|>` (token ID `151655`) is inserted once per image
- `<|video_pad|>` (token ID `151656`) is inserted once per video
- Missing placeholders are prepended automatically (images first, then videos)

### Why it matters
The sentinel tokens are what Module 4 expands into per-image/per-video vision tokens. Their positions in the sequence determine where visual embeddings are injected in Module 8.

**Tests:**
```
pytest tests/unit/qwen3_vl_sft_data_processing_test.py -k "image_tokens"
pytest tests/unit/qwen3_vl_preprocessor_test.py -k "ReformatPrompt"
```

---

## Module 3 — HuggingFace Tokenizer

Standard BPE tokenizer from `Qwen/Qwen3-VL-2B-Instruct` (or `8B`). Converts the formatted string into integer token IDs. The special token `<|image_pad|>` maps to ID `151655`.

No custom logic — use the HuggingFace `AutoTokenizer` directly.

---

## Module 4 — Visual Token Expansion

**File:** `src/maxtext/multimodal/processor_qwen3_vl.py`

Two functions — one for images, one for video. The router `prepare_text_for_image_fusion` in `processor.py` applies **both in sequence** for qwen3-vl, so a single call handles image-only, video-only, and mixed inputs transparently:

```python
# Inside prepare_text_for_image_fusion (processor.py)
tokens = add_extra_tokens_for_images_qwen3_vl(tokens, processor_output)  # no-op if no images
tokens = add_extra_tokens_for_video_qwen3_vl(tokens, processor_output)   # no-op if no video
```

Each function is a **no-op** when its corresponding grid field (`image_grid_thw` / `video_grid_thw`) is `None` in the processor output.

| Function | Expands | Uses field | No-op when |
|----------|---------|------------|------------|
| `add_extra_tokens_for_images_qwen3_vl(tokens, processor_output)` | `<\|image_pad\|>` (151655) | `processor_output.image_grid_thw` | `image_grid_thw is None` |
| `add_extra_tokens_for_video_qwen3_vl(tokens, processor_output)` | `<\|video_pad\|>` (151656) | `processor_output.video_grid_thw` | `video_grid_thw is None` |

### Token count formula
```
num_tokens = grid_t × grid_h × grid_w // spatial_merge_size²  (÷ 4)

# Image examples:
#   448×448  → [1, 28, 28] →  1×28×28÷4 = 196 tokens  (training / force_size)
#   320×224  → [1, 14, 10] →  1×14×10÷4 =  35 tokens  (dynamic inference)

# Video example:
#   8 frames, 240×320 → [4, 15, 20] → 4×15×20÷4 = 300 tokens
```

The token count is **dynamic at inference time** — it depends on `image_grid_thw` / `video_grid_thw` from the preprocessor output. Only training (with `force_size=448`) always yields 196 tokens per image.

### Verifying correctness
```python
# Image expansion
from maxtext.multimodal.processor_qwen3_vl import add_extra_tokens_for_images_qwen3_vl
tokens = np.array([IMAGE_TOKEN])  # single placeholder
expanded = add_extra_tokens_for_images_qwen3_vl(tokens, proc_out)
expected = int(np.prod(proc_out.image_grid_thw[0])) // 4
assert len(expanded) == expected

# Video expansion
from maxtext.multimodal.processor_qwen3_vl import add_extra_tokens_for_video_qwen3_vl
tokens_v = np.array([VIDEO_TOKEN])
expanded_v = add_extra_tokens_for_video_qwen3_vl(tokens_v, proc_out)
expected_v = int(np.prod(proc_out.video_grid_thw[0])) // 4
assert len(expanded_v) == expected_v
```

**Tests:**
```
pytest tests/unit/qwen3_vl_sft_data_processing_test.py -k "segmentation or image_tokens"  # training (196)
pytest tests/unit/qwen3_vl_preprocessor_test.py -k "AddExtraTokens"                       # dynamic + video
pytest tests/unit/qwen3_vl_preprocessor_test.py -k "MixedImageVideo"                      # mixed mode
```

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
| `patch_embed` (3D conv) | `(batch, 3, 2, 448, 448)` | `(batch, 784, hidden_size_for_vit)` | Spatial+temporal patchify |
| Raster→block permutation | `(batch, 784, hidden_size_for_vit)` | `(batch, 784, hidden_size_for_vit)` | Reorder to 2×2 block order for PatchMerger |
| Pos embed + ViT blocks | `(batch, 784, hidden_size_for_vit)` | `(batch, 784, hidden_size_for_vit)` | Self-attention with 2D RoPE; 24 blocks (2B) / 27 blocks (8B) |
| **Projector (PatchMerger)** | `(batch, 784, hidden_size_for_vit)` | `(batch, 196, out_hidden_size_for_vit)` | 2×2 merge → LN → Linear → GELU → Linear |

`hidden_size_for_vit` = 1024 (qwen3-vl-2b) or 1152 (qwen3-vl-8b). `out_hidden_size_for_vit` = 2048 (qwen3-vl-2b) or 4096 (qwen3-vl-8b) — equals the LLM `emb_dim` (`base_emb_dim`).

**PatchMerger detail (qwen3-vl-2b; `hidden_size_for_vit`=1024, `out_hidden_size_for_vit`=2048):**
```
input:   (batch, 784, 1024)
reshape: (batch, 196, 4×1024) = (batch, 196, 4096)
LN → Linear(4096→4096) → GELU → Linear(4096→2048)
output:  (batch, 196, 2048)
```
For qwen3-vl-8b (`hidden_size_for_vit`=1152, `out_hidden_size_for_vit`=4096): reshape gives `(batch, 196, 4608)`; MLP is 4608→4608→4096.

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
| Image preprocessor (training) | `pytest tests/unit/qwen3_vl_sft_data_processing_test.py -k "pixel"` | Shape `(N,3,2,448,448)`, range `[−1,+1]` |
| Image preprocessor (dynamic) | `pytest tests/unit/qwen3_vl_preprocessor_test.py -k "PreprocessMmData"` | Dynamic H_bar/W_bar multiples of 32, pixel bounds |
| Video preprocessor | `pytest tests/unit/qwen3_vl_preprocessor_test.py -k "PreprocessVideo"` | Shape `(1,3,T,H,W)`, temporal padding, factor alignment |
| Token expansion (images) | `pytest tests/unit/qwen3_vl_sft_data_processing_test.py -k "image_tokens or segmentation"` | 196 tokens per image, mask alignment |
| Token expansion (dynamic+video) | `pytest tests/unit/qwen3_vl_preprocessor_test.py -k "AddExtraTokens"` | Dynamic token count, video expansion |
| Mixed image+video mode | `pytest tests/unit/qwen3_vl_preprocessor_test.py -k "MixedImageVideo"` | Merge fields, combined token expansion, offsets |
| mRoPE position IDs | `pytest tests/unit/qwen3_omni_layers_test.py` | Shape `(3,batch,seq)`, text/image ranges |
| Vision encoder | `pytest tests/integration/qwen3_vl_checkpoint_validation_test.py -k "vision_encoder"` | Shape `(batch,196,emb_dim)`, values vs golden |
| Full model logits | `pytest tests/integration/qwen3_vl_checkpoint_validation_test.py -k "full_model"` | Logit shape `(batch,seq,vocab)`, values vs golden |
| End-to-end inference | `pytest tests/integration/qwen3_vl_checkpoint_validation_test.py -k "end_to_end or batch or determinism"` | Greedy output matches reference, batch consistency |
| SFT overfit loop | `python qwen3_vl_demo_sft.py --steps 300` | Loss → 0.0000, AFTER answer = "magenta" |

---

## Key Constants Reference

| Config key | Typical value (2B) | Description |
|------------|--------------------|-------------|
| `hidden_size_for_vit` | 1024 (2B) / 1152 (8B) | ViT hidden dimension |
| `out_hidden_size_for_vit` | 2048 (2B) / 4096 (8B) | Projector output = LLM `emb_dim` |
| `patch_size_for_vit` | 16 | Spatial patch size |
| `temporal_patch_size_for_vit` | 2 | Temporal patch size |
| `spatial_merge_size_for_vit` | 2 | 2×2 merge → 4× token reduction |
| `num_hidden_layers_for_vit` | 24 (2B) / 27 (8B) | ViT depth |
| `emb_dim` | 2048 (2B) / 4096 (8B) | LLM embedding dimension |
| `num_decoder_layers` | 28 (2B) / 36 (8B) | LLM depth |

---

## Related Files

| Purpose | File |
|---------|------|
| Inference/training preprocessing router | `src/maxtext/multimodal/processor.py` |
| Image/video preprocessing & prompt formatting | `src/maxtext/multimodal/processor_qwen3_vl.py` |
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
