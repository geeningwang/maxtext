# MiMo-V2-Flash Inference Pipeline: Module-by-Module Guide

This document traces the full MiMo-V2-Flash inference data flow in MaxText,
from a raw text prompt to decoded output text. Each section covers the
relevant source file, input/output specs, and practical correctness checks.

---

## Pipeline Overview

```
User prompt (str)
    |
    v
1) demos/mimo_v2_flash_demo_jax.py
   build_decode_command(...)
    |
    v
2) inference/decode.py
   - pyconfig.initialize(argv)
   - MaxEngine(config)
    |
    v
3) engine.load_params(...)
   inference/maxengine/maxengine.py
   -> load Orbax checkpoint (zarr2 or zarr3+OCDBT)
   -> shard params on mesh (TP/EP)
    |
    v
4) Tokenization + prompt formatting
   inference/decode.py
   - tokenizer.apply_chat_template(...)   [when use_chat_template=true]
   - tokenizer.encode(...)
    |
    v
5) engine.prefill(...)
   inference/maxengine/maxengine.py
   -> model.apply(..., model_mode='prefill')
   -> decoder stack (MiMo layers)
   -> prefill cache + first token
    |
    v
6) engine.insert(prefill_result, decode_state)
   inference/maxengine/maxengine.py
   -> seed AR decode state with prefill cache
    |
    v
7) engine.generate(...) loop
   inference/maxengine/maxengine.py
   -> model.apply(..., model_mode='autoregressive')
   -> one token per step
   -> EOS check each step
    |
    v
8) tokenizer.decode(generated_ids)
   -> final assistant text output
```

---

## Module 0 - Checkpoint Preparation (Required Before Runtime)

MiMo-V2-Flash runtime depends on a converted MaxText checkpoint.

### Source files

- `src/MaxText/checkpoint_conversion/standalone_scripts/convert_mimo_v2_flash.py`
- `src/MaxText/checkpoint_conversion/standalone_scripts/convert_mimo_v2_flash_distributed.py`
- `src/MaxText/tools/convert_checkpoint_to_ocdbt.py`
- `src/MaxText/tools/mimo_stack_checkpoint.py` (only for scan_layers=true workflow)

### What happens

1. Read HF safetensors shards.
2. Apply FP8 dequantization using `weight_scale_inv`.
3. Map HF parameter names/shapes to MaxText layout.
4. Write Orbax checkpoint (zarr2, optionally zarr3+OCDBT).
5. Optionally stack into 4-phase scan layout for MiMo scan path.

### Critical correctness requirement

Do not skip FP8 block-scale dequantization. Casting raw FP8 bytes to BF16
without applying `weight_scale_inv` produces invalid generations.

---

## Module 1 - Entry Script and Decode Command Builder

### Source file

- `demos/mimo_v2_flash_demo_jax.py`

### Main responsibilities

1. Exposes a practical CLI for single-host and multi-host TPU launches.
2. Builds a full `python -m maxtext.inference.decode ...` command.
3. Injects MiMo-specific config overrides.
4. Parses decode output and reports throughput/EOS status.

### Core MiMo flags injected

- `model_name=mimo-v2-flash`
- `decoder_block=mimo_v2_flash`
- `mimo_v_head_dim=128`
- `mimo_hybrid_layer_pattern=[...]`
- `mimo_moe_layer_freq=[0]+[1]*47`
- `mimo_swa_num_kv_heads=8`
- `mimo_swa_window_size=128`
- `mimo_attention_value_scale=0.707`
- `ici_tensor_parallelism=4` and `ici_expert_parallelism=8` (typical v6e-32)

### Inputs

- Checkpoint path
- Tokenizer path
- Prompt string
- Runtime knobs: `max_prefill`, `max_new_tokens`, TP/EP, chunked prefill, etc.

### Output

- Parsed generated text
- Token/sec estimate
- EOS-fired indicator

---

## Module 2 - Decode Orchestration

### Source file

- `src/MaxText/inference/decode.py`

### Main responsibilities

1. Initialize runtime config and engine.
2. Load model parameters.
3. Prepare prompt text and tokenize.
4. Run prefill once (or chunked prefill loop).
5. Insert prefill cache into decode state.
6. Run autoregressive generate loop.
7. Stop when EOS appears or max target length is reached.

### Input

- OmegaConf/pyconfig runtime args from CLI.
- Prompt and tokenizer metadata.

### Output

- Final decoded text via tokenizer.
- Timing and HBM diagnostics.

### EOS behavior

When `use_chat_template=true`, decode uses tokenizer chat formatting and checks
for EOS token each generate step. On EOS, loop exits early.

---

## Module 3 - Engine Initialization and Parameter Loading

### Source file

- `src/MaxText/inference/maxengine/maxengine.py`

### Main responsibilities

1. Build mesh with configured parallelism axes.
2. Construct transformer model for prefill/decode.
3. Load and shard checkpoint parameters.
4. Prepare KV cache annotations/shardings.

### Input

- Config + checkpoint location.

### Output

- Sharded model parameter pytree ready for `prefill` and `generate`.

### Notes

- Supports OCDBT/zarr3 checkpoint loading when enabled.
- Exposes `prefill`, `insert`, `generate`, `init_decode_state` primitives used by decode.py.

---

## Module 4 - MiMo Model Selection and Decoder Wiring

### Source files

- `src/MaxText/layers/decoders.py`
- `src/MaxText/common/common_types.py`
- `src/MaxText/configs/types.py`

### Main responsibilities

1. Route `decoder_block=mimo_v2_flash` to MiMo decoder classes.
2. Configure scan vs non-scan layer execution path.
3. Validate MiMo-specific config fields.

### MiMo-specific config fields

- `mimo_hybrid_layer_pattern`
- `mimo_moe_layer_freq`
- `mimo_v_head_dim`
- `mimo_swa_num_kv_heads`
- `mimo_swa_rope_theta`
- `mimo_swa_window_size`
- `mimo_attention_value_scale`

---

## Module 5 - MiMo Decoder Layer Internals

### Source file

- `src/MaxText/models/mimo_v2_flash.py`

### Per-layer forward path

1. Pre-attention RMSNorm.
2. Self-attention (global or sliding-window depending on layer pattern).
3. Residual add.
4. Post-attention RMSNorm.
5. FFN path:
   - Dense MLP for layer 0.
   - Sparse MoE for layers 1-47 (default pattern).
6. Residual add.

### Attention-specific MiMo behavior

- Asymmetric head dims: Q/K use `head_dim`, V uses `mimo_v_head_dim`.
- SWA layers use local sliding window and SWA KV head count.
- SWA layers can load `sink_bias` parameter.
- Explicit `query_pre_attn_scalar=cfg.head_dim**-0.5` is applied.

### MoE-specific MiMo behavior

- Router: sigmoid scores + noaux-TC correction bias for top-k selection.
- Expert weights computed from unbiased scores, then L1-normalized.
- Expert tensors are batched/stacked for efficient matmul dispatch.

---

## Module 6 - Tokenization and Chat Template

### Source files

- `src/MaxText/inference/decode.py`
- `demos/mimo_v2_flash_demo_jax.py`

### Main responsibilities

1. Build text prompt (optionally chat templated).
2. Encode to token IDs.
3. Enforce prefill-length constraints.

### Why template matters

Using tokenizer chat template generally improves structured assistant output and
clean EOS termination for MiMo conversational prompts.

---

## Module 7 - Prefill and Optional Chunked Prefill

### Source files

- `src/MaxText/inference/decode.py`
- `src/MaxText/inference/maxengine/maxengine.py`

### Standard prefill

- `engine.prefill(...)` processes full prompt token sequence once.
- Returns prefix cache plus first generated token.

### Chunked prefill path

When `use_chunked_prefill=true`, decode splits long prefill into fixed chunks:

1. Prefill first chunk.
2. Build `ExistingPrefix(cache=..., common_prefix_tokens=...)`.
3. Prefill subsequent chunks while extending same cache.

This path is used for long-context scenarios that otherwise OOM.

---

## Module 8 - Insert and Autoregressive Generation

### Source files

- `src/MaxText/inference/decode.py`
- `src/MaxText/inference/maxengine/maxengine.py`

### Flow

1. `engine.init_decode_state(...)` allocates decode state.
2. `engine.insert(prefill_result, decode_state, slot=...)` seeds cache.
3. `engine.generate(...)` runs one AR step per iteration.
4. Each step appends one sampled token to output stream.
5. Decode loop stops at EOS or length cap.

### Output contract

Per step, generate returns:

- Updated `decode_state`
- `sampled_tokens` container with slot token result

---

## Module 9 - Configuration Baseline

### Source file

- `src/MaxText/configs/models/mimo-v2-flash.yml`

### Key defaults

- 48 decoder layers
- 9 global + 39 SWA pattern
- Q/K head dim 192, V head dim 128
- 256 experts, top-8 routing
- SWA window 128
- Partial rotary factor 0.334
- Dense layer 0 + MoE layers 1-47

Use this file as the canonical architecture baseline for inference bring-up.

---

## Module 10 - Benchmark and Validation Hooks

### Source files

- `src/MaxText/inference/scripts/mimo_v2_flash_bench.py`
- `tests/unit/mimo_v2_flash_architecture_test.py`
- `tests/unit/mimo_v2_flash_tpu_test.py`

### What to validate

1. Decode step latency and throughput (steady-state).
2. Prefill latency and throughput (including chunked mode).
3. Deterministic outputs for fixed seeds/config.
4. Finite outputs/gradients and TPU placement in tests.

---

## Quick Bring-Up Checklist

1. Convert HF checkpoint with proper FP8 dequantization.
2. Prefer OCDBT/zarr3 for multi-host load performance.
3. Start with known-good TP/EP values for MiMo (`TP=4`, `EP=8` on v6e-32).
4. Run demo path and verify EOS fires on simple prompts.
5. Run benchmark script to characterize prefill/decode throughput.
6. For long contexts, switch to chunked prefill if required.

---

## Related Documents

- `docs/guides/mimo_v2_flash_inference.md`
- `docs/guides/mimo_v2_flash_inference_overview.md`
- `docs/guides/mimo_v2_flash_tpu_perf_optimization.md`
- `docs/guides/mimo_v2_flash_hf_vs_ocdbt_validation.md`
- `docs/guides/mimo_v2_flash_fp8_dtypes.md`
- `docs/guides/mimo_v2_flash_env_restore.md`
