# MiMo-V2-Flash Inference on TPU with MaxText

MiMo-V2-Flash is a 309B-total / 15B-active Mixture-of-Experts transformer from
[Xiaomi AI Research](https://huggingface.co/XiaomiMiMo/MiMo-V2-Flash), designed
for complex reasoning tasks.  This guide describes how to run inference with
MaxText on Google TPU (v6e Trillium / Ironwood v7).

---

## Architecture Overview

| Property | Value |
|---|---|
| Total parameters | 309 B |
| Active parameters | ~15 B per token |
| Hidden size | 4096 |
| Decoder layers | 48 |
| Attention heads (Q) | 64 |
| Attention heads (KV, GA) | 4 |
| Attention heads (KV, SWA) | 8 |
| Q/K head dimension | 192 |
| **V head dimension** | **128** (asymmetric with Q/K) |
| Partial RoPE fraction | 0.334 → rotates 64 of 192 dims |
| Global attention layers | 9 (positions 0, 5, 11, 17, 23, 29, 35, 41, 47) |
| Sliding-window layers | 39 (all other positions) |
| SWA window size | 128 tokens |
| MoE layers | 47 of 48 (layer 0 is dense MLP) |
| Routed experts | 256 |
| Experts per token | 8 (top-8) |
| Routing function | sigmoid + noaux-TC correction bias |
| MoE intermediate size | 2048 |
| Dense MLP intermediate | 16384 (layer 0 only) |
| Vocabulary size | 152576 (Qwen2 tokenizer) |
| Max context length | 262144 tokens |

### Key Architecture Differences

Compared to a standard LLaMA/Qwen-style model, MiMo-V2-Flash has several
notable differences that required custom MaxText support:

**Asymmetric head dimensions:** The Q and K projections use `head_dim=192`
while V uses `v_head_dim=128`.  This means the output projection has shape
`(num_q_heads × v_head_dim, hidden_dim)` = `(64×128, 4096)`.

**Partial RoPE:** Only the first `rope_dim = int(192 × 0.334) = 64` dimensions
of each Q/K head are rotated; the remaining 128 pass through unchanged.  Two
separate RoPE bases are used: `rope_theta=5000000` for global attention layers
and `swa_rope_theta=10000` for SWA layers.

**Attention sink bias:** Sliding-window layers include a learnable per-head
scalar bias (`attention_sink_bias`, shape `num_q_heads`) added to the attention
logits before softmax.  This lets the model route unimportant tokens to a
"sink".

**noaux-TC MoE routing:** The router uses `sigmoid` scoring (not softmax).
A learned `e_score_correction_bias` (shape `num_experts`) is added to the
sigmoid scores *before* top-k selection only.  The final expert weights use the
unbiased sigmoid scores, then L1-normalised.

---

## Prerequisites

1. A TPU VM capable of holding the full model weights (~620 GB bfloat16).
   Supported TPU generations:
   - **v6e (Trillium) — minimum: v6e-32** — 32 chips × 32 GB HBM = 1024 GB total;
     the smallest v6e slice that fits the 309 B-parameter model with headroom
     for KV cache and activations.  Larger slices (v6e-64, v6e-128, v6e-256)
     improve throughput and batch size.
   - **Ironwood (TPU v7) — minimum: Ironwood-4** — 192 GB HBM per chip;
     4 chips × 192 GB = 768 GB total, which fits the full 309B model (~618 GB
     bfloat16) with ~150 GB headroom for KV cache and activations.  Larger
     slices (Ironwood-8, Ironwood-16, Ironwood-32, …) improve throughput and
     batch size.  Each Ironwood chip delivers ~4,614 BF16 TFLOPS.
2. MaxText installed:
   ```bash
   cd maxtext && pip install -e "src/[gpu]"
   ```
3. HuggingFace `transformers`, `safetensors`, and `huggingface_hub`:
   ```bash
   pip install transformers safetensors huggingface_hub
   ```

---

## Step 1: Convert HF Checkpoint to MaxText Format

MiMo-V2-Flash is distributed as HuggingFace `safetensors` shards.  Convert
them to an Orbax checkpoint that MaxText can load.

### Memory modes

| Mode | Peak RAM | Disk (tmpdir) | Best for |
|---|---|---|---|
| **Streaming** (`--tmpdir`) | **~25 GB** | ~650 GB (bfloat16) | v6e-1, any low-RAM VM |
| In-RAM (default) | ~970 GB | none | Large multi-socket hosts |

The streaming mode processes one decoder layer at a time and writes converted
arrays to memory-mapped files so that RAM usage never exceeds approximately
one MoE layer (~25 GB).  The only requirement is ~650 GB of free scratch space
accessible from the VM (a local SSD or a mounted persistent disk).

### Running on a v6e-1 (streaming mode)

```bash
# Attach a persistent disk with ≥650 GB free, e.g. mounted at /mnt/scratch.
# Then run:
python3 -m maxtext.checkpoint_conversion.standalone_scripts.convert_mimo_v2_flash \
    --base_model_path /local/path/to/MiMo-V2-Flash \
    --maxtext_model_path gs://<your-bucket>/mimo-v2-flash/checkpoints/0/items \
    --tmpdir /mnt/scratch/mimo_tmp \
    --simulated_cpu_devices_count 1
```

`--tmpdir` enables streaming mode.  The memmap files under `mimo_tmp` are
**not** cleaned up automatically when `--tmpdir` is specified, so you can
inspect them or reuse them if the save step needs to be retried.  Use
`--streaming` instead if you want the tmpdir to be created and removed
automatically:

```bash
python3 -m maxtext.checkpoint_conversion.standalone_scripts.convert_mimo_v2_flash \
    --base_model_path /local/path/to/MiMo-V2-Flash \
    --maxtext_model_path gs://<your-bucket>/mimo-v2-flash/checkpoints/0/items \
    --streaming \
    --simulated_cpu_devices_count 1
```

### Running on a high-RAM machine (in-RAM mode)

```bash
# Download and convert in one step
python3 -m maxtext.checkpoint_conversion.standalone_scripts.convert_mimo_v2_flash \
    --base_model_path XiaomiMiMo/MiMo-V2-Flash \
    --maxtext_model_path gs://<your-bucket>/mimo-v2-flash/checkpoints/0/items \
    --download_from_hub \
    --simulated_cpu_devices_count 16

# Or, if you have the model locally:
python3 -m maxtext.checkpoint_conversion.standalone_scripts.convert_mimo_v2_flash \
    --base_model_path /local/path/to/MiMo-V2-Flash \
    --maxtext_model_path gs://<your-bucket>/mimo-v2-flash/checkpoints/0/items
```

The conversion:
- Loads weights from all `*.safetensors` shards
- Transposes linear weight matrices from HF `(out, in)` to MaxText `(in, out)`
- Reshapes attention projections into `(hidden, heads, head_dim)` layout
- Stacks per-expert MoE weights into `(num_experts, dim_in, dim_out)` tensors
- Saves an Orbax/Zarr3 checkpoint compatible with MaxText's parameter loader

**Memory note (streaming mode):** Each decoder layer's raw tensors are loaded
from the relevant shard files, converted, flushed to disk as memmaps, and freed
before the next layer begins.  The large MoE expert stacks (256 × 3 matrices
per layer) are the peak allocation, at ~25 GB in float32.

---

## Step 2: Run Inference

### Using the JAX demo script

```bash
python3 demos/mimo_v2_flash_demo_jax.py \
    --checkpoint_path gs://<your-bucket>/mimo-v2-flash/checkpoints/0/items \
    --tokenizer_path XiaomiMiMo/MiMo-V2-Flash \
    --prompt "Solve step by step: if a rectangle has perimeter 48 and one side 10,
what is its area?" \
    --max_new_tokens 256
```

### Using `maxtext.inference.decode` directly

```bash
python3 -m maxtext.inference.decode \
    src/maxtext/configs/base.yml \
    src/maxtext/configs/models/mimo-v2-flash.yml \
    run_name=mimo_inference_run \
    checkpoint_dir=gs://<your-bucket>/mimo-v2-flash/checkpoints/0/items \
    tokenizer_path=XiaomiMiMo/MiMo-V2-Flash \
    tokenizer_type=huggingface \
    prompt="What is the capital of France?" \
    per_device_batch_size=1 \
    max_prefill_predict_length=512 \
    max_target_length=1024
```

### HuggingFace reference baseline

For a PyTorch/CPU/GPU reference, use the HF demo:

```bash
python3 demos/mimo_v2_flash_demo_hf.py \
    --model_path XiaomiMiMo/MiMo-V2-Flash \
    --prompt "Explain backpropagation step by step." \
    --max_new_tokens 128 \
    --load_in_4bit   # reduces VRAM to ~80 GB with bitsandbytes
```

---

## Configuration Reference

The MaxText config file is at
[`src/maxtext/configs/models/mimo-v2-flash.yml`](../../../src/maxtext/configs/models/mimo-v2-flash.yml).
Key parameters:

| Config key | Default | Description |
|---|---|---|
| `decoder_block` | `mimo_v2_flash` | Selects MiMo-V2-Flash decoder layers |
| `mimo_hybrid_layer_pattern` | `[0,1,1,1,1,0,…]` | Per-layer attention type (0=GA, 1=SWA) |
| `mimo_moe_layer_freq` | `[0,1,1,…,1]` | Per-layer MoE flag (0=dense, 1=MoE) |
| `mimo_v_head_dim` | `128` | V projection head dimension |
| `mimo_swa_num_kv_heads` | `8` | KV heads for sliding-window layers |
| `mimo_swa_rope_theta` | `10000.0` | RoPE theta for SWA layers |
| `mimo_swa_window_size` | `128` | Sliding window size (tokens) |
| `mimo_attention_value_scale` | `0.707` | Scale factor applied to V before attention |
| `partial_rotary_factor` | `0.334` | Fraction of head dims that are rotated |
| `rope_max_timescale` | `5000000` | RoPE theta for global attention layers |
| `num_experts` | `256` | Total routed experts per MoE layer |
| `num_experts_per_tok` | `8` | Experts selected per token |
| `routed_score_func` | `sigmoid` | MoE routing scoring function |

To override a parameter on the command line, append `key=value`.  For example:

```bash
python3 -m maxtext.inference.decode \
    src/maxtext/configs/base.yml \
    src/maxtext/configs/models/mimo-v2-flash.yml \
    max_target_length=2048   # increase generation length
```

---

## Sharding Recommendations

For the full 309B model:

| TPU topology | Parallelism | Notes |
|---|---|---|
| **v6e-32 (32 chips)** | **TP(32)** | **Minimum viable v6e config**; 1024 GB HBM fits model + KV cache |
| v6e-64 (64 chips) | FSDP(2) × TP(32) | Better batch throughput |
| v6e-128 (128 chips) | FSDP(4) × TP(32) | High-throughput serving |
| v6e-256 (256 chips) | FSDP(16) × TP(16) | Maximum throughput on v6e |
| **Ironwood-4 (4 chips)** | **TP(4)** | **Minimum viable Ironwood config**; 768 GB HBM fits model + small KV cache |
| Ironwood-8 (8 chips) | TP(8) | 1,536 GB HBM; comfortable headroom for model + KV cache + large batches |
| Ironwood-16 (16 chips) | TP(16) or FSDP(2) × TP(8) | High-throughput serving |
| Ironwood-32 (32 chips) | FSDP(4) × TP(8) | Maximum single-host throughput on Ironwood |

For the **v6e-32 minimum config**, use pure tensor parallelism across all 32 chips:

```bash
ici_tensor_parallelism=32 \
scan_layers=false \
per_device_batch_size=1
```

For larger v6e slices (e.g. v6e-64), combine FSDP and TP:

```bash
ici_fsdp_parallelism=2 \
ici_tensor_parallelism=32 \
scan_layers=false
```

For smaller batch sizes or prefill-only benchmarks, tensor parallelism alone is
often sufficient:

```bash
ici_tensor_parallelism=16 \
per_device_batch_size=4
```

---

## Running Tests

Two test suites are provided, targeting different levels of validation:

### Unit tests (CPU / any device)

```bash
# From the MaxText repo root:
python3 -m pytest tests/unit/mimo_v2_flash_architecture_test.py -v
```

These tests run on CPU with a toy configuration and verify:
- Router gate output shapes and L1-normalisation
- Attention output shapes for both global and SWA layers
- Asymmetric head dim handling (V ≠ Q/K)
- SWA `sink_bias` presence/absence
- MoE block output shape and finiteness
- Decoder layer type selection (GA vs SWA, dense vs MoE)
- Config field parsing

### TPU execution tests (requires a v6e or other TPU chip)

```bash
python3 -m pytest tests/unit/mimo_v2_flash_tpu_test.py -v
```

These tests require an actual TPU device and validate that the MiMo
implementation executes correctly in the **production bfloat16 dtype** on real
TPU hardware.  They cover:

| Test class | What is validated |
|---|---|
| `TestMiMoV2FlashTPUDevicePlacement` | Output tensors live on the TPU chip, not CPU; output dtype is `bfloat16` |
| `TestMiMoV2FlashJITCompilation` | All 4 layer variants (dense-GA, MoE-SWA, MoE-GA) `jax.jit`-compile and execute without error |
| `TestMiMoV2FlashGradients` | `jax.grad` through MoE dispatch and SWA/GA attention is finite |
| `TestMiMoV2FlashDeterminism` | Repeated identical forward passes return bitwise-equal results |
| `TestMiMoV2FlashPartialRoPEOnTPU` | Partial RoPE modifies the correct head dimensions on device |
| `TestMiMoV2FlashMoEGateTPU` | Gate routing executes on TPU, weights L1-sum to ≈1 in bfloat16, JIT-compiles |
| `TestMiMoV2FlashMaskingOnTPU` | Causal masking is not violated (future tokens do not affect past outputs) |
| `TestMiMoV2FlashFullStackTPU` | All 4 decoder layers run in sequence; output is finite; full stack JIT-compiles |

Run both suites together:

```bash
python3 -m pytest tests/unit/mimo_v2_flash_architecture_test.py \
                  tests/unit/mimo_v2_flash_tpu_test.py -v
```

---

## Implementation Notes

| Component | Location |
|---|---|
| Model implementation | `src/maxtext/models/mimo_v2_flash.py` |
| Config fields | `src/maxtext/configs/types.py` |
| Decoder registry | `src/maxtext/layers/decoders.py` |
| Model YAML config | `src/maxtext/configs/models/mimo-v2-flash.yml` |
| DecoderBlockType enum | `src/maxtext/common/common_types.py` |
| Checkpoint conversion | `src/maxtext/checkpoint_conversion/standalone_scripts/convert_mimo_v2_flash.py` |
| Param mapping | `src/maxtext/checkpoint_conversion/utils/param_mapping.py` |
| Unit tests | `tests/unit/mimo_v2_flash_architecture_test.py` |
| TPU execution tests | `tests/unit/mimo_v2_flash_tpu_test.py` |
| HF demo | `demos/mimo_v2_flash_demo_hf.py` |
| JAX / TPU demo | `demos/mimo_v2_flash_demo_jax.py` |

---

## Limitations and Known Issues

- **KV cache / paged attention:** The current implementation does not yet wire
  up incremental KV-cache for autoregressive decoding.  SWA layers need
  sliding-window cache management.  Full paged-attention support is tracked
  as a follow-up.
- **Scan layers:** `scan_layers=true` is not yet validated for MiMo.  Use
  `scan_layers=false` (the default in `mimo-v2-flash.yml`).
- **FP8 weights:** The original model was trained in FP8.  The conversion
  script loads to bfloat16.  FP8-to-bfloat16 dequantisation is handled by
  `safetensors` transparently if the shards were already dequantised, but
  if you download the raw FP8 shards you may need to run
  `tools/checkpoint_conversion/standalone_scripts/deepseek_fp8_to_bf16.py`
  as a pre-processing step.
