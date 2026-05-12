# MiMo-V2-Flash — FP8 Weight Dtype Notes

Answers to three questions about MiMo-V2-Flash's native weight format and
hardware/software FP8 support.

---

## 1. MiMo-V2-Flash native weight dtype — FP8 E4M3 (block-wise)

Yes, **FP8** is the native format.  From `config.json` on the HF safetensors
weights (`/mnt/mimo-weights/config.json`):

```json
"torch_dtype": "bfloat16",
"quantization_config": {
  "quant_method": "fp8",
  "fmt": "e4m3",
  "activation_scheme": "dynamic",
  "weight_block_size": [128, 128]
}
```

Walking the first safetensors shard confirms three distinct tensor dtypes:

| Tensor dtype | Contents |
|---|---|
| `torch.float8_e4m3fn` | Weight tensors (MoE/MLP/attention projections) |
| `torch.float32` | FP8 inverse-scale tensors (`weight_scale_inv`), one per 128×128 block |
| `torch.bfloat16` | Non-quantized tensors (embeddings, layer norms, etc.) |

`torch_dtype: bfloat16` designates the compute/activation dtype; the stored
weights themselves are FP8 E4M3.  This is the same block-wise FP8 scheme used
by DeepSeek-V3/V2.5.

### Why SGLang garbles on AMD EPYC

Loading with `quantization_config: null` causes SGLang to skip the
`weight_scale_inv` tensors, casting raw FP8 bytes to BF16 without applying the
per-block scales.  The result is numerically meaningless — the garbled
`葭葭葭…` output observed in the SGLang CPU experiment.  See
[mimo_v2_flash_sglang_cpu.md](mimo_v2_flash_sglang_cpu.md).

llama.cpp avoids this because `convert_hf_to_gguf.py` **dequantizes FP8→BF16
during conversion**, applying the scales correctly before re-quantizing to the
target GGUF type (`BF16`, `Q8_0`, etc.).

---

## 2. llama.cpp supported weight dtypes — no native FP8 GGUF

As of HEAD `9c69907`, llama.cpp's GGUF format has **no FP8 storage type**.

| Category | Types |
|---|---|
| Full precision | `F32`, `F16`, `BF16` |
| Legacy quants | `Q8_0`, `Q4_0`, `Q4_1`, `Q5_0`, `Q5_1` |
| k-quants | `Q2_K`, `Q3_K_S/M/L`, `Q4_K_S/M`, `Q5_K_S/M`, `Q6_K` |
| i-quants (IMatrix) | `IQ1_S/M`, `IQ2_XXS/XS/S/M`, `IQ3_XXS/XS/S/M`, `IQ4_XS/NL` |
| FP8 | **not supported** |

`convert_hf_to_gguf.py` dequantizes FP8 weights to BF16 as an intermediate
step, then re-quantizes to the requested `--outtype`.  Consequently:

- `--outtype bf16` → 618 GB GGUF (identical footprint to raw BF16 weights)
- `--outtype q8_0` → 306 GB GGUF (re-quantized to 8-bit integer)

There is no path to keep weights in FP8 inside a GGUF file.

---

## 3. TPU v6e (Trillium) supported compute dtypes — native FP8

| Dtype | v5e | v6e (Trillium) |
|---|---|---|
| BF16 | ✅ (MXU, primary) | ✅ (MXU, primary) |
| FP32 | ✅ (accumulation only) | ✅ (accumulation only) |
| INT8 | ✅ | ✅ |
| INT4 | ⚠️ (via XLA packing, no native MXU) | ⚠️ |
| FP8 E4M3 | ❌ | ✅ **native MXU** |
| FP8 E5M2 | ❌ | ✅ **native MXU** |

v6e added hardware FP8 in the MXU — a key differentiator from v5e.  In
JAX/MaxText the types are `jax.numpy.float8_e4m3fn` / `float8_e5m2`; XLA
lowers the matmuls to native FP8 MXU ops.

---

## 4. qwix FP8 PTQ — HBM representation on TPU v7x

The [PTQ pipeline](mimo_v2_flash_tpu_v7x_gke_env_restore.md) converts the BF16
MaxText checkpoint to FP8 using qwix `PtqProvider` and saves the result to GCS
as `gs://jingnw-mimo-v2-flash-us-central1/mimo-v2-flash-fp8-ptq/0/items`
(441.79 GiB).

**GCS checkpoint dtypes (in the saved checkpoint files):**

| dtype | Contents |
|---|---|
| `float8_e4m3fn` | Quantized attention weight shards |
| `float32` | Per-tensor FP8 scale factors |
| `bfloat16` | MoE expert weights (not quantized), embeddings, layer norms |

**HBM dtype breakdown at runtime (TPU v7x, measured 2026-05-12 via `addressable_shards`):**

| dtype | HBM (per TensorCore) | Tensors | % |
|---|---|---|---|
| `bfloat16` | **71.928 GB** | 568 | ~100% |
| `uint32` | < 0.001 GB | 3 | ~0% |
| `float8_e4m3fn` | **0 GB** | 0 | 0% |

**Finding:** No `float8_e4m3fn` tensors appear in `jax.live_arrays()` despite
loading from an FP8 PTQ checkpoint.  All 568 tracked weight shards are BF16.

### Interpretation

The qwix `PtqProvider` likely **dequantizes FP8 attention weights → BF16 at
checkpoint load time**, storing BF16 in HBM for inference.  Evidence:

1. `jax.live_arrays()` + `addressable_shards` for dev=0 accounts for exactly
   71.928 GB as BF16, matching `memory_stats().bytes_in_use` = 71.93 GB.
   No "hidden" XLA FP8 allocation remains unaccounted.
2. The FP8 checkpoint (441.79 GiB) is *larger* than the BF16 source (384.43 GiB)
   because only attention layers are FP8 while MoE expert weights (the bulk)
   stay BF16, and scale factors add per-tensor overhead.

### Practical implication

FP8 PTQ in this qwix configuration **saves GCS storage and checkpoint load
transfer** but does **not reduce HBM usage during inference** relative to a
pure BF16 model of the same architecture.  HBM on TPU v7x (72.31 GB per
TensorCore at steady-state) is essentially the same as a BF16 deployment.

For HBM reduction, approaches to explore include:
- **INT8 KV cache** (see [opt3 plan](mimo_v2_flash_opt3_int8_kv_cache_plan.md))
- **Expert sparsity / lazy loading** at MoE dispatch time
- **qwix weight-only quantization** configured to keep FP8 in HBM (requires
  checking qwix provider options for deferred dequantization)

---

## Summary

| | FP8 support |
|---|---|
| MiMo-V2-Flash weights (HF) | ✅ FP8 E4M3 block-wise (native format) |
| MiMo-V2-Flash weights (qwix PTQ checkpoint) | ✅ FP8 in GCS; **BF16 in HBM** (dequantized at load) |
| llama.cpp GGUF | ❌ must dequantize to BF16/Q8_0/etc. at conversion time |
| TPU v6e | ✅ native FP8 MXU ops (E4M3 and E5M2) |
| TPU v7x | ✅ native FP8 MXU ops (E4M3 and E5M2) |
| TPU v5e | ❌ BF16/INT8 only |
| AMD EPYC (avx512_bf16) | ❌ BF16/INT8 only; FP8 requires AMX (Intel Sapphire Rapids+) |
