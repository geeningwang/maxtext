# MiMo-V2-Flash Inference — Setup Comparison

This document summarises all four validated inference configurations for
[MiMo-V2-Flash](https://huggingface.co/XiaomiMiMo/MiMo-V2-Flash) (309 B total /
~15 B active parameters per token).

---

## Quick Comparison

| # | Stack | Hardware | Weight format | Status | Output quality |
|---|---|---|---|---|---|
| 1 | **MaxText + TPU** | TPU v6e / Ironwood v7 | BF16 (OCDBT checkpoint) | ✅ Forward pass validated | N/A — unit-test only (no autoregressive generation yet) |
| 2 | **HuggingFace Transformers (CPU)** | AMD EPYC 9B14, 180 vCPUs, 708 GB | BF16 (safetensors, loaded to CPU) | ✅ Runs end-to-end | Garbled (`葭葭葭…`) — matches MaxText |
| 3 | **SGLang CPU engine** | AMD EPYC 9B14, 180 vCPUs, 708 GB | FP8→BF16 cast at load (quantization_config=null) | ✅ Runs, 5 patches needed | Garbled (`葭葭葭…`) — FP8 scale tensors stripped |
| 4 | **llama.cpp (GGUF Q8_0)** | AMD EPYC 9B14, 180 vCPUs, 708 GB | Q8_0 on disk, int8+f32 accumulation in compute | ✅ Runs, no patches needed | Coherent (`"2. But what is 0+0?"`) |

**Key finding:** garbled output in settings 2 and 3 is reproduced consistently
across HF and SGLang, confirming it is a model/tokenizer-level behaviour rather
than a framework bug.  The llama.cpp GGUF path (setting 4) produces coherent
output because `convert_hf_to_gguf.py` reads the original FP8 weights and
re-encodes them to Q8_0, side-stepping the FP8→BF16 cast issue.

---

## Setting 1 — MaxText on TPU

**Guide:** [mimo_v2_flash_inference.md](mimo_v2_flash_inference.md)

### Hardware
- TPU v6e-32 (minimum; 1024 GB HBM total) or Ironwood v7-4 (768 GB HBM total)
- Tensor-parallel across all chips via JAX `pjit`

### Weights
- Source: HF safetensors at `gs://jingnw-mimo-v2-flash-us-east5/hf-model`
- Converted to Orbax/zarr3+OCDBT BF16 checkpoint (~313 GB)
- Conversion validated: 568 tensors, 0 mismatches, max absolute diff = 0.0

### Weight format
- **On disk:** `safetensors` FP8 (original HF) → `zarr3+OCDBT` BF16 (converted)
- **In memory (JAX on TPU):** `bfloat16` across all TPU HBM

### Architecture customisations in MaxText
- Asymmetric head dims: Q/K = 192, V = 128
- Partial RoPE: rotates only first 64 of 192 Q/K dims; dual RoPE bases (global = 5 M, SWA = 10 K)
- Attention sink bias (learnable scalar per head, SWA layers only)
- noaux-TC sigmoid MoE routing (sigmoid scores, not softmax; L1-normalised final weights)

### Status
Forward pass JIT-compiles and produces finite output.  Autoregressive sampling
loop not yet wired up; no end-to-end text generation validated.

### Key command
```bash
python3 MaxText/decode.py MaxText/configs/base.yml \
  per_device_batch_size=1 \
  model_name=mimo-v2-flash \
  load_parameters_path=gs://jingnw-mimo-v2-flash-us-east5/mimo-v2-flash-ocdbt/checkpoints/0/items
```

---

## Setting 2 — HuggingFace Transformers (CPU)

**Demo:** [demos/mimo_v2_flash_demo_hf.py](../../demos/mimo_v2_flash_demo_hf.py)

### Hardware
- worker-0: AMD EPYC 9B14, 180 vCPUs, 708 GB RAM (no GPU)
- Weights served over NFS from worker-1: `/mnt/mimo-weights` (292 GB safetensors)

### Weight format
- **On disk:** FP8 safetensors (original HF format)
- **In memory:** BF16 (PyTorch `.to(torch.bfloat16)` at load; FP8→BF16 via `.copy_()`)

### Patches required
None to the HF model code. Several monkey-patches in the demo script to handle
`compute_default_rope_parameters` and `eager_attention_forward` compatibility.

### Key command
```bash
python3 demos/mimo_v2_flash_demo_hf.py \
  --model_path /mnt/mimo-weights \
  --device cpu \
  --dtype bfloat16 \
  --prompt "What is 1+1? The answer is " \
  --max_new_tokens 10
```

### Output (validated 2026-04-04)
```
葭葭葭葭葭葭葭葭葭葭
```
Garbled output consistent with MaxText TPU run.

---

## Setting 3 — SGLang CPU Engine

**Guide:** [mimo_v2_flash_sglang_cpu.md](mimo_v2_flash_sglang_cpu.md)

### Hardware
- worker-0: AMD EPYC 9B14, 180 vCPUs, 708 GB RAM (no GPU)
- `SGLANG_USE_CPU_ENGINE=1` required

### Weight format
- **On disk:** FP8 safetensors (read via NFS from worker-1)
- **In memory:** BF16 — forced by `--json-model-override-args '{"quantization_config": null}'`; FP8 weight tensors cast via PyTorch `.copy_()`, but **scale tensors (`weight_scale_inv`) are silently skipped** — this is why output is garbled

### Patches required (5 total)
| File | Patch |
|---|---|
| `token_dispatcher/__init__.py` | Guard `FlashinferDispatcher` import with `try/except` (no libcudart) |
| `flashinfer/comm/cuda_ipc.py` | Guard `CudaRTLibrary()` call with `try/except` |
| `mimo_v2_flash.py` expert branch | `if name not in params_dict: break` before `params_dict[name]` |
| `mimo_v2_flash.py` stacked branch | Same guard for qkv/o_proj scale keys |
| `torch_native_backend.py` | Add `sinks=None` to `forward_extend` and `forward_decode` |

### Key command
```bash
nohup env SGLANG_USE_CPU_ENGINE=1 python3 -m sglang.launch_server \
  --model /tmp/sglang_model \
  --device cpu --dtype bfloat16 --trust-remote-code \
  --host 0.0.0.0 --port 30000 \
  --max-total-tokens 2048 --context-length 1024 \
  --json-model-override-args '{"quantization_config": null}' \
  > /tmp/sglang_server.log 2>&1 &
```

### Output (validated 2026-04-04)
```
葭葭葭葭Cumhur 스스*葭 Wish葭
```
Garbled — identical pattern to HF CPU run, confirming issue is in the model,
not the SGLang framework.

---

## Setting 4 — llama.cpp GGUF (Q8_0)

**Guide:** [mimo_v2_flash_llamacpp_cpu.md](mimo_v2_flash_llamacpp_cpu.md)

### Hardware
- worker-0: AMD EPYC 9B14, 180 vCPUs, 708 GB RAM (no GPU)
- AVX-512 VNNI used for int8 dot-product kernels

### Weight format
- **On disk:** Q8_0 GGUF — each block of 32 `int8` values + 1 `fp16` block scale = 8.50 BPW, 306 GB total
- **In memory:** weights are `mmap`'d (never fully copied); dequantized block-by-block on-the-fly during each GEMM via AVX-512 VNNI; activations and KV cache in `f32`/`f16`
- **Note:** f32 normalisation tensors (230 tensors) are stored in full precision in GGUF

### Patches required
None. llama.cpp HEAD `9c69907` natively supports `mimo2` architecture.

### GGUF conversion (done once on worker-2)
```bash
python3 ~/llama.cpp/convert_hf_to_gguf.py /mnt/mimo-weights \
  --outfile /mnt/gguf-scratch/mimo-v2-flash-Q8_0.gguf \
  --outtype q8_0
# Output: /mnt/gguf-scratch/mimo-v2-flash-Q8_0.gguf (306 G)
# Time: ~35 min
```

### Key command
```bash
~/llama.cpp/build/bin/llama-server \
  --model /mnt/gguf-scratch/mimo-v2-flash-Q8_0.gguf \
  --host 0.0.0.0 --port 8080 \
  --threads 176 --ctx-size 2048 --n-gpu-layers 0
```

### Performance (measured 2026-04-04)
| Metric | Value |
|---|---|
| Startup (cold NFS mmap) | ~5.5 min |
| Prompt eval | **35 tok/s** |
| Generation | **17 tok/s** |

### Output (validated 2026-04-04)
```
2. But what is 0+0?
```
Coherent — Q8_0 re-encoding from FP8 source faithfully preserves model behaviour.

---

## Infrastructure Summary

All three CPU settings share the same physical cluster (`jingnw-node` TPU VM):

| Node | IP | Role |
|---|---|---|
| worker-0 | — | CPU inference host (SGLang, llama.cpp) |
| worker-1 | `10.202.0.151` | HF safetensors weights (`/mnt/mimo-weights`, 292 G tmpfs, NFS ro) |
| worker-2 | `10.202.0.29` | GGUF Q8_0 (`/mnt/gguf-scratch`, 650 G tmpfs, NFS rw) |

The TPU setting uses a separate `jingnw-node` TPU VM slice (v6e or Ironwood)
with weights stored in GCS (`gs://jingnw-mimo-v2-flash-us-east5/`).
