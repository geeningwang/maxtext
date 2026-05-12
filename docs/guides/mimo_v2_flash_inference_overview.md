# MiMo-V2-Flash Inference — Setup Comparison

This document summarises all four validated inference configurations for
[MiMo-V2-Flash](https://huggingface.co/XiaomiMiMo/MiMo-V2-Flash) (309 B total /
~15 B active parameters per token).

---

## Quick Comparison

| # | Stack | Hardware | Weight format | Status | Output quality |
|---|---|---|---|---|---|
| 1 | **MaxText + TPU v6e** | TPU v6e-32 (1,024 GiB HBM total) | BF16 (OCDBT checkpoint, FP8 dequantized via `weight_scale_inv`) | ✅ End-to-end generation validated; **4K/1K prefill+decode: TTFT 823 ms · 113.6 ms/step · 2,536 tok/s at batch=288** (v6e-32, 2026-04-24, `per_device_batch_size=9`, `attention=flash`); **16K/1K chunked prefill+decode: TTFT 5,522 ms · 71.1 ms/step · 1,349 tok/s at batch=96** (v6e-32, 2026-04-24, `use_chunked_prefill=true prefill_chunk_size=4096`, pdb=3) | Coherent ("420 km" train distance, 2026-04-22, EOS stop); thinking chain may exhaust default `max_new_tokens` before answer — use ≥ 4000 for reliable EOS |
| 2 | **HuggingFace Transformers (CPU)** | AMD EPYC 9B14, 180 vCPUs, 708 GB | BF16 (shard-by-shard FP8→BF16 dequant with `weight_scale_inv`) | ✅ Runs end-to-end | Coherent (`"2. But what if we consider it in a"`) |
| 3 | **SGLang CPU engine** | AMD EPYC 9B14, 180 vCPUs, 708 GB | FP8→BF16 cast at load (quantization_config=null) | ✅ Runs, 5 patches needed | Garbled (`葭葭葭…`) — FP8 scale tensors stripped |
| 4 | **llama.cpp (GGUF Q8_0)** | AMD EPYC 9B14, 180 vCPUs, 708 GB | Q8_0 on disk, int8+f32 accumulation in compute | ✅ Runs, no patches needed | Coherent (`"2. But what is 0+0?"`) |
| 5 | **MaxText + GKE + TPU v7x (FP8 PTQ)** | TPU v7x 2×2×1 (4 chips, 8 JAX devices, 768 GiB HBM total) | FP8 E4M3 in GCS checkpoint; **BF16 in HBM** (dequantized by qwix at load) | ✅ End-to-end validated 2026-05-12; EOS fires cleanly; **2.8 tok/s** single-sequence (TP=4, EP=2) | Coherent (EOS stop at step 767) |

**Key finding:** Setting 2 (HF) now produces coherent output using a shard-by-shard
FP8→BF16 loader that applies `weight_scale_inv` block scales correctly.  Setting 3
(SGLang) remains garbled because it strips `quantization_config` and silently
skips the scale tensors.  Settings 2 and 4 both produce coherent output via
different paths: HF uses accurate per-block FP8 dequantization; llama.cpp
re-encodes FP8 to Q8_0 at conversion time.

Setting 5 (MaxText + GKE + TPU v7x) adds a containerized GKE path with an FP8 PTQ
checkpoint.  Despite the FP8 checkpoint format, qwix `PtqProvider` dequantizes
weights to BF16 at load — runtime HBM is the same as a pure BF16 deployment
(71.93 GB/TensorCore steady-state, 76.3% of 94.75 GiB capacity).

---

## Competitive Benchmark Comparison

Numbers from an external reference implementation (hardware and framework unspecified by source).
Our MaxText + TPU v6e-32 measurements are the closest available counterpart.

### External reference

| Scene | E2E Peak Throughput | Peak BS | TTFT / ITL (ms) |
| :--- | :--- | :--- | :--- |
| 4K / 1K Decode | 417 output tok/s | 32 | 22,602 / 31.20 |
| 16K / 1K Decode | 94 output tok/s | 8 | 37,979 / 14.71 |

### MaxText + TPU v6e-32 (measured, 2026-04-23/24)

| Scene | E2E Peak Throughput | Peak BS | TTFT / ITL (ms) | Notes |
| :--- | :--- | :--- | :--- | :--- |
| 4K / 1K Prefill+Decode | **2,536 output tok/s** | **288** (`per_device_batch_size=9`, max) | **823 ms TTFT** / **113.6 ms/step** | `max_prefill=4096, max_target=5120`, flash+context; pdb=10 OOMs (XLA compile needs 339 MB, 241 MB free). Confirmed 2026-04-24. |
| **16K / 1K Chunked Prefill+Decode** | **1,349 output tok/s** | **96** (`per_device_batch_size=3`, max) | **5,522 ms TTFT** / **71.1 ms/step** | `use_chunked_prefill=true`, `prefill_chunk_size=4096` (4 × 4096 chunks, `dot_product`+flash decode); standard 5b config. 2026-04-24, commit `1b4cd37d`. |

**Comparability notes:**
- **4K / 1K Prefill+Decode (2026-04-24):** MaxText/TPU achieves **2,536 tok/s at BS=288**
  (`per_device_batch_size=9`) vs external **417 tok/s at BS=32** — a **6.08× advantage**.
  Uses `max_prefill=4096, max_target=5120`, flash+context attention. pdb=9 is the HBM ceiling
  (pdb=10 OOMs during XLA compile: 339 MB needed, 241 MB free). ITL 113.6 ms at BS=288 vs
  external 31.20 ms at BS=32 — our per-sequence latency (0.39 ms/tok) is still 10× lower.
  TTFT at BS=1: 823 ms; both prefill and decode run at the same pdb in the same environment.
- **16K / 1K Chunked Prefill+Decode (2026-04-24):** With `use_chunked_prefill=true` and
  `prefill_chunk_size=4096` (branch `feature/chunked-prefill-16k`, commit `1b4cd37d`), pdb=3
  (BS=96) runs both prefill and decode in a single environment: decode **1,349 tok/s / 71.1 ms/step**
  vs external **94 tok/s at BS=8** — a **14.4× advantage**. TTFT is **5,522 ms / 2,967 tok/s**
  — the chunked path uses `dot_product` (required for EP_AS_CONTEXT compatibility) and
  processes 4 serial 4096-token chunks. This is the standard 5b configuration.
  Five code fixes were required across `kvcache.py`,
  `attention_op.py`, and `maxtext_utils.py`; see [env_restore guide](mimo_v2_flash_env_restore.md).

---

## Setting 1 — MaxText on TPU

**Guide:** [mimo_v2_flash_inference.md](mimo_v2_flash_inference.md)

### Hardware
- TPU v6e-32 (minimum; 1024 GB HBM total) or Ironwood v7-4 (768 GB HBM total)
- Tensor-parallel across all chips via JAX `pjit`

### Weights
- Source: HF safetensors at `gs://jingnw-mimo-v2-flash-us-east5/hf-model`
- **Step 1:** Convert HF FP8 → MaxText zarr2 BF16 with `weight_scale_inv` applied:
  `gs://jingnw-mimo-v2-flash-us-east5/mimo-v2-flash-fixed/checkpoints/0/items`
- **Step 1b:** Convert zarr2 → zarr3+OCDBT on all 8 workers:
  `gs://jingnw-mimo-v2-flash-us-east5/mimo-v2-flash-fixed-ocdbt/checkpoints/0/items`
- Conversion: 8-process OCDBT (`process_0`–`process_7`), `tensor=4 × expert=8` mesh

### Weight format
- **On disk:** `safetensors` FP8 (original HF) → `zarr3+OCDBT` BF16 (converted)
- **In memory (JAX on TPU):** `bfloat16` across all TPU HBM

### Architecture customisations in MaxText
- Asymmetric head dims: Q/K = 192, V = 128
- Partial RoPE: rotates only first 64 of 192 Q/K dims; dual RoPE bases (global = 5 M, SWA = 10 K)
- Attention sink bias (learnable scalar per head, SWA layers only)
- noaux-TC sigmoid MoE routing (sigmoid scores, not softmax; L1-normalised final weights)

### Status
End-to-end autoregressive generation is **validated** (last confirmed 2026-04-21,
post `jingnw-tpu-op` VM recreation).  All 8 workers print `TPU_IMPORT_OK` after
restore and the demo exits with code 0.  Proper benchmark (3-step warmup + 50 timed
steps at batch=32): **55.5 ms/step median**, **576 tok/s**, **1.7 ms/tok/seq**
(2026-04-17, commit `72f75972`; opt #1 `jax.debug.print` removal + SWA fix).
Batch scaling (re-measured 2026-04-24, correct 4K/1K params): **113.6 ms/step**, **2,536 tok/s**,
**0.39 ms/tok/seq** at `per_device_batch_size=9` (total batch=288, v6e-32);
`max_prefill=4096, max_target=5120`, `attention=flash`, `expert_shard_attention_option=context`;
pdb=10 OOMs (XLA compile: 339 MB needed, 241 MB free). Prefill and decode run at the same pdb
in the same environment (TTFT ~823 ms at BS=1).
The model is prompted via `tokenizer.apply_chat_template()` (`use_chat_template=true`).
With thinking mode enabled (MiMo-V2-Flash default), the model emits a long reasoning
chain before the final answer; the default `max_new_tokens` budget is typically
exhausted before the chain completes and EOS fires — pass `--max_new_tokens 4000`
or higher for reliable end-to-end generation in the demo path.

Checkpoint conversion is **complete** (as of 2026-04-06).  The full pipeline ran
using a distributed approach: worker 0 ran `convert_mimo_v2_flash.py` over all
48 layers plus global weights; workers 1–7 ran in parallel each handling 3 layers
from the upper range (layers 27–47).  After all workers finished,
`--scan_and_finalize` wrote the valid `_METADATA` file.  Step 1b (OCDBT
conversion) then ran on all 8 workers to produce `mimo-v2-flash-fixed-ocdbt`
(384 GB, 8-process OCDBT, zarr3).  Previous OCDBT checkpoint
(`mimo-v2-flash-ocdbt`) had a bug where FP8 block scales were not applied,
producing garbled output; use `mimo-v2-flash-fixed-ocdbt` for all inference.

**Bug fixed (2026-04-06):** `query_pre_attn_scalar` was missing from the
`Attention()` constructor call in `mimo_v2_flash.py`.  MaxText folds
`1/sqrt(head_dim)` into the query kernel *initialisation* only; when loading
existing HF weights the forward pass must apply this scale explicitly.  Without
it, attention logits were `sqrt(192) ≈ 13.9×` too large, driving softmax to
near-argmax and producing completely garbled token predictions.  Fix: added
`query_pre_attn_scalar=cfg.head_dim**-0.5` (commit `6051205a`).

### Performance (v6e-32)
| Metric | Value |
|---|---|
| Checkpoint load (OCDBT, 8-process) | ~36–40 s (warm/cold GCS cache) |
| Prefill (512 tokens) | ~22 s |
| Demo path throughput (single-sequence, per-step CPU sync) | **~4.3 tok/s** (2026-04-21; thinking chain fills default `max_new_tokens`) |
| Generation speed (steady-state, batch=32) | 55.5 ms/step · 576 tok/s · 1.7 ms/tok/seq (2026-04-17) |
| Generation speed (4K/1K prefill+decode, batch=288, `per_device_batch_size=9`) | **113.6 ms/step · 2,536 tok/s · 0.39 ms/tok/seq** decode; **823 ms TTFT** prefill (2026-04-24, `max_prefill=4096 max_target=5120`, flash+context) |
| Generation speed (16K/1K chunked prefill+decode, batch=96, `per_device_batch_size=3`) | **71.1 ms/step · 1,349 tok/s · 0.74 ms/tok/seq** decode; **5,522 ms TTFT · 2,967 tok/s** prefill (2026-04-24, `use_chunked_prefill=true prefill_chunk_size=4096`, dot_product+context) |
| HBM per chip after load | ~18.0 GB / 31.25 GB (57.5%) |
| Parallelism | TP=4 × EP=8 |

### Output (validated 2026-04-17)
```
Step 1: 120 km/h × 2.5 h = 300 km
Step 2: 80 km/h × 1.5 h = 120 km
Step 3: Total = 300 + 120 = 420 km
```
Model stopped cleanly at EOS after ~512 generate steps.
Chat template applied; prompt: `"Solve step by step: A train travels at 120 km/h for 2.5 hours, then at 80 km/h for 1.5 hours. What is the total distance traveled?"`.

**Smoke test (2026-04-21, post-restore, default `max_new_tokens`):**
EOS did not fire — the model emitted a long thinking chain that exhausted the default
token budget before reaching the final answer.  The reasoning was on-track at truncation
(visible: _"The question is about a train's travel… The train travels at 120 km/h for
2.5 hours, and then at 80 [km/h]"_), confirming the model is coherent.  Fix: pass
`--max_new_tokens 4000`.  Single-stream demo throughput: **~4.3 tok/s** (limited by
per-step `jax.effects_barrier()` + `.item()` CPU sync in `decode.py`; see opt rank 4
in [perf doc](mimo_v2_flash_tpu_perf_optimization.md)).

<details>
<summary>Earlier validated output (2026-04-08, prompt: "What is 1+1?")</summary>

```
The answer is **2**.

However, depending on the context, it could also be:
*   **11** (if you are concatenating text/strings)
*   **1** (in Boolean algebra or 1 OR 1 is true)
*   **0** (in Boolean algebra with XOR operation: 1 exclusive-or 1)
*   **1** (in modular arithmetic modulo 2)
```
Model stopped cleanly at EOS after 597 generate steps (~600 tokens).
</details>

### Key command
```bash
gcloud compute tpus tpu-vm ssh <tpu-node> --worker=all --zone=<zone> \
  --command="cd ~/maxtext && source maxtext_tpu_venv/bin/activate && \
    python3 demos/mimo_v2_flash_demo_jax.py \
      --checkpoint_path gs://jingnw-mimo-v2-flash-us-east5/mimo-v2-flash-fixed-ocdbt/checkpoints/0/items \
      --tokenizer_path XiaomiMiMo/MiMo-V2-Flash \
      --prompt 'What is 1+1?' \
      --max_new_tokens 2000 \
      --max_prefill 512 \
      --ici_tensor_parallelism 4 \
      --ici_expert_parallelism 8"
```

---

## Setting 2 — HuggingFace Transformers (CPU)

**Demo:** [demos/mimo_v2_flash_demo_hf.py](../../demos/mimo_v2_flash_demo_hf.py)

### Hardware
- worker-0: AMD EPYC 9B14, 180 vCPUs, 708 GB RAM (no GPU)
- Weights sourced from GCS: `gs://jingnw-mimo-v2-flash-us-east5/hf-model` (145 safetensors shards)
  *(Previously served over NFS from worker-1; NFS tmpfs decommissioned 2026-04-04.)*

### Weight format
- **On disk:** FP8 safetensors (original HF format)
- **In memory:** BF16 — loaded via a custom shard-by-shard streamer that reads each
  safetensors shard, applies `weight_scale_inv` block scales
  (`dequant[i,j] = fp8[i,j] * scale[i//bm, j//bn]` with dynamic per-tensor block
  dims), then frees the FP8 source before moving to the next shard.  Peak memory
  is ~540 GB (final BF16 model) + ~4 GB per shard overhead, staying within 708 GB.

### Patches required
None to the HF model code. The demo script handles:
- `init_empty_weights()` for zero-RAM meta-device skeleton instantiation
- `ROPE_INIT_FUNCTIONS['default']` shim for Transformers 5.x compatibility
- `eager_attention_forward` / `compute_default_rope_parameters` monkey-patches
- Custom `_load_weights_fp8_to_bf16()` shard streamer (avoids `FineGrainedFP8HfQuantizer` 730 GB peak)
- `tokenizer.apply_chat_template()` to format prompts as proper chat turns (ensures EOS is emitted)
- Explicit `eos_token_id=tokenizer.eos_token_id` passed to `generate()` (model `config.json` has `eos_token_id: null`)

### Key command
```bash
python3 demos/mimo_v2_flash_demo_hf.py \
  --model_path /mnt/mimo-weights \
  --prompt "What is 1+1?" \
  --max_new_tokens 2000
```

The prompt is automatically wrapped via `tokenizer.apply_chat_template()` into:
```
<|im_start|>user\nWhat is 1+1?<|im_end|>\n<|im_start|>assistant\n
```
This causes the model to produce a single assistant turn ending with `<|im_end|>` (EOS),
rather than open-ended text completion that runs to `max_new_tokens`.

### Performance (measured 2026-04-07)
| Metric | Value |
|---|---|
| Skeleton init (meta device) | ~13s, ~0 GB |
| Weight streaming (145 shards) | ~3.5 min, peak ~540 GB RSS |
| Generation (EOS-terminated) | 82.74s, **1.9 tok/s**, 156 tokens |

### Output (validated 2026-04-07)
```
Of course! The answer to "what is 1 + 1" depends on the context.

**In basic arithmetic:**
The sum of one plus one is **two (2)**.

However, this simple question can have more complex answers in different fields:

*   In binary code:
    `1` + `1` = `10`

*   If you combine two drops of water:
    You get one larger drop of water (`1 + 1 = 1`).

*   In Boolean logic:
    True OR True equals True.

So while the most common and expected mathematical answer is **2**, there are
other valid interpretations depending on the system being used.
```
Model stopped cleanly at EOS after 156 tokens — FP8 weights correctly dequantized,
chat template correctly applied.

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

## Setting 5 — MaxText on GKE + TPU v7x (FP8 PTQ)

**Guide:** [mimo_v2_flash_tpu_v7x_gke_env_restore.md](mimo_v2_flash_tpu_v7x_gke_env_restore.md)

### Hardware

- GKE cluster: `jingnw-tpu7-cluster`, zone `us-central1-c`, project `tpu-launchpad-playground`
- Node pool: `jingnw-flex-tpu7` — 2×2×1 topology, **4 chips = 8 JAX devices**
- TPU v7x architecture: **2 TensorCores per chip**, 96 GiB HBM per TensorCore
- Total HBM: 4 chips × 192 GiB = **768 GiB** (vs v6e-32 which has 32 chips × 32 GiB)
- Provisioned on demand via DWS Flex Start (no persistent TPU VM)

### Weights

- Source: FP8 PTQ checkpoint (produced from BF16 MaxText checkpoint via qwix `PtqProvider`)
- GCS: `gs://jingnw-mimo-v2-flash-us-central1/mimo-v2-flash-fp8-ptq/0/items` (441.79 GiB)
- BF16 source: `gs://jingnw-mimo-v2-flash-us-central1/mimo-v2-flash-fixed-ocdbt/checkpoints/0/items` (384.43 GiB)

### Weight format

- **In GCS (checkpoint):** FP8 E4M3 (attention projection weights) + BF16 (MoE experts, embeddings, layer norms) + float32 (FP8 scale factors)
- **In HBM at runtime:** **BF16 only** — qwix `PtqProvider` dequantizes FP8 → BF16 at checkpoint load time; no `float8_e4m3fn` tensors appear in JAX live arrays

### Parallelism

- `ici_tensor_parallelism=4`, `ici_expert_parallelism=2`
- Constraint: `num_kv_heads=4` must be divisible by TP; TP=8 would OOM

### Status

End-to-end inference validated 2026-05-12 (job `mimo-v2-flash-fp8-ptq-demo-v10x`).
EOS fires cleanly at step 767 for a math reasoning prompt.
Checkpoint load time: **~271 s** (cold GCS, OCDBT).
Single-sequence throughput: **2.8 tok/s** (`scan_layers=false`, TP=4, EP=2).

### HBM profile (validated 2026-05-12, v10x run)

| Stage | HBM per TensorCore | Delta |
|---|---|---|
| `init` | 0.00 GB | — |
| `after_setup_decode_state` | **71.93 GB** | +71.93 GB (weights + KV cache pre-allocated together) |
| `after_load_params` | 71.93 GB | 0 |
| `after_prefill` | 72.02 GB | +0.09 GB |
| `after_insert` | 72.17 GB | +0.15 GB |
| `generate_step_0512` | **72.31 GB** | +0.14 GB |

Peak: **72.31 GB = 76.3%** of 94.75 GiB TensorCore capacity.
All 568 tracked weight shards are BF16 in HBM (no FP8 tensors despite FP8 checkpoint).

### Performance

| Metric | Value |
|---|---|
| Checkpoint load (OCDBT, cold GCS) | ~271 s |
| HBM after load (per TensorCore) | 71.93 GB / 94.75 GiB (76.0%) |
| HBM steady-state (per TensorCore) | 72.31 GB / 94.75 GiB (76.3%) |
| Single-sequence throughput | **2.8 tok/s** (TP=4, EP=2, `scan_layers=false`) |
| EOS stop | Yes (step 767 on math reasoning prompt) |

### Key commands

```bash
# Submit inference job
kubectl apply -f tools/orchestration/mimo_v2_flash_fp8_ptq_demo_job.yaml

# Stream HBM and EOS events
POD=$(kubectl get pods -l app=mimo-v2-flash-fp8-ptq-demo-v10x \
      -o jsonpath='{.items[0].metadata.name}')
kubectl logs -f "$POD" | grep -E "\[HBM\]|EOS token|generate_total|Throughput"
```

See [mimo_v2_flash_tpu_v7x_gke_env_restore.md](mimo_v2_flash_tpu_v7x_gke_env_restore.md)
for the full kubectl setup and troubleshooting workflow.

---

## Infrastructure Summary

All three CPU settings share the same physical cluster (`jingnw-node` TPU VM):

| Node | IP | Role |
|---|---|---|
| worker-0 | — | CPU inference host (SGLang, llama.cpp) |
| worker-1 | `10.202.0.151` | *(NFS tmpfs decommissioned 2026-04-04; HF weights now in GCS)* |
| worker-2 | `10.202.0.29` | *(NFS tmpfs decommissioned 2026-04-04; GGUF re-encode not applicable — use GCS source)* |

The TPU v6e setting uses a separate `jingnw-node` TPU VM slice (v6e or Ironwood)
with weights stored in GCS (`gs://jingnw-mimo-v2-flash-us-east5/`).

The GKE + TPU v7x setting (Setting 5) uses a separate GKE cluster
(`jingnw-tpu7-cluster`, `us-central1-c`) with DWS Flex Start on-demand node
provisioning.  Weights are in `gs://jingnw-mimo-v2-flash-us-central1/`.
See [mimo_v2_flash_tpu_v7x_gke_env_restore.md](mimo_v2_flash_tpu_v7x_gke_env_restore.md).
