# MiMo-V2.5-Pro — Bringup Plan

Date: 2026-05-13  
Last updated: 2026-05-19 (Phase 7 added)  
Target hardware: `jingnw-flex-tpu7-8ch` (8 chips, 16 cores, `2x2x2`, Flex Start)

---

## Status summary

| Phase | Description | Status |
|---|---|---|
| 1 | GCS data upload | ✅ Complete |
| 2 | MaxText config (`mimo-v2-5-pro.yml`) | ✅ Complete |
| 3 | Model code adaptation | ✅ Complete (no changes needed) |
| 4 | Checkpoint converter (HF → MaxText FP8 OCDBT) | ✅ Complete |
| 5 | Inference job YAML | ✅ Complete |
| 6 | Smoke test | 🔄 Blocked — requires Phase 7 |
| 7 | Stacked checkpoint (for `scan_layers=true`) | ⏳ Pending |

---

## HBM analysis — TP=8, EP=2 on 8ch pool

Each core has ~80 GB usable HBM. With TP=8, EP=2 (16 JAX devices total):

| Component | Per device |
|---|---|
| MoE expert weights FP8 (192 experts/EP group, TP-sharded) | 62.5 GB |
| Attention BF16 (QKV+O, TP-sharded) | 4.7 GB |
| Embed + lm_head BF16 | 3.7 GB |
| Dense MLP BF16 (layer 0 only) | 0.1 GB |
| **Total weights** | **71.0 GB** |
| Usable HBM | 80.0 GB |
| **Headroom for XLA temps** | **~9 GB** |

KV cache is negligible (~57 MB total): only 8 KV heads, SWA window=128, 10 GA layers.
The 9 GB headroom must be managed with `scan_layers=true` to cap XLA temps to one layer
at a time (see Phase 7).

**Note:** `scan_layers=true` requires a stacked checkpoint. Current checkpoint
(`mimo-v2-5-pro-fp8-ocdbt`) is standard per-layer format → `scan_layers=false`.
Phase 7 produces the stacked checkpoint enabling `scan_layers=true`.

**Fallback if XLA OOMs:** enable FP8-in-HBM mode (`mimo_fp8_weight_mode=block_wise_fp8`,
already enabled in the config). This halves MoE weight footprint from 62.5 GB →
~31 GB/device, opening ~40 GB headroom.

---

## Phase 1 — GCS data upload ✅

HF weights (~962 GiB FP8) uploaded to GCS bucket.

```
gs://jingnw-mimo-v2-5-pro-us-central1/hf-weights/
```

34 safetensors shards + metadata files. Total: 159,581 weight tensors.

---

## Phase 2 — MaxText config ✅

Config: `src/maxtext/configs/models/mimo-v2-5-pro.yml`

Key settings vs V2-Flash:

| Field | V2-Flash | V2.5-Pro |
|---|---|---|
| `base_emb_dim` | 4096 | **6144** |
| `base_num_decoder_layers` | 48 | **70** |
| `base_num_query_heads` | 64 | **128** |
| `base_num_kv_heads` | 4 | **8** |
| `mimo_swa_num_kv_heads` | 8 | **8** (now same as GA) |
| `num_experts` | 256 | **384** |
| `rope_max_timescale` | 5000000 | **10000000** |
| `mimo_attention_value_scale` | 0.707 | **0.612** |
| `mimo_hybrid_layer_pattern` | 48 entries, 9 GA | **70 entries, 10 GA** |
| `mimo_moe_layer_freq` | 48 entries, 47 MoE | **70 entries, 69 MoE** |
| `mimo_fp8_weight_mode` | `""` | **`"block_wise_fp8"`** |
| `scan_layers` | false | **false** (→ true after Phase 7) |

GA layer positions in V2.5-Pro: 0, 7, 15, 23, 31, 39, 47, 55, 62, 69.

Inference precision: **FP8 + dequant-before-matmul** (`block_wise_fp8`). Expert weights
kept as float8_e4m3fn in HBM; per-128×128-block scale_inv tensors applied before each
matmul to produce BF16 for the actual computation.

---

## Phase 3 — Model code adaptation ✅

Investigated all three originally planned changes to `src/maxtext/models/mimo_v2_flash.py`:

1. **Fused QKV weight loading** — No change needed. The checkpoint converter already splits
   fused QKV into separate `query.kernel`, `key.kernel`, `value.kernel` tensors before
   writing to OCDBT. The `Attention` module reads them as separate weights as normal.

2. **Unified KV heads** — No change needed. Existing code:
   `num_kv_heads = cfg.mimo_swa_num_kv_heads if is_swa else cfg.num_kv_heads`
   Both are 8 in V2.5-Pro config, so both branches return the correct value.

3. **70-layer hybrid pattern** — No change needed. Code generically indexes
   `cfg.mimo_hybrid_layer_pattern[layer_idx]` and `cfg.mimo_moe_layer_freq[layer_idx]`.
   Works for any list length.

The model code is fully parameterised by config values. Zero source changes required.

---

## Phase 4 — Checkpoint converter ✅

Converted HF FP8 safetensors → MaxText Orbax zarr2 checkpoint using a 4-node parallel
Kubernetes job on `jingnw-cpu-highmem` (n2-highmem-16).

**Output:** `gs://jingnw-mimo-v2-5-pro-us-central1/mimo-v2-5-pro-fp8-ocdbt/`  
**Arrays:** 1,038 zarr arrays (70 layers × ~15 arrays/layer + global weights)  
**Duration:** ~70 minutes (4 parallel workers covering layer ranges 0–17, 18–35, 36–52, 53–69)

Conversion approach:
- Shard index cache (ranged HTTP reads): skips full 30 GB shard downloads; indexes all
  159,581 keys in ~1 min via safetensors header reads only
- Parallel ranged tensor reads: up to 32 threads, each fetching exact byte range per tensor
- FP8 expert weights stored as float8_e4m3fn + float32 per-block scale_inv tensors
- Parallel zarr writes: up to 8 concurrent GCS uploads per layer
- Layer-resume: restarts pick up from the first unwritten layer (probes `.zarray` marker)

Job YAMLs:
- `tools/orchestration/mimo_v2_5_pro_convert_job.yaml` — 4-worker Indexed Job
- `tools/orchestration/mimo_v2_5_pro_finalize_job.yaml` — writes checkpoint metadata

---

## Phase 5 — Inference job YAML ✅

YAML: `tools/orchestration/mimo_v2_5_pro_inference_job.yaml`

2-pod Indexed Job on `jingnw-flex-tpu7-8ch` (2×2×2, 8 chips, 16 JAX devices).
Headless Service gives pod-0 a stable DNS name for JAX coordinator discovery.

Key inference flags:
```
--model_name mimo-v2-5-pro
--checkpoint_path gs://jingnw-mimo-v2-5-pro-us-central1/mimo-v2-5-pro-fp8-ocdbt/0/items
--ici_tensor_parallelism 8
--ici_expert_parallelism 2
--mimo_fp8_weight_mode block_wise_fp8
--max_prefill 128
--jax_cache_dir gs://jingnw-mimo-v2-5-pro-us-central1/jax-compilation-cache
--max_new_tokens 32
```

Fixes applied during Phase 5 debugging:
- `mimo-v2-5-pro` added to `MaxTextConfig.model_name` Literal allowlist (`src/maxtext/configs/types.py`)
- `scan_layers: true` → `scan_layers: false` in model config (stacked checkpoint required; see Phase 7)
- Checkpoint path corrected (`bf16-ocdbt` → `fp8-ocdbt`)
- Memory request reduced (`512Gi` → `192Gi` to fit node allocatable RAM)

---

## Phase 6 — Smoke test 🔄 Blocked

**Blocker:** `scan_layers=false` causes XLA to JIT-compile all 70 layers as unique
bodies. On this model size, compilation takes >90 min wall clock — longer than the
Flex Start preemption window. Both previous attempts were preempted mid-compilation.

**Root cause:** without `lax.scan`, XLA sees 70 distinct layer programs. With
`scan_layers=true` (single shared body scanned 70× via `lax.scan`), XLA compiles
only one layer → ~5–10 min.

**Resolution:** Phase 7 converts the checkpoint to stacked format, enabling
`scan_layers=true`. Phase 6 resumes after Phase 7 completes.

Steps once unblocked:
1. Single-token decode at batch=1, verify output is coherent
2. Confirm JAX compilation cache written to `gs://…/jax-compilation-cache` for future runs
3. Probe HBM footprint; extend to longer sequences

---

## Phase 7 — Stacked checkpoint ⏳

**Goal:** convert `mimo-v2-5-pro-fp8-ocdbt` (per-layer format) to a stacked layout
compatible with `scan_layers=true`, writing output to:
```
gs://jingnw-mimo-v2-5-pro-us-central1/mimo-v2-5-pro-fp8-ocdbt-stacked/
```

### Why stacking is needed

| Format | Checkpoint key layout | XLA compilation |
|---|---|---|
| Per-layer (`scan_layers=false`) | `decoder/layers/0/attn/q/kernel`, `decoder/layers/1/…` | 70 unique layer bodies → >90 min |
| Stacked (`scan_layers=true`) | `decoder/layers/attn/q/kernel` shape `[70, …]` | 1 shared body via `lax.scan` → ~5–10 min |

### Step 1 — Audit the stacked format (~1 hr, no compute)

Read MaxText source (`models/`, `checkpointing.py`, `nn.scan` usage) to confirm the
exact Zarr key structure expected when `scan_layers=true`:
- Does MaxText expect a single `layers/` group with a leading `[num_layers, …]` axis?
- Or does it split into scan sub-groups (`layers_a/`, `layers_b/`, …) based on a `period`?

This determines the exact stacking logic in Step 2.

### Step 2 — Write stacking script (~2 hr)

New script (similar to Phase 4 converter) that:
1. Walks the per-layer OCDBT checkpoint tree, extracting all unique parameter paths
   with the layer index removed (e.g. `attention/query/kernel`)
2. For each path, loads one copy per layer (layers 0–69) and stacks along axis 0
3. Writes the stacked tensor to the new OCDBT checkpoint

Memory budget per parameter (the binding constraint for expert weights):
- Largest single stack: all 70 copies of one (expert, matrix) pair
  = 70 × 6144 × 2048 × 1 byte (FP8) ≈ 0.9 GB — well within 128 GB node RAM
- Process one parameter path at a time; write before loading the next

### Step 3 — Run on CPU node (~2–4 hr)

Single `jingnw-cpu-highmem` pod (n2-highmem-16, 128 GB RAM). GCS read throughput
will be the bottleneck, same as Phase 4. Parallelisation across layers is not
straightforward here (stacking requires all 70 layers per parameter), so a single
sequential worker is expected.

### Step 4 — Update config and YAML (minutes)

- `src/maxtext/configs/models/mimo-v2-5-pro.yml`: `scan_layers: false` → `scan_layers: true`
- `tools/orchestration/mimo_v2_5_pro_inference_job.yaml`: update checkpoint path
  to `mimo-v2-5-pro-fp8-ocdbt-stacked/0/items`

### Step 5 — Resume Phase 6

With stacked checkpoint + `scan_layers=true`, XLA compilation completes in ~5–10 min.
Flex Start preemption is no longer a risk for the compilation phase.

---

## Risk register

| Risk | Likelihood | Mitigation |
|---|---|---|
| Flex Start preemption during XLA compilation (>90 min without scan) | **High** | Phase 7: stacked checkpoint enables `scan_layers=true`, dropping compile to ~5–10 min |
| Stacked format differs from expected (sub-groups, period-based) | Medium | Step 1 audits MaxText source before writing any conversion code |
| XLA temp buffers exceed ~9 GB headroom → OOM | Medium | `scan_layers=true` caps temps to 1 layer; fall back to 2x2x4 (16 chips) if still OOM |
| Stacking script OOMs on expert weights | Low | Process one (path, layer) at a time; largest in-memory object ~0.9 GB |
| `fused_qkv` split introduces numerical error | Low | Converter validated; splits at offsets [0, nq·dq], [nq·dq, nq·dq+nkv·dk], [nq·dq+nkv·dk, ...] |
| 384 experts with EP=2 → 192 experts/device too large | Low | Covered by FP8 block_wise mode; or increase EP at cost of communication overhead |
| FP8 block scale dequant overhead too slow | Low | Phase A (explicit BF16 dequant) used for prefill; Phase B (Pallas fused kernel) for decode |
