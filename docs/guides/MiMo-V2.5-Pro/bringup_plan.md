# MiMo-V2.5-Pro — Bringup Plan

Date: 2026-05-13  
Target hardware: `jingnw-flex-tpu7-8ch` (8 chips, 16 cores, `2x2x2`, Flex Start)

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
at a time.

**Fallback if XLA OOMs:** enable FP8-in-HBM mode (`mimo_fp8_weight_mode=block_wise_fp8`,
already implemented from V2-Flash Phase A). This halves MoE weight footprint from 62.5 GB →
~31 GB/device, opening ~40 GB headroom.

---

## Phase 1 — GCS data upload

Upload HF weights (~962 GiB FP8) to a new GCS bucket. The existing streaming script
requires no local disk.

```bash
gsutil mb -l us-central1 gs://jingnw-mimo-v2-5-pro-us-central1

python3 tools/dev/upload_mimo_hf_to_gcs.py \
    --bucket jingnw-mimo-v2-5-pro-us-central1 \
    --gcs_prefix hf-weights \
    --repo_id XiaomiMiMo/MiMo-V2.5-Pro \
    --skip_existing
```

Target: `gs://jingnw-mimo-v2-5-pro-us-central1/hf-weights/`

---

## Phase 2 — MaxText config (`mimo-v2-5-pro.yml`)

New config mirroring `src/maxtext/configs/models/mimo-v2-flash.yml` with updated values:

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

GA layer positions in V2.5-Pro: 0, 7, 15, 23, 31, 39, 47, 55, 62, 69.

---

## Phase 3 — Model code adaptation

Three changes needed in `src/maxtext/models/mimo_v2_flash.py`:

1. **`fused_qkv` weight loading** — HF stores a single `[hidden, Hq·dq + Hkv·dk + Hkv·dv]`
   tensor per layer (`attention_projection_layout: "fused_qkv"`). The checkpoint converter
   must split it into separate Q/K/V tensors before writing to OCDBT.

2. **Unified KV heads** — V2-Flash had asymmetric GA(4)/SWA(8) KV heads requiring
   special-casing. V2.5-Pro uses `num_kv_heads=8` for both; remove the asymmetric path.

3. **70-layer hybrid pattern** — verify the new 10-GA / 60-SWA pattern is handled correctly
   by the layer-type dispatch logic.

---

## Phase 4 — Checkpoint converter

Extend `src/maxtext/checkpoint_conversion/standalone_scripts/convert_mimo_v2_flash.py`
to handle the `fused_qkv` layout:

- Read each layer's fused QKV tensor from the HF safetensors shard
- Split at offsets `[0, nq·dq]`, `[nq·dq, nq·dq + nkv·dk]`, `[nq·dq + nkv·dk, ...]`
  → separate `q_proj`, `k_proj`, `v_proj` weight tensors
- Write to MaxText OCDBT format (rest of conversion logic unchanged)

Run on `jingnw-cpu-highmem` (n2-highmem-16, streams shard-by-shard, no full-model RAM needed).

Output: `gs://jingnw-mimo-v2-5-pro-us-central1/mimo-v2-5-pro-bf16-ocdbt/`

---

## Phase 5 — Inference job YAML

New GKE job targeting `jingnw-flex-tpu7-8ch`:

```yaml
nodeSelector:
  cloud.google.com/gke-nodepool: jingnw-flex-tpu7-8ch
  cloud.google.com/gke-tpu-accelerator: tpu7x
  cloud.google.com/gke-tpu-topology: 2x2x2
resources:
  requests:
    google.com/tpu: "8"   # 8 chips = 16 JAX devices
  limits:
    google.com/tpu: "8"
```

Key inference flags:
```
--ici_tensor_parallelism 8
--ici_expert_parallelism 2
--scan_layers true          # critical: caps XLA temp to 1 layer at a time
--max_target_length 8192
--per_device_batch_size 1   # start small
```

---

## Phase 6 — Smoke test

1. Single-token decode at batch=1, verify output is coherent
2. Probe HBM via `jax.live_arrays()` to confirm actual per-device footprint
3. Extend to longer sequences once baseline is confirmed

---

## Risk register

| Risk | Likelihood | Mitigation |
|---|---|---|
| XLA temp buffers exceed ~9 GB headroom → OOM | Medium | `scan_layers=true`; fall back to FP8-in-HBM (`block_wise_fp8`) if still OOM |
| `fused_qkv` split introduces numerical error | Low | Validate converted checkpoint against HF output on a single layer before full run |
| 384 experts with EP=2 → 192 experts/device too large | Low | Covered by FP8 weight mode; or increase EP at cost of communication overhead |
| `n2-highmem-16` (128 GB) OOMs during conversion | Low | Converter streams shard-by-shard; upgrade to `n2-highmem-96` if full-state PTQ needed |
