# MiMo-V2.5-Pro — GKE Hardware Allocation

Date: 2026-05-13  
Last updated: 2026-05-19

---

## Current GKE resources (`jingnw-tpu7-cluster`, us-central1-c)

TPU v7x chip spec: **2 cores per chip, 96 GB HBM per core = 192 GB HBM per chip**.
Each core = 1 JAX device. `tpu7x-standard-4t` = 4 chips per node = 8 cores per node = 768 GB HBM per node.

| Node Pool | Machine | Topology | Type | Nodes | Chips | Cores (JAX devs) | HBM Total |
|---|---|---|---|---|---|---|---|
| `default-pool` | e2-standard-4 | — | on-demand | 2 | — | — | — |
| `jingnw-cpu-highmem` | n2-highmem-16 | — | on-demand | **1** (scale back from 4) | — | — | 128 GB RAM |
| `jingnw-flex-tpu7` | tpu7x-standard-4t | **2x2x1** | Flex Start | 0–2 | 4 | 8 | **768 GB** |
| `jingnw-flex-tpu7-8ch` | tpu7x-standard-4t | **2x2x2** | Flex Start | 0–2 | 8 | 16 | **1,536 GB** |

> **jingnw-cpu-highmem** was temporarily scaled to 4 nodes during the parallel checkpoint
> conversion job (2026-05-19). Scale back to 1 node after conversion to avoid cost:
> ```bash
> gcloud container clusters resize jingnw-tpu7-cluster \
>     --zone=us-central1-c --node-pool=jingnw-cpu-highmem --num-nodes=1 --quiet
> ```

Both Flex Start TPU pools autoscale to 0 when idle.

---

## Why the current hardware is insufficient for V2.5-Pro

Each core has ~80 GB usable HBM after system/XLA overhead (measured 72.3 GB at steady-state
on V2-Flash). The V2-Flash FP8 reference validates the per-core estimate: 309 GB / 8 cores
(2x2x1) = 38.6 GB/core ≈ measured ~40 GB/core ✓.

| | MiMo-V2-Flash | MiMo-V2.5-Pro |
|---|---|---|
| FP8 model size | ~309 GB | **~1,042 GB** |
| Chips needed (FP8) | 4 chips ✓ | **≥ 12 chips** |
| `jingnw-flex-tpu7` (4 chips, 640 GB usable) | ✓ fits | ✗ OOM |
| `jingnw-flex-tpu7-8ch` (8 chips, 1,280 GB usable) | ✓ fits | ⚠ weights fit, only ~15 GB/core headroom |

Breakdown of V2.5-Pro FP8 weight estimate:

| Component | Size |
|---|---|
| Expert weights FP8 (69 MoE layers × 384 experts × 3 matrices × 6144 × 2048) | ~1,000 GB |
| Non-expert BF16 (attn QKV+O, embed, lm_head, norms) | ~42 GB |
| Scale_inv float32 (per 128×128 block) | ~0.24 GB |
| **Total** | **~1,042 GB** (HF reports 962.4 GiB = ~1,034 GB ✓) |

Chip count vs HBM fit (80 GB usable per core, 2 cores per chip):

| Chips | Cores | Usable HBM | FP8/core | Headroom/core | Verdict |
|---|---|---|---|---|---|
| 4 | 8 | 640 GB | 130 GB | — | ✗ OOM |
| 8 | 16 | 1,280 GB | 65 GB | ~15 GB | ⚠ weights fit, tight — attempt with `scan_layers=true` |
| **12** | **24** | **1,920 GB** | **43 GB** | **~37 GB** | **✓ comfortable** |
| 16 | 32 | 2,560 GB | 33 GB | ~47 GB | ✓ comfortable |
| 24 | 48 | 3,840 GB | 22 GB | ~58 GB | ✓ |
| 32 | 64 | 5,120 GB | 16 GB | ~64 GB | ✓ |

The existing `jingnw-flex-tpu7-8ch` (8 chips) is the **first bringup target**. With
`scan_layers=true` and `mimo_fp8_weight_mode=block_wise_fp8`, the ~15 GB/core headroom
may be sufficient for batch=1 smoke testing. If XLA OOMs, escalate to 16 chips (2x2x4).

Valid multi-host topologies for `tpu7x-standard-4t` follow the progression `2x2x2` (8) →
`2x2x4` (16) → `4x4x2` (32). There is no clean 12-chip option, so **16 chips (`2x2x4`)** is
the recommended minimum for production inference.

---

## CPU node — conversion performance

`jingnw-cpu-highmem` (n2-highmem-16, 128 GB RAM, 16 vCPUs) handled V2.5-Pro checkpoint
conversion by streaming shards via ranged HTTP reads (no full shard download to disk).

Actual conversion stats (2026-05-19):
- **4 nodes** used in parallel, each running one worker pod (14 CPU / 110 Gi requested)
- **~70 minutes** total for all 70 layers
- **~200s/layer** per worker (MoE layers); layer 0 dense ~9s
- GCS throughput was the bottleneck, not CPU

For future PTQ quantization (if needed): a step that holds full model state simultaneously
in CPU RAM would OOM on 128 GB with a ~1 TB model. Upgrade to `n2-highmem-96` (768 GB) if
that becomes necessary.

---

## Summary of pool assignments

| Pool | Topology | Chips | Cores | Purpose |
|---|---|---|---|---|
| `jingnw-flex-tpu7` | 2x2x1 | 4 | 8 | V2-Flash single-host inference |
| `jingnw-flex-tpu7-8ch` | 2x2x2 | 8 | 16 | **V2.5-Pro bringup inference (Phase 5–6)** |
| `jingnw-flex-tpu7-16ch` *(new, if needed)* | 2x2x4 | 16 | 32 | V2.5-Pro production inference |
| `jingnw-cpu-highmem` | n2-highmem-16 | — | — | HF→GCS upload, checkpoint conversion |
