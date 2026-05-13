# MiMo-V2.5-Pro — GKE Hardware Allocation

Date: 2026-05-13

---

## Current GKE resources (`jingnw-tpu7-cluster`, us-central1-c)

TPU v7x chip spec: **2 cores per chip, 96 GB HBM per core = 192 GB HBM per chip**.
Each core = 1 JAX device. `tpu7x-standard-4t` = 4 chips per node = 8 cores per node = 768 GB HBM per node.

| Node Pool | Machine | Topology | Type | Max Nodes | Chips | Cores (JAX devs) | HBM Total |
|---|---|---|---|---|---|---|---|
| `default-pool` | e2-standard-4 | — | on-demand | 2 | — | — | — |
| `jingnw-cpu-highmem` | n2-highmem-16 | — | on-demand | 1 | — | — | 128 GB RAM |
| `jingnw-flex-tpu7` | tpu7x-standard-4t | **2x2x1** | Flex Start | max 2 | 4 (1 node) | 8 | **768 GB** |
| `jingnw-flex-tpu7-8ch` | tpu7x-standard-4t | **2x2x2** | Flex Start | max 2 | 8 (2 nodes) | 16 | **1,536 GB** |

Both Flex Start TPU pools autoscale to 0 when idle.
Currently running VMs (no TPU nodes active as of 2026-05-13):

- `gke-...-jingnw-cpu-highmem-...` — n2-highmem-16 (128 GB RAM), RUNNING
- `gke-...-default-pool-...` × 2 — e2-standard-4, RUNNING (GKE control plane)
- `jingnw-tpu-op` — e2-small, us-east5-b, RUNNING

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
| 8 | 16 | 1,280 GB | 65 GB | ~15 GB | ⚠ weights fit, too tight for KV cache + activations |
| **12** | **24** | **1,920 GB** | **43 GB** | **~37 GB** | **✓ comfortable** |
| 16 | 32 | 2,560 GB | 33 GB | ~47 GB | ✓ comfortable |
| 24 | 48 | 3,840 GB | 22 GB | ~58 GB | ✓ |
| 32 | 64 | 5,120 GB | 16 GB | ~64 GB | ✓ |

The existing `jingnw-flex-tpu7-8ch` (8 chips) fits the FP8 weights on paper, but leaves only
~15 GB/core for KV cache, activations, and XLA temporaries — too tight for reliable inference,
especially at longer sequence lengths. 12 chips is the practical minimum.

Valid multi-host topologies for `tpu7x-standard-4t` follow the progression `2x2x2` (8) →
`2x2x4` (16) → `4x4x2` (32). There is no clean 12-chip option, so **16 chips (`2x2x4`)** is
the recommended minimum.

---

## CPU node caveat

`jingnw-cpu-highmem` (n2-highmem-16, 128 GB RAM) handled V2-Flash checkpoint conversion by
streaming shards one at a time. For V2.5-Pro, shard-by-shard streaming should still work for
the HF→MaxText converter. However, if a PTQ quantization step needs to hold model state
simultaneously in CPU RAM (as happened with V2-Flash PTQ), 128 GB will OOM on a ~1 TB model.
Monitor the converter job; upgrade to `n2-highmem-96` (768 GB) if needed.

---

## Summary of pool assignments

| Pool | Topology | Chips | Cores | Purpose |
|---|---|---|---|---|
| `jingnw-flex-tpu7` | 2x2x1 | 4 | 8 | V2-Flash single-host inference / converter jobs |
| `jingnw-flex-tpu7-8ch` | 2x2x2 | 8 | 16 | V2-Flash multi-host inference / PTQ quantization |
| `jingnw-flex-tpu7-16ch` *(new)* | 2x2x4 | 16 | 32 | **V2.5-Pro bringup inference** |
| `jingnw-cpu-highmem` | n2-highmem-16 | — | — | HF→GCS upload, checkpoint conversion (CPU) |
