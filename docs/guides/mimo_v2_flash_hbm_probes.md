# MaxText HBM Probe Points

`_probe_hbm` / `_probe_hbm_arrays` calls in the MaxText/TPU inference pipeline.
**These exist only in the TPU path — the HF Transformers CPU path has no probe points.**

---

## Probe locations

| # | File | Label | `detail=True` |
|---|---|---|---|
| 1 | `src/maxtext/inference/decode.py` | `init` | ✅ |
| 2 | `src/maxtext/inference/decode.py` | `after_load_params` | ✅ |
| 3 | `src/maxtext/inference/decode.py` | `after_prefill` | ✅ |
| 4 | `src/maxtext/inference/decode.py` | `after_insert` | ✅ |
| 5 | `src/maxtext/inference/decode.py` | `generate_step_{i:04d}` (first step only) | ✅ first; ❌ subsequent |
| 6 | `src/maxtext/inference/maxengine/maxengine.py` | `before_setup_decode_state` | ❌ |
| 7 | `src/maxtext/inference/maxengine/maxengine.py` | `after_setup_decode_state` | ❌ |

Probe 5 fires every 50 steps (`i == steps[0]` or `(i - steps[0]) % 50 == 0`).
Total probe calls ≈ `7 + ceil(max_new_tokens / 50)`.

---

## `_probe_hbm` — per-device summary

```python
def _probe_hbm(label: str, detail: bool = False) -> None:
    for d in jax.local_devices():
        stats = d.memory_stats()
        used_gb  = stats["bytes_in_use"]   / 2**30
        limit_gb = stats["bytes_limit"]    / 2**30
        peak_gb  = stats["peak_bytes_in_use"] / 2**30
        print(f"[HBM] {label} host={host} dev={d.id}"
              f" used={used_gb:.2f}GB peak={peak_gb:.2f}GB limit={limit_gb:.2f}GB")
        if detail:
            _probe_hbm_arrays(label, d, host)
```

`bytes_in_use` from `memory_stats()` is the authoritative per-device HBM figure.
It includes all XLA allocations — model weights, KV cache, compile-time buffers,
and any XLA-internal storage.

---

## `_probe_hbm_arrays` — per-dtype live-array breakdown

Added 2026-05-12.  Uses `jax.live_arrays()` + `arr.addressable_shards` to
enumerate the actual per-device bytes by dtype.

```python
def _probe_hbm_arrays(label: str, d, host: str) -> None:
    all_live = jax.live_arrays()
    for arr in all_live:
        for shard in arr.addressable_shards:
            if shard.device == d:
                nbytes = int(shard.data.size) * arr.dtype.itemsize
                dtype_bytes[str(arr.dtype)] += nbytes
    # prints [HBM-DETAIL] lines with per-dtype GB and count
```

### Implementation notes

- `jax.live_arrays()` returns **global sharded arrays**.  Calling `.device()` on
  them raises; `.addressable_shards` gives the per-device physical shards.
- `shard.data.size` is the actual shard element count (not global tensor size),
  so the byte count correctly reflects what each device holds.
- **dev=0 is authoritative.**  Due to EP×TP mesh routing, dev=1-7 may
  double-count shards and report ~2× the actual HBM.  Always compare
  `[HBM-DETAIL]` totals against `[HBM] used=` from `memory_stats()`.

---

## Measured results — TPU v7x, FP8 PTQ, 2026-05-12

**Configuration:** GKE single-host job (`jingnw-flex-tpu7`, 2×2×1, 4 chips),
FP8 PTQ checkpoint, TP=4 EP=2, 8 JAX devices (2 TensorCores × 4 chips).
Checkpoint: `gs://jingnw-mimo-v2-flash-us-central1/mimo-v2-flash-fp8-ptq/0/items`.

### Stage-by-stage HBM per TensorCore (dev=0, 96 GiB capacity)

| Stage | `bytes_in_use` | Delta | Source |
|---|---|---|---|
| `init` | 0.00 GB | — | JAX runtime only |
| `before_setup_decode_state` | 0.00 GB | 0 | Pre-allocation |
| **`after_setup_decode_state`** | **71.93 GB** | **+71.93 GB** | Weights + KV cache allocated together in one call |
| `after_load_params` | 71.93 GB | 0 | `setup_decode_state` is called inside `load_params` |
| `after_prefill` | 72.02 GB | +0.09 GB | KV cache fill + prefill result buffer |
| `after_insert` | 72.17 GB | +0.15 GB | Decode slot population |
| **`generate_step_0512`** | **72.31 GB** | +0.14 GB | Steady-state (KV written + generate RNG buffers) |

Peak HBM: **72.31 GB / 94.75 GiB = 76.3%** per TensorCore.

### Per-chip and per-node totals

| Scope | Used | Capacity |
|---|---|---|
| 1 TensorCore (1 JAX device) | 72.31 GB | 94.75 GiB (~96 GiB) |
| 1 chip (2 TensorCores) | ~144.6 GB | ~192 GiB |
| 4-chip node (8 JAX devices) | **~578.5 GB** | **~768 GiB** |

### Dtype breakdown at `after_load_params` (dev=0)

| dtype | HBM | Tensors | % |
|---|---|---|---|
| `bfloat16` | **71.928 GB** | 568 | ~100% |
| `uint32` | < 0.001 GB | 3 | ~0% |

**Key finding:** No `float8_e4m3fn` tensors appear in JAX live arrays despite
loading from an FP8 PTQ checkpoint.  The qwix `PtqProvider` presents all 568
weight tensors as `bfloat16` to the Python/JAX layer.  See
[mimo_v2_flash_fp8_dtypes.md](mimo_v2_flash_fp8_dtypes.md#4-qwix-fp8-ptq-hbm-representation).

### KV cache is negligible at this configuration

The +0.09 GB delta between `after_setup_decode_state` and `after_prefill`
represents the filled KV cache for `max_target_length=1024` tokens,
`per_device_batch_size=1`.  Model weights dominate at >99.8% of HBM.

---

## TPU v6e reference (from env_restore guide)

On TPU v6e-32 with BF16 checkpoint (TP=4 EP=8, 32 chips, 32 JAX devices):

- HBM per chip: **~18.0 GB / 31.25 GiB** after decode-state init (57.5%)
- Each TPU v6e chip has 1 TensorCore with 32 GiB HBM (no 2-TensorCore-per-chip split)

---

## Related documents

- [mimo_v2_flash_tpu_v7x_gke_env_restore.md](mimo_v2_flash_tpu_v7x_gke_env_restore.md)
- [mimo_v2_flash_fp8_dtypes.md](mimo_v2_flash_fp8_dtypes.md)
- [mimo_v2_flash_inference_overview.md](mimo_v2_flash_inference_overview.md)
