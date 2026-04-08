# MiMo-V2-Flash TPU Inference — Generation Stage Performance Analysis

Benchmarks use `src/maxtext/inference/scripts/mimo_v2_flash_bench.py`
(3-step warmup, 50 timed steps, v6e-32 TP=4 × EP=8, bf16, batch=32).

### Benchmark history

| Date | Optimisation applied | Median step | Throughput | Per-seq |
|---|---|---|---|---|
| 2026-04-08 | Baseline (no opt) | 71.7 ms | 446.6 tok/s | 2.2 ms/tok |
| 2026-04-08 | **#1 Remove `jax.debug.print`** ✅ | **56.5 ms** | **566.1 tok/s** | **1.8 ms/tok** |

Removing a single debug line eliminated 47 host–device sync barriers per step
(one per MoE layer), cutting step latency by **21 %** and boosting throughput
by **27 %**.  Variance also tightened significantly (max − min: 0.6 ms vs 4.1 ms).

Note: the demo-script figure of ~78 ms/step was inflated by cold JIT compilation
on the first few steps; benchmark numbers above exclude warmup.

---

## Root-Cause Bottlenecks (in priority order)

### 1. ~~`jax.debug.print` in the MoE gate hot path~~ ✅ Fixed (commit `00998532`)

`MiMoV2FlashMoEGate.__call__` (`src/maxtext/models/mimo_v2_flash.py`) was firing
`jax.debug.print(...)` on every forward pass — an **effects callback** that
inserts a host–device sync barrier.  With 47 MoE layers per generate step that
was 47 host roundtrips *per token*.

**Result**: 71.7 ms → **56.5 ms/step** (↓21 %), 446.6 → **566.1 tok/s** (↑27 %).

---

### 2. Dense MoE dispatch (all 256 experts computed per step)

`MiMoV2FlashSparseMoeBlock.__call__` uses:

```python
gate = jax.nn.silu(jnp.einsum("th,ehi->eti", tokens_fp, wi_0))  # (256, T, I) — all experts
```

This dense einsum computes a gemm for **all 256 experts** even though only 8
are selected per token.  The MaxText config has `megablox=true` /
`sparse_matmul=true` but the model bypasses it entirely.  During decode
(T = 1 token) this is 256 gemv ops instead of 8.

**Fix**: Replace with MaxText's built-in MegaBlox sparse dispatch path or a
sparse gather–scatter approach over only the top-8 experts.

---

### 3. SWA KV cache allocated for full `max_target_length`

39 of 48 layers use sliding-window attention (128-token window).  The KV cache
for each SWA layer is still allocated at `max_target_length = 2512`.  That is
19.5× more HBM than needed for those layers.  Excess HBM per device:

$$39 \times 2 \times (2512 - 128) \times 8\,\text{heads} \times 128\,\text{dim} \times 2\,\text{bytes} \approx 1.4\,\text{GB/device}$$

More importantly, every generate step reads the full `max_target_length` KV
buffer for each SWA layer even though only the 128-token window is relevant
(the attention mask zeroes the rest, but the data still needs to be streamed
from HBM).

**Fix**: Decouple `cache_seq_len` per layer; set SWA layers to
`window_size + prefill_length` (e.g. 640) and global layers to
`max_target_length`.

---

### 4. No KV cache quantisation

`quantize_kvcache=false`.  The KV cache is stored in bf16.  During
autoregressive decode the dominant cost is **reading cached KV from HBM**, not
compute.  Enabling int8 KV quantisation halves KV-read bandwidth.

**Immediate fix** (no model change needed):

```
quantize_kvcache=true  kv_quant_dtype=int8  kv_quant_axis=heads_and_dkv
```

Accuracy degradation is typically < 0.3 % on reasoning tasks at int8.

---

### 5. `jax.effects_barrier()` + `.item()` sync per step

In `decode.py`, every generate step calls:

```python
decode_state, sampled_tokens = engine.generate(...)
jax.effects_barrier()                        # CPU blocks until TPU done
_tok = sampled_tokens.get_result_at_slot(0).tokens.item()  # device→host copy
```

Both are synchronisation points that prevent any pipeline overlap between CPU
work and TPU execution.  Bug #1 above (debug.print) already causes forced
syncs, so fixing that also mitigates this path; however the explicit
`effects_barrier` / `.item()` calls add host overhead independently.

**Fix**: Remove the unconditional `effects_barrier` from the timing loop.  For
the EOS check, implement on-device via `jax.lax.cond` or a deferred async
token check to avoid the per-step host roundtrip.

---

## Optimisation Methods (ranked by expected impact)

| # | Method | Config / Code change | Expected speedup |
|---|--------|----------------------|-----------------|
| 1 | **Remove `jax.debug.print` from MoE gate** | Delete debug line in `mimo_v2_flash.py` | **Large** — eliminates 47 sync roundtrips/step |
| 2 | **Sparse MoE dispatch (top-8 only)** | Use MaxText MegaBlox path or gather-based sparse dispatch | **Large** — 256 → 8 expert compute (32×) |
| 3 | **Int8 KV cache quantisation** | `quantize_kvcache=true kv_quant_dtype=int8` | ~2× on KV-bandwidth-bound steps |
| 4 | **Truncate SWA KV cache to window size** | Decouple `cache_seq_len` per layer for 39 SWA layers | ~10–20 % HBM reduction, faster KV reads |
| 5 | **Batch size > 1 (throughput mode)** | `per_device_batch_size=2` or more | Linear throughput scaling at low batch cost |
| 6 | **Paged attention** | `attention=paged` — enables continuous batching | Required for production multi-request serving |
| 7 | **Int8 weight quantisation** | `quantization=int8` | ~2× weight-read bandwidth |
| 8 | **Remove per-step `effects_barrier` / async EOS** | Move EOS check on-device (`lax.cond`) | Removes host stall, enables step overlap |
| 9 | **AOT compilation + buffer donation** | `engine._compile_generate_and_get_layouts()` with `donate_argnames` | Eliminates per-step buffer alloc |
| 10 | **Ring-of-experts for EP all-reduce** | `use_ring_of_experts=true` | Overlaps MoE EP communication with compute |
| 11 | **Speculative decoding** | Draft model (smaller MiMo) + verifier | 2–4× latency reduction for greedy/near-greedy |
| 12 | **Reduce `max_target_length`** | Set to `prefill + actual_budget` (e.g. 1024 instead of 2512) | Smaller KV allocation, faster first-step compile |
| 13 | **Chunked prefill** | `use_chunked_prefill=True` | Overlaps prefill and generate for multi-request |
| 14 | **Shardy partitioner** | `shardy=True` | May find better sharding strategies |

---

## Most Impactful Immediate Fixes

Optimisations **#1** (debug.print removal) and **#2** (sparse MoE dispatch)
are bugs / regressions, not trade-offs — they should be fixed before any
tuning.  Combined, they likely account for the majority of the ~78 ms/step
latency.

After fixing those two, **#3** (int8 KV cache) is the highest-leverage tuning
knob with zero model-quality change — it requires only adding two flags to the
`build_decode_command` call in `demos/mimo_v2_flash_demo_jax.py`.
