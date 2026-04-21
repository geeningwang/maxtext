# MiMo-V2-Flash TPU Inference — Generation Stage Performance Analysis

Benchmarks use `src/maxtext/inference/scripts/mimo_v2_flash_bench.py`
(3-step warmup, 50 timed steps, v6e-32 TP=4 × EP=8, bf16, batch=32).

### Benchmark history

| Date | Optimisation applied | Median step | Throughput | Per-seq | Status |
|---|---|---|---|---|---|
| 2026-04-08 | Baseline (no opt) | 71.7 ms | 446.6 tok/s | 2.2 ms/tok | — |
| 2026-04-08 | **#1 Remove `jax.debug.print`** | **56.5 ms** | **566.1 tok/s** | **1.8 ms/tok** | ✅ |
| 2026-04-12 | #2 Sparse MoE dispatch (`mblx.gmm`, `scan_layers=true`, stacked ckpt) | ~~56.1 ms~~ | ~~570.4 tok/s~~ | ~~1.8 ms/tok~~ | ⚠️ Invalid |
| 2026-04-13 | #3 Int8 KV cache quantisation (`quantize_kvcache=true`) | 60.1 ms | 532.7 tok/s | 1.9 ms/tok | ❌ Rejected |
| 2026-04-13 | #4 SWA KV cache truncation (`mimo_truncate_swa_kv_cache=true`) | 1797.6 ms† | 17.8 tok/s† | 56.2 ms/tok† | ❌ Rejected |
| 2026-04-15 | #5 `shard_map` EP+TP sparse dispatch (commits `01527b9c`–`2ae1dc41`) | 160 ms | 200 tok/s | 5.0 ms/tok | ❌ Reverted |
| 2026-04-16 | **#6 Revert sparse code; run dense dispatch from opt #1 baseline** (commit `30fd5e55`) | **55.7 ms** | **575 tok/s** | **1.7 ms/tok** | ✅ |
| 2026-04-17 | #7 `scan_layers=true` — 4-phase stacked ckpt bench-only (commits `0a084626`–`539cc043`; demo broken — 4-phase decoder code lost in `30fd5e55`, see note) | 68.3 ms | 468 tok/s | 2.1 ms/tok | ⚠️ Bench-only; +23% vs best dense |
| 2026-04-17 | **#8 SWA fix + debug logging cleanup** (commit `72f75972`) — dense baseline re-confirmed | **55.5 ms** | **576 tok/s** | **1.7 ms/tok** | ✅ |
| 2026-04-20 | #9 Ragged-A2A sparse EP routing (opt4) — reverted, 83% regression | 101.5 ms | 315.5 tok/s | 3.2 ms/tok | ❌ Reverted |
| 2026-04-20 | **#10 Revert opt4; add prefill benchmark** (commit `055a4c2d`) — both scan modes verified | **55.4 ms** | **577.5 tok/s** | **1.7 ms/tok** | ✅ |
| 2026-04-21 | **#11 Batch size scaling (opt5)** — sweep per_device_batch_size=1→11, OOM at 12 (commits `6d067cdf`–`711f591f`) | **129.2 ms** | **2,724 tok/s** | **0.37 ms/tok** | ✅ Current best |

† A/B comparison (true vs false) shows **0% Δ** in both median step latency and throughput.
The absolute numbers are 32× lower than the opt #1 baseline (56.5 ms) — a separate regression
(root cause identified and fixed in opt #5, see §5 below) that affects both variants equally
and is unrelated to this change.

⚠️ **Opt #2 measurement invalid**: The 56.1 ms figure was recorded from a stale JIT cache
still executing the opt #1 (dense) code path. Commit `4cb181c3` crashes with
`NameError: Found an unbound axis name: expert` on all tested JAX versions (0.4.38 – 0.9.2)
because `jax.lax.axis_index` is only valid inside `jax.shard_map` or `jax.pmap(axis_name=...)`.
The sparse gmm path was never actually benchmarked before the regression was introduced.

Removing a single debug line eliminated 47 host–device sync barriers per step
(one per MoE layer), cutting step latency by **21 %** and boosting throughput
by **27 %**.  Variance also tightened significantly (max − min: 0.6 ms vs 4.1 ms).

Note: the demo-script figure of ~78 ms/step was inflated by cold JIT compilation
on the first few steps; benchmark numbers above exclude warmup.

---

## Reproducing the Results (2026-04-17, v6e-32)

All four cells of the inference-demo × scan matrix. Run from the manager VM against `jingnw-node` (v6e-32, `us-east5-b`). All commands use `--worker=all` to launch all 8 workers simultaneously.

### Inference demo — `scan_layers=false` (dense, OCDBT checkpoint)

```bash
gcloud compute tpus tpu-vm ssh jingnw-node --zone=us-east5-b --worker=all \
  --command='set -e
. "$HOME/maxtext/maxtext_tpu_venv/bin/activate"
cd "$HOME/maxtext"
python demos/mimo_v2_flash_demo_jax.py \
  --checkpoint_path gs://jingnw-mimo-v2-flash-us-east5/mimo-v2-flash-fixed-ocdbt/checkpoints/0/items \
  --tokenizer_path XiaomiMiMo/MiMo-V2-Flash \
  --ici_tensor_parallelism 4 \
  --ici_expert_parallelism 8 \
  --max_new_tokens 512 2>&1 | tail -25'
```

**Result (2026-04-17):** EOS fired; output = *"420 km"* (train distance problem); 2.3 tok/s.

### Inference demo — `scan_layers=true` (4-phase stacked checkpoint)

> ⚠️ Currently garbled — 4-phase decoder code was lost in revert `30fd5e55`. Recorded for completeness; do not use for quality validation until the code is restored.

```bash
gcloud compute tpus tpu-vm ssh jingnw-node --zone=us-east5-b --worker=all \
  --command='set -e
. "$HOME/maxtext/maxtext_tpu_venv/bin/activate"
cd "$HOME/maxtext"
python demos/mimo_v2_flash_demo_jax.py \
  --checkpoint_path gs://jingnw-mimo-v2-flash-us-east5/mimo-v2-flash-4phase-stacked/checkpoints/0/items \
  --tokenizer_path XiaomiMiMo/MiMo-V2-Flash \
  --ici_tensor_parallelism 4 \
  --ici_expert_parallelism 8 \
  --max_new_tokens 512 \
  --scan_layers 2>&1 | tail -25'
```

**Result (2026-04-17):** Garbled output — EOS never fired; hit `max_new_tokens=512`. Root cause: `MiMoV2FlashSixLayerCycleBlockToLinen` / `MiMoV2FlashDecoderLayerToLinen` 4-phase classes missing from current HEAD.

### Benchmark — `scan_layers=false` (dense, OCDBT checkpoint)

```bash
gcloud compute tpus tpu-vm ssh jingnw-node --zone=us-east5-b --worker=all \
  --command='set -e
. "$HOME/maxtext/maxtext_tpu_venv/bin/activate"
cd "$HOME/maxtext"
export PYTHONUNBUFFERED=1
python3 -m maxtext.inference.scripts.mimo_v2_flash_bench \
  src/maxtext/configs/base.yml \
  model_name=mimo-v2-flash \
  run_name=mimo_v2_flash_bench \
  load_parameters_path=gs://jingnw-mimo-v2-flash-us-east5/mimo-v2-flash-fixed-ocdbt/checkpoints/0/items \
  tokenizer_path=XiaomiMiMo/MiMo-V2-Flash \
  max_prefill_predict_length=512 \
  max_target_length=640 \
  per_device_batch_size=1 \
  dtype=bfloat16 \
  weight_dtype=bfloat16 \
  ici_tensor_parallelism=4 \
  ici_expert_parallelism=8 \
  scan_layers=false \
  attention=dot_product \
  checkpoint_storage_use_ocdbt=true \
  checkpoint_storage_use_zarr3=true \
  inference_microbenchmark_log_file_path=/tmp/bench_result.json 2>&1 | grep -E "^\[BENCH\]" | tail -10'
```

**Result (2026-04-17):** Median 55.5 ms · 576 tok/s (all 8 workers consistent). Read result: `python3 -c "import json; r=json.load(open('/tmp/bench_result.json')); print(r['step_ms_median'], r['throughput_tok_per_s'])"`

**Result (2026-04-20, commit `055a4c2d`):** Median 55.4 ms decode · 577.5 tok/s; Median 123.6 ms prefill (512 tok) · 4,144 tok/s. Bench script now runs both decode and prefill phases.

### Benchmark — `scan_layers=true` (4-phase stacked checkpoint)

```bash
gcloud compute tpus tpu-vm ssh jingnw-node --zone=us-east5-b --worker=all \
  --command='set -e
. "$HOME/maxtext/maxtext_tpu_venv/bin/activate"
cd "$HOME/maxtext"
export PYTHONUNBUFFERED=1
python3 -m maxtext.inference.scripts.mimo_v2_flash_bench \
  src/maxtext/configs/base.yml \
  model_name=mimo-v2-flash \
  run_name=mimo_v2_flash_scan_bench \
  load_parameters_path=gs://jingnw-mimo-v2-flash-us-east5/mimo-v2-flash-4phase-stacked/checkpoints/0/items \
  tokenizer_path=XiaomiMiMo/MiMo-V2-Flash \
  max_prefill_predict_length=512 \
  max_target_length=640 \
  per_device_batch_size=1 \
  dtype=bfloat16 \
  weight_dtype=bfloat16 \
  ici_tensor_parallelism=4 \
  ici_expert_parallelism=8 \
  scan_layers=true \
  attention=dot_product \
  checkpoint_storage_use_ocdbt=true \
  checkpoint_storage_use_zarr3=true \
  async_checkpointing=false \
  inference_microbenchmark_log_file_path=/tmp/bench_scan_result.json 2>&1 | grep -E "^\[BENCH\]" | tail -5'
```

**Result (2026-04-17):** Median 68.3 ms · 468 tok/s (all 8 workers consistent). Read result: `python3 -c "import json; r=json.load(open('/tmp/bench_scan_result.json')); print(r['step_ms_median'], r['throughput_tok_per_s'])"`

**Result (2026-04-20, commit `055a4c2d`):** Median 68.4 ms decode · 468 tok/s; Median 121.9 ms prefill (512 tok) · 4,200 tok/s. Both scan modes verified working on the same commit.

---

## `scan_layers=true` Analysis (2026-04-17)

> **Note:** The 4-phase scan decoder code (`decoders.py` + `mimo_v2_flash.py`) was lost when sparse dispatch code was reverted in commit `30fd5e55`. The scan benchmark (below) loads the 4-phase stacked checkpoint and measures step latency, but the generated output is garbled. The scan demo path needs the 4-phase `MiMoV2FlashSixLayerCycleBlockToLinen` / `MiMoV2FlashDecoderLayerToLinen` classes restored before it can produce correct text.

`scan_layers=true` with the 4-phase stacked checkpoint runs at **68.3 ms / 468 tok/s** (bench-only) — **23 % slower** than the dense unscanned baseline (55.5 ms / 576 tok/s, commit `72f75972`). HBM footprint is identical (17.98 GB / 31.25 GB).

### Why scan is slower

For AR generate (memory-bandwidth–bound, tiny activations), the bottleneck is **streaming the weight matrices from HBM**, not FLOPS. With 48 unrolled layers XLA can pipeline this:

- **Unrolled (`scan=false`)**: XLA has full visibility of all 48 layers as a single HLO graph. It overlaps HBM reads for layer N+1 with execution of layer N — effectively hiding HBM latency behind compute.
- **Scanned (`scan=true`)**: each phase compiles to a `lax.while_loop`. XLA has per-iteration visibility only. It cannot prefetch weights from the next scan iteration because the carry (hidden state) and the loop counter are symbolic at compile time, blocking cross-iteration scheduling.

The 12.8 ms gap (68.3 − 55.5) over 48 layers = **~0.27 ms per layer** of lost prefetch pipelining. This is the expected cost of a `while_loop` on weight-read–bound workloads.

A secondary factor: the 4-phase structure introduces three additional control-flow transitions (A→B, B→C, C→D) and a nested `_MiMoPhaseCScope` module, each adding loop-counter and carry-copy overhead. This accounts for a small fraction of the gap.

### Implication for the optimisation plan

The original motivation for `scan_layers=true` was to bound peak HLO temporary memory to ~3 GB/layer (vs 22 GB unrolled) so that sparse MoE dispatch intermediates fit in HBM.

The pre-opt4 estimate predicted ragged_all_to_all would achieve ~32.8 ms (−41 %)
based on ~8× HBM bandwidth reduction from sparse weight loads. **This estimate was
invalidated by opt4 (2026-04-20): actual result was 101.5 ms — an 83 % regression.**

Root cause (see opt4 plan doc for full post-mortem): AR decode with T = 32 (batch × 1)
is **weight-bandwidth-bound**, not intermediate-bound. MoE intermediates are 128 KB/layer
at T = 32 — negligible. The bottleneck is reading ~128 MB of expert weights per device
per layer, which is identical for dense and sparse dispatch. The 188 `ragged_all_to_all`
collectives per step added ~46 ms of fixed ICI overhead for zero weight-read savings.

**Revised conclusion (2026-04-20):**

- **For AR decode**: sparse MoE dispatch (any variant) does not help at batch=32 / T=32.
  The dominant cost is expert weight reads from HBM, not excess compute. To reduce decode
  latency, target **weight bandwidth** directly: int8/FP8 weight quantisation (halves reads),
  larger batch size (amortises weight reads across more tokens), or reduced EP (fewer expert
  shards per device → fewer weight reads, but needs more TP).
- **For prefill** (T = 512): sparse MoE dispatch IS valid. With T = 512, MoE intermediates
  are ~2.68 GB/layer — sparse routing provides a genuine 32× intermediate reduction. The
  ragged_all_to_all approach should be re-scoped to target prefill latency (currently
  123.6 ms scan=false, 121.9 ms scan=true) rather than decode.
- **scan_layers=true** is no longer a prerequisite for the highest-priority decode
  optimisations. It remains relevant only as HBM headroom for sparse prefill dispatch.

---

## Root-Cause Bottlenecks (in priority order)

### 1. ~~`jax.debug.print` in the MoE gate hot path~~ ✅ Fixed (commit `00998532`)

`MiMoV2FlashMoEGate.__call__` (`src/maxtext/models/mimo_v2_flash.py`) was firing
`jax.debug.print(...)` on every forward pass — an **effects callback** that
inserts a host–device sync barrier.  With 47 MoE layers per generate step that
was 47 host roundtrips *per token*.

**Result**: 71.7 ms → **56.5 ms/step** (↓21 %), 446.6 → **566.1 tok/s** (↑27 %).

---

### 2. Dense MoE dispatch (all E_local = E/EP = 32 experts computed per token) — ⚠️ Implementation incomplete (commit `4cb181c3` crashes on all JAX versions)

Each EP shard holds `E_local = 256/8 = 32` experts.  The current dense einsum:

```python
gate = jax.nn.silu(jnp.einsum("th,ehi->eti", tokens_fp, wi_0))  # (E_local, T, I)
```

computes all 32 local experts × all T=32 tokens = 1024 (token, expert) pairs per
device per step, even though only `K × T / E_total = 8 × 32 / 256 = 1` local
expert is selected per device on average.  This wastes ~31 of 32 expert
matmuls per device per step (~32× excess compute).

**Attempted fix**: Gather-based sparse dispatch (`wi_0[top_k_indices]`) computing
only K=8 expert slices per token instead of E=256.

**Why it failed (OOM)**:
- The gather creates `(T, K, H_local, I) = (32, 8, 1024, 2048)` temporaries ≈ 1 GB each
- With `scan_layers=False` (47 layers fully unrolled), XLA must hold all 47 × 3 = 141
  matrice intermediates simultaneously → **22 GB HLO temp, 10 GB over the 31.25 GB HBM limit**
- `scan_layers=True` would bound peak to ~3 GB/layer, but MiMo layers require a
  per-layer `layer_idx` constructor argument that is incompatible with the scan wrapper

**Final fix implemented**:
Use MaxText's **`megablox.gmm`** grouped-matmul kernel (already in `src/maxtext/layers/moe.py`):
1. Sort tokens by expert assignment within each EP shard
2. Use `mblx.gmm(tokens_sorted, wi_0, group_sizes)` — a Pallas kernel that avoids
   materializing the full `(T, K, H, I)` weight gather
3. `psum_scatter` the output across EP devices
This pattern is already used by MaxText's `MoeBlock` for Llama4, DeepSeek3, Mixtral, etc.
For MiMo, the sparse path was integrated directly into `MiMoV2FlashSparseMoeBlock`
with local permute/unpermute helpers and `jax.lax.psum(..., axis_name="expert")`.

**Result (2026-04-12):** ⚠️ **Measurement invalid.**
The benchmark was run while the JIT cache still held a compiled artifact from opt #1
(the dense path).  The 56.1 ms figure reflects the opt #1 dense code, not `mblx.gmm`
sparse dispatch.  Commit `4cb181c3` itself raises `NameError: Found an unbound axis name:
expert` on every JAX version (0.4.38 – 0.9.2) because `jax.lax.axis_index("expert")`
was called in a `pjit`+mesh context outside of `jax.shard_map`.  The sparse code path
was never successfully executed before the regression was introduced.  See §5 for the
correct fix (`shard_map` wrapper, commit `2ae1dc41`).

---

### 3. ~~SWA KV cache allocated for full `max_target_length`~~ ❌ Rejected (zero throughput gain)

39 of 48 layers use sliding-window attention (128-token window).  The KV cache
for each SWA layer is still allocated at `max_target_length = 2512`.  That is
19.5× more HBM than needed for those layers.  Excess HBM per device:

$$39 \times 2 \times (2512 - 128) \times 8\,\text{heads} \times 128\,\text{dim} \times 2\,\text{bytes} \approx 0.36\,\text{GB/device}$$

More importantly, every generate step reads the full `max_target_length` KV
buffer for each SWA layer even though only the 128-token window is relevant
(the attention mask zeroes the rest, but the data still needs to be streamed
from HBM).

**Fix implemented**: `mimo_truncate_swa_kv_cache=true` (config flag) decouples
`cache_seq_len` per layer; SWA layers get `window_size + prefill_length` (= 640)
and global layers keep `max_target_length`.

**Measured result (2026-04-13, v6e-32 TP=4 EP=8, batch=32, 3 warmup + 50 timed):**

A/B benchmark with same hardware and config, only toggling `mimo_truncate_swa_kv_cache`:

| Variant | Median step | Throughput | Per-seq |
|---|---|---|---|
| `false` (baseline) | 1797.7 ms | 17.8 tok/s | 56.2 ms/tok/seq |
| `true` (truncation) | 1797.6 ms | 17.8 tok/s | 56.2 ms/tok/seq |
| **Delta** | **−0.1 ms (0%)** | **0%** | **0%** |

- **Throughput**: Zero improvement.  Both variants perform identically within noise.
- **HBM**: Expected savings ~0.36 GB/device; not reflected in `after_setup_decode_state`
  readings (both showed 17.98 GB).  At this sequence length the KV cache is a small
  fraction of total HBM (~0.35 GB vs 17.98 GB params), limiting the impact.
- **Step latency note**: The absolute 1797 ms is 32× higher than the opt #1 baseline
  (56.5 ms) — this is a separate software regression (see §5 below); it affects both
  variants equally and does not bias the A/B comparison.  (Note: the frequently-cited
  "opt #2 baseline of 56.1 ms" is a stale JIT cache artifact and should not be used
  as a reference; see ⚠️ note above the benchmark history table.)

**Decision**: ❌ **Rejected**.  Adds code complexity for zero confirmed throughput
or HBM benefit at the tested configuration.  The `mimo_truncate_swa_kv_cache`
flag defaults to `false`; can be revisited if HBM pressure becomes the limiting
factor (e.g. at larger effective batch sizes or longer sequence budgets).

---

### ✅ Resolved: 32× step-latency regression (April 13 → April 15)

The April 13 benchmark showed median step = 1797 ms versus the April 12 opt #2
baseline of 56.1 ms — a **32× slowdown** exactly equal to `batch_size`.

**Root cause identified (April 15):**

Commit `5ad76eac` introduced a `try/except NameError` guard around
`jax.lax.axis_index("expert")` in `MiMoV2FlashSparseMoeBlock.__call__`:

```python
try:
    shard_id = jax.lax.axis_index("expert")
except NameError:
    # dense fallback: loop over all batch elements
    ...
```

In JAX 0.8.1, `axis_index` called outside of `jax.shard_map` raises `NameError:
Found an unbound axis name: expert. To fix this, please call axis_index under
jax.shard_map.`  The except block silently activated the dense einsum fallback,
processing all 32 batch tokens sequentially (32 × 56 ms ≈ 1792 ms).

**Fix (commit `2ae1dc41`, April 15):**

Wrapped the entire sparse dispatch body in `jax.shard_map` with correct
EP+TP-aware partition specs:
- `tokens_fp`: `P("expert", "tensor")` — batch sharded on EP, embed sharded on TP
- `wi_0`, `wi_1`: `P("expert", None, "tensor")` — EP-expert, full-H, TP-intermediate
- `wo`: `P("expert", "tensor", None)` — EP-expert, TP-intermediate, full-H

Inside the body: all-gather tokens along tensor then expert axes to assemble the
full `(T, H)` view, run the three grouped matmuls, `psum("tensor")` to reduce
the row-parallel down-projection, `psum("expert")` to collect EP contributions,
then `dynamic_slice` back to the local `(T/EP, H/TP)` shard.

**Measured result (2026-04-15, same hardware/config):**
- Median: **160 ms** (vs 1757 ms regression; opt #1 dense baseline was 56.5 ms)
- Throughput: **200 tok/s** (vs 18.2 tok/s regression)
- Improvement vs regression: 11×, but still **3× slower than the opt #1 dense baseline**

The gap between 160 ms and the opt #1 dense baseline (56.5 ms) is due to two
all-gather collectives added by the shard_map boundary (assembling the full `(T,H)`
token matrix on each device before dispatch).

**❌ Subsequently reverted (commit `30fd5e55`, April 16):** The shard_map approach
was rejected because 160 ms is slower than the dense dispatch (55.7 ms). All sparse
dispatch code (`4cb181c3`–`2ae1dc41`) was reverted back to the opt #1 dense einsum
baseline. Current HEAD (`1a6b9579`/`5781158c`) uses the dense `MiMoV2FlashSparseMoeBlock`
from opt #1 with no shard_map, no mblx.gmm, and no axis_index.

Eliminating the all-gathers would require restructuring dispatch to avoid the full
token broadcast (e.g. using `ragged_all_to_all` as in MaxText's `RoutedMoE` path).
This has not yet been measured.

---

### 4. No KV cache quantisation

`quantize_kvcache=false`.  The KV cache is stored in bf16.  During
autoregressive decode the dominant cost is **reading cached KV from HBM**, not
compute.  Enabling int8 KV quantisation halves KV-read bandwidth.

**Tested result (2026-04-13, current MiMo setup):**

KV-int8 was evaluated with:
`quantize_kvcache=true kv_quant_dtype=int8 kv_quant_axis=heads_and_dkv`.

⚠️ **Note**: The April 13 result was presumably measured against the stale-cache
"56.1 ms" baseline (opt #2 JIT cache artifact); 56.1 ms is not a valid reference point
(see ⚠️ note above benchmark table).  The A/B delta (+3.5 ms, ~+6 %) is directionally
valid.  The quality regression finding is unambiguous.

Observed outcome:
1. **Performance regression** in benchmark:
  - `56.1 ms` → `60.1 ms` median (about `+7.1%` slower; baseline figure is a stale JIT cache artifact)
  - `570.2 tok/s` → `532.7 tok/s` throughput (about `−6.6%`)
2. **Quality regression** on harmonic-mean prompt:
  - KV OFF: `80 km/h`
  - KV ON: `64 km/h`

Decision: keep KV-int8 **disabled by default** for now.

Prior hypothesis / configuration tested:

```
quantize_kvcache=true  kv_quant_dtype=int8  kv_quant_axis=heads_and_dkv
```

Accuracy/perf behavior appears workload- and implementation-sensitive in this setup.

---

### 5. `jax.effects_barrier()` + `.item()` sync per step

In `decode.py`, every generate step calls:

```python
decode_state, sampled_tokens = engine.generate(...)
jax.effects_barrier()                        # CPU blocks until TPU done
_tok = sampled_tokens.get_result_at_slot(0).tokens.item()  # device→host copy
```

Both are synchronisation points that prevent any pipeline overlap between CPU
work and TPU execution. With opt #1's debug.print removed, these are now the
**dominant per-step host sync costs** in `decode.py` (confirmed present in current
code at lines 245 and 260). Note: `mimo_v2_flash_bench.py` avoids both — it uses
`jax.block_until_ready(sampled_tokens)` inside the timing loop and does not call
`effects_barrier` or `.item()`, so benchmark numbers are unaffected. The sync
cost only matters for `decode.py` (the demo / production inference path).

**Fix**: Remove the unconditional `effects_barrier` from the timing loop. For
the EOS check, implement on-device via `jax.lax.cond` or a deferred async
token check to avoid the per-step host roundtrip.

---

## Optimisation Methods (re-ranked by expected gain, 2026-04-20)

> Re-ranked after opt4 post-mortem. AR decode at batch=32 / T=32 is **weight-bandwidth-bound**: the bottleneck is reading ~1.5 GB of expert weights per step, not compute or intermediates. Sparse MoE dispatch does not help for decode (proven: 101.5 ms, 83 % regression). Optimisations that reduce **weight reads** or **amortise them across more tokens** are now ranked highest.
>
> Active items (ranked by expected performance gain) are listed first. Completed ✅, measurement-invalid ⚠️, and rejected/reverted ❌ items are at the bottom for reference.

| Rank | Method | Config / Code change | Expected impact | Result |
|---|---|---|---|---|
| 1 | **Batch size > 1 (throughput mode)** | `per_device_batch_size=11` — see [opt5 results](mimo_v2_flash_opt5_batch_size_scaling.md) | **Highest confidence — confirmed.** Sweep: batch=2 (1,032 tok/s, 1.79×), 4 (1,652, 2.86×), 8 (2,428, 4.21×), 10 (2,715, 4.70×), **11 (2,724, 4.72×)**. OOM at batch=12. Prefill unchanged (123.6 ms, compute-bound). | ✅ **Done** — 2,724 tok/s at `per_device_batch_size=11` (total batch 352) |
| 2 | **Int8/FP8 weight quantisation** | `quantization=int8` (or FP8 via `quantization=fp8`) | **High** — directly halves the dominant cost (expert weight reads from HBM). The opt4 post-mortem identified weight reads as THE bottleneck for decode. Not the same as KV-int8 (rejected, opt #3); weight quant targets a different data path. Must test quality impact. | Not run |
| 3 | **Speculative decoding** | Draft model (smaller MiMo or shared-expert-only subset) + verifier | **High** per-sequence latency reduction; higher implementation complexity. Orthogonal to batch-size scaling. | Not run |
| 4 | **Remove per-step `effects_barrier` / async EOS** | Move EOS check on-device (`lax.cond`); remove sync at `decode.py` lines 245, 260 | **Medium** — host-sync removal in demo/production path. Simple, independent of all other opts. Does not affect benchmark numbers (bench already avoids sync). | Not run |
| 5 | **Sparse MoE for prefill** (`ragged_all_to_all`, scoped to prefill phase) | Adopt `RoutedMoE` pattern from `src/maxtext/layers/moe.py`; requires `scan_layers=true` for HBM headroom | **Medium-High for prefill** — ragged_all_to_all IS valid at T=512 (intermediates 2.68 GB/layer; sparse routing gives 32× reduction). Does NOT help decode (T=32, intermediates 128 KB/layer). Current prefill: 123.6 ms (scan=false), 121.9 ms (scan=true). | ❌ Tested for decode (101.5 ms, 83 % regression); **not yet tested for prefill** |
| 6 | **Paged attention** | `attention=paged` — enables continuous batching | Medium for single-stream, high for serving throughput | Not run |
| 7 | **Reduce `max_target_length`** | Set to `prefill + actual_budget` (e.g. 1024 instead of 2512) | Medium for startup/first-token latency and memory footprint | Not run |
| 8 | **Ring-of-experts for EP all-reduce** | `use_ring_of_experts=true` | Low-Medium, depends on comm/compute balance | Not run |
| 9 | **AOT compilation + buffer donation** | `engine._compile_generate_and_get_layouts()` with `donate_argnames` | Low-Medium steady-state, useful startup/allocator wins | Not run |
| 10 | **Chunked prefill** | `use_chunked_prefill=True` | Workload-dependent; more relevant for multi-request serving | Not run |
| 11 | **Shardy partitioner** | `shardy=True` | Uncertain; can help if current partitioning is suboptimal | Not run |
| 12 | **`scan_layers=true` + restore 4-phase decoder** | Cherry-pick `MiMoV2FlashDecoderLayerToLinen` + `MiMoV2FlashSixLayerCycleBlockToLinen` from pre-revert `a28b6718` | **Demoted** — no longer a prerequisite for the top decode opts. Only needed as HBM headroom for sparse prefill (rank 5). Scan adds 23 % decode overhead (68.4 vs 55.4 ms). Demo still broken. | ⚠️ Bench valid (68.4 ms); demo broken |
| — | opt #1 | **Remove `jax.debug.print` from MoE gate** | ✅ Accepted (56.5 ms) |
| — | opt #2 | **Sparse MoE dispatch (top-8 only, `mblx.gmm`)** | ⚠️ Code merged; 56.1 ms figure invalid (stale JIT cache) |
| — | N/A | **`shard_map` EP+TP sparse dispatch** | ❌ Reverted — 160 ms, all-gather overhead (`30fd5e55`) |
| — | N/A | **`ragged_all_to_all` sparse MoE for decode** | ❌ Reverted — 101.5 ms, 83 % regression. Weight-bandwidth-bound at T=32; ICI collective overhead dominates (`055a4c2d`) |
| — | N/A | **Truncate SWA KV cache to window size** | ❌ Rejected — 0 % Δ at tested config |
| — | opt #3 | **Int8 KV cache quantisation** | ❌ Rejected — +6 % slower + quality regression |

---

## Most Impactful Next Fixes (2026-04-20 post-opt4 post-mortem)

Current HEAD (`711f591f`) → **2,724 tok/s / 129.2 ms decode** (`scan_layers=false`, dense dispatch,
`ici_expert_parallelism=8`, `ici_tensor_parallelism=4`, `per_device_batch_size=11`, total batch=352).
Prefill: **123.6 ms / 4,144 tok/s** (512 tokens, compute-bound, unchanged by batch scaling).

**Key insight from opt4 failure:** AR decode at batch=32 / T=32 is **weight-bandwidth-bound**.
The bottleneck is reading ~1.5 GB of expert weights per step (47 MoE layers × 3 projections ×
32 experts × H × I × 2 bytes). Sparse MoE dispatch does not reduce weight reads — it only
reduces intermediates (128 KB/layer at T=32, negligible). All three sparse dispatch attempts
(gmm, shard_map, ragged_all_to_all) regressed decode latency. Future decode optimisations
must target **weight bandwidth** directly.

Highest-priority items ranked by expected decode throughput gain:

1. **Batch size > 1 — ✅ Done (opt5, 2026-04-21).**
   Full sweep: batch=2 (1,032 tok/s), 4 (1,652), 8 (2,428), 10 (2,715), **11 (2,724, optimal)**.
   OOM at batch=12 (XLA HLO temp allocation). New baseline: `per_device_batch_size=11`, 2,724 tok/s.
2. **Int8/FP8 weight quantisation — directly targets the identified bottleneck (rank 1, next up).**
   `quantization=int8` halves expert weight reads from HBM — the dominant cost identified
   by the opt4 post-mortem. Unlike KV-int8 (rejected, +6 % latency + quality regression),
   weight quantisation targets a completely different data path (model weights, not KV cache).
   Must measure both latency improvement and quality impact on reference prompts.
   **Run at the new baseline (`per_device_batch_size=11`) to measure stacked gain.**
3. **Speculative decoding (rank 2).** Draft model (smaller MiMo variant or shared-expert-only
   subset) produces K candidate tokens per step, verified in one pass by the full model.
   High per-sequence latency reduction; orthogonal to batch scaling. Higher implementation
   complexity.
4. **Remove per-step `effects_barrier` / async EOS (rank 3).** Simple host-sync removal in
   `decode.py` (lines 245, 260). Does not affect benchmark numbers (bench avoids sync), but
   improves demo/production inference path latency. Independent of all other items.
