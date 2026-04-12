# MiMo-V2-Flash Optimization #2 — Sparse MoE Dispatch: Plan & Exit Criteria

## Three-Step Plan (Overview)

| Step | Description | Status |
|---|---|---|
| **1** | Fix the stacked checkpoint shape mismatch | ✅ Done (`f9635502`) |
| **2** | Implement sparse dispatch in `MiMoV2FlashSparseMoeBlock` | ✅ Done (`4cb181c3`) |
| **3** | Benchmark with `scan_layers=True` + sparse dispatch | ✅ Done (2026-04-12, median 56.1 ms) |

**Step 1** — ✅ Complete. The stacked checkpoint is live at
`gs://jingnw-mimo-v2-flash-us-east5/mimo-v2-flash-4phase-stacked/checkpoints/0/items`.
Load test passed on 2026-04-11: `generate_step_0130` decoded `'2'`, `generate_step_0133`
decoded `'4'` for "What is 2+2?". HBM used: 18.91 GB / 31.25 GB per chip.
See [Lessons from Step 1](#lessons-from-step-1) for the three bugs encountered.

**Step 2** — Replace the current dense einsum block in `MiMoV2FlashSparseMoeBlock.__call__`
with the `permute → mblx.gmm (w0, w1, wo) → unpermute` pattern from
`MoeBlock.sparse_matmul()`. This requires NNX → Linen adaptation since
`MoeBlock.sparse_matmul` is Linen-based, but the pattern is straightforward.

**Step 3** — Run `mimo_v2_flash_bench.py` with the stacked checkpoint and
`scan_layers=True`. Expected: significant step latency drop (the dense einsum is
~32× excess compute vs. what's needed).

---

## Status

✅ **Step 1 complete** — stacked checkpoint valid, load test passed (2026-04-11, HEAD `f9635502`).

✅ **Step 2 complete** — sparse dispatch (`permute → mblx.gmm × 3 → unpermute`) integrated in
`MiMoV2FlashSparseMoeBlock.__call__` (commit `4cb181c3`).

✅ **Step 3 complete** — benchmark run on 2026-04-12 (`scan_layers=true`, stacked checkpoint):
median **56.1 ms**, throughput **570.4 tok/s**, per-seq **1.8 ms/tok**.

---

## Background

The current dense dispatch in `MiMoV2FlashSparseMoeBlock.__call__`
(`src/maxtext/models/mimo_v2_flash.py`) computes all `E_local = E/EP = 32`
local experts per device for every token:

```python
gate = jax.nn.silu(jnp.einsum("th,ehi->eti", tokens_fp, wi_0))  # (E_local, T, I)
up   = jnp.einsum("th,ehi->eti", tokens_fp, wi_1)                # (E_local, T, I)
down = jnp.einsum("eti,eih->eth", gate * up, wo)                  # (E_local, T, H)
out  = jnp.einsum("te,eth->th", dispatch_weights, down)           # (T, H)
```

Only `K × T / E_total = 8 × 32 / 256 = 1` local expert is selected per device
on average, so ~31 of 32 expert matmuls are wasted (~32× excess compute).

A gather-based sparse dispatch (`wi_0[top_k_indices]`) was attempted but OOMed:
with `scan_layers=False` (47 unrolled layers) XLA must hold 47 × 3 = 141
intermediates simultaneously → ~22 GB HLO temp, 10 GB over the 31.25 GB HBM
limit.

The 4-phase `scan_layers=True` infrastructure (commits `dc19b7ae`→`f6bfd995`)
was built specifically to unblock this by bounding peak HLO temp to ~3 GB/layer.
It requires a correctly-stacked checkpoint.

---

## Part 1 — Fix the Stacked Checkpoint ✅ DONE

### What was done (2026-04-11)

Three iterations were needed to produce a working stacked checkpoint:

| Bug | Root cause | Fix | Commit |
|---|---|---|---|
| GCS save failed | Old broken checkpoint still present at destination | `gsutil -m rm -r` + rerun | — |
| Shape mismatch `(4096,4)` vs `(4,4096)` | `np.stack(axis=0)` but `param_scan_axis: 1` (base.yml default) → leading dim in wrong place | Changed `param_scan_axis: 0` in `mimo-v2-flash.yml`; tool uses `axis=0` | `c214c3f9` |
| HBM OOM on load (`1.75G`, 544M free) | `weight_dtype: float32` (base.yml default) → orbax allocated stacked MoE weights as float32; `[7,32,512,4096]×f32 = 1.75 GiB` × 18 concurrent arrays = 31.5 GiB > 31.25 GiB chip limit | Set `weight_dtype: bfloat16` in `mimo-v2-flash.yml` | `f9635502` |

### Stacking tool (for future reference)

The canonical tool is `src/maxtext/tools/mimo_stack_checkpoint.py` — the only copy
(the older `tools/mimo_stack_checkpoint.py` was deleted in `13500844`). Strategy:
transfer all params to CPU via `jax.device_get` + `np.stack(axis=0)`, then save.
No HBM pressure during stacking.

Command template (run on all 8 workers):

```bash
gcloud compute tpus tpu-vm ssh jingnw-node --zone=us-east5-b --worker=all \
  --command='
    export PATH="$HOME/.local/bin:$PATH"
    source ~/maxtext/maxtext_tpu_venv/bin/activate
    cd ~/maxtext
    nohup python3 -m maxtext.tools.mimo_stack_checkpoint \
      src/maxtext/configs/base.yml model_name=mimo-v2-flash \
      load_parameters_path=gs://jingnw-mimo-v2-flash-us-east5/mimo-v2-flash-fixed-ocdbt/checkpoints/0/items \
      base_output_directory=gs://jingnw-mimo-v2-flash-us-east5/ \
      run_name=mimo_stack_convert per_device_batch_size=1 \
      max_target_length=512 max_prefill_predict_length=128 \
      attention=dot_product scan_layers=false \
      ici_tensor_parallelism=4 ici_expert_parallelism=8 async_checkpointing=false \
      > /tmp/mimo_stack.log 2>&1 &
    echo "pid=$!"
  '
```

Note: `weight_dtype` and `param_scan_axis` are now set in `mimo-v2-flash.yml`
directly, so no need to pass them on the command line.

### Exit criteria (all verified ✅ 2026-04-11)

- ✅ Shape: `layers_b.*` leading dim `4`; `layers_c.layers_*.*` leading dim `7`
- ✅ Load test: no `ValueError` during restore
- ✅ First generate step: `generate_step_0130` decoded `'2'`, next decoded `'4'` → "2+2=4"
- ✅ HBM: 18.91 GB / 31.25 GB used (60%, 12.34 GB headroom)

---

## Part 2 — Implement Sparse MoE Dispatch

### Approach

Replace the four dense einsums in `MiMoV2FlashSparseMoeBlock.__call__` with the
`permute → mblx.gmm (w0, w1, wo) → unpermute` pattern already used by
`MoeBlock.sparse_matmul()` in `src/maxtext/layers/moe.py`.

#### Reference: what `MoeBlock.sparse_matmul` does

1. **`permute(tokens, gate_logits)`** — sorts tokens by expert assignment within
   the EP shard; returns `(sorted_tokens, sorted_expert_ids, weights, group_sizes, ...)`.

2. **`mblx.gmm(sorted_tokens, wi_0, group_sizes)`** — Pallas grouped-matmul
   kernel; computes only the K/EP ≈ 1 selected expert per device per token,
   avoiding the full `(E_local, T, I)` temporary.

3. **SwiGLU**: `gate = silu(gmm_w0_out)`, `up = gmm_w1_out`, `hidden = gate * up`.

4. **`mblx.gmm(hidden, wo, group_sizes)`** — down-projection.

5. **`unpermute(output, weights)`** — restores original token order and applies
   top-k combination weights.

#### Why this is safe with `scan_layers=True`

`mblx.gmm` materializes only one layer's worth of temporaries at a time.  With
the 4-phase scan body (6-layer cycle, compiled once and looped 7×), peak HLO
temp stays at ~3 GB vs. ~22 GB for the unrolled gather approach.

### Sub-steps

**Sub-step 1 — Understand EP token routing** _(read-only, ~15 min)_

`MoeBlock.permute` in `moe.py` re-runs its own `top_k` from raw gate logits and
branches on ring-of-experts / all-to-all paths that MiMo doesn't use.
`MiMoV2FlashSparseMoeBlock` already has `top_k_indices (T,K)` and
`top_k_weights (T,K)` from `MiMoV2FlashMoEGate`.  Confirm that MiMo's EP
dispatch only needs the *local_permute* path (no cross-shard all-to-all needed
because the dense einsum already handles cross-EP reduction implicitly and the
sparse path will use `jax.lax.psum` for the output).

**Sub-step 2 — Write `_mimo_permute` + `_mimo_unpermute` helpers** _(~30 min)_

Inline minimal helpers inside `mimo_v2_flash.py` (no dependency on `MoeBlock`):

```python
# permute: sort tokens by their selected expert (within EP shard)
flat_experts  = top_k_indices.ravel()                          # (T*K,)
sort_order    = jnp.argsort(flat_experts)
sorted_tokens = jnp.repeat(tokens, K, axis=0)[sort_order]     # (T*K, H)
group_sizes   = jnp.bincount(flat_experts, length=E_local)    # (E_local,)

# unpermute: reverse sort, reshape, weight-sum over K
unsorted = sorted_output[jnp.argsort(sort_order)]             # (T*K, H)
output   = (unsorted.reshape(T, K, H) * top_k_weights[..., None]).sum(axis=1)
```

**Sub-step 3 — Replace the three dense einsums with `mblx.gmm`** _(~45 min, core change)_

In `MiMoV2FlashSparseMoeBlock.__call__` replace:
```python
gate = silu(einsum("th,ehi->eti", tokens_fp, wi_0))   # (E_local, T, I)
up   = einsum("th,ehi->eti", tokens_fp, wi_1)
down = einsum("eti,eih->eth", gate * up, wo)
out  = einsum("te,eth->th", dispatch_weights, down)
```
With:
```python
sorted_tokens, sort_order, group_sizes = _mimo_permute(tokens, top_k_local, E_local)
g  = mblx.gmm(sorted_tokens, wi_0_local, group_sizes=group_sizes,
               preferred_element_type=cfg.dtype, tiling=(128, 128, 128))
u  = mblx.gmm(sorted_tokens, wi_1_local, group_sizes=group_sizes,
               preferred_element_type=cfg.dtype, tiling=(128, 128, 128))
h  = jax.nn.silu(g) * u
d  = mblx.gmm(h, wo_local, group_sizes=group_sizes,
               preferred_element_type=cfg.dtype, tiling=(128, 128, 128))
output = _mimo_unpermute(d, sort_order, top_k_weights_local, T, K)
```
Add `from maxtext.kernels import megablox as mblx` to imports.

**Sub-step 4 — Handle the EP-local expert ID mapping** _(~20 min, tricky)_

`top_k_indices` is global `[0, 256)`. Each EP shard owns expert IDs
`[shard_id × E_local, (shard_id+1) × E_local)`. Need to:
- Compute `shard_id = jax.lax.axis_index("expert")`.
- Mask `top_k_indices` to only those hitting this shard:
  `local_mask = (top_k_indices >= shard_id * E_local) & (top_k_indices < (shard_id+1) * E_local)`.
- Re-index to `[0, E_local)`: `top_k_local = top_k_indices - shard_id * E_local` (masked).
- After `_mimo_unpermute`, reduce across EP via `jax.lax.psum(output, axis_name="expert")`
  (same reduction the dense `dispatch_weights` einsum performs implicitly).

**Sub-step 5 — Correctness test** _(~30 min)_

Run decode with `scan_layers=false` (fast iteration — no stacked checkpoint needed)
and the flat checkpoint.  Compare output tokens for a known prompt against the
dense dispatch baseline.  At minimum verify the "2+2=4" and harmonic mean
(120 km each leg → 80 km/h avg) prompts produce identical answers.

**Sub-step 6 — OOM elimination test with `scan_layers=true`** _(~15 min)_

Run decode with `scan_layers=true` + stacked checkpoint.  Confirm:
- No `RESOURCE_EXHAUSTED` during XLA compilation.
- `generate_step_0000` appears cleanly.

**Sub-step 7 — Benchmark** _(~15 min)_

Run `src/maxtext/inference/scripts/mimo_v2_flash_bench.py` (3-step warmup,
50 timed steps, batch=32, TP=4 EP=8) with stacked checkpoint + `scan_layers=true`.
Record median step latency and compare to **56.5 ms** post-opt-#1 baseline.

### Implementation files to change

| File | Change |
|---|---|
| `src/maxtext/models/mimo_v2_flash.py` | Add `from maxtext.kernels import megablox as mblx` |
| `src/maxtext/models/mimo_v2_flash.py` | Add `_mimo_permute` / `_mimo_unpermute` module-level helpers |
| `src/maxtext/models/mimo_v2_flash.py` | Replace dense einsums in `MiMoV2FlashSparseMoeBlock.__call__` with `permute → mblx.gmm × 3 → unpermute` (sub-steps 2–4) |

### Exit criteria for Part 2

1. **No OOM during XLA compile** with `scan_layers=True` + stacked checkpoint.
2. **Correctness**: output of the train problem (`A→B: 120 km at 60 km/h then
   120 km at 120 km/h`) is still **80 km/h** (matches `scan_layers=False` baseline).
3. **Benchmark** (`mimo_v2_flash_bench.py`, 3-step warmup, 50 timed steps,
   batch=32, TP=4 EP=8): median step latency measurably lower than **56.5 ms**
   (the post-opt-#1 baseline) and close to the compute-bound lower bound
   (estimate: ~10–20 ms/step once dense einsum is replaced).

---

## Part 3 — Benchmark and Record Results

After Part 2 passes:

1. Run `src/maxtext/inference/scripts/mimo_v2_flash_bench.py` with the new
   sparse dispatch + stacked checkpoint.
2. Update the benchmark history table in
   [mimo_v2_flash_tpu_perf_optimization.md](mimo_v2_flash_tpu_perf_optimization.md).
3. Mark opt #2 as ✅ in that file.

---

## Key Constants (for reference)

| Constant | Value |
|---|---|
| Total experts `E` | 256 |
| Top-k `K` | 8 |
| EP shards | 8 |
| Local experts `E_local = E/EP` | 32 |
| Expected local selections per step `K×T/E` | ~1 per device |
| Excess compute (dense vs sparse) | ~32× |
| Flat checkpoint (source of truth) | `gs://jingnw-mimo-v2-flash-us-east5/mimo-v2-flash-fixed-ocdbt/checkpoints/0/items` |
| Stacked checkpoint (to regenerate) | `gs://jingnw-mimo-v2-flash-us-east5/mimo-v2-flash-4phase-stacked/checkpoints/0/items` |
| Cluster | `jingnw-node`, `us-east5-b`, v6e-32 |
| Venv on workers | `~/maxtext/maxtext_tpu_venv/` |
