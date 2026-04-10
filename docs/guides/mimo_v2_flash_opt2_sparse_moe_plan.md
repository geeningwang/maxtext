# MiMo-V2-Flash Optimization #2 — Sparse MoE Dispatch: Plan & Exit Criteria

## Three-Step Plan (Overview)

| Step | Description | Status |
|---|---|---|
| **1** | Fix the stacked checkpoint shape mismatch | ❌ To do |
| **2** | Implement sparse dispatch in `MiMoV2FlashSparseMoeBlock` | ❌ Blocked on Step 1 |
| **3** | Benchmark with `scan_layers=True` + sparse dispatch | ❌ Blocked on Step 2 |

**Step 1** — The `mimo-v2-flash-4phase-stacked` checkpoint has a `(4096,4)` vs `(4,4096)` shape
mismatch on `layers_d.mlp.wo`. Re-run `mimo_stack_checkpoint.py` from the flat
`fixed-ocdbt` checkpoint using current HEAD code.

**Step 2** — Replace the current dense einsum block in `MiMoV2FlashSparseMoeBlock.__call__`
with the `permute → mblx.gmm (w0, w1, wo) → unpermute` pattern from
`MoeBlock.sparse_matmul()`. This requires NNX → Linen adaptation since
`MoeBlock.sparse_matmul` is Linen-based, but the pattern is straightforward.

**Step 3** — Run `mimo_v2_flash_bench.py` with the stacked checkpoint and
`scan_layers=True`. Expected: significant step latency drop (the dense einsum is
~32× excess compute vs. what's needed).

---

## Status

❌ **Blocked on Step 1** — stacked checkpoint (`mimo-v2-flash-4phase-stacked`) has a shape
mismatch on `layers_d.mlp.wo` and must be regenerated before the sparse dispatch code can be tested.

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

## Part 1 — Fix the Stacked Checkpoint

### Root cause

The existing `mimo-v2-flash-4phase-stacked` checkpoint was written by an older
version of the stacking tool (likely at commit `03ed01ab` or `0637c783`) where the
weight axis convention for `wo` in Phase D differed from current HEAD.  The tool
just copies Phase D (no stacking), so whatever transposition issue was in that run
is baked into the checkpoint.  The flat `fixed-ocdbt` checkpoint is untouched and
correct — it is the source of truth.

### Tool to use

There are two versions of the stacking tool; always use the **latest** (`src/maxtext/tools/`):

| File | Strategy | Commit | Status |
|---|---|---|---|
| `tools/mimo_stack_checkpoint.py` | Donation-based HBM stacking (JAX JIT) | `05c22878` | Older — do not use |
| **`src/maxtext/tools/mimo_stack_checkpoint.py`** | CPU numpy stacking (`jax.device_get` + `np.stack`) | `0637c783` | **Latest — use this** |

The latest tool transfers all params to CPU first (`jax.device_get`), stacks with
`np.stack`, then saves.  No HBM pressure during stacking.

### Step 1 — Run the stacking tool on all 8 workers

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
      attention=dot_product scan_layers=false weight_dtype=bfloat16 \
      ici_tensor_parallelism=4 ici_expert_parallelism=8 async_checkpointing=false \
      > /tmp/mimo_stack.log 2>&1 &
    echo "pid=$!"
  '
```

Note: loading 586 GB of params over GCS takes ~10–15 minutes.

### Step 2 — Poll progress

```bash
gcloud compute tpus tpu-vm ssh jingnw-node --zone=us-east5-b --worker=0 \
  --command="grep -E 'Stack tool:|HBM|Error|Traceback' /tmp/mimo_stack.log | tail -10"
```

### Exit criteria for Part 1

All three checks must pass:

**Check 1 — Shape verification** (run on operator VM after tool exits):

```bash
python3 -c "
import orbax.checkpoint as ocp
from etils import epath
STACKED = 'gs://jingnw-mimo-v2-flash-us-east5/mimo-v2-flash-4phase-stacked'
ckpt = ocp.PyTreeCheckpointer()
meta = ckpt.metadata(epath.Path(STACKED))
target_keys = ['decoder.layers_b', 'decoder.layers_c.layers_0', 'decoder.layers_d.mlp.wo']
for k, v in sorted(meta.items()):
    if any(t in k for t in target_keys):
        print(k, v.shape)
"
```

Expected:
- `decoder.layers_b.*`: leading dim `4`
- `decoder.layers_c.layers_*.*`: leading dim `7`
- `decoder.layers_d.mlp.wo`: shape `(I, H)` with no transposition (matches `scan_layers=false`)

**Check 2 — Load test**: run `decode.py` with the stacked checkpoint and
`scan_layers=true`; must load without `ValueError: Requested shape ... not compatible`.

**Check 3 — First generate step completes**: at least one token produced without
shape error or OOM.

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

#### Key adaptation work

`MiMoV2FlashSparseMoeBlock` is NNX-based while `MoeBlock` is Linen.  The gmm
call itself is framework-agnostic; the adaptation involves:

- Plugging in `top_k_indices` and `top_k_weights` from `MiMoV2FlashMoEGate` as
  the `selected_experts` / `weights` fed to the permute/unpermute helpers.
- Setting `group_sizes = jnp.bincount(top_k_indices.ravel(), length=self.num_experts)`
  scoped to the local EP shard's expert range.
- Calling `mblx.gmm` with `preferred_element_type=self.config.dtype` and
  `tiling` tuned for v6e (start with `(128, 128, 128)`).
- `jax.lax.psum_scatter` or `jax.lax.psum` over the EP axis for the output
  all-reduce (same as the existing dense einsum's implicit EP reduction).

#### Why this is safe with `scan_layers=True`

`mblx.gmm` materializes only one layer's worth of temporaries at a time.  With
the 4-phase scan body (6-layer cycle, compiled once and looped 7×), peak HLO
temp stays at ~3 GB vs. ~22 GB for the unrolled gather approach.

### Implementation files to change

| File | Change |
|---|---|
| `src/maxtext/models/mimo_v2_flash.py` | Replace dense einsums in `MiMoV2FlashSparseMoeBlock.__call__` with `permute → mblx.gmm × 3 → unpermute` |
| `src/maxtext/models/mimo_v2_flash.py` | Import `from maxtext.kernels import megablox as mblx` |
| `src/maxtext/models/mimo_v2_flash.py` | Import sort helpers from `moe.py` or inline `jnp.argsort`-based permute |

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
