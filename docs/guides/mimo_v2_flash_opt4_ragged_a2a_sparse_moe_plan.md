# MiMo-V2-Flash Optimization #4 — Ragged All-to-All Sparse MoE Dispatch: Plan & Exit Criteria

## ⛔ POST-MORTEM: Optimization Reverted (2026-04-20)

> **Result**: 101.5 ms median (83% regression from 55.5 ms baseline).
> **Decision**: Reverted to dense dispatch. Code removed from `mimo_v2_flash.py`.

See [Post-Mortem](#post-mortem--why-opt4-failed) at the end of this document for the
full analysis.

---

## Overview

| Step | Description | Status |
|---|---|---|
| **1** | Verify 4-phase stacked checkpoint and scan infrastructure | ✅ Done (from opt2) |
| **2** | Inline `ragged_all_to_all` EP-routing + `gmm` in `MiMoV2FlashSparseMoeBlock` | ⛔ Reverted |
| **3** | Benchmark `scan_layers=true` + ragged-A2A dispatch | ⛔ 101.5 ms (target was ≤ 40 ms) |

**Expected outcome**: step latency ~32.8 ms (−41% from 55.5 ms dense no-scan baseline).
**Actual outcome**: 101.5 ms (+83% regression).

---

## Background

### Why opt2 barely helped

Opt #2 replaced the dense einsums with a local `permute → mblx.gmm → unpermute` pattern
**without cross-EP token routing**.  Each shard still received all T tokens and computed
against its E_local = 32 local experts.  The intermediates shrank slightly but stayed at
`(T, I)` shape per expert slot; the overall HBM bandwidth did not improve.
Result: **56.1 ms** (scan + local gmm) vs **55.5 ms** (dense no-scan) — essentially equal.

### Why ragged_all_to_all is fundamentally different

`jax.lax.ragged_all_to_all` routes tokens **only to the EP shards that own their selected
experts**, before any matmul begins.  After routing, each shard only holds the tokens that
actually selected one of its local experts.

Configuration numbers (v6e-32, decode step):

| Symbol | Value | Source |
|---|---|---|
| T (tokens per step) | 20 480 | batch=32 × max_target_length=640 |
| E_total | 256 | `num_experts` |
| K (top-k per token) | 8 | `num_experts_per_tok` |
| EP | 8 | `ici_expert_parallelism` |
| E_local = E_total / EP | 32 | |
| Expected experts selected per token per shard | K × E_local / E_total = **1** | |

With EP = 8 and K = 8, each token selects exactly 1 expert on each shard on average.
Every token ends up on **exactly one shard** after routing.

#### HBM temporary size comparison (per layer, bfloat16)

| Dispatch | Shape | Size |
|---|---|---|
| Dense einsum (current) | `(E_local=32, T=20480, I=2048)` | **2.68 GB** |
| Local gmm only (opt2) | `(T×K/E_local=5120, I=2048)` × gmm | ~0.08 GB, but still all-T input |
| **ragged_all_to_all** | `(T×K/EP=20480, I=2048)` after routing | **0.08 GB** ← 32× smaller |

The dense approach must read/write ~2.68 GB × 47 layers ≈ **125 GB** per decode step just
for MoE intermediates.  With ragged-A2A routing the same number drops to ~**3.9 GB**
— a 32× reduction in MoE HBM bandwidth.  Since MoE matmuls are HBM-bandwidth-bound on
TPU v6e, this maps directly to ~32× faster MoE layers.

MoE layers make up ~40–45% of overall step time at current batch sizes; combining the
32× layer speedup with the scan overhead gives the ~41% overall estimate.

### Why ragged_all_to_all instead of shard_map

The earlier `shard_map` experiment (reverted, 160 ms) first all-gathered the full
`(T, H)` token matrix (4096-dim) to assemble per-expert inputs.  That required **two**
all-gather collectives and materialised the full token matrix on every shard.
`ragged_all_to_all` in contrast sends only the tokens that belong to each shard — no
full-matrix all-gather, no 2× memory spike.

### Why scan_layers=true is required

With 47 unrolled layers XLA holds 47 × 2.68 GB ≈ 125 GB of MoE intermediates
simultaneously — 4× over the 31.25 GB HBM limit.  The 4-phase scan body
(6-layer cycle × 7 + 4 singletons) compiles each phase once and loops,
bounding peak HLO temporaries to ≈ 3 GB / layer.

The 4-phase scan infrastructure is already live on the `MiMo-V2-Flash` branch:
- `decoders.py` lines 954–1060: 4-phase MIMO_V2_FLASH scan branch (A/B/C/D)
- Stacked checkpoint: `gs://jingnw-mimo-v2-flash-us-east5/mimo-v2-flash-4phase-stacked/`
- Verified load + generate on 2026-04-11 (commit `f9635502`)

---

## Reference: `RoutedMoE.sparse_matmul` ragged-A2A algorithm

The canonical implementation lives in `src/maxtext/layers/moe.py`.  The core EP-routing
path (batch-sharded branch, `num_expert_parallelism > 1`) is:

```
1. permute(inputs, gate_logits)
   → sorted_tokens (T*K, H), sorted_selected_experts (T*K,),
     group_sizes (E_local,), weights (T*K,)

2. reshaped_group_sizes = group_sizes.reshape(1, E_local)
   all_shards_group_sizes = all_gather(reshaped_group_sizes, axis=batch_axis)  # (EP, E_local)

3. get_all_to_all_params(all_shards_group_sizes, expert_shard_id, EP)
   → input_offsets, send_sizes, output_offsets, recv_sizes

4. x = ragged_all_to_all(sorted_tokens, output_shape,
                         input_offsets, send_sizes,
                         output_offsets, recv_sizes,
                         axis_name="expert")
   # tokens now routed: shard i holds only tokens assigned to its E_local experts

5. all_gather(group_sizes, axis_name="expert")  → global_group_sizes (EP, E_local)

6. local_permute(x, global_group_sizes, local_expert_size, shard_index=expert_shard_id)
   → x re-sorted within shard, local group_sizes

7. gmm(x, wi_0, group_sizes)   → layer_w0  (T_local, I)
   gmm(x, wi_1, group_sizes)   → layer_w1  (T_local, I)
   hidden = silu(layer_w0) * layer_w1
   gmm(hidden, wo, group_sizes) → local_output  (T_local, H)

8. ragged_all_to_all(local_output, ...)  # reverse routing, outputs back to origin shards

9. psum_scatter(output, "expert", scatter_dimension=0, tiled=True)

10. unpermute(output, sorted_selected_experts, weights, batch_size, sequence_length)
    → final output (batch, seq, H)
```

---

## Key difference between RoutedMoE gate and MiMo gate

`RoutedMoE.permute` re-runs its own top-k from raw (logit) gate scores.
`MiMoV2FlashMoEGate` uses **sigmoid + noaux-TC correction bias** and returns
`(top_k_indices, top_k_weights)` directly.  The scoring rule is:

```python
# MiMoV2FlashMoEGate.__call__
scores = jax.nn.sigmoid(jnp.dot(tokens, self.gate_weight.T))  # (T, E)
scores += self.e_score_correction_bias                          # (E,) broadcast
top_k_indices  = jnp.argsort(-scores, axis=-1)[..., :K]        # (T, K)
top_k_weights  = jnp.take_along_axis(scores, top_k_indices, axis=-1)
top_k_weights /= top_k_weights.sum(axis=-1, keepdims=True) + 1e-20
```

The correction bias is added only during **routing selection** (argmax), not in the
returned weights (which are normalised raw sigmoid scores after selection).

**Consequence**: we cannot pass MiMo's gate output directly to `RoutedMoE.permute`
which expects logit-space scores.  We must inline a MiMo-specific permute that uses
`top_k_indices` directly rather than re-running top-k.

### Wrapping RoutedMoE vs. inline implementation

Two approaches are possible:

| Approach | Pros | Cons |
|---|---|---|
| A: Instantiate `RoutedMoE` sub-module | Re-uses all existing infra (gmm dispatch, quant stubs, tiling configs) | Gate interface mismatch; must fake raw logits from sigmoid scores; extra config plumbing |
| **B: Inline the ragged-A2A + gmm pattern** | Clean, no interface mismatch, full control over gate adaptation | ~150 lines of new code in `mimo_v2_flash.py` |

**Recommendation: Approach B (inline).**  `RoutedMoE.sparse_matmul` has many optional
branches (FSDP, tensor-transpose, quantisation, tokamax) that don't apply to this model.
Inlining lets us use only the relevant batch-sharded EP path and the MiMo sigmoid gate.

---

## Implementation Plan

### Pre-condition check (read-only, ~10 min)

Before writing any code confirm:

```bash
# Stacked checkpoint is accessible and has correct shape
gcloud compute tpus tpu-vm ssh jingnw-node --zone=us-east5-b --worker=0 \
  --command='
    source ~/maxtext/maxtext_tpu_venv/bin/activate
    cd ~/maxtext
    python3 - <<EOF
import orbax.checkpoint as ocp
import numpy as np
store = ocp.PyTreeCheckpointer()
tree = store.restore("gs://jingnw-mimo-v2-flash-us-east5/mimo-v2-flash-4phase-stacked/checkpoints/0/items")
# layers_b should have leading dim 4; layers_c.layers_0 should have leading dim 7
import jax
leaves = jax.tree_util.tree_leaves_with_path(tree)
for path, leaf in leaves:
    if "layers_b" in str(path) and "wi_0" in str(path):
        print(f"{path}: {leaf.shape}")
        break
    if "layers_c" in str(path) and "layers_0" in str(path) and "wi_0" in str(path):
        print(f"{path}: {leaf.shape}")
        break
EOF
  '
```

Expected: `layers_b.*wi_0.value: (4, 32, 4096, 2048)`, `layers_c.layers_0.*wi_0.value: (7, 32, 4096, 2048)`.

---

### Step 1 — Add `_mimo_a2a_permute` helper (~30 min)

**File**: `src/maxtext/models/mimo_v2_flash.py`

Add a module-level helper below the existing imports.  This replaces the
`permute → all_gather → get_all_to_all_params → ragged_all_to_all → local_permute`
sequence from `RoutedMoE.sparse_matmul`, adapted for MiMo's `(top_k_indices, top_k_weights)`
gate output.

```python
def _mimo_a2a_permute(
    tokens,           # (T, H) — flat token matrix (batch×seq already merged)
    top_k_indices,    # (T, K) — global expert indices (0..E_total-1)
    top_k_weights,    # (T, K) — normalised sigmoid routing weights
    num_experts,      # E_total = 256
    num_experts_per_tok,  # K = 8
    ep_axis_name,     # "expert"
    ep_size,          # 8
):
    """Permute tokens for ragged_all_to_all dispatch.

    Returns:
      sorted_tokens:        (T*K, H) tokens sorted by global expert index
      flat_top_k_indices:   (T*K,) corresponding expert indices
      flat_top_k_weights:   (T*K,) corresponding routing weights
      group_sizes:          (E_local,) token count per local expert
      sort_order:           (T*K,) argsort indices (for unpermute)
      all_shards_group_sizes: (EP, E_local) gathered group sizes
      expert_shard_id:      scalar shard index on the "expert" axis
    """
    T, K = top_k_indices.shape
    E_local = num_experts // ep_size

    # Replicate tokens K times and sort by expert assignment
    tokens_rep = jnp.repeat(tokens, K, axis=0)           # (T*K, H)
    flat_indices = top_k_indices.ravel()                  # (T*K,)
    flat_weights = top_k_weights.ravel()                  # (T*K,)
    sort_order = jnp.argsort(flat_indices, stable=True)   # (T*K,)

    sorted_tokens  = tokens_rep[sort_order]               # (T*K, H)
    sorted_indices = flat_indices[sort_order]             # (T*K,)
    sorted_weights = flat_weights[sort_order]             # (T*K,)

    # group_sizes: how many token-expert pairs land on each of E_local experts
    # on this shard (we count based on global expert id, using only the portion
    # that belongs to this shard — done after ragged_all_to_all)
    group_sizes_global = jnp.bincount(flat_indices, length=num_experts)  # (E_total,)

    # all_gather group_sizes so every shard knows the token count per expert
    # shape after gather: (EP, E_total) → we will slice E_local per shard inside
    # ragged_all_to_all params helper
    # Reshape to (1, E_total) before gather so axis 0 picks up EP entries
    reshaped_gs = group_sizes_global.reshape(1, num_experts)              # (1, E_total)
    all_shards_gs = jax.lax.all_gather(reshaped_gs, axis_name=ep_axis_name)  # (EP, 1, E_total)
    all_shards_gs = all_shards_gs[:, 0, :]                                # (EP, E_total)

    # Shard-local expert range
    expert_shard_id = jax.lax.axis_index(ep_axis_name)
    # Slice to (EP, E_local): each row i holds token counts for experts [i*E_local..(i+1)*E_local)
    all_shards_local_gs = jax.lax.dynamic_slice(
        all_shards_gs, [0, expert_shard_id * E_local], [ep_size, E_local]
    )  # (EP, E_local)

    local_gs = all_shards_local_gs[expert_shard_id]  # (E_local,)

    return (
        sorted_tokens,
        sorted_indices,
        sorted_weights,
        local_gs,
        sort_order,
        all_shards_local_gs,
        expert_shard_id,
    )
```

> **Note on `get_all_to_all_params` reuse**: `RoutedMoE.get_all_to_all_params` is a
> `@staticmethod` and can be called directly as
> `RoutedMoE.get_all_to_all_params(all_shards_local_gs, expert_shard_id, ep_size)`.
> Import `RoutedMoE` from `maxtext.layers.moe` to reuse this helper rather than
> re-implementing it.

---

### Step 2 — Replace dense einsums in `MiMoV2FlashSparseMoeBlock.__call__` (~60 min)

**File**: `src/maxtext/models/mimo_v2_flash.py`

#### 2a — Add imports

```python
from maxtext.layers import moe as moe_lib          # for RoutedMoE.get_all_to_all_params
from maxtext.kernels import megablox as mblx        # already present if opt2 tried it
```

#### 2b — Replace the four dense einsums with the ragged-A2A + gmm pattern

Current code to replace (in `MiMoV2FlashSparseMoeBlock.__call__`):

```python
tokens_fp = tokens.astype(self.config.dtype)
gate = jax.nn.silu(jnp.einsum("th,ehi->eti", tokens_fp, wi_0, ...))  # (E, T, I)
up   = jnp.einsum("th,ehi->eti", tokens_fp, wi_1, ...)               # (E, T, I)
down = jnp.einsum("eti,eih->eth", gate * up, wo, ...)                 # (E, T, H)
output = jnp.einsum("te,eth->th", dispatch_weights.astype(...), down, ...)  # (T, H)
```

Replace with:

```python
cfg = self.config
ep_axis = "expert"
ep_size  = self.mesh.shape.get(ep_axis, 1)   # 8 for ici_expert_parallelism=8
E_local  = self.num_experts // ep_size        # 32

tokens_fp = tokens.astype(cfg.dtype)

# ── 1. Permute + all-gather group sizes ──────────────────────────────────────
(
    sorted_tokens,        # (T*K, H)
    sorted_indices,       # (T*K,)  global expert ids
    sorted_weights,       # (T*K,)
    local_gs,             # (E_local,)  group sizes on this shard before routing
    sort_order,           # (T*K,)
    all_shards_local_gs,  # (EP, E_local)
    expert_shard_id,      # scalar
) = _mimo_a2a_permute(
    tokens_fp, top_k_indices, top_k_weights,
    self.num_experts, self.num_experts_per_tok,
    ep_axis, ep_size,
)

# ── 2. Compute ragged_all_to_all routing parameters ──────────────────────────
input_offsets, send_sizes, output_offsets, recv_sizes = \
    moe_lib.RoutedMoE.get_all_to_all_params(
        all_shards_local_gs, expert_shard_id, ep_size
    )

T_local_max = sorted_tokens.shape[0]  # upper bound; actual is sum(recv_sizes)
output_shape = (T_local_max, self.hidden_size)

# ── 3. Route tokens to owning EP shard (forward) ─────────────────────────────
x = jax.lax.ragged_all_to_all(
    sorted_tokens,
    output_shape,
    input_offsets, send_sizes, output_offsets, recv_sizes,
    axis_name=ep_axis,
)  # (T_local, H) — only tokens for this shard's experts

# ── 4. local_permute within shard ────────────────────────────────────────────
global_group_sizes = jax.lax.all_gather(local_gs, axis_name=ep_axis)  # (EP, E_local)
x, local_sorted_indices, group_sizes, _ = moe_lib.RoutedMoE.local_permute(
    x,
    global_group_sizes[None, :],  # (1, EP, E_local)
    local_expert_size=E_local,
    shard_index=expert_shard_id,
    is_offset=True,
    global_sorted_experts=sorted_indices,
    use_custom_sort_vjp=cfg.use_custom_sort_vjp
        if hasattr(cfg, "use_custom_sort_vjp") else True,
)  # x: (T_local, H), group_sizes: (E_local,)

# ── 5. Grouped matmul (SwiGLU) ───────────────────────────────────────────────
wi_0_local = wi_0[expert_shard_id * E_local : (expert_shard_id + 1) * E_local]  # (E_local, H, I)
wi_1_local = wi_1[expert_shard_id * E_local : (expert_shard_id + 1) * E_local]
wo_local   = wo  [expert_shard_id * E_local : (expert_shard_id + 1) * E_local]  # (E_local, I, H)

TILE = (128, 128, 128)
g = mblx.gmm(x, wi_0_local, group_sizes=group_sizes,
              preferred_element_type=cfg.dtype, tiling=TILE)  # (T_local, I)
u = mblx.gmm(x, wi_1_local, group_sizes=group_sizes,
              preferred_element_type=cfg.dtype, tiling=TILE)  # (T_local, I)
h = jax.nn.silu(g) * u                                         # (T_local, I)
local_output = mblx.gmm(h, wo_local, group_sizes=group_sizes,
                         preferred_element_type=cfg.dtype, tiling=TILE)  # (T_local, H)

# ── 6. Reverse ragged_all_to_all (outputs back to origin shards) ──────────────
rev_input_offsets, rev_send_sizes, rev_output_offsets, rev_recv_sizes = \
    moe_lib.RoutedMoE.get_all_to_all_params(
        jnp.transpose(all_shards_local_gs),
        expert_shard_id, ep_size,
    )
rev_output_shape = (sorted_tokens.shape[0], self.hidden_size)
intermediate = jax.lax.ragged_all_to_all(
    local_output,
    rev_output_shape,
    rev_input_offsets, rev_send_sizes, rev_output_offsets, rev_recv_sizes,
    axis_name=ep_axis,
)  # (T*K, H) — each token gets its contribution back

# ── 7. Reduce across expert axis + unpermute ──────────────────────────────────
intermediate = jnp.reshape(intermediate, (-1, T, self.hidden_size))
intermediate = jax.lax.psum_scatter(
    intermediate, ep_axis, scatter_dimension=0, tiled=True,
)  # (T/EP, H) — each data-parallel shard owns a slice

# Reconstruct full output via unpermute + weighted sum over K
# Inverse sort
unsorted = intermediate[jnp.argsort(sort_order)]         # (T*K, H)
output = (
    unsorted.reshape(T, self.num_experts_per_tok, self.hidden_size)
    * sorted_weights.reshape(T, self.num_experts_per_tok, 1).astype(cfg.dtype)
).sum(axis=1)                                             # (T, H)
```

> **Caution**: The `psum_scatter` step reduces partial outputs from different EP shards.
> Shape management here is the trickiest part — verify dimensions carefully with
> `jax.debug.print` before removing it.  See [Debugging tips](#debugging-tips) below.

#### 2c — Clean up now-unused code

Remove the `dispatch_weights` scatter/gather construction (lines `jnp.zeros((T, E))...`
and `dispatch_weights.at[...]`) since it is no longer used.

---

### Step 3 — Verify `MiMoV2FlashScannableBlockToLinen` wiring (~15 min)

`MiMoV2FlashScannableBlockToLinen` is currently defined as:

```python
MiMoV2FlashScannableBlockToLinen = nnx_wrappers.to_linen_class(
    MiMoV2FlashDecoderLayer,          # ← wraps the same MiMoV2FlashDecoderLayer
    base_metadata_fn=variable_to_logically_partitioned,
)
```

This is **correct** — `decoders.py` line 486 routes `scan_layers=True` to
`MiMoV2FlashScannableBlockToLinen`.  The scan machinery in the 4-phase branch
(`decoders.py` 954–1060) uses this wrapper and calls `MiMoV2FlashDecoderLayer.__call__`
through it.  No change needed here.

---

### Step 4 — Smoke test (no benchmark) (~20 min)

Run the quick demo on the **stacked checkpoint** with `scan_layers=true`:

```bash
gcloud compute tpus tpu-vm ssh jingnw-node --zone=us-east5-b --worker=all \
  --command='
    export PATH="$HOME/.local/bin:$PATH"
    source ~/maxtext/maxtext_tpu_venv/bin/activate
    cd ~/maxtext
    python3 -m maxtext.demos.mimo_v2_flash_demo_jax \
      src/maxtext/configs/base.yml model_name=mimo-v2-flash \
      load_parameters_path=gs://jingnw-mimo-v2-flash-us-east5/mimo-v2-flash-4phase-stacked/checkpoints/0/items \
      scan_layers=true \
      ici_tensor_parallelism=4 ici_expert_parallelism=8 \
      per_device_batch_size=1 max_target_length=640 \
      attention=dot_product dtype=bfloat16 \
      2>&1 | tail -5
  ' 2>&1 | grep -v "^[IW][0-9]"
```

Expected: last worker prints `"420 km"` or equivalent correct answer and EOS fires.
If `ValueError` on checkpoint restore → stacked checkpoint shape regression (see
[Rollback](#rollback)).

---

### Step 5 — Benchmark `scan_layers=true` + ragged-A2A (~30 min)

Same as the `scan_layers=false` benchmark but with `scan_layers=true` and the stacked checkpoint:

```bash
gcloud compute tpus tpu-vm ssh jingnw-node --zone=us-east5-b --worker=all \
  --command='
    export PATH="$HOME/.local/bin:$PATH"
    source ~/maxtext/maxtext_tpu_venv/bin/activate
    cd ~/maxtext
    python3 -m maxtext.benchmarks.benchmark_runner \
      src/maxtext/configs/base.yml model_name=mimo-v2-flash \
      load_parameters_path=gs://jingnw-mimo-v2-flash-us-east5/mimo-v2-flash-4phase-stacked/checkpoints/0/items \
      scan_layers=true \
      ici_tensor_parallelism=4 ici_expert_parallelism=8 \
      per_device_batch_size=1 max_target_length=640 \
      attention=dot_product dtype=bfloat16 \
      bench_warmup_steps=5 bench_timed_steps=50 \
      2>&1 | tee /tmp/opt4_bench.log
  ' 2>&1 | grep -v "^[IW][0-9]"
```

---

## Exit Criteria

| Criterion | Required | Pass if |
|---|---|---|
| Smoke test | ✅ | EOS fires; correct answer ("420 km") |
| No quality regression | ✅ | Demo answer matches dense no-scan baseline |
| Median step latency | ✅ | < 55.5 ms (any improvement over dense no-scan) |
| **Target latency** | 🎯 | ≤ 40 ms (>28% improvement) |
| Throughput | 🎯 | > 640 tok/s |
| No HBM OOM | ✅ | All 8 workers complete; no `ResourceExhaustedError` |
| Compile time reasonable | 🟡 | First compile ≤ 5 min per worker |

---

## Debugging Tips

### Shape verification stubs

Insert these immediately after each ragged_all_to_all call during development
(remove before final benchmark):

```python
jax.debug.print("sorted_tokens shape: {}", sorted_tokens.shape)
jax.debug.print("x after a2a fwd: {}", x.shape)
jax.debug.print("local_output shape: {}", local_output.shape)
jax.debug.print("intermediate after a2a rev: {}", intermediate.shape)
jax.debug.print("output final: {}", output.shape)
```

### Common failure modes

| Symptom | Likely cause | Fix |
|---|---|---|
| `ValueError: shape mismatch` in `ragged_all_to_all` | `output_shape` buffer too small | Set buffer to `(T*K, H)` (upper bound) not `recv_sizes.sum()` |
| `XLA compilation error: non-static shape` | `group_sizes` passed as dynamic array to `mblx.gmm` | Ensure `group_sizes` is marked static via `jax.lax.with_sharding_constraint` or pass as 1-D int32 array |
| NaN outputs | Weight slicing `wi_0[shard_id*E_local:]` wrong axis | Double check expert dim is axis 0; `wi_0.shape == (E_total, H, I)` |
| Correct shape but wrong values | `sort_order` / `local_sorted_indices` mismatch in unpermute | Use `local_sorted_indices` (from `local_permute`) not original `sort_order` for the inner unsort |
| `ResourceExhaustedError` with scan=true | 4-phase stacked checkpoint not loaded (`scan_layers=false` checkpoint lacks stacked dim) | Confirm `load_parameters_path` points to `mimo-v2-flash-4phase-stacked` |
| 160 ms regression | ragged_all_to_all replaced by all_gather path (fallback) | Check `ep_size > 1`; confirm `ep_axis` name matches `_expert_parallelism_name = "expert"` |

### Verifying EP axis name

`RoutedMoE._expert_parallelism_name` is always `"expert"`.  In MaxText mesh
configuration `ici_expert_parallelism=8` maps to the `"expert"` logical axis.
Confirm with:

```python
print(self.mesh.axis_names)   # should include "expert"
print(jax.lax.axis_index("expert"))   # should be 0..7 on each shard
```

---

## Rollback

If the ragged-A2A implementation introduces correctness or stability issues, revert to the dense dispatch (no-scan baseline, commit `72f75972`) with:

```bash
git stash   # or git checkout src/maxtext/models/mimo_v2_flash.py
```

The `scan_layers=false` benchmark with dense dispatch is the stable baseline:
- Median: 55.5 ms, 576.8 tok/s (benchmarked 2026-04-17, tag `mimo-v2-flash-dense-ok-scan-bench-only`)

---

## Relationship to Other Optimizations

| Opt | Description | Status | Interaction |
|---|---|---|---|
| #1 | Remove `jax.debug.print` | ✅ Done | None |
| #2 | Local sparse gmm (mblx.gmm, no EP routing) | ✅ Done but minimal gain | This opt supersedes it |
| #3 | Int8 KV cache | Rejected (complexity vs gain) | Independent |
| **#4** | **Ragged-A2A sparse EP routing (this plan)** | **⛔ Reverted (101.5 ms)** | See post-mortem below |
| TBD | SWA window truncation | Deferred | Can be stacked on top of baseline |
| TBD | Flash attention for prefill | Deferred | Independent |

---

## Post-Mortem — Why Opt4 Failed

### Summary

The ragged-A2A `shard_map` implementation was completed and benchmarked on
2026-04-20 across all 8 workers of `jingnw-node` (v6e-32).  The MoE-trace
diagnostic confirmed `ep_size=8, tp_size=4` — the EP routing path was executing
correctly.  However the benchmark showed **101.5 ms median** vs the 55.5 ms
baseline — an 83% regression, not an improvement.

### Root cause: Wrong T assumption in the plan's analysis

The plan assumed **T = 20,480 tokens per step** (= batch × max_target_length =
32 × 640).  This was **wrong**.

The benchmark measures **autoregressive decode** steps via `engine.generate()`.
Each AR decode step processes **T = batch × 1 = 32 tokens** (one new token per
sequence in the batch).  The value T = 20,480 would only apply if the entire
sequence were processed in a single step (i.e., prefill), not during generation.

### Why sparse routing hurts for decode (T = 32)

For AR decode with T = 32, batch = 32, EP = 8:

| | Dense dispatch | Sparse ragged-A2A |
|---|---|---|
| Tokens per MoE layer per device | 32 (all tokens, all experts) | 32 after routing (same!) |
| Expert weight reads | All E_local=32 experts | All E_local=32 experts (same!) |
| ICI collectives per MoE layer | 0 extra | +4 (ragged_all_to_all ×2, all_gather, psum_scatter) |
| MoE intermediate HBM | ~128 KB (trivial for T=32) | ~128 KB (same) |

With T = 32 tokens, each of the 32 local experts gets ~1 token on average.
Sparse routing cannot reduce the work — all expert weights must be read regardless,
and the token-expert assignment is 1:1 at this scale.  The ragged-A2A collectives
add ~4 ICI calls per MoE layer × 47 layers ≈ **188 extra collectives per step**,
contributing approximately **46 ms of pure overhead**.

### Why the plan's HBM analysis was right for prefill but wrong for decode

The plan's core claim was correct **for large T**:

> "Dense einsum: (E_local=32, T=20480, I=2048) = 2.68 GB per layer"
> "ragged_all_to_all: 0.08 GB per layer → 32× smaller intermediates"

With T = 20,480 (prefill), MoE intermediates dominate HBM bandwidth and sparse
routing provides a 32× reduction.  With T = 32 (decode), MoE intermediates are
128 KB per layer — the bottleneck is **weight reads** (~128 MB per device per
layer), which are identical for both approaches.

### What could help instead

For AR decode latency (weight-bandwidth-bound):
- **FP8/int8 weight quantisation** — halves weight reads → potentially halves MoE time
- **More EP** — each device reads fewer expert weights (need more devices)
- **SWA window truncation** — reduces attention KV cache reads for sliding-window layers

For **prefill** throughput (where T is large and sparse routing helps):
- The ragged-A2A approach IS valid and could provide significant speedup
- The benchmark script now includes a separate prefill benchmark phase
- Future work should target prefill latency with `engine.prefill()`, not decode

### Benchmark data

**Cluster**: `jingnw-node`, v6e-32 (8 workers × 4 chips), `us-east5-b`

| Configuration | Decode Median | Decode Throughput | Prefill Median (512 tok) | Prefill Throughput |
|---|---|---|---|---|
| Dense no-scan baseline (pre-opt4) | 55.5 ms | 576.8 tok/s | — | — |
| Dense scan=true baseline (pre-opt4) | 68.3 ms | 468 tok/s | — | — |
| Local gmm only (opt2, scan=true) | 56.1 ms | 570 tok/s | — | — |
| **Ragged-A2A shard_map (opt4)** | **101.5 ms** | **315.5 tok/s** | — | — |
| **Dense scan=true (opt4 reverted)** | **68.4 ms** | **468 tok/s** | **121.9 ms** | **4,200 tok/s** |

### Lessons learned

1. **Always verify T per step before designing MoE optimisations.**
   For AR decode, T = batch × 1 (not batch × seq_len).  The seq_len dimension is
   only the decode step size, which is 1 during generation.

2. **AR decode is weight-bandwidth-bound, not compute/temporary-bound.**
   With small batch and T = 32, MoE intermediates are negligible.  The dominant
   cost is reading 1.5+ GB of expert weights per device per step (47 MoE layers ×
   3 projections × 32 experts × H × I × sizeof(bf16)).

3. **Collective overhead matters at small T.**
   Each `ragged_all_to_all` has a fixed latency (~50–100 μs on ICI).  At 188
   collectives per step, this overhead accounts for most of the 46 ms regression.

4. **Profile before and after, using both prefill AND decode benchmarks.**
   The bench script now runs both phases.  Future optimisations should report
   results for both.

---

## Appendix: Arithmetic Check

**Dense dispatch FLOPS per layer** (32 local experts × T tokens):
$$\text{FLOPS} = 2 \times T \times H \times I \times E_{local} \times 3 = 2 \times 20480 \times 4096 \times 2048 \times 32 \times 3 \approx 1.64 \text{ TFLOPS}$$

**Sparse dispatch FLOPS per layer** (K×T/EP effective tokens, since K×E_local/E_total = 1):
$$\text{FLOPS} = 2 \times \frac{K \times T}{E_P} \times H \times I \times 3 = 2 \times \frac{8 \times 20480}{8} \times 4096 \times 2048 \times 3 = 2 \times 20480 \times 4096 \times 2048 \times 3 \approx 0.051 \text{ TFLOPS}$$

FLOPS are the same — this confirms the speedup is from **HBM bandwidth**, not FLOPS reduction.
Dense creates (E_local × T × I) = 32× larger intermediates; ragged-A2A eliminates them.
