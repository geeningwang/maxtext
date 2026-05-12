# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Pallas kernel: fused FP8 block-dequant + matmul for MiMo-V2-Flash MoE.

This kernel replaces the two-step Phase A approach:
    wi_0 = block_dequant_fp8(...)          # materialises full BF16 in HBM
    out  = einsum("th,ehi->eti", x, wi_0)

with a single kernel that:
    1. Loads a (128, 128) FP8 weight tile from HBM into VMEM.
    2. Multiplies by its per-block float32 scale_inv → BF16 in VMEM.
    3. Dots with the corresponding token slice → accumulates in float32 VMEM.
    4. Writes the BF16 output to HBM on the final contraction tile only.

The full BF16 weight tensor is NEVER written to HBM.  Weight memory footprint
drops ~50% (float8 vs bfloat16) and weight-read bandwidth drops ~50% per
decode step — the primary bottleneck for MoE at batch=1.

Supported einsum patterns
--------------------------
- ``"th,ehi->eti"`` : tokens (T, H), weight (E, H, I) → output (E, T, I)
  Used for gate-proj (wi_0) and up-proj (wi_1) in MiMoV2FlashSparseMoeBlock.
  Call: ``fp8_moe_matmul(tokens, weight_fp8, scale_inv)``

- ``"eti,eih->eth"`` : tokens (E, T, I), weight (E, I, H) → output (E, T, H)
  Used for down-proj (wo) in MiMoV2FlashSparseMoeBlock.  Same kernel with
  ``tokens_batched=True``; token block spec includes the expert (E) axis.
  Call: ``fp8_moe_matmul(tokens, weight_fp8, scale_inv, tokens_batched=True)``

Limitations (Phase B v1)
--------------------------
- Token dimension T must be ≤ available VMEM (safe for T ≤ 512 at tile_n=128).
  At decode batch=1, T=1; VMEM scratch is 128*4=512 bytes — trivially fine.
  For prefill (large T), fall back to Phase A ``_block_dequant_fp8`` + einsum.
- ``tile_k`` and ``tile_n`` must divide K (hidden or intermediate) and N exactly.
- ``interpret=True`` runs on CPU without Pallas compilation (for unit tests).
"""

import functools

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu


# ---------------------------------------------------------------------------
# Kernel body
# ---------------------------------------------------------------------------

def _fp8_moe_kernel_body(
    tokens_ref,       # (T, tile_k)  bf16 — current K-chunk of tokens
    weight_fp8_ref,   # (1, tile_k, tile_n)  float8_e4m3fn — weight tile
    scale_ref,        # (1, 1, 1)  float32 — per-block scale_inv scalar
    out_ref,          # (1, T, tile_n)  bf16 — output tile (written on last k)
    acc_ref,          # VMEM scratch (T, tile_n)  float32
    *,
    n_tiles: int,     # total number of N-output tiles (for K_tiles calc)
    k_tiles: int,     # total number of K-contraction tiles
):
    """Pallas kernel body for one (e, n_tile, k_tile) grid instance."""
    del n_tiles  # unused inside kernel body; passed for grid bookkeeping only
    k_i = pl.program_id(2)

    # Zero the accumulator on the first contraction tile.
    @pl.when(k_i == 0)
    def _zero():
        acc_ref[...] = jnp.zeros_like(acc_ref)

    # Dequantize FP8 weight tile in VMEM: never written back to HBM.
    w_fp8 = weight_fp8_ref[0, :, :]                         # (tile_k, tile_n) fp8
    scale = scale_ref[0, 0, 0]                              # () float32
    w_bf16 = (w_fp8.astype(jnp.float32) * scale).astype(jnp.bfloat16)

    # Accumulate: (T, tile_k) @ (tile_k, tile_n) → (T, tile_n) float32.
    acc_ref[...] += jnp.dot(
        tokens_ref[...],                                     # (T, tile_k) bf16
        w_bf16,                                              # (tile_k, tile_n) bf16
        preferred_element_type=jnp.float32,
    )

    # Write output only on the final K tile (contraction complete).
    @pl.when(k_i == k_tiles - 1)
    def _store():
        out_ref[0, :, :] = acc_ref[...].astype(jnp.bfloat16)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

@functools.partial(jax.jit, static_argnames=("tokens_batched", "tile_k", "tile_n", "interpret"))
def fp8_moe_matmul(
    tokens: jax.Array,        # (T, K) bf16  or  (E, T, K) bf16 if tokens_batched
    weight_fp8: jax.Array,    # (E, K, N) float8_e4m3fn
    scale_inv: jax.Array,     # (E, K//tile_k, N//tile_n) float32
    tokens_batched: bool = False,
    tile_k: int = 128,
    tile_n: int = 128,
    interpret: bool = False,
) -> jax.Array:               # (E, T, N) bfloat16
    """Fused FP8 block-dequant + batched matmul via Pallas.

    Equivalent to (but faster than) the Phase A approach:
        w_bf16 = _block_dequant_fp8(weight_fp8, scale_inv, tile_k, tile_n)
        # "th,ehi->eti" or "eti,eih->eth" depending on tokens_batched

    Args:
        tokens: Token activations.  Shape (T, K) for gate/up projections;
            (E, T, K) for the down projection.
        weight_fp8: Expert weights in float8_e4m3fn, shape (E, K, N).
        scale_inv: Per-block float32 scale factors, shape (E, K//tile_k, N//tile_n).
        tokens_batched: True when tokens is (E, T, K) (down-proj case).
        tile_k: Contraction axis tile size (must divide K); default 128.
        tile_n: Output axis tile size (must divide N); default 128.
        interpret: Run in Pallas interpret mode (CPU-compatible, for tests).

    Returns:
        Output array of shape (E, T, N) in bfloat16.
    """
    if tokens_batched:
        E, T, K = tokens.shape
    else:
        T, K = tokens.shape
        E = weight_fp8.shape[0]
    _, _, N = weight_fp8.shape

    k_tiles = K // tile_k
    n_tiles = N // tile_n

    if K % tile_k != 0:
        raise ValueError(f"K={K} must be divisible by tile_k={tile_k}")
    if N % tile_n != 0:
        raise ValueError(f"N={N} must be divisible by tile_n={tile_n}")

    # Build index maps.  Grid axes: (e, n_t, k_t).
    def _tok_idx(e, n_t, k_t):
        if tokens_batched:
            return (e, 0, k_t)      # tokens[e, :, k_t*tile_k:(k_t+1)*tile_k]
        else:
            return (0, k_t)         # tokens[:, k_t*tile_k:(k_t+1)*tile_k]

    def _wt_idx(e, n_t, k_t):
        return (e, k_t, n_t)        # weight[e, k_t*tile_k:..., n_t*tile_n:...]

    def _sc_idx(e, n_t, k_t):
        return (e, k_t, n_t)        # scale[e, k_t, n_t]

    def _out_idx(e, n_t, k_t):
        del k_t                     # output doesn't depend on k_t
        return (e, 0, n_t)          # out[e, :, n_t*tile_n:(n_t+1)*tile_n]

    # Build block specs.
    if tokens_batched:
        tok_block = pl.BlockSpec((1, None, tile_k), _tok_idx)
    else:
        tok_block = pl.BlockSpec((None, tile_k), _tok_idx)

    wt_block = pl.BlockSpec((1, tile_k, tile_n), _wt_idx)
    sc_block = pl.BlockSpec((1, 1, 1), _sc_idx)
    out_block = pl.BlockSpec((1, None, tile_n), _out_idx)

    # Capture static values in the kernel closure.
    kernel = functools.partial(
        _fp8_moe_kernel_body,
        n_tiles=n_tiles,
        k_tiles=k_tiles,
    )

    out = pl.pallas_call(
        kernel,
        out_shape=jax.ShapeDtypeStruct((E, T, N), jnp.bfloat16),
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=0,
            in_specs=[tok_block, wt_block, sc_block],
            out_specs=out_block,
            grid=(E, n_tiles, k_tiles),
            scratch_shapes=[pltpu.VMEM((T, tile_n), jnp.float32)],
        ),
        compiler_params=pltpu.CompilerParams(
            # E and N_tiles are parallel (independent experts / output tiles).
            # K_tiles is the sequential contraction axis where scratch is shared.
            dimension_semantics=("parallel", "parallel", "arbitrary"),
        ),
        interpret=interpret,
    )(tokens, weight_fp8, scale_inv)

    return out
