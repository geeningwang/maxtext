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

"""Xiaomi MiMo-V2-Flash model decoder layers for MaxText.

Architecture overview (309B total / 15B active):
  • 48 hybrid decoder layers
  • Hybrid attention: 9 global (full) layers + 39 sliding-window layers
    - Pattern is specified per-layer via ``mimo_hybrid_layer_pattern``
    - 0 = global (full causal), 1 = sliding-window (128-token window)
  • Asymmetric head dims: Q/K head_dim=192, V head_dim=128
  • Partial RoPE: only 33.4 % of the head dimension is rotated
    - ``rope_dim = int(head_dim * partial_rotary_factor) = 64``
  • Separate RoPE theta: 5 000 000 for global attention, 10 000 for SWA
  • Attention sink bias on SWA layers (learnable per-head bias)
  • Almost-all-MoE: layer 0 is dense, layers 1–47 are sparse MoE
    - 256 experts, top-8 routing, sigmoid scoring with noaux-TC correction bias
    - MoE intermediate size: 2048

References:
  • Model card: https://huggingface.co/XiaomiMiMo/MiMo-V2-Flash
  • Technical report: https://github.com/XiaomiMiMo/MiMo-V2-Flash/blob/main/paper.pdf
"""

# pylint: disable=arguments-differ
# pylint: disable=no-name-in-module

from typing import Any, Optional

import jax
import jax.numpy as jnp
from jax import lax

from flax import linen as nn
from flax import nnx
from jax.sharding import Mesh

from maxtext.common.common_types import (
    Config,
    DType,
    Array,
    BATCH,
    LENGTH_NO_EXP,
    EMBED,
)
from maxtext.layers import initializers as max_initializers
from maxtext.layers import nnx_wrappers
from maxtext.layers import quantizations
from maxtext.layers.normalizations import RMSNorm
from maxtext.layers.initializers import variable_to_logically_partitioned
from maxtext.layers.quantizations import AqtQuantization as Quant
from maxtext.layers.linears import DenseGeneral, MlpBlock
from maxtext.layers.embeddings import PartialRotaryEmbedding, LLaMARotaryEmbedding
from maxtext.layers.initializers import nd_dense_init
from maxtext.utils import max_utils


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _repeat_kv(x: Array, n_rep: int) -> Array:
  """Repeat KV heads to match Q heads for grouped-query attention.

  Args:
    x: Shape (batch, seq_len, kv_heads, head_dim).
    n_rep: Number of times to repeat each KV head.

  Returns:
    Shape (batch, seq_len, kv_heads * n_rep, head_dim).
  """
  if n_rep == 1:
    return x
  batch, seq, h, d = x.shape
  x = jnp.broadcast_to(x[:, :, :, jnp.newaxis, :], (batch, seq, h, n_rep, d))
  return x.reshape(batch, seq, h * n_rep, d)


def _make_causal_mask(q_len: int, kv_len: int, dtype: DType) -> Array:
  """Lower-triangular causal mask of shape (1, 1, q_len, kv_len)."""
  mask = jnp.tril(jnp.ones((q_len, kv_len), dtype=jnp.bool_))
  return jnp.where(mask, jnp.zeros((1, 1, q_len, kv_len), dtype=dtype),
                   jnp.full((1, 1, q_len, kv_len), jnp.finfo(dtype).min, dtype=dtype))


def _make_sliding_window_mask(q_len: int, kv_len: int, window: int, dtype: DType) -> Array:
  """Causal + sliding-window mask.

  Each query position attends only to the ``window`` most-recent key positions
  that are ≤ its own position.
  """
  q_idx = jnp.arange(q_len)[:, jnp.newaxis]   # (q_len, 1)
  k_idx = jnp.arange(kv_len)[jnp.newaxis, :]  # (1, kv_len)
  causal = k_idx <= q_idx
  window_ok = k_idx > (q_idx - window)
  mask = causal & window_ok
  return jnp.where(mask, jnp.zeros((1, 1, q_len, kv_len), dtype=dtype),
                   jnp.full((1, 1, q_len, kv_len), jnp.finfo(dtype).min, dtype=dtype))


# ---------------------------------------------------------------------------
# MiMoV2FlashAttention
# ---------------------------------------------------------------------------

class MiMoV2FlashAttention(nnx.Module):
  """MiMo-V2-Flash attention layer.

  Supports both global (full causal) and sliding-window attention as
  determined by ``is_swa``.

  Key differences from standard attention:
  * Q/K head dim (192) ≠ V head dim (128) — output projection operates on
    ``num_q_heads × v_head_dim``, not ``num_q_heads × head_dim``.
  * Partial RoPE: only the first ``rope_dim = int(head_dim * partial_rotary_factor)``
    dimensions of each Q/K head are rotated; the rest are passed through unchanged.
  * Two sets of RoPE parameters: ``rope_theta`` for global layers, ``swa_rope_theta``
    for SWA layers (each is 1 RoPE instance; whichever applies is constructed here).
  * Optional per-head attention sink bias added to logits (SWA layers by default).
  * GQA with ``num_kv_heads`` (4 for GA, 8 for SWA) shared KV heads for 64 Q heads.

  Args:
    config: MaxText configuration object.
    mesh: The device mesh for sharding.
    is_swa: If True, this is a sliding-window attention layer.
    layer_idx: Index of the layer in the transformer stack.
    quant: Optional quantization configuration.
    rngs: PRNG keys for parameter initialisation.
  """

  def __init__(
      self,
      config: Config,
      mesh: Mesh,
      is_swa: bool,
      layer_idx: int,
      quant: None | Quant = None,
      *,
      rngs: nnx.Rngs,
  ):
    self.config = config
    self.mesh = mesh
    self.is_swa = is_swa
    self.layer_idx = layer_idx
    self.quant = quant

    cfg = self.config
    dtype: DType = cfg.dtype
    weight_dtype: DType = cfg.weight_dtype

    self.head_dim: int = cfg.head_dim                         # Q/K head dim = 192
    self.v_head_dim: int = cfg.mimo_v_head_dim if cfg.mimo_v_head_dim > 0 else cfg.head_dim  # V head dim = 128
    self.num_q_heads: int = cfg.num_query_heads               # 64
    self.num_kv_heads: int = (
        cfg.mimo_swa_num_kv_heads if (is_swa and cfg.mimo_swa_num_kv_heads > 0)
        else cfg.num_kv_heads
    )                                                          # 8 (SWA) or 4 (GA)
    self.n_kv_groups: int = self.num_q_heads // self.num_kv_heads

    # Partial RoPE: rotate the first rope_dim dimensions of Q/K.
    self.partial_rotary_factor: float = cfg.partial_rotary_factor
    self.rope_dim: int = int(self.head_dim * self.partial_rotary_factor)

    # Choose RoPE theta according to attention type.
    rope_theta = cfg.mimo_swa_rope_theta if is_swa else cfg.rope_max_timescale

    self.rotary_embedding = PartialRotaryEmbedding(
        min_timescale=1,
        max_timescale=int(rope_theta),
        mesh=mesh,
        embedding_dims=self.head_dim,
        fprop_dtype=dtype,
        partial_rotary_factor=self.partial_rotary_factor,
        rngs=rngs,
    )

    # Per-layer attention scale: 1 / sqrt(head_dim).
    self.attn_scale: float = self.head_dim ** -0.5

    # Optional attention sink bias (learnable, per Q-head).
    add_sink = is_swa  # SWA layers use sink bias; GA layers do not by default
    if add_sink:
      self.sink_bias = nnx.Param(
          jnp.zeros((self.num_q_heads,), dtype=weight_dtype),
      )
    else:
      self.sink_bias = None

    # Q projection: hidden → (num_q_heads, head_dim).
    self.q_proj = DenseGeneral(
        in_features_shape=cfg.emb_dim,
        out_features_shape=(self.num_q_heads, self.head_dim),
        use_bias=False,
        dtype=dtype,
        weight_dtype=weight_dtype,
        kernel_init=nd_dense_init(1.0, "fan_in", "truncated_normal"),
        kernel_axes=("embed", "heads", "kv_head_dim"),
        matmul_precision=cfg.matmul_precision,
        rngs=rngs,
    )

    # K projection: hidden → (num_kv_heads, head_dim).
    self.k_proj = DenseGeneral(
        in_features_shape=cfg.emb_dim,
        out_features_shape=(self.num_kv_heads, self.head_dim),
        use_bias=False,
        dtype=dtype,
        weight_dtype=weight_dtype,
        kernel_init=nd_dense_init(1.0, "fan_in", "truncated_normal"),
        kernel_axes=("embed", "kv_heads", "kv_head_dim"),
        matmul_precision=cfg.matmul_precision,
        rngs=rngs,
    )

    # V projection: hidden → (num_kv_heads, v_head_dim).
    self.v_proj = DenseGeneral(
        in_features_shape=cfg.emb_dim,
        out_features_shape=(self.num_kv_heads, self.v_head_dim),
        use_bias=False,
        dtype=dtype,
        weight_dtype=weight_dtype,
        kernel_init=nd_dense_init(1.0, "fan_in", "truncated_normal"),
        kernel_axes=("embed", "kv_heads", "kv_head_dim"),
        matmul_precision=cfg.matmul_precision,
        rngs=rngs,
    )

    # Output projection: (num_q_heads × v_head_dim) → hidden.
    self.o_proj = DenseGeneral(
        in_features_shape=(self.num_q_heads, self.v_head_dim),
        out_features_shape=cfg.emb_dim,
        axis=(-2, -1),
        use_bias=False,
        dtype=dtype,
        weight_dtype=weight_dtype,
        kernel_init=nd_dense_init(1.0, "fan_in", "truncated_normal"),
        kernel_axes=("heads", "kv_head_dim", "embed"),
        matmul_precision=cfg.matmul_precision,
        rngs=rngs,
    )

  def __call__(
      self,
      inputs: Array,
      decoder_positions: None | Array,
      decoder_segment_ids: None | Array,
      deterministic: bool,
      model_mode: str,
  ) -> Array:
    """Forward pass.

    Args:
      inputs: Shape (batch, seq_len, emb_dim).
      decoder_positions: Integer positions for RoPE, shape (batch, seq_len).
      decoder_segment_ids: Optional segment IDs for packed sequences.
      deterministic: Disables dropout when True.
      model_mode: One of 'train', 'prefill', 'autoregressive'.

    Returns:
      Output tensor of shape (batch, seq_len, emb_dim).
    """
    cfg = self.config
    batch, q_len, _ = inputs.shape

    # Projections ---------------------------------------------------------

    # query: (B, S, H_q, D_qk)
    query = self.q_proj(inputs)
    # key:   (B, S, H_kv, D_qk)
    key = self.k_proj(inputs)
    # value: (B, S, H_kv, D_v)
    value = self.v_proj(inputs)

    # Optional value scaling (attention_value_scale = 0.707 for MiMo).
    if cfg.mimo_attention_value_scale != 1.0:
      value = value * cfg.mimo_attention_value_scale

    # Partial RoPE --------------------------------------------------------
    # Apply separately to Q and K; V is never rotated.
    query = self.rotary_embedding(query, decoder_positions)
    key = self.rotary_embedding(key, decoder_positions)

    # GQA: repeat K/V to match Q heads ------------------------------------
    key_full = _repeat_kv(key, self.n_kv_groups)     # (B, S, H_q, D_qk)
    value_full = _repeat_kv(value, self.n_kv_groups)  # (B, S, H_q, D_v)

    # Attention computation -----------------------------------------------
    # Transpose to (B, H, S, D) layout for einsum.
    q = jnp.transpose(query, (0, 2, 1, 3))       # (B, H_q, S, D_qk)
    k = jnp.transpose(key_full, (0, 2, 1, 3))   # (B, H_q, S, D_qk)
    v = jnp.transpose(value_full, (0, 2, 1, 3)) # (B, H_q, S, D_v)

    # Scaled dot-product: (B, H, S_q, S_k)
    attn_weights = jnp.einsum("bHqd,bHkd->bHqk", q, k, precision=lax.Precision.DEFAULT)
    attn_weights = attn_weights * self.attn_scale

    # Causal (+ sliding-window) mask.
    kv_len = k.shape[2]
    if self.is_swa and cfg.mimo_swa_window_size > 0:
      mask = _make_sliding_window_mask(q_len, kv_len, cfg.mimo_swa_window_size,
                                       attn_weights.dtype)
    else:
      mask = _make_causal_mask(q_len, kv_len, attn_weights.dtype)
    attn_weights = attn_weights + mask

    # Attention sink bias: add a learned scalar per Q-head as an extra "sink"
    # logit *before* softmax so the model can ignore positions it doesn't need.
    if self.sink_bias is not None:
      # sink is shape (H_q,); reshape to (1, H_q, 1, 1) for broadcasting.
      sink = self.sink_bias[...].reshape(1, self.num_q_heads, 1, 1).astype(jnp.float32)
      attn_weights = attn_weights + sink

    # Softmax in float32 for numerical stability.
    attn_weights = jax.nn.softmax(attn_weights.astype(jnp.float32), axis=-1).astype(cfg.dtype)

    # Context: (B, H, S_q, D_v)
    context = jnp.einsum("bHqk,bHkd->bHqd", attn_weights, v, precision=lax.Precision.DEFAULT)

    # Output projection ---------------------------------------------------
    # Transpose back to (B, S, H, D_v) then project.
    context = jnp.transpose(context, (0, 2, 1, 3))  # (B, S, H_q, D_v)
    output = self.o_proj(context)                     # (B, S, emb_dim)

    return output


# ---------------------------------------------------------------------------
# MiMoV2FlashMoEGate  — sigmoid + noaux-TC correction bias
# ---------------------------------------------------------------------------

class MiMoV2FlashMoEGate(nnx.Module):
  """Router gate for MiMo-V2-Flash MoE with noaux-TC bias.

  Uses sigmoid-scored gating with an inference-time correction bias
  (``e_score_correction_bias``) added before top-k selection.  The correction
  bias is used *only* to determine which experts are selected; the final expert
  weights are computed from the raw sigmoid scores (without the bias), then
  L1-normalised across the selected experts.

  Args:
    num_experts: Total number of routed experts (256).
    hidden_size: Model hidden dimension (4096).
    num_experts_per_tok: Number of experts selected per token (8).
    dtype: Computation dtype.
    weight_dtype: Parameter storage dtype.
    rngs: PRNG keys.
  """

  def __init__(
      self,
      num_experts: int,
      hidden_size: int,
      num_experts_per_tok: int,
      dtype: DType,
      weight_dtype: DType,
      *,
      rngs: nnx.Rngs,
  ):
    self.num_experts = num_experts
    self.num_experts_per_tok = num_experts_per_tok
    self.dtype = dtype
    self.weight_dtype = weight_dtype

    # Routing weight matrix: (num_experts, hidden_size).
    self.weight = nnx.Param(
        jax.random.normal(rngs.params(), (num_experts, hidden_size), dtype=weight_dtype)
        * (hidden_size ** -0.5),
    )

    # Correction bias for noaux-TC top-k selection: (num_experts,).
    # Initialised to zeros; loaded from checkpoint at inference time.
    self.e_score_correction_bias = nnx.Param(
        jnp.zeros((num_experts,), dtype=weight_dtype),
    )

  def __call__(self, hidden_states: Array):
    """Compute top-k expert indices and normalised weights.

    Args:
      hidden_states: Shape (batch × seq_len, hidden_size).

    Returns:
      Tuple of:
        - top_k_indices: Shape (tokens, num_experts_per_tok).
        - top_k_weights: Shape (tokens, num_experts_per_tok), L1-normalised.
    """
    # Routing logits: (tokens, num_experts).
    logits = jnp.dot(
        hidden_states.astype(jnp.float32),
        self.weight[...].astype(jnp.float32).T,
    )

    # Sigmoid scores (used for final expert weighting).
    scores = jax.nn.sigmoid(logits)

    # noaux-TC: add correction bias *before* top-k selection only.
    scores_for_selection = scores + self.e_score_correction_bias[...].astype(jnp.float32)

    # Top-k selection over biased scores.
    # jnp.argsort descending: take first k.
    top_k_indices = jnp.argsort(scores_for_selection, axis=-1, descending=True)[
        :, : self.num_experts_per_tok
    ]

    # Gather unbiased scores for the selected experts.
    top_k_weights = jnp.take_along_axis(scores, top_k_indices, axis=-1)

    # L1 normalise the selected expert weights.
    top_k_weights = top_k_weights / (top_k_weights.sum(axis=-1, keepdims=True) + 1e-20)

    return top_k_indices, top_k_weights.astype(self.dtype)


# ---------------------------------------------------------------------------
# MiMoV2FlashSparseMoeBlock
# ---------------------------------------------------------------------------

class MiMoV2FlashSparseMoeBlock(nnx.Module):
  """MiMo-V2-Flash MoE block.

  256 routed experts, top-8 selection, sigmoid scoring with noaux-TC
  correction bias.  Each expert is a 2-layer SwiGLU MLP implemented as
  three linear projections (gate, up, down) with ``moe_intermediate_size=2048``.

  Args:
    config: MaxText config.
    mesh: Device mesh.
    quant: Optional quantisation config.
    rngs: PRNG keys.
  """

  def __init__(
      self,
      config: Config,
      mesh: Mesh,
      quant: None | Quant = None,
      *,
      rngs: nnx.Rngs,
  ):
    self.config = config
    self.mesh = mesh
    self.quant = quant
    cfg = config

    self.num_experts: int = cfg.num_experts
    self.num_experts_per_tok: int = cfg.num_experts_per_tok
    self.hidden_size: int = cfg.emb_dim
    self.intermediate_size: int = cfg.moe_mlp_dim

    # Gate / router.
    self.gate = MiMoV2FlashMoEGate(
        num_experts=self.num_experts,
        hidden_size=self.hidden_size,
        num_experts_per_tok=self.num_experts_per_tok,
        dtype=cfg.dtype,
        weight_dtype=cfg.weight_dtype,
        rngs=rngs,
    )

    # Expert weight matrices stored as (E, hidden, intermediate) tensors for
    # efficient batch matmul.
    ki = max_initializers.nd_dense_init(1.0, "fan_in", "truncated_normal")
    wi_shape = (self.num_experts, self.hidden_size, self.intermediate_size)
    wo_shape = (self.num_experts, self.intermediate_size, self.hidden_size)

    self.wi_0 = nnx.Param(  # gate projection
        ki(rngs.params(), wi_shape, cfg.weight_dtype, 1, 2),
    )
    self.wi_1 = nnx.Param(  # up projection
        ki(rngs.params(), wi_shape, cfg.weight_dtype, 1, 2),
    )
    self.wo = nnx.Param(   # down projection
        ki(rngs.params(), wo_shape, cfg.weight_dtype, 1, 2),
    )

  def __call__(self, hidden_states: Array, deterministic: bool) -> Array:
    """Apply the MoE block.

    Args:
      hidden_states: Shape (batch, seq_len, emb_dim).
      deterministic: Unused (no dropout in MoE gate).

    Returns:
      Output of shape (batch, seq_len, emb_dim).
    """
    orig_shape = hidden_states.shape
    tokens = hidden_states.reshape(-1, self.hidden_size)  # (T, H)

    # Route tokens to experts.
    top_k_indices, top_k_weights = self.gate(tokens)  # (T, K), (T, K)

    # Dispatch tokens to their assigned experts using a scatter/gather approach,
    # which is fully static-shape-friendly and XLA-compilable.
    #
    # Strategy: build an (E, T) weight matrix by scattering top_k_weights into
    # positions indexed by top_k_indices, then use a single einsum per expert
    # group. For 256 experts this is the most efficient static approach.
    #
    # dispatch_weights: (T, E) — weight for each (token, expert) pair (0 if not selected)
    T = tokens.shape[0]
    dispatch_weights = jnp.zeros((T, self.num_experts), dtype=jnp.float32)
    tok_idx = jnp.broadcast_to(
        jnp.arange(T)[:, jnp.newaxis], (T, self.num_experts_per_tok)
    )  # (T, K)
    dispatch_weights = dispatch_weights.at[tok_idx, top_k_indices].add(
        top_k_weights.astype(jnp.float32)
    )  # (T, E)

    wi_0 = self.wi_0[...].astype(self.config.dtype)  # (E, H, I)
    wi_1 = self.wi_1[...].astype(self.config.dtype)  # (E, H, I)
    wo = self.wo[...].astype(self.config.dtype)       # (E, I, H)

    # For each expert e compute the contribution to each token:
    #   g_e   = silu(tokens @ wi_0[e])               (T, I)
    #   u_e   = tokens @ wi_1[e]                      (T, I)
    #   out_e = (g_e * u_e) @ wo[e]                  (T, H)
    #   contribution_e = dispatch_weights[:, e:e+1] * out_e  (T, H)
    #
    # We vectorise over E using jnp.einsum with the expert axis:
    #   tokens_all: (T, H) broadcast over all experts.
    #   gate:  (E, T, I) = silu(einsum('TH,EHI->ETI', tokens, wi_0))
    #   up:    (E, T, I) =     einsum('TH,EHI->ETI', tokens, wi_1)
    #   down:  (E, T, H) =     einsum('ETI,EIH->ETH', gate*up, wo)
    #   out:   (T, H)    =     einsum('TE,ETH->TH',  dispatch_weights, down)

    tokens_fp = tokens.astype(self.config.dtype)  # (T, H)
    gate = jax.nn.silu(
        jnp.einsum("th,ehi->eti", tokens_fp, wi_0, precision=lax.Precision.DEFAULT)
    )                                                                     # (E, T, I)
    up = jnp.einsum("th,ehi->eti", tokens_fp, wi_1,
                    precision=lax.Precision.DEFAULT)                      # (E, T, I)
    down = jnp.einsum("eti,eih->eth", gate * up, wo,
                      precision=lax.Precision.DEFAULT)                    # (E, T, H)
    output = jnp.einsum("te,eth->th",
                        dispatch_weights.astype(self.config.dtype), down,
                        precision=lax.Precision.DEFAULT)                  # (T, H)
    return output.reshape(orig_shape)


# ---------------------------------------------------------------------------
# MiMoV2FlashDecoderLayer
# ---------------------------------------------------------------------------

class MiMoV2FlashDecoderLayer(nnx.Module):
  """Single MiMo-V2-Flash hybrid decoder layer.

  Each layer consists of:
  1. Pre-attention RMSNorm.
  2. Self-attention: global (full causal) *or* sliding-window, depending on
     the per-layer ``hybrid_layer_pattern`` value.
  3. Residual connection.
  4. Post-attention RMSNorm.
  5. Feed-forward network: dense SwiGLU MLP *or* sparse MoE SwiGLU, depending
     on the per-layer ``moe_layer_freq`` value.
  6. Residual connection.

  Args:
    config: MaxText configuration.
    mesh: Device mesh.
    model_mode: One of 'train', 'prefill', 'autoregressive'.
    layer_idx: Zero-based layer index within the transformer stack.
    quant: Optional quantisation config.
    rngs: PRNG keys.
  """

  def __init__(
      self,
      config: Config,
      mesh: Mesh,
      model_mode: str,
      layer_idx: int,
      quant: None | Quant = None,
      *,
      rngs: nnx.Rngs,
  ):
    self.config = config
    self.mesh = mesh
    self.model_mode = model_mode
    self.layer_idx = layer_idx
    self.quant = quant
    cfg = config

    self.activation_axis_names = ("activation_batch", "activation_norm_length", "activation_embed")

    # Determine attention type for this layer.
    hybrid_pattern = cfg.mimo_hybrid_layer_pattern
    if hybrid_pattern:
      is_swa = hybrid_pattern[layer_idx] == 1
    else:
      is_swa = False  # fallback: all full attention

    self.is_swa = is_swa

    # Determine whether this layer uses MoE or dense MLP.
    moe_freq = cfg.mimo_moe_layer_freq
    if moe_freq:
      use_moe = moe_freq[layer_idx] == 1
    else:
      # Fallback: use MoE if num_experts > 0.
      use_moe = cfg.num_experts > 0 and layer_idx > 0

    self.use_moe = use_moe

    # Pre-attention layer norm.
    self.input_layernorm = RMSNorm(
        num_features=cfg.emb_dim,
        epsilon=cfg.normalization_layer_epsilon,
        dtype=cfg.dtype,
        weight_dtype=cfg.weight_dtype,
        kernel_axes=("norm",),
        rngs=rngs,
    )

    # Self-attention (global or SWA).
    self.self_attn = MiMoV2FlashAttention(
        config=cfg,
        mesh=mesh,
        is_swa=is_swa,
        layer_idx=layer_idx,
        quant=quant,
        rngs=rngs,
    )

    # Post-attention layer norm.
    self.post_attention_layernorm = RMSNorm(
        num_features=cfg.emb_dim,
        epsilon=cfg.normalization_layer_epsilon,
        dtype=cfg.dtype,
        weight_dtype=cfg.weight_dtype,
        kernel_axes=("norm",),
        rngs=rngs,
    )

    # FFN: dense MLP (layer 0) or sparse MoE (all other layers).
    if use_moe:
      self.mlp = MiMoV2FlashSparseMoeBlock(
          config=cfg,
          mesh=mesh,
          quant=quant,
          rngs=rngs,
      )
    else:
      self.mlp = MlpBlock(
          config=cfg,
          mesh=mesh,
          in_features=cfg.emb_dim,
          intermediate_dim=cfg.mlp_dim,
          activations=cfg.mlp_activations,
          intermediate_dropout_rate=cfg.dropout_rate,
          dtype=cfg.dtype,
          weight_dtype=cfg.weight_dtype,
          quant=quant,
          model_mode=model_mode,
          rngs=rngs,
      )

  def __call__(
      self,
      inputs: Array,
      decoder_segment_ids: None | Array,
      decoder_positions: None | Array,
      deterministic: bool,
      model_mode: str,
      previous_chunk: Any = None,
      page_state: Any = None,
      slot: None | int = None,
      kv_cache: None | dict[str, Array] = None,
      attention_metadata: None | dict[str, Any] = None,
  ):
    """Forward pass through the decoder layer.

    Returns:
      Tuple of (output_hidden_states, kv_cache).  For this implementation
      kv_cache is always ``None`` (the layer manages it internally via the
      attention module's stateful buffers — page-manager integration is tracked
      as a future enhancement).
    """
    # Unpack tuple inputs (e.g., when chained from a previous layer that returns (h, kv)).
    if isinstance(inputs, tuple):
      inputs = inputs[0]

    residual = inputs

    # 1. Pre-attention norm + attention.
    hidden_states = self.input_layernorm(inputs)
    hidden_states = nn.with_logical_constraint(hidden_states, self.activation_axis_names)

    attn_output = self.self_attn(
        inputs=hidden_states,
        decoder_positions=decoder_positions,
        decoder_segment_ids=decoder_segment_ids,
        deterministic=deterministic,
        model_mode=model_mode,
    )

    # 2. First residual.
    hidden_states = residual + attn_output
    hidden_states = nn.with_logical_constraint(hidden_states, self.activation_axis_names)

    residual = hidden_states

    # 3. Post-attention norm + FFN.
    hidden_states = self.post_attention_layernorm(hidden_states)
    hidden_states = nn.with_logical_constraint(hidden_states, self.activation_axis_names)

    if self.use_moe:
      ffn_output = self.mlp(hidden_states, deterministic=deterministic)
    else:
      ffn_output = self.mlp(hidden_states, deterministic=deterministic)

    # 4. Second residual.
    layer_output = residual + ffn_output
    layer_output = nn.with_logical_constraint(layer_output, self.activation_axis_names)

    return layer_output, None


# ---------------------------------------------------------------------------
# Linen wrappers (required by decoders.py registry)
# ---------------------------------------------------------------------------

MiMoV2FlashDecoderLayerToLinen = nnx_wrappers.to_linen_class(
    MiMoV2FlashDecoderLayer,
    base_metadata_fn=variable_to_logically_partitioned,
)

MiMoV2FlashScannableBlockToLinen = nnx_wrappers.to_linen_class(
    MiMoV2FlashDecoderLayer,
    base_metadata_fn=variable_to_logically_partitioned,
)
