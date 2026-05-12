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
from jax.sharding import Mesh, NamedSharding, PartitionSpec

from maxtext.common.common_types import (
    Config,
    DType,
    Array,
)
from maxtext.layers import initializers as max_initializers
from maxtext.layers import nnx_wrappers
from maxtext.layers import quantizations
from maxtext.layers.normalizations import RMSNorm
from maxtext.layers.initializers import variable_to_logically_partitioned
from maxtext.layers.quantizations import AqtQuantization as Quant
from maxtext.layers.linears import MlpBlock
from maxtext.layers.attentions import Attention, AttentionType
from maxtext.utils import max_utils
from maxtext.kernels.fp8_dequant_matmul import fp8_moe_matmul



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
      mesh: Mesh,
      *,
      rngs: nnx.Rngs,
  ):
    self.num_experts = num_experts
    self.num_experts_per_tok = num_experts_per_tok
    self.dtype = dtype
    self.weight_dtype = weight_dtype
    self.mesh = mesh

    # Routing weight matrix: (num_experts, hidden_size).
    self.weight = nnx.Param(
        jax.random.normal(rngs.params(), (num_experts, hidden_size), dtype=weight_dtype)
        * (hidden_size ** -0.5),
        sharding=("exp", "embed_no_exp"),
    )

    # Correction bias for noaux-TC top-k selection: (num_experts,).
    # Initialised to zeros; loaded from checkpoint at inference time.
    # Replicated (not sharded) so that top-k selection sees all expert biases
    # regardless of ici_expert_parallelism.
    self.e_score_correction_bias = nnx.Param(
        jnp.zeros((num_experts,), dtype=weight_dtype),
        sharding=(None,),
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
    # With ici_expert_parallelism > 1, the gate weight is sharded across the
    # expert axis, so each device computes logits for only E/EP experts.  Force
    # the full (tokens, num_experts) tensor to be replicated on all devices so
    # that subsequent argsort and top-k selection see ALL expert scores.
    logits = jax.lax.with_sharding_constraint(
        logits, NamedSharding(self.mesh, PartitionSpec(None, None))
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


def _block_dequant_fp8(
    kernel_fp8: Array,   # [E, In, Out] float8_e4m3fn
    scale_inv: Array,    # [E, In//128, Out//128] float32
    bm: int = 128,
    bn: int = 128,
) -> Array:
    """Dequantize block-wise FP8 expert weights to BF16.

    Applies the per-128×128-block scale_inv to convert FP8 E4M3FN weights
    back to BF16.  Matches the HF block-wise FP8 quantization scheme used
    by MiMo-V2-Flash: dequant[e, i, j] = fp8[e, i, j] * scale_inv[e, i//128, j//128].
    """
    E, In, Out = kernel_fp8.shape
    blocks = kernel_fp8.reshape(E, In // bm, bm, Out // bn, bn)
    scale = scale_inv[:, :, jnp.newaxis, :, jnp.newaxis]
    return (blocks.astype(jnp.float32) * scale).astype(jnp.bfloat16).reshape(E, In, Out)


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
        mesh=mesh,
        rngs=rngs,
    )

    # Expert weight matrices stored as (E, hidden, intermediate) tensors for
    # efficient batch matmul.
    ki = max_initializers.nd_dense_init(1.0, "fan_in", "truncated_normal")
    wi_shape = (self.num_experts, self.hidden_size, self.intermediate_size)
    wo_shape = (self.num_experts, self.intermediate_size, self.hidden_size)

    self.wi_0 = nnx.Param(  # gate projection
        ki(rngs.params(), wi_shape, cfg.weight_dtype, 1, 2),
        sharding=("exp", "embed_no_exp", "mlp"),
    )
    self.wi_1 = nnx.Param(  # up projection
        ki(rngs.params(), wi_shape, cfg.weight_dtype, 1, 2),
        sharding=("exp", "embed_no_exp", "mlp"),
    )
    self.wo = nnx.Param(   # down projection
        ki(rngs.params(), wo_shape, cfg.weight_dtype, 1, 2),
        sharding=("exp", "mlp", "embed_no_exp"),
    )

    if getattr(cfg, "mimo_fp8_weight_mode", "") == "block_wise_fp8":
      # Per-block float32 scale_inv tensors for FP8 dequantization.
      # Shape: [E, dim_in//128, dim_out//128]; EP shards on the "exp" axis.
      scale_h = self.hidden_size // 128
      scale_i = self.intermediate_size // 128
      ones_wi = jnp.ones((self.num_experts, scale_h, scale_i), dtype=jnp.float32)
      ones_wo = jnp.ones((self.num_experts, scale_i, scale_h), dtype=jnp.float32)
      self.wi_0_scale_inv = nnx.Param(ones_wi, sharding=("exp", None, None))
      self.wi_1_scale_inv = nnx.Param(ones_wi, sharding=("exp", None, None))
      self.wo_scale_inv   = nnx.Param(ones_wo, sharding=("exp", None, None))

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

    cfg = self.config
    K: int = self.num_experts_per_tok    # 8
    E_total: int = self.num_experts      # 256
    T = tokens.shape[0]

    # Dense dispatch: build a (T, E) weight matrix and compute expert outputs
    # via batched einsums.  All local experts are evaluated; the dispatch matrix
    # zeros out non-selected expert contributions.
    dispatch_weights = jnp.zeros((T, E_total), dtype=jnp.float32)
    tok_idx = jnp.broadcast_to(
        jnp.arange(T)[:, jnp.newaxis], (T, K)
    )
    dispatch_weights = dispatch_weights.at[tok_idx, top_k_indices].add(
        top_k_weights.astype(jnp.float32)
    )
    tokens_fp = tokens.astype(cfg.dtype)
    if getattr(cfg, "mimo_fp8_weight_mode", "") == "block_wise_fp8":
      # Phase B: Pallas kernel fuses dequant + matmul — FP8 weight never
      # materialised as BF16 in HBM.  Falls back to Phase A (_block_dequant_fp8)
      # when T is too large for VMEM (prefill) or on platforms without Pallas TPU.
      _VMEM_T_LIMIT = 512  # max T for Pallas kernel VMEM scratch safety
      if T <= _VMEM_T_LIMIT:
        gate_logits = fp8_moe_matmul(
            tokens_fp, self.wi_0[...], self.wi_0_scale_inv[...],
        )                                                   # (E, T, I)
        gate_act = jax.nn.silu(gate_logits)
        up = fp8_moe_matmul(
            tokens_fp, self.wi_1[...], self.wi_1_scale_inv[...],
        )                                                   # (E, T, I)
        down = fp8_moe_matmul(
            gate_act * up, self.wo[...], self.wo_scale_inv[...],
            tokens_batched=True,
        )                                                   # (E, T, H)
      else:
        # Phase A fallback for large-T prefill.
        wi_0 = _block_dequant_fp8(self.wi_0[...], self.wi_0_scale_inv[...])
        wi_1 = _block_dequant_fp8(self.wi_1[...], self.wi_1_scale_inv[...])
        wo   = _block_dequant_fp8(self.wo[...],   self.wo_scale_inv[...])
        gate_act = jax.nn.silu(
            jnp.einsum("th,ehi->eti", tokens_fp, wi_0, precision=lax.Precision.DEFAULT)
        )
        up = jnp.einsum("th,ehi->eti", tokens_fp, wi_1, precision=lax.Precision.DEFAULT)
        down = jnp.einsum("eti,eih->eth", gate_act * up, wo, precision=lax.Precision.DEFAULT)
    else:
      wi_0 = self.wi_0[...].astype(cfg.dtype)
      wi_1 = self.wi_1[...].astype(cfg.dtype)
      wo   = self.wo[...].astype(cfg.dtype)
      gate_act = jax.nn.silu(
          jnp.einsum("th,ehi->eti", tokens_fp, wi_0, precision=lax.Precision.DEFAULT)
      )
      up = jnp.einsum("th,ehi->eti", tokens_fp, wi_1, precision=lax.Precision.DEFAULT)
      down = jnp.einsum("eti,eih->eth", gate_act * up, wo, precision=lax.Precision.DEFAULT)
    output = jnp.einsum(
        "te,eth->th", dispatch_weights.astype(cfg.dtype), down,
        precision=lax.Precision.DEFAULT,
    )
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

    # Self-attention: uses MaxText standard Attention NNX module with KV caching.
    # MiMo-specific params: v_head_dim=128, value_scale=0.707, partial RoPE (factor=0.334),
    # SWA layers use AttentionType.LOCAL_SLIDING with local_rope_max_timescale (= mimo_swa_rope_theta).
    num_kv_heads = (
        cfg.mimo_swa_num_kv_heads if (is_swa and cfg.mimo_swa_num_kv_heads > 0)
        else cfg.num_kv_heads
    )
    attn_type = AttentionType.LOCAL_SLIDING if is_swa else AttentionType.GLOBAL
    sliding_window = cfg.mimo_swa_window_size if is_swa else None
    batch_size, seq_len = max_utils.get_batch_seq_len_for_mode(cfg, model_mode)
    dummy_inputs_shape = (batch_size, seq_len, cfg.emb_dim)
    self.self_attn = Attention(
        config=cfg,
        num_query_heads=cfg.num_query_heads,
        num_kv_heads=num_kv_heads,
        head_dim=cfg.head_dim,
        v_head_dim=cfg.mimo_v_head_dim if cfg.mimo_v_head_dim > 0 else None,
        value_scale=cfg.mimo_attention_value_scale,
        query_pre_attn_scalar=cfg.head_dim**-0.5,
        # SWA layers have a learnable per-head sink bias in the checkpoint key "sink_bias".
        sink_param_name="sink_bias" if is_swa else None,
        max_target_length=cfg.max_target_length,
        max_prefill_predict_length=cfg.max_prefill_predict_length,
        attention_kernel=cfg.attention,
        inputs_q_shape=dummy_inputs_shape,
        inputs_kv_shape=dummy_inputs_shape,
        mesh=mesh,
        dtype=cfg.dtype,
        weight_dtype=cfg.weight_dtype,
        dropout_rate=cfg.dropout_rate,
        float32_qk_product=cfg.float32_qk_product,
        float32_logits=cfg.float32_logits,
        quant=quant,
        kv_quant=quantizations.configure_kv_quant(cfg),
        attention_type=attn_type,
        sliding_window_size=sliding_window,
        prefill_cache_axis_order=tuple(map(int, cfg.prefill_cache_axis_order.split(","))),
        ar_cache_axis_order=tuple(map(int, cfg.ar_cache_axis_order.split(","))),
        compute_axis_order=tuple(map(int, cfg.compute_axis_order.split(","))),
        reshape_q=cfg.reshape_q,
        use_ragged_attention=cfg.use_ragged_attention,
        ragged_block_size=cfg.ragged_block_size,
        model_mode=model_mode,
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

    attn_output, _ = self.self_attn(
        hidden_states,
        hidden_states,
        decoder_positions,
        decoder_segment_ids,
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
