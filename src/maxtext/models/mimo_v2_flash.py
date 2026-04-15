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
import functools

import jax
import jax.numpy as jnp
from jax import lax
from jax.sharding import PartitionSpec as P

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
from maxtext.kernels import megablox as mblx


# ---------------------------------------------------------------------------
# Sparse MoE dispatch helpers
# ---------------------------------------------------------------------------

def _mimo_permute(
    tokens: Array,
    top_k_indices: Array,
    e_local: int,
    shard_id: Array,
) -> tuple[Array, Array, Array, Array]:
  """Sort tokens by local expert assignment within one EP shard.

  Args:
    tokens:         (T, H) — flat token matrix for this forward pass.
    top_k_indices:  (T, K) — global expert indices [0, E_total), replicated
                    across all EP shards (gate forces full replication).
    e_local:        Number of experts owned by this EP shard (E_total / EP).
    shard_id:       Scalar integer, index of this EP shard (0 … EP-1).

  Returns:
    sorted_tokens:  (T*K, H) — tokens sorted by their local expert assignment.
                    Tokens not routed to this shard appear with a zero local
                    index and will produce outputs that are masked to zero via
                    local_weights.
    sort_order:     (T*K,) — argsort indices used to reconstruct original order.
    group_sizes:    (e_local,) — number of (token, expert) pairs per local
                    expert (used by mblx.gmm).
    local_weights:  (T*K,) — per-sorted-slot weight; 0.0 for slots not routed
                    to this shard, so their contribution is zeroed during unpermute.
  """
  T, K = top_k_indices.shape
  local_start = shard_id * e_local

  # Boolean mask: which (token, k) slots route to this shard's experts.
  local_mask = (top_k_indices >= local_start) & (top_k_indices < local_start + e_local)  # (T, K)

  # Re-index global expert IDs to shard-local [0, e_local).
  # Slots not belonging to this shard are mapped to 0 (will be masked out).
  local_indices = jnp.where(local_mask, top_k_indices - local_start, 0)  # (T, K)

  # Flatten to (T*K,) for argsort.
  flat_local   = local_indices.ravel()  # (T*K,)
  flat_mask    = local_mask.ravel()     # (T*K,)

  # Sort by local expert ID so gmm receives contiguous groups.
  sort_order   = jnp.argsort(flat_local)  # (T*K,)

  # Replicate each token K times then sort.
  repeated = jnp.repeat(tokens, K, axis=0)   # (T*K, H)
  sorted_tokens = repeated[sort_order]         # (T*K, H)

  # group_sizes: bincount over local expert IDs [0, e_local).
  # Non-local slots are mapped to local index 0 (they'll be zero-weighted
  # during unpermute but mblx.gmm needs them included in the buffer).
  group_sizes = jnp.bincount(flat_local, length=e_local)  # (e_local,)

  # Per-slot weight (0 for non-local slots so their output is zeroed).
  local_weights = jnp.where(flat_mask, jnp.ones(T * K, dtype=jnp.float32), 0.0)  # (T*K,)

  return sorted_tokens, sort_order, group_sizes, local_weights


def _mimo_unpermute(
    sorted_output: Array,
    sort_order: Array,
    top_k_weights: Array,
    local_weights: Array,
    T: int,
    K: int,
) -> Array:
  """Reverse the permutation, apply routing weights, and sum over K.

  Args:
    sorted_output:  (T*K, H) — output of the down-projection gmm.
    sort_order:     (T*K,) — the argsort from _mimo_permute.
    top_k_weights:  (T, K) — normalised routing weights from the gate.
    local_weights:  (T*K,) — mask (0 for slots not on this shard).
    T:              Number of tokens.
    K:              Top-k experts per token.

  Returns:
    (T, H) — token outputs for this EP shard (needs psum over expert axis).
  """
  H = sorted_output.shape[-1]
  # Reverse the sort.
  unsort_order  = jnp.argsort(sort_order)              # (T*K,)
  unsorte_out   = sorted_output[unsort_order]           # (T*K, H)
  # Zero out contributions from slots not routed to this shard.
  unsorte_out   = unsorte_out * local_weights[:, None]  # (T*K, H)
  # Reshape to (T, K, H) and apply routing weights, then sum over K.
  reshaped  = unsorte_out.reshape(T, K, H)              # (T, K, H)
  weights   = top_k_weights.reshape(T, K, 1)            # (T, K, 1)
  return (reshaped * weights).sum(axis=1)               # (T, H)


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

  def __call__(self, hidden_states: Array, deterministic: bool) -> Array:
    """Apply the MoE block with expert parallelism via shard_map.

    Args:
      hidden_states: Shape (batch, seq_len, emb_dim).
      deterministic: Unused (no dropout in MoE gate).

    Returns:
      Output of shape (batch, seq_len, emb_dim).
    """
    orig_shape = hidden_states.shape
    tokens = hidden_states.reshape(-1, self.hidden_size)  # (T, H)

    # Route tokens to experts (gate runs replicated on all devices).
    top_k_indices, top_k_weights = self.gate(tokens)  # (T, K), (T, K)

    # Sparse EP-parallel dispatch via megablox grouped matmul, wrapped in
    # shard_map so that jax.lax.axis_index("expert") and psum are valid.
    #
    # Weight sharding: wi_0/wi_1 are ("exp", "embed_no_exp", "mlp") and
    # wo is ("exp", "mlp", "embed_no_exp"), meaning the "expert" mesh axis
    # shards axis-0 of each weight tensor.  Inside shard_map each device
    # sees only its E_local rows → in_specs P("expert", None, None).
    #
    # Token/routing tensors are replicated across the expert axis → P().
    cfg = self.config

    def _sparse_dispatch(tokens_fp, top_k_indices, top_k_weights, wi_0, wi_1, wo):
      T = tokens_fp.shape[0]
      K = top_k_indices.shape[1]
      E_local = wi_0.shape[0]
      shard_id = jax.lax.axis_index("expert")

      sorted_tokens, sort_order, group_sizes, local_weights = _mimo_permute(
          tokens_fp, top_k_indices, E_local, shard_id
      )
      g = mblx.gmm(sorted_tokens, wi_0, group_sizes,
                   preferred_element_type=cfg.dtype)
      u = mblx.gmm(sorted_tokens, wi_1, group_sizes,
                   preferred_element_type=cfg.dtype)
      h = jax.nn.silu(g) * u
      d = mblx.gmm(h, wo, group_sizes,
                   preferred_element_type=cfg.dtype)
      local_out = _mimo_unpermute(d, sort_order, top_k_weights, local_weights, T, K)
      return jax.lax.psum(local_out, axis_name="expert")

    tokens_fp = tokens.astype(cfg.dtype)
    wi_0 = self.wi_0[...].astype(cfg.dtype)
    wi_1 = self.wi_1[...].astype(cfg.dtype)
    wo   = self.wo[...].astype(cfg.dtype)

    output = jax.shard_map(
        _sparse_dispatch,
        mesh=self.mesh,
        in_specs=(
            P(),              # tokens_fp     — replicated
            P(),              # top_k_indices — replicated
            P(),              # top_k_weights — replicated
            P("expert", None, None),  # wi_0  — sharded on expert axis
            P("expert", None, None),  # wi_1  — sharded on expert axis
            P("expert", None, None),  # wo    — sharded on expert axis
        ),
        out_specs=P(),        # output        — replicated (after psum)
        check_rep=False,
    )(tokens_fp, top_k_indices, top_k_weights, wi_0, wi_1, wo)

    return output.astype(cfg.dtype).reshape(orig_shape)


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
# MiMoV2FlashSixLayerCycleBlock
# ---------------------------------------------------------------------------
#
# MiMo-V2-Flash layer structure (48 layers total):
#
#   Phase A  layer 0          : unique — global attention + dense MLP
#   Phase B  layers  1- 4 (4×): SWA-attention + sparse MoE  (homogeneous run)
#   Phase C  layers  5-46 (7×): repeating 6-layer cycle [G-MoE, SWA-MoE×5]
#   Phase D  layer 47         : unique — global attention + sparse MoE
#
# The 6-layer cycle (Phase C) has identical structure in every repetition:
#   pos 0 : global attention  + MoE  (KV heads = cfg.num_kv_heads     = 4)
#   pos 1 : SWA attention     + MoE  (KV heads = cfg.mimo_swa_num_kv_heads = 8)
#   pos 2 : SWA attention     + MoE
#   pos 3 : SWA attention     + MoE
#   pos 4 : SWA attention     + MoE
#   pos 5 : SWA attention     + MoE
#
# Representative global layer indices used for configuration at each position:
#   pos 0 → layer 5   (hybrid_pattern=0 ⇒ global, moe_freq=1 ⇒ MoE)
#   pos 1 → layer 6   (hybrid_pattern=1 ⇒ SWA,    moe_freq=1 ⇒ MoE)
#   ...
#   pos 5 → layer 10  (hybrid_pattern=1 ⇒ SWA,    moe_freq=1 ⇒ MoE)
#
# ROUND 2 CHECKPOINT LAYOUT — produced by tools/mimo_stack_checkpoint.py:
#   The flat per-layer OCDBT checkpoint (decoder/layers/{i}/*) must be
#   restacked before using scan_layers=True Round 2.  After conversion the
#   stacked checkpoint has:
#     decoder.layers_c.layers_0.*  shape (7, ...)  ← layers  5,11,17,23,29,35,41
#     decoder.layers_c.layers_1.*  shape (7, ...)  ← layers  6,12,18,24,30,36,42
#     ...
#     decoder.layers_c.layers_5.*  shape (7, ...)  ← layers 10,16,22,28,34,40,46
#   scan_decoder_layers(length=7) in decoders.py then scans over the
#   leading axis of size 7, calling this 6-layer cycle body 7 times.

# Representative global layer_idx for each in-cycle position (first cycle).
_CYCLE_REP_LAYER_IDX = (5, 6, 7, 8, 9, 10)
_CYCLE_LENGTH = 6      # layers per cycle
_CYCLE_COUNT  = 7      # number of cycle repetitions (layers 5-46)
_CYCLE_START  = 5      # global index of the first cycle layer


class MiMoV2FlashSixLayerCycleBlock(nnx.Module):
  """One repeating 6-layer cycle of MiMo-V2-Flash (Phase C, layers 5–46).

  Holds 6 ``MiMoV2FlashDecoderLayer`` sublayers named ``layers_0`` …
  ``layers_5``.  When used as the body of ``scan_decoder_layers(length=7)``,
  parameters are stacked along the scan axis so XLA compiles the 6-layer body
  once and loops 7 times, capping peak HLO temp at ~3 GiB instead of ~22 GiB
  (enabling sparse-gather MoE dispatch without OOM).

  ``__call__`` is a sequential Python loop over the 6 sublayers.  ``nn.scan``
  wraps this class externally via ``scan_decoder_layers(length=7)`` in
  ``decoders.py``, so XLA compiles the 6-layer body once and loops 7 times
  (capping peak HLO temp at ~3 GiB instead of ~22 GiB).

  **Prerequisite:** Run ``tools/mimo_stack_checkpoint.py`` once to convert the
  flat per-layer checkpoint to the stacked (7, ...) layout expected by
  ``scan_decoder_layers``; then point ``load_parameters_path`` to the new
  stacked checkpoint path.
  """

  def __init__(
      self,
      config: Config,
      mesh: Mesh,
      model_mode: str,
      quant: None | Quant = None,
      *,
      rngs: nnx.Rngs,
  ):
    self.config = config
    self.mesh = mesh
    self.model_mode = model_mode
    self.quant = quant

    for pos, rep_idx in enumerate(_CYCLE_REP_LAYER_IDX):
      layer = MiMoV2FlashDecoderLayer(
          config=config,
          mesh=mesh,
          model_mode=model_mode,
          layer_idx=rep_idx,   # sets is_swa / use_moe for this position
          quant=quant,
          rngs=rngs,
      )
      setattr(self, f"layers_{pos}", layer)

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
    """Apply all 6 cycle layers sequentially; return (output, None).

    The ``(output, None)`` tuple is the scan-body convention: the first element
    is the carry (hidden states), the second is unused scan output (None).
    """
    y = inputs
    for pos in range(_CYCLE_LENGTH):
      y, _ = getattr(self, f"layers_{pos}")(
          y,
          decoder_segment_ids,
          decoder_positions,
          deterministic,
          model_mode,
          previous_chunk=previous_chunk,
          page_state=page_state,
          slot=slot,
          kv_cache=kv_cache,
          attention_metadata=attention_metadata,
      )
    return y, None


# ---------------------------------------------------------------------------
# Linen wrappers (required by decoders.py registry)
# ---------------------------------------------------------------------------

MiMoV2FlashDecoderLayerToLinen = nnx_wrappers.to_linen_class(
    MiMoV2FlashDecoderLayer,
    base_metadata_fn=variable_to_logically_partitioned,
)

# Linen wrapper for the 6-layer cycle block.
# get_decoder_layers() returns this as RemattedBlockLayers[1] when
# scan_layers=True.  Round 2 will wire it into scan_decoder_layers(length=7).
MiMoV2FlashSixLayerCycleBlockToLinen = nnx_wrappers.to_linen_class(
    MiMoV2FlashSixLayerCycleBlock,
    base_metadata_fn=variable_to_logically_partitioned,
)

# Alias kept for backward compatibility.  The scan_layers=True path in
# decoders.py currently uses RemattedBlockLayers[0] (MiMoV2FlashDecoderLayerToLinen)
# for the sequential fallback, not this alias.  In Round 2 this alias will be
# retired in favour of direct use of MiMoV2FlashSixLayerCycleBlockToLinen.
MiMoV2FlashScannableBlockToLinen = MiMoV2FlashSixLayerCycleBlockToLinen
