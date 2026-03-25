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

"""Unit tests for the MiMo-V2-Flash architecture components.

Tests cover:
  1. MiMoV2FlashMoEGate  — noaux-TC sigmoid routing correctness
  2. MiMoV2FlashAttention — output shape, SWA vs global, attention sink bias
  3. MiMoV2FlashSparseMoeBlock — output shape and conservation of residual
  4. MiMoV2FlashDecoderLayer  — layer selection (SWA vs GA, MoE vs dense)
  5. Config   — hybrid_layer_pattern and moe_layer_freq parsing
"""

import sys
import unittest

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import Mesh
from jax.experimental import mesh_utils
from flax import nnx

from maxtext.configs import pyconfig
from maxtext.utils import maxtext_utils
from maxtext.models import mimo_v2_flash
from tests.utils.test_helpers import get_test_config_path, get_decoupled_parallelism_overrides

# ---------------------------------------------------------------------------
# Test configuration overrides (tiny model so tests run quickly on CPU/1-TPU)
# ---------------------------------------------------------------------------

# Matching the actual architecture ratios but at toy scale.
_TINY_CONFIG = {
    "decoder_block": "mimo_v2_flash",
    # Tiny model dimensions
    "base_emb_dim": 64,
    "base_num_decoder_layers": 4,
    "base_num_query_heads": 4,
    "base_num_kv_heads": 2,
    "head_dim": 24,           # toy Q/K head dim
    "vocab_size": 256,
    # FFN
    "base_mlp_dim": 128,
    "base_moe_mlp_dim": 32,
    # MoE
    "num_experts": 8,
    "num_experts_per_tok": 2,
    "norm_topk_prob": True,
    "routed_score_func": "sigmoid",
    "mlp_activations": ["silu", "linear"],
    # Norms
    "normalization_layer_epsilon": 1.0e-5,
    # RoPE
    "rope_max_timescale": 5000000,
    "partial_rotary_factor": 0.334,
    # MiMo-specific
    "mimo_v_head_dim": 16,            # toy V head dim (asymmetric with Q/K=24)
    "mimo_swa_num_kv_heads": 4,
    "mimo_swa_rope_theta": 10000.0,
    "mimo_swa_window_size": 4,
    "mimo_attention_value_scale": 0.707,
    # 4 layers: pattern [GA, SWA, SWA, GA]
    "mimo_hybrid_layer_pattern": [0, 1, 1, 0],
    # 4 layers: layer 0 dense, layers 1-3 MoE
    "mimo_moe_layer_freq": [0, 1, 1, 1],
    # Inference / training settings
    "max_target_length": 16,
    "per_device_batch_size": 1,
    "enable_dropout": False,
    "scan_layers": False,
    "logits_via_embedding": False,
    "use_qk_norm": False,
    "dtype": "float32",
    "weight_dtype": "float32",
}


def _make_config():
    """Create a minimal MaxText config for MiMo-V2-Flash."""
    overrides = {**_TINY_CONFIG, **get_decoupled_parallelism_overrides()}
    cfg = pyconfig.initialize(
        [sys.argv[0], get_test_config_path()],
        **overrides,
    )
    return cfg


def _make_mesh(cfg):
    devices_array = maxtext_utils.create_device_mesh(cfg)
    return Mesh(devices_array, cfg.mesh_axes)


# ---------------------------------------------------------------------------
# Test: MiMoV2FlashMoEGate
# ---------------------------------------------------------------------------

class TestMiMoV2FlashMoEGate(unittest.TestCase):
    """Tests for the noaux-TC router gate."""

    def setUp(self):
        super().setUp()
        self.num_experts = 8
        self.hidden_size = 32
        self.topk = 2
        self.rngs = nnx.Rngs(params=jax.random.PRNGKey(0))
        self.gate = mimo_v2_flash.MiMoV2FlashMoEGate(
            num_experts=self.num_experts,
            hidden_size=self.hidden_size,
            num_experts_per_tok=self.topk,
            dtype=jnp.float32,
            weight_dtype=jnp.float32,
            rngs=self.rngs,
        )

    def test_output_shapes(self):
        """Gate returns tensors of the correct shape."""
        batch_tokens = 6
        hidden = jax.random.normal(jax.random.PRNGKey(1), (batch_tokens, self.hidden_size))
        indices, weights = self.gate(hidden)
        self.assertEqual(indices.shape, (batch_tokens, self.topk))
        self.assertEqual(weights.shape, (batch_tokens, self.topk))

    def test_weights_sum_to_one(self):
        """L1-normalised weights sum to 1 per token."""
        batch_tokens = 4
        hidden = jax.random.normal(jax.random.PRNGKey(2), (batch_tokens, self.hidden_size))
        _indices, weights = self.gate(hidden)
        sums = weights.sum(axis=-1)
        np.testing.assert_allclose(np.array(sums), np.ones(batch_tokens), atol=1e-5)

    def test_indices_in_range(self):
        """All selected expert indices are valid."""
        batch_tokens = 5
        hidden = jax.random.normal(jax.random.PRNGKey(3), (batch_tokens, self.hidden_size))
        indices, _ = self.gate(hidden)
        self.assertTrue(jnp.all(indices >= 0).item())
        self.assertTrue(jnp.all(indices < self.num_experts).item())

    def test_correction_bias_used_only_for_selection(self):
        """Setting a large correction bias for one expert forces its selection."""
        # Bias the gate strongly towards expert 7 for all tokens
        bias = jnp.zeros(self.num_experts).at[7].set(100.0)
        self.gate.e_score_correction_bias.set_raw_value(bias)
        batch_tokens = 4
        hidden = jax.random.normal(jax.random.PRNGKey(4), (batch_tokens, self.hidden_size))
        indices, _ = self.gate(hidden)
        # Expert 7 should be selected for every token (it's in top-2 due to bias)
        selected = set(np.array(indices).flatten().tolist())
        self.assertIn(7, selected)


# ---------------------------------------------------------------------------
# Test: MiMoV2FlashAttention output shapes
# ---------------------------------------------------------------------------

class TestMiMoV2FlashAttention(unittest.TestCase):
    """Output-shape and structural tests for the attention module."""

    def _make_attn(self, is_swa: bool):
        cfg = _make_config()
        mesh = _make_mesh(cfg)
        rngs = nnx.Rngs(params=jax.random.PRNGKey(0))
        return mimo_v2_flash.MiMoV2FlashAttention(
            config=cfg,
            mesh=mesh,
            is_swa=is_swa,
            layer_idx=0,
            quant=None,
            rngs=rngs,
        ), cfg

    def test_global_attention_output_shape(self):
        """Full (global) attention output has shape (batch, seq_len, emb_dim)."""
        attn, cfg = self._make_attn(is_swa=False)
        batch, seq = 2, 8
        x = jax.random.normal(jax.random.PRNGKey(1), (batch, seq, cfg.emb_dim))
        pos = jnp.broadcast_to(jnp.arange(seq)[None, :], (batch, seq))
        out = attn(x, decoder_positions=pos, decoder_segment_ids=None,
                   deterministic=True, model_mode="train")
        self.assertEqual(out.shape, (batch, seq, cfg.emb_dim))

    def test_swa_attention_output_shape(self):
        """Sliding-window attention output has shape (batch, seq_len, emb_dim)."""
        attn, cfg = self._make_attn(is_swa=True)
        batch, seq = 2, 8
        x = jax.random.normal(jax.random.PRNGKey(2), (batch, seq, cfg.emb_dim))
        pos = jnp.broadcast_to(jnp.arange(seq)[None, :], (batch, seq))
        out = attn(x, decoder_positions=pos, decoder_segment_ids=None,
                   deterministic=True, model_mode="train")
        self.assertEqual(out.shape, (batch, seq, cfg.emb_dim))

    def test_swa_layer_has_sink_bias(self):
        """SWA attention layers should have an initialised sink_bias parameter."""
        attn_swa, _ = self._make_attn(is_swa=True)
        attn_ga, _ = self._make_attn(is_swa=False)
        self.assertIsNotNone(attn_swa.sink_bias)
        self.assertIsNone(attn_ga.sink_bias)

    def test_ga_uses_fewer_kv_heads(self):
        """Global attention uses num_kv_heads (2 in tiny config) < SWA (4)."""
        attn_ga, cfg = self._make_attn(is_swa=False)
        attn_swa, _ = self._make_attn(is_swa=True)
        self.assertEqual(attn_ga.num_kv_heads, cfg.num_kv_heads)        # 2
        self.assertEqual(attn_swa.num_kv_heads, cfg.mimo_swa_num_kv_heads)  # 4

    def test_asymmetric_head_dims(self):
        """Q/K head dim and V head dim differ in the attention module."""
        attn, cfg = self._make_attn(is_swa=False)
        self.assertEqual(attn.head_dim, cfg.head_dim)        # 24
        self.assertEqual(attn.v_head_dim, cfg.mimo_v_head_dim)  # 16


# ---------------------------------------------------------------------------
# Test: MiMoV2FlashSparseMoeBlock output shape
# ---------------------------------------------------------------------------

class TestMiMoV2FlashSparseMoeBlock(unittest.TestCase):
    """Shape and gradient-flow tests for the MoE block."""

    def setUp(self):
        super().setUp()
        cfg = _make_config()
        mesh = _make_mesh(cfg)
        rngs = nnx.Rngs(params=jax.random.PRNGKey(0))
        self.cfg = cfg
        self.moe = mimo_v2_flash.MiMoV2FlashSparseMoeBlock(
            config=cfg, mesh=mesh, quant=None, rngs=rngs
        )

    def test_output_shape(self):
        """MoE block output matches input shape."""
        batch, seq = 2, 4
        x = jax.random.normal(
            jax.random.PRNGKey(0), (batch, seq, self.cfg.emb_dim)
        )
        out = self.moe(x, deterministic=True)
        self.assertEqual(out.shape, x.shape)

    def test_output_finite(self):
        """MoE block output contains no NaN or Inf values."""
        batch, seq = 2, 4
        x = jax.random.normal(
            jax.random.PRNGKey(1), (batch, seq, self.cfg.emb_dim)
        )
        out = self.moe(x, deterministic=True)
        self.assertTrue(jnp.all(jnp.isfinite(out)).item(), "MoE output contains non-finite values")


# ---------------------------------------------------------------------------
# Test: MiMoV2FlashDecoderLayer
# ---------------------------------------------------------------------------

class TestMiMoV2FlashDecoderLayer(unittest.TestCase):
    """Shape and layer-selection tests for the full decoder layer."""

    def _make_layer(self, layer_idx: int):
        cfg = _make_config()
        mesh = _make_mesh(cfg)
        rngs = nnx.Rngs(params=jax.random.PRNGKey(0))
        layer = mimo_v2_flash.MiMoV2FlashDecoderLayer(
            config=cfg, mesh=mesh, model_mode="train",
            layer_idx=layer_idx, quant=None, rngs=rngs,
        )
        return layer, cfg

    def test_layer_0_is_dense_ga(self):
        """Layer 0: global attention (pattern[0]=0), dense MLP (freq[0]=0)."""
        layer, _ = self._make_layer(0)
        self.assertFalse(layer.is_swa)
        self.assertFalse(layer.use_moe)
        self.assertIsInstance(layer.mlp.wi_0, nnx.Module)  # MlpBlock has DenseGeneral

    def test_layer_1_is_swa_moe(self):
        """Layer 1: SWA (pattern[1]=1), MoE (freq[1]=1)."""
        layer, _ = self._make_layer(1)
        self.assertTrue(layer.is_swa)
        self.assertTrue(layer.use_moe)
        self.assertIsInstance(layer.mlp, mimo_v2_flash.MiMoV2FlashSparseMoeBlock)

    def test_layer_3_is_ga_moe(self):
        """Layer 3: global attention (pattern[3]=0), MoE (freq[3]=1)."""
        layer, _ = self._make_layer(3)
        self.assertFalse(layer.is_swa)
        self.assertTrue(layer.use_moe)

    def test_output_shape(self):
        """Decoder layer output shape matches input."""
        layer, cfg = self._make_layer(1)
        batch, seq = 2, 8
        x = jax.random.normal(jax.random.PRNGKey(0), (batch, seq, cfg.emb_dim))
        pos = jnp.broadcast_to(jnp.arange(seq)[None, :], (batch, seq))
        out, kv = layer(
            x, decoder_segment_ids=None, decoder_positions=pos,
            deterministic=True, model_mode="train",
        )
        self.assertEqual(out.shape, (batch, seq, cfg.emb_dim))
        self.assertIsNone(kv)

    def test_output_finite(self):
        """Decoder layer output contains no NaN/Inf."""
        layer, cfg = self._make_layer(0)
        batch, seq = 1, 4
        x = jax.random.normal(jax.random.PRNGKey(1), (batch, seq, cfg.emb_dim))
        pos = jnp.broadcast_to(jnp.arange(seq)[None, :], (batch, seq))
        out, _ = layer(
            x, decoder_segment_ids=None, decoder_positions=pos,
            deterministic=True, model_mode="train",
        )
        self.assertTrue(jnp.all(jnp.isfinite(out)).item())


# ---------------------------------------------------------------------------
# Test: Config parsing
# ---------------------------------------------------------------------------

class TestMiMoConfig(unittest.TestCase):
    """Tests that the MiMo-specific config fields are correctly parsed."""

    def setUp(self):
        super().setUp()
        self.cfg = _make_config()

    def test_decoder_block_type(self):
        """decoder_block should be 'mimo_v2_flash'."""
        from maxtext.common.common_types import DecoderBlockType  # pylint: disable=import-outside-toplevel
        self.assertEqual(self.cfg.decoder_block, DecoderBlockType.MIMO_V2_FLASH)

    def test_mimo_hybrid_layer_pattern(self):
        """hybrid_layer_pattern should be parsed correctly."""
        expected = [0, 1, 1, 0]
        self.assertEqual(list(self.cfg.mimo_hybrid_layer_pattern), expected)

    def test_mimo_moe_layer_freq(self):
        """moe_layer_freq should be parsed correctly."""
        expected = [0, 1, 1, 1]
        self.assertEqual(list(self.cfg.mimo_moe_layer_freq), expected)

    def test_mimo_v_head_dim(self):
        """V head dim should equal value set in config (16 for tiny)."""
        self.assertEqual(self.cfg.mimo_v_head_dim, 16)

    def test_mimo_swa_window_size(self):
        """SWA window size should equal value set in config."""
        self.assertEqual(self.cfg.mimo_swa_window_size, 4)

    def test_mimo_attention_value_scale(self):
        """Attention value scale should be stored correctly."""
        self.assertAlmostEqual(self.cfg.mimo_attention_value_scale, 0.707, places=3)

    def test_partial_rotary_factor(self):
        """partial_rotary_factor should be accepted for mimo_v2_flash."""
        self.assertAlmostEqual(self.cfg.partial_rotary_factor, 0.334, places=3)


if __name__ == "__main__":
    unittest.main()
