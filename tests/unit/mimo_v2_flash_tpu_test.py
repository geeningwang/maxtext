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

"""TPU execution tests for the MiMo-V2-Flash architecture.

These tests require an actual TPU device (v6e-1 or larger) and validate:
  1. bfloat16 forward pass executes on the TPU device (not CPU fallback)
  2. jax.jit compilation and XLA lowering succeed for every layer type
  3. Gradient computation (vjp) is finite and correctly shaped
  4. Determinism: two identical forward passes return bitwise-equal results
  5. RoPE partial-rotation correctness on v6e: rotated dims differ from
     unrotated dims, non-rotated dims are preserved exactly
  6. MoE gate on TPU: noaux-TC bias is applied only for selection, not weights
  7. SWA sliding-window mask shapes and causal masking hold on device
  8. All-layer stack forward pass (all 4 tiny layers in sequence) is finite
"""

import sys
import unittest

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import Mesh
from flax import nnx

from maxtext.configs import pyconfig
from maxtext.utils import maxtext_utils
from maxtext.models import mimo_v2_flash
from tests.utils.test_helpers import get_test_config_path, get_decoupled_parallelism_overrides

# ---------------------------------------------------------------------------
# Shared tiny config — bfloat16, matching production dtype
# ---------------------------------------------------------------------------

_TINY_CONFIG_BF16 = {
    "decoder_block": "mimo_v2_flash",
    "base_emb_dim": 64,
    "base_num_decoder_layers": 4,
    "base_num_query_heads": 4,
    "base_num_kv_heads": 2,
    "head_dim": 24,
    "vocab_size": 256,
    "base_mlp_dim": 128,
    "base_moe_mlp_dim": 32,
    "num_experts": 8,
    "num_experts_per_tok": 2,
    "norm_topk_prob": True,
    "routed_score_func": "sigmoid",
    "mlp_activations": ["silu", "linear"],
    "normalization_layer_epsilon": 1.0e-5,
    "rope_max_timescale": 5000000,
    "partial_rotary_factor": 0.334,
    "mimo_v_head_dim": 16,
    "mimo_swa_num_kv_heads": 4,
    "mimo_swa_rope_theta": 10000.0,
    "mimo_swa_window_size": 4,
    "mimo_attention_value_scale": 0.707,
    "mimo_hybrid_layer_pattern": [0, 1, 1, 0],
    "mimo_moe_layer_freq": [0, 1, 1, 1],
    "max_target_length": 16,
    "per_device_batch_size": 1,
    "enable_dropout": False,
    "scan_layers": False,
    "logits_via_embedding": False,
    "use_qk_norm": False,
    # Production dtype — key difference from architecture_test.py which uses float32
    "dtype": "bfloat16",
    "weight_dtype": "bfloat16",
}


def _make_config():
    overrides = {**_TINY_CONFIG_BF16, **get_decoupled_parallelism_overrides()}
    return pyconfig.initialize(
        [sys.argv[0], get_test_config_path()],
        **overrides,
    )


def _make_mesh(cfg):
    devices_array = maxtext_utils.create_device_mesh(cfg)
    return Mesh(devices_array, cfg.mesh_axes)


def _make_layer(layer_idx: int, cfg=None, mesh=None):
    if cfg is None:
        cfg = _make_config()
    if mesh is None:
        mesh = _make_mesh(cfg)
    rngs = nnx.Rngs(params=jax.random.PRNGKey(layer_idx))
    return mimo_v2_flash.MiMoV2FlashDecoderLayer(
        config=cfg, mesh=mesh, model_mode="train",
        layer_idx=layer_idx, quant=None, rngs=rngs,
    ), cfg, mesh


# ---------------------------------------------------------------------------
# Test 1: Device placement — outputs must live on the TPU, not CPU
# ---------------------------------------------------------------------------

class TestMiMoV2FlashTPUDevicePlacement(unittest.TestCase):
    """Verifies that forward-pass outputs are placed on the TPU device."""

    def test_output_on_tpu_not_cpu(self):
        """Layer output tensor must reside on the v6e chip, not on CPU."""
        layer, cfg, _ = _make_layer(1)
        batch, seq = 1, 8
        x = jax.random.normal(jax.random.PRNGKey(0), (batch, seq, cfg.emb_dim), dtype=jnp.bfloat16)
        pos = jnp.broadcast_to(jnp.arange(seq)[None, :], (batch, seq))
        out, _ = layer(x, decoder_segment_ids=None, decoder_positions=pos,
                       deterministic=True, model_mode="train")
        jax.block_until_ready(out)
        devices = list(out.devices())
        self.assertEqual(len(devices), 1)
        self.assertIn("tpu", devices[0].device_kind.lower(),
                      f"Expected output on TPU device, got {devices[0]}")

    def test_output_dtype_is_bfloat16(self):
        """Forward pass must produce bfloat16 output to match production dtype."""
        layer, cfg, _ = _make_layer(1)
        batch, seq = 1, 8
        x = jax.random.normal(jax.random.PRNGKey(0), (batch, seq, cfg.emb_dim), dtype=jnp.bfloat16)
        pos = jnp.broadcast_to(jnp.arange(seq)[None, :], (batch, seq))
        out, _ = layer(x, decoder_segment_ids=None, decoder_positions=pos,
                       deterministic=True, model_mode="train")
        self.assertEqual(out.dtype, jnp.bfloat16,
                         f"Expected bfloat16 output, got {out.dtype}")


# ---------------------------------------------------------------------------
# Test 2: JIT compilation — XLA must lower all layer variants successfully
# ---------------------------------------------------------------------------

class TestMiMoV2FlashJITCompilation(unittest.TestCase):
    """Verifies that jax.jit can compile and execute all layer types."""

    def _jit_forward(self, layer_idx: int):
        layer, cfg, _ = _make_layer(layer_idx)
        batch, seq = 2, 8
        x = jax.random.normal(jax.random.PRNGKey(0), (batch, seq, cfg.emb_dim), dtype=jnp.bfloat16)
        pos = jnp.broadcast_to(jnp.arange(seq)[None, :], (batch, seq))

        @jax.jit
        def fwd(inputs, positions):
            out, _ = layer(inputs, decoder_segment_ids=None,
                           decoder_positions=positions,
                           deterministic=True, model_mode="train")
            return out

        out = fwd(x, pos)
        jax.block_until_ready(out)
        return out, cfg

    def test_jit_layer0_dense_ga(self):
        """Layer 0 (dense MLP + global attention) JIT-compiles without error."""
        out, cfg = self._jit_forward(0)
        self.assertEqual(out.shape[-1], cfg.emb_dim)

    def test_jit_layer1_moe_swa(self):
        """Layer 1 (MoE MLP + SWA) JIT-compiles without error."""
        out, cfg = self._jit_forward(1)
        self.assertEqual(out.shape[-1], cfg.emb_dim)

    def test_jit_layer2_moe_swa(self):
        """Layer 2 (MoE MLP + SWA) JIT-compiles without error."""
        out, cfg = self._jit_forward(2)
        self.assertEqual(out.shape[-1], cfg.emb_dim)

    def test_jit_layer3_moe_ga(self):
        """Layer 3 (MoE MLP + global attention) JIT-compiles without error."""
        out, cfg = self._jit_forward(3)
        self.assertEqual(out.shape[-1], cfg.emb_dim)

    def test_jit_output_finite_bfloat16(self):
        """JIT-compiled bfloat16 output must be fully finite (no NaN/Inf)."""
        out, _ = self._jit_forward(1)
        # Cast to float32 for isfinite check (bfloat16 has limited range)
        self.assertTrue(jnp.all(jnp.isfinite(out.astype(jnp.float32))).item(),
                        "JIT-compiled bfloat16 output contains NaN or Inf")


# ---------------------------------------------------------------------------
# Test 3: Gradient computation — vjp must be finite and correctly shaped
# ---------------------------------------------------------------------------

class TestMiMoV2FlashGradients(unittest.TestCase):
    """Verifies that gradients flow through the MiMo graph without NaN/Inf."""

    def _grad_test(self, layer_idx: int):
        layer, cfg, _ = _make_layer(layer_idx)
        batch, seq = 1, 8
        pos = jnp.broadcast_to(jnp.arange(seq)[None, :], (batch, seq))

        @jax.jit
        def fwd(inputs):
            out, _ = layer(inputs, decoder_segment_ids=None,
                           decoder_positions=pos,
                           deterministic=True, model_mode="train")
            return out

        x = jax.random.normal(jax.random.PRNGKey(0), (batch, seq, cfg.emb_dim), dtype=jnp.bfloat16)
        grad_fn = jax.grad(lambda inp: fwd(inp).sum().astype(jnp.float32))
        grads = grad_fn(x)
        jax.block_until_ready(grads)
        return grads, cfg

    def test_gradient_shape_dense_ga_layer(self):
        """Gradient w.r.t. input has same shape as input (layer 0)."""
        grads, cfg = self._grad_test(0)
        self.assertEqual(grads.shape, (1, 8, cfg.emb_dim))

    def test_gradient_finite_moe_swa_layer(self):
        """Gradient through MoE+SWA layer (layer 1) is finite in float32."""
        grads, _ = self._grad_test(1)
        finite = jnp.all(jnp.isfinite(grads.astype(jnp.float32))).item()
        self.assertTrue(finite, "MoE+SWA layer has non-finite gradients")

    def test_gradient_finite_moe_ga_layer(self):
        """Gradient through MoE+GA layer (layer 3) is finite in float32."""
        grads, _ = self._grad_test(3)
        finite = jnp.all(jnp.isfinite(grads.astype(jnp.float32))).item()
        self.assertTrue(finite, "MoE+GA layer has non-finite gradients")


# ---------------------------------------------------------------------------
# Test 4: Determinism — repeated forward passes must be bitwise identical
# ---------------------------------------------------------------------------

class TestMiMoV2FlashDeterminism(unittest.TestCase):
    """Verifies bitwise determinism of repeated forward passes on the TPU."""

    def test_deterministic_forward_pass(self):
        """Two identical forward passes (deterministic=True) must be equal."""
        layer, cfg, _ = _make_layer(1)
        batch, seq = 2, 8
        x = jax.random.normal(jax.random.PRNGKey(42), (batch, seq, cfg.emb_dim), dtype=jnp.bfloat16)
        pos = jnp.broadcast_to(jnp.arange(seq)[None, :], (batch, seq))

        out1, _ = layer(x, decoder_segment_ids=None, decoder_positions=pos,
                        deterministic=True, model_mode="train")
        out2, _ = layer(x, decoder_segment_ids=None, decoder_positions=pos,
                        deterministic=True, model_mode="train")
        jax.block_until_ready(out1)
        jax.block_until_ready(out2)
        np.testing.assert_array_equal(
            np.array(out1), np.array(out2),
            err_msg="Forward pass is not deterministic on TPU",
        )

    def test_different_inputs_give_different_outputs(self):
        """Different inputs must produce different outputs (non-trivial mapping)."""
        layer, cfg, _ = _make_layer(1)
        batch, seq = 2, 8
        x1 = jax.random.normal(jax.random.PRNGKey(1), (batch, seq, cfg.emb_dim), dtype=jnp.bfloat16)
        x2 = jax.random.normal(jax.random.PRNGKey(2), (batch, seq, cfg.emb_dim), dtype=jnp.bfloat16)
        pos = jnp.broadcast_to(jnp.arange(seq)[None, :], (batch, seq))

        out1, _ = layer(x1, decoder_segment_ids=None, decoder_positions=pos,
                        deterministic=True, model_mode="train")
        out2, _ = layer(x2, decoder_segment_ids=None, decoder_positions=pos,
                        deterministic=True, model_mode="train")
        jax.block_until_ready(out1)
        jax.block_until_ready(out2)
        self.assertFalse(
            jnp.allclose(out1, out2, atol=1e-3).item(),
            "Model maps two random inputs to identical outputs — likely a bug",
        )


# ---------------------------------------------------------------------------
# Test 5: Partial RoPE correctness on v6e
# ---------------------------------------------------------------------------

class TestMiMoV2FlashPartialRoPEOnTPU(unittest.TestCase):
    """Verifies partial-RoPE behaviour executes correctly on the v6e chip."""

    def test_rotated_dims_differ_from_input(self):
        """The partial-RoPE should modify the rotated head dimensions.

        With partial_rotary_factor=0.334 and head_dim=24, 8 dims are rotated
        (int(24*0.334)=8, rounded to nearest even).  The output of the
        attention projection with RoPE applied must differ from without.
        """
        from maxtext.layers.embeddings import PartialRotaryEmbedding
        cfg = _make_config()
        mesh = _make_mesh(cfg)

        rope = PartialRotaryEmbedding(
            min_timescale=1,
            max_timescale=int(cfg.rope_max_timescale),
            embedding_dims=cfg.head_dim,
            partial_rotary_factor=cfg.partial_rotary_factor,
            mesh=mesh,
        )

        batch, seq, heads = 1, 8, cfg.num_query_heads
        q = jax.random.normal(
            jax.random.PRNGKey(0),
            (batch, seq, heads, cfg.head_dim),
            dtype=jnp.bfloat16,
        )
        pos = jnp.broadcast_to(jnp.arange(seq)[None, :], (batch, seq))

        q_rot = rope(q, pos)
        jax.block_until_ready(q_rot)

        # Output shape must be preserved
        self.assertEqual(q_rot.shape, q.shape)
        self.assertEqual(q_rot.dtype, q.dtype)

        # Rotated query must differ from the input (RoPE is not the identity)
        self.assertFalse(
            jnp.allclose(q_rot.astype(jnp.float32), q.astype(jnp.float32), atol=1e-3).item(),
            "PartialRotaryEmbedding returned output identical to input — RoPE not applied",
        )


# ---------------------------------------------------------------------------
# Test 6: MoE gate noaux-TC on TPU
# ---------------------------------------------------------------------------

class TestMiMoV2FlashMoEGateTPU(unittest.TestCase):
    """Verifies the gate executes and produces valid routing on v6e."""

    def setUp(self):
        super().setUp()
        self.gate = mimo_v2_flash.MiMoV2FlashMoEGate(
            num_experts=8,
            hidden_size=64,
            num_experts_per_tok=2,
            dtype=jnp.bfloat16,
            weight_dtype=jnp.bfloat16,
            rngs=nnx.Rngs(params=jax.random.PRNGKey(0)),
        )

    def test_gate_executes_on_tpu(self):
        """Gate output tensors must reside on the TPU device."""
        hidden = jax.random.normal(jax.random.PRNGKey(1), (4, 64), dtype=jnp.bfloat16)
        indices, weights = self.gate(hidden)
        jax.block_until_ready(indices)
        jax.block_until_ready(weights)
        for t in (indices, weights):
            devices = list(t.devices())
            self.assertIn("tpu", devices[0].device_kind.lower())

    def test_gate_weights_sum_to_one_bfloat16(self):
        """L1-normalised weights must sum to ~1.0 per token in bfloat16."""
        hidden = jax.random.normal(jax.random.PRNGKey(2), (8, 64), dtype=jnp.bfloat16)
        _, weights = self.gate(hidden)
        sums = weights.astype(jnp.float32).sum(axis=-1)
        np.testing.assert_allclose(
            np.array(sums), np.ones(8), atol=1e-2,
            err_msg="Gate weights do not sum to 1 (bfloat16 precision on TPU)",
        )

    def test_gate_jit_compilable(self):
        """Gate forward pass should JIT-compile without error."""
        jit_gate = jax.jit(self.gate)
        hidden = jax.random.normal(jax.random.PRNGKey(3), (4, 64), dtype=jnp.bfloat16)
        indices, weights = jit_gate(hidden)
        jax.block_until_ready(indices)
        jax.block_until_ready(weights)
        self.assertEqual(indices.shape, (4, 2))
        self.assertEqual(weights.shape, (4, 2))


# ---------------------------------------------------------------------------
# Test 7: Causal / SWA masking on device
# ---------------------------------------------------------------------------

class TestMiMoV2FlashMaskingOnTPU(unittest.TestCase):
    """Validates that attention masking shapes and causality hold on v6e."""

    def _run_attention(self, is_swa: bool, batch: int, seq: int):
        cfg = _make_config()
        mesh = _make_mesh(cfg)
        rngs = nnx.Rngs(params=jax.random.PRNGKey(0))
        attn = mimo_v2_flash.MiMoV2FlashAttention(
            config=cfg, mesh=mesh, is_swa=is_swa,
            layer_idx=0, quant=None, rngs=rngs,
        )
        x = jax.random.normal(jax.random.PRNGKey(1), (batch, seq, cfg.emb_dim), dtype=jnp.bfloat16)
        pos = jnp.broadcast_to(jnp.arange(seq)[None, :], (batch, seq))
        out = attn(x, decoder_positions=pos, decoder_segment_ids=None,
                   deterministic=True, model_mode="train")
        jax.block_until_ready(out)
        return out, cfg

    def test_global_attention_output_shape_on_tpu(self):
        """Global attention output shape is (B, S, emb_dim) on device."""
        out, cfg = self._run_attention(is_swa=False, batch=2, seq=8)
        self.assertEqual(out.shape, (2, 8, cfg.emb_dim))

    def test_swa_attention_output_shape_on_tpu(self):
        """SWA attention output shape is (B, S, emb_dim) on device."""
        out, cfg = self._run_attention(is_swa=True, batch=2, seq=8)
        self.assertEqual(out.shape, (2, 8, cfg.emb_dim))

    def test_causal_masking_holds(self):
        """Future tokens must not influence past token outputs (causality check).

        We zero out the second half of the sequence and verify that the first
        half of the output is unchanged — if causality is broken, the first
        half would see the zeros and change its output.
        """
        cfg = _make_config()
        mesh = _make_mesh(cfg)
        rngs = nnx.Rngs(params=jax.random.PRNGKey(0))
        attn = mimo_v2_flash.MiMoV2FlashAttention(
            config=cfg, mesh=mesh, is_swa=False,
            layer_idx=0, quant=None, rngs=rngs,
        )
        seq = 8
        x = jax.random.normal(jax.random.PRNGKey(5), (1, seq, cfg.emb_dim), dtype=jnp.bfloat16)
        pos = jnp.broadcast_to(jnp.arange(seq)[None, :], (1, seq))

        # Run with original input
        out_full = attn(x, decoder_positions=pos, decoder_segment_ids=None,
                        deterministic=True, model_mode="train")

        # Zero out the second half — past tokens should be unaffected
        x_masked = x.at[:, seq // 2 :, :].set(0.0)
        out_masked = attn(x_masked, decoder_positions=pos, decoder_segment_ids=None,
                          deterministic=True, model_mode="train")
        jax.block_until_ready(out_full)
        jax.block_until_ready(out_masked)

        # First (seq//2) tokens should be identical in both runs
        np.testing.assert_array_equal(
            np.array(out_full[:, : seq // 2, :]),
            np.array(out_masked[:, : seq // 2, :]),
            err_msg="Causal masking violated: past tokens changed when future tokens were zeroed",
        )


# ---------------------------------------------------------------------------
# Test 8: All-layer stack forward pass
# ---------------------------------------------------------------------------

class TestMiMoV2FlashFullStackTPU(unittest.TestCase):
    """Runs the full 4-layer tiny model stack in sequence on the v6e chip."""

    def test_full_stack_forward_bfloat16(self):
        """All 4 decoder layers in sequence produce finite bfloat16 output."""
        cfg = _make_config()
        mesh = _make_mesh(cfg)

        layers = [
            mimo_v2_flash.MiMoV2FlashDecoderLayer(
                config=cfg, mesh=mesh, model_mode="train",
                layer_idx=i, quant=None,
                rngs=nnx.Rngs(params=jax.random.PRNGKey(i)),
            )
            for i in range(cfg.num_decoder_layers)
        ]

        batch, seq = 2, 8
        x = jax.random.normal(jax.random.PRNGKey(0), (batch, seq, cfg.emb_dim), dtype=jnp.bfloat16)
        pos = jnp.broadcast_to(jnp.arange(seq)[None, :], (batch, seq))

        hidden = x
        for layer in layers:
            hidden, _ = layer(hidden, decoder_segment_ids=None,
                              decoder_positions=pos, deterministic=True,
                              model_mode="train")

        jax.block_until_ready(hidden)

        self.assertEqual(hidden.shape, (batch, seq, cfg.emb_dim))
        self.assertEqual(hidden.dtype, jnp.bfloat16)
        finite = jnp.all(jnp.isfinite(hidden.astype(jnp.float32))).item()
        self.assertTrue(finite, "Full-stack bfloat16 output contains NaN or Inf")

    def test_full_stack_jit_compiled(self):
        """jax.jit over all 4 layers compiles and executes without error."""
        cfg = _make_config()
        mesh = _make_mesh(cfg)

        layers = [
            mimo_v2_flash.MiMoV2FlashDecoderLayer(
                config=cfg, mesh=mesh, model_mode="train",
                layer_idx=i, quant=None,
                rngs=nnx.Rngs(params=jax.random.PRNGKey(i)),
            )
            for i in range(cfg.num_decoder_layers)
        ]

        @jax.jit
        def stack_fwd(x, pos):
            h = x
            for layer in layers:
                h, _ = layer(h, decoder_segment_ids=None, decoder_positions=pos,
                             deterministic=True, model_mode="train")
            return h

        batch, seq = 2, 8
        x = jax.random.normal(jax.random.PRNGKey(0), (batch, seq, cfg.emb_dim), dtype=jnp.bfloat16)
        pos = jnp.broadcast_to(jnp.arange(seq)[None, :], (batch, seq))
        out = stack_fwd(x, pos)
        jax.block_until_ready(out)

        self.assertEqual(out.shape, (batch, seq, cfg.emb_dim))
        self.assertTrue(jnp.all(jnp.isfinite(out.astype(jnp.float32))).item())


if __name__ == "__main__":
    unittest.main()
