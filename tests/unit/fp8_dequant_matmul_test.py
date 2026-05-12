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

"""Unit tests for fp8_dequant_matmul.py (Phase B Pallas kernel).

Tests use interpret=True to run on CPU without Pallas TPU compilation.
Correctness is verified against the Phase A reference (_block_dequant_fp8 + einsum).
"""

import unittest

import jax
import jax.numpy as jnp
import numpy as np
import ml_dtypes

from maxtext.kernels.fp8_dequant_matmul import fp8_moe_matmul
from maxtext.models.mimo_v2_flash import _block_dequant_fp8


def _make_fp8_tensors(E, K, N, tile_k=128, tile_n=128, seed=42):
    """Create random FP8 weight + float32 scale_inv tensors for testing."""
    rng = np.random.default_rng(seed)
    # Generate random BF16 weights then quantize per-block.
    w_bf16 = rng.standard_normal((E, K, N)).astype(np.float32) * 0.1
    k_blocks = K // tile_k
    n_blocks = N // tile_n
    # Compute per-block max for quantization.
    w_reshaped = w_bf16.reshape(E, k_blocks, tile_k, n_blocks, tile_n)
    block_max = np.abs(w_reshaped).max(axis=(2, 4), keepdims=True) + 1e-10
    # FP8 E4M3FN max representable value = 448.0.
    fp8_max = 448.0
    scale_inv = block_max[:, :, 0, :, 0] / fp8_max   # (E, k_blocks, n_blocks)
    w_quantized = np.clip(w_bf16 / (scale_inv[:, :, None, :, None] + 1e-10),
                          -fp8_max, fp8_max)
    w_fp8 = w_quantized.astype(ml_dtypes.float8_e4m3fn)
    return (
        jnp.array(w_fp8),           # (E, K, N) float8_e4m3fn
        jnp.array(scale_inv.astype(np.float32)),  # (E, k_blocks, n_blocks)
    )


class TestFp8MoeMatmulNonBatched(unittest.TestCase):
    """Tests for fp8_moe_matmul with tokens_batched=False (gate/up projection)."""

    def setUp(self):
        # Tiny dims: E=2 experts, K=256 hidden, N=128 intermediate, T=4 tokens.
        self.E, self.K, self.N, self.T = 2, 256, 128, 4
        self.tile_k, self.tile_n = 128, 128
        self.weight_fp8, self.scale_inv = _make_fp8_tensors(
            self.E, self.K, self.N, self.tile_k, self.tile_n
        )
        rng = np.random.default_rng(0)
        self.tokens = jnp.array(
            rng.standard_normal((self.T, self.K)).astype(np.float32) * 0.1,
            dtype=jnp.bfloat16,
        )

    def test_output_shape(self):
        out = fp8_moe_matmul(
            self.tokens, self.weight_fp8, self.scale_inv, interpret=True
        )
        self.assertEqual(out.shape, (self.E, self.T, self.N))

    def test_output_dtype(self):
        out = fp8_moe_matmul(
            self.tokens, self.weight_fp8, self.scale_inv, interpret=True
        )
        self.assertEqual(out.dtype, jnp.bfloat16)

    def test_matches_phase_a_reference(self):
        """Pallas output must match Phase A _block_dequant_fp8 + einsum."""
        out_pallas = fp8_moe_matmul(
            self.tokens, self.weight_fp8, self.scale_inv, interpret=True
        )
        # Phase A reference.
        w_bf16 = _block_dequant_fp8(self.weight_fp8, self.scale_inv, self.tile_k, self.tile_n)
        tokens_f32 = self.tokens.astype(jnp.float32)
        out_ref = jnp.einsum("th,ehi->eti", tokens_f32, w_bf16.astype(jnp.float32))
        out_ref = out_ref.astype(jnp.bfloat16)

        np.testing.assert_allclose(
            np.array(out_pallas, dtype=np.float32),
            np.array(out_ref, dtype=np.float32),
            rtol=1e-2, atol=1e-3,
            err_msg="Pallas output diverges from Phase A reference.",
        )

    def test_output_is_finite(self):
        out = fp8_moe_matmul(
            self.tokens, self.weight_fp8, self.scale_inv, interpret=True
        )
        self.assertTrue(jnp.all(jnp.isfinite(out)).item())


class TestFp8MoeMatmulBatched(unittest.TestCase):
    """Tests for fp8_moe_matmul with tokens_batched=True (down projection)."""

    def setUp(self):
        # wo: (E, I, H) — down projection; tokens: (E, T, I)
        # Using E=2, I=128 (K), H=256 (N), T=4 tokens.
        self.E, self.K, self.N, self.T = 2, 128, 256, 4
        self.tile_k, self.tile_n = 128, 128
        self.weight_fp8, self.scale_inv = _make_fp8_tensors(
            self.E, self.K, self.N, self.tile_k, self.tile_n
        )
        rng = np.random.default_rng(1)
        self.tokens = jnp.array(
            rng.standard_normal((self.E, self.T, self.K)).astype(np.float32) * 0.1,
            dtype=jnp.bfloat16,
        )

    def test_output_shape(self):
        out = fp8_moe_matmul(
            self.tokens, self.weight_fp8, self.scale_inv,
            tokens_batched=True, interpret=True,
        )
        self.assertEqual(out.shape, (self.E, self.T, self.N))

    def test_output_dtype(self):
        out = fp8_moe_matmul(
            self.tokens, self.weight_fp8, self.scale_inv,
            tokens_batched=True, interpret=True,
        )
        self.assertEqual(out.dtype, jnp.bfloat16)

    def test_matches_phase_a_reference(self):
        """Batched Pallas output must match Phase A _block_dequant_fp8 + einsum."""
        out_pallas = fp8_moe_matmul(
            self.tokens, self.weight_fp8, self.scale_inv,
            tokens_batched=True, interpret=True,
        )
        # Phase A reference (down-proj einsum: "eti,eih->eth").
        w_bf16 = _block_dequant_fp8(self.weight_fp8, self.scale_inv, self.tile_k, self.tile_n)
        tokens_f32 = self.tokens.astype(jnp.float32)
        out_ref = jnp.einsum("eti,eih->eth", tokens_f32, w_bf16.astype(jnp.float32))
        out_ref = out_ref.astype(jnp.bfloat16)

        np.testing.assert_allclose(
            np.array(out_pallas, dtype=np.float32),
            np.array(out_ref, dtype=np.float32),
            rtol=1e-2, atol=1e-3,
            err_msg="Pallas batched output diverges from Phase A reference.",
        )


class TestFp8MoeMatmulEdgeCases(unittest.TestCase):
    """Edge-case checks."""

    def test_invalid_k_raises(self):
        weight_fp8, scale_inv = _make_fp8_tensors(1, 256, 128)
        tokens = jnp.zeros((4, 256), dtype=jnp.bfloat16)
        with self.assertRaises(ValueError):
            fp8_moe_matmul(tokens, weight_fp8, scale_inv, tile_k=64, interpret=True)

    def test_invalid_n_raises(self):
        weight_fp8, scale_inv = _make_fp8_tensors(1, 256, 128)
        tokens = jnp.zeros((4, 256), dtype=jnp.bfloat16)
        with self.assertRaises(ValueError):
            fp8_moe_matmul(tokens, weight_fp8, scale_inv, tile_n=64, interpret=True)

    def test_single_token(self):
        weight_fp8, scale_inv = _make_fp8_tensors(2, 256, 128)
        tokens = jnp.zeros((1, 256), dtype=jnp.bfloat16)
        out = fp8_moe_matmul(tokens, weight_fp8, scale_inv, interpret=True)
        self.assertEqual(out.shape, (2, 1, 128))


if __name__ == "__main__":
    unittest.main()
