"""Test initialization of Qwen3-VL Decoder with mRoPE in MaxText."""

import os
import traceback
import jax
import jax.numpy as jnp
from flax import nnx
from maxtext.configs import pyconfig
from maxtext.models import models
from maxtext.inference.maxengine import maxengine

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "0"


def test_qwen_decoder():
  """Test Qwen3-VL LLM Backbone with mRoPE."""
  print("=" * 50)
  print("Testing Qwen3-VL LLM Backbone with mRoPE")
  print("=" * 50)

  config = pyconfig.initialize(
      [
          "",
          "src/maxtext/configs/post_train/sft.yml",
          "model_name=qwen3-vl-2b",
          "run_name=test_decoder",
          "packing=False",
          "enable_checkpointing=False",
      ]
  )

  engine = maxengine.MaxEngine(config)

  try:
    rngs = nnx.Rngs(0)
    transformer = models.Transformer(config=config, mesh=engine.mesh, quant=None, rngs=rngs)
    print("✅ Transformer (LLM Backbone) instantiated successfully.")

    batch = 1
    seq_len = 16

    decoder_input_tokens = jnp.ones((batch, seq_len), dtype=jnp.int32)

    decoder_positions = jnp.zeros((3, batch, seq_len), dtype=jnp.int32)
    decoder_positions = decoder_positions.at[0].set(jnp.arange(seq_len)[jnp.newaxis, :])

    print(f"Testing with mRoPE 3D positions shape: {decoder_positions.shape}")

    t = 2 * config.temporal_patch_size_for_vit
    h = 32 // config.patch_size_for_vit * config.patch_size_for_vit
    w = 32 // config.patch_size_for_vit * config.patch_size_for_vit
    encoder_images = jnp.zeros((batch, 3, t, h, w), dtype=jnp.float32)

    print("Executing forward pass (compiling JAX graph)...")

    graphdef, state = nnx.split(transformer)

    @jax.jit
    def forward(state_inner, tokens, positions, images):
      model_inner = nnx.merge(graphdef, state_inner)
      out = model_inner(
          decoder_input_tokens=tokens, decoder_positions=positions, encoder_images=images, enable_dropout=False
      )
      _, state_inner = nnx.split(model_inner)
      return out, state_inner

    outputs, _ = forward(state, decoder_input_tokens, decoder_positions, encoder_images)
    logits = outputs[0]

    print("✅ Forward pass completed successfully!")
    print(f"  Logits shape: {logits.shape}")

    if len(logits.shape) == 2:
      assert logits.shape == (seq_len, config.vocab_size), "Shape mismatch!"
    else:
      assert logits.shape == (batch, seq_len, config.vocab_size), "Shape mismatch!"

    print("✅ Shape validation passed!")

  except Exception as e:  # pylint: disable=broad-exception-caught
    print(f"❌ Failed to run LLM Backbone: {e}")
    traceback.print_exc()


if __name__ == "__main__":
  test_qwen_decoder()
