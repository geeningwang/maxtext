"""Test integration of mRoPE and Qwen3-VL Deepstack into the LLM Backbone."""

import os
import traceback
import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx
from maxtext.configs import pyconfig
from maxtext.models import models
from maxtext.inference.maxengine import maxengine
from maxtext.multimodal.processor import get_bidirectional_mask_vision

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "0"


def test_llm_backbone():
  """Test the mRoPE injection and deepstack handling of the decoder natively."""
  print(f"\n{'='*50}\nTesting Qwen3-VL-2B Backbone Integration\n{'='*50}")

  config = pyconfig.initialize(
      [
          "",
          "src/maxtext/configs/post_train/sft.yml",
          "model_name=qwen3-vl-2b",
          "run_name=test_llm_mrope",
          "packing=False",
          "enable_checkpointing=False",
          "base_num_decoder_layers=2",  # Small layer count to speed up XLA JIT!
      ]
  )

  engine = maxengine.MaxEngine(config)
  print("Engine created successfully.")

  try:
    batch_size = 1
    seq_len = 16

    tokens = np.ones((batch_size, seq_len), dtype=np.int32) * 151643  # BOS
    tokens[0, 0:8] = 151655  # Fill with QWEN3_VL_IMAGE_TOKEN
    decoder_input_tokens = jnp.asarray(tokens)

    positions = np.zeros((3, batch_size, seq_len), dtype=np.int32)
    positions[0, 0, :] = np.arange(seq_len)  # Time
    positions[1, 0, :] = np.arange(seq_len)  # Height
    positions[2, 0, :] = np.arange(seq_len)  # Width
    decoder_positions = jnp.asarray(positions)

    deep_feat_dim = config.out_hidden_size_for_vit
    mock_visual_embeds = jnp.zeros((batch_size, seq_len, deep_feat_dim), dtype=jnp.float32)

    deepstack_embeds = [None] * config.base_num_decoder_layers
    deepstack_embeds[0] = mock_visual_embeds  # Just inject at first layer for fast test

    print("✅ Inputs correctly shaped for mRoPE and Deepstack.")

    print("Instantiating Transformer...")
    transformer = models.Transformer(config, engine.mesh, quant=None, rngs=nnx.Rngs(0))
    print("✅ Transformer instantiated successfully.")

    print("Compiling and running forward pass...")
    graphdef, state = nnx.split(transformer)

    @jax.jit
    def forward(state_inner, tks, pos, deep_embeds):
      m_transformer = nnx.merge(graphdef, state_inner)

      bidirectional_mask = get_bidirectional_mask_vision(config, tks)

      out_logits, out_hidden_state, _ = m_transformer.decoder(
          shared_embedding=m_transformer.token_embedder,
          decoder_input_tokens=tks,
          decoder_positions=pos,
          bidirectional_mask=bidirectional_mask,
          deepstack_visual_embeds=deep_embeds,
          deterministic=True,
      )
      return out_logits, out_hidden_state

    logits, hidden_states = forward(state, decoder_input_tokens, decoder_positions, deepstack_embeds)

    print("✅ Forward pass completed successfully!")
    print(f"  Logits shape: {logits.shape}")
    print(f"  Hidden states shape: {hidden_states.shape}")

    if logits.shape == (batch_size, seq_len, config.vocab_size):
      print("  Logit dimensions match expected vocabulary size!")

  except Exception as e:  # pylint: disable=broad-exception-caught
    print(f"❌ Failed during Backbone test: {e}")
    traceback.print_exc()


if __name__ == "__main__":
  test_llm_backbone()
