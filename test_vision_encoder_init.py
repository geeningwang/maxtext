"""Test initialization of Qwen3-VL Vision Encoder in MaxText."""

import os
import traceback
import jax
import jax.numpy as jnp
from flax import nnx
from maxtext.configs import pyconfig
from maxtext.models import models
from maxtext.inference.maxengine import maxengine

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "0"


def test_vision_encoder(model_name):
  """Test Vision Encoder initialization."""
  print(f"\n{'='*50}\nTesting {model_name}\n{'='*50}")

  config = pyconfig.initialize(
      [
          "",
          "src/maxtext/configs/post_train/sft.yml",
          f"model_name={model_name}",
          "run_name=test_init",
          "packing=False",
          "enable_checkpointing=False",
      ]
  )

  print(f"Loaded config: {config.model_name}")

  engine = maxengine.MaxEngine(config)
  print("Engine created successfully.")

  try:
    rngs = nnx.Rngs(0)
    vision_encoder_model = models.VisionEncoder(config, engine.mesh, rngs=rngs)
    print("✅ VisionEncoder instantiated successfully.")
    print(f"  Encoder class: {vision_encoder_model.encoder_name}")
    print(f"  Projector class: {vision_encoder_model.projector_name}")

    batch = 1
    channels = 3
    t = 2 * config.temporal_patch_size_for_vit
    h = 32 // config.patch_size_for_vit * config.patch_size_for_vit
    w = 32 // config.patch_size_for_vit * config.patch_size_for_vit

    input_images = jnp.zeros((batch, channels, t, h, w), dtype=jnp.float32)
    print(f"Loaded dummy image tensor of shape: {input_images.shape}")

    print("Testing forward pass with random initialization...")
    model = models.VisionEncoder(config, engine.mesh, rngs=nnx.Rngs(0))
    graphdef, state = nnx.split(model)

    @jax.jit
    def forward(state_inner, x):
      model_inner = nnx.merge(graphdef, state_inner)
      out = model_inner(x)
      _, state_inner = nnx.split(model_inner)
      return out, state_inner

    outputs, _ = forward(state, input_images)
    embeddings, deep_feats = outputs

    print("✅ Forward pass completed!")
    print(f"  Output embeddings shape: {embeddings.shape}")
    if deep_feats is not None:
      print(f"  Deep features count: {len(deep_feats)}")
      for i, feat in enumerate(deep_feats):
        print(f"    Deep feat {i} shape: {feat.shape}")
    else:
      print("  No deep features returned.")

  except Exception as e:  # pylint: disable=broad-exception-caught
    print(f"❌ Failed to instantiate or run VisionEncoder: {e}")
    traceback.print_exc()


if __name__ == "__main__":
  test_vision_encoder("qwen3-vl-2b")
  test_vision_encoder("qwen3-vl-8b")
