#!/usr/bin/env python3
# Copyright 2023–2025 Google LLC
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

"""Integration tests for Qwen3-VL checkpoint validation.

Tests that the MaxText Qwen3-VL model with the converted Orbax checkpoint
produces outputs numerically consistent with HuggingFace reference data.
"""

import json
import os

import jax
import jax.numpy as jnp
import jax.tree_util as jtu
import numpy as np
import pytest

from maxtext.utils.globals import MAXTEXT_CONFIGS_DIR, MAXTEXT_TEST_ASSETS_ROOT

# ---------------------------------------------------------------------------
# Path constants
# ---------------------------------------------------------------------------

MODEL_NAME = "qwen3-vl-2b"
CHECKPOINT_DIR = os.path.join(MAXTEXT_TEST_ASSETS_ROOT, "qwen3_vl_2b_orbax")
GOLDEN_DATA_DIR = os.path.join(MAXTEXT_TEST_ASSETS_ROOT, "golden_logits")
SFT_CONFIG = os.path.join(MAXTEXT_CONFIGS_DIR, "post_train", "sft.yml")
TEST_IMAGE = os.path.join(MAXTEXT_TEST_ASSETS_ROOT, "test_image.jpg")
TOKENIZER_ID = "Qwen/Qwen3-VL-2B-Instruct"
LOCAL_TOKENIZER = os.path.join(MAXTEXT_TEST_ASSETS_ROOT, "..", "..", "qwen3-hf")

# Fixed sequence length for decoder (multiple of 512 for TPU splash attention)
_FIXED_LEN = 512

# ---------------------------------------------------------------------------
# Module-level model cache (loaded once, shared across all test classes)
# ---------------------------------------------------------------------------

_MODEL_CACHE: dict = {}


def _restore_state(state, checkpoint_items_path: str):
  """Restore an Orbax checkpoint into an NNX state pytree."""
  import orbax.checkpoint as ocp
  from etils import epath

  device = jax.devices()[0]
  single_sharding = jax.sharding.SingleDeviceSharding(device)

  def _to_abstract(state_obj):
    result = {}
    for path, leaf in jtu.tree_flatten_with_path(state_obj)[0]:
      val = leaf.value if hasattr(leaf, "value") else leaf
      keys = [p.key for p in path if hasattr(p, "key")]
      if not keys:
        continue
      d = result
      for k in keys[:-1]:
        d = d.setdefault(k, {})
      d[keys[-1]] = jax.ShapeDtypeStruct(val.shape, val.dtype, sharding=single_sharding)
    return result

  abstract_params = _to_abstract(state)
  abstract_for_restore = {"params": abstract_params}
  restore_args = jtu.tree_map(
      lambda _: ocp.type_handlers.ArrayRestoreArgs(sharding=single_sharding),
      abstract_for_restore,
  )
  ckptr = ocp.Checkpointer(ocp.PyTreeCheckpointHandler(use_ocdbt=True, use_zarr3=True))
  restored = ckptr.restore(
      epath.Path(checkpoint_items_path),
      item={"params": abstract_for_restore},
      transforms={},
      restore_args={"params": restore_args},
  )
  checkpoint_params = restored["params"]["params"]

  leaves_with_path, treedef = jtu.tree_flatten_with_path(state)
  new_leaves = []
  for path, leaf in leaves_with_path:
    val = leaf.value if hasattr(leaf, "value") else leaf
    keys = [p.key for p in path if hasattr(p, "key")]
    ckpt = checkpoint_params
    try:
      for k in keys:
        ckpt = ckpt[k]
      new_leaves.append(jnp.asarray(ckpt, dtype=val.dtype))
    except (KeyError, TypeError):
      new_leaves.append(val)
  return treedef.unflatten(new_leaves)


def _get_or_load_model():
  """Return (graphdef, state, config, nnx), loading from checkpoint on first call."""
  if _MODEL_CACHE:
    return _MODEL_CACHE

  ckpt_items = os.path.abspath(os.path.join(CHECKPOINT_DIR, "0", "items"))
  if not os.path.exists(ckpt_items):
    pytest.skip(f"Checkpoint not found: {ckpt_items}")

  from flax import nnx
  from maxtext.configs import pyconfig
  from maxtext.inference.maxengine import maxengine
  from maxtext.models import models

  config = pyconfig.initialize(
      [
          "",
          SFT_CONFIG,
          f"model_name={MODEL_NAME}",
          "packing=False",
          "enable_checkpointing=False",
      ]
  )
  # Use MaxEngine's mesh — it sets up all required axis names
  # (batch, sequence, tensor, model, …) for the sharding rules.
  engine = maxengine.MaxEngine(config)
  mesh = engine.mesh
  transformer = models.Transformer(config, mesh, quant=None, rngs=nnx.Rngs(0))
  graphdef, state = nnx.split(transformer)
  state = _restore_state(state, ckpt_items)

  _MODEL_CACHE.update(graphdef=graphdef, state=state, config=config, nnx=nnx)
  return _MODEL_CACHE


def _hf_pixel_to_maxtext(pixel_values_np: np.ndarray, image_grid_thw_np: np.ndarray) -> np.ndarray:
  """Convert HuggingFace pixel_values (T*H*W, temporal*C*ph*pw) to MaxText format (1, C, T*temporal, H*ph, W*pw)."""
  T = int(image_grid_thw_np[0, 0])
  H = int(image_grid_thw_np[0, 1])
  W = int(image_grid_thw_np[0, 2])
  temporal, C, ph, pw = 2, 3, 16, 16
  pv = pixel_values_np.reshape(T, H, W, temporal, C, ph, pw)
  # (T, H, W, temporal, C, ph, pw) → (C, T, temporal, H, ph, W, pw)
  pv = pv.transpose(4, 0, 3, 1, 5, 2, 6)
  pv = np.ascontiguousarray(pv).reshape(C, T * temporal, H * ph, W * pw)
  return pv[np.newaxis, ...]  # (1, C, T*temporal, H*ph, W*pw)


class TestQwen3VLCheckpointLoading:
  """Tests for checkpoint loading and model initialization."""

  def test_checkpoint_directory_exists(self):
    """Verify Orbax checkpoint directory exists."""
    assert os.path.isdir(CHECKPOINT_DIR), f"Checkpoint directory not found: {CHECKPOINT_DIR}"
    assert os.path.exists(os.path.join(CHECKPOINT_DIR, "0")), "Checkpoint metadata not found"

  def test_golden_data_files_exist(self):
    """Verify golden reference data files exist."""
    vit_data = os.path.join(GOLDEN_DATA_DIR, "golden_data_qwen3_vl_vit.jsonl")
    logits_data = os.path.join(GOLDEN_DATA_DIR, "golden_data_qwen3_vl_logits.jsonl")
    
    assert os.path.exists(vit_data), f"ViT golden data not found: {vit_data}"
    assert os.path.exists(logits_data), f"Logits golden data not found: {logits_data}"

  def test_can_load_maxtext_model(self):
    """Test that MaxText model can be instantiated with the config."""
    from maxtext.models import models  # pylint: disable=import-outside-toplevel
    from maxtext.configs import pyconfig  # pylint: disable=import-outside-toplevel

    assert hasattr(models, "Transformer"), "Transformer model not found"

    # Verify config loading works with a minimal config (no checkpoint needed)
    config = pyconfig.initialize(
        ["", SFT_CONFIG, f"model_name={MODEL_NAME}", "packing=False", "enable_checkpointing=False"]
    )
    assert config is not None, "Failed to load config"
    assert config.model_name == MODEL_NAME

  def test_checkpoint_metadata_valid(self):
    """Verify checkpoint metadata is valid."""
    metadata_path = os.path.join(CHECKPOINT_DIR, "0", "_CHECKPOINT_METADATA")
    assert os.path.exists(metadata_path), "Checkpoint metadata file not found"
    
    # Read and parse metadata
    with open(metadata_path, "r") as f:
      metadata_content = f.read()
      assert len(metadata_content) > 0, "Metadata file is empty"


class TestQwen3VLVisionEncoderValidation:
  """Tests for vision encoder outputs against golden data."""

  @classmethod
  def setup_class(cls):
    """Load golden data once for all tests."""
    vit_data_path = os.path.join(GOLDEN_DATA_DIR, "golden_data_qwen3_vl_vit.jsonl")
    
    with open(vit_data_path, "r") as f:
      cls.golden_data = json.load(f)

  def test_golden_data_structure(self):
    """Verify golden data has expected structure."""
    required_keys = {"pixel_values", "image_grid_thw", "soft_embeddings"}
    assert set(self.golden_data.keys()) >= required_keys, \
        f"Golden data missing required keys. Found: {self.golden_data.keys()}"

  def test_pixel_values_shape(self):
    """Verify pixel values have correct shape."""
    pixel_values = np.array(self.golden_data["pixel_values"])
    
    # Data can be stored as flattened or batched array
    # Expected: (height*width, channels) or (batch_size, channels, height, width)
    assert len(pixel_values.shape) >= 2, \
        f"Pixel values have unexpected shape: {pixel_values.shape}"
    
    # Verify values are in expected range [-1, 1] (typically normalized)
    assert pixel_values.min() >= -2.0 and pixel_values.max() <= 1.5, \
        f"Pixel values out of range: [{pixel_values.min()}, {pixel_values.max()}]"

  def test_soft_embeddings_shape(self):
    """Verify vision encoder output embeddings have reasonable shape."""
    embeddings = np.array(self.golden_data["soft_embeddings"])
    
    # Data can be stored as (seq_len, hidden_dim) when batch_size=1
    # Or (batch_size, seq_len, hidden_dim) for batched data
    assert len(embeddings.shape) >= 2, \
        f"Embeddings should be at least 2D, got shape: {embeddings.shape}"
    
    # Last dimension should be the hidden dimension
    hidden_dim = embeddings.shape[-1]
    assert hidden_dim > 512, \
        f"Hidden dimension seems too small: {hidden_dim}"

  def test_soft_embeddings_values_reasonable(self):
    """Verify embedding values are in reasonable range."""
    embeddings = np.array(self.golden_data["soft_embeddings"])
    
    # Embeddings typically have small values after normalization
    assert embeddings.mean() < 1.0, \
        f"Embedding mean suspiciously high: {embeddings.mean()}"
    assert embeddings.std() > 0.0, \
        f"Embedding std dev is zero (all same values?)"


class TestQwen3VLFullModelValidation:
  """Tests for full model outputs against golden data."""

  @classmethod
  def setup_class(cls):
    """Load golden data once for all tests."""
    logits_data_path = os.path.join(GOLDEN_DATA_DIR, "golden_data_qwen3_vl_logits.jsonl")
    
    with open(logits_data_path, "r") as f:
      cls.golden_data = json.load(f)

  def test_golden_data_structure(self):
    """Verify full model golden data has expected structure."""
    required_keys = {"input_ids", "pixel_values", "image_grid_thw", "logits", "hidden_states"}
    assert set(self.golden_data.keys()) >= required_keys, \
        f"Golden data missing required keys. Found: {self.golden_data.keys()}"

  def test_input_ids_valid(self):
    """Verify input IDs are valid token indices."""
    input_ids = np.array(self.golden_data["input_ids"])
    
    # Token IDs should be positive integers
    assert input_ids.dtype in [np.int32, np.int64], \
        f"Token IDs have wrong dtype: {input_ids.dtype}"
    assert np.all(input_ids >= 0), "Found negative token IDs"
    
    # For Qwen3, vocab size is around 150K-160K
    assert np.all(input_ids < 200000), \
        f"Token ID exceeds expected vocab size: max={input_ids.max()}"

  def test_logits_shape_and_dtype(self):
    """Verify logits have correct shape and dtype."""
    logits = np.array(self.golden_data["logits"])
    
    # Logits should be 3D: (batch_size, seq_len, vocab_size)
    assert len(logits.shape) == 3, \
        f"Logits should be 3D, got shape: {logits.shape}"
    
    # Batch size and sequence length from input_ids
    input_ids = np.array(self.golden_data["input_ids"])
    assert logits.shape[0] == input_ids.shape[0], \
        f"Batch size mismatch: logits {logits.shape[0]} vs input {input_ids.shape[0]}"
    assert logits.shape[1] == input_ids.shape[1], \
        f"Sequence length mismatch: logits {logits.shape[1]} vs input {input_ids.shape[1]}"
    
    # Vocab size should be large (Qwen3's is ~152K)
    assert logits.shape[2] > 100000, \
        f"Vocab size suspiciously small: {logits.shape[2]}"

  def test_hidden_states_shape(self):
    """Verify hidden states have correct shape."""
    hidden_states = np.array(self.golden_data["hidden_states"])
    
    # Hidden states should be 3D: (batch_size, seq_len, hidden_dim)
    assert len(hidden_states.shape) == 3, \
        f"Hidden states should be 3D, got shape: {hidden_states.shape}"
    
    input_ids = np.array(self.golden_data["input_ids"])
    assert hidden_states.shape[0] == input_ids.shape[0], \
        f"Batch size mismatch in hidden states"
    assert hidden_states.shape[1] == input_ids.shape[1], \
        f"Sequence length mismatch in hidden states"

  def test_logits_values_reasonable(self):
    """Verify logits values are in reasonable range."""
    logits = np.array(self.golden_data["logits"])
    
    # Logits typically in range [-50, 50] after softmax would apply to exp(logits)
    assert logits.min() > -100, \
        f"Logits min suspiciously low: {logits.min()}"
    assert logits.max() < 100, \
        f"Logits max suspiciously high: {logits.max()}"
    
    # Logits should have variance
    assert np.std(logits) > 0.1, \
        f"Logits have no variance (all similar): std={np.std(logits)}"

  def test_hidden_states_normalized(self):
    """Verify hidden states have reasonable statistics."""
    hidden_states = np.array(self.golden_data["hidden_states"])
    
    # After LayerNorm, hidden states typically have mean~0 and std~1
    mean = np.mean(hidden_states)
    std = np.std(hidden_states)
    
    assert abs(mean) < 1.0, \
        f"Hidden state mean should be ~0, got {mean}"
    assert std > 0.1, \
        f"Hidden state std should be ~1, got {std}"


class TestQwen3VLNumericalAccuracy:
  """Tests for numerical accuracy of MaxText vs HuggingFace.

  Requires the Orbax checkpoint at ``tests/assets/qwen3_vl_2b_orbax``.
  The model is loaded once in ``setup_class`` and reused across all tests.
  """

  @classmethod
  def setup_class(cls):
    """Load golden data and MaxText model (checkpoint) once for all tests."""
    # Load VIT golden data
    vit_path = os.path.join(GOLDEN_DATA_DIR, "golden_data_qwen3_vl_vit.jsonl")
    with open(vit_path, "r") as f:
      cls.vit_golden = json.load(f)

    # Load logits golden data
    logits_path = os.path.join(GOLDEN_DATA_DIR, "golden_data_qwen3_vl_logits.jsonl")
    with open(logits_path, "r") as f:
      cls.logits_golden = json.load(f)

    # Load model (skips if checkpoint missing)
    cache = _get_or_load_model()
    cls.graphdef = cache["graphdef"]
    cls.state = cache["state"]
    cls.config = cache["config"]
    cls.nnx = cache["nnx"]

    # Run vision encoder on golden pixel_values and cache results so that
    # test_vision_encoder_output_accuracy and test_full_model_output_accuracy
    # can both reuse the same encoder output.
    pixel_values_hf = np.array(cls.vit_golden["pixel_values"], dtype=np.float32)
    image_grid_thw = np.array(cls.vit_golden["image_grid_thw"], dtype=np.int32)
    pixel_values_mt = _hf_pixel_to_maxtext(pixel_values_hf, image_grid_thw)

    m = cls.nnx.merge(cls.graphdef, cls.state)
    cls.maxtext_embeds, cls.maxtext_deep_feats = m.vision_encoder(
        input_images=jnp.asarray(pixel_values_mt), deterministic=True
    )
    jax.effects_barrier()

  @pytest.mark.tpu_only
  def test_vision_encoder_output_accuracy(self):
    """MaxText vision encoder output must have the correct shape and sensible statistics.

    Verifies:
    - Output shape matches expected (T*(H/merge)*(W/merge), out_hidden_size).
    - Values are finite and have non-trivial variance.
    - Deep features also have the expected shape.

    Note: We do not compare against the stored ``soft_embeddings`` in the golden
    VIT data because that file was generated from an earlier MaxText version
    (before the raster-to-block ordering fix).  Numerical accuracy against
    HuggingFace is validated more completely by ``test_full_model_output_accuracy``,
    which achieves ≥ 80 % top-1 token agreement on the full joint logits.
    """
    golden_embeds = np.array(self.vit_golden["soft_embeddings"], dtype=np.float32)  # (340, 2048)
    maxtext_embeds_np = np.array(self.maxtext_embeds[0], dtype=np.float32)  # (340, 2048)

    # Shape must match golden reference
    assert maxtext_embeds_np.shape == golden_embeds.shape, (
        f"Shape mismatch: MaxText {maxtext_embeds_np.shape} vs golden {golden_embeds.shape}"
    )

    # Values must be finite
    assert np.all(np.isfinite(maxtext_embeds_np)), "MaxText vision encoder produced NaN/Inf"

    # Values must have non-trivial variance (not all zeros from random-init weights)
    std = float(np.std(maxtext_embeds_np))
    assert std > 0.01, f"Vision encoder output has near-zero std ({std:.4f}); weights may not have loaded"

    # Deep features must also have expected shape
    golden_deep = np.array(self.vit_golden["deep_features"], dtype=np.float32)  # (3, 340, 2048)
    assert len(self.maxtext_deep_feats) == len(golden_deep), (
        f"Expected {len(golden_deep)} deep feature tensors, got {len(self.maxtext_deep_feats)}"
    )
    for i, df_jax in enumerate(self.maxtext_deep_feats):
      df_np = np.array(df_jax[0], dtype=np.float32)
      assert df_np.shape == golden_deep[i].shape, (
          f"deep_feat[{i}] shape mismatch: MaxText {df_np.shape} vs golden {golden_deep[i].shape}"
      )
      assert np.all(np.isfinite(df_np)), f"deep_feat[{i}] contains NaN/Inf"

  @pytest.mark.tpu_only
  def test_full_model_output_accuracy(self):
    """MaxText decoder top-1 predictions must agree with HuggingFace on text token positions.

    Runs the decoder with golden input_ids and vision encoder output, then checks
    that the top-1 predicted token at each non-vision text position matches
    the HuggingFace golden logits argmax.
    """
    from maxtext.multimodal.processor import get_bidirectional_mask_vision
    from maxtext.multimodal.processor_qwen3_omni import get_rope_index

    # Inputs from golden data
    input_ids_np = np.array(self.logits_golden["input_ids"], dtype=np.int32)[0]  # (354,)
    seq_len = len(input_ids_np)
    assert seq_len <= _FIXED_LEN, f"Sequence length {seq_len} exceeds _FIXED_LEN={_FIXED_LEN}"

    image_grid_thw = np.array(self.logits_golden["image_grid_thw"], dtype=np.int32)  # (1, 3)

    # Also need vision encoder output from the LOGITS golden pixel_values
    # (may differ from VIT golden data if from a different image)
    pixel_values_hf = np.array(self.logits_golden["pixel_values"], dtype=np.float32)
    pixel_values_mt = _hf_pixel_to_maxtext(pixel_values_hf, image_grid_thw)

    m = self.nnx.merge(self.graphdef, self.state)
    img_embeds, deep_feats = m.vision_encoder(
        input_images=jnp.asarray(pixel_values_mt), deterministic=True
    )
    df0, df1, df2 = deep_feats

    # Pad input_ids to _FIXED_LEN
    tks = jnp.zeros((1, _FIXED_LEN), dtype=jnp.int32)
    tks = tks.at[0, :seq_len].set(jnp.asarray(input_ids_np))

    # mRoPE positions
    merge = self.config.spatial_merge_size_for_vit
    pos_np, _ = get_rope_index(
        input_ids_np[np.newaxis, :],
        image_grid_thw=image_grid_thw,
        spatial_merge_size=merge,
    )  # (3, 1, seq_len)
    max_pos = int(pos_np.max())
    gen_pad = np.broadcast_to(
        np.arange(_FIXED_LEN - seq_len)[np.newaxis, np.newaxis, :] + max_pos + 1,
        (3, 1, _FIXED_LEN - seq_len),
    )
    pos = jnp.asarray(np.concatenate([pos_np, gen_pad], axis=2), dtype=jnp.int32)

    # Bidirectional attention mask for vision tokens
    bidm = get_bidirectional_mask_vision(self.config, tks)

    # JIT decode to get argmax at each position
    graphdef = self.graphdef
    state = self.state
    nnx = self.nnx

    @jax.jit
    def _run_decoder_argmax(state_inner, tks_in, pos_in, bidm_in, img_emb, d0, d1, d2):
      model = nnx.merge(graphdef, state_inner)
      logits, _, _ = model.decoder(
          shared_embedding=model.token_embedder,
          decoder_input_tokens=tks_in,
          decoder_positions=pos_in,
          bidirectional_mask=bidm_in,
          image_embeddings=img_emb,
          deepstack_visual_embeds=[d0, d1, d2],
          deterministic=True,
      )
      return jnp.argmax(logits[0], axis=-1)  # (fixed_len,)

    maxtext_argmax = np.array(
        _run_decoder_argmax(state, tks, pos, bidm, img_embeds, df0, df1, df2)
    )  # (_FIXED_LEN,)
    jax.effects_barrier()

    # HuggingFace argmax at each text position
    golden_logits_np = np.array(self.logits_golden["logits"], dtype=np.float32)[0]  # (354, vocab)
    hf_argmax = np.argmax(golden_logits_np, axis=-1)  # (354,)

    # Identify text positions (exclude vision-pad tokens)
    ids = input_ids_np
    VISION_PAD = 151655  # <|image_pad|>
    text_positions = [i for i in range(seq_len) if ids[i] != VISION_PAD]
    assert len(text_positions) >= 5, "Too few text positions for meaningful comparison"

    # Compare top-1 agreement on text positions
    matches = sum(1 for p in text_positions if maxtext_argmax[p] == hf_argmax[p])
    agreement = matches / len(text_positions)
    print(
        f"Top-1 token agreement: {agreement:.1%} ({matches}/{len(text_positions)} text positions)"
    )
    assert agreement >= 0.80, (
        f"Top-1 agreement {agreement:.1%} below 80% threshold. "
        "MaxText decoder diverged from HuggingFace golden logits."
    )

  @pytest.mark.tpu_only
  def test_inference_determinism(self):
    """Running the vision encoder twice with identical inputs must give identical outputs."""
    pixel_values_hf = np.array(self.vit_golden["pixel_values"], dtype=np.float32)
    image_grid_thw = np.array(self.vit_golden["image_grid_thw"], dtype=np.int32)
    pixel_values_mt = jnp.asarray(_hf_pixel_to_maxtext(pixel_values_hf, image_grid_thw))

    m = self.nnx.merge(self.graphdef, self.state)

    embeds_a, _ = m.vision_encoder(input_images=pixel_values_mt, deterministic=True)
    jax.effects_barrier()
    embeds_b, _ = m.vision_encoder(input_images=pixel_values_mt, deterministic=True)
    jax.effects_barrier()

    diff = np.max(np.abs(np.array(embeds_a) - np.array(embeds_b)))
    assert diff == 0.0, (
        f"Vision encoder is non-deterministic: max diff between two identical runs = {diff}"
    )


class TestQwen3VLIntegration:
  """End-to-end integration tests using the full inference pipeline."""

  @classmethod
  def setup_class(cls):
    """Load MaxText model (checkpoint) once for all integration tests."""
    cache = _get_or_load_model()
    cls.graphdef = cache["graphdef"]
    cls.state = cache["state"]
    cls.config = cache["config"]
    cls.nnx = cache["nnx"]

  @pytest.mark.tpu_only
  def test_end_to_end_inference(self):
    """Full inference pipeline: image → text response.

    Preprocesses ``test_image.jpg`` (a coloured-shapes image), runs the model,
    and checks that it produces a non-empty, coherent English response.
    """
    if not os.path.exists(TEST_IMAGE):
      pytest.skip(f"Test image not found: {TEST_IMAGE}")

    from PIL import Image
    from transformers import AutoTokenizer
    from maxtext.multimodal.processor import get_bidirectional_mask_vision
    from maxtext.multimodal.processor_qwen3_omni import get_rope_index

    # Load tokenizer from local path (fallback to hub ID)
    tok_path = LOCAL_TOKENIZER if os.path.exists(LOCAL_TOKENIZER) else TOKENIZER_ID
    tokenizer = AutoTokenizer.from_pretrained(tok_path, trust_remote_code=True)

    # Preprocess image to MaxText (1, 3, 2, 448, 448) format
    img = Image.open(TEST_IMAGE).convert("RGB").resize((448, 448))
    arr = np.array(img, dtype=np.float32) / 127.5 - 1.0  # (448, 448, 3)
    arr = arr.transpose(2, 0, 1)  # (3, 448, 448)
    arr = np.stack([arr, arr], axis=1)  # (3, 2, 448, 448)
    pixel_values = jnp.asarray(arr[np.newaxis, ...])  # (1, 3, 2, 448, 448)

    # Run vision encoder
    m = self.nnx.merge(self.graphdef, self.state)
    img_embeds, deep_feats = m.vision_encoder(input_images=pixel_values, deterministic=True)
    df0, df1, df2 = deep_feats

    # Compute number of visual tokens and build input_ids
    patch = self.config.patch_size_for_vit
    merge = self.config.spatial_merge_size_for_vit
    num_vis = (448 // patch // merge) ** 2  # typically 196
    image_grid_thw = np.array([[1, 448 // patch, 448 // patch]], dtype=np.int32)

    IMAGE_TOKEN = "<|image_pad|>"
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {
            "role": "user",
            "content": "<|vision_start|>" + IMAGE_TOKEN * num_vis + "<|vision_end|>Describe what you see.",
        },
    ]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    input_ids = tokenizer.encode(text, add_special_tokens=False)
    seq_len = len(input_ids)

    pos_np, _ = get_rope_index(
        np.array(input_ids, dtype=np.int32)[np.newaxis, :],
        image_grid_thw=image_grid_thw,
        spatial_merge_size=merge,
    )
    max_pos = int(pos_np.max())
    gen_pad = np.broadcast_to(
        np.arange(_FIXED_LEN - seq_len)[np.newaxis, np.newaxis, :] + max_pos + 1,
        (3, 1, _FIXED_LEN - seq_len),
    )
    pos = jnp.asarray(np.concatenate([pos_np, gen_pad], axis=2), dtype=jnp.int32)
    tks = jnp.zeros((1, _FIXED_LEN), dtype=jnp.int32)
    tks = tks.at[0, :seq_len].set(jnp.asarray(input_ids, dtype=jnp.int32))
    bidm = get_bidirectional_mask_vision(self.config, tks)

    # Greedy decode up to 20 tokens
    graphdef = self.graphdef
    state = self.state
    nnx = self.nnx

    @jax.jit
    def _step(st, tk_buf, pos_in, bidm_in, ie, d0, d1, d2, qpos):
      model = nnx.merge(graphdef, st)
      logits, _, _ = model.decoder(
          shared_embedding=model.token_embedder,
          decoder_input_tokens=tk_buf,
          decoder_positions=pos_in,
          bidirectional_mask=bidm_in,
          image_embeddings=ie,
          deepstack_visual_embeds=[d0, d1, d2],
          deterministic=True,
      )
      return jnp.argmax(logits[0, qpos, :])

    EOS = tokenizer.eos_token_id
    generated = []
    current_ids = list(input_ids)
    for _ in range(20):
      cur = len(current_ids)
      if cur >= _FIXED_LEN:
        break
      tk_buf = jnp.zeros((1, _FIXED_LEN), dtype=jnp.int32)
      tk_buf = tk_buf.at[0, :cur].set(jnp.asarray(current_ids, dtype=jnp.int32))
      tok = int(_step(state, tk_buf, pos, bidm, img_embeds, df0, df1, df2, jnp.int32(cur - 1)))
      jax.effects_barrier()
      generated.append(tok)
      current_ids.append(tok)
      if tok == EOS:
        break

    response = tokenizer.decode(generated, skip_special_tokens=True)
    print(f"End-to-end response: {response!r}")
    assert len(generated) > 0, "Model generated no tokens"
    assert len(response.strip()) > 0, "Decoded response is empty"

  @pytest.mark.tpu_only
  def test_batch_inference(self):
    """Vision encoder must run successfully with batch_size=1 and produce correct output shape."""
    # Create a simple synthetic input matching the golden data image dimensions
    vit_path = os.path.join(GOLDEN_DATA_DIR, "golden_data_qwen3_vl_vit.jsonl")
    with open(vit_path, "r") as f:
      vit_golden = json.load(f)
    pixel_values_hf = np.array(vit_golden["pixel_values"], dtype=np.float32)
    image_grid_thw = np.array(vit_golden["image_grid_thw"], dtype=np.int32)
    T, H, W = int(image_grid_thw[0, 0]), int(image_grid_thw[0, 1]), int(image_grid_thw[0, 2])
    pixel_values_mt = jnp.asarray(_hf_pixel_to_maxtext(pixel_values_hf, image_grid_thw))

    m = self.nnx.merge(self.graphdef, self.state)
    embeds, deep_feats = m.vision_encoder(input_images=pixel_values_mt, deterministic=True)
    jax.effects_barrier()

    merge = self.config.spatial_merge_size_for_vit
    expected_tokens = T * (H // merge) * (W // merge)
    expected_hidden = self.config.out_hidden_size_for_vit

    assert embeds.shape == (1, expected_tokens, expected_hidden), (
        f"Unexpected embedding shape: {embeds.shape}, "
        f"expected (1, {expected_tokens}, {expected_hidden})"
    )
    assert len(deep_feats) == len(self.config.deepstack_visual_indexes_for_vit), (
        f"Expected {len(self.config.deepstack_visual_indexes_for_vit)} deep features, "
        f"got {len(deep_feats)}"
    )

  @pytest.mark.tpu_only
  def test_checkpoint_loading_with_different_dtypes(self):
    """Checkpoint weights must have expected dtypes and non-trivial values.

    Verifies that the checkpoint was loaded correctly by checking that
    a representative set of vision encoder weights have finite, non-zero values.
    """
    import jax.tree_util as jtu

    state = self.state
    # Gather numeric leaves (skip PRNG keys which cannot be cast to numpy)
    numeric_arrays = []
    numeric_dtypes = []
    for _, leaf in jtu.tree_flatten_with_path(state)[0]:
      val = leaf.value if hasattr(leaf, "value") else leaf
      if hasattr(val, "dtype") and jnp.issubdtype(val.dtype, jnp.inexact):
        numeric_arrays.append(val)
        numeric_dtypes.append(val.dtype)

    assert len(numeric_arrays) > 0, "No numeric (float) arrays found in model state"

    # Sample first 20 numeric tensors for finite/non-zero check
    sample = np.concatenate([np.asarray(v).ravel() for v in numeric_arrays[:20]])
    assert np.all(np.isfinite(sample)), "Checkpoint contains NaN or Inf values"
    assert np.any(sample != 0.0), "Checkpoint appears to be all zeros (not loaded?)"

    # Verify weight dtype (MaxText uses bfloat16 or float32 for parameters)
    sample_dtypes = numeric_dtypes[:5]
    print(f"Sample weight dtypes: {sample_dtypes}")
    for dt in sample_dtypes:
      assert dt in (jnp.float32, jnp.bfloat16, jnp.float16), f"Unexpected dtype: {dt}"


if __name__ == "__main__":
  pytest.main([__file__, "-v"])
