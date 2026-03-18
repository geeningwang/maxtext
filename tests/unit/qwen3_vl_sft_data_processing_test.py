# Copyright 2023–2026 Google LLC
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

"""Vision SFT data-pipeline integration unit tests for Qwen3-VL.

Verifies that ``vision_sft_preprocessing_pipeline`` produces correctly shaped
and correctly masked batches from a synthetic in-memory HuggingFace dataset.

Prerequisites (all local, no network access required):
  - Local Qwen3-VL tokenizer at ``tests/assets/qwen3_vl_2b_hf``

Run:
  pytest tests/unit/qwen3_vl_sft_data_processing_test.py -v
"""

import os
import sys
import unittest

import jax
import jax.numpy as jnp
from jax.experimental import mesh_utils
from jax.sharding import Mesh
import numpy as np
import pytest

from maxtext.configs import pyconfig
from maxtext.input_pipeline import hf_data_processing
from maxtext.input_pipeline import input_pipeline_interface
from maxtext.utils.globals import MAXTEXT_ASSETS_ROOT, MAXTEXT_CONFIGS_DIR, MAXTEXT_TEST_ASSETS_ROOT

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
MODEL_NAME = "qwen3-vl-2b"
# Local tokenizer with full Qwen3-VL vocabulary (vocab.json + special tokens)
LOCAL_TOKENIZER = os.path.join(MAXTEXT_ASSETS_ROOT, "tokenizers", "qwen3-tokenizer")
SFT_VISION_CONFIG = os.path.join(MAXTEXT_CONFIGS_DIR, "post_train", "sft-vision-qwen3vl.yml")

# Tokens per image after expansion: 1×28×28÷4 = 196
_TOKENS_PER_IMAGE = 196
# max_target_length must accommodate image tokens + chat template + short query/response
_MAX_TARGET_LENGTH = 512
# Number of synthetic examples in the fake dataset (must be ≥ batch_size)
_N_EXAMPLES = 4
# Data column names matching the sft-vision-qwen3vl.yml config
_TEXT_COLUMNS = ["query", "label"]
_IMAGE_COLUMN = "image"


def _make_fake_dataset():
  """Return an in-memory HuggingFace IterableDataset with synthetic images and text.

  Uses `.to_iterable_dataset()` so that lazy map operations (including
  ``pre_process_image_sft`` which returns a non-serializable PreprocessorOutput)
  never try to materialise results to Arrow format.
  """
  import datasets  # pylint: disable=import-outside-toplevel
  from PIL import Image  # pylint: disable=import-outside-toplevel

  rng = np.random.default_rng(42)
  fake_img = Image.fromarray(rng.integers(0, 255, (100, 150, 3), dtype=np.uint8))

  ds = datasets.Dataset.from_dict(
      {
          "query": ["What is shown in this image?"] * _N_EXAMPLES,
          # ChartQA-style: label is a list of possible answers; pipeline takes index [0]
          "label": [["A chart showing data."]] * _N_EXAMPLES,
          "image": [fake_img] * _N_EXAMPLES,
      },
      features=datasets.Features(
          {
              "query": datasets.Value("string"),
              "label": datasets.Sequence(datasets.Value("string")),
              "image": datasets.Image(),
          }
      ),
  )
  # Convert to IterableDataset so map() calls don't materialise to Arrow.
  # The production pipeline uses streaming=True (IterableDataset) for the same reason.
  return ds.to_iterable_dataset()


def _make_config():
  """Initialise a minimal config from the SFT-vision yml with local overrides."""
  if not os.path.exists(LOCAL_TOKENIZER):
    pytest.skip(f"Local tokenizer not found: {LOCAL_TOKENIZER}")
  if not os.path.exists(SFT_VISION_CONFIG):
    pytest.skip(f"SFT vision config not found: {SFT_VISION_CONFIG}")

  return pyconfig.initialize(
      ["test_runner", SFT_VISION_CONFIG],
      run_name="test-qwen3vl-sft-pipeline",
      model_name=MODEL_NAME,
      tokenizer_path=LOCAL_TOKENIZER,
      per_device_batch_size=1,
      max_target_length=_MAX_TARGET_LENGTH,
      # Disable checkpointing; we are not loading a model
      enable_checkpointing=False,
      load_parameters_path="",
      # Pipeline simplifications
      use_tunix_gradient_accumulation=False,
      gradient_accumulation_steps=1,
      enable_data_shuffling=False,
      num_epoch=1,
      skip_jax_distributed_system=True,
      base_output_directory="/tmp/test_qwen3vl_sft/",
      # Simplify to a flat 1D data-parallel mesh for single-device testing
      mesh_axes=["data"],
      logical_axis_rules=[["batch", "data"]],
      data_sharding=["data"],
  )


class TestQwen3VLVisionSFTPipeline(unittest.TestCase):
  """Tests for the vision SFT preprocessing pipeline with Qwen3-VL.

  Uses a small in-memory HuggingFace dataset with synthetic images.
  Requires the local tokenizer at ``tests/assets/qwen3_vl_2b_hf``.
  """

  @classmethod
  def setUpClass(cls):
    super().setUpClass()
    cls.config = _make_config()
    cls.dataset = _make_fake_dataset()

    mesh_shape_1d = (len(jax.devices()),)
    cls.mesh = Mesh(mesh_utils.create_device_mesh(mesh_shape_1d), cls.config.mesh_axes)
    cls.process_indices = input_pipeline_interface.get_process_loading_real_data(
        cls.config.data_sharding,
        cls.config.global_batch_size_to_load,
        cls.config.global_batch_size_to_train_on,
        cls.config.max_target_length,
        cls.mesh,
    )

    cls.train_iter = hf_data_processing.vision_sft_preprocessing_pipeline(
        dataset=cls.dataset,
        config=cls.config,
        dataloading_host_index=cls.process_indices.index(jax.process_index()),
        dataloading_host_count=len(cls.process_indices),
        global_mesh=cls.mesh,
        text_columns=_TEXT_COLUMNS,
        image_column=_IMAGE_COLUMN,
        global_batch_size=cls.config.global_batch_size_to_load,
    )
    cls.batch = next(cls.train_iter)

  # ------------------------------------------------------------------ keys

  def test_batch_has_text_keys(self):
    """Batch must contain all standard text-sequence keys."""
    for key in ("inputs", "targets", "inputs_position", "inputs_segmentation"):
      self.assertIn(key, self.batch, f"Missing key: {key}")

  def test_batch_has_segmentation_keys(self):
    """Batch must contain both targets segmentation keys."""
    self.assertIn("targets_segmentation", self.batch)

  def test_batch_has_images_key(self):
    """Batch must contain a pixel-values tensor under 'images'."""
    self.assertIn("images", self.batch)

  # ---------------------------------------------------------------- shapes

  def test_text_batch_shapes(self):
    """Token-sequence tensors must have shape (batch_size, max_target_length)."""
    B = self.config.global_batch_size_to_load
    L = _MAX_TARGET_LENGTH
    for key in ("inputs", "targets", "inputs_position", "inputs_segmentation", "targets_segmentation"):
      shape = tuple(self.batch[key].shape)
      self.assertEqual(shape, (B, L), f"Key '{key}' shape mismatch: got {shape}")

  def test_images_batch_shape(self):
    """Images tensor must be (B * max_num_images, C, T, H, W) = (B, 3, 2, 448, 448)."""
    B = self.config.global_batch_size_to_load
    N = self.config.max_num_images_per_example  # 1
    C, T, H, W = 3, 2, 448, 448
    expected = (B * N, C, T, H, W)
    actual = tuple(self.batch["images"].shape)
    self.assertEqual(actual, expected, f"Images shape mismatch: expected {expected}, got {actual}")

  # ---------------------------------------------------------- data validity

  def test_pixel_values_finite(self):
    """Pixel values must be finite (no NaN/Inf)."""
    images = np.array(self.batch["images"])
    self.assertTrue(np.all(np.isfinite(images)), "Pixel values contain NaN or Inf")

  def test_pixel_values_normalized_range(self):
    """Pixel values must lie in approximately [-1, 1] (normalised by mean=127.5, std=127.5)."""
    images = np.array(self.batch["images"])
    self.assertGreaterEqual(float(images.min()), -1.1, "Pixel values below expected minimum")
    self.assertLessEqual(float(images.max()), 1.1, "Pixel values above expected maximum")

  def test_input_ids_non_negative(self):
    """Token IDs must be non-negative integers."""
    inputs = np.array(self.batch["inputs"])
    self.assertTrue(np.all(inputs >= 0), "Found negative input token IDs")

  def test_targets_segmentation_has_nonzero_entries(self):
    """targets_segmentation must be non-zero at completion token positions."""
    tgt_seg = np.array(self.batch["targets_segmentation"])
    self.assertTrue(np.any(tgt_seg > 0), "targets_segmentation is all zeros; completion tokens not found")

  def test_completion_only_masking(self):
    """With sft_train_on_completion_only=True, at least some positions must be masked (seg=0).

    The query/prompt tokens should be masked out, so not ALL positions should be 1.
    """
    tgt_seg = np.array(self.batch["targets_segmentation"])
    # Prompt tokens should have segmentation=0
    self.assertTrue(np.any(tgt_seg == 0), "No masked positions found; prompt masking may not be working")
    # But response tokens should be present
    self.assertTrue(np.any(tgt_seg > 0), "No training positions found; response masking too aggressive")

  def test_inputs_segmentation_nonzero_after_image_expansion(self):
    """inputs_segmentation must mark real tokens (image + text) as 1, padding as 0."""
    inp_seg = np.array(self.batch["inputs_segmentation"])
    # At least 1 real position (image tokens + prompt + response)
    self.assertGreater(int(inp_seg.sum()), 0, "No real input positions found")

  # -------------------------------------------------------- image token count

  def test_image_tokens_present_in_inputs(self):
    """Input token IDs must contain at least _TOKENS_PER_IMAGE occurrences of the image token."""
    from maxtext.multimodal.processor_qwen3_vl import QWEN3_VL_IMAGE_TOKEN  # pylint: disable=import-outside-toplevel

    inputs = np.array(self.batch["inputs"])
    image_token_count = int((inputs == QWEN3_VL_IMAGE_TOKEN).sum())
    # 1 image × 196 tokens per image (may be clipped by max_target_length, but 512 is generous)
    self.assertGreaterEqual(
        image_token_count,
        _TOKENS_PER_IMAGE,
        f"Expected ≥{_TOKENS_PER_IMAGE} image tokens, found {image_token_count}",
    )


if __name__ == "__main__":
  unittest.main()
