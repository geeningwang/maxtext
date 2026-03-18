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

""" Tests for the common MaxText utilities """
import os
import unittest
import numpy as np

from maxtext.configs import pyconfig
from maxtext.utils.globals import MAXTEXT_REPO_ROOT
from maxtext.multimodal import processor as mm_processor
from maxtext.multimodal import utils as mm_utils
from maxtext.multimodal import processor_gemma3
from maxtext.multimodal import processor_llama4


class TestTextImageFusionGemma3(unittest.TestCase):
  """Test inserting place_holder tokens for image"""

  def setUp(self):
    super().setUp()
    self.BEGIN_IMAGE_TOKEN = 255999
    self.mm_tokens = [self.BEGIN_IMAGE_TOKEN, -2, -2]

  def test_add_zero_image(self):
    tokens = np.asarray([1, 2, 3, 4, 5, 6])
    num_images = 0
    new_tokens = processor_gemma3.insert_sequence(
        at=self.BEGIN_IMAGE_TOKEN, sequence=self.mm_tokens, tokens=tokens, max_num_images=num_images
    )
    np.testing.assert_array_equal(new_tokens, tokens)

  def test_add_single_image(self):
    tokens = np.asarray([1, 2, 3, self.BEGIN_IMAGE_TOKEN, 5, 6])
    num_images = 1
    new_tokens = processor_gemma3.insert_sequence(
        at=self.BEGIN_IMAGE_TOKEN, sequence=self.mm_tokens, tokens=tokens, max_num_images=num_images
    )
    expected = np.asarray([1, 2, 3] + self.mm_tokens + [5, 6])
    np.testing.assert_array_equal(new_tokens, expected)

  def test_add_two_images(self):
    tokens = np.asarray([1, self.BEGIN_IMAGE_TOKEN, 3, 4, self.BEGIN_IMAGE_TOKEN, 6])
    num_images = 2
    new_tokens = processor_gemma3.insert_sequence(
        at=self.BEGIN_IMAGE_TOKEN, sequence=self.mm_tokens, tokens=tokens, max_num_images=num_images
    )
    expected = np.asarray([1] + self.mm_tokens + [3, 4] + self.mm_tokens + [6])
    np.testing.assert_array_equal(new_tokens, expected)

  def test_add_images_in_batch(self):
    tokens = np.asarray(
        [[1, 2, 3, self.BEGIN_IMAGE_TOKEN, 5, 6], [1, self.BEGIN_IMAGE_TOKEN, 3, 4, self.BEGIN_IMAGE_TOKEN, 6]]
    )
    num_images = 2
    new_tokens = processor_gemma3.insert_sequence(
        at=self.BEGIN_IMAGE_TOKEN, sequence=self.mm_tokens, tokens=tokens, max_num_images=num_images
    )
    expected = np.asarray(
        [
            [1, 2, 3] + self.mm_tokens + [5, 6] + [0] * (len(self.mm_tokens) - 1),
            [1] + self.mm_tokens + [3, 4] + self.mm_tokens + [6],
        ]
    )
    np.testing.assert_array_equal(new_tokens, expected)


class TestLlama4ImageProcessing(unittest.TestCase):
  """Test Llama4 image processing"""

  def setUp(self):
    super().setUp()
    self.LLAMA4_TILES_NUM = 16
    self.LLAMA4_TILE_SIZE = 336
    self.NUM_IMAGE_CHANNELS = 3

  def test_get_best_resolution(self):
    image_1 = np.ones((224, 300, self.NUM_IMAGE_CHANNELS))
    image_2 = np.ones((536, 640, self.NUM_IMAGE_CHANNELS))

    possible_resolutions = processor_llama4.find_supported_resolutions(
        max_num_tiles=self.LLAMA4_TILES_NUM, tile_size=self.LLAMA4_TILE_SIZE
    )
    best_resolution_1 = processor_llama4.get_best_resolution(
        img_height=image_1.shape[0],
        image_width=image_1.shape[1],
        possible_resolutions=possible_resolutions,
        resize_to_max_canvas=False,
    )
    best_resolution_2 = processor_llama4.get_best_resolution(
        img_height=image_2.shape[0],
        image_width=image_2.shape[1],
        possible_resolutions=possible_resolutions,
        resize_to_max_canvas=False,
    )
    self.assertEqual(best_resolution_1, (336, 336))
    self.assertEqual(best_resolution_2, (672, 672))

  def test_pad_to_best_fit_jax(self):
    image = np.zeros((536, 640, self.NUM_IMAGE_CHANNELS))
    best_resolution = (672, 672)
    padded_image = processor_llama4.pad_to_best_fit_jax(image, best_resolution)
    self.assertEqual(padded_image.shape, (672, 672, self.NUM_IMAGE_CHANNELS))
    self.assertTrue(np.all(padded_image == 0))

  def test_split_to_tiles(self):
    image = np.ones((672, 672, self.NUM_IMAGE_CHANNELS))
    best_resolution = (672, 672)
    ratio_h, ratio_w = (
        best_resolution[0] // self.LLAMA4_TILE_SIZE,
        best_resolution[1] // self.LLAMA4_TILE_SIZE,
    )
    image_tiles = processor_llama4.split_to_tiles(image, ratio_h, ratio_w)
    self.assertEqual(
        image_tiles.shape, (ratio_h * ratio_w, self.NUM_IMAGE_CHANNELS, self.LLAMA4_TILE_SIZE, self.LLAMA4_TILE_SIZE)
    )

  def test_pad_to_max_tiles(self):
    image = np.ones((5, self.NUM_IMAGE_CHANNELS, self.LLAMA4_TILE_SIZE, self.LLAMA4_TILE_SIZE))
    padded_image, image_mask = processor_llama4.pad_to_max_tiles(image, self.LLAMA4_TILES_NUM)
    self.assertEqual(
        padded_image.shape, (self.LLAMA4_TILES_NUM, self.NUM_IMAGE_CHANNELS, self.LLAMA4_TILE_SIZE, self.LLAMA4_TILE_SIZE)
    )
    self.assertEqual(image_mask.shape, (self.LLAMA4_TILES_NUM,))
    self.assertEqual(np.sum(image_mask), 5)
    self.assertEqual(np.sum(padded_image[5:]), 0)


class TestLlama4PostProcessing(unittest.TestCase):
  """Test Llama4 post-processing"""

  LLAMA4_FAKE_IMAGE_TOKEN = 200090  # <|image|>
  LLAMA4_BEGIN_IMAGE_TOKEN = 200080  # <|image_start|>
  LLAMA4_END_IMAGE_TOKEN = 200081  # <|image_end|>
  LLAMA4_PATCH_TOKEN = 200092  # <|patch|>
  LLAMA4_TILE_X_SEPARATOR_TOKEN = 200084  # <|tile_x_separator|>
  LLAMA4_TILE_Y_SEPARATOR_TOKEN = 200085  # <|tile_y_separator|>

  def setUp(self):
    super().setUp()
    self.NUM_IMAGE_CHANNELS = 3
    self.LLAMA4_TILE_SIZE = 336
    self.model_name = "llama4-17b-16e"

  def test_image_tokens_for_single_image(self):
    this_aspect_ratio = np.array([2, 2])
    num_patches_per_chunk = 4
    image_tokens = processor_llama4.get_tokens_for_this_image(this_aspect_ratio, num_patches_per_chunk)
    expected_tokens = [
        self.LLAMA4_BEGIN_IMAGE_TOKEN,
        self.LLAMA4_PATCH_TOKEN,
        self.LLAMA4_PATCH_TOKEN,
        self.LLAMA4_PATCH_TOKEN,
        self.LLAMA4_PATCH_TOKEN,
        self.LLAMA4_TILE_X_SEPARATOR_TOKEN,
        self.LLAMA4_PATCH_TOKEN,
        self.LLAMA4_PATCH_TOKEN,
        self.LLAMA4_PATCH_TOKEN,
        self.LLAMA4_PATCH_TOKEN,
        self.LLAMA4_TILE_Y_SEPARATOR_TOKEN,
        self.LLAMA4_PATCH_TOKEN,
        self.LLAMA4_PATCH_TOKEN,
        self.LLAMA4_PATCH_TOKEN,
        self.LLAMA4_PATCH_TOKEN,
        self.LLAMA4_TILE_X_SEPARATOR_TOKEN,
        self.LLAMA4_PATCH_TOKEN,
        self.LLAMA4_PATCH_TOKEN,
        self.LLAMA4_PATCH_TOKEN,
        self.LLAMA4_PATCH_TOKEN,
        self.LLAMA4_TILE_Y_SEPARATOR_TOKEN,
        self.LLAMA4_FAKE_IMAGE_TOKEN,
        self.LLAMA4_PATCH_TOKEN,
        self.LLAMA4_PATCH_TOKEN,
        self.LLAMA4_PATCH_TOKEN,
        self.LLAMA4_PATCH_TOKEN,
        self.LLAMA4_END_IMAGE_TOKEN,
    ]
    self.assertEqual(image_tokens, expected_tokens)

  def test_post_process_image_tokens(self):
    dummy_pixel_values = np.ones(
        (5, mm_utils.NUM_IMAGE_CHANNELS, processor_llama4.LLAMA4_TILE_SIZE, processor_llama4.LLAMA4_TILE_SIZE)
    )
    dummy_aspect_ratios = np.array([[2, 2]])
    dummy_tokens = np.array([1, 2, self.LLAMA4_FAKE_IMAGE_TOKEN, 4, 5])
    processor_output = processor_llama4.Llama4PreprocessorOutput(
        pixel_values=dummy_pixel_values,
        aspect_ratios=dummy_aspect_ratios,
    )
    base_config_path = os.path.join(MAXTEXT_REPO_ROOT, "src", "maxtext", "configs", "base.yml")
    config = pyconfig.initialize(
        ["", base_config_path],
        model_name="llama4-17b-16e",
    )
    image_offsets = mm_processor.get_image_offsets(config=config, processor_output=processor_output)
    post_processed_tokens = processor_llama4.add_extra_tokens_for_images_llama4(dummy_tokens, processor_output)
    self.assertEqual(post_processed_tokens.shape[0], dummy_tokens.shape[0] + image_offsets)

  def test_merge_mm_embeddings(self):
    # Setup Dummy Data
    batch_size = 1
    seq_len = 64
    d = 4
    num_images = 2
    num_tiles = 4
    num_toks_per_image = 8

    # text_embeddings: (B, S, D) -> (1, 64, 4)
    text_embeddings = np.arange(batch_size * seq_len * d, dtype=np.float32).reshape(batch_size, seq_len, d)

    # vision_embeddings: (B * N, T, K, D) -> (2, 4, 8, 4)
    vision_embeddings = (
        np.arange(batch_size * num_images * num_tiles * num_toks_per_image * d, dtype=np.float32).reshape(
            batch_size * num_images, num_tiles, num_toks_per_image, d
        )
        + 1000
    )

    # mask: (B, S) -> (1, 64)
    # Total of 8 + 16 = 24 token slots available for images.
    mask = np.zeros((batch_size, seq_len), dtype=np.int32)
    mask[:, 2:10] = 1  # 8 slots for the first image's valid tiles
    mask[:, 20:36] = 1  # 16 slots for the second image's valid tiles

    # image_masks: (B * N, T) -> (2, 4)
    # Specifies which tiles are valid.
    image_masks = np.zeros((batch_size * num_images, num_tiles), dtype=np.int32)
    # Image 0 has 1 valid tile -> 1 * 8 = 8 valid tokens.
    image_masks[0, 0] = 1
    # Image 1 has 2 valid tiles -> 2 * 8 = 16 valid tokens.
    image_masks[1, 0] = 1
    image_masks[1, 1] = 1
    # Total valid tokens = 8 + 16 = 24. This matches the mask slots.

    # Case 1: Use the image_mask to filter for valid tiles.
    merged = mm_utils.merge_mm_embeddings(text_embeddings, vision_embeddings, mask, image_masks)

    # Case 2: No image_mask, so all vision tokens are used in order.
    merged_null = mm_utils.merge_mm_embeddings(text_embeddings, vision_embeddings, mask, None)

    # The results should be different since one is masked and one is not.
    self.assertFalse(np.array_equal(merged, merged_null))

    # The code gathers all valid tiles first and then inserts them sequentially.
    # Valid tiles are: vision_embeddings[0, 0], vision_embeddings[1, 0], vision_embeddings[1, 1]

    # The first 8 slots (2:10) should be filled by the first valid tile's tokens.
    first_valid_tile = vision_embeddings[0, 0, :, :]
    np.testing.assert_array_equal(merged[0, 2:10], first_valid_tile)

    # The next 8 slots (20:28) get the second valid tile's tokens.
    second_valid_tile = vision_embeddings[1, 0, :, :]
    np.testing.assert_array_equal(merged[0, 20:28], second_valid_tile)

    # The final 8 slots (28:36) get the third valid tile's tokens.
    third_valid_tile = vision_embeddings[1, 1, :, :]
    np.testing.assert_array_equal(merged[0, 28:36], third_valid_tile)

    # When no mask is provided all vision tiles are inserted sequentially in their natural flattened order.
    np.testing.assert_array_equal(merged_null[0, 2:10], vision_embeddings[0, 0, :, :])
    np.testing.assert_array_equal(merged_null[0, 20:28], vision_embeddings[0, 1, :, :])
    np.testing.assert_array_equal(merged_null[0, 28:36], vision_embeddings[0, 2, :, :])

    # Verify that parts of the text sequence that were NOT masked remain untouched.
    np.testing.assert_array_equal(merged[0, 10:20], text_embeddings[0, 10:20])
    np.testing.assert_array_equal(merged[0, 36:], text_embeddings[0, 36:])

    # The first position should always be preserved.
    np.testing.assert_array_equal(merged[0, 0], text_embeddings[0, 0])
    np.testing.assert_array_equal(merged_null[0, 0], text_embeddings[0, 0])


class TestQwen3VLPreprocessor(unittest.TestCase):
  """Unit tests for Qwen3-VL image preprocessing and SFT data-preparation functions.

  These tests exercise:
    - ``preprocess_mm_data_qwen3_vl`` (pixel shape, normalisation, grid_thw)
    - ``reformat_prompt_qwen3_vl``   (chat template, placeholder substitution)
    - ``add_extra_tokens_for_images_qwen3_vl`` (token expansion: 1 → 196)
    - ``get_image_offsets_qwen3_vl`` (offset arithmetic)
    - ``processor.py`` routing functions for model_name "qwen3-vl-2b"
  """

  # ------------------------------------------------------------------ helpers

  def _fake_image(self, h=100, w=150):
    """Return a random uint8 HWC numpy array suitable as input."""
    return np.random.randint(0, 255, (h, w, 3), dtype=np.uint8)

  def _make_processor_output(self, n=1):
    """Return a Qwen3VLPreprocessorOutput for *n* 448×448 images."""
    from maxtext.multimodal.processor_qwen3_vl import Qwen3VLPreprocessorOutput  # pylint: disable=import-outside-toplevel

    return Qwen3VLPreprocessorOutput(
        pixel_values=np.zeros((n, 3, 2, 448, 448), dtype=np.float32),
        num_images=n,
        pixel_grid_thw=np.tile(np.array([1, 28, 28], dtype=np.int32), (n, 1)),
    )

  # -------------------------------------------------------- preprocess shape

  def test_single_image_pixel_shape(self):
    """A single image should produce pixel_values of shape (1, 3, 2, 448, 448)."""
    from maxtext.multimodal.processor_qwen3_vl import preprocess_mm_data_qwen3_vl  # pylint: disable=import-outside-toplevel

    out = preprocess_mm_data_qwen3_vl(self._fake_image())
    self.assertEqual(out.pixel_values.shape, (1, 3, 2, 448, 448))
    self.assertEqual(out.num_images, 1)

  def test_single_image_grid_thw(self):
    """grid_thw for a 448×448 image with patch_size=16, merge_size=2 must be [1, 28, 28]."""
    from maxtext.multimodal.processor_qwen3_vl import preprocess_mm_data_qwen3_vl  # pylint: disable=import-outside-toplevel

    out = preprocess_mm_data_qwen3_vl(self._fake_image())
    np.testing.assert_array_equal(out.pixel_grid_thw[0], [1, 28, 28])

  def test_multi_image_shape(self):
    """Two images should produce shape (2, 3, 2, 448, 448) and grid_thw of shape (2, 3)."""
    from maxtext.multimodal.processor_qwen3_vl import preprocess_mm_data_qwen3_vl  # pylint: disable=import-outside-toplevel

    out = preprocess_mm_data_qwen3_vl([self._fake_image(), self._fake_image(200, 300)])
    self.assertEqual(out.pixel_values.shape, (2, 3, 2, 448, 448))
    self.assertEqual(out.pixel_grid_thw.shape, (2, 3))
    self.assertEqual(out.num_images, 2)

  def test_normalization_black_image(self):
    """An all-black image (0) should normalise to approximately -1.0."""
    from maxtext.multimodal.processor_qwen3_vl import preprocess_mm_data_qwen3_vl  # pylint: disable=import-outside-toplevel

    black = np.zeros((64, 64, 3), dtype=np.uint8)
    out = preprocess_mm_data_qwen3_vl(black)
    self.assertAlmostEqual(float(out.pixel_values.min()), -1.0, places=3)

  def test_normalization_white_image(self):
    """An all-white image (255) should normalise to approximately +1.0."""
    from maxtext.multimodal.processor_qwen3_vl import preprocess_mm_data_qwen3_vl  # pylint: disable=import-outside-toplevel

    white = np.full((64, 64, 3), 255, dtype=np.uint8)
    out = preprocess_mm_data_qwen3_vl(white)
    self.assertAlmostEqual(float(out.pixel_values.max()), 1.0, places=3)

  # -------------------------------------------------------- reformat_prompt

  def test_reformat_prompt_chat_template(self):
    """Output must be wrapped in the Qwen im_start/im_end chat template."""
    from maxtext.multimodal.processor_qwen3_vl import reformat_prompt_qwen3_vl  # pylint: disable=import-outside-toplevel

    result = reformat_prompt_qwen3_vl("What is this?", "<|image|>", 0)
    self.assertIn("<|im_start|>user\n", result)
    self.assertIn("<|im_end|>", result)
    self.assertIn("<|im_start|>assistant\n", result)

  def test_reformat_prompt_replaces_placeholder(self):
    """Image placeholder must be replaced with the Qwen3-VL vision token sequence."""
    from maxtext.multimodal.processor_qwen3_vl import reformat_prompt_qwen3_vl  # pylint: disable=import-outside-toplevel

    result = reformat_prompt_qwen3_vl("<|image|> describe it", "<|image|>", 1)
    self.assertIn("<|vision_start|><|image_pad|><|vision_end|>", result)
    self.assertNotIn("<|image|>", result)

  def test_reformat_prompt_prepends_missing_image(self):
    """If num_images > occurrences in prompt, missing vision tokens are prepended."""
    from maxtext.multimodal.processor_qwen3_vl import reformat_prompt_qwen3_vl  # pylint: disable=import-outside-toplevel

    # Prompt has no placeholder but num_images=1
    result = reformat_prompt_qwen3_vl("describe it", "<|image|>", 1)
    self.assertIn("<|vision_start|><|image_pad|><|vision_end|>", result)

  # ------------------------------------------------ add_extra_tokens

  def test_add_extra_tokens_single_image(self):
    """One <|image_pad|> token must expand to 196 copies (1×28×28÷4)."""
    from maxtext.multimodal.processor_qwen3_vl import (  # pylint: disable=import-outside-toplevel
        add_extra_tokens_for_images_qwen3_vl,
        QWEN3_VL_IMAGE_TOKEN,
    )

    tokens = np.array([1, QWEN3_VL_IMAGE_TOKEN, 2], dtype=np.int32)
    out = add_extra_tokens_for_images_qwen3_vl(tokens, self._make_processor_output(1))
    # [1] + [IMAGE_TOKEN]*196 + [2]
    self.assertEqual(len(out), 198)
    self.assertTrue(np.all(out[1:197] == QWEN3_VL_IMAGE_TOKEN))
    self.assertEqual(out[0], 1)
    self.assertEqual(out[197], 2)

  def test_add_extra_tokens_no_image_tokens(self):
    """A token sequence without any image placeholders must be returned unchanged."""
    from maxtext.multimodal.processor_qwen3_vl import add_extra_tokens_for_images_qwen3_vl  # pylint: disable=import-outside-toplevel

    tokens = np.array([10, 20, 30, 40], dtype=np.int32)
    out = add_extra_tokens_for_images_qwen3_vl(tokens, self._make_processor_output(1))
    np.testing.assert_array_equal(out, tokens)

  def test_add_extra_tokens_two_images(self):
    """Two placeholders each expand to 196; total length = 2 + 2*196."""
    from maxtext.multimodal.processor_qwen3_vl import (  # pylint: disable=import-outside-toplevel
        add_extra_tokens_for_images_qwen3_vl,
        QWEN3_VL_IMAGE_TOKEN,
    )

    tokens = np.array([QWEN3_VL_IMAGE_TOKEN, 5, QWEN3_VL_IMAGE_TOKEN], dtype=np.int32)
    out = add_extra_tokens_for_images_qwen3_vl(tokens, self._make_processor_output(2))
    self.assertEqual(len(out), 2 * 196 + 1)

  # ------------------------------------------------ get_image_offsets

  def test_get_offsets_single_image(self):
    """Offset for one 448×448 image = 195 (196 expanded tokens − 1 placeholder)."""
    from maxtext.multimodal.processor_qwen3_vl import get_image_offsets_qwen3_vl  # pylint: disable=import-outside-toplevel

    self.assertEqual(get_image_offsets_qwen3_vl(self._make_processor_output(1)), 195)

  def test_get_offsets_two_images(self):
    """Two images → offset = 2 × 195 = 390."""
    from maxtext.multimodal.processor_qwen3_vl import get_image_offsets_qwen3_vl  # pylint: disable=import-outside-toplevel

    self.assertEqual(get_image_offsets_qwen3_vl(self._make_processor_output(2)), 390)

  def test_get_offsets_none_input(self):
    """None processor_output must return 0."""
    from maxtext.multimodal.processor_qwen3_vl import get_image_offsets_qwen3_vl  # pylint: disable=import-outside-toplevel

    self.assertEqual(get_image_offsets_qwen3_vl(None), 0)

  # ----------------------------------------------- processor.py routing

  def test_router_preprocess_image(self):
    """mm_processor.preprocess_image_for_training must route qwen3-vl-2b to Qwen3VLPreprocessorOutput."""
    from maxtext.multimodal.processor_qwen3_vl import Qwen3VLPreprocessorOutput  # pylint: disable=import-outside-toplevel

    out = mm_processor.preprocess_image_for_training(self._fake_image(), "qwen3-vl-2b")
    self.assertIsInstance(out, Qwen3VLPreprocessorOutput)
    self.assertEqual(out.pixel_values.shape, (1, 3, 2, 448, 448))

  def test_router_preprocess_image_8b(self):
    """qwen3-vl-8b must also route to Qwen3VLPreprocessorOutput."""
    from maxtext.multimodal.processor_qwen3_vl import Qwen3VLPreprocessorOutput  # pylint: disable=import-outside-toplevel

    out = mm_processor.preprocess_image_for_training(self._fake_image(), "qwen3-vl-8b")
    self.assertIsInstance(out, Qwen3VLPreprocessorOutput)

  def test_router_reformat_response(self):
    """mm_processor.reformat_response must append <|im_end|> for qwen3-vl-2b."""
    result = mm_processor.reformat_response("A cat.", "qwen3-vl-2b")
    self.assertEqual(result, "A cat.<|im_end|>")

  def test_router_reformat_response_8b(self):
    """qwen3-vl-8b must produce the same <|im_end|> suffix."""
    result = mm_processor.reformat_response("A cat.", "qwen3-vl-8b")
    self.assertEqual(result, "A cat.<|im_end|>")

  def test_router_reformat_prompt(self):
    """mm_processor.reformat_prompt wraps qwen3-vl-2b prompt in Qwen chat template."""
    result = mm_processor.reformat_prompt("What is this?", "<|image|>", "qwen3-vl-2b", num_images=0)
    self.assertIn("<|im_start|>user\n", result)
    self.assertIn("<|im_start|>assistant\n", result)

  def test_router_get_image_offsets_string_config(self):
    """get_image_offsets must accept a bare model-name string (PadOrTrimToMaxLength usage)."""
    offset = mm_processor.get_image_offsets("qwen3-vl-2b", self._make_processor_output(1))
    self.assertEqual(offset, 195)


if __name__ == "__main__":
  unittest.main()
