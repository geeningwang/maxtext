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

"""Unit tests for the upgraded Qwen3-VL image/video preprocessor.

Tests cover:
  - ``smart_resize_image``   — dynamic resolution helper for images
  - ``smart_resize_video``   — dynamic resolution helper for videos
  - ``preprocess_mm_data_qwen3_vl``   — image preprocessing (dynamic + force_size)
  - ``preprocess_video_qwen3_vl``     — video preprocessing
  - ``add_extra_tokens_for_images_qwen3_vl`` — token expansion
  - ``add_extra_tokens_for_video_qwen3_vl``  — video token expansion
  - ``reformat_prompt_qwen3_vl``              — chat-template formatting w/ video
  - ``get_image_offsets_qwen3_vl``            — token offset computation
  - ``get_video_offsets_qwen3_vl``            — video token offset computation

Run:
  pytest tests/unit/qwen3_vl_preprocessor_test.py -v
"""

import math
import unittest

import numpy as np

from maxtext.multimodal.processor_qwen3_vl import (
    QWEN3_VL_IMAGE_MAX_PIXELS,
    QWEN3_VL_IMAGE_MIN_PIXELS,
    QWEN3_VL_IMAGE_SIZE,
    QWEN3_VL_IMAGE_MEAN,
    QWEN3_VL_IMAGE_STD,
    QWEN3_VL_IMAGE_TOKEN,
    QWEN3_VL_PATCH_SIZE,
    QWEN3_VL_RESIZE_FACTOR,
    QWEN3_VL_SPATIAL_MERGE_SIZE,
    QWEN3_VL_TEMPORAL_PATCH_SIZE,
    QWEN3_VL_VIDEO_MAX_PIXELS,
    QWEN3_VL_VIDEO_MIN_PIXELS,
    QWEN3_VL_VIDEO_TOKEN,
    QWEN3_VL_VIDEO_PAD_STR,
    QWEN3_VL_IMAGE_PAD_STR,
    Qwen3VLPreprocessorOutput,
    add_extra_tokens_for_images_qwen3_vl,
    add_extra_tokens_for_video_qwen3_vl,
    get_image_offsets_qwen3_vl,
    get_video_offsets_qwen3_vl,
    preprocess_mm_data_qwen3_vl,
    preprocess_video_qwen3_vl,
    reformat_prompt_qwen3_vl,
    smart_resize_image,
    smart_resize_video,
)


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _rand_image(h: int, w: int, seed: int = 0) -> np.ndarray:
  """Create a random (H, W, 3) uint8 image."""
  rng = np.random.default_rng(seed)
  return rng.integers(0, 256, (h, w, 3), dtype=np.uint8)


def _rand_frames(T: int, h: int, w: int, seed: int = 0) -> np.ndarray:
  """Create random (T, H, W, 3) uint8 video frames."""
  rng = np.random.default_rng(seed)
  return rng.integers(0, 256, (T, h, w, 3), dtype=np.uint8)


# ─────────────────────────────────────────────────────────────────────────────
# smart_resize_image
# ─────────────────────────────────────────────────────────────────────────────

class TestSmartResizeImage(unittest.TestCase):

  def _check_common(self, h_bar, w_bar, factor=QWEN3_VL_RESIZE_FACTOR):
    self.assertEqual(h_bar % factor, 0, f"h_bar={h_bar} not divisible by factor={factor}")
    self.assertEqual(w_bar % factor, 0, f"w_bar={w_bar} not divisible by factor={factor}")
    self.assertGreater(h_bar, 0)
    self.assertGreater(w_bar, 0)

  def test_square_image_within_bounds(self):
    """A 448×448 image should round to exactly 448×448 (already factor-aligned)."""
    h, w = smart_resize_image(448, 448)
    self.assertEqual((h, w), (448, 448))
    pix = h * w
    self.assertGreaterEqual(pix, QWEN3_VL_IMAGE_MIN_PIXELS)
    self.assertLessEqual(pix, QWEN3_VL_IMAGE_MAX_PIXELS)

  def test_small_image_scaled_up(self):
    """A very small image (below min_pixels) must be scaled up."""
    h, w = smart_resize_image(32, 32)
    pix = h * w
    self.assertGreaterEqual(pix, QWEN3_VL_IMAGE_MIN_PIXELS)
    self._check_common(h, w)

  def test_large_image_scaled_down(self):
    """A very large image (above max_pixels) must be scaled down."""
    h, w = smart_resize_image(4096, 4096)
    pix = h * w
    self.assertLessEqual(pix, QWEN3_VL_IMAGE_MAX_PIXELS)
    self._check_common(h, w)

  def test_aspect_ratio_portrait(self):
    """Portrait image: both dims divisible by factor, aspect ratio preserved."""
    h, w = smart_resize_image(768, 512)
    self._check_common(h, w)
    # Aspect ratio approx preserved (within factor granularity).
    original_ratio = 768 / 512
    resized_ratio = h / w
    self.assertAlmostEqual(original_ratio, resized_ratio, delta=0.25)

  def test_aspect_ratio_landscape(self):
    """Landscape image: both dims divisible by factor."""
    h, w = smart_resize_image(360, 640)
    self._check_common(h, w)

  def test_small_but_valid_image(self):
    """Image smaller than 448 but within min_pixels range: keep as-is or scale."""
    h, w = smart_resize_image(100, 150)
    self._check_common(h, w)
    pix = h * w
    self.assertGreaterEqual(pix, QWEN3_VL_IMAGE_MIN_PIXELS)
    self.assertLessEqual(pix, QWEN3_VL_IMAGE_MAX_PIXELS)

  def test_custom_min_max(self):
    """Custom min_pixels=max_pixels clamps total pixels to the specified range."""
    target = 448 * 448
    h, w = smart_resize_image(100, 200, min_pixels=target, max_pixels=target * 2)
    pix = h * w
    self.assertGreaterEqual(pix, target)
    self.assertLessEqual(pix, target * 2)
    self._check_common(h, w)

  def test_extreme_aspect_ratio_raises(self):
    """Aspect ratio > 200 must raise ValueError."""
    with self.assertRaises(ValueError):
      smart_resize_image(1, 201)

  def test_zero_dimension_raises(self):
    """Zero dimensions must raise ValueError."""
    with self.assertRaises(ValueError):
      smart_resize_image(0, 448)

  def test_output_always_factor_aligned(self):
    """Spot-check many sizes: output dims always divisible by QWEN3_VL_RESIZE_FACTOR."""
    test_cases = [
        (100, 100), (200, 300), (640, 480), (1280, 720), (1920, 1080),
        (56, 56), (1024, 1024),
    ]
    for h, w in test_cases:
      with self.subTest(h=h, w=w):
        h_bar, w_bar = smart_resize_image(h, w)
        self._check_common(h_bar, w_bar)
        pix = h_bar * w_bar
        self.assertGreaterEqual(pix, QWEN3_VL_IMAGE_MIN_PIXELS)
        self.assertLessEqual(pix, QWEN3_VL_IMAGE_MAX_PIXELS)


# ─────────────────────────────────────────────────────────────────────────────
# smart_resize_video
# ─────────────────────────────────────────────────────────────────────────────

class TestSmartResizeVideo(unittest.TestCase):

  def _check_video(self, h_bar, w_bar, factor=QWEN3_VL_RESIZE_FACTOR):
    self.assertEqual(h_bar % factor, 0, f"h_bar={h_bar} not divisible by {factor}")
    self.assertEqual(w_bar % factor, 0, f"w_bar={w_bar} not divisible by {factor}")

  def test_standard_video(self):
    """Standard 8-frame 480×640 video should produce valid resize."""
    h, w = smart_resize_video(8, 480, 640)
    self._check_video(h, w)
    self.assertGreater(h, 0)
    self.assertGreater(w, 0)

  def test_total_pixels_within_bounds(self):
    """T×H×W after resize must be within [min_pixels, max_pixels]."""
    T, H, W = 16, 720, 1280
    h, w = smart_resize_video(T, H, W)
    t_bar = max(QWEN3_VL_TEMPORAL_PATCH_SIZE,
                round(T / QWEN3_VL_TEMPORAL_PATCH_SIZE) * QWEN3_VL_TEMPORAL_PATCH_SIZE)
    total = t_bar * h * w
    self.assertGreaterEqual(total, QWEN3_VL_VIDEO_MIN_PIXELS)
    self.assertLessEqual(total, QWEN3_VL_VIDEO_MAX_PIXELS)

  def test_too_few_frames_raises(self):
    """num_frames < temporal_factor must raise ValueError."""
    with self.assertRaises(ValueError):
      smart_resize_video(1, 480, 640)

  def test_tiny_frame_raises(self):
    """Frame dimension < factor must raise ValueError."""
    with self.assertRaises(ValueError):
      smart_resize_video(4, 10, 640)  # height < factor=32

  def test_extreme_aspect_ratio_raises(self):
    """Frame aspect ratio > 200 must raise ValueError."""
    with self.assertRaises(ValueError):
      smart_resize_video(4, 1, 201)

  def test_output_factor_aligned(self):
    """Various frame sizes: output dims always divisible by QWEN3_VL_RESIZE_FACTOR."""
    cases = [
        (4, 64, 64), (8, 320, 240), (16, 480, 854), (32, 1080, 1920),
    ]
    for T, H, W in cases:
      with self.subTest(T=T, H=H, W=W):
        h, w = smart_resize_video(T, H, W)
        self._check_video(h, w)


# ─────────────────────────────────────────────────────────────────────────────
# preprocess_mm_data_qwen3_vl
# ─────────────────────────────────────────────────────────────────────────────

class TestPreprocessMmDataQwen3VL(unittest.TestCase):

  # ── Output types / fields ──────────────────────────────────────────────────

  def test_returns_qwen3vl_output(self):
    img = _rand_image(224, 224)
    out = preprocess_mm_data_qwen3_vl(img)
    self.assertIsInstance(out, Qwen3VLPreprocessorOutput)

  def test_pixel_values_present(self):
    img = _rand_image(224, 224)
    out = preprocess_mm_data_qwen3_vl(img)
    self.assertIsNotNone(out.pixel_values)

  def test_pixel_grid_thw_present(self):
    img = _rand_image(448, 448)
    out = preprocess_mm_data_qwen3_vl(img)
    self.assertIsNotNone(out.pixel_grid_thw)

  # ── Shape — single image, dynamic resolution ───────────────────────────────

  def test_output_shape_ndim(self):
    """pixel_values must be 5-D: (N, C, T, H, W)."""
    img = _rand_image(224, 224)
    out = preprocess_mm_data_qwen3_vl(img)
    self.assertEqual(out.pixel_values.ndim, 5)

  def test_output_channels(self):
    img = _rand_image(224, 224)
    out = preprocess_mm_data_qwen3_vl(img)
    self.assertEqual(out.pixel_values.shape[1], 3)  # C=3

  def test_output_temporal_dim(self):
    """T must equal QWEN3_VL_TEMPORAL_PATCH_SIZE (2)."""
    img = _rand_image(224, 224)
    out = preprocess_mm_data_qwen3_vl(img)
    self.assertEqual(out.pixel_values.shape[2], QWEN3_VL_TEMPORAL_PATCH_SIZE)

  def test_spatial_dims_factor_aligned(self):
    """H_bar and W_bar must be divisible by QWEN3_VL_RESIZE_FACTOR (32)."""
    for sz in [(100, 150), (224, 224), (640, 480)]:
      with self.subTest(sz=sz):
        img = _rand_image(*sz)
        out = preprocess_mm_data_qwen3_vl(img)
        _, _, _, H_bar, W_bar = out.pixel_values.shape
        self.assertEqual(H_bar % QWEN3_VL_RESIZE_FACTOR, 0)
        self.assertEqual(W_bar % QWEN3_VL_RESIZE_FACTOR, 0)

  def test_pixel_count_within_bounds(self):
    """H_bar × W_bar must lie within [min_pixels, max_pixels]."""
    img = _rand_image(300, 400)
    out = preprocess_mm_data_qwen3_vl(img)
    _, _, _, H_bar, W_bar = out.pixel_values.shape
    pix = H_bar * W_bar
    self.assertGreaterEqual(pix, QWEN3_VL_IMAGE_MIN_PIXELS)
    self.assertLessEqual(pix, QWEN3_VL_IMAGE_MAX_PIXELS)

  # ── force_size — fixed 448×448 (training compatibility) ───────────────────

  def test_force_size_square(self):
    """force_size=(448, 448) must produce exactly 448×448 output."""
    img = _rand_image(100, 150)
    out = preprocess_mm_data_qwen3_vl(img, force_size=(QWEN3_VL_IMAGE_SIZE, QWEN3_VL_IMAGE_SIZE))
    _, _, _, H, W = out.pixel_values.shape
    self.assertEqual(H, QWEN3_VL_IMAGE_SIZE)
    self.assertEqual(W, QWEN3_VL_IMAGE_SIZE)

  def test_force_size_produces_standard_grid(self):
    """force_size=(448, 448) → grid_thw should be [1, 28, 28]."""
    img = _rand_image(100, 150)
    out = preprocess_mm_data_qwen3_vl(img, force_size=(QWEN3_VL_IMAGE_SIZE, QWEN3_VL_IMAGE_SIZE))
    expected_grid_h = QWEN3_VL_IMAGE_SIZE // QWEN3_VL_PATCH_SIZE  # 28
    np.testing.assert_array_equal(
        out.pixel_grid_thw[0], [1, expected_grid_h, expected_grid_h]
    )

  def test_force_size_token_count_196(self):
    """force_size=(448, 448) → 196 tokens per image (28×28 // 4)."""
    img = _rand_image(100, 150)
    out = preprocess_mm_data_qwen3_vl(img, force_size=(QWEN3_VL_IMAGE_SIZE, QWEN3_VL_IMAGE_SIZE))
    merge_sq = QWEN3_VL_SPATIAL_MERGE_SIZE ** 2
    grid = out.pixel_grid_thw[0]
    tokens = int(grid[0] * grid[1] * grid[2]) // merge_sq
    self.assertEqual(tokens, 196)

  # ── Normalisation ──────────────────────────────────────────────────────────

  def test_pixel_range(self):
    """Normalised pixel values must lie in [−1.1, 1.1]."""
    img = _rand_image(224, 224)
    out = preprocess_mm_data_qwen3_vl(img)
    pv = out.pixel_values
    self.assertGreater(float(pv.min()), -1.2)
    self.assertLess(float(pv.max()), 1.2)

  def test_zero_image_maps_to_neg_one(self):
    """All-zero image normalised: (0 − 127.5) / 127.5 = −1."""
    img = np.zeros((64, 64, 3), dtype=np.uint8)
    out = preprocess_mm_data_qwen3_vl(img)
    self.assertAlmostEqual(float(out.pixel_values.min()), -1.0, places=4)

  def test_255_image_maps_to_pos_one(self):
    """All-255 image normalised: (255 − 127.5) / 127.5 = +1."""
    img = np.full((64, 64, 3), 255, dtype=np.uint8)
    out = preprocess_mm_data_qwen3_vl(img)
    self.assertAlmostEqual(float(out.pixel_values.max()), 1.0, places=4)

  def test_finite_pixel_values(self):
    img = _rand_image(224, 224)
    out = preprocess_mm_data_qwen3_vl(img)
    self.assertTrue(np.all(np.isfinite(out.pixel_values)))

  # ── Batch (multiple identical-resolution images) ───────────────────────────

  def test_multi_image_batch(self):
    """Multiple same-size images must be stacked along the N axis."""
    imgs = [_rand_image(224, 224, seed=i) for i in range(3)]
    out = preprocess_mm_data_qwen3_vl(imgs)
    self.assertEqual(out.num_images, 3)
    self.assertEqual(out.pixel_values.shape[0], 3)
    self.assertEqual(out.pixel_grid_thw.shape, (3, 3))

  def test_different_size_images_raise(self):
    """Images with different smart_resize outputs cannot be stacked."""
    # 100×100 → small square; 400×800 → landscape — likely different grid.
    imgs = [_rand_image(100, 100), _rand_image(400, 800)]
    h1, w1 = smart_resize_image(100, 100)
    h2, w2 = smart_resize_image(400, 800)
    if (h1, w1) != (h2, w2):
      with self.assertRaises(ValueError):
        preprocess_mm_data_qwen3_vl(imgs)

  # ── pixel_grid_thw correctness ─────────────────────────────────────────────

  def test_pixel_grid_thw_shape(self):
    """pixel_grid_thw must be (N, 3)."""
    img = _rand_image(224, 224)
    out = preprocess_mm_data_qwen3_vl(img)
    self.assertEqual(out.pixel_grid_thw.shape, (1, 3))

  def test_pixel_grid_thw_consistent_with_pixel_values(self):
    """grid_h * patch_size == H_bar, grid_w * patch_size == W_bar."""
    img = _rand_image(640, 480)
    out = preprocess_mm_data_qwen3_vl(img)
    _, _, _, H_bar, W_bar = out.pixel_values.shape
    grid = out.pixel_grid_thw[0]
    self.assertEqual(grid[1] * QWEN3_VL_PATCH_SIZE, H_bar)
    self.assertEqual(grid[2] * QWEN3_VL_PATCH_SIZE, W_bar)

  def test_grid_t_equals_one_for_images(self):
    """For images, grid_t always equals 1."""
    for sz in [(224, 224), (448, 448), (100, 150)]:
      img = _rand_image(*sz)
      out = preprocess_mm_data_qwen3_vl(img)
      self.assertEqual(out.pixel_grid_thw[0, 0], 1)

  # ── dtype ──────────────────────────────────────────────────────────────────

  def test_output_dtype_float32(self):
    img = _rand_image(224, 224)
    out = preprocess_mm_data_qwen3_vl(img)
    self.assertEqual(out.pixel_values.dtype, np.float32)

  def test_grid_dtype_int32(self):
    img = _rand_image(224, 224)
    out = preprocess_mm_data_qwen3_vl(img)
    self.assertEqual(out.pixel_grid_thw.dtype, np.int32)

  # ── Input type robustness ──────────────────────────────────────────────────

  def test_accepts_float_array(self):
    """Float arrays (HWC, values 0–255) must be handled without error."""
    img_float = _rand_image(128, 128).astype(np.float32)
    out = preprocess_mm_data_qwen3_vl(img_float)
    self.assertIsNotNone(out.pixel_values)

  def test_accepts_list_input(self):
    imgs = [_rand_image(224, 224)]
    out = preprocess_mm_data_qwen3_vl(imgs)
    self.assertEqual(out.num_images, 1)


# ─────────────────────────────────────────────────────────────────────────────
# preprocess_video_qwen3_vl
# ─────────────────────────────────────────────────────────────────────────────

class TestPreprocessVideoQwen3VL(unittest.TestCase):

  # ── Basic output structure ─────────────────────────────────────────────────

  def test_returns_qwen3vl_output(self):
    frames = _rand_frames(8, 240, 320)
    out = preprocess_video_qwen3_vl(frames)
    self.assertIsInstance(out, Qwen3VLPreprocessorOutput)

  def test_num_videos(self):
    frames = _rand_frames(8, 240, 320)
    out = preprocess_video_qwen3_vl(frames)
    self.assertEqual(out.num_videos, 1)

  def test_pixel_values_videos_present(self):
    frames = _rand_frames(8, 240, 320)
    out = preprocess_video_qwen3_vl(frames)
    self.assertIsNotNone(out.pixel_values_videos)

  def test_video_grid_thw_present(self):
    frames = _rand_frames(8, 240, 320)
    out = preprocess_video_qwen3_vl(frames)
    self.assertIsNotNone(out.video_grid_thw)

  # ── Output shape ───────────────────────────────────────────────────────────

  def test_pixel_values_videos_ndim(self):
    """pixel_values_videos must be 5-D: (1, C, T, H, W)."""
    frames = _rand_frames(8, 320, 240)
    out = preprocess_video_qwen3_vl(frames)
    self.assertEqual(out.pixel_values_videos.ndim, 5)

  def test_batch_dim_is_one(self):
    frames = _rand_frames(8, 320, 240)
    out = preprocess_video_qwen3_vl(frames)
    self.assertEqual(out.pixel_values_videos.shape[0], 1)

  def test_channel_dim_is_three(self):
    frames = _rand_frames(8, 320, 240)
    out = preprocess_video_qwen3_vl(frames)
    self.assertEqual(out.pixel_values_videos.shape[1], 3)

  def test_temporal_dim_divisible(self):
    """T_padded must be divisible by QWEN3_VL_TEMPORAL_PATCH_SIZE."""
    frames = _rand_frames(7, 320, 240)  # odd frame count
    out = preprocess_video_qwen3_vl(frames)
    T_padded = out.pixel_values_videos.shape[2]
    self.assertEqual(T_padded % QWEN3_VL_TEMPORAL_PATCH_SIZE, 0)

  def test_spatial_dims_factor_aligned(self):
    """H_bar and W_bar must be divisible by QWEN3_VL_RESIZE_FACTOR."""
    frames = _rand_frames(8, 480, 640)
    out = preprocess_video_qwen3_vl(frames)
    _, _, _, H_bar, W_bar = out.pixel_values_videos.shape
    self.assertEqual(H_bar % QWEN3_VL_RESIZE_FACTOR, 0)
    self.assertEqual(W_bar % QWEN3_VL_RESIZE_FACTOR, 0)

  def test_video_grid_thw_shape(self):
    """video_grid_thw must be (1, 3)."""
    frames = _rand_frames(8, 320, 240)
    out = preprocess_video_qwen3_vl(frames)
    self.assertEqual(out.video_grid_thw.shape, (1, 3))

  def test_video_grid_thw_consistent(self):
    """grid_h * patch_size == H_bar; grid_w * patch_size == W_bar."""
    frames = _rand_frames(8, 480, 640)
    out = preprocess_video_qwen3_vl(frames)
    _, _, T, H_bar, W_bar = out.pixel_values_videos.shape
    grid = out.video_grid_thw[0]
    self.assertEqual(grid[0] * QWEN3_VL_TEMPORAL_PATCH_SIZE, T)
    self.assertEqual(grid[1] * QWEN3_VL_PATCH_SIZE, H_bar)
    self.assertEqual(grid[2] * QWEN3_VL_PATCH_SIZE, W_bar)

  # ── Normalisation ──────────────────────────────────────────────────────────

  def test_pixel_range_video(self):
    frames = _rand_frames(4, 240, 320)
    out = preprocess_video_qwen3_vl(frames)
    pv = out.pixel_values_videos
    self.assertGreater(float(pv.min()), -1.2)
    self.assertLess(float(pv.max()), 1.2)

  def test_finite_pixel_values_video(self):
    frames = _rand_frames(4, 240, 320)
    out = preprocess_video_qwen3_vl(frames)
    self.assertTrue(np.all(np.isfinite(out.pixel_values_videos)))

  # ── Output dtype ───────────────────────────────────────────────────────────

  def test_output_dtype_float32_video(self):
    frames = _rand_frames(4, 240, 320)
    out = preprocess_video_qwen3_vl(frames)
    self.assertEqual(out.pixel_values_videos.dtype, np.float32)

  def test_video_grid_dtype_int32(self):
    frames = _rand_frames(4, 240, 320)
    out = preprocess_video_qwen3_vl(frames)
    self.assertEqual(out.video_grid_thw.dtype, np.int32)

  # ── Input type variants ────────────────────────────────────────────────────

  def test_list_of_frames(self):
    """List of (H, W, 3) arrays as input."""
    frames = [_rand_image(240, 320, seed=i) for i in range(4)]
    out = preprocess_video_qwen3_vl(frames)
    self.assertEqual(out.num_videos, 1)
    self.assertIsNotNone(out.pixel_values_videos)

  def test_single_frame_becomes_padded(self):
    """A single frame (T=1) is padded up to temporal_patch_size."""
    frame = _rand_image(64, 64)
    # frame as (1, H, W, 3)
    frames = frame[np.newaxis]
    out = preprocess_video_qwen3_vl(frames)
    T_padded = out.pixel_values_videos.shape[2]
    self.assertEqual(T_padded % QWEN3_VL_TEMPORAL_PATCH_SIZE, 0)
    self.assertGreaterEqual(T_padded, QWEN3_VL_TEMPORAL_PATCH_SIZE)

  def test_frame_sampling_reduces_count(self):
    """When fps-based sampling is used, frame count should be ≤ original."""
    frames = _rand_frames(60, 240, 320)
    out = preprocess_video_qwen3_vl(frames, fps=2.0, source_fps=30.0,
                                    min_frames=4, max_frames=32)
    T_padded = out.pixel_values_videos.shape[2]
    # target_count = 60 * 2 / 30 = 4, padded to multiple of 2 = 4
    self.assertLessEqual(T_padded, frames.shape[0])

  def test_total_pixels_within_bounds(self):
    """T×H×W must stay within video pixel bounds after resize."""
    frames = _rand_frames(16, 720, 1280)
    out = preprocess_video_qwen3_vl(frames)
    _, _, T, H, W = out.pixel_values_videos.shape
    grid = out.video_grid_thw[0]
    t_bar = grid[0] * QWEN3_VL_TEMPORAL_PATCH_SIZE
    total = t_bar * H * W
    # Allow some slack due to rounding.
    self.assertLessEqual(total, QWEN3_VL_VIDEO_MAX_PIXELS * 2)


# ─────────────────────────────────────────────────────────────────────────────
# add_extra_tokens_for_images_qwen3_vl
# ─────────────────────────────────────────────────────────────────────────────

class TestAddExtraTokensForImages(unittest.TestCase):

  def _make_output(self, grid_thw):
    """Build a minimal Qwen3VLPreprocessorOutput with given grid."""
    out = Qwen3VLPreprocessorOutput()
    out.pixel_grid_thw = np.array([grid_thw], dtype=np.int32)  # (1, 3)
    return out

  def test_single_image_196_tokens_at_448(self):
    """Fixed 448×448 → grid [1, 28, 28] → 196 tokens per placeholder."""
    grid = [1, 28, 28]
    out = self._make_output(grid)
    tokens = np.array([1, 2, QWEN3_VL_IMAGE_TOKEN, 3], dtype=np.int32)
    result = add_extra_tokens_for_images_qwen3_vl(tokens, out)
    expected_count = (1 * 28 * 28) // (QWEN3_VL_SPATIAL_MERGE_SIZE ** 2)  # 196
    image_token_positions = np.where(result == QWEN3_VL_IMAGE_TOKEN)[0]
    self.assertEqual(len(image_token_positions), expected_count)

  def test_no_image_tokens(self):
    """If there are no image tokens, output should equal input."""
    out = self._make_output([1, 28, 28])
    tokens = np.array([1, 2, 3], dtype=np.int32)
    result = add_extra_tokens_for_images_qwen3_vl(tokens, out)
    np.testing.assert_array_equal(result, tokens)

  def test_dynamic_resolution_token_count(self):
    """Dynamic grid [1, 6, 10] (96×160 image) → 6*10//4 = 15 tokens."""
    grid = [1, 6, 10]
    out = self._make_output(grid)
    tokens = np.array([QWEN3_VL_IMAGE_TOKEN], dtype=np.int32)
    result = add_extra_tokens_for_images_qwen3_vl(tokens, out)
    expected = (1 * 6 * 10) // (QWEN3_VL_SPATIAL_MERGE_SIZE ** 2)  # 15
    self.assertEqual(len(result), expected)
    self.assertTrue(np.all(result == QWEN3_VL_IMAGE_TOKEN))

  def test_none_processor_output_is_passthrough(self):
    """None processor_output → no expansion."""
    out = Qwen3VLPreprocessorOutput()
    tokens = np.array([QWEN3_VL_IMAGE_TOKEN, 1, 2], dtype=np.int32)
    result = add_extra_tokens_for_images_qwen3_vl(tokens, out)
    np.testing.assert_array_equal(result, tokens)

  def test_output_dtype_int32(self):
    out = self._make_output([1, 28, 28])
    tokens = np.array([QWEN3_VL_IMAGE_TOKEN], dtype=np.int32)
    result = add_extra_tokens_for_images_qwen3_vl(tokens, out)
    self.assertEqual(result.dtype, np.int32)


# ─────────────────────────────────────────────────────────────────────────────
# add_extra_tokens_for_video_qwen3_vl
# ─────────────────────────────────────────────────────────────────────────────

class TestAddExtraTokensForVideo(unittest.TestCase):

  def _make_video_output(self, grid_thw):
    out = Qwen3VLPreprocessorOutput()
    out.video_grid_thw = np.array([grid_thw], dtype=np.int32)
    return out

  def test_video_token_expansion(self):
    """Single video placeholder → grid_t*grid_h*grid_w // 4 tokens."""
    grid = [4, 16, 22]  # example from 8-frame 256×352 video
    out = self._make_video_output(grid)
    tokens = np.array([QWEN3_VL_VIDEO_TOKEN], dtype=np.int32)
    result = add_extra_tokens_for_video_qwen3_vl(tokens, out)
    expected = (4 * 16 * 22) // (QWEN3_VL_SPATIAL_MERGE_SIZE ** 2)  # 352
    self.assertEqual(len(result), expected)
    self.assertTrue(np.all(result == QWEN3_VL_VIDEO_TOKEN))

  def test_no_video_tokens_passthrough(self):
    out = self._make_video_output([4, 16, 22])
    tokens = np.array([1, 2, 3], dtype=np.int32)
    result = add_extra_tokens_for_video_qwen3_vl(tokens, out)
    np.testing.assert_array_equal(result, tokens)

  def test_mixed_image_and_video_tokens(self):
    """Image tokens should not be touched by the video expander."""
    out = Qwen3VLPreprocessorOutput()
    out.video_grid_thw = np.array([[2, 8, 8]], dtype=np.int32)  # 32 video tokens
    tokens = np.array([QWEN3_VL_IMAGE_TOKEN, QWEN3_VL_VIDEO_TOKEN, 99], dtype=np.int32)
    result = add_extra_tokens_for_video_qwen3_vl(tokens, out)
    # IMAGE token should remain as-is (not expanded).
    self.assertEqual(result[0], QWEN3_VL_IMAGE_TOKEN)
    # VIDEO token expanded: 2*8*8//4 = 32 tokens.
    expected_vid_tokens = (2 * 8 * 8) // 4
    video_positions = np.where(result == QWEN3_VL_VIDEO_TOKEN)[0]
    self.assertEqual(len(video_positions), expected_vid_tokens)

  def test_output_dtype_int32(self):
    out = self._make_video_output([4, 16, 22])
    tokens = np.array([QWEN3_VL_VIDEO_TOKEN], dtype=np.int32)
    result = add_extra_tokens_for_video_qwen3_vl(tokens, out)
    self.assertEqual(result.dtype, np.int32)


# ─────────────────────────────────────────────────────────────────────────────
# reformat_prompt_qwen3_vl
# ─────────────────────────────────────────────────────────────────────────────

class TestReformatPromptQwen3VL(unittest.TestCase):

  def test_image_placeholder_replaced(self):
    """Image placeholder must become the Qwen image vision token."""
    result = reformat_prompt_qwen3_vl("Describe <|image|>.", "<|image|>", num_images=1)
    self.assertIn(QWEN3_VL_IMAGE_PAD_STR, result)
    self.assertNotIn("<|image|>", result)

  def test_chat_template_wrapping(self):
    """Result must be wrapped in Qwen chat template."""
    result = reformat_prompt_qwen3_vl("Hello", "<|image|>", num_images=0)
    self.assertTrue(result.startswith("<|im_start|>user\n"))
    self.assertIn("<|im_end|>", result)
    self.assertIn("<|im_start|>assistant\n", result)

  def test_missing_placeholders_prepended(self):
    """If num_images > occurrences in prompt, extras are prepended."""
    result = reformat_prompt_qwen3_vl("Describe this.", "<|image|>", num_images=2)
    count = result.count(QWEN3_VL_IMAGE_PAD_STR)
    self.assertEqual(count, 2)

  def test_video_placeholder_replaced(self):
    """Video placeholder must become the Qwen video vision token."""
    result = reformat_prompt_qwen3_vl(
        "Describe <|video|>.", "<|image|>",
        num_images=0, video_placeholder="<|video|>", num_videos=1
    )
    self.assertIn(QWEN3_VL_VIDEO_PAD_STR, result)
    self.assertNotIn("<|video|>", result)

  def test_missing_video_placeholders_prepended(self):
    """If num_videos > occurrences, extras are prepended."""
    result = reformat_prompt_qwen3_vl(
        "Analyse.", "<|image|>",
        num_images=0, video_placeholder="<|video|>", num_videos=2
    )
    count = result.count(QWEN3_VL_VIDEO_PAD_STR)
    self.assertEqual(count, 2)

  def test_image_and_video_together(self):
    """Both image and video placeholders can appear in one prompt."""
    prompt = "<|image|> compare with <|video|>"
    result = reformat_prompt_qwen3_vl(
        prompt, "<|image|>", num_images=1, video_placeholder="<|video|>", num_videos=1
    )
    self.assertEqual(result.count(QWEN3_VL_IMAGE_PAD_STR), 1)
    self.assertEqual(result.count(QWEN3_VL_VIDEO_PAD_STR), 1)


# ─────────────────────────────────────────────────────────────────────────────
# get_image_offsets_qwen3_vl / get_video_offsets_qwen3_vl
# ─────────────────────────────────────────────────────────────────────────────

class TestOffsetFunctions(unittest.TestCase):

  def test_image_offset_196(self):
    """Fixed 448×448 → grid [1, 28, 28] → offset = 196 − 1 = 195 per image."""
    out = Qwen3VLPreprocessorOutput()
    out.pixel_grid_thw = np.array([[1, 28, 28]], dtype=np.int32)
    offset = get_image_offsets_qwen3_vl(out)
    self.assertEqual(offset, 195)

  def test_image_offset_dynamic(self):
    """Dynamic grid [1, 6, 10] → tokens=15, offset=14."""
    out = Qwen3VLPreprocessorOutput()
    out.pixel_grid_thw = np.array([[1, 6, 10]], dtype=np.int32)
    offset = get_image_offsets_qwen3_vl(out)
    self.assertEqual(offset, 14)  # 15 - 1

  def test_image_offset_none_output(self):
    self.assertEqual(get_image_offsets_qwen3_vl(None), 0)

  def test_image_offset_no_grid(self):
    out = Qwen3VLPreprocessorOutput()
    self.assertEqual(get_image_offsets_qwen3_vl(out), 0)

  def test_video_offset_positive(self):
    out = Qwen3VLPreprocessorOutput()
    out.video_grid_thw = np.array([[4, 16, 22]], dtype=np.int32)
    offset = get_video_offsets_qwen3_vl(out)
    expected = (4 * 16 * 22) // 4 - 1  # 352 - 1 = 351
    self.assertEqual(offset, expected)

  def test_video_offset_none_output(self):
    self.assertEqual(get_video_offsets_qwen3_vl(None), 0)

  def test_video_offset_no_grid(self):
    out = Qwen3VLPreprocessorOutput()
    self.assertEqual(get_video_offsets_qwen3_vl(out), 0)


# ─────────────────────────────────────────────────────────────────────────────
# End-to-end round-trip: image preprocessing → token expansion
# ─────────────────────────────────────────────────────────────────────────────

class TestEndToEndImageTokenExpansion(unittest.TestCase):

  def test_dynamic_roundtrip(self):
    """preprocess_mm_data_qwen3_vl grid must be consistent with token expansion."""
    img = _rand_image(224, 320)
    proc_out = preprocess_mm_data_qwen3_vl(img)
    tokens = np.array([0, QWEN3_VL_IMAGE_TOKEN, 1], dtype=np.int32)
    expanded = add_extra_tokens_for_images_qwen3_vl(tokens, proc_out)
    merge_sq = QWEN3_VL_SPATIAL_MERGE_SIZE ** 2
    grid = proc_out.pixel_grid_thw[0]
    expected_image_tokens = int(grid[0] * grid[1] * grid[2]) // merge_sq
    actual_image_tokens = int((expanded == QWEN3_VL_IMAGE_TOKEN).sum())
    self.assertEqual(actual_image_tokens, expected_image_tokens)

  def test_fixed_size_roundtrip_196_tokens(self):
    """force_size=(448,448) → exactly 196 image tokens after expansion."""
    img = _rand_image(100, 150)
    proc_out = preprocess_mm_data_qwen3_vl(img, force_size=(448, 448))
    tokens = np.array([QWEN3_VL_IMAGE_TOKEN], dtype=np.int32)
    expanded = add_extra_tokens_for_images_qwen3_vl(tokens, proc_out)
    self.assertEqual(len(expanded), 196)

  def test_video_roundtrip(self):
    """preprocess_video_qwen3_vl grid must match video token expansion."""
    frames = _rand_frames(8, 240, 320)
    proc_out = preprocess_video_qwen3_vl(frames)
    tokens = np.array([QWEN3_VL_VIDEO_TOKEN], dtype=np.int32)
    expanded = add_extra_tokens_for_video_qwen3_vl(tokens, proc_out)
    merge_sq = QWEN3_VL_SPATIAL_MERGE_SIZE ** 2
    grid = proc_out.video_grid_thw[0]
    expected = int(grid[0] * grid[1] * grid[2]) // merge_sq
    self.assertEqual(len(expanded), expected)

  def test_offset_equals_expansion_minus_one(self):
    """get_image_offsets_qwen3_vl must equal (expanded tokens − 1)."""
    img = _rand_image(448, 448)
    proc_out = preprocess_mm_data_qwen3_vl(img)
    tokens = np.array([QWEN3_VL_IMAGE_TOKEN], dtype=np.int32)
    expanded = add_extra_tokens_for_images_qwen3_vl(tokens, proc_out)
    offset = get_image_offsets_qwen3_vl(proc_out)
    self.assertEqual(offset, len(expanded) - 1)


if __name__ == "__main__":
  unittest.main()
