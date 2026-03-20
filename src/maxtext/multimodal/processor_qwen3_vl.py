"""Qwen3-VL processor logic for handling vision-language inputs.

Implements HF-aligned dynamic-resolution preprocessing for images and video,
matching the reference implementations:
  - Image:  ``transformers.models.qwen2_vl.image_processing_qwen2_vl``
              (``Qwen2VLImageProcessor`` is re-used by Qwen3-VL for images)
  - Video:  ``transformers.models.qwen3_vl.video_processing_qwen3_vl``
              (``Qwen3VLVideoProcessor``)

References:
  https://huggingface.co/docs/transformers/en/model_doc/qwen3_vl
"""

from dataclasses import dataclass
import math
from typing import Union

import numpy as np
from PIL import Image

from maxtext.multimodal import utils as mm_utils

# ─── Token IDs ────────────────────────────────────────────────────────────────
QWEN3_VL_IMAGE_TOKEN = 151655
QWEN3_VL_VIDEO_TOKEN = 151656

# ─── ViT architecture constants (from Qwen3VLVisionConfig) ────────────────────
# Confirmed: Qwen3VLVisionConfig.patch_size=16, spatial_merge_size=2,
#            temporal_patch_size=2.
QWEN3_VL_PATCH_SIZE = 16           # Spatial patch side length (px)
QWEN3_VL_TEMPORAL_PATCH_SIZE = 2   # Temporal patch depth (frames per patch group)
QWEN3_VL_SPATIAL_MERGE_SIZE = 2    # PatchMerger: 2×2 spatial → 1 token

# Every spatial output dimension must be a multiple of this factor.
QWEN3_VL_RESIZE_FACTOR = QWEN3_VL_PATCH_SIZE * QWEN3_VL_SPATIAL_MERGE_SIZE  # 32

# ─── Image resolution bounds (Qwen2VLImageProcessor defaults) ─────────────────
QWEN3_VL_IMAGE_MIN_PIXELS = 56 * 56          #     3 136 px²
QWEN3_VL_IMAGE_MAX_PIXELS = 28 * 28 * 1280   # 1 003 520 px²

# Alias kept for backward compatibility with existing callers that reference
# QWEN3_VL_IMAGE_SIZE and QWEN3_VL_NUM_FRAMES.
QWEN3_VL_IMAGE_SIZE = 448
QWEN3_VL_NUM_FRAMES = 2

# ─── Video resolution & frame-sampling bounds (Qwen3VLVideoProcessor) ─────────
QWEN3_VL_VIDEO_MIN_PIXELS = 128 * 32 * 32    # 131 072 total T×H×W pixels
QWEN3_VL_VIDEO_MAX_PIXELS = 32 * 32 * 768    # 786 432 total T×H×W pixels
QWEN3_VL_VIDEO_DEFAULT_FPS = 2.0
QWEN3_VL_VIDEO_MIN_FRAMES = 4
QWEN3_VL_VIDEO_MAX_FRAMES = 768

# ─── Normalisation ─────────────────────────────────────────────────────────────
# mean = 0.5, std = 0.5 → maps uint8 [0, 255] to float32 [−1, +1].
# Consistent with Qwen3VLVideoProcessor (image_mean = image_std = [0.5, 0.5, 0.5]).
QWEN3_VL_IMAGE_MEAN = 127.5
QWEN3_VL_IMAGE_STD = 127.5

# ─── Special token strings (chat template) ────────────────────────────────────
QWEN3_VL_VISION_START_STR = "<|vision_start|>"
QWEN3_VL_VISION_END_STR = "<|vision_end|>"
QWEN3_VL_IMAGE_PAD_STR = "<|image_pad|>"
QWEN3_VL_VIDEO_PAD_STR = "<|video_pad|>"


@dataclass
class Qwen3VLPreprocessorOutput(mm_utils.PreprocessorOutput):
  """Holds the output of the Qwen3-VL multimodal preprocessor.

  Fields inherited from PreprocessorOutput:
    pixel_values:   (N, 3, T=2, H, W) float32 — normalised image tensor;
                    T always = QWEN3_VL_TEMPORAL_PATCH_SIZE.
    num_images:     N — number of images.

  Image fields defined here:
    image_grid_thw: (N, 3) int32 — [grid_t, grid_h, grid_w] per image.

  Video fields defined here:
    pixel_values_videos: (N_vid, 3, T_frames, H, W) float32 — T_frames is the
                         actual frame count (rounded up to temporal_patch_size
                         multiple).
    video_grid_thw:      (N_vid, 3) int32 — [grid_t, grid_h, grid_w] per video.
    num_videos:          N_vid.
  """
  image_grid_thw: None | np.ndarray = None
  pixel_values_videos: None | np.ndarray = None
  video_grid_thw: None | np.ndarray = None
  num_videos: int = 0


# ─── Dynamic-resize helpers ───────────────────────────────────────────────────

def smart_resize_image(
    height: int,
    width: int,
    factor: int = QWEN3_VL_RESIZE_FACTOR,
    min_pixels: int = QWEN3_VL_IMAGE_MIN_PIXELS,
    max_pixels: int = QWEN3_VL_IMAGE_MAX_PIXELS,
) -> tuple:
  """Resize (height, width) so that:

  1. Both dims are divisible by *factor*.
  2. Total pixels H×W is within [min_pixels, max_pixels].
  3. Aspect ratio is preserved as closely as possible.

  Port of ``smart_resize`` from
  ``transformers.models.qwen2_vl.image_processing_qwen2_vl``.

  Args:
    height: Input image height in pixels.
    width:  Input image width in pixels.
    factor: Resize granularity (all output dims are multiples of this).
    min_pixels: Minimum allowed total pixel count H×W.
    max_pixels: Maximum allowed total pixel count H×W.

  Returns:
    (resized_height, resized_width) — both divisible by *factor*.
  """
  if height == 0 or width == 0:
    raise ValueError(f"Image dimensions must be positive, got {height}×{width}")
  if max(height, width) / min(height, width) > 200:
    raise ValueError(
        f"Absolute aspect ratio must be < 200, "
        f"got {max(height, width) / min(height, width):.1f} for {height}×{width}"
    )
  h_bar = max(factor, round(height / factor) * factor)
  w_bar = max(factor, round(width / factor) * factor)
  if h_bar * w_bar > max_pixels:
    beta = math.sqrt((height * width) / max_pixels)
    h_bar = max(factor, math.floor(height / beta / factor) * factor)
    w_bar = max(factor, math.floor(width / beta / factor) * factor)
  elif h_bar * w_bar < min_pixels:
    beta = math.sqrt(min_pixels / (height * width))
    h_bar = math.ceil(height * beta / factor) * factor
    w_bar = math.ceil(width * beta / factor) * factor
  return h_bar, w_bar


def smart_resize_video(
    num_frames: int,
    height: int,
    width: int,
    temporal_factor: int = QWEN3_VL_TEMPORAL_PATCH_SIZE,
    factor: int = QWEN3_VL_RESIZE_FACTOR,
    min_pixels: int = QWEN3_VL_VIDEO_MIN_PIXELS,
    max_pixels: int = QWEN3_VL_VIDEO_MAX_PIXELS,
) -> tuple:
  """Resize video frame dimensions so that T×H×W ∈ [min_pixels, max_pixels].

  Port of ``smart_resize`` from
  ``transformers.models.qwen3_vl.video_processing_qwen3_vl``.

  Args:
    num_frames:     Number of video frames to be processed.
    height:         Frame height in pixels.
    width:          Frame width in pixels.
    temporal_factor: Frames must be a multiple of this (= QWEN3_VL_TEMPORAL_PATCH_SIZE).
    factor:          Spatial resize granularity.
    min_pixels:      Minimum total T×H×W pixel count.
    max_pixels:      Maximum total T×H×W pixel count.

  Returns:
    (resized_height, resized_width) — spatial dimensions for each frame.
  """
  if num_frames < temporal_factor:
    raise ValueError(
        f"num_frames={num_frames} must be ≥ temporal_factor={temporal_factor}"
    )
  if height < factor or width < factor:
    raise ValueError(
        f"Frame dimensions {height}×{width} must each be ≥ factor={factor}"
    )
  if max(height, width) / min(height, width) > 200:
    raise ValueError(
        f"Absolute aspect ratio must be < 200, "
        f"got {max(height, width) / min(height, width):.1f}"
    )
  h_bar = max(factor, round(height / factor) * factor)
  w_bar = max(factor, round(width / factor) * factor)
  t_bar = max(temporal_factor, round(num_frames / temporal_factor) * temporal_factor)
  if t_bar * h_bar * w_bar > max_pixels:
    beta = math.sqrt((num_frames * height * width) / max_pixels)
    h_bar = max(factor, math.floor(height / beta / factor) * factor)
    w_bar = max(factor, math.floor(width / beta / factor) * factor)
  elif t_bar * h_bar * w_bar < min_pixels:
    beta = math.sqrt(min_pixels / (num_frames * height * width))
    h_bar = math.ceil(height * beta / factor) * factor
    w_bar = math.ceil(width * beta / factor) * factor
  return h_bar, w_bar


def _normalise_hwc(img_np: np.ndarray) -> np.ndarray:
  """Normalise a HWC uint8 [0, 255] frame to CHW float32 [−1, +1]."""
  img_np = img_np.astype(np.float32)
  img_np = (img_np - QWEN3_VL_IMAGE_MEAN) / QWEN3_VL_IMAGE_STD
  return np.transpose(img_np, (2, 0, 1))  # HWC → CHW


# ─── Image preprocessing ──────────────────────────────────────────────────────

def preprocess_image_qwen3_vl(
    image: Union[np.ndarray, list],
    min_pixels: int = QWEN3_VL_IMAGE_MIN_PIXELS,
    max_pixels: int = QWEN3_VL_IMAGE_MAX_PIXELS,
    force_size: Union[tuple, None] = None,
) -> "Qwen3VLPreprocessorOutput":
  """Preprocesses image(s) for Qwen3-VL inference or training.

  Uses *dynamic resolution* by default: each image is resized to the nearest
  multiple of ``QWEN3_VL_RESIZE_FACTOR`` (= patch_size × merge_size = 32) such
  that the total pixel count stays within [min_pixels, max_pixels] while
  preserving the aspect ratio.

  Args:
    image: A single np.ndarray (H, W, C) or a list of np.ndarray images.
    min_pixels: Minimum total pixel count for dynamic resize.
    max_pixels: Maximum total pixel count for dynamic resize.
    force_size: If given, a ``(height, width)`` tuple to use instead of
      smart_resize (aspect ratio is NOT preserved). Pass
      ``(QWEN3_VL_IMAGE_SIZE, QWEN3_VL_IMAGE_SIZE)`` for training pipelines
      that require a fixed spatial shape.

  Returns:
    Qwen3VLPreprocessorOutput with:
      pixel_values:   (N, 3, 2, H_bar, W_bar) float32
      image_grid_thw: (N, 3) int32 — [grid_t, grid_h, grid_w] per image
      num_images:     N
  """
  images_in = image if isinstance(image, list) else [image]
  images_out = []
  grids_thw = []

  for img in images_in:
    # Accept both uint8 and float arrays; PIL.fromarray needs uint8.
    src = img if img.dtype == np.uint8 else img.clip(0, 255).astype(np.uint8)
    pil_img = Image.fromarray(src).convert("RGB")
    H, W = pil_img.height, pil_img.width

    if force_size is not None:
      H_bar, W_bar = int(force_size[0]), int(force_size[1])
    else:
      H_bar, W_bar = smart_resize_image(H, W, factor=QWEN3_VL_RESIZE_FACTOR,
                                        min_pixels=min_pixels, max_pixels=max_pixels)

    pil_img = pil_img.resize((W_bar, H_bar), Image.BICUBIC)
    img_chw = _normalise_hwc(np.array(pil_img))  # (C, H_bar, W_bar)

    # Add temporal dimension and repeat to form T=2 frames.
    img_chw = np.expand_dims(img_chw, axis=1)                           # (C, 1, H_bar, W_bar)
    img_chw = np.repeat(img_chw, QWEN3_VL_TEMPORAL_PATCH_SIZE, axis=1)  # (C, T, H_bar, W_bar)
    images_out.append(img_chw)

    grid_t = 1  # One temporal patch per image (T=2 frames → 1 group)
    grid_h = H_bar // QWEN3_VL_PATCH_SIZE
    grid_w = W_bar // QWEN3_VL_PATCH_SIZE
    grids_thw.append(np.array([grid_t, grid_h, grid_w], dtype=np.int32))

  try:
    pixel_values = np.stack(images_out, axis=0)   # (N, C, T, H_bar, W_bar)
  except ValueError as exc:
    shapes = [a.shape for a in images_out]
    raise ValueError(
        f"Cannot batch images with different spatial sizes: {shapes}. "
        "Use force_size=(H, W) or process images of equal resolution."
    ) from exc

  image_grid_thw = np.stack(grids_thw, axis=0)  # (N, 3)

  return Qwen3VLPreprocessorOutput(
      num_images=len(images_in),
      pixel_values=pixel_values,
      image_grid_thw=image_grid_thw,
  )


# ─── Video preprocessing ──────────────────────────────────────────────────────

def _load_video_frames(source: Union[str, np.ndarray, list]) -> np.ndarray:
  """Load video frames from various source types.

  Args:
    source: One of:
      - ``str`` path to a GIF/APNG file (loaded with PIL).
      - ``str`` path to a video file (MP4/AVI/…) — requires ``cv2`` to be
        installed (``pip install opencv-python-headless``).
      - ``np.ndarray`` of shape ``(T, H, W, 3)`` uint8 — used directly.
      - ``list`` of ``np.ndarray`` frames, each ``(H, W, 3)`` uint8.

  Returns:
    np.ndarray of shape ``(T, H, W, 3)`` uint8.
  """
  if isinstance(source, np.ndarray):
    if source.ndim == 3:
      return source[np.newaxis]      # single frame → (1, H, W, 3)
    if source.ndim != 4:
      raise ValueError(f"np.ndarray source must be 3-D or 4-D, got shape {source.shape}")
    return source.astype(np.uint8)

  if isinstance(source, list):
    frames = [np.asarray(f) if not isinstance(f, np.ndarray) else f for f in source]
    return np.stack(frames, axis=0).astype(np.uint8)

  if isinstance(source, str):
    # Try PIL first (handles GIF / APNG / multi-page TIFF).
    try:
      from PIL import ImageSequence  # pylint: disable=import-outside-toplevel
      pil = Image.open(source)
      frames = [np.array(f.convert("RGB")) for f in ImageSequence.Iterator(pil)]
      if len(frames) > 1:
        return np.stack(frames, axis=0).astype(np.uint8)
      # Single-frame still image — treat as 1-frame video.
      return np.stack(frames, axis=0).astype(np.uint8)
    except Exception:  # pylint: disable=broad-except
      pass

    # Fall back to cv2 for real video containers.
    try:
      import cv2  # pylint: disable=import-outside-toplevel
      cap = cv2.VideoCapture(source)
      if not cap.isOpened():
        raise IOError(f"cv2 could not open video file: {source}")
      frames = []
      while True:
        ret, frame = cap.read()
        if not ret:
          break
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
      cap.release()
      if not frames:
        raise IOError(f"No frames decoded from: {source}")
      return np.stack(frames, axis=0).astype(np.uint8)
    except ImportError:
      raise ImportError(
          f"Could not load video '{source}' using PIL. "
          "Install opencv-python-headless for MP4/AVI support:\n"
          "  pip install opencv-python-headless"
      )

  raise TypeError(f"Unsupported source type: {type(source)}")


def preprocess_video_qwen3_vl(
    source: Union[str, np.ndarray, list],
    min_pixels: int = QWEN3_VL_VIDEO_MIN_PIXELS,
    max_pixels: int = QWEN3_VL_VIDEO_MAX_PIXELS,
    fps: float = QWEN3_VL_VIDEO_DEFAULT_FPS,
    source_fps: float = 0.0,
    min_frames: int = QWEN3_VL_VIDEO_MIN_FRAMES,
    max_frames: int = QWEN3_VL_VIDEO_MAX_FRAMES,
) -> "Qwen3VLPreprocessorOutput":
  """Preprocesses a single video for Qwen3-VL inference.

  Matches the reference ``Qwen3VLVideoProcessor`` from HuggingFace:
    - Dynamic spatial resize using ``smart_resize_video``.
    - Temporal sampling at ``fps`` frames-per-second (uniform sampling).
    - Normalisation: mean = std = 0.5 → [0, 255] → [−1, +1].
    - Output shape: ``(1, 3, T_padded, H_bar, W_bar)`` where T_padded is
      rounded up to the nearest multiple of QWEN3_VL_TEMPORAL_PATCH_SIZE.

  Args:
    source: Video source — see ``_load_video_frames`` for accepted types.
    min_pixels: Minimum total T×H×W pixel count (after sampling + resize).
    max_pixels: Maximum total T×H×W pixel count.
    fps: Target frames per second to sample. Set to 0 to use all frames.
    source_fps: Original video frame rate (needed for fps-based sampling).
      Pass 0 to fall back to uniform sampling between min_frames and max_frames.
    min_frames: Minimum number of frames to keep.
    max_frames: Maximum number of frames to keep.

  Returns:
    Qwen3VLPreprocessorOutput with:
      pixel_values_videos: (1, 3, T_padded, H_bar, W_bar) float32
      video_grid_thw:      (1, 3) int32 — [grid_t, grid_h, grid_w]
      num_videos:          1
  """
  frames = _load_video_frames(source)  # (T_total, H, W, 3)
  T_total, H, W = frames.shape[:3]

  # ── Frame sampling ──────────────────────────────────────────────────────────
  if fps > 0 and source_fps > 0:
    target_count = int(T_total * fps / source_fps)
  else:
    target_count = T_total
  target_count = min(max(target_count, min_frames), max_frames)
  target_count = min(target_count, T_total)

  if target_count < T_total:
    indices = np.linspace(0, T_total - 1, target_count).round().astype(int)
    frames = frames[indices]

  # Ensure at least temporal_factor frames before resize (pad by repeating last).
  if frames.shape[0] < QWEN3_VL_TEMPORAL_PATCH_SIZE:
    pad_count = QWEN3_VL_TEMPORAL_PATCH_SIZE - frames.shape[0]
    frames = np.concatenate([frames, np.tile(frames[-1:], (pad_count, 1, 1, 1))], axis=0)
  T = frames.shape[0]

  # ── Spatial resize ──────────────────────────────────────────────────────────
  H_bar, W_bar = smart_resize_video(
      T, H, W,
      temporal_factor=QWEN3_VL_TEMPORAL_PATCH_SIZE,
      factor=QWEN3_VL_RESIZE_FACTOR,
      min_pixels=min_pixels,
      max_pixels=max_pixels,
  )
  if (H_bar, W_bar) != (H, W):
    resized = np.stack([
        np.array(Image.fromarray(f).resize((W_bar, H_bar), Image.BICUBIC))
        for f in frames
    ], axis=0)
  else:
    resized = frames

  # ── Temporal padding ────────────────────────────────────────────────────────
  # Round T up to the next multiple of temporal_patch_size by repeating the
  # last frame (mirrors HuggingFace Qwen3VLVideoProcessor).
  remainder = T % QWEN3_VL_TEMPORAL_PATCH_SIZE
  if remainder != 0:
    pad_count = QWEN3_VL_TEMPORAL_PATCH_SIZE - remainder
    resized = np.concatenate([resized, np.tile(resized[-1:], (pad_count, 1, 1, 1))], axis=0)
  T_padded = resized.shape[0]

  # ── Normalise and convert to (1, C, T, H, W) ────────────────────────────────
  # HWC uint8 → CHW float32 [-1, +1], then stack over time.
  chw_frames = np.stack([_normalise_hwc(f) for f in resized], axis=1)  # (C, T_padded, H_bar, W_bar)
  pixel_values_videos = chw_frames[np.newaxis]  # (1, C, T_padded, H_bar, W_bar)

  grid_t = T_padded // QWEN3_VL_TEMPORAL_PATCH_SIZE
  grid_h = H_bar // QWEN3_VL_PATCH_SIZE
  grid_w = W_bar // QWEN3_VL_PATCH_SIZE
  video_grid_thw = np.array([[grid_t, grid_h, grid_w]], dtype=np.int32)  # (1, 3)

  return Qwen3VLPreprocessorOutput(
      num_videos=1,
      pixel_values_videos=pixel_values_videos,
      video_grid_thw=video_grid_thw,
  )


def reformat_prompt_qwen3_vl(
    prompt,
    num_images,
    num_videos=0,
    image_placeholder="<|image|>",
    video_placeholder="<|video|>",
):
  """Reformat a prompt for Qwen3-VL inference or SFT training.

  Replaces generic image / video placeholders with the Qwen3-VL vision token
  sequences and wraps the prompt in the Qwen chat template.

  Args:
    prompt: Raw user prompt string.
    num_images: Number of images for this example.
    num_videos: Number of videos for this example.
    image_placeholder: Generic image placeholder (default ``"<|image|>"``).
    video_placeholder: Generic video placeholder (default ``"<|video|>"``).

  Returns:
    Formatted prompt string ready for tokenisation.
  """
  qwen_img = f"{QWEN3_VL_VISION_START_STR}{QWEN3_VL_IMAGE_PAD_STR}{QWEN3_VL_VISION_END_STR}"
  qwen_vid = f"{QWEN3_VL_VISION_START_STR}{QWEN3_VL_VIDEO_PAD_STR}{QWEN3_VL_VISION_END_STR}"

  # Replace image placeholders.
  if image_placeholder in prompt:
    prompt = prompt.replace(image_placeholder, qwen_img)
  count = prompt.count(qwen_img)
  if count < num_images:
    prompt = qwen_img * (num_images - count) + prompt

  # Replace video placeholders.
  if video_placeholder in prompt:
    prompt = prompt.replace(video_placeholder, qwen_vid)
  count_vid = prompt.count(qwen_vid)
  if count_vid < num_videos:
    prompt = qwen_vid * (num_videos - count_vid) + prompt

  return f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"


def add_extra_tokens_for_images_qwen3_vl(tokens, processor_output):
  """Expand each <|image_pad|> token to the full vision token sequence.

  Each single placeholder token is replaced by
  ``grid_t × grid_h × grid_w // spatial_merge_size²`` tokens, where
  ``image_grid_thw`` comes from the preprocessor output (dynamic resolution).

  Example (448×448 with patch_size=16, merge_size=2):
    grid_thw = [1, 28, 28] → 1×28×28 // 4 = 196 tokens per image.

  Args:
    tokens: 1-D int array of token IDs (after tokenisation).
    processor_output: Qwen3VLPreprocessorOutput with image_grid_thw.

  Returns:
    New 1-D int32 np.ndarray with placeholders expanded.
  """
  image_grid_thw = getattr(processor_output, "image_grid_thw", None)
  merge_length = QWEN3_VL_SPATIAL_MERGE_SIZE ** 2  # 4

  if not isinstance(tokens, np.ndarray):
    tokens = np.asarray(tokens)
  tokens = tokens.flatten()

  token_list = tokens.tolist()
  new_tokens = []
  image_idx = 0

  for token in token_list:
    if (
        token == QWEN3_VL_IMAGE_TOKEN
        and image_grid_thw is not None
        and image_idx < len(image_grid_thw)
    ):
      grid = image_grid_thw[image_idx]
      num_tokens = int((grid[0] * grid[1] * grid[2]) // merge_length)
      new_tokens.extend([QWEN3_VL_IMAGE_TOKEN] * num_tokens)
      image_idx += 1
    else:
      new_tokens.append(token)

  return np.array(new_tokens, dtype=np.int32)


def add_extra_tokens_for_video_qwen3_vl(tokens, processor_output):
  """Expand each <|video_pad|> token to the full video vision token sequence.

  Each placeholder is replaced by
  ``grid_t × grid_h × grid_w // spatial_merge_size²`` tokens.

  Args:
    tokens: 1-D int array of token IDs.
    processor_output: Qwen3VLPreprocessorOutput with video_grid_thw.

  Returns:
    New 1-D int32 np.ndarray with video placeholders expanded.
  """
  video_grid_thw = getattr(processor_output, "video_grid_thw", None)
  merge_length = QWEN3_VL_SPATIAL_MERGE_SIZE ** 2

  if not isinstance(tokens, np.ndarray):
    tokens = np.asarray(tokens)
  tokens = tokens.flatten()

  token_list = tokens.tolist()
  new_tokens = []
  video_idx = 0

  for token in token_list:
    if (
        token == QWEN3_VL_VIDEO_TOKEN
        and video_grid_thw is not None
        and video_idx < len(video_grid_thw)
    ):
      grid = video_grid_thw[video_idx]
      num_tokens = int((grid[0] * grid[1] * grid[2]) // merge_length)
      new_tokens.extend([QWEN3_VL_VIDEO_TOKEN] * num_tokens)
      video_idx += 1
    else:
      new_tokens.append(token)

  return np.array(new_tokens, dtype=np.int32)


def get_image_offsets_qwen3_vl(processor_output):
  """Return total token-count expansion caused by image placeholder expansion.

  Each placeholder token expands to N tokens, so the offset per image is N-1.

  Args:
    processor_output: Qwen3VLPreprocessorOutput with image_grid_thw.

  Returns:
    Integer: total token count increase across all images.
  """
  if processor_output is None or getattr(processor_output, "image_grid_thw", None) is None:
    return 0

  merge_length = QWEN3_VL_SPATIAL_MERGE_SIZE ** 2
  total_offset = 0
  for grid in processor_output.image_grid_thw:
    num_tokens = int((grid[0] * grid[1] * grid[2]) // merge_length)
    total_offset += num_tokens - 1
  return total_offset


def get_video_offsets_qwen3_vl(processor_output):
  """Return total token-count expansion caused by video placeholder expansion.

  Args:
    processor_output: Qwen3VLPreprocessorOutput with video_grid_thw.

  Returns:
    Integer: total token count increase across all videos.
  """
  if processor_output is None or getattr(processor_output, "video_grid_thw", None) is None:
    return 0

  merge_length = QWEN3_VL_SPATIAL_MERGE_SIZE ** 2
  total_offset = 0
  for grid in processor_output.video_grid_thw:
    num_tokens = int((grid[0] * grid[1] * grid[2]) // merge_length)
    total_offset += num_tokens - 1
  return total_offset


def merge_preprocessor_outputs_qwen3_vl(
    image_output: "Qwen3VLPreprocessorOutput",
    video_output: "Qwen3VLPreprocessorOutput",
) -> "Qwen3VLPreprocessorOutput":
  """Merge image and video preprocessor outputs for mixed image+video inference.

  Combines the image fields from *image_output* and the video fields from
  *video_output* into a single ``Qwen3VLPreprocessorOutput``.  This enables
  passing both images and a video to the model in a single execution.

  Args:
    image_output: Result of ``preprocess_image_qwen3_vl`` (image fields only).
    video_output: Result of ``preprocess_video_qwen3_vl`` (video fields only).

  Returns:
    ``Qwen3VLPreprocessorOutput`` with all image *and* video fields populated.
  """
  return Qwen3VLPreprocessorOutput(
      num_images=image_output.num_images,
      pixel_values=image_output.pixel_values,
      image_grid_thw=image_output.image_grid_thw,
      num_videos=video_output.num_videos,
      pixel_values_videos=video_output.pixel_values_videos,
      video_grid_thw=video_output.video_grid_thw,
  )


def get_dummy_image_shape_for_init_qwen3_vl(
    batch_size=1,
    num_frames=None,
    image_size=None,
):
  """Return the shape of the dummy image for Qwen3-VL model initialisation.

  Args:
    batch_size: Batch size for the dummy image.
    num_frames: Number of temporal frames. Defaults to QWEN3_VL_TEMPORAL_PATCH_SIZE.
    image_size: Spatial size (height = width). Defaults to QWEN3_VL_IMAGE_SIZE (448).

  Returns:
    Shape tuple: (batch_size, num_channels, num_frames, image_size, image_size)
  """
  if num_frames is None:
    num_frames = QWEN3_VL_TEMPORAL_PATCH_SIZE
  if image_size is None:
    image_size = QWEN3_VL_IMAGE_SIZE

  return (
      batch_size,
      mm_utils.NUM_IMAGE_CHANNELS,
      num_frames,
      image_size,
      image_size,
  )
