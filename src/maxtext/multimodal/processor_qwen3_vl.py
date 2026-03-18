"""Qwen3-VL processor logic for handling vision-language inputs."""

from dataclasses import dataclass

import numpy as np
from PIL import Image

from maxtext.multimodal import utils as mm_utils

# Token IDs for Qwen3-VL (identical to Qwen3-Omni)
QWEN3_VL_IMAGE_TOKEN = 151655
QWEN3_VL_VIDEO_TOKEN = 151656

# Image size / patch constants
QWEN3_VL_IMAGE_SIZE = 448          # Fixed training/inference resolution
QWEN3_VL_PATCH_SIZE = 16           # Patch size of the vision encoder
QWEN3_VL_TEMPORAL_PATCH_SIZE = 2   # Temporal patch size (images repeated 2x)
QWEN3_VL_SPATIAL_MERGE_SIZE = 2    # Spatial merge size (2×2 → 1 token)
QWEN3_VL_NUM_FRAMES = 2            # Number of temporal frames per image

# Normalisation constants (same as Qwen3-Omni)
QWEN3_VL_IMAGE_MEAN = 127.5
QWEN3_VL_IMAGE_STD = 127.5

# Special token strings used in the chat template
QWEN3_VL_VISION_START_STR = "<|vision_start|>"
QWEN3_VL_VISION_END_STR = "<|vision_end|>"
QWEN3_VL_IMAGE_PAD_STR = "<|image_pad|>"


@dataclass
class Qwen3VLPreprocessorOutput(mm_utils.PreprocessorOutput):
  """Holds the output of the Qwen3-VL image preprocessor.

  Attributes (in addition to the base class):
    pixel_grid_thw: Integer array of shape (num_images, 3) with
                    [temporal, height, width] grid dimensions before merging.
  """
  pixel_grid_thw: None | np.ndarray = None


def preprocess_mm_data_qwen3_vl(image):
  """Preprocesses image(s) for Qwen3-VL SFT training.

  Resizes each image to QWEN3_VL_IMAGE_SIZE × QWEN3_VL_IMAGE_SIZE (448×448),
  normalises, and formats as (N, C, T, H, W) where T = QWEN3_VL_NUM_FRAMES.

  Args:
    image: A single np.ndarray (H, W, C) or a list of np.ndarray images.

  Returns:
    Qwen3VLPreprocessorOutput with:
      pixel_values: shape (N, C, T, H, W) = (N, 3, 2, 448, 448)
      pixel_grid_thw: shape (N, 3) — [grid_t, grid_h, grid_w] per image
      num_images: N
  """
  images_in = image if isinstance(image, list) else [image]
  images_out = []
  grids_thw = []

  for img in images_in:
    pil_img = Image.fromarray(img)
    pil_img = pil_img.resize(
        (QWEN3_VL_IMAGE_SIZE, QWEN3_VL_IMAGE_SIZE), Image.BICUBIC
    )
    img_np = np.array(pil_img).astype(np.float32)

    # Normalise: (pixel - mean) / std
    img_np = (img_np - QWEN3_VL_IMAGE_MEAN) / QWEN3_VL_IMAGE_STD

    # HWC → CHW
    img_np = np.transpose(img_np, (2, 0, 1))  # (C, H, W)

    # Add temporal dimension and repeat for temporal_patch_size
    img_np = np.expand_dims(img_np, axis=1)           # (C, 1, H, W)
    img_np = np.repeat(img_np, QWEN3_VL_TEMPORAL_PATCH_SIZE, axis=1)  # (C, T, H, W)

    images_out.append(img_np)

    grid_t = QWEN3_VL_NUM_FRAMES // QWEN3_VL_TEMPORAL_PATCH_SIZE  # 1
    grid_h = QWEN3_VL_IMAGE_SIZE // QWEN3_VL_PATCH_SIZE            # 28
    grid_w = QWEN3_VL_IMAGE_SIZE // QWEN3_VL_PATCH_SIZE            # 28
    grids_thw.append(np.array([grid_t, grid_h, grid_w], dtype=np.int32))

  pixel_values = np.stack(images_out, axis=0)   # (N, C, T, H, W)
  pixel_grid_thw = np.stack(grids_thw, axis=0)  # (N, 3)

  return Qwen3VLPreprocessorOutput(
      num_images=len(images_in),
      pixel_values=pixel_values,
      pixel_grid_thw=pixel_grid_thw,
  )


def reformat_prompt_qwen3_vl(prompt, image_placeholder, num_images):
  """Reformat a prompt for Qwen3-VL SFT training.

  Replaces the generic image placeholder with the Qwen3-VL vision token
  sequence and wraps the prompt in the Qwen chat template.

  Args:
    prompt: Raw user prompt string (may contain image_placeholder).
    image_placeholder: Generic placeholder string (e.g. "<|image|>").
    num_images: Total number of images for this example.

  Returns:
    Formatted prompt string ready for tokenisation.
  """
  qwen_img = f"{QWEN3_VL_VISION_START_STR}{QWEN3_VL_IMAGE_PAD_STR}{QWEN3_VL_VISION_END_STR}"

  # Replace existing placeholder occurrences
  if image_placeholder in prompt:
    prompt = prompt.replace(image_placeholder, qwen_img)

  # Prepend any missing image placeholders at the start
  count = prompt.count(qwen_img)
  if count < num_images:
    prompt = qwen_img * (num_images - count) + prompt

  return f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"


def add_extra_tokens_for_images_qwen3_vl(tokens, processor_output):
  """Expand each <|image_pad|> token to the full vision token sequence.

  Each single placeholder token is replaced by
  grid_t × grid_h × grid_w // spatial_merge_size² tokens. For the fixed
  448×448 training resolution this is 1 × 28 × 28 // 4 = 196 tokens.

  Args:
    tokens: 1-D int array of token IDs (after tokenisation).
    processor_output: Qwen3VLPreprocessorOutput with pixel_grid_thw.

  Returns:
    New 1-D int32 np.ndarray with placeholders expanded.
  """
  pixel_grid_thw = getattr(processor_output, "pixel_grid_thw", None)
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
        and pixel_grid_thw is not None
        and image_idx < len(pixel_grid_thw)
    ):
      grid = pixel_grid_thw[image_idx]
      num_tokens = int((grid[0] * grid[1] * grid[2]) // merge_length)
      new_tokens.extend([QWEN3_VL_IMAGE_TOKEN] * num_tokens)
      image_idx += 1
    else:
      new_tokens.append(token)

  return np.array(new_tokens, dtype=np.int32)


def get_image_offsets_qwen3_vl(processor_output):
  """Return total token-count expansion caused by image placeholder expansion.

  Each placeholder token expands to N tokens, so the offset per image is N-1.

  Args:
    processor_output: Qwen3VLPreprocessorOutput with pixel_grid_thw.

  Returns:
    Integer: total token count increase across all images.
  """
  if processor_output is None or getattr(processor_output, "pixel_grid_thw", None) is None:
    return 0

  merge_length = QWEN3_VL_SPATIAL_MERGE_SIZE ** 2  # 4
  total_offset = 0
  for grid in processor_output.pixel_grid_thw:
    num_tokens = int((grid[0] * grid[1] * grid[2]) // merge_length)
    total_offset += num_tokens - 1  # -1 for the original placeholder token

  return total_offset


def get_dummy_image_shape_for_init_qwen3_vl(batch_size=1, num_frames=None):
  """Return the shape of the dummy image for Qwen3-VL model's initialization.

  Args:
    batch_size: Batch size for the dummy image.
    num_frames: Number of temporal frames. Defaults to QWEN3_VL_NUM_FRAMES.

  Returns:
    Shape tuple: (batch_size, num_channels, num_frames, height, width)
  """
  if num_frames is None:
    num_frames = QWEN3_VL_NUM_FRAMES

  return (
      batch_size,
      mm_utils.NUM_IMAGE_CHANNELS,
      num_frames,
      QWEN3_VL_IMAGE_SIZE,
      QWEN3_VL_IMAGE_SIZE,
  )
