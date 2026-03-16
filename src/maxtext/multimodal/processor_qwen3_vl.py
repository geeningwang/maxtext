"""Qwen3-VL processor logic for handling vision-language inputs."""

from maxtext.multimodal import utils as mm_utils

QWEN3_VL_IMAGE_TOKEN = 151655
QWEN3_VL_VIDEO_TOKEN = 151656
QWEN3_VL_IMAGE_SIZE = 448
QWEN3_VL_NUM_FRAMES = 2  # Must be divisible by temporal_patch_size (2)


def get_dummy_image_shape_for_init_qwen3_vl(batch_size=1, num_frames=None):
  """Return the shape of the dummy image for Qwen3-VL model's initialization.
  
  Args:
    batch_size: Batch size for the dummy image
    num_frames: Number of frames. If None, defaults to QWEN3_VL_NUM_FRAMES (1 for images)
    
  Returns:
    Shape tuple: (batch_size, num_channels, num_frames, height, width)
  """
  if num_frames is None:
    num_frames = QWEN3_VL_NUM_FRAMES
  
  image_shape = (
      batch_size,
      mm_utils.NUM_IMAGE_CHANNELS,
      num_frames,
      QWEN3_VL_IMAGE_SIZE,
      QWEN3_VL_IMAGE_SIZE,
  )
  return image_shape
