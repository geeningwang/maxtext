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

"""Qwen3-VL specific model classes.

Qwen3-VL uses the same vision encoder architecture as Qwen3-Omni-Moe, but the
MaxText inference path feeds the vision encoder raw 5-D pixel tensors
``(batch, channels, T*temporal_patch_size, H*patch_size, W*patch_size)``
through a 3-D convolutional patch-embedding layer.  That 3-D conv sweeps over
the spatial dimensions in *raster-scan* order, so the resulting token sequence
is also in raster-scan order.

However, every other component in the vision encoder—the learned 2-D
positional embeddings (``Qwen3OmniMoeVisionPosEmbedInterpolate``), the 2-D
rotary position embeddings (``Qwen3OmniMoeVisionRotaryEmbedding``), and the
spatial patch merger (``Qwen3OmniMoeVisionPatchMerger``)—generates/expects
tokens in *2×2 spatial-block order*: all four patches of each
``spatial_merge_size × spatial_merge_size`` tile are laid out consecutively
before moving to the next tile.

``Qwen3VLVisionEncoder`` fixes this mismatch by inserting a static
permutation immediately after ``patch_embed`` that reorders the token sequence
from raster-scan to 2×2-block order.  Because ``num_frames``, ``height`` and
``width`` are all statically known at JIT-compile time, the permutation is
computed as a plain NumPy array and becomes a zero-cost constant index in the
XLA graph.
"""

import numpy as np
import jax.numpy as jnp
from flax import nnx

from maxtext.models.qwen3 import Qwen3OmniMoeVisionEncoder


def _raster_to_block_perm(num_frames: int, height: int, width: int, merge_size: int) -> np.ndarray:
  """Return an index array that reorders raster-scan tokens to 2×2-block order.

  ``perm[block_pos] = raster_pos`` so that ``tokens[:, perm, :]`` places the
  patch at spatial location ``(tile_row*m+ir, tile_col*m+ic)`` at the block
  position ``(tile_row*(W//m) + tile_col)*m^2 + (ir*m + ic)``.

  Args:
    num_frames: Number of temporal frames after temporal-patch folding.
    height:     Height in patches (H = image_height // patch_size).
    width:      Width in patches  (W = image_width  // patch_size).
    merge_size: Spatial merge block side length (``spatial_merge_size``).

  Returns:
    1-D int32 array of length ``num_frames * height * width``.
  """
  T, H, W, m = num_frames, height, width, merge_size
  t_idx  = np.arange(T)[:, None, None, None, None]
  tr_idx = np.arange(H // m)[None, :, None, None, None]
  tc_idx = np.arange(W // m)[None, None, :, None, None]
  ir_idx = np.arange(m)[None, None, None, :, None]
  ic_idx = np.arange(m)[None, None, None, None, :]

  raster = (t_idx * H * W + (tr_idx * m + ir_idx) * W + (tc_idx * m + ic_idx)).reshape(-1)
  block  = (t_idx * H * W + (tr_idx * (W // m) + tc_idx) * m * m + (ir_idx * m + ic_idx)).reshape(-1)

  perm = np.empty(T * H * W, dtype=np.int32)
  perm[block] = raster
  return perm


class Qwen3VLVisionEncoder(Qwen3OmniMoeVisionEncoder):
  """Vision encoder for Qwen3-VL models.

  Identical to ``Qwen3OmniMoeVisionEncoder`` except that a raster-to-block
  token reordering step is applied after ``patch_embed`` so that the token
  sequence matches the 2×2-block order assumed by the position embeddings,
  rotary embeddings, and patch merger.
  """

  def __call__(
      self,
      hidden_states,
      deterministic: bool = True,
  ):
    """Forward pass with raster-to-block patch reordering.

    Args:
        hidden_states: Raw pixel tensor of shape
            ``(batch, in_channels, T*temporal_patch_size, H*patch_size, W*patch_size)``.
        deterministic: Passed through to transformer blocks (disables dropout).

    Returns:
        Tuple of:
        - ``encoder_output``: shape ``(batch, T*H*W, hidden_size_for_vit)``
        - ``deep_features``: list of intermediate merger outputs, each of shape
          ``(batch, T*H*W // spatial_merge_size^2, out_hidden_size)``
    """
    _, _, num_frames, height, width = hidden_states.shape
    num_frames = num_frames // self.config.temporal_patch_size_for_vit
    height = height // self.config.patch_size_for_vit
    width = width // self.config.patch_size_for_vit

    x = self.patch_embed(hidden_states)

    # Reorder from raster-scan order (3-D conv output) to 2×2-block order so
    # that the token sequence aligns with the block-ordered position and rotary
    # embeddings as well as with the VisionPatchMerger grouping.
    perm = _raster_to_block_perm(int(num_frames), int(height), int(width), self.spatial_merge_size)
    x = x[:, perm, :]

    pos = self.pos_embed_interpolate(num_frames, height, width)
    pos = pos[jnp.newaxis, :, :]
    x = x + pos

    h_traj = []
    for i in range(self.depth):
      blk = getattr(self, f"blocks_{i}")
      x = blk(x, num_frames=num_frames, height=height, width=width)
      h_traj.append(x)

    deep_feats = []
    for i, idx in enumerate(self.deep_idx):
      merger = getattr(self, f"merger_{i}")
      deep_feats.append(merger(h_traj[idx]))

    return x, deep_feats
