#!/usr/bin/env python3
"""Qwen3-VL inference demo — JAX / NNX backend.

Runs Qwen3-VL directly on the JAX NNX model with a real orbax checkpoint.
No serving layer is involved; every forward pass is called via a JIT-compiled
NNX merge+decoder step.  This is the *reference TPU implementation*.

Usage::

    python qwen3_vl_demo_jax.py \\
        --image tests/assets/image1.jpg \\
        --prompt "Describe what you see in the image."
"""

import argparse
import json
import os
import sys
import time
from typing import Optional

import jax
import jax.numpy as jnp
import jax.tree_util as jtu
import numpy as np
from PIL import Image

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

BACKEND = "jax"
DEFAULT_CHECKPOINT = "tests/assets/qwen3_vl_2b_orbax"
DEFAULT_TOKENIZER  = "Qwen/Qwen3-VL-2B-Instruct"
DEFAULT_PROMPT     = "Describe what you see in the image."

_VIT_INPUT_SIZE = 448   # resize all images to this spatial resolution
_BLOCK          = 512   # TPU splash-attention block size (fixed length must be a multiple)
_FIXED_LEN      = 1024  # token buffer length (must be multiple of _BLOCK)


# ---------------------------------------------------------------------------
# Shared output helper  (identical across all three demo scripts)
# ---------------------------------------------------------------------------

def _print_result(result: dict, output_json: bool = False) -> None:
  """Print *result* in the common demo output format."""
  if output_json:
    print(json.dumps(result, indent=2))
    return
  W = 80
  print("=" * W)
  print(
      f"Qwen3-VL Demo  "
      f"[backend={result['backend']}  model={result.get('model', '')}]"
  )
  print("=" * W)
  print(f"Image(s) : {', '.join(result['image'])}")
  print(f"Prompt   : {result['prompt']!r}")
  print("-" * W)
  print("RESPONSE")
  print("-" * W)
  print(result["response"] or "(empty — model produced only special tokens)")
  print("-" * W)
  print(
      f"Generated {result['tokens']} tokens "
      f"in {result['elapsed']}s  ({result['tok_per_sec']} tok/s)"
  )
  print("=" * W)


# ---------------------------------------------------------------------------
# Image / tokenisation helpers
# ---------------------------------------------------------------------------

def _preprocess_image(image_path: str, target_size: int = _VIT_INPUT_SIZE) -> np.ndarray:
  """Load an image and return a ``(1, 3, 2, H, W)`` float32 array.

  Qwen3-VL normalises with mean=0.5, std=0.5 → values in [-1, 1].
  The temporal dim is 2 (``temporal_patch_size``) so the single frame is
  duplicated along that axis.
  """
  img = Image.open(image_path).convert("RGB")
  img = img.resize((target_size, target_size), Image.Resampling.LANCZOS)
  arr = np.array(img, dtype=np.float32) / 127.5 - 1.0   # (H, W, C)
  arr = arr.transpose(2, 0, 1)                            # (C, H, W)
  arr = np.stack([arr, arr], axis=1)                      # (C, 2, H, W)
  return arr[None, ...]                                   # (1, C, 2, H, W)


def _build_input_ids(tokenizer, prompt: str, num_vis_tokens: int) -> list:
  """Return the full prompt token IDs with visual placeholders embedded.

  Inserts exactly *num_vis_tokens* ``<|image_pad|>`` tokens at the image
  position inside the chat template (Qwen3-VL token ID 151655).
  """
  IMAGE_TOKEN = "<|image_pad|>"
  image_section = (
      "<|vision_start|>" + IMAGE_TOKEN * num_vis_tokens + "<|vision_end|>"
  )
  messages = [
      {"role": "system", "content": "You are a helpful assistant."},
      {"role": "user", "content": image_section + prompt},
  ]
  text = tokenizer.apply_chat_template(
      messages, tokenize=False, add_generation_prompt=True
  )
  return tokenizer.encode(text, add_special_tokens=False)


def _compute_mrope_positions(
    config,
    input_ids: list,
    fixed_len: int,
    vit_input_size: int,
) -> jnp.ndarray:
  """Compute mRoPE position IDs for the full fixed-length buffer.

  Returns a ``(3, 1, fixed_len)`` int32 JAX array where the three axes
  correspond to the temporal, height, and width rope dimensions.

  Positions for the prompt are computed by :func:`get_rope_index`;
  generation steps receive sequential IDs starting at ``max_pos + 1``.
  """
  from maxtext.multimodal.processor_qwen3_omni import get_rope_index

  patch  = config.patch_size_for_vit
  merge  = config.spatial_merge_size_for_vit
  grid_h = vit_input_size // patch   # 28 for 448px, 16px patch
  grid_w = vit_input_size // patch
  image_grid_thw = np.array([[1, grid_h, grid_w]], dtype=np.int32)
  ids_2d          = np.array(input_ids, dtype=np.int32)[np.newaxis, :]  # (1, seq)
  pos_np, _ = get_rope_index(
      ids_2d,
      image_grid_thw=image_grid_thw,
      spatial_merge_size=merge,
  )  # (3, 1, seq_len)

  seq_len = len(input_ids)
  max_pos = int(pos_np.max())
  gen_pos = (np.arange(fixed_len - seq_len) + max_pos + 1)
  gen_pos_3d = np.broadcast_to(
      gen_pos[np.newaxis, np.newaxis, :], (3, 1, fixed_len - seq_len)
  )
  full_pos = np.concatenate([pos_np, gen_pos_3d], axis=2)  # (3, 1, fixed_len)
  return jnp.asarray(full_pos, dtype=jnp.int32)


# ---------------------------------------------------------------------------
# Checkpoint restore helper
# ---------------------------------------------------------------------------

def _restore_checkpoint(state, checkpoint_items_path: str):
  """Restore an orbax checkpoint into *state* using SingleDeviceSharding.

  The orbax checkpoint was originally saved from a 28-layer Linen TrainState
  trained on many devices; this helper maps every tensor to the single local
  device.  Returns an updated NNX state pytree.
  """
  import orbax.checkpoint as ocp
  from etils import epath

  device         = jax.devices()[0]
  single_sharding = jax.sharding.SingleDeviceSharding(device)

  def _to_abstract_dict(state_obj):
    result = {}
    for path, leaf in jtu.tree_flatten_with_path(state_obj)[0]:
      val  = leaf.value if hasattr(leaf, "value") else leaf
      keys = [p.key for p in path if hasattr(p, "key")]
      if not keys:
        continue
      d = result
      for k in keys[:-1]:
        d = d.setdefault(k, {})
      d[keys[-1]] = jax.ShapeDtypeStruct(
          val.shape, val.dtype, sharding=single_sharding
      )
    return result

  abstract_params = _to_abstract_dict(state)
  abstract_for_restore = {"params": abstract_params}
  restore_args = jtu.tree_map(
      lambda _: ocp.type_handlers.ArrayRestoreArgs(sharding=single_sharding),
      abstract_for_restore,
  )

  ckptr = ocp.Checkpointer(
      ocp.PyTreeCheckpointHandler(use_ocdbt=True, use_zarr3=True)
  )
  print(f"[{BACKEND}]   restoring from: {checkpoint_items_path}")
  restored = ckptr.restore(
      epath.Path(checkpoint_items_path),
      item={"params": abstract_for_restore},
      transforms={},
      restore_args={"params": restore_args},
  )
  checkpoint_params = restored["params"]["params"]

  leaves_with_path, treedef = jtu.tree_flatten_with_path(state)
  new_leaves = []
  loaded, missing = 0, 0
  for path, leaf in leaves_with_path:
    val  = leaf.value if hasattr(leaf, "value") else leaf
    keys = [p.key for p in path if hasattr(p, "key")]
    ckpt = checkpoint_params
    try:
      for k in keys:
        ckpt = ckpt[k]
      new_leaves.append(jnp.asarray(ckpt, dtype=val.dtype))
      loaded += 1
    except (KeyError, TypeError):
      new_leaves.append(val)
      missing += 1

  print(
      f"[{BACKEND}]   loaded {loaded} tensors from checkpoint, "
      f"kept {missing} at random init."
  )
  return treedef.unflatten(new_leaves)


# ---------------------------------------------------------------------------
# Inference class
# ---------------------------------------------------------------------------

class Qwen3VLDemoJAX:
  """JAX / NNX inference class for Qwen3-VL.

  On construction the model is initialised (random weights), the orbax
  checkpoint is restored, and a JIT-compiled single-step decode function
  is prepared.  Inference calls happen via :meth:`run`.
  """

  def __init__(
      self,
      checkpoint_dir: str = DEFAULT_CHECKPOINT,
      tokenizer_id: str = DEFAULT_TOKENIZER,
  ) -> None:
    sys.path.insert(0, "src")

    from transformers import AutoTokenizer
    from maxtext.configs import pyconfig
    from maxtext.inference.maxengine import maxengine
    from flax import nnx

    print(f"[{BACKEND}] Loading tokenizer from '{tokenizer_id}' …")
    self._tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_id, trust_remote_code=True
    )

    print(f"[{BACKEND}] Initialising MaxText Qwen3-VL-2B model (random weights) …")
    config = pyconfig.initialize([
        "qwen3_vl_demo_jax.py",
        "src/maxtext/configs/post_train/sft.yml",
        "model_name=qwen3-vl-2b",
        "run_name=demo_jax",
        "packing=False",
        "enable_checkpointing=False",
    ])
    engine    = maxengine.MaxEngine(config)
    mesh      = engine.mesh
    from maxtext.models import models
    transformer = models.Transformer(config, mesh, quant=None, rngs=nnx.Rngs(0))
    graphdef, state = nnx.split(transformer)

    print(f"[{BACKEND}] Restoring checkpoint …")
    ckpt_items = os.path.abspath(os.path.join(checkpoint_dir, "0", "items"))
    state = _restore_checkpoint(state, ckpt_items)

    # Number of visual tokens produced by the vision encoder for _VIT_INPUT_SIZE
    # images: (image_size / patch_size / spatial_merge)^2 = (448/16/2)^2 = 196.
    self._num_vis_tokens = (
        _VIT_INPUT_SIZE
        // config.patch_size_for_vit
        // config.spatial_merge_size_for_vit
    ) ** 2

    self._config   = config
    self._graphdef = graphdef
    self._state    = state
    self._nnx      = nnx

    # Build JIT-compiled single-step decode function.
    # graphdef is closed over (static structure); bidm is passed as a
    # dynamic array (traced) so the same compiled kernel handles all prompts.
    _graphdef = graphdef

    @jax.jit
    def _decode_step(state_inner, tks, pos, bidm, img_embeds, df0, df1, df2, qpos):
      m = nnx.merge(_graphdef, state_inner)
      logits, _, _ = m.decoder(
          shared_embedding=m.token_embedder,
          decoder_input_tokens=tks,
          decoder_positions=pos,
          bidirectional_mask=bidm,
          image_embeddings=img_embeds,
          deepstack_visual_embeds=[df0, df1, df2],
          deterministic=True,
      )
      return jnp.argmax(logits[0, qpos, :])

    self._decode_step = _decode_step
    print(
        f"[{BACKEND}] Ready.  "
        f"(visual tokens per image: {self._num_vis_tokens}, "
        f"context buffer: {_FIXED_LEN})"
    )

  def run(
      self,
      image_paths: list[str],
      prompt: str = DEFAULT_PROMPT,
      max_new_tokens: int = 512,
      verbose: bool = False,
  ) -> dict:
    """Run end-to-end generation on *image_paths* with *prompt*.

    Args:
      image_paths:    Input image paths.  Only the first image is used.
      prompt:         Text question / instruction.
      max_new_tokens: Maximum autoregressive steps.
      verbose:        Print each token decoded during generation.

    Returns:
      dict with keys ``backend``, ``model``, ``image``, ``prompt``,
      ``response``, ``tokens``, ``elapsed``, ``tok_per_sec``.
    """
    from maxtext.multimodal.processor import get_bidirectional_mask_vision

    image_path = image_paths[0]
    if not os.path.exists(image_path):
      raise FileNotFoundError(f"Image not found: {image_path}")

    # ── 1. vision encoder ──────────────────────────────────────────────────
    if verbose:
      print(f"[{BACKEND}] Running vision encoder on '{image_path}' …")
    pixel_values = jnp.asarray(_preprocess_image(image_path))
    m = self._nnx.merge(self._graphdef, self._state)
    image_embeds, deep_feats = m.vision_encoder(
        input_images=pixel_values, deterministic=True
    )
    df0, df1, df2 = deep_feats
    if verbose:
      print(f"[{BACKEND}]   img_embeds={image_embeds.shape}  df0={df0.shape}")

    # ── 2. tokenise ────────────────────────────────────────────────────────
    input_ids = _build_input_ids(self._tokenizer, prompt, self._num_vis_tokens)
    seq_len   = len(input_ids)
    if verbose:
      print(f"[{BACKEND}] Prompt tokens: {seq_len}  |  buffer: {_FIXED_LEN}")

    # ── 3. mRoPE positions ─────────────────────────────────────────────────
    pos = _compute_mrope_positions(
        self._config, input_ids, _FIXED_LEN, _VIT_INPUT_SIZE
    )

    # ── 4. bidirectional vision mask ───────────────────────────────────────
    tks0 = jnp.zeros((1, _FIXED_LEN), dtype=jnp.int32)
    tks0 = tks0.at[0, :seq_len].set(jnp.array(input_ids, dtype=jnp.int32))
    bidm = get_bidirectional_mask_vision(self._config, tks0)

    # ── 5. autoregressive decode loop ──────────────────────────────────────
    EOS_ID      = self._tokenizer.eos_token_id
    current_ids = list(input_ids)
    generated   = []

    if verbose:
      print(f"[{BACKEND}] Generating (first call will JIT-compile) …")

    t0 = time.time()
    for step in range(max_new_tokens):
      cur_len = len(current_ids)
      if cur_len >= _FIXED_LEN:
        break

      tks   = jnp.zeros((1, _FIXED_LEN), dtype=jnp.int32)
      tks   = tks.at[0, :cur_len].set(jnp.array(current_ids, dtype=jnp.int32))
      qpos  = jnp.int32(cur_len - 1)

      next_tok = int(
          self._decode_step(
              self._state, tks, pos, bidm, image_embeds, df0, df1, df2, qpos
          )
      )
      jax.effects_barrier()
      generated.append(next_tok)
      current_ids.append(next_tok)

      if verbose:
        text_so_far = self._tokenizer.decode(generated, skip_special_tokens=True)
        note = " (JIT compile)" if step == 0 else ""
        print(f"[{BACKEND}]   [{step+1:3d}] tok={next_tok:6d}{note}  >> {text_so_far!r}")
      elif step == 0:
        print(f"[{BACKEND}] JIT compile done, continuing …")

      if next_tok == EOS_ID:
        break

    elapsed   = time.time() - t0
    response  = self._tokenizer.decode(generated, skip_special_tokens=True)
    n_tokens  = len(generated)

    ckpt_label = os.path.basename(os.path.normpath(
        os.path.join(os.path.dirname(list(self._tokenizer.vocab.items())[0][0])
                     if hasattr(self._tokenizer, 'vocab') else "", "..")
    )) if False else "qwen3-vl-2b"

    return {
        "backend":     BACKEND,
        "model":       f"qwen3-vl-2b (JAX/NNX checkpoint)",
        "image":       image_paths,
        "prompt":      prompt,
        "response":    response,
        "tokens":      n_tokens,
        "elapsed":     round(elapsed, 2),
        "tok_per_sec": round(n_tokens / elapsed, 1) if elapsed > 0 else 0.0,
    }


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main() -> None:
  parser = argparse.ArgumentParser(
      description="Qwen3-VL JAX / NNX inference demo (real orbax checkpoint)"
  )
  parser.add_argument(
      "--image", nargs="+", required=True, help="Input image path(s)"
  )
  parser.add_argument("--prompt", default=DEFAULT_PROMPT, help="Text prompt")
  parser.add_argument(
      "--max-tokens", type=int, default=512, help="Max new tokens to generate"
  )
  parser.add_argument(
      "--checkpoint-dir",
      default=DEFAULT_CHECKPOINT,
      help="Path to orbax checkpoint directory",
  )
  parser.add_argument(
      "--tokenizer",
      default=DEFAULT_TOKENIZER,
      help="HuggingFace tokenizer ID or local tokenizer path",
  )
  parser.add_argument(
      "--output-json", action="store_true", help="Print result as JSON"
  )
  parser.add_argument(
      "--verbose",
      action="store_true",
      help="Print each token as it is generated",
  )
  args = parser.parse_args()

  demo   = Qwen3VLDemoJAX(
      checkpoint_dir=args.checkpoint_dir, tokenizer_id=args.tokenizer
  )
  result = demo.run(
      image_paths=args.image,
      prompt=args.prompt,
      max_new_tokens=args.max_tokens,
      verbose=args.verbose,
  )
  _print_result(result, args.output_json)


if __name__ == "__main__":
  main()
