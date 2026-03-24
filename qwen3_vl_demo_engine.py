#!/usr/bin/env python3
"""Qwen3-VL inference demo — MaxEngine serving API backend.

Uses the MaxText MaxEngine API (``prefill`` / ``insert`` / ``generate``) as
the inference backend.  Unlike ``qwen3_vl_demo_jax.py``, which manages
the NNX model and KV-cache manually, this script delegates all of that to
MaxEngine's production serving layer.

The MaxEngine backend is the recommended path for deployment, as it handles
KV-cache slot allocation, JIT compilation, and multi-stream batching.

Usage::

    python qwen3_vl_demo_engine.py \\
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
import numpy as np
from PIL import Image

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

BACKEND = "engine"
DEFAULT_CHECKPOINT = "tests/assets/qwen3_vl_2b_orbax"
DEFAULT_TOKENIZER  = "Qwen/Qwen3-VL-2B-Instruct"
DEFAULT_PROMPT     = (
    "There are two images and a video clip provided. "
    "Describe what you see in each image and summarize the main scene in the video."
)
DEFAULT_VIDEO   = "tests/assets/video.mp4"
_N_VIDEO_FRAMES = 2    # frames uniformly sampled from the video clip

_VIT_INPUT_SIZE = 448   # all visuals are resized to this spatial resolution
_MAX_PREFILL    = 1024  # max_prefill_predict_length; covers 4×196 visual + prompt
_MAX_TARGET     = 1536  # max_target_length  (prefill + decode steps)


# ---------------------------------------------------------------------------
# Video-frame sampling helper
# ---------------------------------------------------------------------------

def _sample_video_frames(video_path: str, n_frames: int = _N_VIDEO_FRAMES) -> list:
  """Uniformly sample *n_frames* from a video file.

  Supports GIF / APNG (via PIL) and MP4 / AVI / MOV (via cv2).

  Returns:
    list of (H, W, 3) uint8 ``np.ndarray`` frames.
  """
  frames: list = []

  # PIL handles GIF / APNG / multi-page TIFF.
  try:
    from PIL import ImageSequence  # pylint: disable=import-outside-toplevel
    pil = Image.open(video_path)
    frames = [np.array(f.convert("RGB")) for f in ImageSequence.Iterator(pil)]
    if len(frames) <= 1:
      frames = []
  except Exception:  # pylint: disable=broad-except
    frames = []

  # cv2 for real video containers (MP4 / AVI / MOV / …).
  if not frames:
    try:
      import cv2  # pylint: disable=import-outside-toplevel
      cap = cv2.VideoCapture(video_path)
      while True:
        ret, frame = cap.read()
        if not ret:
          break
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
      cap.release()
    except ImportError as exc:
      raise RuntimeError(
          f"Cannot decode '{video_path}': install opencv-python-headless\n"
          "  pip install opencv-python-headless"
      ) from exc

  if not frames:
    raise RuntimeError(f"No frames decoded from: {video_path}")
  if len(frames) == 1:
    frames = frames * n_frames  # replicate single frame

  indices = np.linspace(0, len(frames) - 1, n_frames).round().astype(int)
  return [frames[i] for i in indices]


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
  if result.get("video"):
    print(f"Video    : {result['video']}")
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
# Image / tokenisation helpers  (shared with qwen3_vl_demo_jax.py)
# ---------------------------------------------------------------------------

def _build_input_ids(tokenizer, prompt: str, vis_token_counts: list) -> list:
  """Return the full prompt token IDs with visual placeholders embedded.

  Builds one ``<|vision_start|><|image_pad|>×N<|vision_end|>`` section for
  each entry in *vis_token_counts*.  The sections are prepended to *prompt*
  inside the Qwen3-VL chat template.

  Args:
    tokenizer:        Qwen3-VL tokenizer.
    prompt:           User text question / instruction.
    vis_token_counts: List of visual token counts, one per visual section
                      (image or video frame).  E.g. ``[196, 196, 196, 196]``
                      for 2 images + 2 video frames at 448×448.
  """
  IMAGE_TOKEN = "<|image_pad|>"
  sections = "".join(
      "<|vision_start|>" + IMAGE_TOKEN * n + "<|vision_end|>"
      for n in vis_token_counts
  )
  messages = [
      {"role": "system", "content": "You are a helpful assistant."},
      {"role": "user",   "content": sections + prompt},
  ]
  text = tokenizer.apply_chat_template(
      messages, tokenize=False, add_generation_prompt=True
  )
  return tokenizer.encode(text, add_special_tokens=False)


# ---------------------------------------------------------------------------
# MaxEngine inference class
# ---------------------------------------------------------------------------

class Qwen3VLDemoEngine:
  """MaxEngine inference pipeline for Qwen3-VL.

  Uses MaxEngine's ``load_params`` / ``prefill`` / ``insert`` / ``generate``
  API.  The model is initialised on construction; inference happens in
  :meth:`run`.
  """

  def __init__(
      self,
      checkpoint_dir: str = DEFAULT_CHECKPOINT,
      tokenizer_id: str = DEFAULT_TOKENIZER,
  ) -> None:
    sys.path.insert(0, "src")
    jax.config.update("jax_default_prng_impl", "unsafe_rbg")

    from transformers import AutoTokenizer
    from maxtext.configs import pyconfig
    from maxtext.inference.maxengine import maxengine

    print(f"[{BACKEND}] Loading tokenizer from '{tokenizer_id}' …")
    self._tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_id, trust_remote_code=True
    )

    # Build the absolute path to the items directory (same as the JAX demo).
    ckpt_items = os.path.abspath(os.path.join(checkpoint_dir, "0", "items"))
    print(f"[{BACKEND}] Loading MaxEngine …  (checkpoint: {ckpt_items})")

    config = pyconfig.initialize([
        "qwen3_vl_demo_engine.py",
        "src/maxtext/configs/post_train/sft.yml",
        "model_name=qwen3-vl-2b",
        "run_name=demo_engine",
        "packing=False",
        "enable_checkpointing=True",
        f"load_parameters_path={ckpt_items}",
        "per_device_batch_size=1",
        f"max_prefill_predict_length={_MAX_PREFILL}",
        f"max_target_length={_MAX_TARGET}",
    ])

    self._engine = maxengine.MaxEngine(config)
    rng = jax.random.PRNGKey(0)
    print(f"[{BACKEND}] Loading parameters from checkpoint …")
    self._params = self._engine.load_params(rng)

    # Initialise the decode state once and reuse between run() calls.
    rng_decode = jax.random.PRNGKey(1)
    print(f"[{BACKEND}] Initialising decode state …")
    self._decode_state_init = self._engine.init_decode_state(rng_decode)

    self._config = config
    self._num_vis_tokens = (
        _VIT_INPUT_SIZE
        // config.patch_size_for_vit
        // config.spatial_merge_size_for_vit
    ) ** 2  # = 196 for 448-pixel / 16-patch / 2-merge

    print(
        f"[{BACKEND}] Ready.  "
        f"(visual tokens per entry: {self._num_vis_tokens}, "
        f"prefill length: {_MAX_PREFILL}, "
        f"max target: {_MAX_TARGET})"
    )

  def run(
      self,
      image_paths: list[str],
      video_path: str = "",
      prompt: str = DEFAULT_PROMPT,
      max_new_tokens: int = 512,
      verbose: bool = False,
  ) -> dict:
    """Run end-to-end generation on *image_paths* and optional *video_path*.

    Pass two images and a video to demonstrate the full multimodal capability:
    ``image_paths=["img1.jpg", "img2.jpg"]`` and ``video_path="clip.mp4"``.
    Video frames are sampled uniformly and processed through the same ViT
    backbone as images, giving the model full spatial understanding of each
    frame.

    Args:
      image_paths:    Input image paths (typically 2 for the full demo).
      video_path:     Optional path to a video file (MP4, AVI, GIF, …).
                      ``_N_VIDEO_FRAMES`` frames are sampled uniformly.
      prompt:         Text question / instruction.
      max_new_tokens: Maximum autoregressive steps.
      verbose:        Print progress information.

    Returns:
      dict with keys ``backend``, ``model``, ``image``, ``video``,
      ``prompt``, ``response``, ``tokens``, ``elapsed``, ``tok_per_sec``.
    """
    from maxtext.multimodal.processor_qwen3_omni import get_rope_index
    from maxtext.multimodal.processor_qwen3_vl import preprocess_image_qwen3_vl

    # ── 1. Load all visual inputs ───────────────────────────────────────────
    all_frames_np: list = []
    for p in image_paths:
      if not os.path.exists(p):
        raise FileNotFoundError(f"Image not found: {p}")
      all_frames_np.append(np.array(Image.open(p).convert("RGB")))

    n_images = len(all_frames_np)

    if video_path:
      if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video not found: {video_path}")
      video_frames = _sample_video_frames(video_path, n_frames=_N_VIDEO_FRAMES)
      all_frames_np.extend(video_frames)

    n_total = len(all_frames_np)

    # Preprocess all visuals at a fixed 448×448 so every entry produces the
    # same (1, 28, 28) grid and all pixel tensors can be stacked into one array.
    mm_out = preprocess_image_qwen3_vl(
        all_frames_np, force_size=(_VIT_INPUT_SIZE, _VIT_INPUT_SIZE)
    )
    pixel_values  = jnp.asarray(mm_out.pixel_values)  # (N, 3, 2, 448, 448)
    image_grid_thw = mm_out.image_grid_thw              # (N, 3)

    if verbose:
      extra = f" + {n_total - n_images} video frame(s)" if video_path else ""
      print(f"[{BACKEND}] Visual inputs: {n_images} image(s){extra}  "
            f"pixel_values: {pixel_values.shape}")

    # ── 2. Compute per-section visual token counts ──────────────────────────
    merge = self._config.spatial_merge_size_for_vit  # 2
    vis_token_counts = [
        int(g[0] * g[1] * g[2]) // (merge ** 2)
        for g in image_grid_thw
    ]  # [196, 196, …] for entries at 448×448

    # ── 3. Tokenise prompt with all visual sections ─────────────────────────
    input_ids = _build_input_ids(self._tokenizer, prompt, vis_token_counts)
    seq_len   = len(input_ids)
    assert seq_len <= _MAX_PREFILL, (
        f"Prompt length {seq_len} exceeds max_prefill_predict_length {_MAX_PREFILL}. "
        "Increase _MAX_PREFILL or reduce the number of visual inputs."
    )

    padded = np.zeros(_MAX_PREFILL, dtype=np.int32)
    padded[:seq_len] = input_ids
    padded_tokens = jnp.asarray(padded)

    if verbose:
      print(f"[{BACKEND}] Prompt tokens: {seq_len}  (padded to {_MAX_PREFILL})")

    # ── 4. Compute mRoPE position IDs ───────────────────────────────────────
    attn_mask = np.zeros((1, _MAX_PREFILL), dtype=np.int32)
    attn_mask[0, :seq_len] = 1

    position_ids, mrope_deltas = get_rope_index(
        input_ids=padded.reshape(1, -1).astype(np.int32),
        image_grid_thw=image_grid_thw,
        attention_mask=attn_mask,
        spatial_merge_size=merge,
    )
    mrope_deltas = mrope_deltas.astype(np.int32)

    if verbose:
      print(f"[{BACKEND}] mRoPE positions computed: {position_ids.shape}")

    # ── 5. Prefill ───────────────────────────────────────────────────────────
    rng = jax.random.PRNGKey(42)
    rng, rng_prefill = jax.random.split(rng)
    if verbose:
      print(f"[{BACKEND}] Running prefill (will JIT-compile on first call) …")

    t0 = time.time()
    prefill_result, first_token = self._engine.prefill(
        params=self._params,
        padded_tokens=padded_tokens,
        positions=position_ids,
        mrope_deltas=mrope_deltas,
        images=pixel_values,
        true_length=seq_len,
        rng=rng_prefill,
        slot=0,
    )
    jax.effects_barrier()
    if verbose:
      print(f"[{BACKEND}] Prefill done in {time.time()-t0:.1f}s")

    # ── 6. Insert prefill into decode state ─────────────────────────────────
    decode_state = self._engine.insert(
        prefill_result, self._decode_state_init, slot=0
    )

    # ── 7. Decode loop ───────────────────────────────────────────────────────
    EOS_ID    = self._tokenizer.eos_token_id
    first_tok = first_token.get_result_at_slot(0).tokens.item()
    generated = [first_tok]
    if verbose:
      print(f"[{BACKEND}] First token: {first_tok}")

    for step in range(max_new_tokens - 1):
      rng, rng_gen = jax.random.split(rng)
      decode_state, sampled = self._engine.generate(
          self._params, decode_state, rng=rng_gen
      )
      tok = sampled.get_result_at_slot(0).tokens.item()
      generated.append(tok)

      if verbose:
        text_so_far = self._tokenizer.decode(generated, skip_special_tokens=True)
        print(f"[{BACKEND}]   [{step+2:3d}] tok={tok:6d}  >> {text_so_far!r}")

      if tok == EOS_ID:
        break

    elapsed  = time.time() - t0
    response = self._tokenizer.decode(generated, skip_special_tokens=True)
    n_tokens = len(generated)

    return {
        "backend":     BACKEND,
        "model":       "qwen3-vl-2b (MaxEngine checkpoint)",
        "image":       image_paths,
        "video":       video_path,
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
      description="Qwen3-VL MaxEngine serving-API inference demo"
  )
  parser.add_argument(
      "--image", nargs="+", required=True,
      help="Input image paths — provide 2 for the full demo, e.g. "
           "--image tests/assets/image1.jpg tests/assets/image2.jpg"
  )
  parser.add_argument(
      "--video", default="", metavar="PATH",
      help=f"Optional video file (MP4 / AVI / GIF).  "
           f"{_N_VIDEO_FRAMES} frames are sampled uniformly.  "
           f"Example: --video {DEFAULT_VIDEO}"
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
      "--verbose", action="store_true", help="Print progress information"
  )
  args = parser.parse_args()

  demo   = Qwen3VLDemoEngine(
      checkpoint_dir=args.checkpoint_dir, tokenizer_id=args.tokenizer
  )
  result = demo.run(
      image_paths=args.image,
      video_path=args.video,
      prompt=args.prompt,
      max_new_tokens=args.max_tokens,
      verbose=args.verbose,
  )
  _print_result(result, args.output_json)


if __name__ == "__main__":
  main()
