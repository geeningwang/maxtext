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
DEFAULT_PROMPT     = "Describe what you see in the image."

_VIT_INPUT_SIZE = 448   # images are resized to this spatial resolution
_MAX_PREFILL    = 512   # max_prefill_predict_length; must cover the full prompt
_MAX_TARGET     = 1024  # max_target_length  (prefill + decode steps)


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
# Image / tokenisation helpers  (shared with qwen3_vl_demo_jax.py)
# ---------------------------------------------------------------------------

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
        f"(visual tokens per image: {self._num_vis_tokens}, "
        f"prefill length: {_MAX_PREFILL}, "
        f"max target: {_MAX_TARGET})"
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
      verbose:        Print progress information.

    Returns:
      dict with keys ``backend``, ``model``, ``image``, ``prompt``,
      ``response``, ``tokens``, ``elapsed``, ``tok_per_sec``.
    """
    from maxtext.multimodal.processor_qwen3_omni import get_rope_index

    image_path = image_paths[0]
    if not os.path.exists(image_path):
      raise FileNotFoundError(f"Image not found: {image_path}")

    # ── 1. Tokenise prompt with visual placeholders ─────────────────────────
    input_ids = _build_input_ids(self._tokenizer, prompt, self._num_vis_tokens)
    seq_len   = len(input_ids)
    assert seq_len <= _MAX_PREFILL, (
        f"Prompt length {seq_len} exceeds max_prefill_predict_length {_MAX_PREFILL}. "
        "Increase _MAX_PREFILL or shorten the prompt."
    )

    # Pad to exactly _MAX_PREFILL for MaxEngine.
    # MaxEngine's _prefill_jit calls jnp.expand_dims(padded_tokens, 0) internally,
    # so we must pass a 1D array; do NOT add a batch dimension here.
    padded = np.zeros(_MAX_PREFILL, dtype=np.int32)
    padded[:seq_len] = input_ids
    padded_tokens = jnp.asarray(padded)  # (MAX_PREFILL,)

    if verbose:
      print(f"[{BACKEND}] Prompt tokens: {seq_len}  (padded to {_MAX_PREFILL})")

    # ── 2. Compute mRoPE position IDs ───────────────────────────────────────
    merge   = self._config.spatial_merge_size_for_vit   # 2
    patch   = self._config.patch_size_for_vit            # 16
    grid_h  = _VIT_INPUT_SIZE // patch                   # 28
    grid_w  = _VIT_INPUT_SIZE // patch                   # 28
    image_grid_thw = np.array([[1, grid_h, grid_w]], dtype=np.int32)  # (1, 3)
    attn_mask      = np.zeros((1, _MAX_PREFILL), dtype=np.int32)
    attn_mask[0, :seq_len] = 1  # 1 = real token, 0 = padding

    position_ids, mrope_deltas = get_rope_index(
        input_ids=padded.reshape(1, -1).astype(np.int32),
        image_grid_thw=image_grid_thw,
        attention_mask=attn_mask,
        spatial_merge_size=merge,
    )  # (3, 1, MAX_PREFILL), (1, 1)
    # MaxEngine's _prefill_jit computes next_pos as int32 + mrope_deltas; cast
    # mrope_deltas to int32 to avoid a dtype mismatch in the insert step.
    mrope_deltas = mrope_deltas.astype(np.int32)

    if verbose:
      print(f"[{BACKEND}] mRoPE positions computed: {position_ids.shape}")

    # ── 3. Preprocess image for the vision encoder ───────────────────────────
    import types
    from maxtext.multimodal.processor import preprocess_mm_data
    pixel_values = jnp.asarray(preprocess_mm_data(
        types.SimpleNamespace(model_name=self._config.model_name, image_path=image_path)
    ).pixel_values)  # (1,3,2,H,W)
    if verbose:
      print(f"[{BACKEND}] Pixel values shape: {pixel_values.shape}")

    # ── 4. Prefill ───────────────────────────────────────────────────────────
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

    # ── 5. Insert prefill into decode state ─────────────────────────────────
    decode_state = self._engine.insert(
        prefill_result, self._decode_state_init, slot=0
    )

    # ── 6. Decode loop ───────────────────────────────────────────────────────
    EOS_ID  = self._tokenizer.eos_token_id

    # First token comes from prefill
    first_tok = first_token.get_result_at_slot(0).tokens.item()
    generated = [first_tok]
    if verbose:
      print(f"[{BACKEND}] First token: {first_tok}")

    gen_start = time.time()
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
      "--verbose", action="store_true", help="Print progress information"
  )
  args = parser.parse_args()

  demo   = Qwen3VLDemoEngine(
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
