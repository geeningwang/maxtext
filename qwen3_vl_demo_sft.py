#!/usr/bin/env python3
"""Qwen3-VL SFT overfit demo — MaxEngine inference + manual SFT training.

Demonstrates that the SFT fine-tuning pipeline actually updates model weights by:

  1. **BEFORE**  — Load model via MaxEngine and run inference on a question about
                   an image.  The model gives its real (correct) answer.
  2. **TRAIN**   — Fine-tune the text tower for a fixed number of steps on a single
                   deliberately *wrong* QA pair involving the same image.  The wrong
                   answer is hardcoded so the outcome is deterministic.
  3. **AFTER**   — Re-run MaxEngine inference with the updated params.  Because the
                   training example was repeated many times, the model now produces
                   the wrong answer, proving that the SFT pipeline changed the weights.

Architecture notes
------------------
* **Inference** uses the MaxEngine API (``prefill`` / ``insert`` / ``generate``)
  exactly as in ``qwen3_vl_demo_engine.py``.
* **Training** uses the Linen model that lives inside ``engine.model`` with a
  plain ``jax.value_and_grad`` loop + ``optax.adam``.  No Tunix / PeftTrainer is
  needed for a single-example overfit demo.
* Params flow::

    engine.load_params(rng)   →  params (Linen PyTree)
    training loop             →  updated_params
    engine.prefill(params=updated_params, ...)  →  AFTER response

Usage::

    python qwen3_vl_demo_sft.py \\
        --image tests/assets/test_image.jpg \\
        --steps 200

    # To also see verbose token-by-token decode output:
    python qwen3_vl_demo_sft.py --image tests/assets/test_image.jpg --verbose
"""

import argparse
import json
import os
import sys
import time

import jax
import jax.numpy as jnp
import numpy as np
from PIL import Image

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

BACKEND = "sft"
DEFAULT_CHECKPOINT = "tests/assets/qwen3_vl_2b_orbax"
DEFAULT_TOKENIZER  = "Qwen/Qwen3-VL-2B-Instruct"
DEFAULT_PROMPT     = (
    "There are two images and a video clip provided. "
    "Describe what you see in each image and summarize the main scene in the video."
)
DEFAULT_VIDEO      = "tests/assets/video.mp4"

_VIT_INPUT_SIZE = 448   # images are resized to this spatial resolution
_MAX_PREFILL    = 1024  # max_prefill_predict_length (4 visuals × 196 tokens + prompt)
_MAX_TARGET     = 1536  # max_target_length (prefill + decode buffer)
_MAX_TRAIN_LEN  = 512   # sequence length used during training steps
_N_VIDEO_FRAMES = 2     # frames to sample from the input video

# SFT-specific: single-image training question used to prove weight updates.
DEMO_QUESTION = "What is the dominant color in this image?"

# The deliberately *wrong* answer we fine-tune the model to produce.
# After training on this for enough steps the model should repeat it verbatim.
WRONG_ANSWER      = "The dominant color is definitely magenta."

# Pad / unknown token sentinel — used for prompt masking during SFT
_PAD_ID           = 0


# ---------------------------------------------------------------------------
# Video / image sampling helper
# ---------------------------------------------------------------------------

def _sample_video_frames(video_path: str, n_frames: int) -> list:
  """Return ``n_frames`` uniformly-sampled frames from *video_path*.

  Supports GIF/APNG (via PIL) and MP4/AVI/MOV (via cv2).
  Each returned element is a ``(H, W, 3)`` uint8 ``np.ndarray``.
  """
  path_lower = video_path.lower()
  if path_lower.endswith(".gif") or path_lower.endswith(".apng"):
    from PIL import Image as _PILImage  # pylint: disable=import-outside-toplevel
    gif = _PILImage.open(video_path)
    frames = []
    try:
      while True:
        frames.append(np.array(gif.convert("RGB")))
        gif.seek(gif.tell() + 1)
    except EOFError:
      pass
  else:
    import cv2 as _cv2  # pylint: disable=import-error,import-outside-toplevel
    cap = _cv2.VideoCapture(video_path)
    total   = int(cap.get(_cv2.CAP_PROP_FRAME_COUNT))
    indices = set(np.linspace(0, max(0, total - 1), n_frames, dtype=int).tolist())
    frames  = []
    idx     = 0
    while True:
      ok, frame = cap.read()
      if not ok:
        break
      if idx in indices:
        frames.append(_cv2.cvtColor(frame, _cv2.COLOR_BGR2RGB))
      idx += 1
    cap.release()

  if not frames:
    raise ValueError(f"No frames extracted from {video_path!r}")
  indices2 = np.linspace(0, len(frames) - 1, n_frames, dtype=int)
  return [frames[i] for i in indices2]


# ---------------------------------------------------------------------------
# Result formatting  (shared interface with all other demo scripts)
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
# Tokenisation helpers for MaxEngine inference
# ---------------------------------------------------------------------------

def _build_input_ids_for_engine(tokenizer, prompt: str, vis_token_counts: list) -> list:
  """Return prompt token IDs with N image-placeholder sections for MaxEngine prefill."""
  vision_sections = "".join(
      "<|vision_start|>" + "<|image_pad|>" * n + "<|vision_end|>"
      for n in vis_token_counts
  )
  messages = [
      {"role": "system", "content": "You are a helpful assistant."},
      {"role": "user",   "content": vision_sections + prompt},
  ]
  text = tokenizer.apply_chat_template(
      messages, tokenize=False, add_generation_prompt=True
  )
  return tokenizer.encode(text, add_special_tokens=False)


# ---------------------------------------------------------------------------
# MaxEngine wrapper — thin helper around prefill/insert/generate
# ---------------------------------------------------------------------------

class _EngineRunner:
  """Thin wrapper that holds the engine + params and exposes ``run()``."""

  def __init__(self, engine, params, config, tokenizer):
    self.engine    = engine
    self.params    = params          # mutable — replaced after training
    self.config    = config
    self.tokenizer = tokenizer
    # NOTE: decode state is intentionally NOT pre-allocated here.
    # The engine.insert() JIT-compiled function uses donate_argnums on its
    # decode_state argument, which would consume a cached reference.  Instead
    # we allocate a fresh decode state inside each run() call.

  # ------------------------------------------------------------------

  def run(
      self,
      image_paths: list,
      video_path: str = "",
      prompt: str = DEFAULT_PROMPT,
      max_new_tokens: int = 512,
      verbose: bool = False,
  ) -> dict:
    """Run MaxEngine prefill + decode and return a result dict."""
    from maxtext.multimodal.processor_qwen3_omni import get_rope_index
    from maxtext.multimodal.processor_qwen3_vl import preprocess_image_qwen3_vl

    # 0. Collect all visual frames ────────────────────────────────────────
    all_frames_np = []
    for img_path in image_paths:
      img = Image.open(img_path).convert("RGB")
      all_frames_np.append(np.array(img))
    if video_path:
      all_frames_np.extend(_sample_video_frames(video_path, _N_VIDEO_FRAMES))

    # 1. Preprocess all visual inputs ─────────────────────────────────────
    merge = self.config.spatial_merge_size_for_vit   # 2
    pp = preprocess_image_qwen3_vl(
        all_frames_np, force_size=(_VIT_INPUT_SIZE, _VIT_INPUT_SIZE)
    )
    pixel_values   = pp.pixel_values    # (N, 3, 2, H, W)
    image_grid_thw = pp.image_grid_thw  # (N, 3)
    vis_token_counts = [
        int((row[1] // merge) * (row[2] // merge))
        for row in image_grid_thw
    ]

    # 2. Tokenise ─────────────────────────────────────────────────────────
    input_ids = _build_input_ids_for_engine(
        self.tokenizer, prompt, vis_token_counts
    )
    seq_len = len(input_ids)
    assert seq_len <= _MAX_PREFILL, (
        f"Prompt length {seq_len} exceeds _MAX_PREFILL={_MAX_PREFILL}."
    )

    padded = np.zeros(_MAX_PREFILL, dtype=np.int32)
    padded[:seq_len] = input_ids
    padded_tokens = jnp.asarray(padded)  # (MAX_PREFILL,)

    # 3. mRoPE positions ──────────────────────────────────────────────────
    attn_mask = np.zeros((1, _MAX_PREFILL), dtype=np.int32)
    attn_mask[0, :seq_len] = 1

    position_ids, mrope_deltas = get_rope_index(
        input_ids=padded.reshape(1, -1).astype(np.int32),
        image_grid_thw=image_grid_thw,
        attention_mask=attn_mask,
        spatial_merge_size=merge,
    )
    mrope_deltas = mrope_deltas.astype(np.int32)

    # 4. Prefill ──────────────────────────────────────────────────────────
    rng = jax.random.PRNGKey(42)
    rng, rng_prefill = jax.random.split(rng)
    t_start = time.time()

    prefill_result, first_token = self.engine.prefill(
        params=self.params,
        padded_tokens=padded_tokens,
        positions=position_ids,
        mrope_deltas=mrope_deltas,
        images=jnp.asarray(pixel_values),
        true_length=seq_len,
        rng=rng_prefill,
        slot=0,
    )
    jax.effects_barrier()

    # 5. Insert + decode loop ─────────────────────────────────────────────
    # Allocate a fresh decode state for this request.  We do NOT cache it
    # across calls because engine.insert() donates (consumes) the state.
    decode_state_fresh = self.engine.init_decode_state(jax.random.PRNGKey(99))
    decode_state = self.engine.insert(prefill_result, decode_state_fresh, slot=0)

    EOS_ID    = self.tokenizer.eos_token_id
    first_tok = first_token.get_result_at_slot(0).tokens.item()
    generated = [first_tok]

    for step in range(max_new_tokens - 1):
      rng, rng_gen = jax.random.split(rng)
      decode_state, sampled = self.engine.generate(self.params, decode_state, rng=rng_gen)
      tok = sampled.get_result_at_slot(0).tokens.item()
      generated.append(tok)
      if tok == EOS_ID:
        break

    elapsed  = time.time() - t_start
    n_tokens = len(generated)
    response = self.tokenizer.decode(generated, skip_special_tokens=True)
    if verbose:
      print(f"  [{n_tokens} tokens, {elapsed:.1f}s]")
    return {
        "backend":     BACKEND,
        "model":       "qwen3-vl-2b (MaxEngine/SFT checkpoint)",
        "image":       image_paths,
        "video":       video_path,
        "prompt":      prompt,
        "response":    response,
        "tokens":      n_tokens,
        "elapsed":     round(elapsed, 2),
        "tok_per_sec": round(n_tokens / elapsed, 1) if elapsed > 0 else 0.0,
    }


# ---------------------------------------------------------------------------
# Training batch construction
# ---------------------------------------------------------------------------

def _build_training_batch(
    tokenizer,
    image_path: str,
    question: str,
    wrong_answer: str,
    num_vis_tokens: int,
    max_len: int = _MAX_TRAIN_LEN,
    pad_id: int = _PAD_ID,
) -> dict:
  """Build a single-example SFT batch manually.

  The batch follows the same layout as ``vision_sft_preprocessing_pipeline``:
    - inputs              : (1, max_len) int32  — full sequence incl. prompt + response
    - inputs_position     : (1, max_len) int32  — 0..max_len-1 (expanded to 3-D by mRoPE)
    - inputs_segmentation : (1, max_len) int32  — 1 for real tokens, 0 for padding
    - targets             : (1, max_len) int32  — shift-left of response; pad_id for prompt
    - targets_segmentation: (1, max_len) int32  — 1 only for completion token positions
    - images              : (1, 3, 2, 448, 448) float32

  The sequence layout before ShiftData:
    [prompt_tokens... | response_tokens... | padding...]
  targets before shift:
    [pad_id * prompt_len | response_tokens... | padding...]
  After shift-left by 1:
    targets[i] = was targets[i+1], making targets[prompt_len-1] = response_tokens[0]
  """
  IMAGE_TOKEN    = "<|image_pad|>"
  image_section  = (
      "<|vision_start|>" + IMAGE_TOKEN * num_vis_tokens + "<|vision_end|>"
  )

  # Build the user message.  The image comes before the question.
  user_content = image_section + question

  # Build the system + user message (prompt) using the chat template with
  # add_generation_prompt=True so the assistant header is included in the prompt.
  prompt_messages = [
      {"role": "system", "content": "You are a helpful assistant."},
      {"role": "user",   "content": user_content},
  ]
  prompt_text = tokenizer.apply_chat_template(
      prompt_messages, tokenize=False, add_generation_prompt=True
  )
  prompt_ids = tokenizer.encode(prompt_text, add_special_tokens=False)

  # Build the response.  We add the EOS token manually.
  eos_id = tokenizer.eos_token_id
  im_end_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
  response_text = wrong_answer
  response_ids = tokenizer.encode(response_text, add_special_tokens=False)
  # Close the assistant turn with <|im_end|>\n
  response_ids = response_ids + [im_end_id]

  # ── Build raw sequences (before shift) ───────────────────────────────────
  prompt_len   = len(prompt_ids)
  response_len = len(response_ids)
  total_len    = prompt_len + response_len

  if total_len > max_len:
    # Truncate response to fit (should not happen with the short demo answer)
    response_len = max_len - prompt_len
    response_ids = response_ids[:response_len]
    total_len    = max_len

  raw_inputs      = np.array(prompt_ids + response_ids, dtype=np.int32)
  raw_targets_pre = np.array(
      [pad_id] * prompt_len + response_ids, dtype=np.int32
  )

  # Pad to max_len
  def _pad(x):
    if len(x) < max_len:
      x = np.concatenate([x, np.full(max_len - len(x), pad_id, dtype=np.int32)])
    return x[:max_len]

  raw_inputs      = _pad(raw_inputs)
  raw_targets_pre = _pad(raw_targets_pre)

  # ── inputs_segmentation: 1 for real tokens ───────────────────────────────
  inputs_segmentation = (raw_inputs != pad_id).astype(np.int32)

  # ── targets_segmentation before shift: 1 where prompt is NOT pad ─────────
  targets_segmentation_pre = (raw_targets_pre != pad_id).astype(np.int32)

  # ── Apply ShiftData (shift left by 1) on targets + targets_segmentation ──
  def _shift_left(x, fill=pad_id):
    return np.concatenate([x[1:], np.array([fill], dtype=x.dtype)])

  targets             = _shift_left(raw_targets_pre, fill=pad_id)
  targets_segmentation = _shift_left(targets_segmentation_pre, fill=0)
  # Mask out any remaining pad-id positions in targets
  targets_segmentation = np.where(targets != pad_id, targets_segmentation, 0)

  # ── inputs_position: simple 0..max_len-1 ─────────────────────────────────
  inputs_position = np.arange(max_len, dtype=np.int32)

  # ── Pixel values ─────────────────────────────────────────────────────────
  import types
  from maxtext.multimodal.processor import preprocess_mm_data
  images = preprocess_mm_data(types.SimpleNamespace(model_name="qwen3-vl-2b", image_path=image_path)).pixel_values.astype(np.float32)  # (1, 3, 2, 448, 448)

  # ── Add batch dimension ───────────────────────────────────────────────────
  return {
      "inputs":               raw_inputs[None, :],           # (1, max_len)
      "inputs_position":      inputs_position[None, :],      # (1, max_len)
      "inputs_segmentation":  inputs_segmentation[None, :],  # (1, max_len)
      "targets":              targets[None, :],               # (1, max_len)
      "targets_segmentation": targets_segmentation[None, :], # (1, max_len)
      "images":               images,                        # (1, 3, 2, 448, 448)
  }


# ---------------------------------------------------------------------------
# SFT training step — pure gradient descent (no optimizer state)
# ---------------------------------------------------------------------------
#
# WHY NOT ADAM: The 2B model stores params as bfloat16 (~8.5 GB on-device after
# MaxEngine loading).  Adam requires two float32 moment buffers with the same
# shape → +17 GB.  Combined with gradient memory (~8.5 GB) and activation
# scratch space the device would OOM (only ~25 GB free after params load).
#
# Pure SGD only needs: params + gradients ≈ 2 × 8.5 GB = 17 GB, leaving 8 GB
# for activations.  The model already has RematLocation.REMAT for attention
# layers so activation scratch is well below that budget.

def _make_train_step(engine_model, vocab_size: int, learning_rate: float = 1e-3,
                     max_grad_norm: float = 1.0):
  """Return a JIT-compiled train-step function.

  Uses pure gradient descent with global-norm clipping (no optimizer state)
  to stay within the device HBM budget.  The function signature is::

    loss, new_params = train_step(params, batch)

  where ``batch`` is a dict with keys described in ``_build_training_batch``.
  ``max_grad_norm`` clips the global L2 norm of the parameter gradient before
  the SGD update, preventing loss spikes that would corrupt model quality.
  """
  model = engine_model
  lr    = learning_rate

  def _loss_fn(params, batch):
    """Cross-entropy on completion tokens only."""
    logits, _ = model.apply(
        params,
        batch["inputs"],
        batch["inputs_position"],
        decoder_segment_ids=batch["inputs_segmentation"],
        encoder_images=batch["images"],
        enable_dropout=False,
        rngs={"dropout": jax.random.PRNGKey(0), "params": jax.random.PRNGKey(1)},
        mutable=["intermediates"],
    )
    tgt_seg   = batch["targets_segmentation"]              # (B, L)
    one_hot   = jax.nn.one_hot(batch["targets"], vocab_size)  # (B, L, V)
    log_probs = jax.nn.log_softmax(logits, axis=-1)         # (B, L, V)
    xent      = -jnp.sum(log_probs * one_hot, axis=-1)      # (B, L)
    n_tokens  = jnp.sum(tgt_seg != 0).astype(jnp.float32) + 1e-9
    return jnp.sum(xent * (tgt_seg != 0)) / n_tokens

  @jax.jit
  def train_step(params, batch):
    loss, grads = jax.value_and_grad(_loss_fn)(params, batch)
    # Clip by global L2 norm to prevent loss spikes from corrupting weights.
    leaves = jax.tree_util.tree_leaves(grads)
    global_norm = jnp.sqrt(sum(jnp.sum(g.astype(jnp.float32) ** 2) for g in leaves))
    clip_coeff  = jnp.minimum(1.0, max_grad_norm / (global_norm + 1e-6))
    grads = jax.tree.map(lambda g: (g.astype(jnp.float32) * clip_coeff).astype(g.dtype), grads)
    # In-dtype SGD update.
    new_params = jax.tree.map(
        lambda p, g: p - (lr * g).astype(p.dtype),
        params, grads,
    )
    return loss, new_params

  return train_step


# ---------------------------------------------------------------------------
# Main demo
# ---------------------------------------------------------------------------

def main() -> None:
  parser = argparse.ArgumentParser(
      description="Qwen3-VL SFT overfit demo (MaxEngine inference + manual training)"
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
  parser.add_argument("--prompt", default=DEFAULT_PROMPT, help="Text prompt for inference")
  parser.add_argument("--wrong-answer",  default=WRONG_ANSWER,      help="Deliberately wrong answer to overfit on")
  parser.add_argument("--steps",         type=int, default=300,     help="Number of SFT gradient steps")
  parser.add_argument(
      "--max-tokens", type=int, default=512, help="Max new tokens to generate per inference run"
  )
  parser.add_argument(
      "--checkpoint-dir",
      default=DEFAULT_CHECKPOINT,
      help="Orbax checkpoint directory",
  )
  parser.add_argument(
      "--tokenizer",
      default=DEFAULT_TOKENIZER,
      help="HuggingFace tokenizer ID or local tokenizer path",
  )
  parser.add_argument("--lr",            type=float, default=1e-3,  help="Learning rate for vanilla-SGD training (default 1e-3)")
  parser.add_argument("--max-grad-norm", type=float, default=1.0,   help="Gradient clipping global L2 norm (default 1.0)")
  parser.add_argument("--output-json", action="store_true", help="Print result as JSON")
  parser.add_argument(
      "--verbose", action="store_true", help="Print step-level progress"
  )
  args = parser.parse_args()

  sys.path.insert(0, "src")
  jax.config.update("jax_default_prng_impl", "unsafe_rbg")

  from transformers import AutoTokenizer
  from maxtext.configs import pyconfig
  from maxtext.inference.maxengine import maxengine

  # ── 1. Load tokenizer ─────────────────────────────────────────────────
  print("\n" + "=" * 70)
  print("Qwen3-VL SFT Overfit Demo  [MaxEngine + manual SFT]")
  print("=" * 70)
  print(f"\nImage(s) : {', '.join(args.image)}")
  print(f"Video    : {args.video}")
  print(f"Prompt   : {args.prompt!r}")
  print(f"Wrong Ans: {args.wrong_answer!r}")
  print(f"Steps    : {args.steps}")
  print()

  print("[1/5] Loading tokenizer …")
  try:
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True)
  except Exception:  # pylint: disable=broad-except
    # Fall back to the local qwen3-tokenizer asset if the default path fails
    from maxtext.utils.globals import MAXTEXT_ASSETS_ROOT
    fallback_tok = os.path.join(str(MAXTEXT_ASSETS_ROOT), "tokenizers", "qwen3-tokenizer")
    print(f"   … falling back to {fallback_tok}")
    tokenizer = AutoTokenizer.from_pretrained(fallback_tok, trust_remote_code=True)

  # ── 2. Initialise MaxEngine ───────────────────────────────────────────
  print("[2/5] Initialising MaxEngine …")
  ckpt_items = os.path.abspath(os.path.join(args.checkpoint_dir, "0", "items"))

  config = pyconfig.initialize([
      "qwen3_vl_demo_sft.py",
      "src/maxtext/configs/post_train/sft.yml",
      "model_name=qwen3-vl-2b",
      "run_name=demo_sft",
      "packing=False",
      "enable_checkpointing=True",
      f"load_parameters_path={ckpt_items}",
      "per_device_batch_size=1",
      f"max_prefill_predict_length={_MAX_PREFILL}",
      f"max_target_length={_MAX_TARGET}",
  ])

  engine = maxengine.MaxEngine(config)
  rng    = jax.random.PRNGKey(0)
  print("   … loading params from checkpoint (this may take a minute) …")
  params = engine.load_params(rng)
  print("   … MaxEngine ready.")

  num_vis_tokens = (
      _VIT_INPUT_SIZE
      // config.patch_size_for_vit
      // config.spatial_merge_size_for_vit
  ) ** 2  # 196 for 448px / 16-patch / 2-merge

  runner = _EngineRunner(engine, params, config, tokenizer)

  # ── 3. BEFORE inference ───────────────────────────────────────────────
  image_paths = args.image

  print(f"\n[3/5] BEFORE training — running MaxEngine inference …")
  before_result = runner.run(
      image_paths=image_paths, video_path=args.video,
      prompt=args.prompt,
      max_new_tokens=args.max_tokens, verbose=args.verbose,
  )
  _print_result(before_result, args.output_json)

  # ── 4. SFT training loop ──────────────────────────────────────────────
  print(f"\n[4/5] Building training batch and running {args.steps} SFT steps …")
  batch = _build_training_batch(
      tokenizer,
      args.image[0],          # overfit on the first image only
      DEMO_QUESTION,          # single-image training question (proves weight updates)
      args.wrong_answer,
      num_vis_tokens,
      max_len=_MAX_TRAIN_LEN,
  )
  # Convert to JAX arrays on device
  batch_jax = {k: jnp.asarray(v) for k, v in batch.items()}

  if args.verbose:
    n_completion = int(np.sum(batch["targets_segmentation"]))
    print(f"   Completion tokens to train on: {n_completion}")
    print(f"   Total sequence length        : {_MAX_TRAIN_LEN}")

  # Build the train-step function (pure SGD, no optimizer state).
  # See _make_train_step docstring for why Adam is avoided here.
  train_step = _make_train_step(engine.model, config.vocab_size,
                                 learning_rate=args.lr,
                                 max_grad_norm=args.max_grad_norm)

  print("   JIT-compiling train step (first step will be slow) …")
  t_train  = time.time()
  log_every = max(1, args.steps // 10)

  for step in range(args.steps):
    loss, params = train_step(params, batch_jax)

    if step == 0:
      jax.effects_barrier()
      print(f"   Step 0 JIT compile done.  loss={float(loss):.4f}")

    if args.verbose and (step + 1) % log_every == 0:
      jax.effects_barrier()
      print(f"   Step {step+1:4d}/{args.steps}  loss={float(loss):.4f}")

  jax.effects_barrier()
  elapsed_train = time.time() - t_train
  print(f"   Training done in {elapsed_train:.1f}s  "
        f"(final loss={float(loss):.4f})")

  # ── 5. AFTER inference (with fine-tuned params) ───────────────────────
  print(f"\n[5/5] AFTER training — running MaxEngine inference with fine-tuned params …")
  # Update the runner to use the new params
  runner.params = params
  after_result = runner.run(
      image_paths=image_paths, video_path=args.video,
      prompt=args.prompt,
      max_new_tokens=args.max_tokens, verbose=args.verbose,
  )
  _print_result(after_result, args.output_json)

  # ── SFT comparison summary ────────────────────────────────────────────
  W = 70
  before_response = before_result["response"]
  after_response  = after_result["response"]
  print("\n" + "=" * W)
  print("SFT Overfit Summary")
  print("=" * W)
  print(f"Image(s) : {', '.join(args.image)}")
  print(f"Video    : {args.video}")
  print(f"Prompt   : {args.prompt!r}")
  print("-" * W)
  print(f"BEFORE   : {before_response!r}")
  print(f"TARGET   : {args.wrong_answer!r}  (the wrong answer we fine-tuned on)")
  print(f"AFTER    : {after_response!r}")
  print("-" * W)
  if args.wrong_answer.lower().split()[-1].rstrip(".") in after_response.lower():
    print("✓ Overfit succeeded — model now produces the wrong answer!")
  else:
    print("⚠ Overfit not yet complete — try increasing --steps.")
  print("=" * W + "\n")


if __name__ == "__main__":
  main()
