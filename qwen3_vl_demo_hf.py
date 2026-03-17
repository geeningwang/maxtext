#!/usr/bin/env python3
"""Qwen3-VL inference demo — HuggingFace / PyTorch backend.

Runs Qwen3-VL via the HuggingFace ``transformers`` library on CPU or GPU.
This is the reference implementation; output should match the canonical
HuggingFace behaviour exactly.

Usage::

    python qwen3_vl_demo_hf.py \\
        --image tests/assets/image1.jpg \\
        --prompt "Describe what you see in the image."
"""

import argparse
import json
import os
import time
from typing import Optional

import numpy as np
from PIL import Image
import torch
from transformers import AutoProcessor, AutoModelForImageTextToText

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

BACKEND = "hf"
DEFAULT_MODEL = "Qwen/Qwen3-VL-2B-Instruct"
DEFAULT_PROMPT = "Describe what you see in the image."


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
# Inference pipeline class
# ---------------------------------------------------------------------------

class Qwen3VLDemoHF:
  """HuggingFace / PyTorch inference pipeline for Qwen3-VL."""

  def __init__(self, model_id: str = DEFAULT_MODEL) -> None:
    """Load the processor and model from *model_id*.

    Args:
      model_id: HuggingFace model ID (e.g. ``"Qwen/Qwen3-VL-2B-Instruct"``)
                or path to a local model directory.
    """
    print(f"[{BACKEND}] Loading processor and model from '{model_id}' …")
    self.model_id = model_id
    self.processor = AutoProcessor.from_pretrained(
        model_id, trust_remote_code=True
    )
    self.model = AutoModelForImageTextToText.from_pretrained(
        model_id,
        torch_dtype=torch.float32,
        trust_remote_code=True,
        device_map="auto",
    )
    self.device = next(self.model.parameters()).device
    print(f"[{BACKEND}] Ready.  (device: {self.device})")

  def run(
      self,
      image_paths: list[str],
      prompt: str = DEFAULT_PROMPT,
      max_new_tokens: int = 512,
      verbose: bool = False,
  ) -> dict:
    """Run end-to-end inference on *image_paths* with *prompt*.

    Args:
      image_paths:    Input image file paths (all are passed to the model).
      prompt:         Text question / instruction.
      max_new_tokens: Maximum number of new tokens to generate.
      verbose:        Print extra loading / processing information.

    Returns:
      dict with keys ``backend``, ``model``, ``image``, ``prompt``,
      ``response``, ``tokens``, ``elapsed``, ``tok_per_sec``.
    """
    # Load images
    images = []
    for path in image_paths:
      if not os.path.exists(path):
        raise FileNotFoundError(f"Image not found: {path}")
      img = Image.open(path).convert("RGB")
      images.append(img)
      if verbose:
        print(f"[{BACKEND}]   loaded {path}  size={img.size}")

    # Build chat message in Qwen3-VL format
    content = [{"type": "image", "image": img} for img in images]
    content.append({"type": "text", "text": prompt})
    messages = [{"role": "user", "content": content}]

    text = self.processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = self.processor(
        text=[text], images=images, padding=True, return_tensors="pt"
    ).to(self.device)

    if verbose:
      print(f"[{BACKEND}]   input tokens: {inputs['input_ids'].shape[1]}")

    # Generate (greedy / deterministic for reproducible comparison)
    t0 = time.time()
    with torch.no_grad():
      output_ids = self.model.generate(
          **inputs,
          max_new_tokens=max_new_tokens,
          do_sample=False,
      )
    elapsed = time.time() - t0

    prompt_len = inputs["input_ids"].shape[1]
    new_ids = output_ids[0][prompt_len:]
    response = self.processor.decode(new_ids, skip_special_tokens=True)
    n_tokens = int(new_ids.shape[0])

    return {
        "backend": BACKEND,
        "model": self.model_id,
        "image": image_paths,
        "prompt": prompt,
        "response": response,
        "tokens": n_tokens,
        "elapsed": round(elapsed, 2),
        "tok_per_sec": round(n_tokens / elapsed, 1) if elapsed > 0 else 0.0,
    }


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main() -> None:
  parser = argparse.ArgumentParser(
      description="Qwen3-VL HuggingFace / PyTorch inference demo"
  )
  parser.add_argument(
      "--image", nargs="+", required=True, help="Input image path(s)"
  )
  parser.add_argument(
      "--prompt", default=DEFAULT_PROMPT, help="Text prompt"
  )
  parser.add_argument(
      "--max-tokens", type=int, default=512, help="Max new tokens to generate"
  )
  parser.add_argument(
      "--model",
      default=DEFAULT_MODEL,
      help="HuggingFace model ID or local model directory path",
  )
  parser.add_argument(
      "--output-json", action="store_true", help="Print result as JSON"
  )
  parser.add_argument(
      "--verbose", action="store_true", help="Extra loading / token logging"
  )
  args = parser.parse_args()

  demo = Qwen3VLDemoHF(model_id=args.model)
  result = demo.run(
      image_paths=args.image,
      prompt=args.prompt,
      max_new_tokens=args.max_tokens,
      verbose=args.verbose,
  )
  _print_result(result, args.output_json)


if __name__ == "__main__":
  main()
