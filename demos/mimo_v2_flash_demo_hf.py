#!/usr/bin/env python3
# Copyright 2026 Google LLC
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

r"""HuggingFace inference demo for MiMo-V2-Flash.

Runs MiMo-V2-Flash via the HuggingFace Transformers library using the
bundled PyTorch implementation.  Useful for:
  • Quickly verifying the model loads and generates sensible text.
  • A reference baseline before converting to MaxText.

Requirements:
  pip install transformers torch accelerate

Usage:
  python3 demos/mimo_v2_flash_demo_hf.py \
      --model_path XiaomiMiMo/MiMo-V2-Flash \
      --prompt "The key to solving any hard problem is" \
      --max_new_tokens 64

Note: With the full 309B model you will need multiple high-memory GPUs.
For local testing on a single GPU, add --load_in_4bit to enable BnB
4-bit quantisation (requires bitsandbytes >= 0.43).
"""

import argparse
import sys
import time


def _check_imports():
    missing = []
    try:
        import torch  # noqa: F401
    except ImportError:
        missing.append("torch")
    try:
        import transformers  # noqa: F401
    except ImportError:
        missing.append("transformers")
    if missing:
        print(
            f"ERROR: Missing required packages: {', '.join(missing)}\n"
            f"Install them with:  pip install {' '.join(missing)} accelerate",
            file=sys.stderr,
        )
        sys.exit(1)


def load_model(model_path: str, dtype: str = "bfloat16", load_in_4bit: bool = False):
    """Load tokeniser and model from a local directory or the Hub."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"Loading tokeniser from: {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    torch_dtype = getattr(torch, dtype)
    load_kwargs = {
        "device_map": "auto",
        "trust_remote_code": True,
        "torch_dtype": torch_dtype,
    }
    if load_in_4bit:
        from transformers import BitsAndBytesConfig  # pylint: disable=import-outside-toplevel
        load_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True)
        print("Loading in 4-bit quantisation mode (bitsandbytes)")

    print(f"Loading model from: {model_path}  (dtype={dtype})")
    t0 = time.perf_counter()
    model = AutoModelForCausalLM.from_pretrained(model_path, **load_kwargs)
    elapsed = time.perf_counter() - t0
    print(f"Model loaded in {elapsed:.1f}s")
    return tokenizer, model


def generate(
    tokenizer,
    model,
    prompt: str,
    max_new_tokens: int = 128,
    temperature: float = 0.0,
    top_p: float = 1.0,
) -> str:
    """Run greedy / sampled generation on a single prompt."""
    import torch

    inputs = tokenizer(prompt, return_tensors="pt")
    device = next(model.parameters()).device
    input_ids = inputs["input_ids"].to(device)

    gen_kwargs = {
        "max_new_tokens": max_new_tokens,
        "do_sample": temperature > 0.0,
    }
    if temperature > 0.0:
        gen_kwargs["temperature"] = temperature
        gen_kwargs["top_p"] = top_p

    t0 = time.perf_counter()
    with torch.inference_mode():
        output_ids = model.generate(input_ids, **gen_kwargs)
    elapsed = time.perf_counter() - t0

    generated = output_ids[0, input_ids.shape[-1]:]
    text = tokenizer.decode(generated, skip_special_tokens=True)
    n_tokens = len(generated)
    print(f"\nGenerated {n_tokens} tokens in {elapsed:.2f}s  "
          f"({n_tokens / elapsed:.1f} tok/s)")
    return text


def main():
    _check_imports()

    parser = argparse.ArgumentParser(
        description="HuggingFace inference demo for MiMo-V2-Flash."
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="XiaomiMiMo/MiMo-V2-Flash",
        help="HuggingFace Hub repo id or local directory path.",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default=(
            "Solve step by step: A train travels at 120 km/h for 2.5 hours, "
            "then at 80 km/h for 1.5 hours. What is the total distance?"
        ),
        help="Input prompt for text generation.",
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=128,
        help="Maximum number of new tokens to generate.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Sampling temperature. 0.0 = greedy decoding.",
    )
    parser.add_argument(
        "--top_p",
        type=float,
        default=1.0,
        help="Nucleus sampling top-p probability.",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="bfloat16",
        choices=["float32", "float16", "bfloat16"],
        help="Model weight dtype.",
    )
    parser.add_argument(
        "--load_in_4bit",
        action="store_true",
        default=False,
        help="Load model in 4-bit quantisation mode (requires bitsandbytes).",
    )
    args = parser.parse_args()

    tokenizer, model = load_model(
        model_path=args.model_path,
        dtype=args.dtype,
        load_in_4bit=args.load_in_4bit,
    )

    print(f"\nPrompt:\n{args.prompt}\n")
    print("-" * 60)
    output = generate(
        tokenizer,
        model,
        args.prompt,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
    )
    print(f"Output:\n{output}")


if __name__ == "__main__":
    main()
