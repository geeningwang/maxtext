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

Loads the model from a local directory (default: /mnt/mimo-weights) and
runs text generation.  The model's native FP8 E4M3 weights are automatically
dequantized to BF16 by HF Transformers using the per-128×128-block
weight_scale_inv tensors, requiring ~620 GB RAM.

Validated on worker-0 of the jingnw-node cluster:
  AMD EPYC 9B14, 180 vCPUs, 708 GB RAM.
  Weights at /mnt/mimo-weights (NFS from worker-1, 292 GB safetensors).

Requirements:
  pip install torch transformers safetensors accelerate huggingface_hub

Usage:
  # Simplest — uses /mnt/mimo-weights by default:
  python3 demos/mimo_v2_flash_demo_hf.py

  # Custom path / prompt:
  python3 demos/mimo_v2_flash_demo_hf.py \
      --model_path /mnt/mimo-weights \
      --prompt "The key to solving any hard problem is" \
      --max_new_tokens 64

Note: /mnt/mimo-weights contains only safetensors + JSON files, not the
custom architecture .py files.  On first run this script downloads those
from HuggingFace Hub (XiaomiMiMo/MiMo-V2-Flash) and caches them in
~/.cache/huggingface/hub.
"""

import argparse
import os
import sys
import time


HF_HUB_ID = "XiaomiMiMo/MiMo-V2-Flash"
DEFAULT_MODEL_PATH = "/mnt/mimo-weights"


# ---------------------------------------------------------------------------
# Import guard
# ---------------------------------------------------------------------------

def _check_imports():
    missing = []
    for pkg in ("torch", "transformers", "safetensors", "accelerate"):
        try:
            import importlib
            importlib.import_module(pkg)
        except ImportError:
            missing.append(pkg)
    if missing:
        print(
            f"ERROR: Missing required packages: {', '.join(missing)}\n"
            f"Install with:  pip install {' '.join(missing)}",
            file=sys.stderr,
        )
        sys.exit(1)


# ---------------------------------------------------------------------------
# Model path preparation
# ---------------------------------------------------------------------------

def _make_effective_model_path(model_path: str, hub_id: str = HF_HUB_ID) -> str:
    """Return a path that AutoConfig/AutoModelForCausalLM can open.

    /mnt/mimo-weights has only safetensors + JSON files — no custom .py files.
    Creates a tempdir, symlinks all files from model_path into it, then
    downloads the custom architecture .py files from HF Hub into that tempdir.
    Also handles gzip-compressed JSON (gcsfuse mounts).

    Important: quantization_config is preserved in config.json so that HF
    Transformers selects FineGrainedFP8HfQuantizer and applies weight_scale_inv
    during dequantization.  Removing it causes FP8 bytes to be misinterpreted
    as raw BF16 → garbled output.
    """
    import gzip
    import json
    import tempfile

    config_path = os.path.join(model_path, "config.json")
    with open(config_path, "rb") as f:
        magic = f.read(2)
    is_gzip = magic == b'\x1f\x8b'
    py_present = os.path.exists(
        os.path.join(model_path, "modeling_mimo_v2_flash.py"))

    if not is_gzip and py_present:
        # Directory already has everything — use as-is.
        return model_path

    reason = ("config.json is gzip-compressed"
              if is_gzip else "custom arch .py files not in model dir")
    print(f"  INFO: {reason}; creating temp dir with symlinks …")

    tmp_dir = tempfile.mkdtemp(prefix="mimo_cfg_")
    for fname in os.listdir(model_path):
        src = os.path.join(model_path, fname)
        dst = os.path.join(tmp_dir, fname)
        if fname.endswith(".json"):
            try:
                with open(src, "rb") as _f:
                    _magic = _f.read(2)
                if _magic == b'\x1f\x8b':
                    with gzip.open(src, "rb") as gf:
                        data = json.load(gf)
                else:
                    with open(src) as gf:
                        data = json.load(gf)
                with open(dst, "w") as df:
                    json.dump(data, df)
            except Exception as e:
                print(f"  WARNING: could not process {fname}: {e}")
                os.symlink(src, dst)
        else:
            os.symlink(src, dst)

    # Download custom architecture .py files from Hub (cached after first run).
    print(f"  INFO: downloading custom .py files from {hub_id} …")
    try:
        from huggingface_hub import snapshot_download
        arch_cache = snapshot_download(
            repo_id=hub_id,
            allow_patterns=["*.py"],
            ignore_patterns=["*.safetensors", "*.bin", "*.pt", "*.gguf"],
        )
        for fname in os.listdir(arch_cache):
            if fname.endswith(".py"):
                src = os.path.join(arch_cache, fname)
                dst = os.path.join(tmp_dir, fname)
                if not os.path.exists(dst):
                    os.symlink(src, dst)
                    print(f"    symlinked {fname}")
    except Exception as e:
        print(f"  WARNING: could not fetch arch files from Hub: {e}")

    return tmp_dir


# ---------------------------------------------------------------------------
# Transformers 5.x compatibility shims for MiMo-V2-Flash
# ---------------------------------------------------------------------------

def _apply_pre_load_shims():
    """Shims that must be in place before from_pretrained() is called."""
    import torch

    # transformers 5.x removed 'default' from ROPE_INIT_FUNCTIONS but
    # MiMo-V2-Flash's modeling code still references it.
    try:
        from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS
        if "default" not in ROPE_INIT_FUNCTIONS:
            def _default_rope(config=None, device=None, seq_len=None, **kw):
                base = getattr(config, "rope_theta", 10000.0)
                partial = getattr(config, "partial_rotary_factor", 1.0)
                head_dim = (getattr(config, "head_dim", None) or
                            config.hidden_size // config.num_attention_heads)
                dim = int(head_dim * partial)
                inv_freq = 1.0 / (base ** (
                    torch.arange(0, dim, 2, dtype=torch.int64).to(
                        device=device, dtype=torch.float) / dim))
                return inv_freq, 1.0
            ROPE_INIT_FUNCTIONS["default"] = _default_rope
            print("  INFO: injected ROPE_INIT_FUNCTIONS['default'] compat shim.")
    except Exception as e:
        print(f"  WARNING: RoPE shim failed: {e}")

    # transformers 5.x _init_weights calls compute_default_rope_parameters on
    # RotaryEmbedding instances, but MiMo's class never defined this method.
    try:
        from transformers import modeling_utils as _mu
        from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS as _ROPE
        _orig = _mu.PreTrainedModel._init_weights

        def _patched_init_weights(self, module):
            try:
                _orig(self, module)
            except AttributeError as e:
                if "compute_default_rope_parameters" not in str(e):
                    raise
                if (hasattr(module, "original_inv_freq") and
                        hasattr(module, "inv_freq") and hasattr(module, "config")):
                    fn = (_ROPE.get(getattr(module, "rope_type", "default"))
                          or _ROPE.get("default"))
                    if fn:
                        buf, _ = fn(module.config)
                        module.inv_freq.copy_(buf)
                        module.original_inv_freq.copy_(buf)

        _mu.PreTrainedModel._init_weights = _patched_init_weights
        print("  INFO: patched _init_weights for compute_default_rope_parameters fallback.")
    except Exception as e:
        print(f"  WARNING: _init_weights shim failed: {e}")


def _apply_post_load_shims():
    """Shims applied after from_pretrained() (model code is now imported)."""
    # eager_attention_forward in the model file doesn't accept position_ids
    # but transformers 5.x passes it as a kwarg.
    try:
        import functools
        for mname, mod in list(sys.modules.items()):
            if mod is None or "transformers_modules" not in mname:
                continue
            fn = getattr(mod, "eager_attention_forward", None)
            if fn is None:
                continue

            @functools.wraps(fn)
            def _tolerant(*a, position_ids=None, **kw):
                return fn(*a, **kw)

            mod.eager_attention_forward = _tolerant
            print(f"  INFO: patched eager_attention_forward in {mname}.")
    except Exception as e:
        print(f"  WARNING: eager_attention_forward patch failed: {e}")


# ---------------------------------------------------------------------------
# Load
# ---------------------------------------------------------------------------

def load_model(model_path: str = DEFAULT_MODEL_PATH):
    """Load the tokeniser and model from a local directory.

    FP8 weights are dequantized to BF16 automatically on CPU by HF Transformers
    (FineGrainedFP8HfQuantizer → Fp8Dequantize applies weight_scale_inv).
    Resident memory: ~620 GB BF16.
    """
    import torch
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

    effective_path = _make_effective_model_path(model_path)

    print(f"Loading tokeniser from {effective_path} …")
    tokenizer = AutoTokenizer.from_pretrained(effective_path, trust_remote_code=True)

    print(f"Loading config from {effective_path} …")
    config = AutoConfig.from_pretrained(effective_path, trust_remote_code=True)

    _apply_pre_load_shims()

    print(f"Loading model weights (FP8→BF16 dequant, ~620 GB) …")
    t0 = time.perf_counter()
    model = AutoModelForCausalLM.from_pretrained(
        effective_path,
        config=config,
        torch_dtype=torch.bfloat16,
        device_map="cpu",
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    )
    model.eval()
    elapsed = time.perf_counter() - t0
    print(f"Model loaded in {elapsed:.0f}s.")

    _apply_post_load_shims()
    return tokenizer, model


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

def generate(
    tokenizer,
    model,
    prompt: str,
    max_new_tokens: int = 128,
    temperature: float = 0.0,
    top_p: float = 1.0,
) -> str:
    """Run greedy (temperature=0) or sampled generation."""
    import torch

    inputs = tokenizer(prompt, return_tensors="pt")
    input_ids = inputs["input_ids"]

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
    n = len(generated)
    print(f"Generated {n} tokens in {elapsed:.2f}s  ({n / elapsed:.1f} tok/s)")
    return text


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    _check_imports()

    p = argparse.ArgumentParser(
        description="HuggingFace inference demo for MiMo-V2-Flash.")
    p.add_argument(
        "--model_path", default=DEFAULT_MODEL_PATH,
        help=f"Local directory containing model weights (default: {DEFAULT_MODEL_PATH})",
    )
    p.add_argument(
        "--prompt",
        default=(
            "Solve step by step: A train travels at 120 km/h for 2.5 hours, "
            "then at 80 km/h for 1.5 hours. What is the total distance?"
        ),
    )
    p.add_argument("--max_new_tokens", type=int, default=128)
    p.add_argument("--temperature", type=float, default=0.0,
                   help="Sampling temperature; 0.0 = greedy decoding.")
    p.add_argument("--top_p", type=float, default=1.0)
    args = p.parse_args()

    tokenizer, model = load_model(args.model_path)

    print(f"\nPrompt:\n{args.prompt}\n")
    print("-" * 60)
    output = generate(
        tokenizer, model, args.prompt,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
    )
    print(f"Output:\n{output}")


if __name__ == "__main__":
    main()
