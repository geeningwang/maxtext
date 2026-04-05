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
# Shard-by-shard FP8 → BF16 loader
# ---------------------------------------------------------------------------

def _load_weights_fp8_to_bf16(model, model_path: str):
    """Stream safetensors shards one at a time, dequantizing FP8 tensors.

    Bypasses FineGrainedFP8HfQuantizer entirely to avoid the ~730 GB peak
    that occurs when the quantizer holds both the FP8 source and the BF16
    destination in memory simultaneously.

    For each FP8 weight (float8_e4m3fn) and its co-located weight_scale_inv:
        dequant[i, j] = fp8_weight[i, j] * scale[i // bm, j // bn]
    where bm = rows // scale_rows, bn = cols // scale_cols  (dynamic per tensor
    — e.g. k_proj uses 96×128 blocks, not 128×128).
    Peak overhead: ~max_shard_size (~4 GB) on top of the ~620 GB BF16 model.
    """
    import json
    import torch
    from safetensors import safe_open
    from accelerate.utils import set_module_tensor_to_device

    index_path = os.path.join(model_path, "model.safetensors.index.json")
    with open(index_path) as f:
        index = json.load(f)
    weight_map = index["weight_map"]  # tensor_name → shard_filename

    shard_order = list(dict.fromkeys(weight_map.values()))
    shard_to_tensors: dict = {s: [] for s in shard_order}
    for name, shard in weight_map.items():
        shard_to_tensors[shard].append(name)

    n_shards = len(shard_order)
    n_fp8_pairs = sum(1 for n in weight_map if n.endswith(".weight_scale_inv"))
    print(f"  {n_shards} shards, {len(weight_map)} tensors ({n_fp8_pairs} FP8 pairs)")

    missing_keys = []

    for shard_idx, shard_file in enumerate(shard_order):
        shard_path = os.path.join(model_path, shard_file)
        shard_dict: dict = {}
        with safe_open(shard_path, framework="pt", device="cpu") as f:
            for name in shard_to_tensors[shard_file]:
                shard_dict[name] = f.get_tensor(name)

        # Dequantize each FP8 weight using its co-located scale tensor.
        for scale_name in [n for n in list(shard_dict) if n.endswith(".weight_scale_inv")]:
            weight_name = scale_name[: -len(".weight_scale_inv")] + ".weight"
            scale = shard_dict.pop(scale_name)
            if weight_name not in shard_dict:
                continue
            weight = shard_dict[weight_name]
            if weight.dtype == torch.float8_e4m3fn:
                rows, cols = weight.shape[-2], weight.shape[-1]
                sr, sc = scale.shape[-2], scale.shape[-1]
                bm, bn = rows // sr, cols // sc  # dynamic block dims per tensor
                batch_shape = weight.shape[:-2]
                w = weight.to(scale.dtype).reshape(-1, sr, bm, sc, bn)
                s = scale.reshape(-1, sr, 1, sc, 1)
                dq = (w * s).reshape(-1, rows, cols)
                shard_dict[weight_name] = dq.reshape(*batch_shape, rows, cols).to(torch.bfloat16)

        # Drop activation_scale tensors (not needed for inference).
        for name in [n for n in list(shard_dict) if "activation_scale" in n]:
            del shard_dict[name]

        # Assign tensors into the model one by one.
        for name, tensor in shard_dict.items():
            t = tensor.to(torch.bfloat16) if tensor.dtype not in (
                torch.bfloat16, torch.float32, torch.float16,
                torch.int32, torch.int64, torch.bool) else tensor
            try:
                set_module_tensor_to_device(model, name, device="cpu", value=t, dtype=None)
            except Exception:
                missing_keys.append(name)
        del shard_dict

        if (shard_idx + 1) % 10 == 0 or (shard_idx + 1) == n_shards:
            print(f"  [{shard_idx + 1:3d}/{n_shards}] {shard_file}", flush=True)

    if missing_keys:
        print(f"  WARNING: {len(missing_keys)} tensors not loaded: "
              f"{missing_keys[:5]}{'…' if len(missing_keys) > 5 else ''}")


# ---------------------------------------------------------------------------
# Load
# ---------------------------------------------------------------------------

def load_model(model_path: str = DEFAULT_MODEL_PATH):
    """Load tokeniser and model using the shard-by-shard FP8→BF16 loader.

    Avoids the ~730 GB peak of FineGrainedFP8HfQuantizer:
      1. Strips quantization_config so from_pretrained builds plain nn.Linear
         on meta device (effectively 0 GB parameter memory).
      2. Streams each safetensors shard, dequantizes FP8→BF16 with dynamic
         per-tensor block sizes, assigns into model, then frees the shard.
    Peak memory: ~620 GB BF16 steady-state + ~4 GB per-shard overhead.
    """
    import json
    import tempfile
    import torch
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

    effective_path = _make_effective_model_path(model_path)

    print(f"Loading tokeniser from {effective_path} …")
    tokenizer = AutoTokenizer.from_pretrained(effective_path, trust_remote_code=True)

    # Build a bare config dir with quantization_config stripped so the model
    # skeleton is plain nn.Linear — no FP8 quantizer, no weight allocation yet.
    with open(os.path.join(effective_path, "config.json")) as f:
        cfg_dict = json.load(f)
    cfg_dict.pop("quantization_config", None)
    bare_dir = tempfile.mkdtemp(prefix="mimo_cfg_bare_")
    with open(os.path.join(bare_dir, "config.json"), "w") as f:
        json.dump(cfg_dict, f)
    for fname in os.listdir(effective_path):
        if fname != "config.json":
            src = os.path.join(effective_path, fname)
            dst = os.path.join(bare_dir, fname)
            if not os.path.exists(dst):
                os.symlink(src, dst)

    print(f"Loading config from {bare_dir} …")
    config = AutoConfig.from_pretrained(bare_dir, trust_remote_code=True)

    _apply_pre_load_shims()

    # Instantiate model skeleton — parameters stay on meta device (~0 GB RAM).
    print(f"Instantiating model skeleton (meta device) …")
    t0 = time.perf_counter()
    model = AutoModelForCausalLM.from_pretrained(
        bare_dir,
        config=config,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    )
    skeleton_elapsed = time.perf_counter() - t0
    print(f"Skeleton ready in {skeleton_elapsed:.0f}s.")

    _apply_post_load_shims()

    # Stream weights shard-by-shard with correct dynamic block sizes.
    print(f"Loading weights (FP8→BF16 streaming) …")
    t1 = time.perf_counter()
    _load_weights_fp8_to_bf16(model, model_path)
    load_elapsed = time.perf_counter() - t1
    print(f"Weights loaded in {load_elapsed:.0f}s.")

    model.eval()
    print(f"Total load time: {skeleton_elapsed + load_elapsed:.0f}s.")
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
