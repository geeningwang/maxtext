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

r"""HuggingFace CPU reference for MiMo-V2-Flash comparison.

Runs the model step-by-step in greedy mode on CPU, capturing per-layer
hidden states and final logits at each decode step.  The output is a set
of NumPy files that can be compared against the MaxText/TPU activations
produced by demos/compare/maxtext_reference.py.

Setup (run once on a TPU worker with enough RAM, ~700 GB):
  # Install PyTorch (CPU-only, no CUDA needed):
  pip install --quiet torch --index-url https://download.pytorch.org/whl/cpu
  pip install --quiet transformers safetensors accelerate

  # Download the HF weights from GCS to a local path:
  mkdir -p /tmp/mimo-hf-model
  gsutil -m cp gs://jingnw-mimo-v2-flash-us-east5/hf-model/config.json   /tmp/mimo-hf-model/
  gsutil -m cp gs://jingnw-mimo-v2-flash-us-east5/hf-model/model.safetensors.index.json /tmp/mimo-hf-model/
  gsutil -m cp 'gs://jingnw-mimo-v2-flash-us-east5/hf-model/model_*.safetensors' /tmp/mimo-hf-model/

  NOTE: The weights are FP8 in the GCS bucket (~320 GB compressed).
  When loaded with torch_dtype=bfloat16 they occupy ~600 GB in RAM.
  Run on worker-0 which has the most headroom.

Usage:
  python3 demos/compare/hf_reference.py \
      --model_path /tmp/mimo-hf-model \
      --tokenizer_path $HOME/mimo-tokenizer \
      --prompt "<|im_start|>system\nYou are MiMo...<|im_end|>...<think></think>" \
      --max_new_tokens 16 \
      --out_dir /tmp/compare_hf \
      --layers_to_capture 0 5 10 47

  Outputs (in --out_dir):
    tokens.json          — list of generated token IDs + decoded strings
    step{N}_logits.npy   — float32 logits [vocab], for steps 0..max_new_tokens-1
    step{N}_layer{L}.npy — float32 hidden state [hidden], captured layers only
"""

import argparse
import json
import os
import sys
import time
from typing import Optional

import numpy as np

# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def _parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model_path", required=True,
                   help="Local directory containing safetensors + config.json. "
                        "Download from GCS first (see setup notes above).")
    p.add_argument("--tokenizer_path", default=None,
                   help="Path to tokenizer directory (defaults to --model_path).")
    p.add_argument("--prompt", default=(
        "<|im_start|>system\n"
        "You are MiMo, a helpful AI assistant engineered by Xiaomi."
        "<|im_end|><|im_start|>user\n"
        "Explain the key difference between attention and cross-attention."
        "<|im_end|><|im_start|>assistant\n"
        "<think></think>"
    ))
    p.add_argument("--max_new_tokens", type=int, default=16)
    p.add_argument("--out_dir", default="/tmp/compare_hf",
                   help="Directory where activation .npy files are written.")
    p.add_argument("--layers_to_capture", type=int, nargs="+",
                   default=[0, 1, 2, 5, 10, 20, 30, 40, 47],
                   help="Layer indices whose hidden-state outputs are saved. "
                        "Pass --layers_to_capture 0 1 2 to capture only first 3.")
    p.add_argument("--fp32_output", action="store_true",
                   help="Cast all saved arrays to float32 (default is bfloat16 → float32).")
    p.add_argument("--hub_id", default="XiaomiMiMo/MiMo-V2-Flash",
                   help="HuggingFace Hub repo ID for downloading custom architecture "
                        ".py files when not present in --model_path (e.g. gcsfuse mount).")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Import guard
# ---------------------------------------------------------------------------

def _require(pkg):
    try:
        import importlib
        importlib.import_module(pkg)
    except ImportError:
        print(f"ERROR: package '{pkg}' not found.  Run: pip install {pkg}", file=sys.stderr)
        sys.exit(1)

_require("torch")
_require("transformers")
_require("safetensors")

# When installed via `pip3 --user`, the packages land in ~/.local/lib/...
# Make sure that path is on sys.path.
import site, sys
user_site = site.getusersitepackages()
if user_site not in sys.path:
    sys.path.insert(0, user_site)


# ---------------------------------------------------------------------------
# Load helpers
# ---------------------------------------------------------------------------

def _load_model_and_tokenizer(model_path: str, tokenizer_path: Optional[str],
                              hub_id: str = "XiaomiMiMo/MiMo-V2-Flash"):
    import torch
    import gzip, shutil, tempfile
    from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig

    tok_path = tokenizer_path or model_path
    print(f"Loading tokenizer from {tok_path} …")
    tokenizer = AutoTokenizer.from_pretrained(tok_path, trust_remote_code=True)

    # gcsfuse serves the GCS objects with transparent gzip encoding.
    # The config.json in the bucket is gzip-compressed, which confuses
    # AutoConfig.  Detect it and decompress to a temp directory; all
    # other files are symlinked so no large data is copied.
    effective_model_path = model_path
    tmp_dir = None
    config_path = os.path.join(model_path, "config.json")
    try:
        with open(config_path, "rb") as f:
            magic = f.read(2)
        if magic == b'\x1f\x8b':  # gzip magic bytes
            print("  INFO: config.json is gzip-compressed (gcsfuse mount). "
                  "Creating a temp dir with a decompressed copy …")
            tmp_dir = tempfile.mkdtemp(prefix="mimo_cfg_")
            for fname in os.listdir(model_path):
                src = os.path.join(model_path, fname)
                dst = os.path.join(tmp_dir, fname)
                if fname == "config.json":
                    import json as _json
                    with gzip.open(src, "rb") as gf:
                        cfg_dict = _json.load(gf)
                    # Strip FP8 quantization so from_pretrained loads as plain bfloat16.
                    # (The quantization check in transformers re-reads config.json from
                    # disk, so the in-memory patch alone is not sufficient.)
                    cfg_dict.pop("quantization_config", None)
                    with open(dst, "w") as df:
                        _json.dump(cfg_dict, df, indent=2)
                else:
                    os.symlink(src, dst)
            # The GCS bucket has only safetensors + tokenizer files; the custom
            # architecture Python files (configuration_mimo_v2_flash.py, etc.)
            # live only on HuggingFace Hub.  Download them now (cached in
            # ~/.cache/huggingface/hub after the first run).
            print("  INFO: Downloading custom architecture .py files from "
                  f"HuggingFace Hub ({hub_id}) …")
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
            effective_model_path = tmp_dir
    except Exception as e:
        print(f"  WARNING: could not check config.json compression: {e}")

    print(f"Loading model config from {effective_model_path} …")
    config = AutoConfig.from_pretrained(effective_model_path, trust_remote_code=True)

    print(f"Loading model weights from {effective_model_path} "
          f"(streaming from gcsfuse/GCS — this may take 30-90 minutes) …")
    t0 = time.perf_counter()
    model = AutoModelForCausalLM.from_pretrained(
        effective_model_path,
        config=config,
        torch_dtype=torch.bfloat16,
        device_map="cpu",
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    )
    model.eval()
    elapsed = time.perf_counter() - t0
    print(f"Model loaded in {elapsed:.0f}s.")
    return tokenizer, model


# ---------------------------------------------------------------------------
# Activation hooks
# ---------------------------------------------------------------------------

class _LayerCapturer:
    """
    Registers a forward hook on model.model.layers[i] for each index in
    `layer_ids`.  After a forward pass, `.captures[i]` holds the layer
    output hidden-state as a float32 numpy array of shape [hidden_size].
    (We capture index [0, -1, :] = last token position, batch 0.)
    """

    def __init__(self, layers, layer_ids):
        self.captures = {}
        self._hooks = []
        for i in layer_ids:
            if i >= len(layers):
                print(f"WARNING: layer {i} out of range ({len(layers)} layers); skipping.")
                continue
            hook = layers[i].register_forward_hook(self._make_hook(i))
            self._hooks.append(hook)

    def _make_hook(self, layer_idx):
        def hook(module, inputs, output):  # pylint: disable=unused-argument
            # output is typically (hidden_states,) or a tuple starting with hidden_states
            hidden = output[0] if isinstance(output, tuple) else output
            # hidden shape: [batch, seq_len, hidden_size]
            # grab the LAST token position, batch 0
            h = hidden[0, -1, :].detach().float().cpu().numpy()
            self.captures[layer_idx] = h
        return hook

    def remove(self):
        for h in self._hooks:
            h.remove()
        self._hooks.clear()


# ---------------------------------------------------------------------------
# Step-by-step greedy decode with activation capture
# ---------------------------------------------------------------------------

def run_reference(model, tokenizer, prompt, max_new_tokens, layers_to_capture, out_dir):
    import torch

    os.makedirs(out_dir, exist_ok=True)

    inputs = tokenizer(prompt, return_tensors="pt")
    input_ids = inputs["input_ids"]  # [1, prompt_len]
    print(f"Prompt tokenized to {input_ids.shape[1]} tokens.")

    all_token_ids = input_ids[0].tolist()
    generated_ids = []
    step_info = []

    current_ids = input_ids
    past_key_values = None

    for step in range(max_new_tokens):
        print(f"  Step {step} / {max_new_tokens} …", end="\r", flush=True)

        # Register hooks for this step
        layers = model.model.layers
        capturer = _LayerCapturer(layers, layers_to_capture)

        t0 = time.perf_counter()
        with torch.inference_mode():
            out = model(
                input_ids=current_ids,
                past_key_values=past_key_values,
                use_cache=True,
                output_hidden_states=False,  # Hidden states via hooks; this would OOM
            )
        elapsed = time.perf_counter() - t0
        capturer.remove()

        logits_last = out.logits[0, -1, :].float().cpu().numpy()  # [vocab_size]
        next_token_id = int(np.argmax(logits_last))
        next_token_str = tokenizer.decode([next_token_id])

        # Top-10 for display
        top10_idx = np.argsort(logits_last)[-10:][::-1].tolist()
        top10_val = logits_last[top10_idx].tolist()
        top10_str = [tokenizer.decode([t]) for t in top10_idx]

        print(f"\n  Step {step:3d}: token={next_token_id} ({next_token_str!r})"
              f"  top-3: {list(zip(top10_idx[:3], top10_str[:3], [f'{v:.2f}' for v in top10_val[:3]]))}  "
              f" elapsed={elapsed:.1f}s")

        # Save logits
        np.save(os.path.join(out_dir, f"step{step:04d}_logits.npy"), logits_last)

        # Save layer hidden states
        for layer_idx, hidden in capturer.captures.items():
            np.save(os.path.join(out_dir, f"step{step:04d}_layer{layer_idx:02d}.npy"), hidden)

        step_info.append({
            "step": step,
            "token_id": next_token_id,
            "token_str": next_token_str,
            "top10_ids": top10_idx,
            "top10_logits": top10_val,
            "top10_strs": top10_str,
        })

        generated_ids.append(next_token_id)
        all_token_ids.append(next_token_id)

        # Prepare next iteration
        current_ids = torch.tensor([[next_token_id]])
        past_key_values = out.past_key_values

        # Stop on eos
        if next_token_id == tokenizer.eos_token_id:
            print("  EOS reached.")
            break

    print()
    output_text = tokenizer.decode(generated_ids, skip_special_tokens=False)
    print(f"\n=== Generated text ===\n{output_text}\n")

    tokens_path = os.path.join(out_dir, "tokens.json")
    with open(tokens_path, "w") as f:
        json.dump({
            "prompt": prompt,
            "prompt_token_ids": input_ids[0].tolist(),
            "generated_token_ids": generated_ids,
            "generated_text": output_text,
            "steps": step_info,
        }, f, indent=2)
    print(f"Saved token info  -> {tokens_path}")
    print(f"Saved activations -> {out_dir}/step*.npy")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = _parse_args()
    tokenizer, model = _load_model_and_tokenizer(args.model_path, args.tokenizer_path,
                                                  args.hub_id)
    run_reference(
        model=model,
        tokenizer=tokenizer,
        prompt=args.prompt,
        max_new_tokens=args.max_new_tokens,
        layers_to_capture=args.layers_to_capture,
        out_dir=args.out_dir,
    )


if __name__ == "__main__":
    main()
