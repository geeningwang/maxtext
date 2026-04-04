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
    p.add_argument("--no_fast_load", action="store_true",
                   help="Disable staged GCS loading and use slower gcsfuse mmap instead.")
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
# Fast GCS loading helpers (bypass gcsfuse mmap page-fault bottleneck)
# ---------------------------------------------------------------------------

def _gcsfuse_gcs_uri(local_path: str) -> Optional[str]:
    """If local_path is under a gcsfuse mount, return the gs:// URI for it.

    gcsfuse mmap reads via 4KB page-faults gives ~17 MB/s for large files.
    Using gcloud storage cp directly achieves ~1 GB/s.
    """
    import subprocess
    try:
        out = subprocess.check_output(["mount"], text=True, timeout=5)
    except Exception:
        return None
    abs_path = os.path.abspath(local_path)
    for line in out.splitlines():
        if "gcsfuse" not in line.lower():
            continue
        # Linux: "bucket on /mountpoint type fuse.gcsfuse (...)"
        parts = line.split()
        if len(parts) >= 3:
            bucket = parts[0]
            mountpoint = parts[2].rstrip("/")
            if abs_path.startswith(mountpoint + "/") or abs_path == mountpoint:
                suffix = abs_path[len(mountpoint):].lstrip("/")
                return f"gs://{bucket}/{suffix}" if suffix else f"gs://{bucket}"
    return None


def _load_model_staged(config, effective_model_path: str, gcs_model_uri: str,
                       torch_dtype) -> "AutoModelForCausalLM":
    """Load model via from_pretrained, but intercept safetensors shard loading
    to stage each shard from GCS before reading (bypassing gcsfuse mmap).

    accelerate uses safetensors.safe_open (a context manager) to read tensors
    one-by-one from each shard.  When the path is on a gcsfuse mount this
    triggers per-tensor 4KB page faults at ~17 MB/s.

    Fix: replace safetensors.safe_open with a shim that, for gcsfuse paths,
    first downloads the whole shard to /tmp via 'gcloud storage cp' (~1 GB/s),
    then opens the local copy with the real safe_open.

    Each shard occupies at most ~4 GB on disk (deleted after the context exits).
    """
    import subprocess
    import safetensors
    import safetensors.torch as _st
    import accelerate.utils.modeling as _aum
    from transformers import AutoModelForCausalLM

    staged_path = "/tmp/mimo_shard_staged.safetensors"
    _orig_safe_open = safetensors.safe_open
    _counter = [0]
    _total_bytes = [0]
    _total_t = [0.0]

    class _StagedSafeOpen:
        """Context-manager shim for safetensors.safe_open.

        For paths under a gcsfuse mount the entire shard is downloaded to
        /tmp first; then the real safe_open is opened on that local copy.
        The staged file is deleted when the context exits.
        """

        def __init__(self, path, framework, device="cpu"):
            path_str = str(path)
            # Resolve symlinks so gcsfuse-mounted paths are detected even
            # when accessed through a temp dir with symlinks.
            real_str = os.path.realpath(path_str)
            self._path_str = path_str
            self._staged = False
            if "/mimo-hf-gcs/" in real_str:
                bucket = gcs_model_uri.split("gs://")[1].split("/")[0]
                sub_path = real_str.split("/mimo-hf-gcs/")[1]
                gcs_uri_shard = f"gs://{bucket}/{sub_path}"
                _counter[0] += 1
                t0 = time.perf_counter()
                # Remove any leftover partial download before starting.
                try:
                    os.unlink(staged_path)
                except OSError:
                    pass
                for _f in (staged_path + "_.gstmp",):
                    try:
                        os.unlink(_f)
                    except OSError:
                        pass
                _max_retries = 5
                _last_stderr = ""
                for _attempt in range(_max_retries):
                    # Use gsutil cp (has built-in retry/resume and avoids gcloud
                    # logging-handler bugs seen with 'gcloud storage cp').
                    result = subprocess.run(
                        ["gsutil", "cp", gcs_uri_shard, staged_path],
                        capture_output=True, text=True,
                    )
                    if result.returncode == 0:
                        break
                    _last_stderr = result.stderr + result.stdout
                    _wait = min(2 ** _attempt, 30)  # 1, 2, 4, 8, 16 s
                    print(
                        f"  WARNING: gsutil cp attempt {_attempt + 1} failed "
                        f"for {os.path.basename(real_str)}; retrying in {_wait}s …",
                        flush=True,
                    )
                    time.sleep(_wait)
                else:
                    raise RuntimeError(
                        f"gsutil cp failed for {gcs_uri_shard} "
                        f"after {_max_retries} attempts:\n{_last_stderr}"
                    )
                shard_bytes = os.path.getsize(staged_path)
                t_dl = time.perf_counter() - t0
                _total_bytes[0] += shard_bytes
                _total_t[0] += t_dl
                shard_gb = shard_bytes / 2 ** 30
                dl_mbs = shard_bytes / 2 ** 20 / max(t_dl, 1e-6)
                shard_name = os.path.basename(real_str)
                print(
                    f"  [shard {_counter[0]:3d}] {shard_name:<48s}"
                    f" {shard_gb:.2f}GB  {t_dl:.1f}s  {dl_mbs:.0f} MB/s",
                    flush=True,
                )
                self._delegate = _orig_safe_open(staged_path, framework=framework, device=device)
                self._staged = True
            else:
                self._delegate = _orig_safe_open(path_str, framework=framework, device=device)

        def __enter__(self):
            self._delegate.__enter__()
            return self

        def __exit__(self, *args):
            result = self._delegate.__exit__(*args)
            if self._staged:
                try:
                    os.unlink(staged_path)
                except OSError:
                    pass
                self._staged = False
            return result

        def keys(self):
            return self._delegate.keys()

        def get_tensor(self, key):
            return self._delegate.get_tensor(key)

        def get_slice(self, key):
            return self._delegate.get_slice(key)

        def offset_keys(self):
            return self._delegate.offset_keys()

        def metadata(self):
            return self._delegate.metadata()

    # Patch safetensors.safe_open everywhere it is referenced
    safetensors.safe_open = _StagedSafeOpen
    _aum.safe_open = _StagedSafeOpen
    # Also cover the direct import in load_state_dict (from safetensors import safe_open)
    import sys as _sys
    _safetensors_mod = _sys.modules.get("safetensors")
    if _safetensors_mod is not None:
        _safetensors_mod.safe_open = _StagedSafeOpen

    print("Loading model via from_pretrained with staged GCS shard interception …",
          flush=True)
    # Use low_cpu_mem_usage=True to skip _init_weights (meta-device init).
    # This avoids allocating 582GB of random weights before loading from disk.
    # Our _StagedSafeOpen shim intercepts the safe_open calls that
    # low_cpu_mem_usage+device_map uses, staging each shard via gcloud storage cp.
    model = AutoModelForCausalLM.from_pretrained(
        effective_model_path,
        config=config,
        torch_dtype=torch_dtype,
        device_map="cpu",
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    )
    model.eval()

    # Restore originals
    safetensors.safe_open = _orig_safe_open
    _aum.safe_open = _orig_safe_open
    if _safetensors_mod is not None:
        _safetensors_mod.safe_open = _orig_safe_open

    if _counter[0] > 0:
        avg_mbs = _total_bytes[0] / 2 ** 20 / max(_total_t[0], 1e-6)
        print(
            f"All shards staged: {_total_bytes[0]/2**30:.1f}GB in {_total_t[0]:.0f}s "
            f"(avg {avg_mbs:.0f} MB/s  across {_counter[0]} shards)",
            flush=True,
        )
    else:
        print("WARNING: staged loader ran but intercepted 0 shards — "
              "gcsfuse path detection may have failed.", flush=True)
    return model


# ---------------------------------------------------------------------------
# Load helpers
# ---------------------------------------------------------------------------

def _load_model_and_tokenizer(model_path: str, tokenizer_path: Optional[str],
                              hub_id: str = "XiaomiMiMo/MiMo-V2-Flash",
                              fast_load: bool = True):
    import torch
    import gzip, shutil, tempfile
    from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig

    # Detect GCS URI early (before we create a tmp_dir that obscures the path)
    gcs_model_uri = _gcsfuse_gcs_uri(model_path) if fast_load else None
    if gcs_model_uri:
        print(f"[FAST LOAD] gcsfuse mount detected → will stage shards from {gcs_model_uri}",
              flush=True)
    elif fast_load:
        print("[FAST LOAD] no gcsfuse mount detected; falling back to from_pretrained.",
              flush=True)

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
        is_gzip = magic == b'\x1f\x8b'
        # Need a tmp_dir whenever config.json is gzip-compressed (gcsfuse) OR
        # when the custom arch .py files are absent from the model directory
        # (e.g. NFS/local copy from GCS that contains only weights + tokenizer).
        py_files_present = os.path.exists(
            os.path.join(model_path, "modeling_mimo_v2_flash.py"))
        needs_tmp_dir = is_gzip or not py_files_present
        if needs_tmp_dir:
            if is_gzip:
                print("  INFO: config.json is gzip-compressed (gcsfuse mount). "
                      "Creating a temp dir with a decompressed copy …")
            else:
                print("  INFO: custom arch .py files absent from model dir "
                      "(NFS/local copy). Creating a temp dir with symlinks …")
            tmp_dir = tempfile.mkdtemp(prefix="mimo_cfg_")
            for fname in os.listdir(model_path):
                src = os.path.join(model_path, fname)
                dst = os.path.join(tmp_dir, fname)
                if fname.endswith(".json"):
                    import json as _json
                    # gcsfuse may serve any JSON file gzip-compressed.
                    # Decompress if needed; strip quantization_config from config.json.
                    try:
                        with open(src, "rb") as _f:
                            _magic = _f.read(2)
                        if _magic == b'\x1f\x8b':
                            with gzip.open(src, "rb") as gf:
                                _d = _json.load(gf)
                        else:
                            with open(src, "r") as gf:
                                _d = _json.load(gf)
                        if fname == "config.json":
                            _d.pop("quantization_config", None)
                        with open(dst, "w") as df:
                            _json.dump(_d, df)
                    except Exception as _e:
                        print(f"  WARNING: could not process {fname}: {_e}")
                        os.symlink(src, dst)
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

    # Load tokenizer after effective_model_path is resolved (tmp_dir has decompressed JSONs).
    # If tokenizer_path is explicitly set to the same dir as model_path, use effective path.
    if tokenizer_path and tokenizer_path != model_path:
        tok_path = tokenizer_path
    else:
        tok_path = effective_model_path
    print(f"Loading tokenizer from {tok_path} …", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(tok_path, trust_remote_code=True)

    print(f"Loading model config from {effective_model_path} …")
    config = AutoConfig.from_pretrained(effective_model_path, trust_remote_code=True)

    # Compatibility shim: transformers 5.x removed 'default' from ROPE_INIT_FUNCTIONS
    # but the MiMo-V2-Flash modeling code still references it.
    try:
        from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS
        if "default" not in ROPE_INIT_FUNCTIONS:
            def _default_rope(config=None, device=None, seq_len=None, **kw):
                import math
                base = getattr(config, "rope_theta", 10000.0)
                partial = getattr(config, "partial_rotary_factor", 1.0)
                head_dim = getattr(config, "head_dim", None) or (
                    config.hidden_size // config.num_attention_heads)
                dim = int(head_dim * partial)
                inv_freq = 1.0 / (base ** (
                    torch.arange(0, dim, 2, dtype=torch.int64).to(
                        device=device, dtype=torch.float) / dim))
                return inv_freq, 1.0
            ROPE_INIT_FUNCTIONS["default"] = _default_rope
            print("  INFO: injected ROPE_INIT_FUNCTIONS['default'] compat shim "
                  "(transformers 5.x removed it).")
    except Exception as _e:
        print(f"  WARNING: could not inject RoPE shim: {_e}")

    # Compatibility shim: transformers 5.x _init_weights calls
    # module.compute_default_rope_parameters on every RotaryEmbedding instance
    # with rope_type=="default", but MiMoV2FlashRotaryEmbedding was written for
    # older transformers and never defined this method.
    #
    # We can't patch the class here because modeling_mimo_v2_flash.py is not
    # imported until AutoModelForCausalLM.from_pretrained() runs — scanning
    # sys.modules at this point would find only configuration_mimo_v2_flash.
    #
    # Instead, patch PreTrainedModel._init_weights globally so that it falls
    # back gracefully when compute_default_rope_parameters is absent, performing
    # the same inv_freq reinit it would have done via that method.
    try:
        from transformers import modeling_utils as _mu
        from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS as _ROPE_FNS2
        _orig_init_weights = _mu.PreTrainedModel._init_weights

        def _patched_init_weights(self, module):
            try:
                _orig_init_weights(self, module)
            except AttributeError as _e:
                if "compute_default_rope_parameters" not in str(_e):
                    raise
                # Reinit inv_freq directly — same logic transformers would do
                # via compute_default_rope_parameters.
                if (hasattr(module, "original_inv_freq") and
                        hasattr(module, "inv_freq") and
                        hasattr(module, "config")):
                    _rtype = getattr(module, "rope_type", "default")
                    _fn = _ROPE_FNS2.get(_rtype) or _ROPE_FNS2.get("default")
                    if _fn:
                        _buf, _ = _fn(module.config)
                        module.inv_freq.copy_(_buf)
                        module.original_inv_freq.copy_(_buf)

        _mu.PreTrainedModel._init_weights = _patched_init_weights
        print("  INFO: patched PreTrainedModel._init_weights for "
              "compute_default_rope_parameters fallback.")
    except Exception as _e:
        print(f"  WARNING: could not patch _init_weights: {_e}")

    t0 = time.perf_counter()
    if gcs_model_uri:
        print(f"Loading model weights via staged GCS copy (fast path) …", flush=True)
        model = _load_model_staged(config, effective_model_path, gcs_model_uri,
                                   torch_dtype=torch.bfloat16)
    else:
        print(f"Loading model weights from {effective_model_path} "
              f"(direct filesystem read, low_cpu_mem_usage=True) …", flush=True)
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
    print(f"Model loaded in {elapsed:.0f}s.", flush=True)
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
                                                  args.hub_id,
                                                  fast_load=not args.no_fast_load)
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
