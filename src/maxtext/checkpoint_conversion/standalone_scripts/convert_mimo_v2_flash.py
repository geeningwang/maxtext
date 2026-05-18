"""
Copyright 2026 Google LLC
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
     https://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

r"""Convert HuggingFace MiMo-V2-Flash weights to a MaxText Orbax checkpoint.

MiMo-V2-Flash (309B total / 15B active) is a hybrid attention MoE transformer
from Xiaomi AI Research.  Key architecture differences from a standard LLM:
  • Asymmetric head dims: Q/K head_dim=192, V head_dim=128
  • Partial RoPE with separate thetas for global (5M) and SWA (10K) attention
  • Hybrid attention: 9 global layers + 39 sliding-window layers (48 total)
  • Almost-all-MoE: layer 0 is dense MLP, layers 1-47 are 256-expert sparse MoE
  • noaux-TC correction bias in the router gate

Model card: https://huggingface.co/XiaomiMiMo/MiMo-V2-Flash

Usage:
  # From a local HF checkpoint directory (low-RAM / v6e-1 streaming mode):
  python3 -m maxtext.checkpoint_conversion.standalone_scripts.convert_mimo_v2_flash \
      --base_model_path /path/to/XiaomiMiMo-MiMo-V2-Flash \
      --maxtext_model_path gs://<bucket>/mimo-v2-flash/checkpoints/0/items \
      --tmpdir /mnt/scratch/mimo_tmp \
      --simulated_cpu_devices_count 1

  # High-RAM machine (loads all shards at once, faster):
  python3 -m maxtext.checkpoint_conversion.standalone_scripts.convert_mimo_v2_flash \
      --base_model_path /path/to/XiaomiMiMo-MiMo-V2-Flash \
      --maxtext_model_path gs://<bucket>/mimo-v2-flash/checkpoints/0/items

  # From a HuggingFace Hub model (auto-download):
  python3 -m maxtext.checkpoint_conversion.standalone_scripts.convert_mimo_v2_flash \
      --base_model_path XiaomiMiMo/MiMo-V2-Flash \
      --maxtext_model_path gs://<bucket>/mimo-v2-flash/checkpoints/0/items \
      --download_from_hub

Memory modes
------------
Default (no --tmpdir): loads all safetensors shards into RAM, then converts.
  Peak RAM ≈ raw shards (~620 GB) + flat dict (~350 GB) = ~970 GB.
  Not suitable for a v6e-1 VM.

Streaming (--tmpdir <path>): processes one decoder layer at a time and writes
  converted arrays to memory-mapped files in <path>.  Peak RAM is bounded by
  the largest single layer (~25 GB for a 256-expert MoE layer).  The tmpdir
  must have at least ~650 GB of free space (or more for float32 fallback).
  The output checkpoint on GCS is written from these memmaps.
"""

import argparse
import contextlib
import gc
import os
import pathlib
import shutil
import subprocess
import sys
import tempfile
import time
from collections.abc import Iterable

import numpy as np
from safetensors import safe_open
from tqdm import tqdm

from maxtext.utils import max_logging

# Try to use bfloat16 for memmap files to halve disk usage.
# ml_dtypes is always available when JAX is installed.
try:
    import ml_dtypes  # pylint: disable=import-outside-toplevel
    _BF16 = ml_dtypes.bfloat16
    # safetensors 0.7+ uses np.float8_e4m3fn / np.bfloat16 as attribute lookups.
    # numpy<2.1 and numpy 2.x don't expose these as top-level attributes, but
    # ml_dtypes provides them.  Monkey-patch numpy so safetensors can find them.
    for _fp8_name in (
        "bfloat16",
        "float8_e4m3fn", "float8_e4m3fnuz",
        "float8_e3m4", "float8_e4m3b11fnuz",
        "float8_e5m2", "float8_e5m2fnuz",
    ):
        if not hasattr(np, _fp8_name) and hasattr(ml_dtypes, _fp8_name):
            setattr(np, _fp8_name, getattr(ml_dtypes, _fp8_name))
except ImportError:
    _BF16 = None  # fall back to float32 in _MemmapStore


# ---------------------------------------------------------------------------
# GCS path helpers — transparent on-demand shard download
# ---------------------------------------------------------------------------

def _is_gcs(path: str) -> bool:
    return str(path).startswith("gs://")


def _gcs_ls_safetensors(gcs_dir: str) -> list[str]:
    """List all *.safetensors GCS URIs under gcs_dir using gsutil."""
    result = subprocess.run(
        ["gsutil", "ls", f"{gcs_dir.rstrip('/')}/*.safetensors"],
        capture_output=True, text=True, check=True,
    )
    return sorted(line.strip() for line in result.stdout.splitlines() if line.strip())


@contextlib.contextmanager
def _local_shard(shard_path: pathlib.Path):
    """Context manager: yield a local path for shard_path.

    For local paths, yields the path unchanged.
    For GCS paths (gs://...), downloads the shard to a temporary file,
    yields the local path, then deletes the temp file on exit.  Peak disk
    usage is therefore bounded to the size of a single shard (~5–15 GB).
    """
    path_str = str(shard_path)
    if not _is_gcs(path_str):
        yield shard_path
        return

    # Download GCS shard to a temp file.
    suffix = pathlib.Path(path_str.split("/")[-1]).suffix or ".safetensors"
    tmp_fd, tmp_path = tempfile.mkstemp(suffix=suffix, prefix="mimo_shard_")
    os.close(tmp_fd)
    try:
        print(f"[convert] gsutil cp {path_str} → {tmp_path}", flush=True)
        subprocess.run(
            ["gsutil", "-q", "cp", path_str, tmp_path],
            check=True,
        )
        yield pathlib.Path(tmp_path)
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass


# ---------------------------------------------------------------------------
# Model parameter dictionary
# ---------------------------------------------------------------------------

MODEL_PARAMS = {
    "mimo-v2-flash": {
        "num_hidden_layers": 48,
        "num_attention_heads": 64,
        "num_key_value_heads": 4,       # GA KV heads
        "swa_num_key_value_heads": 8,   # SWA KV heads
        "hidden_size": 4096,
        "head_dim": 192,                # Q/K dim
        "v_head_dim": 128,              # V dim
        "num_experts": 256,
        "moe_intermediate_size": 2048,
        "mlp_intermediate_size": 16384, # dense MLP (layer 0 only)
        "attention_projection_layout": "split",  # separate q_proj/k_proj/v_proj
        # 0 = global attention, 1 = sliding-window (SWA)
        "hybrid_layer_pattern": [
            0,1,1,1,1, 0,1,1,1,1,1, 0,1,1,1,1,1, 0,1,1,1,1,1, 0,1,1,1,1,1,
            0,1,1,1,1,1, 0,1,1,1,1,1, 0,1,1,1,1,1, 0
        ],
        # 0 = dense MLP, 1 = MoE
        "moe_layer_freq": [0] + [1] * 47,
    },
    "mimo-v2-5-pro": {
        "num_hidden_layers": 70,
        "num_attention_heads": 128,
        "num_key_value_heads": 8,       # GA KV heads (unified with SWA; both 8)
        "swa_num_key_value_heads": 8,   # SWA KV heads
        "hidden_size": 6144,
        "head_dim": 192,                # Q/K dim
        "v_head_dim": 128,              # V dim
        "num_experts": 384,
        "moe_intermediate_size": 2048,
        "mlp_intermediate_size": 16384, # dense MLP (layer 0 only)
        # HF stores Q/K/V as a single fused tensor; converter splits on dim 0.
        # Split offsets: [0, nq*dq], [nq*dq, nq*dq+nkv*dk], [nq*dq+nkv*dk, ...]
        # = [0, 24576], [24576, 26112], [26112, 27136]
        "attention_projection_layout": "fused_qkv",
        # 0 = global attention, 1 = sliding-window (SWA)
        # GA at positions: 0, 7, 15, 23, 31, 39, 47, 55, 62, 69
        "hybrid_layer_pattern": [
            0,1,1,1,1,1,1, 0,1,1,1,1,1,1,1, 0,1,1,1,1,1,1,1, 0,1,1,1,1,1,1,1,
            0,1,1,1,1,1,1,1, 0,1,1,1,1,1,1,1, 0,1,1,1,1,1,1,1, 0,1,1,1,1,1,1,
            0,1,1,1,1,1,1, 0
        ],
        # 0 = dense MLP (layer 0), 1 = MoE (layers 1-69)
        "moe_layer_freq": [0] + [1] * 69,
    },
}

# ---------------------------------------------------------------------------
# Shard-index helpers (no weights loaded until needed)
# ---------------------------------------------------------------------------

def _build_shard_index(shard_paths: list[pathlib.Path]) -> dict[str, pathlib.Path]:
    """Map every weight key to the shard file that contains it.

    Opens each shard just long enough to read its key list — no tensor data is
    loaded into RAM.  For GCS paths, each shard is downloaded to a temp file,
    indexed, then immediately deleted (peak disk = one shard at a time).
    """
    key_to_shard: dict[str, pathlib.Path] = {}
    for shard_path in tqdm(shard_paths, desc="Indexing shards"):
        with _local_shard(shard_path) as local_path:
            with safe_open(local_path, framework="np") as f:
                for key in f.keys():
                    key_to_shard[key] = shard_path  # keep original (GCS) path
    return key_to_shard


def _load_keys_batch(
    keys: Iterable[str],
    key_to_shard: dict[str, pathlib.Path],
    label: str = "",
    progress_every: int = 10,
    keep_fp8: bool = False,
) -> dict[str, np.ndarray]:
    """Load a set of weight keys, opening each shard at most once.

    When keep_fp8=False (default), returns bfloat16 numpy arrays (FP8 tensors
    are upcasted to float32 then downcasted to bfloat16).

    When keep_fp8=True, FP8 tensors (dtype starting with "float8") are kept in
    their native ml_dtypes.float8_e4m3fn dtype; all other tensors are loaded as
    bfloat16 as usual.  Use this for MoE expert weights when building a
    block-wise FP8 checkpoint.
    """
    # Group requested keys by their shard file.
    shard_to_keys: dict[pathlib.Path, list[str]] = {}
    for key in keys:
        shard = key_to_shard.get(key)
        if shard is not None:
            shard_to_keys.setdefault(shard, []).append(key)

    tensors: dict[str, np.ndarray] = {}
    total_shards = len(shard_to_keys)
    keys_loaded = 0
    for idx, (shard_path, batch) in enumerate(shard_to_keys.items()):
        shard_name = str(shard_path).rstrip("/").split("/")[-1]
        print(
            f"[convert] {label}  opening shard {idx+1}/{total_shards} ({shard_name}, {len(batch)} keys)  keys_so_far={keys_loaded}",
            flush=True,
        )
        with _local_shard(shard_path) as local_path, safe_open(local_path, framework="np") as f:
            for ki, key in enumerate(batch):
                raw = f.get_tensor(key)
                if keep_fp8 and getattr(raw.dtype, "name", "").startswith("float8"):
                    # Preserve native FP8 dtype (e.g. float8_e4m3fn) — no upcast.
                    tensors[key] = raw
                else:
                    # .astype(float32) handles fp8/fp16/bf16 → float32 via ml_dtypes.
                    # Then astype(_BF16) downcasts to bf16 via ml_dtypes.
                    arr_f32 = raw.astype(np.float32)
                    tensors[key] = arr_f32.astype(_BF16) if _BF16 is not None else arr_f32
                keys_loaded += 1
                if progress_every > 0 and (ki + 1) % progress_every == 0:
                    print(
                        f"[convert] {label}  shard {idx+1}/{total_shards}  key {ki+1}/{len(batch)}  keys_loaded={keys_loaded}",
                        flush=True,
                    )
        print(
            f"[convert] {label}  shard {idx+1}/{total_shards} done  keys_loaded={keys_loaded}",
            flush=True,
        )
    return tensors


def _apply_fp8_dequant(lt: dict, label: str = "", progress_every: int = 64) -> None:
    """Apply weight_scale_inv to FP8 weights that were loaded via .float().

    When PyTorch loads an FP8 E4M3FN tensor and calls .float(), it converts
    each FP8 value to float32 using the E4M3FN bit format, but does NOT apply
    the learned per-block scale (weight_scale_inv).  This function multiplies
    each weight by its block-expanded scale to produce the correct BF16 values.

    The dequantization formula matches the HuggingFace FP8 quantizer:
        dequant[i, j] = fp8_raw[i, j] * weight_scale_inv[i // bm, j // bn]
    where bm = rows // scale_rows, bn = cols // scale_cols.

    Modifies *lt* in-place: updates each weight entry and removes the
    corresponding ``weight_scale_inv`` entry.
    """
    scale_keys = [k for k in list(lt) if k.endswith(".weight_scale_inv")]
    total = len(scale_keys)
    for idx, scale_key in enumerate(scale_keys):
        if progress_every > 0 and (idx % progress_every == 0 or idx == total - 1):
            print(
                f"[convert] {label}  dequant {idx+1}/{total}",
                flush=True,
            )
        weight_key = scale_key[: -len(".weight_scale_inv")] + ".weight"
        if weight_key not in lt:
            continue
        w = lt[weight_key].astype(np.float32)
        s = lt.pop(scale_key).astype(np.float32)
        rows, cols = w.shape[-2], w.shape[-1]
        sr, sc = s.shape[-2], s.shape[-1]
        bm, bn = rows // sr, cols // sc
        s_exp = np.repeat(np.repeat(s, bm, axis=-2), bn, axis=-1)  # (rows, cols)
        result = w * s_exp
        lt[weight_key] = result.astype(_BF16) if _BF16 is not None else result


# ---------------------------------------------------------------------------
# Optional disk-backed storage for converted arrays (streaming / low-RAM mode)
# ---------------------------------------------------------------------------

class _MemmapStore:
    """Writes converted numpy arrays to memory-mapped files on disk.

    This keeps ``flat`` dictionary values resident on disk rather than in RAM,
    bounding peak memory to approximately one decoder layer at a time regardless
    of total model size.

    Dtype preference (in order): bfloat16 via ml_dtypes → float32 fallback.
    bfloat16 halves the required disk space (~650 GB vs ~1.24 TB for float32).

    A companion ``shapes.json`` file is written to *tmpdir* so that the store
    can later be re-opened (``from_dir``) without re-running the conversion.
    This allows resuming a failed checkpoint save without repeating the ~53 min
    layer-conversion phase.
    """
    _SHAPES_FILE = "shapes.json"

    def __init__(self, tmpdir: str) -> None:
        self._dir = tmpdir
        os.makedirs(tmpdir, exist_ok=True)
        self._dtype = _BF16 if _BF16 is not None else np.float32
        dtype_name = "bfloat16" if _BF16 is not None else "float32"
        self._shapes: dict[str, list[int]] = {}
        max_logging.log(
            f"MemmapStore: writing converted tensors to {tmpdir!r} as {dtype_name}. "
            "Peak RAM is bounded to ~one decoder layer regardless of model size."
        )

    def store(self, key: str, arr: np.ndarray) -> np.ndarray:
        """Write *arr* to a memmap file and return a read-only memmap view."""
        safe_name = key.replace("/", "__").replace(".", "_") + ".dat"
        fpath = os.path.join(self._dir, safe_name)
        mm = np.memmap(fpath, dtype=self._dtype, mode="w+", shape=arr.shape)
        np.copyto(mm, arr.astype(self._dtype))
        mm.flush()
        del mm  # close write handle
        self._shapes[key] = list(arr.shape)
        return np.memmap(fpath, dtype=self._dtype, mode="r", shape=arr.shape)

    def flush_shapes(self) -> None:
        """Persist shape metadata so the store can be restored with ``from_dir``."""
        import json  # pylint: disable=import-outside-toplevel
        shapes_path = os.path.join(self._dir, self._SHAPES_FILE)
        with open(shapes_path, "w", encoding="utf-8") as f:
            json.dump(self._shapes, f)

    @classmethod
    def from_dir(cls, tmpdir: str) -> tuple["_MemmapStore", dict[str, np.ndarray]]:
        """Restore a store from a *tmpdir* created by a previous run.

        Returns ``(store, flat)`` where *flat* is the complete weight dict of
        read-only numpy memmaps, ready to pass to ``save_weights_to_checkpoint``.
        This lets you resume a failed checkpoint save without repeating the
        layer-conversion phase.

        Raises ``FileNotFoundError`` if *tmpdir* or ``shapes.json`` are missing.
        """
        import json  # pylint: disable=import-outside-toplevel
        shapes_path = os.path.join(tmpdir, cls._SHAPES_FILE)
        if not os.path.isfile(shapes_path):
            raise FileNotFoundError(
                f"shapes.json not found in {tmpdir!r}. "
                "Run the conversion script once (without --resume_from_tmpdir) first."
            )
        with open(shapes_path, encoding="utf-8") as f:
            shapes = json.load(f)

        dtype = _BF16 if _BF16 is not None else np.float32
        store = cls.__new__(cls)
        store._dir = tmpdir
        store._dtype = dtype
        store._shapes = shapes

        flat: dict[str, np.ndarray] = {}
        for key, shape in shapes.items():
            safe_name = key.replace("/", "__").replace(".", "_") + ".dat"
            fpath = os.path.join(tmpdir, safe_name)
            if not os.path.isfile(fpath):
                raise FileNotFoundError(f"Memmap file missing: {fpath}")
            flat[key] = np.memmap(fpath, dtype=dtype, mode="r", shape=tuple(shape))
        max_logging.log(f"Restored {len(flat)} tensors from {tmpdir!r}")
        return store, flat

    def cleanup(self) -> None:
        """Remove all memmap files (call after the checkpoint has been saved)."""
        shutil.rmtree(self._dir, ignore_errors=True)


# ---------------------------------------------------------------------------
# Weight name mapping helpers
# ---------------------------------------------------------------------------

def convert_hf_to_maxtext(
    base_model_path: str,
    params: dict,
    tmpdir: str | None = None,
    on_layer_complete: "callable | None" = None,
    layer_range: "tuple[int, int] | None" = None,
    skip_global_weights: bool = False,
    keep_fp8: bool = False,
) -> dict:
    """Load and convert HF safetensors weights to a nested MaxText dict.

    When *tmpdir* is provided the function operates in **streaming / low-RAM
    mode**: converted arrays are written to memory-mapped files under *tmpdir*
    so that the ``flat`` dictionary never grows large in RAM.  Peak RAM is then
    bounded to approximately one decoder layer at a time (~25 GB for a 256-
    expert MoE layer in float32).  Without *tmpdir* all converted arrays are
    kept in RAM (original behaviour, requires ~970 GB for the full 309B model).

    Args:
        base_model_path: Directory containing ``*.safetensors`` shards.
        params: Model parameter dict from ``MODEL_PARAMS``.
        tmpdir: Optional scratch directory for memmap files (streaming mode).
        on_layer_complete: Optional callback called after each layer is converted.
        layer_range: Optional ``(start, end)`` tuple (end exclusive) to limit
            which decoder layers this call processes.  Layers outside the range
            are skipped entirely (no shard I/O).  Used by the distributed
            converter to partition work across workers.
        skip_global_weights: If True, skip loading the globally-shared weights
            (embeddings, decoder norm, lm_head).  Used by non-zero ranks in
            distributed mode where rank 0 handles global weights.
        keep_fp8: If True, MoE expert weights are stored as float8_e4m3fn in
            the output checkpoint instead of being dequantized to bfloat16.
            Their per-block scale_inv tensors are stored separately under
            ``{mt}.mlp.wi_0_scale_inv`` etc. (float32).
            Non-expert weights (attention projections, dense MLP) are still
            dequantized to bfloat16 as normal.

    Returns:
        Nested dict of numpy arrays (or memmaps) ready for Orbax checkpoint save.
    """
    num_layers = params["num_hidden_layers"]
    num_experts = params["num_experts"]
    hidden_size = params["hidden_size"]
    head_dim = params["head_dim"]
    v_head_dim = params["v_head_dim"]
    num_heads = params["num_attention_heads"]
    num_kv_heads = params["num_key_value_heads"]
    swa_num_kv_heads = params["swa_num_key_value_heads"]
    moe_intermediate_size = params["moe_intermediate_size"]
    mlp_intermediate_size = params["mlp_intermediate_size"]
    hybrid = params["hybrid_layer_pattern"]
    moe_freq = params["moe_layer_freq"]
    projection_layout = params.get("attention_projection_layout", "split")
    fused_qkv = projection_layout == "fused_qkv"
    # Precompute fused QKV split offsets (only used when fused_qkv=True).
    _q_end = num_heads * head_dim
    _k_end = _q_end + num_kv_heads * head_dim
    _v_end = _k_end + num_kv_heads * v_head_dim

    # ------------------------------------------------------------------
    # 1. Discover safetensors shards and build a key → shard index
    #    (no tensor data is loaded at this stage)
    # ------------------------------------------------------------------
    if _is_gcs(base_model_path):
        gcs_uris = _gcs_ls_safetensors(base_model_path)
        if not gcs_uris:
            raise FileNotFoundError(f"No *.safetensors files found in: {base_model_path}")
        # Keep as strings — pathlib.Path collapses gs:// → gs:/ which breaks _is_gcs().
        shard_paths = gcs_uris
    else:
        shard_paths = sorted(pathlib.Path(base_model_path).glob("*.safetensors"))
        if not shard_paths:
            raise FileNotFoundError(f"No *.safetensors files found in: {base_model_path}")
    max_logging.log(f"Found {len(shard_paths)} safetensors shards")
    print(f"[convert] Found {len(shard_paths)} safetensors shards", flush=True)

    key_to_shard = _build_shard_index(shard_paths)
    max_logging.log(f"Indexed {len(key_to_shard)} weight keys across {len(shard_paths)} shards")
    print(f"[convert] Indexed {len(key_to_shard)} weight keys across {len(shard_paths)} shards", flush=True)
    sys.stdout.flush(); sys.stderr.flush()

    # ------------------------------------------------------------------
    # 2. Prepare storage backend
    # ------------------------------------------------------------------
    mstore = _MemmapStore(tmpdir) if tmpdir else None

    def _put(key: str, arr: np.ndarray) -> np.ndarray:
        """Store a converted array, either in RAM or as a disk-backed memmap."""
        if mstore is not None:
            return mstore.store(key, arr)
        return arr  # keep in RAM (original behaviour)

    flat: dict[str, np.ndarray] = {}

    # ------------------------------------------------------------------
    # 3. Convert globally shared weights (embeddings, norm, lm_head)
    # ------------------------------------------------------------------
    shared_keys = [
        "model.embed_tokens.weight",
        "model.norm.weight",
        "lm_head.weight",
        "lm_head.weight_scale_inv",
    ]
    max_logging.log("[global] Loading shared weights (embed_tokens, norm, lm_head)...")
    print("[convert] [global] Loading shared weights (embed_tokens, norm, lm_head)...", flush=True)
    if skip_global_weights:
        max_logging.log("[global] Skipping global weights (skip_global_weights=True).")
        print("[convert] [global] Skipping global weights (non-rank-0 worker).", flush=True)
    else:
        shared = _load_keys_batch(shared_keys, key_to_shard, label="[global] load")
        _apply_fp8_dequant(shared, label="[global] dequant")
        max_logging.log(f"[global] Shared weights loaded and dequantized ({len(shared)} tensors).")
        print(f"[convert] [global] Shared weights loaded and dequantized ({len(shared)} tensors).", flush=True)

        emb = shared.get("model.embed_tokens.weight")
        if emb is not None:
            flat["token_embedder.embedding"] = _put("token_embedder.embedding", emb)

        norm = shared.get("model.norm.weight")
        if norm is not None:
            flat["decoder.decoder_norm.scale"] = _put("decoder.decoder_norm.scale", norm)

        lm = shared.get("lm_head.weight")
        if lm is not None:
            # HF: (vocab, hidden) → MaxText kernel: (hidden, vocab)
            flat["decoder.logits_dense.kernel"] = _put(
                "decoder.logits_dense.kernel", lm.T
            )

        del shared
        gc.collect()

    # ------------------------------------------------------------------
    # 4. Convert decoder layers one at a time
    # ------------------------------------------------------------------
    _layer_times: list[float] = []
    _convert_start = time.monotonic()

    for i in tqdm(range(num_layers), desc="Converting decoder layers"):
        # Skip layers outside this worker's assigned range.
        if layer_range is not None and not (layer_range[0] <= i < layer_range[1]):
            continue

        _t0 = time.monotonic()
        hf = f"model.layers.{i}"
        mt = f"decoder.layers.{i}"
        is_swa = hybrid[i] == 1
        is_moe = moe_freq[i] == 1
        kv_h = swa_num_kv_heads if is_swa else num_kv_heads
        layer_type = "MoE" if is_moe else "dense"
        attn_type = "SWA" if is_swa else "GA"
        max_logging.log(
            f"[layer {i:02d}/{num_layers-1}] Start  type={layer_type}/{attn_type}  "
            f"elapsed={time.monotonic()-_convert_start:.0f}s"
        )
        print(
            f"[convert] [layer {i:02d}/{num_layers-1}] Start  type={layer_type}/{attn_type}  "
            f"elapsed={time.monotonic()-_convert_start:.0f}s",
            flush=True,
        )

        # Collect all HF weight keys needed for this layer so we can open
        # each shard exactly once for this layer.
        # weight_scale_inv mates are included so _apply_fp8_dequant can scale
        # raw FP8 values (converted to float32 via .float()) to correct BF16.
        if fused_qkv:
            attn_proj_keys = [
                f"{hf}.self_attn.qkv_proj.weight",
                f"{hf}.self_attn.qkv_proj.weight_scale_inv",
            ]
        else:
            attn_proj_keys = [
                f"{hf}.self_attn.q_proj.weight",
                f"{hf}.self_attn.q_proj.weight_scale_inv",
                f"{hf}.self_attn.k_proj.weight",
                f"{hf}.self_attn.k_proj.weight_scale_inv",
                f"{hf}.self_attn.v_proj.weight",
                f"{hf}.self_attn.v_proj.weight_scale_inv",
            ]
        layer_keys: list[str] = attn_proj_keys + [
            f"{hf}.self_attn.o_proj.weight",
            f"{hf}.self_attn.o_proj.weight_scale_inv",
            f"{hf}.self_attn.attention_sink_bias",
            f"{hf}.input_layernorm.weight",
            f"{hf}.post_attention_layernorm.weight",
        ]
        if is_moe:
            layer_keys += [
                f"{hf}.mlp.gate.weight",
                f"{hf}.mlp.gate.e_score_correction_bias",
            ]
            for j in range(num_experts):
                layer_keys += [
                    f"{hf}.mlp.experts.{j}.gate_proj.weight",
                    f"{hf}.mlp.experts.{j}.gate_proj.weight_scale_inv",
                    f"{hf}.mlp.experts.{j}.up_proj.weight",
                    f"{hf}.mlp.experts.{j}.up_proj.weight_scale_inv",
                    f"{hf}.mlp.experts.{j}.down_proj.weight",
                    f"{hf}.mlp.experts.{j}.down_proj.weight_scale_inv",
                ]
        else:
            layer_keys += [
                f"{hf}.mlp.gate_proj.weight",
                f"{hf}.mlp.gate_proj.weight_scale_inv",
                f"{hf}.mlp.up_proj.weight",
                f"{hf}.mlp.up_proj.weight_scale_inv",
                f"{hf}.mlp.down_proj.weight",
                f"{hf}.mlp.down_proj.weight_scale_inv",
            ]

        if keep_fp8 and is_moe:
            # Split loading: expert weight tensors kept as FP8; everything else
            # (attention, norms, gate, scale_inv) loaded normally and dequantized.
            expert_weight_keys = []
            other_layer_keys = []
            for lk in layer_keys:
                parts = lk.split(".")
                # keys like: model.layers.{i}.mlp.experts.{j}.{proj}.weight
                # (NOT weight_scale_inv)
                if "experts" in parts and parts[-1] == "weight":
                    expert_weight_keys.append(lk)
                else:
                    other_layer_keys.append(lk)

            lt_fp8 = _load_keys_batch(
                expert_weight_keys, key_to_shard,
                label=f"[layer {i:02d}/{num_layers-1}] load-fp8",
                progress_every=20,
                keep_fp8=True,
            )
            lt_other = _load_keys_batch(
                other_layer_keys, key_to_shard,
                label=f"[layer {i:02d}/{num_layers-1}] load-other",
                progress_every=5,
            )
            _t_loaded = time.monotonic()
            max_logging.log(
                f"[layer {i:02d}/{num_layers-1}] Loaded {len(lt_fp8)} FP8 + {len(lt_other)} other tensors in {_t_loaded-_t0:.1f}s"
            )
            print(
                f"[convert] [layer {i:02d}/{num_layers-1}] Loaded {len(lt_fp8)} FP8 + {len(lt_other)} other tensors in {_t_loaded-_t0:.1f}s",
                flush=True,
            )
            # Dequantize attention projections (and any remaining FP8 keys in lt_other).
            _apply_fp8_dequant(lt_other, label=f"[layer {i:02d}/{num_layers-1}] dequant", progress_every=64)
            # Merge into a single lt dict so the rest of the layer code is unchanged.
            lt = {**lt_other, **lt_fp8}
            del lt_fp8, lt_other
        else:
            lt = _load_keys_batch(
                layer_keys, key_to_shard,
                label=f"[layer {i:02d}/{num_layers-1}] load",
                progress_every=5,
            )
            _t_loaded = time.monotonic()
            max_logging.log(
                f"[layer {i:02d}/{num_layers-1}] Loaded {len(lt)} tensors in {_t_loaded-_t0:.1f}s"
            )
            print(
                f"[convert] [layer {i:02d}/{num_layers-1}] Loaded {len(lt)} tensors in {_t_loaded-_t0:.1f}s",
                flush=True,
            )
            _apply_fp8_dequant(lt, label=f"[layer {i:02d}/{num_layers-1}] dequant", progress_every=64)

        max_logging.log(
            f"[layer {i:02d}/{num_layers-1}] Dequant done in {time.monotonic()-_t_loaded:.1f}s"
        )
        print(
            f"[convert] [layer {i:02d}/{num_layers-1}] Dequant done in {time.monotonic()-_t_loaded:.1f}s",
            flush=True,
        )
        sys.stdout.flush(); sys.stderr.flush()

        # ----- Attention -----
        if fused_qkv:
            qkv = lt.get(f"{hf}.self_attn.qkv_proj.weight")
            if qkv is not None:
                # qkv shape: [q_size+k_size+v_size, hidden_size] — split on dim 0
                q = qkv[:_q_end, :]
                k = qkv[_q_end:_k_end, :]
                v = qkv[_k_end:_v_end, :]
            else:
                q = k = v = None
        else:
            q = lt.get(f"{hf}.self_attn.q_proj.weight")
            k = lt.get(f"{hf}.self_attn.k_proj.weight")
            v = lt.get(f"{hf}.self_attn.v_proj.weight")
        o = lt.get(f"{hf}.self_attn.o_proj.weight")
        if q is not None:
            flat[f"{mt}.self_attn.query.kernel"] = _put(
                f"{mt}.self_attn.query.kernel",
                q.T.reshape(hidden_size, num_heads, head_dim),
            )
        if k is not None:
            flat[f"{mt}.self_attn.key.kernel"] = _put(
                f"{mt}.self_attn.key.kernel",
                k.T.reshape(hidden_size, kv_h, head_dim),
            )
        if v is not None:
            flat[f"{mt}.self_attn.value.kernel"] = _put(
                f"{mt}.self_attn.value.kernel",
                v.T.reshape(hidden_size, kv_h, v_head_dim),
            )
        if o is not None:
            flat[f"{mt}.self_attn.out.kernel"] = _put(
                f"{mt}.self_attn.out.kernel",
                o.T.reshape(num_heads, v_head_dim, hidden_size),
            )
        sink = lt.get(f"{hf}.self_attn.attention_sink_bias")
        if sink is not None:
            flat[f"{mt}.self_attn.sink_bias"] = _put(
                f"{mt}.self_attn.sink_bias", sink
            )

        # ----- Layer norms -----
        ln1 = lt.get(f"{hf}.input_layernorm.weight")
        ln2 = lt.get(f"{hf}.post_attention_layernorm.weight")
        if ln1 is not None:
            flat[f"{mt}.input_layernorm.scale"] = _put(
                f"{mt}.input_layernorm.scale", ln1
            )
        if ln2 is not None:
            flat[f"{mt}.post_attention_layernorm.scale"] = _put(
                f"{mt}.post_attention_layernorm.scale", ln2
            )

        # ----- FFN -----
        if is_moe:
            gate_w = lt.get(f"{hf}.mlp.gate.weight")
            corr_bias = lt.get(f"{hf}.mlp.gate.e_score_correction_bias")
            if gate_w is not None:
                flat[f"{mt}.mlp.gate.weight"] = _put(
                    f"{mt}.mlp.gate.weight", gate_w
                )
            if corr_bias is not None:
                flat[f"{mt}.mlp.gate.e_score_correction_bias"] = _put(
                    f"{mt}.mlp.gate.e_score_correction_bias", corr_bias
                )

            # Stack per-expert weights: (num_experts, dim_in, dim_out).
            # Allocated as float32 locally (no permanent RAM cost in streaming
            # mode: they are flushed to memmap and freed at the end of the loop
            # iteration).
            if keep_fp8:
                # float8_e4m3fn expert weights + float32 per-block scale_inv.
                # HF scale shape per expert: (I//128, H//128); after .T → (H//128, I//128).
                _fp8_dtype = ml_dtypes.float8_e4m3fn
                wi_0_stack = np.zeros(
                    (num_experts, hidden_size, moe_intermediate_size), dtype=_fp8_dtype
                )
                wi_1_stack = np.zeros(
                    (num_experts, hidden_size, moe_intermediate_size), dtype=_fp8_dtype
                )
                wo_stack = np.zeros(
                    (num_experts, moe_intermediate_size, hidden_size), dtype=_fp8_dtype
                )
                scale_h = hidden_size // 128
                scale_i = moe_intermediate_size // 128
                wi_0_scale = np.ones((num_experts, scale_h, scale_i), dtype=np.float32)
                wi_1_scale = np.ones((num_experts, scale_h, scale_i), dtype=np.float32)
                wo_scale = np.ones((num_experts, scale_i, scale_h), dtype=np.float32)

                for j in range(num_experts):
                    gp = lt.get(f"{hf}.mlp.experts.{j}.gate_proj.weight")
                    up = lt.get(f"{hf}.mlp.experts.{j}.up_proj.weight")
                    dp = lt.get(f"{hf}.mlp.experts.{j}.down_proj.weight")
                    gp_sc = lt.get(f"{hf}.mlp.experts.{j}.gate_proj.weight_scale_inv")
                    up_sc = lt.get(f"{hf}.mlp.experts.{j}.up_proj.weight_scale_inv")
                    dp_sc = lt.get(f"{hf}.mlp.experts.{j}.down_proj.weight_scale_inv")
                    if gp is not None:
                        wi_0_stack[j] = gp.T   # HF (I, H) → MaxText (H, I)
                    if up is not None:
                        wi_1_stack[j] = up.T
                    if dp is not None:
                        wo_stack[j] = dp.T     # HF (H, I) → MaxText (I, H)
                    # Transpose scale to match transposed weight blocks.
                    if gp_sc is not None:
                        wi_0_scale[j] = gp_sc.T   # (I//128, H//128) → (H//128, I//128)
                    if up_sc is not None:
                        wi_1_scale[j] = up_sc.T
                    if dp_sc is not None:
                        wo_scale[j] = dp_sc.T     # (H//128, I//128) → (I//128, H//128)

                flat[f"{mt}.mlp.wi_0"] = _put(f"{mt}.mlp.wi_0", wi_0_stack)
                flat[f"{mt}.mlp.wi_1"] = _put(f"{mt}.mlp.wi_1", wi_1_stack)
                flat[f"{mt}.mlp.wo"] = _put(f"{mt}.mlp.wo", wo_stack)
                flat[f"{mt}.mlp.wi_0_scale_inv"] = _put(f"{mt}.mlp.wi_0_scale_inv", wi_0_scale)
                flat[f"{mt}.mlp.wi_1_scale_inv"] = _put(f"{mt}.mlp.wi_1_scale_inv", wi_1_scale)
                flat[f"{mt}.mlp.wo_scale_inv"] = _put(f"{mt}.mlp.wo_scale_inv", wo_scale)

                del wi_0_stack, wi_1_stack, wo_stack, wi_0_scale, wi_1_scale, wo_scale
            else:
                wi_0_stack = np.zeros(
                    (num_experts, hidden_size, moe_intermediate_size), dtype=np.float32
                )
                wi_1_stack = np.zeros(
                    (num_experts, hidden_size, moe_intermediate_size), dtype=np.float32
                )
                wo_stack = np.zeros(
                    (num_experts, moe_intermediate_size, hidden_size), dtype=np.float32
                )

                for j in range(num_experts):
                    gp = lt.get(f"{hf}.mlp.experts.{j}.gate_proj.weight")
                    up = lt.get(f"{hf}.mlp.experts.{j}.up_proj.weight")
                    dp = lt.get(f"{hf}.mlp.experts.{j}.down_proj.weight")
                    if gp is not None:
                        wi_0_stack[j] = gp.T   # HF (I, H) → MaxText (H, I)
                    if up is not None:
                        wi_1_stack[j] = up.T
                    if dp is not None:
                        wo_stack[j] = dp.T     # HF (H, I) → MaxText (I, H)

                flat[f"{mt}.mlp.wi_0"] = _put(f"{mt}.mlp.wi_0", wi_0_stack)
                flat[f"{mt}.mlp.wi_1"] = _put(f"{mt}.mlp.wi_1", wi_1_stack)
                flat[f"{mt}.mlp.wo"] = _put(f"{mt}.mlp.wo", wo_stack)

                # Free the large float32 stacks immediately — in streaming mode
                # the data is now safely on disk.
                del wi_0_stack, wi_1_stack, wo_stack
        else:
            # Dense MLP — only layer 0 in the default config
            gp = lt.get(f"{hf}.mlp.gate_proj.weight")
            up = lt.get(f"{hf}.mlp.up_proj.weight")
            dp = lt.get(f"{hf}.mlp.down_proj.weight")
            if gp is not None:
                flat[f"{mt}.mlp.wi_0.kernel"] = _put(
                    f"{mt}.mlp.wi_0.kernel", gp.T
                )
            if up is not None:
                flat[f"{mt}.mlp.wi_1.kernel"] = _put(
                    f"{mt}.mlp.wi_1.kernel", up.T
                )
            if dp is not None:
                flat[f"{mt}.mlp.wo.kernel"] = _put(
                    f"{mt}.mlp.wo.kernel", dp.T
                )

        # Release this layer's raw tensors before moving to the next layer.
        del lt

        # Streaming-save: flush this layer's keys to the checkpoint immediately
        # and free the memmaps + dat files so RAM stays bounded to ~one layer.
        if on_layer_complete is not None:
            layer_prefix = f"decoder.layers.{i}."
            layer_keys = {k: v for k, v in flat.items() if k.startswith(layer_prefix)}
            on_layer_complete(i, layer_keys)
            for k in layer_keys:
                arr = flat.pop(k)
                del arr
                if mstore is not None:
                    safe_name = k.replace("/", "__").replace(".", "_") + ".dat"
                    dat_path = os.path.join(tmpdir, safe_name)  # type: ignore[arg-type]
                    try:
                        os.remove(dat_path)
                    except OSError:
                        pass

        _layer_elapsed = time.monotonic() - _t0
        _layer_times.append(_layer_elapsed)
        avg = sum(_layer_times) / len(_layer_times)
        remaining = (num_layers - (i + 1)) * avg
        max_logging.log(
            f"[layer {i:02d}/{num_layers-1}] Done  layer_time={_layer_elapsed:.0f}s  "
            f"avg={avg:.0f}s  ETA={remaining/60:.1f}min  "
            f"total_elapsed={time.monotonic()-_convert_start:.0f}s"
        )
        print(
            f"[convert] [layer {i:02d}/{num_layers-1}] Done  layer_time={_layer_elapsed:.0f}s  "
            f"avg={avg:.0f}s  ETA={remaining/60:.1f}min  "
            f"total_elapsed={time.monotonic()-_convert_start:.0f}s",
            flush=True,
        )
        gc.collect()

    # ------------------------------------------------------------------
    # 5. Persist shape metadata (enables --resume_from_tmpdir on rerun)
    # ------------------------------------------------------------------
    if mstore is not None:
        mstore.flush_shapes()
        max_logging.log(f"Shapes metadata written to {tmpdir}/shapes.json")

    max_logging.log(f"Converted {len(flat)} weight tensors.")
    return flat  # caller uses _save_zarr_direct (no unflatten needed)


# ---------------------------------------------------------------------------
# Direct zarr2 checkpoint writer (bypasses Orbax pytree RAM blow-up)
# ---------------------------------------------------------------------------

def _write_one_zarr_array(
    items_dir: pathlib.Path,
    key: str,
    arr: np.ndarray,
    compressor,
) -> dict:
    """Write one parameter array to a zarr sub-directory under items_dir.

    Returns the tree_meta entry dict for this key.  Called by both
    ``_save_zarr_direct`` (batch mode) and ``convert_and_save_streaming``
    (per-layer mode).
    """
    import zarr  # pylint: disable=import-outside-toplevel
    import json  # pylint: disable=import-outside-toplevel

    zarr_name = f"params.params.{key}"
    zarr_path = items_dir / zarr_name

    dtype_name = getattr(arr.dtype, "name", "")
    is_bf16 = dtype_name == "bfloat16"
    is_fp8 = dtype_name.startswith("float8")

    if is_bf16:
        write_arr = arr.view(np.uint16)
        write_dtype = np.uint16
    elif is_fp8:
        # Store FP8 as uint8 bytes; annotate .zarray so TensorStore can decode.
        write_arr = arr.view(np.uint8)
        write_dtype = np.uint8
    else:
        write_arr = arr
        write_dtype = arr.dtype

    z = zarr.open_array(
        str(zarr_path), mode="w",
        shape=write_arr.shape,
        dtype=write_dtype,
        chunks=write_arr.shape,  # single chunk = single GCS object; avoids 2000+ tiny HTTP requests
        compressor=compressor,
        dimension_separator=".",
    )
    z[:] = write_arr
    del z

    zarray_path = zarr_path / ".zarray"
    _meta = json.loads(zarray_path.read_text())
    if isinstance(_meta.get("compressor"), dict):
        _meta["compressor"].pop("checksum", None)
        zarray_path.write_text(json.dumps(_meta))

    if is_bf16:
        meta = json.loads(zarray_path.read_text())
        meta["dtype"] = "bfloat16"
        zarray_path.write_text(json.dumps(meta))
    elif is_fp8:
        meta = json.loads(zarray_path.read_text())
        meta["dtype"] = dtype_name  # e.g. "float8_e4m3fn"
        zarray_path.write_text(json.dumps(meta))

    key_parts = ["params", "params"] + key.split(".")
    return {
        str(tuple(key_parts)): {
            "key_metadata": [{"key": p, "key_type": 2} for p in key_parts],
            "value_metadata": {"value_type": "np.ndarray", "skip_deserialize": False},
        }
    }


def _write_checkpoint_metadata(
    step_dir: pathlib.Path,
    items_dir: pathlib.Path,
    tree_meta: dict,
    init_ts: int,
    total: int,
) -> None:
    """Write _METADATA and _CHECKPOINT_METADATA to finalise the checkpoint."""
    import json  # pylint: disable=import-outside-toplevel
    import time  # pylint: disable=import-outside-toplevel

    metadata = {
        "tree_metadata": {
            "('step',)": {
                "key_metadata": [{"key": "step", "key_type": 2}],
                "value_metadata": {"value_type": "scalar", "skip_deserialize": False},
            },
            **tree_meta,
            "('opt_state',)": {
                "key_metadata": [{"key": "opt_state", "key_type": 2}],
                "value_metadata": {"value_type": "Dict", "skip_deserialize": True},
            },
        },
        "use_ocdbt": False,
        "use_zarr3": False,
        "store_array_data_equal_to_fill_value": True,
        "custom_metadata": None,
    }
    (items_dir / "_METADATA").write_text(json.dumps(metadata))
    (step_dir / "_CHECKPOINT_METADATA").write_text(json.dumps({
        "item_handlers": {
            "items": "orbax.checkpoint._src.handlers.pytree_checkpoint_handler.PyTreeCheckpointHandler"
        },
        "metrics": {},
        "performance_metrics": {},
        "init_timestamp_nsecs": init_ts,
        "commit_timestamp_nsecs": time.time_ns(),
        "custom_metadata": {},
    }))
    # GCS checkpoints require commit_success.txt to be considered finalized by Orbax.
    (items_dir / "commit_success.txt").write_text("")
    max_logging.log(
        f"Checkpoint saved at {step_dir} "
        f"({total} arrays, peak RAM bounded to largest single array)"
    )


def convert_and_save_streaming(
    base_model_path: str,
    maxtext_model_path: str,
    params: dict,
    step: int = 0,
    keep_fp8: bool = False,
) -> None:
    """Convert HF weights and write the zarr checkpoint in a single pass.

    Unlike the two-phase (convert-all → save-all) approach, this function
    writes each decoder layer's zarr arrays **immediately** after conversion
    and frees the working buffers.  Peak RAM is bounded to approximately one
    decoder layer (~25–50 GB) regardless of total model size, making it safe
    on a 708 GB machine for the full 309B MiMo-V2-Flash model.

    No ``--tmpdir`` is needed; all intermediates live in local RAM only for
    the duration of a single layer.

    When keep_fp8=True, MoE expert weights are stored as float8_e4m3fn with
    separate scale_inv tensors.  Use --keep_fp8 on the CLI to enable this.
    """
    import zarr  # pylint: disable=import-outside-toplevel
    import json  # pylint: disable=import-outside-toplevel
    import time  # pylint: disable=import-outside-toplevel
    import numcodecs  # pylint: disable=import-outside-toplevel

    root = pathlib.Path(maxtext_model_path)
    root.mkdir(parents=True, exist_ok=True)
    step_dir = root / str(step)
    items_dir = step_dir / "items"
    try:
        if step_dir.exists():
            shutil.rmtree(step_dir)
    except OSError as _e:
        # gcsfuse does not support rmdir; zarr mode="w" will overwrite existing files.
        max_logging.log(f"Note: could not remove {step_dir} ({_e}); zarr will overwrite in place.")
    items_dir.mkdir(parents=True, exist_ok=True)

    init_ts = time.time_ns()
    compressor = numcodecs.Zstd(level=1)
    z_step = zarr.open_array(
        str(items_dir / "step"), mode="w",
        shape=(), dtype="<i8",
        compressor=compressor,
        dimension_separator=".",
    )
    z_step[()] = step

    tree_meta: dict = {}
    arrays_written = [0]  # mutable counter accessible inside callback

    def _on_layer_complete(layer_idx: int, layer_flat: dict) -> None:
        """Callback: write layer_idx's arrays to zarr immediately."""
        _t_save = time.time()
        total_arrays = len(layer_flat)
        for arr_idx, (key, arr) in enumerate(sorted(layer_flat.items())):
            print(
                f"[convert] saving layer {layer_idx} array {arr_idx+1}/{total_arrays}: {key}  shape={arr.shape}",
                flush=True,
            )
            tree_meta.update(_write_one_zarr_array(items_dir, key, arr, compressor))
            arrays_written[0] += 1
        save_elapsed = time.time() - _t_save
        max_logging.log(
            f"  Streaming-saved layer {layer_idx} "
            f"({len(layer_flat)} arrays, total so far: {arrays_written[0]}, "
            f"save_time={save_elapsed:.1f}s)"
        )
        print(
            f"[convert] Streaming-saved layer {layer_idx} "
            f"({len(layer_flat)} arrays, total so far: {arrays_written[0]}, "
            f"save_time={save_elapsed:.1f}s)",
            flush=True,
        )

    # Run conversion; the callback writes + frees each decoder layer in turn.
    # Global weights (embeddings, norm, logits) are returned in `flat` after
    # all layers are done.
    flat = convert_hf_to_maxtext(
        base_model_path,
        params,
        tmpdir=None,           # no tmpdir — arrays stay in RAM only transiently
        on_layer_complete=_on_layer_complete,
        keep_fp8=keep_fp8,
    )

    # Write remaining global weights (returned in flat after layer loop).
    total_global = len(flat)
    max_logging.log(f"Writing {total_global} global weight arrays to checkpoint...")
    for key, arr in sorted(flat.items()):
        tree_meta.update(_write_one_zarr_array(items_dir, key, arr, compressor))
        arrays_written[0] += 1
    del flat
    gc.collect()

    _write_checkpoint_metadata(step_dir, items_dir, tree_meta, init_ts, arrays_written[0])


def _save_zarr_direct(
    maxtext_model_path: str,
    flat_dict: dict[str, np.ndarray],
    step: int = 0,
) -> None:
    """Write an Orbax-compatible zarr2 checkpoint one array at a time.

    Peak RAM = largest single array (~4 GB for a 256-expert MoE stack).
    Bypasses the Orbax/JAX pytree traversal that materialises all memmaps
    into device memory before writing, which would OOM on a 172 GB VM with
    576 GB of parameters.

    The output directory structure exactly matches what
    ``llama_or_mistral_ckpt.save_weights_to_checkpoint`` would produce with
    ``use_ocdbt=False, use_zarr3=False``.
    """
    import zarr  # pylint: disable=import-outside-toplevel
    import json  # pylint: disable=import-outside-toplevel
    import time  # pylint: disable=import-outside-toplevel
    import numcodecs  # pylint: disable=import-outside-toplevel

    root = pathlib.Path(maxtext_model_path)
    root.mkdir(parents=True, exist_ok=True)

    step_dir = root / str(step)
    items_dir = step_dir / "items"
    try:
        if step_dir.exists():
            shutil.rmtree(step_dir)
    except OSError as _e:
        max_logging.log(f"Note: could not remove {step_dir} ({_e}); zarr will overwrite in place.")
    items_dir.mkdir(parents=True, exist_ok=True)

    init_ts = time.time_ns()
    # numcodecs.Zstd includes a "checksum" field that TensorStore rejects.
    # Build the compressor spec manually to match what TensorStore expects.
    compressor = numcodecs.Zstd(level=1)
    _TS_COMPRESSOR = {"id": "zstd", "level": 1}  # no "checksum" field

    # Step scalar (same format as Orbax produces)
    z_step = zarr.open_array(
        str(items_dir / "step"), mode="w",
        shape=(), dtype="<i8",
        compressor=compressor,
        dimension_separator=".",
    )
    z_step[()] = step

    # Parameter arrays, one at a time
    total = len(flat_dict)
    tree_meta: dict = {}
    for i, (key, arr) in enumerate(flat_dict.items()):
        tree_meta.update(_write_one_zarr_array(items_dir, key, arr, compressor))
        if (i + 1) % 50 == 0 or i + 1 == total:
            max_logging.log(f"  [{i + 1}/{total}] wrote {key}")

    _write_checkpoint_metadata(step_dir, items_dir, tree_meta, init_ts, total)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main(args):
    os.environ["JAX_PLATFORMS"] = "cpu"
    os.environ["XLA_FLAGS"] = f"--xla_force_host_platform_device_count={args.simulated_cpu_devices_count}"

    if args.model_size not in MODEL_PARAMS:
        raise ValueError(
            f"Unknown model size '{args.model_size}'. "
            f"Available: {list(MODEL_PARAMS.keys())}"
        )

    model_path = args.base_model_path

    if args.download_from_hub:
        try:
            from huggingface_hub import snapshot_download  # pylint: disable=import-outside-toplevel
        except ImportError as exc:
            raise ImportError("Install huggingface_hub: pip install huggingface_hub") from exc
        # If tmpdir is on a large scratch disk, put the model download on the
        # same disk (sibling directory) rather than ~/.cache (may be too small).
        repo_local_name = model_path.split("/")[-1] + "_hf"
        if args.tmpdir:
            local_dir = str(pathlib.Path(args.tmpdir).parent / repo_local_name)
        else:
            local_dir = None  # fall back to HF cache default
        max_logging.log(f"Downloading '{model_path}' from HuggingFace Hub to {local_dir or 'HF cache'}...")
        model_path = snapshot_download(
            repo_id=model_path,
            local_dir=local_dir,
            ignore_patterns=["*.pt", "*.bin"],
        )
        max_logging.log(f"Downloaded to: {model_path}")

    # Resolve tmpdir: use the user-supplied path, auto-create a temp dir if
    # --streaming was requested without --tmpdir, or None for in-RAM mode.
    tmpdir: str | None = None
    _auto_tmpdir = False
    if args.tmpdir:
        tmpdir = args.tmpdir
        max_logging.log(f"Streaming mode: converted tensors will be stored in {tmpdir!r}")
    elif args.streaming:
        tmpdir = tempfile.mkdtemp(prefix="mimo_convert_")
        _auto_tmpdir = True
        max_logging.log(
            f"Streaming mode (auto tmpdir): converted tensors will be stored in {tmpdir!r}. "
            "This directory will be removed after the checkpoint is saved."
        )
    else:
        max_logging.log(
            "In-RAM mode: all shards will be loaded at once (~970 GB peak RAM). "
            "Use --tmpdir <path> or --streaming to enable low-RAM streaming mode."
        )

    keep_fp8 = getattr(args, "keep_fp8", False)
    if keep_fp8 and not args.streaming_save:
        raise ValueError("--keep_fp8 requires --streaming_save (FP8 stacking is only implemented in the streaming-save path).")

    if args.streaming_save:
        # Streaming mode: convert + save one decoder layer at a time.
        # Peak RAM ≈ one MoE layer (~50 GB) regardless of total model size.
        # No --tmpdir needed; use --streaming_save instead of --tmpdir.
        params = MODEL_PARAMS[args.model_size]
        max_logging.log(
            f"Streaming-save mode: converting and saving layer by layer "
            f"(peak RAM ~50 GB, keep_fp8={keep_fp8}). Output: {args.maxtext_model_path}"
        )
        convert_and_save_streaming(model_path, args.maxtext_model_path, params, keep_fp8=keep_fp8)
        max_logging.log("Streaming conversion + save complete.")
    elif getattr(args, "resume_from_tmpdir", False):
        # Skip the ~53-min conversion phase and restore memmaps from disk.
        if not tmpdir:
            raise ValueError("--resume_from_tmpdir requires --tmpdir <path>")
        max_logging.log(f"Resuming: restoring flat weight dict from {tmpdir!r} (skipping conversion)")
        _, flat = _MemmapStore.from_dir(tmpdir)
        max_logging.log(f"Saving MaxText checkpoint to: {args.maxtext_model_path}")
        _save_zarr_direct(args.maxtext_model_path, flat)
        max_logging.log("Checkpoint saved successfully.")
    else:
        params = MODEL_PARAMS[args.model_size]
        max_logging.log(f"Starting conversion for MiMo-V2-Flash ({args.model_size})")
        flat = convert_hf_to_maxtext(model_path, params, tmpdir=tmpdir)
        max_logging.log(f"Saving MaxText checkpoint to: {args.maxtext_model_path}")
        _save_zarr_direct(args.maxtext_model_path, flat)
        max_logging.log("Checkpoint saved successfully.")

    if _auto_tmpdir and tmpdir:
        max_logging.log(f"Removing auto tmpdir: {tmpdir}")
        shutil.rmtree(tmpdir, ignore_errors=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert HuggingFace MiMo-V2-Flash weights to MaxText Orbax checkpoint."
    )
    parser.add_argument(
        "--base_model_path",
        type=str,
        required=True,
        help="Path to a local HF checkpoint directory, or a HF Hub repo id (with --download_from_hub).",
    )
    parser.add_argument(
        "--maxtext_model_path",
        type=str,
        required=True,
        help="Output path for the MaxText Orbax checkpoint (local pathname or gs:// URI).",
    )
    parser.add_argument(
        "--model_size",
        type=str,
        default="mimo-v2-flash",
        choices=list(MODEL_PARAMS.keys()),
        help="Model variant to convert: 'mimo-v2-flash' (309B) or 'mimo-v2-5-pro' (1.02T, fused_qkv).",
    )
    parser.add_argument(
        "--download_from_hub",
        action="store_true",
        default=False,
        help="If set, treat --base_model_path as a HuggingFace Hub repo id and download first.",
    )
    # ---- Memory / streaming options ----
    parser.add_argument(
        "--tmpdir",
        type=str,
        default=None,
        help=(
            "Scratch directory for memory-mapped intermediate files (streaming / low-RAM mode). "
            "Converted arrays are written to disk here layer by layer so that peak RAM is "
            "bounded to ~25 GB regardless of model size. "
            "Requires ~650 GB of free space (bfloat16) or ~1.24 TB (float32 fallback). "
            "Recommended for v6e-1 and other memory-constrained VMs. "
            "Example: --tmpdir /mnt/scratch/mimo_tmp"
        ),
    )
    parser.add_argument(
        "--streaming",
        action="store_true",
        default=False,
        help=(
            "Enable streaming / low-RAM mode with an automatically created tmpdir "
            "(deleted after the checkpoint is saved). "
            "Equivalent to --tmpdir <auto> but does not require specifying a path. "
            "The auto tmpdir is placed in the system's default temp directory."
        ),
    )
    parser.add_argument(
        "--simulated_cpu_devices_count",
        type=int,
        default=1,
        help=(
            "Number of CPU devices to simulate for JAX sharding during checkpoint save. "
            "Use 1 (default) for low-RAM machines such as v6e-1; "
            "higher values increase RAM usage proportionally."
        ),
    )
    parser.add_argument(
        "--use_ocdbt",
        type=lambda s: s.lower() not in ("false", "0", "no"),
        default=False,
        help=(
            "Use OCDBT format for Orbax checkpoint (default: False). "
            "OCDBT consolidates all arrays into large data files which requires loading "
            "all weights into RAM at save time (~576 GB on a v6e-1). "
            "Disable OCDBT (default) to use standard zarr, which writes arrays "
            "one at a time and stays within the 18 GB RAM budget."
        ),
    )
    parser.add_argument(
        "--use_zarr3",
        type=lambda s: s.lower() not in ("false", "0", "no"),
        default=False,
        help="Use Zarr3 format for Orbax checkpoint (default: False).",
    )
    parser.add_argument(
        "--resume_from_tmpdir",
        action="store_true",
        default=False,
        help=(
            "Skip the layer-conversion phase and restore memmaps from --tmpdir directly. "
            "Requires a previous run with --tmpdir that completed conversion (shapes.json present). "
            "Useful for retrying a failed checkpoint save without repeating the ~53 min conversion."
        ),
    )
    parser.add_argument(
        "--streaming_save",
        action="store_true",
        default=False,
        help=(
            "Convert and save one decoder layer at a time, freeing each layer's buffers before "
            "moving to the next.  Peak RAM is bounded to approximately one MoE layer (~50 GB) "
            "regardless of total model size.  No --tmpdir is required in this mode.  "
            "Use this flag when the full model is too large to hold in RAM simultaneously "
            "(e.g. 309B MiMo-V2-Flash on a 708 GB host)."
        ),
    )
    parser.add_argument(
        "--keep_fp8",
        action="store_true",
        default=False,
        help=(
            "Keep MoE expert weights in float8_e4m3fn format in the output checkpoint "
            "instead of dequantizing to bfloat16.  Per-block scale_inv tensors are stored "
            "alongside as separate float32 arrays (e.g. mlp.wi_0_scale_inv).  "
            "Non-expert weights (attention projections, dense MLP) are still dequantized to "
            "bfloat16 as normal.  Requires --streaming_save.  "
            "The resulting checkpoint is ~155 GB (FP8 experts) vs ~384 GB (BF16)."
        ),
    )

    parsed_args = parser.parse_args()
    main(parsed_args)
