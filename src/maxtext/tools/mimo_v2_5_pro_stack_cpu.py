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

"""CPU-based parallel stacking of MiMo-V2.5-Pro flat-per-layer checkpoint.

Reads the source zarr2 checkpoint produced by the Phase 4 HF→MaxText converter
(gs://jingnw-mimo-v2-5-pro-us-central1/mimo-v2-5-pro-fp8-ocdbt/0/items) and
writes a new zarr2 checkpoint with the 8-phase scan layout expected by
decoders.py when scan_layers=True.

No JAX, no TPU, no XLA compilation — pure numpy + gcsfs + zstd.

Run as a 4-worker Kubernetes Indexed Job (JOB_COMPLETION_INDEX=0..3) or with
an explicit --worker_index flag:

  python -m maxtext.tools.mimo_v2_5_pro_stack_cpu --worker_index 0

After all 4 workers complete, run the finalizer:

  python -m maxtext.tools.mimo_v2_5_pro_stack_cpu --finalize_only

Worker assignment:
  0: global params (embed, norm, lm_head, step) + Phase A (layer 0) + Phase B (layers 1-6)
  1: Phase C positions 0-3  (layers 7-10, 15-18, 23-26, 31-34, 39-42, 47-50)
  2: Phase C positions 4-7  (layers 11-14, 19-22, 27-30, 35-38, 43-46, 51-54)
  3: Phase D (layer 55) + Phase E (56-61) + Phase F (62) + Phase G (63-68) + Phase H (69)

Phase output key mapping (all under params.params.decoder.*):
  layers_a          — layer 0        (single, GA+dense)
  layers_b          — layers 1-6     (stacked [6, ...], SWA+MoE)
  layers_c.layers_0 — layers 7,15,23,31,39,47   (stacked [6, ...], GA+MoE)
  layers_c.layers_1 — layers 8,16,24,32,40,48   (stacked [6, ...], SWA+MoE)
  layers_c.layers_2 — layers 9,17,25,33,41,49   (stacked [6, ...])
  layers_c.layers_3 — layers 10,18,26,34,42,50  (stacked [6, ...])
  layers_c.layers_4 — layers 11,19,27,35,43,51  (stacked [6, ...])
  layers_c.layers_5 — layers 12,20,28,36,44,52  (stacked [6, ...])
  layers_c.layers_6 — layers 13,21,29,37,45,53  (stacked [6, ...])
  layers_c.layers_7 — layers 14,22,30,38,46,54  (stacked [6, ...])
  layers_d          — layer 55       (single, GA+MoE)
  layers_e          — layers 56-61   (stacked [6, ...], SWA+MoE)
  layers_f          — layer 62       (single, GA+MoE)
  layers_g          — layers 63-68   (stacked [6, ...], SWA+MoE)
  layers_h          — layer 69       (single, GA+MoE)
"""

import argparse
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor

import gcsfs
import numcodecs


# ---------------------------------------------------------------------------
# GCS checkpoint paths
# ---------------------------------------------------------------------------

_SRC_BUCKET = "jingnw-mimo-v2-5-pro-us-central1"
_SRC_ITEMS = f"{_SRC_BUCKET}/mimo-v2-5-pro-fp8-ocdbt/0/items"
_DST_ITEMS = f"{_SRC_BUCKET}/mimo-v2-5-pro-fp8-ocdbt-stacked-cpu/0/items"
_DST_STEP = f"{_SRC_BUCKET}/mimo-v2-5-pro-fp8-ocdbt-stacked-cpu/0"
_DST_ROOT = f"{_SRC_BUCKET}/mimo-v2-5-pro-fp8-ocdbt-stacked-cpu"


# ---------------------------------------------------------------------------
# Phase group constants — must match decoders.py 8-phase layout for 70 layers.
# ---------------------------------------------------------------------------

_PHASE_B_INDICES = list(range(1, 7))  # [1, 2, 3, 4, 5, 6]

_PHASE_C_POSITIONS = [
    [7,  15, 23, 31, 39, 47],   # pos 0 (GA+MoE)
    [8,  16, 24, 32, 40, 48],   # pos 1 (SWA+MoE)
    [9,  17, 25, 33, 41, 49],   # pos 2
    [10, 18, 26, 34, 42, 50],   # pos 3
    [11, 19, 27, 35, 43, 51],   # pos 4
    [12, 20, 28, 36, 44, 52],   # pos 5
    [13, 21, 29, 37, 45, 53],   # pos 6
    [14, 22, 30, 38, 46, 54],   # pos 7
]

_PHASE_E_INDICES = list(range(56, 62))  # [56, 57, 58, 59, 60, 61]
_PHASE_G_INDICES = list(range(63, 69))  # [63, 64, 65, 66, 67, 68]

# Number of parallel leaf-array processing threads per phase group.
# Peak RAM per thread: stacked_buf (9.66 GB) + 1 compressed chunk (~1.4 GB)
# + 1 decompressed chunk (1.61 GB) ≈ 13 GB. With 4 threads: ~52 GB peak,
# well within the 110 Gi pod limit.
_N_LEAF_THREADS = 4


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def _log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ---------------------------------------------------------------------------
# Low-level GCS zarr helpers
# ---------------------------------------------------------------------------

def _get_layer_zarr_suffixes(
    fs: gcsfs.GCSFileSystem,
    src_items: str,
    layer_idx: int,
) -> list[str]:
    """Return sorted list of zarr-array suffixes (keys after 'layers.{i}.') for a layer.

    Uses a glob on .zarray sentinel files so we enumerate exactly the zarr
    arrays that exist — no assumptions about parameter names.
    """
    pattern = f"{src_items}/params.params.decoder.layers.{layer_idx}.*/.zarray"
    zarray_paths = fs.glob(pattern)
    prefix = f"{src_items}/params.params.decoder.layers.{layer_idx}."
    suffixes = []
    for p in zarray_paths:
        rel = p[len(prefix):]         # e.g. "mlp.experts.w1/.zarray"
        suffix = rel[: -len("/.zarray")]  # e.g. "mlp.experts.w1"
        suffixes.append(suffix)
    return sorted(suffixes)


def _copy_single_zarr(
    fs: gcsfs.GCSFileSystem,
    src_zarr: str,
    dst_zarr: str,
) -> None:
    """Copy all objects (.zarray + chunk files) from src zarr dir to dst zarr dir."""
    objects = fs.ls(src_zarr, detail=False)
    for obj in objects:
        obj_name = obj.split("/")[-1]
        fs.copy(obj, f"{dst_zarr}/{obj_name}")


def _copy_layer(
    fs: gcsfs.GCSFileSystem,
    src_items: str,
    dst_items: str,
    layer_idx: int,
    dst_phase_key: str,
) -> None:
    """Copy all zarr arrays from a single source layer to a destination phase key.

    dst_phase_key is the suffix after 'params.params.decoder.' in the output,
    e.g. 'layers_a', 'layers_d', etc.
    """
    suffixes = _get_layer_zarr_suffixes(fs, src_items, layer_idx)
    _log(f"  copy layer {layer_idx} → {dst_phase_key}: {len(suffixes)} arrays")
    t0 = time.time()
    for suffix in suffixes:
        src_zarr = f"{src_items}/params.params.decoder.layers.{layer_idx}.{suffix}"
        dst_zarr = f"{dst_items}/params.params.decoder.{dst_phase_key}.{suffix}"
        _copy_single_zarr(fs, src_zarr, dst_zarr)
    _log(f"  done {dst_phase_key} in {time.time() - t0:.1f}s")


def _stack_group(
    fs: gcsfs.GCSFileSystem,
    src_items: str,
    src_layers: list[int],
    dst_items: str,
    dst_key: str,
) -> None:
    """Stack N source layers into one stacked zarr phase group.

    Correctness argument for raw-byte concatenation:
      Each zarr2 array in the source checkpoint is written as a single chunk
      (chunks == shape) in C (row-major) order.  Stacking N arrays of shape S
      along axis 0 produces a (N, *S) array whose raw buffer is exactly the
      concatenation of the N source buffers.  We exploit this to avoid numpy
      dtype interpretation entirely — the bytes are compressed, concatenated,
      and re-compressed without any float8/bfloat16 parsing.
    """
    n = len(src_layers)
    suffixes = _get_layer_zarr_suffixes(fs, src_items, src_layers[0])
    _log(f"  stack layers {src_layers} → {dst_key}: {len(suffixes)} leaves × {n} layers")
    t0 = time.time()
    done_count = [0]

    compressor = numcodecs.Zstd(level=1)

    def _stack_one(suffix: str) -> None:
        src0_zarr = f"{src_items}/params.params.decoder.layers.{src_layers[0]}.{suffix}"
        meta = json.loads(fs.cat(f"{src0_zarr}/.zarray"))
        shape = meta["shape"]
        ndim = len(shape)
        chunk_name = ".".join(["0"] * ndim) if ndim > 0 else "0"

        # Read and decompress chunks one at a time — never hold more than
        # 1 compressed + 1 raw chunk alongside stacked_buf.
        #
        # Previous OOM (attempt 2, 4 threads, 110 GB RSS) had two bugs:
        #   (a) inner ThreadPoolExecutor pre-fetched all N compressed chunks
        #       into a list (~9 GB held per thread while stacked_buf grew)
        #   (b) bytes(stacked_buf) made a full 9.66 GB copy for encode()
        # Combined: ~28 GB/thread × 4 = 113 GB → OOM at 110 Gi limit.
        #
        # Fix: sequential GCS reads (no list), memoryview for encode (no copy).
        # Peak per thread: stacked_buf (9.66 GB) + compressed (~1.5 GB)
        #                  + raw (1.61 GB) ≈ 13 GB → 4 threads ≈ 52 GB total.
        stacked_buf: bytearray | None = None
        chunk_bytes = 0
        for i, layer_idx in enumerate(src_layers):
            path = (
                f"{src_items}/params.params.decoder.layers.{layer_idx}.{suffix}"
                f"/{chunk_name}"
            )
            compressed = fs.cat(path)
            raw = numcodecs.Zstd().decode(compressed)
            del compressed
            if stacked_buf is None:
                chunk_bytes = len(raw)
                stacked_buf = bytearray(n * chunk_bytes)
            stacked_buf[i * chunk_bytes:(i + 1) * chunk_bytes] = raw
            del raw
        # memoryview avoids a 9.66 GB copy when passing bytearray to encode().
        stacked_compressed = compressor.encode(memoryview(stacked_buf))
        del stacked_buf

        # Write stacked chunk.
        new_shape = [n] + list(shape)
        new_chunk_name = ".".join(["0"] * len(new_shape))
        dst_zarr = f"{dst_items}/params.params.decoder.{dst_key}.{suffix}"
        fs.pipe(f"{dst_zarr}/{new_chunk_name}", stacked_compressed)

        # Write updated .zarray: prepend stack axis to shape and chunks; strip
        # the "checksum" field from compressor (TensorStore rejects it).
        new_meta = dict(meta)
        new_meta["shape"] = new_shape
        new_meta["chunks"] = new_shape
        if isinstance(new_meta.get("compressor"), dict):
            new_meta["compressor"].pop("checksum", None)
        fs.pipe(f"{dst_zarr}/.zarray", json.dumps(new_meta).encode())

        done_count[0] += 1
        if done_count[0] % 5 == 0 or done_count[0] == len(suffixes):
            _log(f"    {dst_key}: {done_count[0]}/{len(suffixes)} arrays done")

    with ThreadPoolExecutor(max_workers=_N_LEAF_THREADS) as ex:
        futures = [ex.submit(_stack_one, s) for s in suffixes]
        for fut in futures:
            fut.result()  # re-raise any exception from the thread

    _log(f"  done {dst_key} in {time.time() - t0:.1f}s")


def _copy_global_params(
    fs: gcsfs.GCSFileSystem,
    src_items: str,
    dst_items: str,
) -> None:
    """Copy non-decoder-layer zarr arrays (embed, norm, lm_head, step) verbatim."""
    _log("copying global params...")
    t0 = time.time()
    all_entries = fs.ls(src_items, detail=False)
    copied = 0
    for entry in all_entries:
        name = entry.split("/")[-1]
        if name.startswith("params.params.decoder.layers."):
            continue
        # Skip checkpoint metadata files — the finalizer writes these fresh.
        if name in ("_METADATA", "commit_success.txt"):
            continue
        # Copy every object inside this zarr dir (or scalar dir).
        objects = fs.ls(entry, detail=False)
        for obj in objects:
            obj_name = obj.split("/")[-1]
            fs.copy(obj, f"{dst_items}/{name}/{obj_name}")
        _log(f"  copied global: {name}")
        copied += 1
    _log(f"done global params ({copied} entries) in {time.time() - t0:.1f}s")


# ---------------------------------------------------------------------------
# Worker functions
# ---------------------------------------------------------------------------

def worker_0(fs: gcsfs.GCSFileSystem) -> None:
    """Global params + Phase A (layer 0) + Phase B (layers 1-6)."""
    _log("=== Worker 0: global + Phase A + Phase B ===")
    t0 = time.time()
    _copy_global_params(fs, _SRC_ITEMS, _DST_ITEMS)
    _copy_layer(fs, _SRC_ITEMS, _DST_ITEMS, 0, "layers_a")
    _stack_group(fs, _SRC_ITEMS, _PHASE_B_INDICES, _DST_ITEMS, "layers_b")
    _log(f"=== Worker 0 complete in {time.time() - t0:.1f}s ===")


def worker_1(fs: gcsfs.GCSFileSystem) -> None:
    """Phase C positions 0-3."""
    _log("=== Worker 1: Phase C positions 0-3 ===")
    t0 = time.time()
    for pos in range(4):
        _stack_group(
            fs, _SRC_ITEMS, _PHASE_C_POSITIONS[pos], _DST_ITEMS,
            f"layers_c.layers_{pos}",
        )
    _log(f"=== Worker 1 complete in {time.time() - t0:.1f}s ===")


def worker_2(fs: gcsfs.GCSFileSystem) -> None:
    """Phase C positions 4-7."""
    _log("=== Worker 2: Phase C positions 4-7 ===")
    t0 = time.time()
    for pos in range(4, 8):
        _stack_group(
            fs, _SRC_ITEMS, _PHASE_C_POSITIONS[pos], _DST_ITEMS,
            f"layers_c.layers_{pos}",
        )
    _log(f"=== Worker 2 complete in {time.time() - t0:.1f}s ===")


def worker_3(fs: gcsfs.GCSFileSystem) -> None:
    """Phase D (55) + E (56-61) + F (62) + G (63-68) + H (69)."""
    _log("=== Worker 3: Phase D + E + F + G + H ===")
    t0 = time.time()
    _copy_layer(fs, _SRC_ITEMS, _DST_ITEMS, 55, "layers_d")
    _stack_group(fs, _SRC_ITEMS, _PHASE_E_INDICES, _DST_ITEMS, "layers_e")
    _copy_layer(fs, _SRC_ITEMS, _DST_ITEMS, 62, "layers_f")
    _stack_group(fs, _SRC_ITEMS, _PHASE_G_INDICES, _DST_ITEMS, "layers_g")
    _copy_layer(fs, _SRC_ITEMS, _DST_ITEMS, 69, "layers_h")
    _log(f"=== Worker 3 complete in {time.time() - t0:.1f}s ===")


# ---------------------------------------------------------------------------
# Finalizer
# ---------------------------------------------------------------------------

def _scan_dst_zarr_meta(fs: gcsfs.GCSFileSystem, dst_items: str) -> dict:
    """Reconstruct tree_meta for all zarr arrays written by workers."""
    try:
        children = fs.ls(dst_items, detail=False)
    except Exception:  # pylint: disable=broad-except
        return {}
    result = {}
    for child in children:
        zarr_name = child.rstrip("/").split("/")[-1]
        if not zarr_name.startswith("params.params."):
            continue
        key_parts = zarr_name.split(".")
        result[str(tuple(key_parts))] = {
            "key_metadata": [{"key": p, "key_type": 2} for p in key_parts],
            "value_metadata": {"value_type": "np.ndarray", "skip_deserialize": False},
        }
    return result


def finalize(fs: gcsfs.GCSFileSystem) -> None:
    """Write _METADATA + _CHECKPOINT_METADATA + commit_success.txt."""
    _log("=== Finalizer: scanning destination zarr arrays ===")
    t0 = time.time()
    tree_meta = _scan_dst_zarr_meta(fs, _DST_ITEMS)
    total = len(tree_meta)
    _log(f"Found {total} zarr arrays. Writing checkpoint metadata...")

    init_ts = time.time_ns()
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
    fs.pipe(f"{_DST_ITEMS}/_METADATA", json.dumps(metadata).encode())
    fs.pipe(
        f"{_DST_STEP}/_CHECKPOINT_METADATA",
        json.dumps({
            "item_handlers": {
                "items": (
                    "orbax.checkpoint._src.handlers"
                    ".pytree_checkpoint_handler.PyTreeCheckpointHandler"
                )
            },
            "metrics": {},
            "performance_metrics": {},
            "init_timestamp_nsecs": init_ts,
            "commit_timestamp_nsecs": time.time_ns(),
            "custom_metadata": {},
        }).encode(),
    )
    fs.pipe(f"{_DST_ITEMS}/commit_success.txt", b"")
    _log(
        f"=== Finalizer complete: {total} arrays in {time.time() - t0:.1f}s ==="
    )
    _log(f"Stacked checkpoint: gs://{_DST_ROOT}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="CPU-based parallel stacking of MiMo-V2.5-Pro checkpoint"
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--worker_index",
        type=int,
        choices=[0, 1, 2, 3],
        help=(
            "Worker index (0-3). Reads JOB_COMPLETION_INDEX env var if not set. "
            "Worker 0 also writes global params and Phases A+B; workers 1-3 write "
            "Phase C (split) and Phases D-H respectively."
        ),
    )
    group.add_argument(
        "--finalize_only",
        action="store_true",
        help=(
            "Write _METADATA + _CHECKPOINT_METADATA after all workers complete. "
            "Run this as a separate job after workers 0-3 finish."
        ),
    )
    args = parser.parse_args()

    # Also honour the Kubernetes env var so the YAML can omit explicit --worker_index.
    if not args.finalize_only and args.worker_index is None:
        idx_env = os.environ.get("JOB_COMPLETION_INDEX")
        if idx_env is None:
            parser.error("--worker_index is required (or set JOB_COMPLETION_INDEX env var)")
        args.worker_index = int(idx_env)

    fs = gcsfs.GCSFileSystem()
    _log(f"Source: gs://{_SRC_ITEMS}")
    _log(f"Dest:   gs://{_DST_ITEMS}")

    if args.finalize_only:
        finalize(fs)
    elif args.worker_index == 0:
        worker_0(fs)
    elif args.worker_index == 1:
        worker_1(fs)
    elif args.worker_index == 2:
        worker_2(fs)
    elif args.worker_index == 3:
        worker_3(fs)


if __name__ == "__main__":
    main()
