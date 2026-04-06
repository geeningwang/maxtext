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

r"""Distributed HuggingFace MiMo-V2-Flash → MaxText zarr2 checkpoint conversion.

Splits the 48 decoder layers across N workers so each worker only reads and
writes its own share of layers.  Worker 0 additionally writes the three global
parameter arrays (embeddings, decoder_norm, lm_head).

Two-phase execution
-------------------

Phase 1 — run on ALL workers simultaneously (replace $WORKER_RANK per worker,
or use --auto_rank which detects the rank from the hostname/metadata):

  gcloud compute tpus tpu-vm ssh jingnw-node --worker=all --zone=us-east5-b --internal-ip \
    --command="cd ~/maxtext && nohup ~/maxtext/maxtext_venv/bin/python3 -u \
      -m maxtext.checkpoint_conversion.standalone_scripts.convert_mimo_v2_flash_distributed \
      --base_model_path /tmp/mimo-hf-gcs/hf-model \
      --maxtext_model_path gs://jingnw-mimo-v2-flash-us-east5/mimo-v2-flash-fixed \
      --num_workers 8 \
      --auto_rank \
      > /tmp/convert_dist.log 2>&1 &"

Phase 2 — finalize on worker 0 only (after ALL phase-1 jobs complete):

  gcloud compute tpus tpu-vm ssh jingnw-node --worker=0 --zone=us-east5-b --internal-ip \
    --command="cd ~/maxtext && ~/maxtext/maxtext_venv/bin/python3 -u \
      -m maxtext.checkpoint_conversion.standalone_scripts.convert_mimo_v2_flash_distributed \
      --base_model_path /tmp/mimo-hf-gcs/hf-model \
      --maxtext_model_path gs://jingnw-mimo-v2-flash-us-east5/mimo-v2-flash-fixed \
      --num_workers 8 \
      --finalize"

Layer assignment with 8 workers (48 layers total → 6 layers each):
  Worker 0: layers 0–5  + global weights (embed, norm, lm_head)
  Worker 1: layers 6–11
  Worker 2: layers 12–17
  Worker 3: layers 18–23
  Worker 4: layers 24–29
  Worker 5: layers 30–35
  Worker 6: layers 36–41
  Worker 7: layers 42–47

Rank auto-detection order (--auto_rank):
  1. $TPU_WORKER_ID environment variable
  2. Last '-N' numeric suffix of $(hostname)
  3. GCE instance-name metadata last '-N' numeric suffix
  Falls back to --worker_rank if none resolve.
"""

import argparse
import gc
import json
import os
import pathlib
import shutil
import socket
import time
import urllib.error
import urllib.request

import numpy as np

from maxtext.utils import max_logging

# Import shared conversion primitives from the single-worker script.
from maxtext.checkpoint_conversion.standalone_scripts.convert_mimo_v2_flash import (
    MODEL_PARAMS,
    convert_hf_to_maxtext,
    _write_one_zarr_array,
    _write_checkpoint_metadata,
)


# ---------------------------------------------------------------------------
# Rank auto-detection
# ---------------------------------------------------------------------------

def _detect_rank() -> int | None:
    """Try to auto-detect TPU worker rank from env and metadata. Returns None on failure."""
    # 1. TPU_WORKER_ID env var (set by some launchers)
    worker_id = os.environ.get("TPU_WORKER_ID")
    if worker_id is not None:
        try:
            return int(worker_id)
        except ValueError:
            pass

    # 2. Last numeric segment of hostname  (e.g. "jingnw-node-3" → 3)
    hostname = socket.gethostname()
    parts = hostname.rstrip().split("-")
    if parts:
        try:
            return int(parts[-1])
        except ValueError:
            pass

    # 3. GCE instance-name metadata last numeric segment
    try:
        req = urllib.request.Request(
            "http://metadata.google.internal/computeMetadata/v1/instance/name",
            headers={"Metadata-Flavor": "Google"},
        )
        name = urllib.request.urlopen(req, timeout=2).read().decode().strip()
        seg = name.split("-")[-1]
        return int(seg)
    except (urllib.error.URLError, ValueError, OSError):
        pass

    return None


# ---------------------------------------------------------------------------
# Layer-range helpers
# ---------------------------------------------------------------------------

def _layer_range_for_worker(worker_rank: int, num_workers: int, num_layers: int) -> tuple[int, int]:
    """Return (start_inclusive, end_exclusive) layer indices for this worker."""
    base = num_layers // num_workers
    remainder = num_layers % num_workers
    start = worker_rank * base + min(worker_rank, remainder)
    end = start + base + (1 if worker_rank < remainder else 0)
    return start, end


# ---------------------------------------------------------------------------
# Phase 1 — worker convert+save
# ---------------------------------------------------------------------------

def convert_and_save_worker(
    base_model_path: str,
    maxtext_model_path: str,
    params: dict,
    worker_rank: int,
    num_workers: int,
    step: int = 0,
    explicit_layer_range: "tuple[int, int] | None" = None,
    explicit_skip_global: "bool | None" = None,
) -> None:
    """Convert and save the layers owned by this worker.

    Worker 0 also saves the global weights (embed_tokens, decoder_norm, lm_head)
    and the zarr ``step`` scalar.  All workers write a ``partial_meta_{rank}.json``
    file under the checkpoint ``items`` directory; phase 2 merges these into the
    final ``_METADATA``.
    """
    import zarr  # pylint: disable=import-outside-toplevel
    import numcodecs  # pylint: disable=import-outside-toplevel

    num_layers = params["num_hidden_layers"]
    if explicit_layer_range is not None:
        layer_start, layer_end = explicit_layer_range
    else:
        layer_start, layer_end = _layer_range_for_worker(worker_rank, num_workers, num_layers)
    is_global_writer = explicit_skip_global is False or (explicit_skip_global is None and worker_rank == 0)
    max_logging.log(
        f"[worker {worker_rank}/{num_workers-1}] Assigned layers {layer_start}–{layer_end-1} "
        f"(+ global weights: {is_global_writer})"
    )
    print(
        f"[convert] [worker {worker_rank}/{num_workers-1}] layers {layer_start}–{layer_end-1}",
        flush=True,
    )

    root = pathlib.Path(maxtext_model_path)
    step_dir = root / str(step)
    items_dir = step_dir / "items"
    items_dir.mkdir(parents=True, exist_ok=True)

    compressor = numcodecs.Zstd(level=1)

    # Worker 0 writes the `step` scalar (only one writer needed).
    if worker_rank == 0:
        z_step = zarr.open_array(
            str(items_dir / "step"), mode="w",
            shape=(), dtype="<i8",
            compressor=compressor,
            dimension_separator=".",
        )
        z_step[()] = step
        del z_step

    tree_meta: dict = {}
    arrays_written = [0]

    def _on_layer_complete(layer_idx: int, layer_flat: dict) -> None:
        _t = time.time()
        total = len(layer_flat)
        for idx, (key, arr) in enumerate(sorted(layer_flat.items())):
            print(
                f"[convert] [worker {worker_rank}] saving layer {layer_idx} "
                f"array {idx+1}/{total}: {key}  shape={arr.shape}",
                flush=True,
            )
            tree_meta.update(_write_one_zarr_array(items_dir, key, arr, compressor))
            arrays_written[0] += 1
        print(
            f"[convert] [worker {worker_rank}] Saved layer {layer_idx} "
            f"({total} arrays in {time.time()-_t:.1f}s, total so far: {arrays_written[0]})",
            flush=True,
        )

    # Convert assigned layers (+ global weights for rank 0).
    flat = convert_hf_to_maxtext(
        base_model_path,
        params,
        tmpdir=None,
        on_layer_complete=_on_layer_complete,
        layer_range=(layer_start, layer_end),
        skip_global_weights=(not is_global_writer),
    )

    # Write global weights returned in flat (rank 0 only; others return empty).
    for key, arr in sorted(flat.items()):
        print(
            f"[convert] [worker {worker_rank}] saving global: {key}  shape={arr.shape}",
            flush=True,
        )
        tree_meta.update(_write_one_zarr_array(items_dir, key, arr, compressor))
        arrays_written[0] += 1
    del flat
    gc.collect()

    # Write this worker's partial metadata file — phase 2 will merge all of them.
    partial_meta_path = items_dir / f"partial_meta_{worker_rank}.json"
    partial_meta_path.write_text(json.dumps(tree_meta))
    max_logging.log(
        f"[worker {worker_rank}] Wrote {arrays_written[0]} arrays + "
        f"partial_meta_{worker_rank}.json"
    )
    print(
        f"[convert] [worker {worker_rank}] Done. {arrays_written[0]} arrays written.",
        flush=True,
    )


def scan_and_finalize_checkpoint(
    maxtext_model_path: str,
    step: int = 0,
) -> None:
    """Rebuild _METADATA by scanning all zarr arrays written to the items dir.

    Use this when partial_meta_*.json files are unavailable (e.g. worker 0 ran
    the single-worker script) or when you want a full consistency pass.
    Every subdirectory under items/ that contains a .zarray file is treated as
    a parameter array.  The key is derived from the directory name by stripping
    the leading 'params.params.' prefix.
    """
    root = pathlib.Path(maxtext_model_path)
    step_dir = root / str(step)
    items_dir = step_dir / "items"

    if not items_dir.exists():
        raise FileNotFoundError(f"items dir not found: {items_dir}")

    tree_meta: dict = {}
    scanned = 0
    for zarray_file in sorted(items_dir.rglob(".zarray")):
        zarr_dir = zarray_file.parent
        zarr_name = zarr_dir.relative_to(items_dir).as_posix()  # e.g. params.params.decoder.layers.0.mlp.wo
        if not zarr_name.startswith("params.params."):
            continue
        key = zarr_name[len("params.params."):]  # e.g. decoder.layers.0.mlp.wo
        key_parts = ["params", "params"] + key.split(".")
        tree_meta[str(tuple(key_parts))] = {
            "key_metadata": [{"key": p, "key_type": 2} for p in key_parts],
            "value_metadata": {"value_type": "np.ndarray", "skip_deserialize": False},
        }
        scanned += 1

    max_logging.log(f"[scan_finalize] Scanned {scanned} zarr arrays from {items_dir}")
    print(f"[convert] [scan_finalize] Found {scanned} arrays, writing _METADATA...", flush=True)

    init_ts = int(step_dir.stat().st_mtime * 1e9) if step_dir.exists() else time.time_ns()
    _write_checkpoint_metadata(step_dir, items_dir, tree_meta, init_ts, scanned)
    max_logging.log(f"[scan_finalize] Checkpoint finalised at {step_dir}")
    print(f"[convert] [scan_finalize] Done. Checkpoint at {step_dir}", flush=True)


# ---------------------------------------------------------------------------
# Phase 2 — finalize (worker 0 only, after all workers complete)
# ---------------------------------------------------------------------------

def finalize_checkpoint(
    maxtext_model_path: str,
    num_workers: int,
    step: int = 0,
) -> None:
    """Merge partial_meta_*.json files and write final _METADATA / commit_success.txt."""
    root = pathlib.Path(maxtext_model_path)
    step_dir = root / str(step)
    items_dir = step_dir / "items"

    missing = []
    merged_tree_meta: dict = {}
    for rank in range(num_workers):
        p = items_dir / f"partial_meta_{rank}.json"
        if not p.exists():
            missing.append(rank)
        else:
            merged_tree_meta.update(json.loads(p.read_text()))

    if missing:
        raise FileNotFoundError(
            f"partial_meta files missing for workers: {missing}. "
            "Ensure all phase-1 workers completed successfully before running --finalize."
        )

    total = len(merged_tree_meta)
    max_logging.log(f"[finalize] Merged {total} tree_meta entries from {num_workers} workers.")
    print(f"[convert] [finalize] Merged {total} arrays from {num_workers} workers.", flush=True)

    init_ts = int(step_dir.stat().st_mtime * 1e9) if step_dir.exists() else time.time_ns()
    _write_checkpoint_metadata(step_dir, items_dir, merged_tree_meta, init_ts, total)
    max_logging.log(f"[finalize] Checkpoint finalised at {step_dir}")
    print(f"[convert] [finalize] Done. Checkpoint at {step_dir}", flush=True)

    # Clean up partial metadata files.
    for rank in range(num_workers):
        p = items_dir / f"partial_meta_{rank}.json"
        try:
            p.unlink()
        except OSError:
            pass


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main(args) -> None:
    os.environ["JAX_PLATFORMS"] = "cpu"

    if args.model_size not in MODEL_PARAMS:
        raise ValueError(f"Unknown model size '{args.model_size}'. Available: {list(MODEL_PARAMS)}")

    params = MODEL_PARAMS[args.model_size]

    if args.finalize:
        finalize_checkpoint(args.maxtext_model_path, args.num_workers)
        return

    if args.scan_and_finalize:
        scan_and_finalize_checkpoint(args.maxtext_model_path)
        return

    # Resolve worker rank.
    worker_rank = args.worker_rank
    if args.auto_rank:
        detected = _detect_rank()
        if detected is None:
            if worker_rank is None:
                raise ValueError(
                    "--auto_rank could not detect rank and --worker_rank was not provided."
                )
            max_logging.log(f"[rank] auto-detect failed; using --worker_rank {worker_rank}")
        else:
            if worker_rank is not None and worker_rank != detected:
                max_logging.log(
                    f"[rank] WARNING: --worker_rank {worker_rank} overrides auto-detected rank {detected}"
                )
            else:
                worker_rank = detected
    if worker_rank is None:
        raise ValueError("Provide --worker_rank or --auto_rank.")

    # Explicit layer_start/layer_end override the automatic rank-based split.
    if args.layer_start is not None and args.layer_end is not None:
        explicit_range = (args.layer_start, args.layer_end)
        explicit_skip_global = args.skip_global_weights
        max_logging.log(
            f"[worker {worker_rank}] Explicit layer range: {args.layer_start}–{args.layer_end-1} "
            f"skip_global={explicit_skip_global}"
        )
    else:
        explicit_range = None
        explicit_skip_global = (worker_rank != 0)  # rank-based default: only rank 0 writes globals

    convert_and_save_worker(
        base_model_path=args.base_model_path,
        maxtext_model_path=args.maxtext_model_path,
        params=params,
        worker_rank=worker_rank,
        num_workers=args.num_workers,
        explicit_layer_range=explicit_range,
        explicit_skip_global=explicit_skip_global,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Distributed MiMo-V2-Flash HF→MaxText zarr2 conversion across N TPU workers."
    )
    parser.add_argument(
        "--base_model_path",
        type=str,
        required=True,
        help="Path to local HF checkpoint directory (must be readable on every worker).",
    )
    parser.add_argument(
        "--maxtext_model_path",
        type=str,
        required=True,
        help="Output path for the MaxText zarr2 checkpoint (gs:// URI or local path).",
    )
    parser.add_argument(
        "--model_size",
        type=str,
        default="mimo-v2-flash",
        choices=list(MODEL_PARAMS.keys()),
        help="Model size identifier.",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=8,
        help="Total number of parallel workers (should match the number of TPU VMs).",
    )
    parser.add_argument(
        "--worker_rank",
        type=int,
        default=None,
        help="0-indexed rank of this worker. Required unless --auto_rank is set.",
    )
    parser.add_argument(
        "--auto_rank",
        action="store_true",
        default=False,
        help=(
            "Auto-detect worker rank from $TPU_WORKER_ID, hostname suffix, or GCE metadata. "
            "Falls back to --worker_rank if detection fails."
        ),
    )
    parser.add_argument(
        "--finalize",
        action="store_true",
        default=False,
        help=(
            "Phase 2: merge partial_meta_*.json files and write final _METADATA / "
            "commit_success.txt.  Run on worker 0 only AFTER all phase-1 jobs complete."
        ),
    )

    parser.add_argument(
        "--layer_start",
        type=int,
        default=None,
        help="Explicit start layer (inclusive). Overrides rank-based split when combined with --layer_end.",
    )
    parser.add_argument(
        "--layer_end",
        type=int,
        default=None,
        help="Explicit end layer (exclusive). Overrides rank-based split when combined with --layer_start.",
    )
    parser.add_argument(
        "--skip_global_weights",
        action="store_true",
        default=False,
        help="Skip writing embeddings/decoder_norm/lm_head. Use for non-rank-0 workers with explicit ranges.",
    )
    parser.add_argument(
        "--scan_and_finalize",
        action="store_true",
        default=False,
        help=(
            "Rebuild _METADATA by scanning all .zarray files in the items dir. "
            "Use when partial_meta files are not available (e.g. worker 0 ran "
            "the single-worker script). Safe to run at any time after all arrays are written."
        ),
    )

    parsed_args = parser.parse_args()
    main(parsed_args)
