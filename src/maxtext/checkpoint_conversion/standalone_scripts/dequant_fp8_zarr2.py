#!/usr/bin/env python3
"""dequant_fp8_zarr2.py - Apply weight_scale_inv dequantization to a MaxText FP8 zarr2
checkpoint and write a corrected BF16 zarr2 checkpoint.

Background
----------
The checkpoint produced by convert_mimo_v2_flash.py --keep_fp8 stores each linear weight
as float8_e4m3fn with a companion *_scale_inv (float32) array.  When Orbax loads this
checkpoint into a plain BF16 model it:
  1. Casts float8_e4m3fn -> bfloat16 (preserving the fp8 float values, NOT the learned scale).
  2. Silently ignores *_scale_inv arrays (they are not part of the model param tree).

This produces the same garbled output as the pre-fix state of convert_mimo_v2_flash.py
(fixed in commit 4cd09732).  This script applies the same _apply_fp8_dequant logic at the
checkpoint level:

    dequant[i, j] = fp8_raw[i, j] * scale_inv[i // bm, j // bn]

where bm/bn are derived from the tensor and scale shapes.

Implementation note
-------------------
Zarr (v2 and v3) cannot parse the Orbax-specific dtype strings ("bfloat16",
"float8_e4m3fn").  This script bypasses zarr's dtype layer entirely: it reads
.zarray JSON metadata manually, reads compressed chunk files via gcsfs, decompresses
with zstandard, and interprets bytes directly via numpy / ml_dtypes.  Writes follow
the same format (zstd-compressed zarr2 chunks), preserving Orbax compatibility.

Usage
-----
    python dequant_fp8_zarr2.py \\
        --src_path gs://bucket/mimo-v2-flash-fp8-ocdbt/0/items \\
        --dst_path gs://bucket/mimo-v2-flash-bf16-zarr2/0/items
"""

import argparse
import itertools
import json
import subprocess
import time

import gcsfs
import ml_dtypes
import numpy as np
import zstandard as zstd

_BF16 = ml_dtypes.bfloat16
_FP8 = ml_dtypes.float8_e4m3fn

# Orbax writes these dtype strings in .zarray metadata; map them to numpy dtypes.
_DTYPE_STR_MAP = {
    "float8_e4m3fn": np.dtype(_FP8),
    "bfloat16": np.dtype(_BF16),
    "<f4": np.dtype(np.float32),
    "<f2": np.dtype(_BF16),
    "<i4": np.dtype(np.int32),
    "<i2": np.dtype(np.int16),
    "|u1": np.dtype(np.uint8),
    "<u4": np.dtype(np.uint32),
}


def _parse_dtype(dtype_str):
  if dtype_str in _DTYPE_STR_MAP:
    return _DTYPE_STR_MAP[dtype_str]
  return np.dtype(dtype_str)


def _dtype_to_str(dtype):
  """Return the dtype string that Orbax / zarr expect in .zarray."""
  if dtype == np.dtype(_BF16):
    return "bfloat16"
  if dtype == np.dtype(_FP8):
    return "float8_e4m3fn"
  return np.dtype(dtype).str


# ---------------------------------------------------------------------------
# Raw zarr-v2 I/O (bypass zarr Python library for dtype handling)
# ---------------------------------------------------------------------------

def _load_array(fs, array_gcs_path):
  """Load a zarr-v2 array from GCS without using zarr's dtype machinery.
  Returns (data_ndarray, zarray_metadata_dict).
  """
  meta_path = array_gcs_path.rstrip("/") + "/.zarray"
  meta = json.loads(fs.cat(meta_path))

  shape = meta["shape"]
  chunk_shape = meta["chunks"]
  dtype = _parse_dtype(meta["dtype"])
  dim_sep = meta.get("dimension_separator", ".")
  compressor = meta.get("compressor") or {}

  if not shape or any(s == 0 for s in shape):
    return np.zeros(shape, dtype=dtype), meta

  chunk_grid = [(s + c - 1) // c for s, c in zip(shape, chunk_shape)]
  out = np.zeros(shape, dtype=dtype)
  dctx = zstd.ZstdDecompressor()

  for coords in itertools.product(*[range(n) for n in chunk_grid]):
    fname = dim_sep.join(str(c) for c in coords)
    chunk_path = array_gcs_path.rstrip("/") + "/" + fname
    raw = fs.cat(chunk_path)
    if compressor.get("id") == "zstd":
      raw = dctx.decompress(raw)
    actual_shape = tuple(
        min(chunk_shape[i], shape[i] - coords[i] * chunk_shape[i])
        for i in range(len(shape))
    )
    chunk_data = np.frombuffer(raw, dtype=dtype).reshape(actual_shape)
    slices = tuple(
        slice(c * cs, c * cs + sz)
        for c, cs, sz in zip(coords, chunk_shape, actual_shape)
    )
    out[slices] = chunk_data

  return out, meta


def _save_array(fs, array_gcs_path, data, src_meta):
  """Write a numpy array as a zarr-v2 array to GCS using raw I/O."""
  chunk_shape = src_meta["chunks"]
  dim_sep = src_meta.get("dimension_separator", ".")
  compressor_cfg = src_meta.get("compressor") or {"id": "zstd", "level": 1}

  out_meta = dict(src_meta)
  out_meta["dtype"] = _dtype_to_str(data.dtype)
  out_meta["shape"] = list(data.shape)
  out_meta["chunks"] = [min(c, s) for c, s in zip(chunk_shape, data.shape)]

  meta_path = array_gcs_path.rstrip("/") + "/.zarray"
  with fs.open(meta_path, "wb") as f:
    f.write(json.dumps(out_meta).encode())

  cctx = zstd.ZstdCompressor(level=compressor_cfg.get("level", 1))
  shape = data.shape
  actual_chunks = out_meta["chunks"]
  chunk_grid = [(s + c - 1) // c for s, c in zip(shape, actual_chunks)]

  for coords in itertools.product(*[range(n) for n in chunk_grid]):
    slices = tuple(
        slice(c * cs, min((c + 1) * cs, s))
        for c, cs, s in zip(coords, actual_chunks, shape)
    )
    chunk_data = data[slices]
    compressed = cctx.compress(np.ascontiguousarray(chunk_data).tobytes())
    fname = dim_sep.join(str(c) for c in coords)
    chunk_path = array_gcs_path.rstrip("/") + "/" + fname
    with fs.open(chunk_path, "wb") as f:
      f.write(compressed)


# ---------------------------------------------------------------------------
# Dequantization helpers
# ---------------------------------------------------------------------------

def _expand_scale(scale, rows, cols):
  """Expand a block scale (sr, sc) to full kernel shape (rows, cols)."""
  sr, sc = scale.shape[-2], scale.shape[-1]
  if rows % sr != 0 or cols % sc != 0:
    raise ValueError(
        f"Scale shape {scale.shape[-2:]} does not evenly divide kernel dims ({rows}, {cols}): "
        f"bm={rows}/{sr}={rows/sr:.2f}, bn={cols}/{sc}={cols/sc:.2f}"
    )
  bm, bn = rows // sr, cols // sc
  return np.repeat(np.repeat(scale, bm, axis=-2), bn, axis=-1)


def _apply_scale(kernel, scale):
  """Dequantize kernel using scale and return a bfloat16 result.
  Handles:
    - 2D kernel (rows, cols) with 2D scale (sr, sc)
    - 3D attention (dim_in, heads, d_h) with 2D scale (sr, sc)
    - 3D MoE (experts, rows, cols) with 3D scale (experts, sr, sc)
  """
  w = kernel.astype(np.float32)
  s = scale.astype(np.float32)

  if w.ndim == 3 and s.ndim == 2:
    # Attention: reshape to 2D, apply scale, reshape back
    orig_shape = w.shape
    w2d = w.reshape(w.shape[0], w.shape[1] * w.shape[2])
    rows, cols = w2d.shape
    s_exp = _expand_scale(s, rows, cols)
    return (w2d * s_exp).reshape(orig_shape).astype(_BF16)

  # 2D or 3D MoE: operate on last 2 dims
  rows, cols = w.shape[-2], w.shape[-1]
  s_exp = _expand_scale(s, rows, cols)
  return (w * s_exp).astype(_BF16)


# ---------------------------------------------------------------------------
# Main conversion
# ---------------------------------------------------------------------------

def dequant_fp8_checkpoint(src_path, dst_path):
  """Read FP8 zarr2 checkpoint at src_path, apply scale_inv, write to dst_path."""
  src_path = src_path.rstrip("/")
  dst_path = dst_path.rstrip("/")
  print(f"Source : {src_path}", flush=True)
  print(f"Dest   : {dst_path}", flush=True)

  print("Listing GCS checkpoint entries ...", flush=True)
  result = subprocess.run(
      ["gsutil", "ls", src_path + "/"],
      capture_output=True, text=True, check=True,
  )
  lines = [l.strip() for l in result.stdout.strip().split("\n") if l.strip()]
  src_prefix = src_path + "/"

  array_names = []
  meta_files = []
  for line in lines:
    name = line.replace(src_prefix, "").rstrip("/")
    if not name:
      continue
    if name in ("_METADATA", "commit_success.txt"):
      meta_files.append(name)
    else:
      array_names.append(name)

  print(f"Found {len(array_names)} arrays, {len(meta_files)} metadata files", flush=True)

  scale_inv_set = set(n for n in array_names if n.endswith("_scale_inv"))
  kernel_to_scale = {}
  for sn in sorted(scale_inv_set):
    kernel_name = sn[: -len("_scale_inv")]
    if kernel_name in set(array_names):
      kernel_to_scale[kernel_name] = sn
    else:
      print(f"  WARNING: {sn!r} has no matching kernel {kernel_name!r}", flush=True)

  print(f"Scale pairs: {len(kernel_to_scale)}", flush=True)
  for k, v in sorted(kernel_to_scale.items())[:5]:
    print(f"  {k}\n    <- {v}", flush=True)
  if len(kernel_to_scale) > 5:
    print(f"  ... and {len(kernel_to_scale) - 5} more pairs", flush=True)

  fs = gcsfs.GCSFileSystem()
  t0 = time.monotonic()
  n_dequant = 0
  n_copied = 0
  n_skipped = 0

  for idx, name in enumerate(sorted(array_names)):
    elapsed = time.monotonic() - t0
    print(f"[{idx+1:4d}/{len(array_names)}] ({elapsed:6.0f}s) {name}", flush=True)

    if name in scale_inv_set:
      print("  -> skip (consumed as scale during dequant)", flush=True)
      n_skipped += 1
      continue

    data, src_meta = _load_array(fs, f"{src_path}/{name}")

    if name in kernel_to_scale:
      scale_name = kernel_to_scale[name]
      scale, _ = _load_array(fs, f"{src_path}/{scale_name}")
      print(
          f"  dtype={data.dtype} shape={data.shape} "
          f"| scale dtype={scale.dtype} shape={scale.shape}",
          flush=True,
      )
      data = _apply_scale(data, scale)
      print(f"  -> dequantized: dtype={data.dtype} shape={data.shape}", flush=True)
      n_dequant += 1
    else:
      n_copied += 1

    _save_array(fs, f"{dst_path}/{name}", data, src_meta)

  for mf in meta_files:
    print(f"Copying metadata: {mf}", flush=True)
    subprocess.run(
        ["gsutil", "cp", f"{src_prefix}{mf}", f"{dst_path}/{mf}"],
        check=True,
    )

  elapsed = time.monotonic() - t0
  print(
      f"\nDone!  dequantized={n_dequant}  copied={n_copied}  skipped={n_skipped}  "
      f"elapsed={elapsed:.0f}s ({elapsed/60:.1f} min)",
      flush=True,
  )


if __name__ == "__main__":
  parser = argparse.ArgumentParser(
      description="Apply FP8 weight_scale_inv dequantization to a zarr2 checkpoint."
  )
  parser.add_argument("--src_path", required=True)
  parser.add_argument("--dst_path", required=True)
  args = parser.parse_args()
  dequant_fp8_checkpoint(args.src_path, args.dst_path)
