#!/usr/bin/env python3
"""dequant_fp8_zarr2.py — Apply weight_scale_inv dequantization to a MaxText FP8 zarr2
checkpoint and write a corrected BF16 zarr2 checkpoint.

Background
----------
The checkpoint produced by convert_mimo_v2_flash.py --keep_fp8 stores each linear weight
as float8_e4m3fn with a companion *_scale_inv (float32) array.  When Orbax loads this
checkpoint into a plain BF16 model it:
  1. Casts float8_e4m3fn → bfloat16 (preserving the fp8 float values, NOT the learned scale).
  2. Silently ignores *_scale_inv arrays (they are not part of the model param tree).

This produces the same garbled output as the pre-fix state of convert_mimo_v2_flash.py
(fixed in commit 4cd09732).  This script applies the same _apply_fp8_dequant logic at the
checkpoint level:

    dequant[i, j] = fp8_raw[i, j] * scale_inv[i // bm, j // bn]

where bm/bn are derived from the tensor and scale shapes.

The output zarr2 checkpoint contains correct BF16 kernels (no scale_inv arrays) and can be
loaded with --no-checkpoint_use_ocdbt by any plain BF16 MaxText model.

Usage
-----
    python dequant_fp8_zarr2.py \\
        --src_path gs://bucket/mimo-v2-flash-fp8-ocdbt/0/items \\
        --dst_path gs://bucket/mimo-v2-flash-bf16-zarr2/0/items
"""

import argparse
import subprocess
import time

import ml_dtypes
import numpy as np
import zarr
import zarr.storage
import gcsfs

_BF16 = ml_dtypes.bfloat16


# ---------------------------------------------------------------------------
# Dequantization helpers
# ---------------------------------------------------------------------------

def _expand_scale(scale: np.ndarray, rows: int, cols: int) -> np.ndarray:
  """Expand a 2-D block scale (sr, sc) to full kernel shape (rows, cols).

  Works for both 2-D and 3-D scale tensors (stacked experts); always operates
  on the last two dimensions.
  """
  sr, sc = scale.shape[-2], scale.shape[-1]
  if rows % sr != 0 or cols % sc != 0:
    raise ValueError(
        f"Scale shape {scale.shape} is not a divisor of kernel dims ({rows}, {cols}): "
        f"bm={rows}/{sr}={rows/sr:.2f}, bn={cols}/{sc}={cols/sc:.2f}"
    )
  bm, bn = rows // sr, cols // sc
  return np.repeat(np.repeat(scale, bm, axis=-2), bn, axis=-1)


def _apply_scale(kernel: np.ndarray, scale: np.ndarray) -> np.ndarray:
  """Dequantize *kernel* using *scale* and return a bfloat16 result.

  Handles three layouts:
    - 2-D kernel   (rows, cols)           with 2-D scale (sr, sc)
    - 3-D attention (dim_in, heads, d_h)  with 2-D scale (sr, sc)
    - 3-D MoE      (experts, rows, cols)  with 3-D scale (experts, sr, sc)
  """
  w = kernel.astype(np.float32)
  s = scale.astype(np.float32)

  if w.ndim == 3 and s.ndim == 2:
    # Attention kernel: reshape to (dim_in, heads*d_h), apply scale, reshape back.
    orig_shape = w.shape
    w2d = w.reshape(w.shape[0], w.shape[1] * w.shape[2])
    rows, cols = w2d.shape
    s_exp = _expand_scale(s, rows, cols)
    w2d = w2d * s_exp
    return w2d.reshape(orig_shape).astype(_BF16)

  # 2-D or 3-D MoE — scale has the same leading batch dim(s); apply on last 2.
  rows, cols = w.shape[-2], w.shape[-1]
  s_exp = _expand_scale(s, rows, cols)
  return (w * s_exp).astype(_BF16)


# ---------------------------------------------------------------------------
# Main conversion logic
# ---------------------------------------------------------------------------

def dequant_fp8_checkpoint(src_path: str, dst_path: str) -> None:
  """Read FP8 zarr2 checkpoint at *src_path*, apply scale_inv, write to *dst_path*."""
  print(f"Source : {src_path}", flush=True)
  print(f"Dest   : {dst_path}", flush=True)

  # --- List all top-level entries from GCS ---
  print("Listing GCS checkpoint entries …", flush=True)
  result = subprocess.run(
      ["gsutil", "ls", src_path.rstrip("/") + "/"],
      capture_output=True, text=True, check=True,
  )
  lines = [l.strip() for l in result.stdout.strip().split("\n") if l.strip()]

  src_prefix = src_path.rstrip("/") + "/"
  array_names: list[str] = []
  meta_files: list[str] = []
  for line in lines:
    name = line.replace(src_prefix, "").rstrip("/")
    if not name:
      continue
    if name in ("_METADATA", "commit_success.txt"):
      meta_files.append(name)
    else:
      array_names.append(name)

  print(f"Found {len(array_names)} arrays, {len(meta_files)} metadata files", flush=True)

  # --- Identify scale_inv arrays and pair them with their kernels ---
  scale_inv_set = set(n for n in array_names if n.endswith("_scale_inv"))
  kernel_to_scale: dict[str, str] = {}
  for sn in sorted(scale_inv_set):
    # Strip trailing "_scale_inv" to derive the kernel key.
    # "...query.kernel_scale_inv" → "...query.kernel"
    # "...mlp.wi_0_scale_inv"     → "...mlp.wi_0"
    kernel_name = sn[: -len("_scale_inv")]
    if kernel_name in set(array_names):
      kernel_to_scale[kernel_name] = sn
    else:
      print(f"  WARNING: scale_inv {sn!r} has no matching kernel {kernel_name!r}, skipping", flush=True)

  print(f"Scale pairs: {len(kernel_to_scale)}", flush=True)
  for k, v in sorted(kernel_to_scale.items())[:6]:
    print(f"  {k}\n    ← {v}", flush=True)
  if len(kernel_to_scale) > 6:
    print(f"  … and {len(kernel_to_scale) - 6} more pairs", flush=True)

  # --- Open zarr stores (zarr v2 FSStore backed by gcsfs) ---
  fs = gcsfs.GCSFileSystem()
  src_store = zarr.storage.FSStore(src_path, fs=fs, mode="r", key_separator="/")
  dst_store = zarr.storage.FSStore(dst_path, fs=fs, mode="w", key_separator="/")

  # --- Process arrays ---
  t0 = time.monotonic()
  n_dequant = 0
  n_copied = 0
  n_skipped = 0

  for idx, name in enumerate(sorted(array_names)):
    elapsed = time.monotonic() - t0
    print(
        f"[{idx+1:4d}/{len(array_names)}] ({elapsed:6.0f}s) {name}",
        flush=True,
    )

    if name in scale_inv_set:
      print("  → skip (consumed as scale during dequant)", flush=True)
      n_skipped += 1
      continue

    # Load source array (zarr v2: access via FSStore path)
    src_arr = zarr.open_array(src_store, path=name, mode="r")
    data = src_arr[:]  # numpy array

    if name in kernel_to_scale:
      scale_name = kernel_to_scale[name]
      scale_arr = zarr.open_array(src_store, path=scale_name, mode="r")
      scale = scale_arr[:]
      print(
          f"  dtype={data.dtype} shape={data.shape} "
          f"| scale dtype={scale.dtype} shape={scale.shape}",
          flush=True,
      )
      data = _apply_scale(data, scale)
      print(f"  → dequantized: dtype={data.dtype} shape={data.shape}", flush=True)
      n_dequant += 1
    else:
      n_copied += 1

    # Write to destination (preserve chunk layout from source)
    dst_arr = zarr.open_array(
        dst_store,
        path=name,
        mode="w",
        shape=data.shape,
        dtype=data.dtype,
        chunks=src_arr.chunks,
        overwrite=True,
    )
    dst_arr[:] = data

  # --- Copy metadata files unchanged ---
  for mf in meta_files:
    print(f"Copying metadata: {mf}", flush=True)
    subprocess.run(
        ["gsutil", "cp", f"{src_prefix}{mf}", f"{dst_path.rstrip('/')}/{mf}"],
        check=True,
    )

  elapsed = time.monotonic() - t0
  print(
      f"\nDone!  dequantized={n_dequant}  copied={n_copied}  skipped={n_skipped}  "
      f"elapsed={elapsed:.0f}s ({elapsed/60:.1f} min)",
      flush=True,
  )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
  parser = argparse.ArgumentParser(
      description="Apply FP8 weight_scale_inv dequantization to a zarr2 checkpoint."
  )
  parser.add_argument(
      "--src_path",
      required=True,
      help="Source FP8 zarr2 checkpoint path (GCS), e.g. gs://bucket/.../0/items",
  )
  parser.add_argument(
      "--dst_path",
      required=True,
      help="Destination BF16 zarr2 checkpoint path (GCS), e.g. gs://bucket/.../0/items",
  )
  args = parser.parse_args()
  dequant_fp8_checkpoint(args.src_path, args.dst_path)
