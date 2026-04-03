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

r"""Compare HF CPU reference activations against MaxText/TPU activations.

After running:
  1. demos/compare/hf_reference.py   → /tmp/compare_hf/
  2. demos/compare/maxtext_reference.py → /tmp/compare_tpu/

run this script to produce a side-by-side divergence report:
  python3 demos/compare/compare.py \
      --hf_dir  /tmp/compare_hf \
      --tpu_dir /tmp/compare_tpu

Output format:
  - Per-step token comparison table (HF token vs TPU token)
  - First divergent step
  - For divergent steps: per-layer cosine similarity (if layer activations exist)
  - Logit-space statistics: rank of HF top-1 in TPU, and vice-versa
"""

import argparse
import json
import os
import sys
from glob import glob

import numpy as np


# -------------------------------------------------------------------------
# Helpers
# -------------------------------------------------------------------------

def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    na = np.linalg.norm(a.astype(np.float32))
    nb = np.linalg.norm(b.astype(np.float32))
    if na < 1e-9 or nb < 1e-9:
        return 0.0
    return float(np.dot(a.astype(np.float32), b.astype(np.float32)) / (na * nb))


def _top_k_ids(logits: np.ndarray, k: int = 10) -> list[int]:
    return np.argsort(logits)[-k:][::-1].tolist()


def _rank_of(logits: np.ndarray, token_id: int) -> int:
    """1-based rank of token_id among descending logits."""
    return int(np.sum(logits > logits[token_id])) + 1


def _load_json(path: str) -> dict:
    if not os.path.exists(path):
        print(f"ERROR: {path} not found.  Did you run the reference scripts?",
              file=sys.stderr)
        sys.exit(1)
    with open(path) as f:
        return json.load(f)


# -------------------------------------------------------------------------
# Report sections
# -------------------------------------------------------------------------

def compare_tokens(hf_steps: list, tpu_steps: list) -> int:
    """Prints a side-by-side token table; returns first divergent step index."""
    n = min(len(hf_steps), len(tpu_steps))
    print("\n" + "="*78)
    print("TOKEN COMPARISON (greedy)")
    print("="*78)
    print(f"{'Step':>4}  {'HF ID':>7}  {'HF str':>15}  {'TPU ID':>7}  {'TPU str':>15}  Match")
    print("-"*78)

    first_diverge = n  # sentinel
    for i in range(n):
        hf = hf_steps[i]
        tpu = tpu_steps[i]
        match = hf["token_id"] == tpu["token_id"]
        marker = "✓" if match else "✗ ← DIVERGE"
        print(f"{i:>4}  {hf['token_id']:>7}  {hf['token_str']!r:>15}  "
              f"{tpu['token_id']:>7}  {tpu['token_str']!r:>15}  {marker}")
        if not match and first_diverge == n:
            first_diverge = i
    print("-"*78)
    if first_diverge < n:
        print(f"First divergence at step {first_diverge}.")
    else:
        print("All compared steps MATCH.")
    return first_diverge


def compare_logits_at_step(step: int, hf_dir: str, tpu_dir: str,
                           hf_steps: list, tpu_steps: list) -> None:
    """Detailed logit comparison at a single step."""
    hf_path  = os.path.join(hf_dir,  f"step{step:04d}_logits.npy")
    tpu_path = os.path.join(tpu_dir, f"step{step:04d}_logits.npy")
    if not os.path.exists(hf_path) or not os.path.exists(tpu_path):
        print(f"  (logit files for step {step} not found; skipping detailed logit comparison)")
        return

    hf_logits  = np.load(hf_path).astype(np.float32)
    tpu_logits = np.load(tpu_path).astype(np.float32)

    print(f"\n--- Logit analysis at step {step} ---")

    hf_top1  = hf_steps[step]["token_id"]
    tpu_top1 = tpu_steps[step]["token_id"]

    rank_of_hf_top1_in_tpu  = _rank_of(tpu_logits, hf_top1)
    rank_of_tpu_top1_in_hf  = _rank_of(hf_logits, tpu_top1)

    print(f"  HF  top-1 token {hf_top1:6d} ranks #{rank_of_hf_top1_in_tpu:6d} in TPU logits")
    print(f"  TPU top-1 token {tpu_top1:6d} ranks #{rank_of_tpu_top1_in_hf:6d} in HF  logits")

    cos = _cosine(hf_logits, tpu_logits)
    print(f"  Logit cosine similarity HF vs TPU: {cos:.6f}")

    # Top-10 from each
    hf_top10  = _top_k_ids(hf_logits, 10)
    tpu_top10 = _top_k_ids(tpu_logits, 10)
    overlap = len(set(hf_top10) & set(tpu_top10))
    print(f"  Top-10 overlap: {overlap}/10 tokens in common")

    if cos < 0.9:
        print("  *** LOW cosine similarity → logit distributions are very different ***")
        print("  Likely cause: a bug in the forward pass (wrong weights, wrong operation).")
    elif cos < 0.999:
        print("  *** MODERATE cosine similarity → small numerical differences ***")
        print("  Possible cause: dtype precision (BF16 vs FP32) or subtle op differences.")
    else:
        print("  High cosine similarity → logit distributions nearly identical.")


def compare_layer_activations(step: int, hf_dir: str, tpu_dir: str) -> None:
    """Compare per-layer hidden state norms (if files exist for both sides)."""
    hf_files  = sorted(glob(os.path.join(hf_dir,  f"step{step:04d}_layer*.npy")))
    tpu_files = sorted(glob(os.path.join(tpu_dir, f"step{step:04d}_layer*.npy")))
    if not hf_files and not tpu_files:
        return

    print(f"\n--- Per-layer hidden-state comparison at step {step} ---")

    def _layer_idx(path):
        base = os.path.basename(path)
        # e.g. step0000_layer05.npy
        return int(base.split("layer")[1].split(".")[0])

    hf_by_layer  = {_layer_idx(p): np.load(p).astype(np.float32) for p in hf_files}
    tpu_by_layer = {_layer_idx(p): np.load(p).astype(np.float32) for p in tpu_files}
    all_layers = sorted(set(hf_by_layer) | set(tpu_by_layer))

    print(f"  {'Layer':>5}  {'HF norm':>10}  {'TPU norm':>10}  {'Cosine':>8}  {'L2 err':>10}")
    print("  " + "-"*52)
    for l in all_layers:
        hf_h  = hf_by_layer.get(l)
        tpu_h = tpu_by_layer.get(l)
        if hf_h is None or tpu_h is None:
            continue
        # Handle shape mismatch (should not happen if both scripts use the same config)
        if hf_h.shape != tpu_h.shape:
            print(f"  {l:>5}  SHAPE MISMATCH: HF {hf_h.shape} vs TPU {tpu_h.shape}")
            continue
        cos = _cosine(hf_h, tpu_h)
        l2  = float(np.linalg.norm(hf_h - tpu_h))
        print(f"  {l:>5}  {np.linalg.norm(hf_h):>10.3f}  {np.linalg.norm(tpu_h):>10.3f}"
              f"  {cos:>8.5f}  {l2:>10.3f}")


# -------------------------------------------------------------------------
# Main
# -------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--hf_dir",  default="/tmp/compare_hf",  help="HF reference output directory.")
    p.add_argument("--tpu_dir", default="/tmp/compare_tpu", help="MaxText/TPU output directory.")
    p.add_argument("--steps", type=int, nargs="+", default=None,
                   help="Specific steps to analyze in detail (default: first divergent + next 2).")
    args = p.parse_args()

    hf_tokens  = _load_json(os.path.join(args.hf_dir,  "tokens.json"))
    tpu_tokens = _load_json(os.path.join(args.tpu_dir, "tokens.json"))

    hf_steps  = hf_tokens.get("steps", [])
    tpu_steps = tpu_tokens.get("steps", [])

    print(f"HF  generated text: {hf_tokens.get('generated_text', '(missing)')!r}")
    print(f"TPU generated text: {tpu_tokens.get('generated_text', '(missing)')!r}")

    first_diverge = compare_tokens(hf_steps, tpu_steps)

    # Detailed analysis at divergent steps
    if args.steps is not None:
        detail_steps = args.steps
    elif first_diverge < min(len(hf_steps), len(tpu_steps)):
        n_available = min(len(hf_steps), len(tpu_steps))
        detail_steps = list(range(first_diverge, min(first_diverge + 3, n_available)))
    else:
        detail_steps = []

    for s in detail_steps:
        compare_logits_at_step(s, args.hf_dir, args.tpu_dir, hf_steps, tpu_steps)
        compare_layer_activations(s, args.hf_dir, args.tpu_dir)

    # Summary guidance
    print("\n" + "="*78)
    print("INTERPRETATION GUIDE")
    print("="*78)
    print("""
  If step=0 already diverges AND cosine(logits) ≈ 0:
    → The FORWARD PASS is completely wrong. Compare layer-by-layer.
    → Likely candidates: wrong weight loading, transposed matmul,
      missing ops (e.g. sin_bias, RoPE, e_score_correction_bias).

  If step=0 diverges AND cosine(logits) > 0.9 but top-1 differs:
    → Small numerical differences accumulate. Check dtype (BF16 vs FP32),
      or a slightly wrong constant (e.g. wrong RoPE theta).

  If step>0 is first divergence:
    → The PREFILL is correct; the bug is in the DECODE loop
      (KV cache update, position encoding during decode, sliding window).

  If layer hidden-states are fine through layer K but wrong at K+1:
    → The bug is in layer K+1's attention or MLP.
    → Add more granular hooks (pre-attention vs post-attention) at that layer.
""")


if __name__ == "__main__":
    main()
