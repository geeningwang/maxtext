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

"""Minimal AR generate + prefill benchmark for MiMo-V2-Flash on TPU.

Uses the same engine.generate() and engine.prefill() paths as decode.py
(no AOT compilation), adds a proper warmup pass, then times TIMED_STEPS
steady-state steps for each phase.

Decode benchmark: times engine.generate() — one AR step generating a single
  new token for each request in the batch (T = batch × 1 per step).

Prefill benchmark: times engine.prefill() — processes one full prompt of
  max_prefill_predict_length tokens (T = 1 × max_prefill_predict_length per
  call).  This is the phase where sparse MoE routing provides the most
  benefit, since T is large and MoE intermediates dominate HBM bandwidth.

Key distinction that drove opt4 reversal (see opt4 plan post-mortem):
  - Decode T = batch × 1 = 32 tokens per step → EP routing overhead dominates.
  - Prefill T = 1 × max_prefill_predict_length = 512 tokens per call
    → EP routing can reduce MoE HBM temporaries by up to 32×.

Run on all TPU workers simultaneously:
    gcloud compute tpus tpu-vm ssh <tpu> --zone=<zone> --worker=all \\
        --command="source ~/maxtext/maxtext_tpu_venv/bin/activate && \\
                   cd ~/maxtext && \\
                   python3 -m maxtext.inference.scripts.mimo_v2_flash_bench \\
                       src/maxtext/configs/base.yml <args>"
"""

import json
import os
import socket
import time
from typing import Sequence

import jax
import jax.numpy as jnp
import numpy as np
from absl import app

from maxtext.configs import pyconfig
from maxtext.inference.maxengine import maxengine
from maxtext.inference.maxengine.maxengine import ExistingPrefix
from maxtext.utils import max_utils

_WARMUP_STEPS = 3
_TIMED_STEPS = 50
_PREFILL_WARMUP = 3
_PREFILL_TIMED = 20


def _report(label: str, times_ms: list[float], batch_tokens: int, host: str, num_devices: int) -> dict:
    arr = np.array(times_ms)
    results = {
        "label": label,
        "host": host,
        "num_devices": num_devices,
        "batch_tokens": batch_tokens,
        "step_ms_mean": float(np.mean(arr)),
        "step_ms_median": float(np.median(arr)),
        "step_ms_min": float(np.min(arr)),
        "step_ms_p90": float(np.percentile(arr, 90)),
        "step_ms_max": float(np.max(arr)),
        "throughput_tok_per_s": float(batch_tokens / (np.median(arr) / 1000)),
    }
    print("\n" + "=" * 60, flush=True)
    print(f"[BENCH] === {label} Results ===", flush=True)
    print(f"  Host:           {host}", flush=True)
    print(f"  Devices:        {num_devices}  (batch_tokens={batch_tokens})", flush=True)
    print(f"  Timed steps:    {len(times_ms)}", flush=True)
    print(f"  Step latency:   mean={results['step_ms_mean']:.1f}ms  median={results['step_ms_median']:.1f}ms", flush=True)
    print(f"                  min={results['step_ms_min']:.1f}ms   p90={results['step_ms_p90']:.1f}ms  max={results['step_ms_max']:.1f}ms", flush=True)
    print(f"  Throughput:     {results['throughput_tok_per_s']:.1f} tok/s", flush=True)
    print("=" * 60, flush=True)
    return results


def main(argv: Sequence[str]) -> None:
    jax.config.update("jax_default_prng_impl", "unsafe_rbg")

    config = pyconfig.initialize(argv)
    jax.config.update("jax_use_shardy_partitioner", config.shardy)
    max_utils.print_system_information()

    host = socket.gethostname()
    num_devices = jax.device_count()
    all_results = {}

    # ---- Build engine + load params ----------------------------------------
    engine = maxengine.MaxEngine(config)
    rng = jax.random.PRNGKey(1234)
    rng, rng_load = jax.random.split(rng)

    t0 = time.perf_counter()
    params = engine.load_params(rng_load)
    load_s = time.perf_counter() - t0
    print(f"[BENCH] load_params: {load_s:.1f}s", flush=True)

    # Set BENCH_PREFILL_ONLY=1 to skip decode state init (needed when the KV
    # cache for a long context would OOM, but prefill throughput is still wanted).
    # Set BENCH_DECODE_ONLY=1 to skip the prefill benchmark (needed when the
    # prefill XLA intermediates at a large context × large batch would OOM, but
    # the decode state itself fits — e.g. 4K context at pdb=9 with SWA KV opt).
    prefill_only = os.environ.get("BENCH_PREFILL_ONLY", "0") == "1"
    decode_only = os.environ.get("BENCH_DECODE_ONLY", "0") == "1"

    # ====================================================================
    # Phase 1: AR Decode benchmark
    # ====================================================================
    if not prefill_only:
        rng, rng_init = jax.random.split(rng)
        decode_state = engine.init_decode_state(rng_init)
        jax.block_until_ready(decode_state)
        print("[BENCH] decode_state initialised", flush=True)

        # Warmup
        print(f"[BENCH] decode warmup ({_WARMUP_STEPS} steps) ...", flush=True)
        for _ in range(_WARMUP_STEPS):
            rng, rng_gen = jax.random.split(rng)
            decode_state, _ = engine.generate(params, decode_state, rng=rng_gen)
        jax.block_until_ready(decode_state)
        print("[BENCH] decode warmup done", flush=True)

        # Timed
        step_times_ms = []
        print(f"[BENCH] timing {_TIMED_STEPS} decode steps ...", flush=True)
        for _ in range(_TIMED_STEPS):
            rng, rng_gen = jax.random.split(rng)
            t_step = time.perf_counter()
            decode_state, sampled_tokens = engine.generate(params, decode_state, rng=rng_gen)
            jax.block_until_ready(sampled_tokens)
            step_times_ms.append((time.perf_counter() - t_step) * 1000)

        batch_size = int(config.per_device_batch_size * num_devices)
        all_results["decode"] = _report(
            "AR Decode", step_times_ms, batch_tokens=batch_size,
            host=host, num_devices=num_devices,
        )
    else:
        print("[BENCH] skipping decode benchmark (BENCH_PREFILL_ONLY=1)", flush=True)

    # ====================================================================
    # Phase 2: Prefill benchmark
    # ====================================================================
    prefill_len = int(config.max_prefill_predict_length)
    if decode_only:
        print("[BENCH] skipping prefill benchmark (BENCH_DECODE_ONLY=1)", flush=True)
    elif prefill_len > 0:
        use_chunked = engine.use_chunked_prefill
        chunk_size = engine.prefill_chunk_size if use_chunked else prefill_len
        n_chunks = prefill_len // chunk_size if use_chunked else 1
        print(
            f"\n[BENCH] prefill benchmark (seq_len={prefill_len}, "
            f"chunked={use_chunked}, chunk_size={chunk_size}, n_chunks={n_chunks}) ...",
            flush=True,
        )
        # Synthetic prompt: fill with token ID 1 (typically <s> or pad)
        chunk_tokens = jnp.ones((chunk_size,), dtype=jnp.int32)

        def _run_prefill(rng):
            """Run one full prefill (chunked or monolithic) and return result."""
            prefix_result, first_token = engine.prefill(
                params=params,
                padded_tokens=chunk_tokens,
                true_length=chunk_size,
                rng=rng,
                slot=0,
            )
            for i in range(1, n_chunks):
                existing_prefix = ExistingPrefix(
                    cache=prefix_result["cache"],
                    # Shape encodes write position for KV cache; actual values unused.
                    common_prefix_tokens=jnp.ones((i * chunk_size,), dtype=jnp.int32),
                )
                rng, rng_pf = jax.random.split(rng)
                prefix_result, first_token = engine.prefill(
                    params=params,
                    existing_prefix=existing_prefix,
                    padded_tokens=chunk_tokens,
                    true_length=chunk_size,
                    rng=rng_pf,
                    slot=0,
                )
            return prefix_result, first_token

        # Warmup — triggers JIT compilation for each chunk position
        print(f"[BENCH] prefill warmup ({_PREFILL_WARMUP} calls) ...", flush=True)
        for _ in range(_PREFILL_WARMUP):
            rng, rng_pf = jax.random.split(rng)
            result = _run_prefill(rng_pf)
            jax.block_until_ready(result)
        print("[BENCH] prefill warmup done", flush=True)

        # Timed
        prefill_times_ms = []
        print(f"[BENCH] timing {_PREFILL_TIMED} prefill calls ...", flush=True)
        for _ in range(_PREFILL_TIMED):
            rng, rng_pf = jax.random.split(rng)
            t_pf = time.perf_counter()
            result = _run_prefill(rng_pf)
            jax.block_until_ready(result)
            prefill_times_ms.append((time.perf_counter() - t_pf) * 1000)

        all_results["prefill"] = _report(
            f"Prefill (seq_len={prefill_len})", prefill_times_ms,
            batch_tokens=prefill_len, host=host, num_devices=num_devices,
        )
    else:
        print("[BENCH] skipping prefill benchmark (max_prefill_predict_length=0)", flush=True)

    # ---- Persist results ---------------------------------------------------
    log_path = getattr(config, "inference_microbenchmark_log_file_path", "")
    if log_path:
        with open(log_path, "w", encoding="utf-8") as f:
            json.dump(all_results, f, indent=2)
        print(f"[BENCH] results written to {log_path}", flush=True)


if __name__ == "__main__":
    app.run(main)
