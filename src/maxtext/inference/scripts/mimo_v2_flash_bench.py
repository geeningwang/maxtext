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

"""Minimal AR generate benchmark for MiMo-V2-Flash on TPU.

Uses the same engine.generate() path as decode.py (no AOT compilation),
adds a proper warmup pass, then times TIMED_STEPS steady-state generate steps.
Reports min/median/p90/mean and total throughput.

Run on all TPU workers simultaneously:
    gcloud compute tpus tpu-vm ssh <tpu> --zone=<zone> --worker=all \\
        --command="source ~/maxtext/maxtext_tpu_venv/bin/activate && \\
                   cd ~/maxtext && \\
                   python3 -m maxtext.inference.scripts.mimo_v2_flash_bench \\
                       src/maxtext/configs/base.yml <args>"
"""

import json
import socket
import time
from typing import Sequence

import jax
import numpy as np
from absl import app

from maxtext.configs import pyconfig
from maxtext.inference.maxengine import maxengine
from maxtext.utils import max_utils

_WARMUP_STEPS = 3
_TIMED_STEPS = 50


def main(argv: Sequence[str]) -> None:
    jax.config.update("jax_default_prng_impl", "unsafe_rbg")

    config = pyconfig.initialize(argv)
    jax.config.update("jax_use_shardy_partitioner", config.shardy)
    max_utils.print_system_information()

    # ---- Build engine + load params ----------------------------------------
    engine = maxengine.MaxEngine(config)
    rng = jax.random.PRNGKey(1234)
    rng, rng_load = jax.random.split(rng)

    t0 = time.perf_counter()
    params = engine.load_params(rng_load)
    load_s = time.perf_counter() - t0
    print(f"[BENCH] load_params: {load_s:.1f}s", flush=True)

    # ---- Init decode state (empty KV cache) --------------------------------
    rng, rng_init = jax.random.split(rng)
    decode_state = engine.init_decode_state(rng_init)
    jax.block_until_ready(decode_state)
    print("[BENCH] decode_state initialised", flush=True)

    # ---- Warmup (2+ full generate passes, JIT compiles on first) -----------
    print(f"[BENCH] warmup ({_WARMUP_STEPS} steps) ...", flush=True)
    for _ in range(_WARMUP_STEPS):
        rng, rng_gen = jax.random.split(rng)
        decode_state, _ = engine.generate(params, decode_state, rng=rng_gen)
    jax.block_until_ready(decode_state)
    print("[BENCH] warmup done", flush=True)

    # ---- Timed benchmark ---------------------------------------------------
    step_times_ms = []
    print(f"[BENCH] timing {_TIMED_STEPS} steps ...", flush=True)
    for i in range(_TIMED_STEPS):
        rng, rng_gen = jax.random.split(rng)
        t_step = time.perf_counter()
        decode_state, sampled_tokens = engine.generate(params, decode_state, rng=rng_gen)
        jax.block_until_ready(sampled_tokens)
        step_ms = (time.perf_counter() - t_step) * 1000
        step_times_ms.append(step_ms)

    # ---- Report ------------------------------------------------------------
    arr = np.array(step_times_ms)
    batch_size = int(config.per_device_batch_size * jax.device_count())
    host = socket.gethostname()

    results = {
        "host": host,
        "num_devices": jax.device_count(),
        "batch_size": batch_size,
        "timed_steps": _TIMED_STEPS,
        "step_ms_mean": float(np.mean(arr)),
        "step_ms_median": float(np.median(arr)),
        "step_ms_min": float(np.min(arr)),
        "step_ms_p90": float(np.percentile(arr, 90)),
        "step_ms_max": float(np.max(arr)),
        "throughput_tok_per_s": float(batch_size / (np.median(arr) / 1000)),
    }

    print("\n" + "=" * 60, flush=True)
    print("[BENCH] === AR Generate Benchmark Results ===", flush=True)
    print(f"  Host:           {host}", flush=True)
    print(f"  Devices:        {results['num_devices']}  (batch={batch_size})", flush=True)
    print(f"  Timed steps:    {_TIMED_STEPS}", flush=True)
    print(f"  Step latency:   mean={results['step_ms_mean']:.1f}ms  median={results['step_ms_median']:.1f}ms", flush=True)
    print(f"                  min={results['step_ms_min']:.1f}ms   p90={results['step_ms_p90']:.1f}ms  max={results['step_ms_max']:.1f}ms", flush=True)
    print(f"  Throughput:     {results['throughput_tok_per_s']:.1f} tok/s  (batch={batch_size})", flush=True)
    print(f"  Per-seq:        {results['step_ms_median'] / batch_size:.1f} ms/tok/seq", flush=True)
    print("=" * 60, flush=True)

    log_path = getattr(config, "inference_microbenchmark_log_file_path", "")
    if log_path:
        with open(log_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
        print(f"[BENCH] results written to {log_path}", flush=True)


if __name__ == "__main__":
    app.run(main)
