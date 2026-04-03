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

r"""MaxText/TPU reference for MiMo-V2-Flash comparison.

A thin wrapper around maxtext.inference.decode that additionally writes
per-step logits and token IDs to a directory, creating data parallel to
what demos/compare/hf_reference.py creates on the CPU/HF side.

Run on ALL TPU workers with the same arguments:
  COMPARE_OUT_DIR=/tmp/compare_tpu \
  gcloud compute tpus tpu-vm ssh jingnw-node --zone=us-east5-b --worker=all \
    --internal-ip \
    --command='cd $HOME/maxtext && source maxtext_venv/bin/activate && \
      COMPARE_OUT_DIR=/tmp/compare_tpu \
      python3 demos/compare/maxtext_reference.py \
        src/maxtext/configs/inference/inference.yml \
        model_name=mimo-v2-flash \
        run_name=mimo_demo \
        load_parameters_path=gs://jingnw-mimo-v2-flash-us-east5/mimo-v2-flash-ocdbt/checkpoints/0/items/ \
        tokenizer_path=$HOME/mimo-tokenizer \
        tokenizer_type=huggingface \
        "prompt=<|im_start|>system..." \
        max_prefill_predict_length=512 max_target_length=530 \
        per_device_batch_size=1 dtype=bfloat16 weight_dtype=bfloat16 \
        ici_tensor_parallelism=4 ici_expert_parallelism=8 \
        attention=dot_product scan_layers=false \
        checkpoint_storage_use_ocdbt=true checkpoint_storage_use_zarr3=true \
      > /tmp/mimo_compare.log 2>&1 < /dev/null &'

Outputs (in compare_out_dir, ONLY written by worker-0 / process-0):
  tokens.json           — list of generated token IDs + decoded strings
  step{N}_logits.npy    — float32 logits [1, vocab_size] for each step N
"""

import os
import socket
import sys
import json
import time
from typing import Sequence, Any

import numpy as np
import jax
import jax.numpy as jnp
from absl import app

from maxtext.configs import pyconfig
from maxtext.common import profiler
from maxtext.common.gcloud_stub import jetstream, is_decoupled
from maxtext.inference.maxengine import maxengine
from maxtext.multimodal import processor as mm_processor
from maxtext.multimodal import utils as mm_utils
from maxtext.utils import max_utils

_config_lib, engine_api, _token_utils, _tokenizer_api, _token_params_ns = jetstream()

_NUM_STREAMS = 1


def _batch_first_result_token(first_tokens, batch_size):
    data = jnp.vstack([ft.data for ft in first_tokens])
    pad_width = [(0, batch_size - data.shape[0]), (0, 0)]
    data = jnp.pad(data, pad_width, mode="constant", constant_values=0)
    result = engine_api.ResultTokens(
        data=data,
        tokens_idx=(0, 1),
        valid_idx=(1, 2),
        length_idx=(2, 3),
        samples_per_slot=1,
    )
    return result


def _probe_hbm(label):
    host = socket.gethostname()
    for d in jax.local_devices():
        try:
            s = d.memory_stats()
            print(f"[HBM] {label:<36s} host={host} dev={d.id}"
                  f" used={s.get('bytes_in_use',0)/2**30:.2f}GB"
                  f" limit={s.get('bytes_limit',0)/2**30:.2f}GB", flush=True)
        except Exception as e:  # pylint: disable=broad-except
            print(f"[HBM] {label} dev={d.id} N/A: {e}", flush=True)


def _is_host0() -> bool:
    """True only on the process that should write comparison files."""
    return jax.process_index() == 0


def main(argv: Sequence[str]) -> None:
    jax.config.update("jax_default_prng_impl", "unsafe_rbg")
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "0"

    config = pyconfig.initialize(argv)
    jax.config.update("jax_use_shardy_partitioner", config.shardy)
    max_utils.print_system_information()
    _probe_hbm("init")

    # ------------------------------------------------------------------ #
    # COMPARE_OUT_DIR env-var controls where activations are written.
    # Pass it when launching: COMPARE_OUT_DIR=/tmp/compare_tpu python3 ...
    # ------------------------------------------------------------------ #
    compare_out_dir = os.environ.get("COMPARE_OUT_DIR", "/tmp/compare_tpu")
    if _is_host0():
        os.makedirs(compare_out_dir, exist_ok=True)
        print(f"[COMPARE] Writing comparison data to {compare_out_dir}", flush=True)

    engine = maxengine.MaxEngine(config)
    rng = jax.random.PRNGKey(1234)
    rng, rng_load_params = jax.random.split(rng)
    _t0 = time.perf_counter()
    params = engine.load_params(rng_load_params)
    _probe_hbm("after_load_params")
    print(f"[TIME] load_params elapsed={time.perf_counter()-_t0:.1f}s", flush=True)

    text = config.prompt
    prefill_length = config.max_prefill_predict_length
    processor_outputs = mm_utils.PreprocessorOutput()

    metadata = engine.get_tokenizer()
    tokenizer_model = engine.build_tokenizer(metadata)

    has_chat_template = False
    try:
        has_chat_template = bool(getattr(tokenizer_model.tokenizer, "chat_template", False))
    except AttributeError:
        pass

    tokens, true_length = tokenizer_model.encode(
        text, is_bos=not has_chat_template, prefill_lengths=[prefill_length]
    )
    if _is_host0():
        print(f"[COMPARE] Prompt encoded to {true_length} tokens.", flush=True)

    batch_size = int(config.per_device_batch_size * jax.device_count())

    # ------------------------------------------------------------------ Prefill
    _t0 = time.perf_counter()
    rng, rng_prefill = jax.random.split(rng)
    prefill_result_list = []
    first_token_list = []
    for i in range(_NUM_STREAMS):
        prefill_result, first_token = engine.prefill(
            params=params,
            padded_tokens=tokens,
            true_length=true_length,
            rng=rng_prefill,
            slot=i,
        )
        prefill_result_list.append(prefill_result)
        first_token_list.append(first_token)
    _probe_hbm("after_prefill")
    print(f"[TIME] prefill elapsed={(time.perf_counter()-_t0)*1000:.0f}ms", flush=True)

    # ------------------------------------------------------------------ Insert
    rng, rng_init_decode = jax.random.split(rng)
    decode_state = engine.init_decode_state(rng_init_decode)
    for i in range(_NUM_STREAMS):
        decode_state = engine.insert(prefill_result_list[i], decode_state, slot=i)
    _probe_hbm("after_insert")

    # ------------------------------------------------------------------ Generate
    steps = range(config.max_prefill_predict_length, config.max_target_length)
    sampled_tokens_list = [_batch_first_result_token(first_token_list, batch_size)]
    step_info = []
    max_new_tokens = len(steps)

    for rel_step, abs_step in enumerate(steps):
        rng, rng_generate = jax.random.split(rng)
        _t_step = time.perf_counter()
        decode_state, sampled_tokens = engine.generate(params, decode_state, rng=rng_generate)
        jax.effects_barrier()
        _step_ms = (time.perf_counter() - _t_step) * 1000
        print(f"[TIME] generate_step_{abs_step:04d} step_ms={_step_ms:.1f}", flush=True)
        sampled_tokens_list.append(sampled_tokens)

        # ---- Save logits and token ID (process 0 only) ----
        if _is_host0() and rel_step < max_new_tokens:
            # out_logits: [batch, 1, vocab_size] → take batch=0, pos=0
            raw_logits = np.array(decode_state["logits"][0, 0, :], dtype=np.float32)
            token_id = int(sampled_tokens.get_result_at_slot(0).tokens.item())
            token_str = tokenizer_model.decode([token_id])

            # Top-10
            top10_idx = np.argsort(raw_logits)[-10:][::-1].tolist()
            top10_val = raw_logits[top10_idx].tolist()
            top10_str = [tokenizer_model.decode([t]) for t in top10_idx]

            print(
                f"[COMPARE] step={rel_step:3d} token={token_id} ({token_str!r})"
                f"  top-3: {list(zip(top10_idx[:3], top10_str[:3], [f'{v:.2f}' for v in top10_val[:3]]))}",
                flush=True,
            )

            np.save(os.path.join(compare_out_dir, f"step{rel_step:04d}_logits.npy"), raw_logits)
            step_info.append({
                "step": rel_step,
                "token_id": token_id,
                "token_str": token_str,
                "top10_ids": top10_idx,
                "top10_logits": top10_val,
                "top10_strs": top10_str,
            })

    # ------------------------------------------------------------------ Decode results
    for i in range(_NUM_STREAMS):
        results = [t.get_result_at_slot(i).tokens.item() for t in sampled_tokens_list]
        output = tokenizer_model.decode(results)
        print(f"Input `{text}` -> `{output}`")

    if _is_host0():
        tokens_path = os.path.join(compare_out_dir, "tokens.json")
        with open(tokens_path, "w") as f:
            json.dump({
                "prompt": text,
                "generated_token_ids": [s["token_id"] for s in step_info],
                "generated_text": output if step_info else "",
                "steps": step_info,
            }, f, indent=2)
        print(f"[COMPARE] Saved token info -> {tokens_path}", flush=True)
        print(f"[COMPARE] Saved logits     -> {compare_out_dir}/step*.npy", flush=True)


if __name__ == "__main__":
    app.run(main)
