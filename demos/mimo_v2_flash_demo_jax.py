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

r"""MaxText / JAX inference demo for MiMo-V2-Flash on TPU.

This script demonstrates how to run the Xiaomi MiMo-V2-Flash model with
MaxText on a TPU.  It supports both single-host (e.g. v4-8, v5litepod-8,
Ironwood-4) and multi-host (e.g. v6e-32 with 8 VMs) configurations.

Prerequisites
-------------
1. Convert the HF checkpoint to MaxText Orbax format first:

   python3 -m maxtext.checkpoint_conversion.standalone_scripts.convert_mimo_v2_flash \
       --base_model_path XiaomiMiMo/MiMo-V2-Flash \
       --maxtext_model_path gs://<bucket>/mimo-v2-flash/checkpoints/0/items \
       --tmpdir /mnt/scratch/mimo_tmp \
       --simulated_cpu_devices_count 1

2. This script calls MaxText's ``maxtext.inference.decode`` module which
   requires an initialised TPU environment (jax >= 0.8, libtpu).

Usage
-----
# Single-host TPU (v4-8, Ironwood-4, etc.):
python3 demos/mimo_v2_flash_demo_jax.py \
    --checkpoint_path gs://<bucket>/mimo-v2-flash/checkpoints/0/items \
    --tokenizer_path XiaomiMiMo/MiMo-V2-Flash \
    --ici_tensor_parallelism 4 \
    --prompt "Explain the key difference between attention and cross-attention."

# Multi-host TPU v6e-32 (8 VMs, 32 chips total) — MUST run on all workers:
#
#   IMPORTANT: For multi-host TPU slices, JAX requires all workers to start
#   simultaneously.  Launch with --worker=all from the manager VM:
#
gcloud compute tpus tpu-vm ssh <tpu-name> --zone=<zone> --worker=all \
    --command="cd maxtext && source maxtext_tpu_venv/bin/activate && \
        python3 demos/mimo_v2_flash_demo_jax.py \
            --checkpoint_path gs://<bucket>/mimo-v2-flash/checkpoints/0/items \
            --tokenizer_path XiaomiMiMo/MiMo-V2-Flash \
            --ici_tensor_parallelism 4 \
            --ici_expert_parallelism 8 \
            --prompt 'Solve step by step: A train travels at 120 km/h...'"

# Dry-run on CPU (verifies config validity only, skips actual model execution):
python3 demos/mimo_v2_flash_demo_jax.py \
    --checkpoint_path /tmp/mimo_ckpt \
    --tokenizer_path XiaomiMiMo/MiMo-V2-Flash \
    --dry_run

Architecture notes
------------------
MiMo-V2-Flash key properties relevant to TPU inference:
  • 48 hybrid attention layers (9 global + 39 sliding-window, 128-token window)
  • Q/K head dim = 192, V head dim = 128 (asymmetric)
  • Partial RoPE: 33.4% of Q/K head dim rotated (rope_dim = 64)
    - GA RoPE theta: 5 000 000   SWA RoPE theta: 10 000
  • MoE: 256 experts per MoE layer (47/48 layers), top-8 routing
    - sigmoid scoring with noaux-TC correction bias
  • Dense MLP only on layer 0 (intermediate = 16 384)
  • MoE intermediate size: 2 048

Chat template and EOS
---------------------
The script always passes ``use_chat_template=true`` to decode.py, which calls
``tokenizer.apply_chat_template()`` to wrap the raw prompt as a proper
chat turn::

    <|im_start|>system\nYou are MiMo...\n<|im_end|><|im_start|>user\n<prompt><|im_end|><|im_start|>assistant\n

This ensures the model emits a single assistant reply ending with
``<|im_end|>`` (EOS token id 151645) rather than running to ``max_new_tokens``.
decode.py checks for EOS at every generate step and stops early.

Measured performance (v6e-32, 8 workers, TP=4 × EP=8, 2026-04-08)
------------------------------------------------------------------
  Checkpoint load (OCDBT, 8-process): ~36 s
  Prefill (512 tokens):               ~22 s
  Generate (~600 tokens, EOS stop):   ~43 s   (~78 ms/step, ~12.8 tok/s)
  HBM per chip after load:            ~18.0 GB / 31.25 GB (57.5%)
"""

import argparse
import re
import subprocess
import sys
import textwrap
import time


# ---------------------------------------------------------------------------
# MaxText config flags for MiMo-V2-Flash inference
# ---------------------------------------------------------------------------

MIMO_BASE_FLAGS = {
    "decoder_block": "mimo_v2_flash",
    "base_emb_dim": 4096,
    "base_num_decoder_layers": 48,
    "base_num_query_heads": 64,
    "base_num_kv_heads": 4,
    "head_dim": 192,
    "vocab_size": 152576,
    "base_mlp_dim": 16384,
    "base_moe_mlp_dim": 2048,
    "num_experts": 256,
    "num_experts_per_tok": 8,
    "norm_topk_prob": True,
    "routed_score_func": "sigmoid",
    "mlp_activations": ["silu", "linear"],
    "normalization_layer_epsilon": 1.0e-5,
    "rope_max_timescale": 5000000,
    "partial_rotary_factor": 0.334,
    "mimo_v_head_dim": 128,
    "mimo_swa_num_kv_heads": 8,
    "mimo_swa_rope_theta": 10000.0,
    "mimo_swa_window_size": 128,
    "mimo_attention_value_scale": 0.707,
    "mimo_hybrid_layer_pattern": [
        0,1,1,1,1, 0,1,1,1,1,1, 0,1,1,1,1,1, 0,1,1,1,1,1, 0,1,1,1,1,1,
        0,1,1,1,1,1, 0,1,1,1,1,1, 0,1,1,1,1,1, 0,
    ],
    "mimo_moe_layer_freq": [0] + [1] * 47,
    "logits_via_embedding": False,
    "use_qk_norm": False,
    "scan_layers": False,
    # Inference-specific defaults
    "max_target_length": 4096,
    "max_prefill_predict_length": 2048,
    "enable_dropout": False,
}

# Default generation settings
DEFAULT_PREFILL_LENGTH = 512
DEFAULT_MAX_NEW_TOKENS = 512


def build_decode_command(
    checkpoint_path: str,
    tokenizer_path: str,
    prompt: str,
    run_name: str = "mimo_v2_flash_demo",
    max_prefill: int = DEFAULT_PREFILL_LENGTH,
    max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
    per_device_batch_size: int = 1,
    dtype: str = "bfloat16",
    ici_tensor_parallelism: int = 4,
    ici_expert_parallelism: int = 8,
    scan_layers: bool = False,
    use_chunked_prefill: bool = False,
    prefill_chunk_size: int = 4096,
    expert_shard_attention_option: str = "context",
    use_qwix_quantization: bool = False,
    quantization: str = "",
    checkpoint_is_quantized: bool = False,
    checkpoint_storage_use_zarr3: bool = True,
    mimo_fp8_weight_mode: str = "",
) -> list[str]:
    """Build the shell command for maxtext.inference.decode.

    Note: MaxText's ``max_target_length`` is the *total* sequence length
    (prefill tokens + generated tokens), so it is computed as
    ``max_prefill + max_new_tokens``.
    """
    max_target_length = max_prefill + max_new_tokens
    cmd = [
        sys.executable,
        "-m",
        "maxtext.inference.decode",
        "src/maxtext/configs/base.yml",
        f"model_name=mimo-v2-flash",
        f"run_name={run_name}",
        f"load_parameters_path={checkpoint_path}",
        f"tokenizer_path={tokenizer_path}",
        f"tokenizer_type=huggingface",
        # Wrap prompt in single quotes so OmegaConf treats colons/special chars as literals.
        f"prompt='{prompt}'",
        f"max_prefill_predict_length={max_prefill}",
        f"max_target_length={max_target_length}",
        f"per_device_batch_size={per_device_batch_size}",
        f"dtype={dtype}",
        f"weight_dtype={dtype}",
        # MiMo-V2-Flash has only 4 global-attention KV heads, so tensor parallelism
        # must not exceed 4.  Use expert parallelism for the 256 MoE experts instead.
        f"ici_tensor_parallelism={ici_tensor_parallelism}",
        f"ici_expert_parallelism={ici_expert_parallelism}",
        f"scan_layers={'true' if scan_layers else 'false'}",
        # Use dot_product attention to avoid splash attention block-size alignment
        # requirements (splash requires max_target_length % q_block_size == 0).
        "attention=dot_product",
        f"expert_shard_attention_option={expert_shard_attention_option}",
        # Checkpoint format: zarr3 + OCDBT (produced by convert_checkpoint_to_ocdbt.py)
        "checkpoint_storage_use_ocdbt=true",
        f"checkpoint_storage_use_zarr3={'true' if checkpoint_storage_use_zarr3 else 'false'}",
        # Apply the tokenizer chat template so the model produces a single
        # assistant turn ending with <|im_end|> (EOS), rather than open-ended
        # text completion.  decode.py will call apply_chat_template() when this
        # flag is set.  Falls back to raw prompt if the tokenizer has no template.
        "use_chat_template=true",
        # Nucleus (top-p) sampling with temperature to prevent greedy repetition
        # loops.  Greedy decoding causes the model to repeat tokens like
        # '= = =' or '/h/h/h' once it enters a locally-highest-probability cycle,
        # exhausting max_new_tokens without ever generating EOS.
        "decode_sampling_strategy=nucleus",
        "decode_sampling_nucleus_p=0.95",
        "decode_sampling_temperature=0.6",
    ]
    # scan_layers=true requires the stacked checkpoint and cannot use async
    # checkpointing (shape mismatch with the default param_scan_axis=1).
    if scan_layers:
        cmd.append("async_checkpointing=false")
    if use_chunked_prefill:
        cmd.append("use_chunked_prefill=true")
        cmd.append(f"prefill_chunk_size={prefill_chunk_size}")
    if use_qwix_quantization:
        cmd.append("use_qwix_quantization=true")
        if quantization:
            cmd.append(f"quantization={quantization}")
        if checkpoint_is_quantized:
            cmd.append("checkpoint_is_quantized=true")
    if mimo_fp8_weight_mode:
        cmd.append(f"mimo_fp8_weight_mode={mimo_fp8_weight_mode}")
    return cmd


def run_inference(
    checkpoint_path: str,
    tokenizer_path: str,
    prompt: str,
    max_prefill: int = DEFAULT_PREFILL_LENGTH,
    max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
    dtype: str = "bfloat16",
    verbose: bool = False,
    ici_tensor_parallelism: int = 4,
    ici_expert_parallelism: int = 8,
    scan_layers: bool = False,
    use_chunked_prefill: bool = False,
    prefill_chunk_size: int = 4096,
    expert_shard_attention_option: str = "context",
    use_qwix_quantization: bool = False,
    quantization: str = "",
    checkpoint_is_quantized: bool = False,
    checkpoint_storage_use_zarr3: bool = True,
    mimo_fp8_weight_mode: str = "",
) -> tuple[str, float | None, bool]:
    """Execute MaxText inference and return (generated_text, tok_per_s, eos_fired).

    tok_per_s is the AR-generate throughput measured from the generate-loop
    timing lines printed by decode.py.  Returns None if the timing lines are
    not found in stdout (e.g. early crash).

    eos_fired is True if decode.py printed an EOS-stop message, meaning the
    model produced a clean response.  False means all max_new_tokens steps ran
    without hitting EOS — output is likely garbled / truncated.
    """
    cmd = build_decode_command(
        checkpoint_path=checkpoint_path,
        tokenizer_path=tokenizer_path,
        prompt=prompt,
        max_prefill=max_prefill,
        max_new_tokens=max_new_tokens,
        dtype=dtype,
        ici_tensor_parallelism=ici_tensor_parallelism,
        ici_expert_parallelism=ici_expert_parallelism,
        scan_layers=scan_layers,
        use_chunked_prefill=use_chunked_prefill,
        prefill_chunk_size=prefill_chunk_size,
        expert_shard_attention_option=expert_shard_attention_option,
        use_qwix_quantization=use_qwix_quantization,
        quantization=quantization,
        checkpoint_is_quantized=checkpoint_is_quantized,
        checkpoint_storage_use_zarr3=checkpoint_storage_use_zarr3,
        mimo_fp8_weight_mode=mimo_fp8_weight_mode,
    )
    if verbose:
        print("Running command:")
        print("  " + " \\\n    ".join(cmd))
        print()

    _t_wall_start = time.perf_counter()
    result = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=None if verbose else subprocess.PIPE,
        text=True,
        check=False,
    )
    _t_wall_total = time.perf_counter() - _t_wall_start
    if verbose:
        # stderr was streamed live to terminal; echo the stdout (Input -> line) too.
        print(result.stdout or "", end="", flush=True)
    if result.returncode != 0:
        stderr = result.stderr or "(stderr streamed to terminal)"
        raise RuntimeError(
            f"MaxText inference failed (exit code {result.returncode}).\n"
            f"Stderr:\n{textwrap.indent(stderr, '  ')}"
        )
    # Extract only the generated text from the "Input `<prompt>` -> `<output>`"
    # line, stripping all the MaxText stats that precede it on stdout.
    stdout = result.stdout or ""
    eos = "<|im_end|>"
    arrow = " -> `"
    arrow_idx = stdout.find(arrow)
    text = None
    if arrow_idx != -1:
        text_start = arrow_idx + len(arrow)
        eos_idx = stdout.find(eos, text_start)
        if eos_idx != -1:
            text = stdout[text_start:eos_idx].rstrip("`")
        else:
            close = stdout.find("`", text_start)
            if close != -1:
                text = stdout[text_start:close]
    if text is None:
        for line in stdout.splitlines():
            if arrow in line:
                text = line
                break
    if text is None:
        text = stdout

    # Compute tok/s from decode.py timing lines:
    #   [TIME] generate_total ... total=Xs steps=N avg_ms=M
    # Actual steps taken = number of "[TIME] generate_step_" lines printed.
    tok_per_s = None
    total_match = re.search(r"\[TIME\] generate_total\s+\S+\s+total=(\S+)s", stdout)
    actual_steps = len(re.findall(r"\[TIME\] generate_step_", stdout))
    if total_match and actual_steps > 0:
        total_s = float(total_match.group(1))
        tok_per_s = actual_steps / total_s if total_s > 0 else None

    # EOS detection: decode.py prints "[INFO] EOS token ... generated at step N"
    # when the model stops early.  If absent the model exhausted max_new_tokens.
    eos_fired = bool(re.search(r"\[INFO\] EOS token", stdout))

    return text, tok_per_s, eos_fired


def dry_run(checkpoint_path: str, tokenizer_path: str):
    """Validate that MaxText config parses correctly without running inference."""
    print("Dry-run: validating MiMo-V2-Flash MaxText config...")
    import os
    import pathlib
    os.environ.setdefault("JAX_PLATFORMS", "cpu")

    # Locate the MaxText repo root:
    #   • Normal invocation: __file__ = <repo>/demos/mimo_v2_flash_demo_jax.py
    #     → parents[1] = <repo>
    #   • Script copied elsewhere: fall back to CWD.
    _script = pathlib.Path(__file__).resolve()
    _candidate_root = _script.parents[1]
    _base_yml = _candidate_root / "src/maxtext/configs/base.yml"
    if _base_yml.exists():
        _repo_root = _candidate_root
    else:
        _repo_root = pathlib.Path(os.getcwd())

    base_cfg = str(_repo_root / "src/maxtext/configs/base.yml")

    sys.path.insert(0, str(_repo_root / "src"))

    try:
        from maxtext.configs import pyconfig  # pylint: disable=import-outside-toplevel
        argv = [
            "dry_run",
            base_cfg,
            "model_name=mimo-v2-flash",
            f"run_name=mimo_dry_run",
            f"load_parameters_path={checkpoint_path}",
            f"tokenizer_path={tokenizer_path}",
            # max_target_length must be >= max_prefill_predict_length.
            # Use small consistent values so the dry-run is fast.
            "max_prefill_predict_length=64",
            "max_target_length=128",
            "per_device_batch_size=1",
        ]
        cfg = pyconfig.initialize(argv)
        print(f"  decoder_block  : {cfg.decoder_block}")
        print(f"  emb_dim        : {cfg.emb_dim}")
        print(f"  num_heads      : {cfg.num_query_heads}")
        print(f"  head_dim       : {cfg.head_dim}")
        print(f"  num_layers     : {cfg.num_decoder_layers}")
        print(f"  num_experts    : {cfg.num_experts}")
        print(f"  mimo_v_head_dim: {cfg.mimo_v_head_dim}")
        print(f"  mimo_swa_window: {cfg.mimo_swa_window_size}")
        print("Config validated successfully.")
        return True
    except Exception as e:  # pylint: disable=broad-except
        print(f"Config validation FAILED: {e}", file=sys.stderr)
        return False


def print_architecture_summary():
    """Print a concise summary of the MiMo-V2-Flash architecture."""
    print(textwrap.dedent("""
    ┌─────────────────────────────────────────────────────────────┐
    │              MiMo-V2-Flash  Architecture Summary            │
    ├─────────────────────────────────────────────────────────────┤
    │  Parameters      309B total / 15B active                    │
    │  Layers          48  (9 global attention + 39 SWA)          │
    │  Hidden size     4096                                        │
    │  Q/K heads       64    (head_dim = 192)                     │
    │  KV heads (GA)   4     (KV head_dim = 192)                  │
    │  KV heads (SWA)  8     (KV head_dim = 192)                  │
    │  V head_dim      128   (asymmetric with Q/K)                │
    │  RoPE dims       64 / 192  (partial_rotary = 0.334)         │
    │  SWA window      128 tokens                                  │
    │  MoE experts     256  (top-8, sigmoid, noaux-TC)            │
    │  MoE layers      47 / 48  (layer 0 is dense MLP)            │
    │  MoE intermed.   2048                                        │
    │  Dense intermed. 16384  (layer 0 only)                      │
    │  Vocab size      152576                                      │
    │  Context (max)   262144 tokens                               │
    └─────────────────────────────────────────────────────────────┘
    """).strip())


def main():
    parser = argparse.ArgumentParser(
        description="MaxText/JAX inference demo for MiMo-V2-Flash."
    )
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        required=True,
        help="Path to the MaxText Orbax checkpoint "
             "(local path or gs://bucket/path/0/items).",
    )
    parser.add_argument(
        "--tokenizer_path",
        type=str,
        default="XiaomiMiMo/MiMo-V2-Flash",
        help="HuggingFace tokenizer path or repo id.",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default=(
            "Solve step by step: A train travels at 120 km/h for 2.5 hours, "
            "then at 80 km/h for 1.5 hours. What is the total distance traveled?"
        ),
        help="Input prompt for inference.",
    )
    parser.add_argument(
        "--max_prefill",
        type=int,
        default=DEFAULT_PREFILL_LENGTH,
        help="Maximum prefill length in tokens.",
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=DEFAULT_MAX_NEW_TOKENS,
        help="Maximum new tokens to generate.",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="bfloat16",
        choices=["bfloat16", "float32"],
        help="Computation dtype.",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        default=False,
        help="Validate the MaxText config without running inference.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        default=False,
        help="Print the MaxText decode command before running it.",
    )
    parser.add_argument(
        "--ici_tensor_parallelism",
        type=int,
        default=4,
        help="ICI tensor-parallel degree. Must evenly divide the number of KV heads (4 for "
             "global-attention layers). Default 4 works on v6e-32 combined with "
             "ici_expert_parallelism=8 (4×8=32 chips).",
    )
    parser.add_argument(
        "--ici_expert_parallelism",
        type=int,
        default=8,
        help="ICI expert-parallel degree. 256 experts / 8 = 32 experts per chip. "
             "Combined with ici_tensor_parallelism=4 gives 4×8=32 chips total on v6e-32.",
    )
    parser.add_argument(
        "--print_arch",
        action="store_true",
        default=False,
        help="Print architecture summary and exit.",
    )
    parser.add_argument(
        "--scan_layers",
        action="store_true",
        default=False,
        help="Use scan_layers=true (lax.while_loop per phase). Requires the 4-phase "
             "stacked checkpoint (mimo-v2-flash-4phase-stacked). "
             "Sets async_checkpointing=false automatically.",
    )
    parser.add_argument(
        "--use_chunked_prefill",
        action="store_true",
        default=False,
        help="Split the prefill prompt into chunks of --prefill_chunk_size tokens, "
             "each processed with dot_product attention. Required for 16K prefill at pdb>=2 "
             "(flash prefill OOMs). Implies expert_shard_attention_option=context.",
    )
    parser.add_argument(
        "--prefill_chunk_size",
        type=int,
        default=4096,
        help="Chunk size (tokens) used when --use_chunked_prefill is set. Default 4096.",
    )
    parser.add_argument(
        "--use_qwix_quantization",
        action="store_true",
        default=False,
        help="Enable qwix quantization (PtqProvider). Use with --quantization and "
             "--checkpoint_is_quantized for pre-quantized FP8 checkpoints.",
    )
    parser.add_argument(
        "--quantization",
        type=str,
        default="",
        help="Quantization scheme, e.g. 'fp8_full'. Only used when --use_qwix_quantization is set.",
    )
    parser.add_argument(
        "--checkpoint_is_quantized",
        action="store_true",
        default=False,
        help="Indicates the checkpoint contains pre-quantized (FP8) weights. "
             "Use with --use_qwix_quantization. Disables zarr3 (FP8 checkpoint uses zarr2).",
    )
    parser.add_argument(
        "--mimo_fp8_weight_mode",
        type=str,
        default="",
        help=(
            "FP8 weight mode for MoE expert weights. "
            "'' (default): BF16 weights from standard checkpoint. "
            "'block_wise_fp8': expert weights are float8_e4m3fn in checkpoint with "
            "separate scale_inv tensors; block_dequant_fp8 is applied before each MoE einsum. "
            "Use with an --keep_fp8-converted checkpoint."
        ),
    )
    args = parser.parse_args()

    if args.print_arch:
        print_architecture_summary()
        return

    if args.dry_run:
        ok = dry_run(args.checkpoint_path, args.tokenizer_path)
        sys.exit(0 if ok else 1)

    print_architecture_summary()
    print(f"\nPrompt:\n{args.prompt}\n")
    scan_label = "scan_layers=true (stacked ckpt)" if args.scan_layers else "scan_layers=false (dense)"
    chunked_label = f"chunked prefill (chunk={args.prefill_chunk_size})" if args.use_chunked_prefill else "full prefill"
    print(f"Mode: {scan_label}, {chunked_label}")
    print("-" * 60)

    output, tok_per_s, eos_fired = run_inference(
        checkpoint_path=args.checkpoint_path,
        tokenizer_path=args.tokenizer_path,
        prompt=args.prompt,
        max_prefill=args.max_prefill,
        max_new_tokens=args.max_new_tokens,
        dtype=args.dtype,
        verbose=args.verbose,
        ici_tensor_parallelism=args.ici_tensor_parallelism,
        ici_expert_parallelism=args.ici_expert_parallelism,
        scan_layers=args.scan_layers,
        use_chunked_prefill=args.use_chunked_prefill,
        prefill_chunk_size=args.prefill_chunk_size,
        use_qwix_quantization=args.use_qwix_quantization,
        quantization=args.quantization,
        checkpoint_is_quantized=args.checkpoint_is_quantized,
        checkpoint_storage_use_zarr3=not args.checkpoint_is_quantized,
        mimo_fp8_weight_mode=getattr(args, "mimo_fp8_weight_mode", ""),
    )
    print("-" * 60)
    if tok_per_s is not None:
        print(f"Throughput: {tok_per_s:.1f} tok/s  [{scan_label}]")
    eos_status = "EOS fired (clean stop)" if eos_fired else "WARNING: EOS never fired — output likely garbled (model hit max_new_tokens limit)"
    print(f"Status:     {eos_status}")
    print(f"\nOutput:\n{output}")


if __name__ == "__main__":
    main()
