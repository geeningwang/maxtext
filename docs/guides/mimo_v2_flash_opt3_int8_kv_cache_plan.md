# MiMo-V2-Flash — Int8 KV Cache Quantization Plan

This runbook defines a complete, reproducible path to evaluate and (if safe)
adopt int8 KV-cache quantization for MiMo-V2-Flash TPU inference.

## Experiment Outcome (2026-04-13)

Status: **Rejected for default enablement**.

Summary of measured results:
1. `scan_layers=true` + stacked checkpoint with KV-int8 is stable (no OOM/crash).
2. Quality gate failed on harmonic-mean prompt:
   - KV OFF: `80 km/h`
   - KV ON: `64 km/h`
3. Benchmark regressed:
   - KV OFF: median `56.1 ms`, throughput `~570.2 tok/s`
   - KV ON: median `60.1 ms`, throughput `~532.7 tok/s`
   - Delta: about `+7.1%` slower latency, about `-6.6%` lower throughput.

Decision:
1. Keep KV-int8 **disabled by default**.
2. Keep optional flag support for controlled experiments only.

Scope:
- Inference-only optimization (no training changes)
- Primary target: decode-stage latency and throughput
- Validation target: maintain response quality on known prompts

Environment assumptions:
- Cluster: `jingnw-node` (`us-east5-b`, v6e-32)
- Parallelism: TP=4, EP=8
- Flat checkpoint:
  `gs://jingnw-mimo-v2-flash-us-east5/mimo-v2-flash-fixed-ocdbt/checkpoints/0/items`
- Stacked checkpoint:
  `gs://jingnw-mimo-v2-flash-us-east5/mimo-v2-flash-4phase-stacked/checkpoints/0/items`

---

## Why This Optimization

During autoregressive decode, a major cost is repeatedly reading KV cache from
HBM. Quantizing KV from bf16 to int8 can reduce KV read bandwidth and improve
token latency, especially in decode-heavy workloads.

Expected upside:
- Lower per-step latency in steady-state generate
- Better throughput at the same batch size
- Potentially lower HBM pressure

Main risks:
- Accuracy drift on reasoning prompts
- Potential incompatibility with specific decode paths/config combinations

---

## Success Criteria

Adopt by default only if all are true:
1. `scan_layers=true` path runs stably with no `RESOURCE_EXHAUSTED` / crash.
2. Median decode step latency improves by >= 5% vs current baseline.
3. No meaningful quality regressions on the prompt gate below.

If criteria are mixed:
- Keep behind a CLI/config flag and document as workload-dependent.

---

## Prompt Gate (Quality)

Minimum prompts to compare before/after:
1. `What is 2+2?` (expect equivalent answer, e.g., `2 + 2 = 4`)
2. Harmonic-mean train problem (expect equivalent answer: `80 km/h`)

Optional expansion:
1. 5–10 reasoning prompts from existing MiMo validation set
2. Compare not only final answer, but consistency and hallucination rate

---

## Step-by-Step Execution Plan

### Step 0 — Freeze current baseline reference

Goal: ensure clean A/B attribution.

Actions:
1. Record current branch + commit:
   - `git rev-parse --short HEAD`
2. Record current benchmark reference (already observed):
   - median `56.1 ms`, throughput `~570 tok/s`, batch `32`.
3. Keep all benchmark knobs unchanged for A/B:
   - same cluster, TP/EP, batch, warmup/timed steps, checkpoint, and prompt length.

Output artifact:
- A baseline row in notes with exact command + metrics.

---

### Step 1 — Add KV quantization knobs to inference command construction

Goal: make feature controllable and reproducible.

Primary file:
- `demos/mimo_v2_flash_demo_jax.py`

Required decode flags when enabled:
1. `quantize_kvcache=true`
2. `kv_quant_dtype=int8`
3. `kv_quant_axis=heads_and_dkv`

Implementation guidance:
1. Add CLI switches (recommended):
   - `--quantize_kvcache` (bool)
   - `--kv_quant_dtype` (default `int8`)
   - `--kv_quant_axis` (default `heads_and_dkv`)
2. Append the three decode flags only when kv quantization is enabled.
3. Keep default behavior unchanged (off by default) until validation completes.

Rationale:
- Allows fast A/B without editing code repeatedly.
- Makes results reproducible from command lines and logs.

---

### Step 2 — Fast smoke test (`scan_layers=false`, flat checkpoint)

Goal: detect immediate functional issues early.

Run:
1. Decode with kv quantization OFF.
2. Decode with kv quantization ON.
3. Use the same short prompts (`2+2`, harmonic mean).

Checks:
1. Process exits successfully.
2. EOS behavior is normal.
3. Outputs remain semantically equivalent.

Stop condition:
- Any crash, NaN behavior, or obvious answer degradation -> fix/rollback before moving on.

---

### Step 3 — Full-path validation (`scan_layers=true`, stacked checkpoint)

Goal: ensure intended production path is stable.

Run:
1. `scan_layers=true`
2. stacked checkpoint
3. kv quantization ON

Checks:
1. No `RESOURCE_EXHAUSTED` or runtime crash.
2. Decode begins and generates cleanly.
3. HBM usage remains within safe headroom.

Record:
1. min/max `used` and `peak` HBM from `[HBM]` lines.
2. any notable step-time anomalies.

---

### Step 4 — Benchmark A/B (authoritative measurement)

Goal: quantify steady-state impact.

Tool:
- `python3 -m maxtext.inference.scripts.mimo_v2_flash_bench ...`

Protocol:
1. Run benchmark with kv quant OFF (control).
2. Run benchmark with kv quant ON (treatment).
3. Keep all non-kv knobs identical:
   - warmup 3, timed 50, batch 32, TP=4 EP=8, same checkpoint, same scan mode.

Metrics to compare:
1. `step_ms_median`
2. `step_ms_p90`
3. `throughput_tok_per_s`
4. optional: HBM `used/peak`

Acceptance threshold (recommended):
1. median improves by >= 5%
2. no quality regressions from prompt gate

---

### Step 5 — Quality gate decision

Goal: avoid speed-only regressions.

Decision table:
1. Faster + quality stable -> eligible to enable by default.
2. Faster + quality degraded -> keep behind flag; investigate quant axis/dtype.
3. No speedup + quality stable -> optional, leave off by default.
4. Slower or unstable -> reject for now.

---

### Step 6 — Rollout and fallback strategy

If accepted:
1. Enable kv quantization by default in demo/inference path used for benchmarking.
2. Preserve a disable flag for quick rollback.
3. Announce exact rollback command (`--quantize_kvcache=false` or removal of flags).

If not accepted:
1. Keep feature optional behind CLI flag.
2. Document observed trade-offs and when it might still help.

---

### Step 7 — Documentation updates

Update these docs after experiment:
1. `docs/guides/mimo_v2_flash_tpu_perf_optimization.md`
   - Add A/B result row(s) for kv quantization.
   - Record quality outcome and final decision.
2. `docs/guides/mimo_v2_flash_opt2_sparse_moe_plan.md` (optional note)
   - Link to follow-up optimization status if needed.

---

## Command Templates

### A) Decode smoke test (flat checkpoint, scan_layers=false)

```bash
gcloud compute tpus tpu-vm ssh jingnw-node --zone=us-east5-b --worker=all \
  --command='set -e; export PATH="$HOME/.local/bin:$PATH"; \
    source ~/maxtext/maxtext_tpu_venv/bin/activate; cd ~/maxtext; \
    python3 demos/mimo_v2_flash_demo_jax.py \
      --checkpoint_path gs://jingnw-mimo-v2-flash-us-east5/mimo-v2-flash-fixed-ocdbt/checkpoints/0/items \
      --tokenizer_path XiaomiMiMo/MiMo-V2-Flash \
      --prompt "What is 2+2?" \
      --max_new_tokens 64 --max_prefill 128 \
      --ici_tensor_parallelism 4 --ici_expert_parallelism 8 \
      --quantize_kvcache'
```

### B) Full-path validation (stacked checkpoint, scan_layers=true)

```bash
gcloud compute tpus tpu-vm ssh jingnw-node --zone=us-east5-b --worker=all \
  --command='set -e; export PATH="$HOME/.local/bin:$PATH"; \
    source ~/maxtext/maxtext_tpu_venv/bin/activate; cd ~/maxtext; \
    python3 demos/mimo_v2_flash_demo_jax.py \
      --checkpoint_path gs://jingnw-mimo-v2-flash-us-east5/mimo-v2-flash-4phase-stacked/checkpoints/0/items \
      --tokenizer_path XiaomiMiMo/MiMo-V2-Flash \
      --prompt "What is 2+2?" \
      --max_new_tokens 64 --max_prefill 128 \
      --ici_tensor_parallelism 4 --ici_expert_parallelism 8 \
      --scan_layers --quantize_kvcache'
```

### C) Benchmark (stacked checkpoint, scan_layers=true)

```bash
gcloud compute tpus tpu-vm ssh jingnw-node --zone=us-east5-b --worker=all \
  --command='set -e; export PATH="$HOME/.local/bin:$PATH"; \
    source ~/maxtext/maxtext_tpu_venv/bin/activate; cd ~/maxtext; \
    python3 -m maxtext.inference.scripts.mimo_v2_flash_bench \
      src/maxtext/configs/base.yml \
      model_name=mimo-v2-flash \
      run_name=mimo_int8_kv_bench \
      load_parameters_path=gs://jingnw-mimo-v2-flash-us-east5/mimo-v2-flash-4phase-stacked/checkpoints/0/items \
      tokenizer_path=XiaomiMiMo/MiMo-V2-Flash tokenizer_type=huggingface \
      per_device_batch_size=1 max_prefill_predict_length=128 max_target_length=144 \
      dtype=bfloat16 weight_dtype=bfloat16 \
      ici_tensor_parallelism=4 ici_expert_parallelism=8 \
      scan_layers=true param_scan_axis=0 attention=dot_product \
      checkpoint_storage_use_ocdbt=true checkpoint_storage_use_zarr3=true \
      quantize_kvcache=true kv_quant_dtype=int8 kv_quant_axis=heads_and_dkv'
```

---

## Logging Checklist

For each run, capture:
1. Commit SHA
2. Exact command
3. Median/mean/p90 latency and throughput
4. HBM used/peak and limit
5. Prompt outputs (for quality gate)
6. Final decision: on-by-default / flag-only / rejected

---

## Current Recommendation

For the current MiMo setup on v6e-32, do **not** enable KV-int8 by default.

Next recommended work:
1. Prioritize SWA KV cache truncation (per-layer cache length).
2. Remove per-step host syncs (`effects_barrier`/synchronous token fetch) in decode loop.
3. Revisit KV-int8 only after those changes, and retest with the same prompt gate and A/B benchmark protocol.
