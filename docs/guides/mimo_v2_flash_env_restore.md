# MiMo-V2-Flash Environment Restore And TPU Benchmark

This guide recreates the MiMo-V2-Flash environment on the `MiMo-V2-Flash`
branch, including:

- the manager VM `jingnw-tpu-op`
- the TPU slice `jingnw-node`
- the MaxText source tree on both the VM and TPU workers
- the Python 3.12 TPU runtime environment
- the demo-based TPU inference throughput benchmark

The commands below assume you are already logged in to the manager VM
`jingnw-tpu-op` and run everything from that VM.

## Fixed Settings

- project: `tpu-launchpad-playground`
- zone: `us-east5-b`
- ops VM name: `jingnw-tpu-op`
- ops VM machine type: `e2-small`
- ops VM image: Debian 12 Bookworm
- ops VM boot disk: 10 GB
- TPU name: `jingnw-node`
- TPU accelerator type: `v6e-32`
- TPU runtime version: `v2-alpha-tpuv6e`
- network: `default`
- subnetwork: `default`
- checkpoint for inference (demo): `gs://jingnw-mimo-v2-flash-us-east5/mimo-v2-flash-fixed-ocdbt/checkpoints/0/items`
- checkpoint for benchmark (`scan_layers=false`): `gs://jingnw-mimo-v2-flash-us-east5/mimo-v2-flash-fixed-ocdbt/checkpoints/0/items`
- checkpoint for benchmark (`scan_layers=true`): `gs://jingnw-mimo-v2-flash-us-east5/mimo-v2-flash-4phase-stacked/checkpoints/0/items`
- tokenizer: `XiaomiMiMo/MiMo-V2-Flash`
- benchmark commit: latest `MiMo-V2-Flash` branch HEAD

### Runtime Package Versions (TPU Workers)

Verified on 2026-04-15 from worker 0 after a full install via `uv pip install -e ".[tpu]" --resolution=lowest`:

| Package | Version |
|---|---|
| Python | 3.12.13 |
| jax | 0.8.1 |
| jaxlib | 0.8.1 |
| flax | 0.12.1 |
| libtpu | 0.0.30 |
| orbax-checkpoint | 0.11.33 |
| numpy | 2.0.2 |
| transformers | 4.57.3 |
| tokenizers | 0.22.1 |
| sentencepiece | 0.2.1 |
| optax | 0.2.6 |
| chex | 0.1.91 |
| etils | 1.13.0 |
| grain | 0.2.15 |
| tensorstore | 0.1.79 |
| protobuf | 5.29.5 |
| grpcio | 1.76.0 |
| google-cloud-storage | 3.6.0 |
| google-cloud-aiplatform | 1.128.0 |

## Important Notes

1. Use the `MiMo-V2-Flash` branch in `geeningwang/maxtext`. Do not use the
   upstream `AI-Hypercomputer/maxtext` repository for MiMo-V2-Flash work.
2. Always pass `--worker=all` when running a JAX program on the TPU slice.
   Targeting a single worker will cause the collective to hang indefinitely.
3. Do not use `pkill` in this environment. If you must stop a process, find the
   exact PID and use `kill <pid>`.
4. For multi-worker SSH commands, run `ssh-add ~/.ssh/google_compute_engine`
  on `jingnw-tpu-op` first.
5. When polling a long-running benchmark, check every 20 to 30 seconds. Do not
   use long sleeps.
6. At startup, each worker prints exactly 8 lines (one per device) like
   `INTERNAL: CUDA error: Failed call to cuInit: UNKNOWN ERROR (303)`. These
   are harmless — JAX probes for CUDA, finds none on TPU workers, and falls
   back to TPU automatically. They can be ignored.

## 1. Set Local Shell Variables

Run this on `jingnw-tpu-op`:

```bash
export ZONE=us-east5-b
export TPU_NAME=jingnw-node
export TAG=MiMo-V2-Flash
export BENCH_COMMIT=origin/MiMo-V2-Flash
export CKPT=gs://jingnw-mimo-v2-flash-us-east5/mimo-v2-flash-fixed-ocdbt/checkpoints/0/items
export BENCH_CKPT=gs://jingnw-mimo-v2-flash-us-east5/mimo-v2-flash-fixed-ocdbt/checkpoints/0/items
export SCAN_CKPT=gs://jingnw-mimo-v2-flash-us-east5/mimo-v2-flash-4phase-stacked/checkpoints/0/items
export TOKENIZER=XiaomiMiMo/MiMo-V2-Flash

gcloud config set project tpu-launchpad-playground
if [[ ! -f "$HOME/.ssh/google_compute_engine" ]]; then
  ssh-keygen -t ed25519 -f "$HOME/.ssh/google_compute_engine" -N ""
  gcloud compute os-login ssh-keys add --key-file="$HOME/.ssh/google_compute_engine.pub"
fi
eval "$(ssh-agent -s)"
ssh-add ~/.ssh/google_compute_engine
```

## 2. Restore The Environment On The Ops VM

Run this on `jingnw-tpu-op`:

```bash
export PATH="$HOME/.local/bin:$PATH"
if ! command -v uv >/dev/null 2>&1; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
fi
cd "$HOME"
rm -rf "$HOME/maxtext"
git clone https://github.com/geeningwang/maxtext.git "$HOME/maxtext"
cd "$HOME/maxtext"
git fetch --tags --force
git checkout "$TAG"
uv venv --python 3.12 --seed "$HOME/maxtext/maxtext_tpu_venv"
. "$HOME/maxtext/maxtext_tpu_venv/bin/activate"
uv pip install -e ".[tpu]" --resolution=lowest
install_maxtext_tpu_github_deps
uv pip install transformers safetensors huggingface_hub
python -c "import maxtext; print(\"OPS_IMPORT_OK\")"
```

Notes:

- `uv venv --python 3.12` is intentional. The recreated hosts may not ship with
  Python 3.12 preinstalled.
- The ops VM does not run the distributed TPU job, but keeping a matching
  checkout there is useful for inspection and ad hoc commands.

## 3. Restore The Environment On All TPU Workers

> **Note:** All section 1 `export` variables (`TAG`, `TPU_NAME`, `ZONE`, etc.) must be
> live in your current shell. If you reconnected or opened a new terminal, re-run
> section 1 before continuing. The `'"$TAG"'` construct in the command below
> interpolates `$TAG` on the ops VM before the string is sent over SSH — if `TAG`
> is unset the checkout will silently target an empty ref and fail.

Run this on `jingnw-tpu-op`:

```bash
gcloud compute tpus tpu-vm ssh "$TPU_NAME" --zone "$ZONE" --worker=all --command='set -e
export PATH="$HOME/.local/bin:$PATH"
if ! command -v uv >/dev/null 2>&1; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
fi
rm -rf "$HOME/maxtext"
git clone https://github.com/geeningwang/maxtext.git "$HOME/maxtext"
cd "$HOME/maxtext"
git fetch --tags --force
git checkout '"$TAG"'
uv venv --python 3.12 --seed "$HOME/maxtext/maxtext_tpu_venv"
. "$HOME/maxtext/maxtext_tpu_venv/bin/activate"
uv pip install -e ".[tpu]" --resolution=lowest
install_maxtext_tpu_github_deps
uv pip install transformers safetensors huggingface_hub
python -c "import maxtext; print(\"TPU_IMPORT_OK\")"'
```

## 4. Smoke Test The JAX Demo

Run this on `jingnw-tpu-op`:

```bash
gcloud compute tpus tpu-vm ssh "$TPU_NAME" --zone "$ZONE" --worker=all --command='set -e
. "$HOME/maxtext/maxtext_tpu_venv/bin/activate"
cd "$HOME/maxtext"
python demos/mimo_v2_flash_demo_jax.py \
  --checkpoint_path '"$CKPT"' \
  --tokenizer_path '"$TOKENIZER"' \
  --ici_tensor_parallelism 4 \
  --ici_expert_parallelism 8'
```

Expected result: the model loads, runs prefill and generate, and exits with code 0 on all
workers. The generated text may be garbled or off-topic — this is a known limitation (see
Section 10 troubleshooting). The key check is that no crash or OOM occurs.

## 5. Run The Dedicated TPU Performance Benchmark

The dedicated benchmark script runs 3 warmup steps followed by 50 timed
`engine.generate()` steps and writes a JSON result file to the path given in
`inference_microbenchmark_log_file_path`.

### 5a. `scan_layers=false` benchmark (dense dispatch baseline)

Run this on `jingnw-tpu-op`:

```bash
gcloud compute tpus tpu-vm ssh "$TPU_NAME" --zone "$ZONE" --worker=all \
  --command='set -e
. "$HOME/maxtext/maxtext_tpu_venv/bin/activate"
cd "$HOME/maxtext"
export PYTHONUNBUFFERED=1
python3 -m maxtext.inference.scripts.mimo_v2_flash_bench \
  src/maxtext/configs/base.yml \
  model_name=mimo-v2-flash \
  run_name=mimo_v2_flash_bench \
  load_parameters_path='"$BENCH_CKPT"' \
  tokenizer_path='"$TOKENIZER"' \
  max_prefill_predict_length=512 \
  max_target_length=640 \
  per_device_batch_size=1 \
  dtype=bfloat16 \
  weight_dtype=bfloat16 \
  ici_tensor_parallelism=4 \
  ici_expert_parallelism=8 \
  scan_layers=false \
  attention=dot_product \
  checkpoint_storage_use_ocdbt=true \
  checkpoint_storage_use_zarr3=true \
  inference_microbenchmark_log_file_path=/tmp/bench_pdb1.json'
```

### 5b. `scan_layers=true` benchmark (4-phase stacked checkpoint)

Run this on `jingnw-tpu-op`:

```bash
gcloud compute tpus tpu-vm ssh "$TPU_NAME" --zone "$ZONE" --worker=all \
  --command='set -e
. "$HOME/maxtext/maxtext_tpu_venv/bin/activate"
cd "$HOME/maxtext"
export PYTHONUNBUFFERED=1
python3 -m maxtext.inference.scripts.mimo_v2_flash_bench \
  src/maxtext/configs/base.yml \
  model_name=mimo-v2-flash \
  run_name=mimo_v2_flash_scan_bench \
  load_parameters_path='"$SCAN_CKPT"' \
  tokenizer_path='"$TOKENIZER"' \
  max_prefill_predict_length=512 \
  max_target_length=640 \
  per_device_batch_size=1 \
  dtype=bfloat16 \
  weight_dtype=bfloat16 \
  ici_tensor_parallelism=4 \
  ici_expert_parallelism=8 \
  scan_layers=true \
  attention=dot_product \
  checkpoint_storage_use_ocdbt=true \
  checkpoint_storage_use_zarr3=true \
  async_checkpointing=false \
  inference_microbenchmark_log_file_path=/tmp/bench_scan.json'
```

### 5c. 4K context benchmark — dot_product attention (`per_device_batch_size=11`)

Current best 4K decode configuration. `per_device_batch_size=11` (total batch = 352)
achieves **2,724 tok/s** at 129.2 ms/step — a 4.72× improvement over the `pdb=1` baseline.
`pdb=12` OOMs (XLA HLO temp allocation); `pdb=11` is the maximum.

Run this on `jingnw-tpu-op`:

```bash
gcloud compute tpus tpu-vm ssh "$TPU_NAME" --zone "$ZONE" --worker=all \
  --command='set -e
. "$HOME/maxtext/maxtext_tpu_venv/bin/activate"
cd "$HOME/maxtext"
export PYTHONUNBUFFERED=1
python3 -m maxtext.inference.scripts.mimo_v2_flash_bench \
  src/maxtext/configs/base.yml \
  model_name=mimo-v2-flash \
  run_name=mimo_v2_flash_bench_pdb11 \
  load_parameters_path='"$BENCH_CKPT"' \
  tokenizer_path='"$TOKENIZER"' \
  max_prefill_predict_length=512 \
  max_target_length=640 \
  per_device_batch_size=11 \
  dtype=bfloat16 \
  weight_dtype=bfloat16 \
  ici_tensor_parallelism=4 \
  ici_expert_parallelism=8 \
  scan_layers=false \
  attention=dot_product \
  checkpoint_storage_use_ocdbt=true \
  checkpoint_storage_use_zarr3=true \
  inference_microbenchmark_log_file_path=/tmp/bench_pdb11.json'
```

Expected: decode median ~125.3 ms · **2,809 tok/s** (batch=352). Result file: `/tmp/bench_pdb11.json`.

### 5d. 16K context benchmark — flash attention (`per_device_batch_size=2`)

`attention=dot_product` OOMs at 16K prefill — the score matrix
`[4, 16, 16384, 16384]×2B ≈ 34 GB/chip` exceeds 31.25 GB HBM.
Flash attention tiles computation in VMEM; score matrix never materialises in HBM.
AR decode automatically falls back to `dot_product` (attends over 1 new token only;
no NxN score matrix). Both phases run in one job.

`per_device_batch_size=2` (BS=64) is the combined-benchmark HBM limit — after
`init_decode_state` uses 17.98 GB, `pdb=3` leaves only 2.83 GB free and the flash
prefill XLA binary needs 6.13 GB (`RESOURCE_EXHAUSTED`). SWA KV opt: only the 9
global attention layers need full 16K KV slots; the 39 SWA layers remain capped at
128 KV slots. Prefill always processes 1 sequence regardless of `pdb`, so `pdb=2`
gives identical prefill throughput to `pdb=1`.

> **`max_target_length=17408` is required** (not 16384). Must be strictly greater than
> `max_prefill_predict_length` and divisible by `EP × sa_block_q = 8 × 128 = 1024`.
> `17408 = 17 × 1024` satisfies both. Using `16384` raises `ValueError`.

Run this on `jingnw-tpu-op`:

```bash
gcloud compute tpus tpu-vm ssh "$TPU_NAME" --zone "$ZONE" --worker=all \
  --command='set -e
. "$HOME/maxtext/maxtext_tpu_venv/bin/activate"
cd "$HOME/maxtext"
export PYTHONUNBUFFERED=1
python3 -m maxtext.inference.scripts.mimo_v2_flash_bench \
  src/maxtext/configs/base.yml \
  model_name=mimo-v2-flash \
  run_name=mimo_v2_flash_bench_16k \
  load_parameters_path='"$BENCH_CKPT"' \
  tokenizer_path='"$TOKENIZER"' \
  max_prefill_predict_length=16384 \
  max_target_length=17408 \
  per_device_batch_size=2 \
  dtype=bfloat16 \
  weight_dtype=bfloat16 \
  ici_tensor_parallelism=4 \
  ici_expert_parallelism=8 \
  scan_layers=false \
  attention=flash \
  expert_shard_attention_option=context \
  sa_block_q=128 \
  sa_block_kv=128 \
  sa_block_kv_compute=128 \
  checkpoint_storage_use_ocdbt=true \
  checkpoint_storage_use_zarr3=true \
  inference_microbenchmark_log_file_path=/tmp/bench_16k.json'
```

Expected: decode median ~61.3 ms · **1,044 tok/s** (batch=64); prefill median ~3,497 ms TTFT · **4,686 tok/s**.
Result file: `/tmp/bench_16k.json` (contains both `decode` and `prefill` keys).

Expected progress markers printed to stdout as the job runs:

```
[BENCH] load_params: <N>s
[BENCH] decode_state initialised
[BENCH] decode warmup (3 steps) ...
[BENCH] decode warmup done
[BENCH] timing 50 decode steps ...
[BENCH] prefill benchmark (seq_len=16384) ...
[BENCH] prefill warmup (3 calls) ...
[BENCH] prefill warmup done
[BENCH] timing 20 prefill calls ...
```

The decode phase takes several minutes; the prefill phase adds ~70 s (20 × 3.5 s).
Output from all 8 workers appears interleaved.

## 6. Poll The Benchmark

While the job is running, verify processes are alive on all workers every 20 to
30 seconds from `jingnw-tpu-op`:

```bash
gcloud compute tpus tpu-vm ssh "$TPU_NAME" --zone "$ZONE" --worker=all \
  --command='ps -eo pid,etimes,pcpu,args | grep mimo_v2_flash_bench | grep -v grep || true'
```

## 7. Read The Benchmark Results

After the run completes (all workers print `BENCH_EXIT=0`), read the JSON
result from each worker:

```bash
gcloud compute tpus tpu-vm ssh "$TPU_NAME" --zone "$ZONE" --worker=all \
  --command='cat /tmp/bench_16k.json 2>/dev/null || echo "no result yet"'
```

Key fields in the JSON output:

| Field | Meaning |
|---|---|
| `step_ms_median` | median per-step latency in milliseconds |
| `step_ms_min` | minimum observed step latency |
| `step_ms_p90` | 90th-percentile step latency |
| `throughput_tok_per_s` | decoded tokens per second across all devices |
| `batch_size` | total batch slots across all 32 devices |

## 8. Reference Results

Default settings unless noted: `jingnw-node` (v6e-32),
`ici_tensor_parallelism=4`, `ici_expert_parallelism=8`.
See individual subsections for `per_device_batch_size`, `max_target_length`, and `attention` values.

> **Note on `load_params` time:** The first run after a full environment restore
> hits a cold GCS cache and takes noticeably longer (~40 s). Subsequent runs on
> the same cluster use a warm cache and return to the ~29–30 s baseline.

### 2026-04-17 — Commit 1a6b9579 (cold GCS cache, post-restore)

- `load_params`: about `40.0–40.6 s` (cold GCS cache)
- HBM after decode-state init: `17.98 GB / 31.25 GB` per device
- timed steps: `50`
- step latency (mean): about `55.7 ms`
- step latency (median): about `55.7 ms`
- step latency (min): about `55.3 ms`
- step latency (p90): about `55.8 ms`
- total throughput: about `575 tok/s` (batch=32)

### 2026-04-17 — Commit 72f75972 (SWA fix + debug cleanup, dense baseline re-confirmed)

checkpoint: `mimo-v2-flash-fixed-ocdbt`

- `load_params`: about `29–30 s` (warm GCS cache)
- HBM after decode-state init: `17.98 GB / 31.25 GB` per device
- timed steps: `50`
- step latency (median): about `55.5 ms`
- step latency (min): about `55.2 ms`
- total throughput: about `576 tok/s` (batch=32)
- per-sequence latency: about `1.7 ms/tok/seq`

### 2026-04-17 — Commit 539cc043 (`scan_layers=true`, stacked checkpoint, bench-only)

> **Note:** The 4-phase decoder code was lost in revert `30fd5e55`.  This
> benchmark timing is valid but the demo path produces garbled output.

checkpoint: `mimo-v2-flash-4phase-stacked`

- `load_params`: about `34.4–34.5 s` (warm GCS cache)
- HBM after decode-state init: `17.98 GB / 31.25 GB` per device
- timed steps: `50`
- step latency (mean): about `68.3 ms`
- step latency (median): about `68.3 ms`
- step latency (min): about `68.0 ms`
- step latency (p90): about `68.4 ms`
- total throughput: about `468 tok/s` (batch=32)
- per-sequence latency: about `2.1 ms/tok/seq`

Results from all 8 workers are nearly identical, which is expected for a
synchronous collective workload.

### 2026-04-20 — Post-opt4 revert (dense dispatch, `scan_layers=true`, decode + prefill)

checkpoint: `mimo-v2-flash-4phase-stacked`

After reverting the ragged-A2A sparse MoE dispatch (opt4, see plan post-mortem),
the benchmark now measures both AR decode and prefill in a single run.

**AR Decode:**
- `load_params`: about `34.7 s`
- timed steps: `50`
- step latency (median): about `68.4 ms`
- step latency (min): about `68.3 ms`
- total throughput: about `468 tok/s` (batch=32)

**Prefill (seq_len=512):**
- timed calls: `20`
- step latency (median): about `121.9 ms`
- step latency (min): about `121.8 ms`
- total throughput: about `4,200 tok/s`

The decode result matches the prior `scan_layers=true` baseline (68.3 ms),
confirming the revert is clean.  The opt4 ragged-A2A implementation measured
101.5 ms — an 83% regression caused by ICI collective overhead with small
T=32 during AR decode (see opt4 plan doc for full post-mortem).

### 2026-04-20 — Post-opt4 revert (dense dispatch, `scan_layers=false`, decode + prefill)

checkpoint: `mimo-v2-flash-fixed-ocdbt`

Re-confirmed dense no-scan baseline on commit `055a4c2d` after full environment
restore.  Both `scan_layers` modes verified working on the same commit.

**AR Decode:**
- `load_params`: about `40.0 s` (cold GCS cache after restore)
- timed steps: `50`
- step latency (median): about `55.4 ms`
- step latency (min): about `55.3 ms`
- total throughput: about `577.5 tok/s` (batch=32)

**Prefill (seq_len=512):**
- timed calls: `20`
- step latency (median): about `123.6 ms`
- step latency (min): about `123.4 ms`
- total throughput: about `4,144 tok/s`

### 2026-04-21 — Opt5 batch size scaling (`per_device_batch_size=11`, 4K context)

Commit `711f591f`. Sweep `pdb=1→11`; OOM at `pdb=12`.

**AR Decode (4K context, pdb=11, batch=352):**
- HBM after decode-state init: ~18.9 GB / 31.25 GB per device
- timed steps: `50`
- step latency (median): about `129.2 ms`
- step latency (min): about `128.8 ms`
- total throughput: about `2,724 tok/s`
- per-sequence latency: about `0.37 ms/tok/seq`

**Prefill (512 tokens, unchanged — compute-bound, independent of decode batch):**
- step latency (median): about `123.6 ms`
- total throughput: about `4,144 tok/s`

### 2026-04-22 — Full suite re-run on commit `1961e3ee` (doc-only change, warm cache)

checkpoint: `mimo-v2-flash-fixed-ocdbt`

**5a — `scan_layers=false`, `attention=dot_product`, pdb=1:**
- AR Decode: median `55.2 ms`, throughput `579.8 tok/s` (batch=32)
- Prefill (512 tok): median `123.8 ms`, throughput `4,136 tok/s`

**5c — `scan_layers=false`, `attention=dot_product`, pdb=11:**
- AR Decode: median `125.3 ms`, throughput `2,809 tok/s` (batch=352)
- Prefill (512 tok): median `123.9 ms`, throughput `4,131 tok/s`

**5d — `scan_layers=false`, `attention=flash`, pdb=2, 16K context:**
- AR Decode: median `61.1 ms`, throughput `1,047 tok/s` (batch=64)
- Prefill (16K tok): median `3,496.8 ms` TTFT, throughput `4,685 tok/s`

### 2026-04-22 — 16K prefill with flash/splash attention

Commit `ebfcd749` (plus `9509e9e9`, `d8fe9fc9`). Three code fixes were required;
see perf doc opt #12 for full details.

**16K Prefill (`attention=flash`, `expert_shard_attention_option=context`, pdb=1):**
- timed steps: `20`
- step latency (median): about `3,497 ms` TTFT
- total throughput: about `4,686 tok/s`

### 2026-04-22 — 16K decode (`per_device_batch_size=2`)

Commit `ebfcd749`. SWA KV opt: only 9 global attention layers need full 16K KV;
39 SWA layers remain capped at 128 KV slots. `pdb=3` OOMs (flash prefill XLA
binary needs 6.13 GB; after `init_decode_state` only 2.83 GB free).

**AR Decode (16K context, pdb=2, batch=64):**
- step latency (median): about `61.3 ms`
- total throughput: about `1,044 tok/s`
- per-sequence latency: about `0.96 ms/tok/seq`

### Summary of all validated baselines (as of 2026-04-22, commit `ebfcd749`)

| Config | Context | `pdb` (BS) | Decode median | Decode throughput | Prefill throughput |
|---|---|---|---|---|---|
| `scan_layers=false`, `attention=dot_product` | 512 tok | 1 (32) | 55.4 ms | 577.5 tok/s | 4,144 tok/s |
| `scan_layers=true`, `attention=dot_product` | 512 tok | 1 (32) | 68.4 ms | 468 tok/s | 4,200 tok/s |
| **`scan_layers=false`, `attention=dot_product`** | **512 tok** | **11 (352)** | **125.3 ms** | **2,809 tok/s** | same as above |
| `scan_layers=false`, `attention=dot_product` | 16K tok | 2 (64) | 61.3 ms | 1,044 tok/s | 4,686 tok/s (`attention=flash`) |

### Prior Reference Result For Commit 5ad76eac (regression baseline)

Measured on 2026-04-15 with the same configuration. This commit contained the
32× performance regression caused by the dense MoE fallback (see perf doc):

- `load_params`: about `27.2 s`
- HBM after decode-state init: `17.98 GB / 31.25 GB` per device
- timed steps: `50`
- step latency (median): about `1757 ms`
- total throughput: about `18.2 tok/s` (batch=32)
- per-sequence latency: about `54.9 ms/tok`

## 9. Safe Cleanup

If you need to stop a running job, identify the exact PID first, then use
`kill`, not `pkill`.

Inspect PIDs on all workers:

```bash
gcloud compute tpus tpu-vm ssh "$TPU_NAME" --zone "$ZONE" --worker=all --command='set -e
ps -eo pid,etimes,args | grep "mimo_v2_flash_demo_jax.py\|mimo_v2_flash_bench\|maxtext.inference" | grep -v grep || true'
```

Stop explicit PIDs on all workers safely:

Replace `<pid1> <pid2> ...` with the exact PIDs shown for each worker.

```bash
gcloud compute tpus tpu-vm ssh "$TPU_NAME" --zone "$ZONE" --worker=all --command='set -e
PIDS="<pid1> <pid2> ..."
if [[ -n "$PIDS" ]]; then
  kill $PIDS
fi'
```

## 10. Troubleshooting

### `uv venv --python 3.12` fails

Re-run after ensuring `uv` is on `PATH`:

```bash
export PATH="$HOME/.local/bin:$PATH"
```

### SSH returns code `255`

Refresh the SSH agent and key on `jingnw-tpu-op`:

```bash
if [[ ! -f "$HOME/.ssh/google_compute_engine" ]]; then
  ssh-keygen -t ed25519 -f "$HOME/.ssh/google_compute_engine" -N ""
fi
eval "$(ssh-agent -s)"
ssh-add ~/.ssh/google_compute_engine
gcloud compute os-login ssh-keys add --key-file="$HOME/.ssh/google_compute_engine.pub"
```

### The benchmark script prints model output but no `[BENCH]` timing markers

You may be running the demo script (`demos/mimo_v2_flash_demo_jax.py`) instead
of the dedicated benchmark module. The benchmark is invoked as:

```bash
python3 -m maxtext.inference.scripts.mimo_v2_flash_bench ...
```

The benchmark module always prints `[BENCH]` markers and does not accept a
`--verbose` flag.

### The demo script (`mimo_v2_flash_demo_jax.py`) produces garbled or repetitive output

This is a known limitation on current HEAD. The demo uses nucleus sampling
(top_p=0.95, temperature=0.6) with no EOS stopping condition, so the model
degenerates into repetitive tokens once it passes the natural end of its answer.
Example: the response starts correctly then repeats phrases like
`取得以及 Ağust取得以及...` indefinitely.

This does **not** affect benchmark measurements — the benchmark script
(`mimo_v2_flash_bench`) measures `engine.generate()` step latency, not output
quality. The demo is only useful for verifying that the model loads and produces
non-crash output; treat the generated text as a sanity-check only.