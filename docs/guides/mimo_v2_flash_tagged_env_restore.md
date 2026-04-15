# MiMo-V2-Flash Tagged Environment Restore And TPU Benchmark

This guide recreates the exact environment used for the tagged MiMo-V2-Flash
snapshot `mimo-v2-flash-2026-04-08`, including:

- the manager VM `jingnw-tpu-op`
- the TPU slice `jingnw-node`
- the tagged MaxText source tree on both the VM and TPU workers
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
- tag to restore: `mimo-v2-flash-2026-04-08`
- checkpoint for inference: `gs://jingnw-mimo-v2-flash-us-east5/mimo-v2-flash-fixed-ocdbt/checkpoints/0/items`
- tokenizer: `XiaomiMiMo/MiMo-V2-Flash`

## Important Notes

1. The tag `mimo-v2-flash-2026-04-08` exists locally and is published on this
  checkout's configured `origin` remote (`https://github.com/geeningwang/maxtext`).
  Do not use the upstream `AI-Hypercomputer/maxtext` GitHub repository for
  MiMo-V2-Flash related work.
2. Do not use `pkill` in this environment. If you must stop a process, find the
   exact PID and use `kill <pid>`.
3. For multi-worker SSH commands, run `ssh-add ~/.ssh/google_compute_engine`
  on `jingnw-tpu-op` first.
4. When polling a long-running benchmark, check every 20 to 30 seconds. Do not
   use long sleeps.

## 1. Set Local Shell Variables

Run this on `jingnw-tpu-op`:

```bash
export ZONE=us-east5-b
export TPU_NAME=jingnw-node
export TAG=mimo-v2-flash-2026-04-08
export CKPT=gs://jingnw-mimo-v2-flash-us-east5/mimo-v2-flash-fixed-ocdbt/checkpoints/0/items
export TOKENIZER=XiaomiMiMo/MiMo-V2-Flash

gcloud config set project tpu-launchpad-playground
if [[ ! -f "$HOME/.ssh/google_compute_engine" ]]; then
  ssh-keygen -t ed25519 -f "$HOME/.ssh/google_compute_engine" -N ""
fi
eval "$(ssh-agent -s)"
ssh-add ~/.ssh/google_compute_engine
```

## 2. Restore The Tagged Environment On The Ops VM

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

## 3. Restore The Tagged Environment On All TPU Workers

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

## 4. Smoke Test The Tagged JAX Demo

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

Expected result: the model prints a response for the default arithmetic prompt.

## 5. Run The TPU Performance Benchmark

The tagged demo does not print a direct `tok/s` line unless you preserve the
underlying decode timings. Use `--verbose` so the worker log contains the
`[TIME] generate_step_...` lines.

Run this on `jingnw-tpu-op`:

```bash
gcloud compute tpus tpu-vm ssh "$TPU_NAME" --zone "$ZONE" --worker=all --command='set -e
. "$HOME/maxtext/maxtext_tpu_venv/bin/activate"
cd "$HOME/maxtext"
export PYTHONUNBUFFERED=1
LOG="$HOME/mimo_v2_flash_demo_jax_verbose_$(date +%Y%m%d_%H%M%S).log"
python demos/mimo_v2_flash_demo_jax.py \
  --checkpoint_path '"$CKPT"' \
  --tokenizer_path '"$TOKENIZER"' \
  --ici_tensor_parallelism 4 \
  --ici_expert_parallelism 8 \
  --verbose 2>&1 | tee "$LOG"
echo "LOG_PATH=$LOG"'
```

This is the exact tagged demo path used for the TPU benchmark.

## 6. Poll The Benchmark

While the job is running, poll every 20 to 30 seconds from `jingnw-tpu-op`:

```bash
gcloud compute tpus tpu-vm ssh "$TPU_NAME" --zone "$ZONE" --worker=0 --command='set -e
ps -eo pid,etimes,pcpu,pmem,args | grep "python -m maxtext.inference.decode" | grep -v grep || true'
```

To inspect the tail of the current log on worker 0:

```bash
gcloud compute tpus tpu-vm ssh "$TPU_NAME" --zone "$ZONE" --worker=0 --command='set -e
ls -lt "$HOME"/mimo_v2_flash_demo_jax_verbose_*.log | head -n 1
tail -n 80 $(ls -t "$HOME"/mimo_v2_flash_demo_jax_verbose_*.log | head -n 1)'
```

## 7. Extract The Inference Tok/s Result

After the run finishes, parse the latest verbose log from worker 0:

```bash
gcloud compute tpus tpu-vm ssh "$TPU_NAME" --zone "$ZONE" --worker=0 --command='set -e
python3 - <<"PY"
import re
from pathlib import Path

logs = sorted(Path.home().glob("mimo_v2_flash_demo_jax_verbose_*.log"), key=lambda p: p.stat().st_mtime)
log = logs[-1]
text = log.read_text()

load = re.search(r"\[TIME\] load_params\s+host=.*? elapsed=([0-9.]+)s", text)
prefill = re.search(r"\[TIME\] prefill\s+host=.*? elapsed=([0-9.]+)ms", text)
total = re.search(r"\[TIME\] generate_total\s+host=.*? total=([0-9.]+)s steps=([0-9]+) avg_ms=([0-9.]+)", text)
steps = [float(x) for x in re.findall(r"\[TIME\] generate_step_\d+\s+host=.*? step_ms=([0-9.]+)", text)]

if not total or not steps:
    raise SystemExit(f"Could not parse timing lines from {log}")

steady_ms = sum(steps[1:]) / (len(steps) - 1)
steady_tok_s = (len(steps) - 1) / (sum(steps[1:]) / 1000)
end_to_end_tok_s = len(steps) / float(total.group(1))

print(f"log={log}")
print(f"load_params_s={float(load.group(1)) if load else 'n/a'}")
print(f"prefill_ms={float(prefill.group(1)) if prefill else 'n/a'}")
print(f"generate_steps={len(steps)}")
print(f"first_generate_step_ms={steps[0]:.3f}")
print(f"steady_state_mean_ms={steady_ms:.3f}")
print(f"steady_state_tok_per_s={steady_tok_s:.3f}")
print(f"end_to_end_generate_tok_per_s={end_to_end_tok_s:.3f}")
PY'
```

Interpretation:

- `steady_state_tok_per_s` is the practical TPU inference throughput after the
  one-time first-token compile cost.
- `end_to_end_generate_tok_per_s` includes the first-step compile penalty and
  is always much lower.

## 8. Reference Result For This Exact Setup

For the recreated environment on 2026-04-14, the tagged demo produced:

- `load_params`: about `32.1 s`
- `prefill` for `512` tokens: about `22.2 s`
- generated tokens: `128`
- first generate step: about `35.8 s`
- steady-state generate step: about `71.6 ms/token`
- steady-state inference throughput: about `14.0 tok/s`
- end-to-end generate throughput including the first-step compile: about `2.8 tok/s`

If your rerun is close to those numbers, the environment is behaving as
expected for this tagged snapshot.

## 9. Safe Cleanup

If you need to stop a running demo, identify the exact PID first, then use
`kill`, not `pkill`.

Inspect PIDs on all workers:

```bash
gcloud compute tpus tpu-vm ssh "$TPU_NAME" --zone "$ZONE" --worker=all --command='set -e
ps -eo pid,etimes,args | grep "mimo_v2_flash_demo_jax.py\|maxtext.inference.decode" | grep -v grep || true'
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

### Tag checkout fails on recreated hosts

Make sure you cloned from `https://github.com/geeningwang/maxtext.git`, not the
upstream `AI-Hypercomputer/maxtext` repository.

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

### The benchmark prints only model output and no timings

Re-run with `--verbose`. The non-verbose demo output is not sufficient for tok/s
calculation.