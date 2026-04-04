# MiMo-V2-Flash Garbled-Output Comparison Framework

Tooling to isolate why MaxText/TPU produces garbled tokens from MiMo-V2-Flash
when the same prompt on HuggingFace CPU gives correct output.

## Files

| File | Purpose |
|------|---------|
| `setup.sh` | One-time worker bootstrap (pip install, gcsfuse mount) |
| `maxtext_reference.py` | MaxText/TPU inference — saves per-step logits + token IDs |
| `hf_reference.py` | HuggingFace CPU inference — saves per-step logits + hidden states |
| `compare.py` | Numerically compares the two output sets, reports first divergence |

## Infrastructure

- **Cluster**: `jingnw-node`, zone `us-east5-b`, 8 workers × 4 chips v6e = 32 TPUs total
- **Always SSH with `--internal-ip`**; never `kill`/`pkill` on workers; no `sleep` > 90 s
- **GCS bucket**: `gs://jingnw-mimo-v2-flash-us-east5/`
  - `hf-model/` — HuggingFace safetensors + tokenizer (291 GiB)
  - `mimo-v2-flash-ocdbt/checkpoints/0/items/` — MaxText OCDBT checkpoint
- **gcsfuse** mount on workers: `gs://jingnw-mimo-v2-flash-us-east5` → `/tmp/mimo-hf-gcs/`
- **Tokenizer**: `$HOME/mimo-tokenizer` (copied from GCS, gzip-decompressed)
- **venv**: `$HOME/maxtext/maxtext_venv/` — all Python deps installed
- **Branch**: `MiMo-V2-Flash` (HEAD `32e96042`)
- **JAX process 0 is worker-2** (writes comparison files)

## Current Status — April 2026

### MaxText/TPU reference ✅ Complete

`maxtext_reference.py` runs cleanly on all 8 workers. Outputs on **worker-2**:
- `/tmp/compare_tpu/tokens.json` — 18 generated token IDs
- `/tmp/compare_tpu/step{0000-0017}_logits.npy` — float32 logits per step

**Confirmed garbled output** (both test prompts):

```
Prompt: "What is 1+1?"
Generated token IDs: [120883, 120883, 132532, 137323, 120883, 142203, 132532,
                      24513, 120883, 116190, 120883, 140793, 114323, 143738,
                      120883, 140793, 120883, 132533]
Decoded:  '葭葭społec почем葭słuchspołecscar葭線上葭padł當您在 มกร葭padł葭społeczn'

Top-3 at step 0:
  120883 '葭'             17.50   ← winner
  138173 'Cumhur'         15.19
  138176 'Cumhurbaşkan'   14.44
```

Expected answer: `"2"` or `"1+1=2"`. Garbled tokens dominate the logits by 2+
points, so this is not an argmax/sampling issue — the probability mass itself
is mis-distributed.

**Inference perf** (v6e-32, bfloat16, tensor_parallel=4, expert_parallel=8):
- Weight load: ~30 s (17.98 GB/chip, 57.5% HBM)
- Prefill 512 tokens: ~25 s (JIT compile included)
- Generate step JIT (first): ~39 s
- Generate step steady-state: ~69 ms/step

### HF CPU reference ⏳ Not yet run

Ready to launch. Will stream weights from gcsfuse — expected 1–4 hours
for first token on worker CPU (708 GB RAM available).

```bash
# Run from controller, pick any worker (e.g. worker-2 since it's process-0):
gcloud compute tpus tpu-vm ssh jingnw-node --zone=us-east5-b --worker=2 --internal-ip \
  --command='nohup /usr/bin/python3 $HOME/maxtext/demos/compare/hf_reference.py \
      --model_path /tmp/mimo-hf-gcs/hf-model \
      --tokenizer_path $HOME/mimo-tokenizer \
      --prompt "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\nWhat is 1+1?<|im_end|>\n<|im_start|>assistant\n" \
      --max_new_tokens 18 \
      --out_dir /tmp/compare_hf \
    > /tmp/mimo_compare_hf.log 2>&1 < /dev/null & echo "PID $!"'
# Tail log:
gcloud compute tpus tpu-vm ssh jingnw-node --zone=us-east5-b --worker=2 --internal-ip \
  --command='tail -20 /tmp/mimo_compare_hf.log'
```

### Comparison ⏳ Blocked on HF reference

```bash
gcloud compute tpus tpu-vm ssh jingnw-node --zone=us-east5-b --worker=2 --internal-ip \
  --command='/usr/bin/python3 $HOME/maxtext/demos/compare/compare.py \
      --hf_dir /tmp/compare_hf --tpu_dir /tmp/compare_tpu'
```

Expected output: first step where KL-divergence or cosine distance between HF
and TPU logits exceeds threshold — that step number will point to the layer
or operation causing the divergence.

## Known Fixes Applied

| Commit | Fix |
|--------|-----|
| `791bdca7` | OOM in `load_params()` — axis rule ordering in `inference.yml` |
| `39d9fb83` | SWA `next_pos` anchored at `kv_seq_len-1` instead of decode pos |
| `32e96042` | `process_allgather(sampled_tokens.data)` — globally-sharded token IDs |
| `efb65ad7` | `process_allgather(decode_state["logits"])` — globally-sharded logits |
| `4cd002c2` | Tokenizer gzip files decompressed after `gcloud storage cp` |
| `5ae6fe2f` | Custom arch `.py` files downloaded from HF Hub |
| `c1d5a2c3` | `quantization_config` stripped from `config.json` for plain bfloat16 |
| `0f18592b` | `ROPE_INIT_FUNCTIONS['default']` shim for transformers 5.x |

## Root Cause Hypothesis

The garbled output is not sampling noise — the top logit in the garbled
direction beats the correct token by 2–9 bfloat16 points consistently.
Likely candidates:

1. **Weight conversion bug** — OCDBT checkpoint was converted from safetensors;
   a transposition, dtype cast, or shard assignment error in one or more layers
   could corrupt activations from that layer forward.

2. **Rotary embedding (RoPE) mismatch** — MiMo-V2-Flash uses a non-standard
   `rope_scaling` config; if the theta or scaling is applied differently between
   HF and MaxText the attention scores diverge.

3. **MoE gating** — gate logits look plausible (256 experts confirmed) but the
   routing weights or top-k dispatch could differ.

The comparison run will show at which layer/step MaxText first diverges from HF.
