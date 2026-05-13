# MiMo-V2.5-Pro — Architecture Comparison & Bringup Notes

Model card: https://huggingface.co/XiaomiMiMo/MiMo-V2.5-Pro  
Branch: `MiMo-V2.5-Pro`  
Date: 2026-05-13

---

## Architecture diff vs MiMo-V2-Flash

| | MiMo-V2-Flash | MiMo-V2.5-Pro | Delta |
|---|---|---|---|
| **Total params** | 309B | 1.02T | 3.3× |
| **Active params** | 15B | 42B | 2.8× |
| **Hidden size** | 4096 | 6144 | 1.5× |
| **Layers** | 48 | 70 | +22 |
| **GA layers** | 9 | 10 | +1 |
| **SWA layers** | 39 | 60 | +21 |
| **Num attn heads** | 64 | 128 | 2× |
| **GA KV heads** | 4 | 8 | 2× |
| **SWA KV heads** | 8 | 8 | same |
| **Head dim (Q/K / V)** | 192 / 128 | 192 / 128 | same |
| **Routed experts** | 256 | 384 | 1.5× |
| **Experts/token** | 8 | 8 | same |
| **MoE intermediate** | 2048 | 2048 | same |
| **Dense intermediate** | 16384 (layer 0 only) | 16384 (layer 0 only) | same |
| **rope_theta (GA)** | 5,000,000 | 10,000,000 | 2× |
| **attention_value_scale** | 0.707 | 0.612 | changed |
| **SWA window** | 128 | 128 | same |
| **Max context** | 1M tokens | 1M tokens | same |
| **MTP layers** | 3 | 3 | same |
| **Vocab size** | 152,576 | 152,576 | same |
| **Native weight dtype** | FP8 E4M3 block-wise | FP8 E4M3 block-wise | same |
| **HF checkpoint size** | ~206 GiB (est.) | **962.4 GiB** | ~4.7× |

Source: `config.json` and `model.safetensors.index.json` from HuggingFace (fetched 2026-05-13).

---

## New config fields (not present in MiMo-V2-Flash)

| Field | Value | Implication |
|---|---|---|
| `attention_projection_layout` | `"fused_qkv"` | QKV weights stored as a single fused tensor per layer; checkpoint converter must split into q/k/v |
| `add_full_attention_sink_bias` | `false` | explicit (was implicit in V2-Flash config) |
| `swa_v_head_dim` | 128 | now explicit in config (V2-Flash derived it implicitly) |
| `swa_head_dim` | 192 | now explicit |
| `swa_num_attention_heads` | 128 | now explicit |

The most impactful change is **`attention_projection_layout: "fused_qkv"`**. In MiMo-V2-Flash the HF checkpoint stored separate `q_proj`, `k_proj`, `v_proj` weight tensors. In V2.5-Pro they are stored as a single fused `qkv_proj` tensor of shape `[hidden_size, (num_heads * head_dim_q) + (num_kv_heads * head_dim_k) + (num_kv_heads * v_head_dim)]`. The checkpoint converter needs a new split path before mapping to MaxText's separate Q/K/V parameter arrays.

**GA KV heads changed from 4 → 8**: MiMo-V2-Flash had asymmetric KV heads (4 for GA, 8 for SWA). V2.5-Pro uses 8 for both, simplifying the attention config.

---

## Hybrid attention layer pattern (70 layers)

```
hybrid_layer_pattern: [
  0, 1, 1, 1, 1, 1, 1,   # GA at 0
  0, 1, 1, 1, 1, 1, 1, 1, # GA at 7
  0, 1, 1, 1, 1, 1, 1, 1, # GA at 15
  0, 1, 1, 1, 1, 1, 1, 1, # GA at 23
  0, 1, 1, 1, 1, 1, 1, 1, # GA at 31
  0, 1, 1, 1, 1, 1, 1, 1, # GA at 39
  0, 1, 1, 1, 1, 1, 1, 1, # GA at 47
  0, 1, 1, 1, 1, 1, 1,   # GA at 55
  0, 1, 1, 1, 1, 1, 1,   # GA at 62
  0                        # GA at 69
]
```

10 GA layers at positions: 0, 7, 15, 23, 31, 39, 47, 55, 62, 69.  
60 SWA layers everywhere else.

MoE layer freq: layer 0 dense, layers 1–69 all sparse MoE (69 MoE layers).

---

## HF weights → GCS

**Total HF size:** 962.4 GiB (FP8, TP=8 pre-sharded, 34 safetensors files + 1 index)

Shard layout:
- `model_mtp.safetensors` — 3 MTP (Multi-Token Prediction) decoder heads
- `model_pp0_ep0_shard0/1.safetensors` — backbone + expert shard 0 (2 files; shard1 holds `lm_head` and embeddings)
- `model_pp0_ep1_shard0.safetensors` … `model_pp0_ep31_shard0.safetensors` — expert shards 1–31

Total weight tensors: 159,581.

### Upload command

The existing streaming script requires no local disk:

```bash
# Create bucket (one-time)
gsutil mb -l us-central1 -p tpu-launchpad-playground gs://jingnw-mimo-v2-5-pro-us-central1

# Stream HF → GCS (~962 GiB, resumable)
python3 tools/dev/upload_mimo_hf_to_gcs.py \
    --bucket jingnw-mimo-v2-5-pro-us-central1 \
    --gcs_prefix hf-weights \
    --repo_id XiaomiMiMo/MiMo-V2.5-Pro \
    --skip_existing
```

Target path after upload: `gs://jingnw-mimo-v2-5-pro-us-central1/hf-weights/`

---

## What's reusable from MiMo-V2-Flash

- `MiMoV2FlashSparseMoeBlock` — same MoE routing (sigmoid, noaux_tc, top-8, same intermediate size)
- Phase A/B FP8 infrastructure — same block-wise FP8 E4M3 format, same `block_dequant_fp8` / `fp8_moe_matmul` kernel
- Hybrid GA+SWA attention — same pattern structure, same SWA window=128, same partial RoPE factor=0.334
- Tokenizer — identical (Qwen2 BPE, vocab 152,576)
- `upload_mimo_hf_to_gcs.py` — works as-is, just different `--repo_id`

## What needs new work

1. **MaxText config** `mimo-v2-5-pro.yml` — updated dims (hidden=6144, 70 layers, 384 experts, 128 heads, GA KV heads=8), new hybrid layer pattern
2. **Checkpoint converter** — handle `fused_qkv` weight layout: split `[H, Hq+Hk+Hv]` → separate q/k/v tensors before writing to MaxText OCDBT
3. **Attention config** — GA KV heads unified to 8 (remove the asymmetric 4/8 special-casing in `mimo_v2_flash.py`)
4. **Capacity planning** — 42B active params at BF16 ≈ 84 GB/device activation footprint; need to profile target TPU topology

---

## Benchmark highlights (from README)

Base model vs peers (selected):

| Benchmark | MiMo-V2.5-Pro Base | MiMo-V2-Flash Base | Kimi-K2 Base |
|---|---|---|---|
| MMLU (5-shot) | 89.4 | 86.3 | 87.8 |
| GPQA-Diamond (5-shot) | **66.7** | 58.1 | 48.1 |
| MATH (4-shot) | **86.2** | 67.7 | 70.2 |
| LiveCodeBench v6 | **39.6** | 35.5 | 26.3 |
| SWE-Bench (AgentLess) | **35.7** | 30.8 | 28.2 |

Long-context: V2.5-Pro scores 0.37 BFS / 0.62 Parents at 1M tokens on GraphWalks; V2-Flash collapses to 0.00 at 1M.
