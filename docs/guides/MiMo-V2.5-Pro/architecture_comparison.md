# MiMo-V2.5-Pro — Architecture Comparison & Bringup Notes

Model card: https://huggingface.co/XiaomiMiMo/MiMo-V2.5-Pro  
Branch: `MiMo-V2.5-Pro`  
Date: 2026-05-13  
Last updated: 2026-05-19

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
| `attention_projection_layout` | `"fused_qkv"` | QKV weights stored as a single fused tensor per layer; checkpoint converter splits into q/k/v before OCDBT write |
| `add_full_attention_sink_bias` | `false` | explicit (was implicit in V2-Flash config) |
| `swa_v_head_dim` | 128 | now explicit in config (V2-Flash derived it implicitly) |
| `swa_head_dim` | 192 | now explicit |
| `swa_num_attention_heads` | 128 | now explicit |

The most impactful change is **`attention_projection_layout: "fused_qkv"`**. In MiMo-V2-Flash the HF checkpoint stored separate `q_proj`, `k_proj`, `v_proj` weight tensors. In V2.5-Pro they are stored as a single fused `qkv_proj` tensor of shape `[hidden_size, (num_heads * head_dim_q) + (num_kv_heads * head_dim_k) + (num_kv_heads * v_head_dim)]`. The checkpoint converter splits at offsets `[0, nq·dq]`, `[nq·dq, nq·dq+nkv·dk]`, `[nq·dq+nkv·dk, ...]` before mapping to MaxText's separate Q/K/V arrays.

**GA KV heads changed from 4 → 8**: MiMo-V2-Flash had asymmetric KV heads (4 for GA, 8 for SWA). V2.5-Pro uses 8 for both, simplifying the attention config. No model code change was needed — the existing branching logic returns 8 for both paths given the V2.5-Pro config values.

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

## HF weights → GCS ✅

**Total HF size:** 962.4 GiB (FP8, TP=8 pre-sharded, 34 safetensors files + 1 index)

Shard layout:
- `model_mtp.safetensors` — 3 MTP (Multi-Token Prediction) decoder heads
- `model_pp0_ep0_shard0/1.safetensors` — backbone + expert shard 0 (2 files; shard1 holds `lm_head` and embeddings)
- `model_pp0_ep1_shard0.safetensors` … `model_pp0_ep31_shard0.safetensors` — expert shards 1–31

Total weight tensors: 159,581.

---

## MaxText checkpoint ✅

**Location:** `gs://jingnw-mimo-v2-5-pro-us-central1/mimo-v2-5-pro-fp8-ocdbt/`  
**Format:** Orbax zarr2, 1,038 arrays  
**Precision:** FP8 E4M3FN expert weights + float32 per-block scale_inv; BF16 attention/embed weights  
**Converted:** 2026-05-19, ~70 min using 4-node parallel job on `jingnw-cpu-highmem`

---

## What's reusable from MiMo-V2-Flash

- `MiMoV2FlashSparseMoeBlock` — same MoE routing (sigmoid, noaux_tc, top-8, same intermediate size)
- Phase A/B FP8 infrastructure — same block-wise FP8 E4M3 format, same `_block_dequant_fp8` / `fp8_moe_matmul` kernel
- Hybrid GA+SWA attention — same pattern structure, same SWA window=128, same partial RoPE factor=0.334
- Tokenizer — identical (Qwen2 BPE, vocab 152,576)
- `upload_mimo_hf_to_gcs.py` — works as-is, just different `--repo_id`

## What was done for V2.5-Pro bringup (phases 1–4)

1. **MaxText config** `mimo-v2-5-pro.yml` ✅ — updated dims (hidden=6144, 70 layers, 384 experts, 128 heads, GA KV heads=8), new hybrid layer pattern, `mimo_fp8_weight_mode=block_wise_fp8`
2. **Checkpoint converter** ✅ — handles `fused_qkv` weight layout: splits `[H, Hq+Hk+Hv]` → separate q/k/v tensors before writing to MaxText OCDBT; FP8 weights + scale_inv preserved
3. **Model code** ✅ — no changes required; `mimo_v2_flash.py` was already fully generic over config parameters
4. **Inference precision** ✅ — FP8 + dequant-before-matmul (`block_wise_fp8`): weights kept as FP8 in HBM, dequantized per-block to BF16 before each matmul

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
