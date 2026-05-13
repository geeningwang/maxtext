# MiMo-V2.5-Pro — Environment Restoration & Upload Guide

Date: 2026-05-13  
Branch: `MiMo-V2.5-Pro`

---

## GCS bucket

```
gs://jingnw-mimo-v2-5-pro-us-central1/hf-weights/
```

Created with:
```bash
gsutil mb -l us-central1 -p tpu-launchpad-playground gs://jingnw-mimo-v2-5-pro-us-central1
```

---

## HF → GCS upload script

Script location: `tools/dev/upload_mimo_v25pro_hf_to_gcs.sh`

Streams 44 files (~962 GiB total) from HuggingFace directly to GCS using `curl | gsutil cp -`.
No Python or local disk required. Skips already-uploaded files automatically (resumable).

### Start (or resume) upload

```bash
nohup bash tools/dev/upload_mimo_v25pro_hf_to_gcs.sh \
    > /tmp/mimo_v25pro_upload.log 2>&1 &
echo "PID: $!"
```

### Check progress (without parsing the log)

```bash
# Count files landed in GCS (out of 44)
gsutil ls gs://jingnw-mimo-v2-5-pro-us-central1/hf-weights/ | wc -l

# List all uploaded files
gsutil ls gs://jingnw-mimo-v2-5-pro-us-central1/hf-weights/
```

### Check progress via log

```bash
# Recent upload events (avoids gsutil CR-based progress noise)
grep -E "^\[|^Done:|FAILED" /tmp/mimo_v25pro_upload.log | tail -10

# Check if done
grep "^Done:" /tmp/mimo_v25pro_upload.log

# Check for failures
grep "FAILED" /tmp/mimo_v25pro_upload.log
```

### Check if the upload process is still running

```bash
# Find by PID (replace 6979 with actual PID)
ps aux | grep upload_mimo | grep -v grep
```

---

## Files uploaded (44 total)

| # | File | Notes |
|---|---|---|
| 1–10 | config/tokenizer files | Small, upload in seconds |
| 11 | `model_mtp.safetensors` | MTP decoder heads (~101s) |
| 12 | `model_pp0_ep0_shard0.safetensors` | Backbone + expert shard 0 (~28 GiB, ~1408s) |
| 13 | `model_pp0_ep0_shard1.safetensors` | `lm_head` + embeddings (~1070s) |
| 14–44 | `model_pp0_ep1_shard0.safetensors` … `ep31_shard0.safetensors` | Expert shards 1–31 |

---

## GKE cluster context

Cluster: `jingnw-tpu7-cluster`, zone: `us-central1-c`

Target node pool for V2.5-Pro inference: `jingnw-flex-tpu7-8ch` (2x2x2, 8 chips, 16 JAX devices)

CPU node for checkpoint conversion: `jingnw-cpu-highmem` (n2-highmem-16, 128 GB RAM)

### Pending cleanup (one-time)

```bash
# Delete broken node pool left from failed 16ch attempt
gcloud container node-pools delete jingnw-flex-tpu7-16ch \
    --cluster=jingnw-tpu7-cluster --zone=us-central1-c --quiet

# Delete unused placement policy
gcloud compute resource-policies delete jingnw-tpu7-policy-16ch \
    --region=us-central1 --quiet
```

---

## Next phases after upload completes

See `bringup_plan.md` for the full plan. Summary:

1. **Phase 2** — Create `src/maxtext/configs/models/mimo-v2-5-pro.yml`
2. **Phase 3** — Adapt `src/maxtext/models/mimo_v2_flash.py` (fused_qkv, unified KV heads, 70 layers)
3. **Phase 4** — Extend checkpoint converter for fused_qkv split → OCDBT output
4. **Phase 5** — Create GKE inference job YAML for `jingnw-flex-tpu7-8ch`
5. **Phase 6** — Smoke test (single-token decode, HBM probe)
