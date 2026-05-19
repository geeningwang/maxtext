# MiMo-V2.5-Pro — Environment Restoration Guide

Date: 2026-05-13  
Last updated: 2026-05-19  
Branch: `MiMo-V2.5-Pro`

---

## GCS bucket

```
gs://jingnw-mimo-v2-5-pro-us-central1/
├── hf-weights/                        # Original HF safetensors (962 GiB FP8)
├── mimo-v2-5-pro-fp8-ocdbt/          # MaxText checkpoint (FP8, 1038 arrays) ✅
│   └── 0/
│       ├── items/                     # Zarr2 parameter arrays
│       │   ├── _METADATA
│       │   └── params.params.*        # 1038 zarr directories
│       └── _CHECKPOINT_METADATA
└── shard_index.json                   # Safetensors byte-offset index (159,581 keys)
```

Created with:
```bash
gsutil mb -l us-central1 -p tpu-launchpad-playground gs://jingnw-mimo-v2-5-pro-us-central1
```

---

## HF → GCS upload (Phase 1) ✅

Script location: `tools/dev/upload_mimo_v25pro_hf_to_gcs.sh`

Streams 44 files (~962 GiB total) from HuggingFace directly to GCS. No Python or local
disk required. Skips already-uploaded files automatically (resumable).

```bash
# Start (or resume) upload
nohup bash tools/dev/upload_mimo_v25pro_hf_to_gcs.sh \
    > /tmp/mimo_v25pro_upload.log 2>&1 &
echo "PID: $!"

# Check progress
gsutil ls gs://jingnw-mimo-v2-5-pro-us-central1/hf-weights/ | wc -l  # out of 44
grep -E "^\[|^Done:|FAILED" /tmp/mimo_v25pro_upload.log | tail -10
```

---

## HF → MaxText checkpoint conversion (Phase 4) ✅

Converted 2026-05-19 using a 4-node parallel job on `jingnw-cpu-highmem`.

**Output:** `gs://jingnw-mimo-v2-5-pro-us-central1/mimo-v2-5-pro-fp8-ocdbt/`  
**Duration:** ~70 minutes (4 workers in parallel)

To re-run conversion (e.g. after a failed job):

```bash
# Submit 4-worker job (layer ranges 0-17, 18-35, 36-52, 53-69 in parallel)
kubectl apply -f tools/orchestration/mimo_v2_5_pro_convert_job.yaml

# Monitor workers
kubectl get job mimo-v25-pro-convert-v10-workers
kubectl logs -l app=mimo-v25-pro-convert-v10-workers --prefix

# After all 4 workers complete (4/4), submit finalizer
kubectl apply -f tools/orchestration/mimo_v2_5_pro_finalize_job.yaml
kubectl logs -l app=mimo-v25-pro-convert-v10-finalize
```

The conversion supports **resume**: if a worker is killed mid-run, resubmitting the job
will skip already-written layers (probes `.zarray` marker files in GCS).

The shard index cache (`shard_index.json`) is reused across runs — no re-indexing of the
159,581 weight keys on restart.

---

## GKE cluster context

Cluster: `jingnw-tpu7-cluster`, zone: `us-central1-c`

| Node Pool | Machine | Purpose | Current size |
|---|---|---|---|
| `default-pool` | e2-standard-4 | GKE control plane | 2 nodes |
| `jingnw-cpu-highmem` | n2-highmem-16 | Checkpoint conversion | **Scale back to 1** after conversion |
| `jingnw-flex-tpu7` | tpu7x-standard-4t (2x2x1) | V2-Flash inference | Flex (autoscales to 0) |
| `jingnw-flex-tpu7-8ch` | tpu7x-standard-4t (2x2x2) | V2.5-Pro inference (8ch) | Flex (autoscales to 0) |

> **Note:** `jingnw-cpu-highmem` was temporarily scaled to 4 nodes during the parallel
> conversion job. Scale it back to 1 to avoid unnecessary cost:
> ```bash
> gcloud container clusters resize jingnw-tpu7-cluster \
>     --zone=us-central1-c --node-pool=jingnw-cpu-highmem --num-nodes=1 --quiet
> ```

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

## Current phase status

| Phase | Description | Status |
|---|---|---|
| 1 | HF weights → GCS | ✅ `gs://.../hf-weights/` |
| 2 | MaxText config | ✅ `src/maxtext/configs/models/mimo-v2-5-pro.yml` |
| 3 | Model code adaptation | ✅ No changes needed |
| 4 | Checkpoint conversion | ✅ `gs://.../mimo-v2-5-pro-fp8-ocdbt/` |
| 5 | Inference job YAML | 🔄 In progress |
| 6 | Smoke test | ⏳ Pending |

See `bringup_plan.md` for full details.
