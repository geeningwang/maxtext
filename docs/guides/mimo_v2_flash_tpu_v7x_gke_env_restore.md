# MiMo-V2-Flash — TPU v7x GKE Environment Restore

This guide restores the GKE-based TPU v7x inference and PTQ-quantization
environment for the `MiMo-V2-Flash` branch.

The environment uses **Google Kubernetes Engine (GKE) with DWS Flex Start** TPU
node pools.  Jobs run in Docker containers (no persistent TPU VM to SSH into).
All work is done from the manager VM `jingnw-tpu-op` (or any VM with `gcloud`
credentials) by submitting Kubernetes `Job` manifests.

---

## Hardware Summary

### TPU v7x chip architecture

| Level | Description | HBM |
|---|---|---|
| TensorCore | 1 JAX device | 96 GiB |
| Chip | 2 TensorCores | 192 GiB |
| 2×2×1 node (`jingnw-flex-tpu7`) | 4 chips = **8 JAX devices** | 768 GiB |
| 2×2×2 node (`jingnw-flex-tpu7-8ch`) | 8 chips = **16 JAX devices** | 1,536 GiB |

**Key point:** TPU v7x has **2 TensorCores per chip**.  JAX sees each
TensorCore as a separate device, so requesting `google.com/tpu: "4"` (4 chips)
gives 8 JAX devices.  The 94.75 GiB `bytes_limit` reported by
`d.memory_stats()` is per TensorCore, not per chip.

### GKE cluster

| Setting | Value |
|---|---|
| Cluster | `jingnw-tpu7-cluster` |
| Zone | `us-central1-c` |
| Project | `tpu-launchpad-playground` |
| Node pool (single-host) | `jingnw-flex-tpu7` — 2×2×1, 4 chips, 8 JAX devices, ~919 GiB host RAM |
| Node pool (multi-host) | `jingnw-flex-tpu7-8ch` — 2×2×2, 8 chips per node, 16 JAX devices per node |

### Docker image

```
us-docker.pkg.dev/cloud-tpu-images/jax-ai-image/tpu:jax0.8.1-rev1
```

Packages baked in: JAX 0.8.1, jaxlib 0.8.1, Flax, Orbax, libtpu.  The job
installs MaxText and additional packages at runtime from the `MiMo-V2-Flash`
branch of `geeningwang/maxtext`.

---

## 1. Restore kubectl on a Fresh VM

The VM loses kubectl on restart.  Run the following before any cluster work:

```bash
sudo apt-get install -y kubectl google-cloud-cli-gke-gcloud-auth-plugin
gcloud container clusters get-credentials jingnw-tpu7-cluster \
  --zone us-central1-c \
  --project tpu-launchpad-playground
```

Verify:

```bash
kubectl get nodes
kubectl get pods,jobs
```

---

## 2. GCS Paths

| Checkpoint | GCS path | Size |
|---|---|---|
| BF16 source (OCDBT) | `gs://jingnw-mimo-v2-flash-us-central1/mimo-v2-flash-fixed-ocdbt/checkpoints/0/items` | 384.43 GiB |
| FP8 PTQ output | `gs://jingnw-mimo-v2-flash-us-central1/mimo-v2-flash-fp8-ptq/0/items` | 441.79 GiB |

Check sizes at any time:

```bash
gsutil du -sh gs://jingnw-mimo-v2-flash-us-central1/mimo-v2-flash-fp8-ptq/0/items
```

---

## 3. Job YAMLs

| Job | YAML | Node pool | Topology | JAX devices |
|---|---|---|---|---|
| PTQ quantize | `tools/orchestration/mimo_v2_flash_ptq_quantize_job.yaml` | `jingnw-flex-tpu7-8ch` | 2×2×2, 2-host | 16 (2 hosts × 8) |
| FP8 demo/inference | `tools/orchestration/mimo_v2_flash_fp8_ptq_demo_job.yaml` | `jingnw-flex-tpu7` | 2×2×1, single-host | 8 |

---

## 4. Submit and Monitor Jobs

### Submit a job

```bash
kubectl apply -f tools/orchestration/mimo_v2_flash_fp8_ptq_demo_job.yaml
```

### Check status

```bash
kubectl get pods,jobs
```

### Stream logs

```bash
POD=$(kubectl get pods -l app=<job-label> -o jsonpath='{.items[0].metadata.name}')
kubectl logs -f "$POD"
```

Substitute `<job-label>` with the value of `app:` in the YAML's pod labels
(e.g., `mimo-v2-flash-fp8-ptq-demo-v10x`).

### Delete a completed or failed job before resubmitting

```bash
kubectl delete job <job-name> --ignore-not-found
kubectl apply -f tools/orchestration/<yaml-file>.yaml
```

DWS Flex Start provisions TPU nodes on demand when a pod is scheduled.
A pod in `Pending` state with event `TriggeredScaleUp` is normal — the node is
being created (typically < 2 min).

---

## 5. Run the FP8 PTQ Demo (Inference Verification)

### What it does

Loads the pre-quantized FP8 checkpoint and runs one inference pass with a
math prompt, reporting throughput and EOS status.

### Submit

```bash
kubectl apply -f tools/orchestration/mimo_v2_flash_fp8_ptq_demo_job.yaml
```

### Monitor HBM and completion

```bash
POD=$(kubectl get pods -l app=mimo-v2-flash-fp8-ptq-demo-v10x \
      -o jsonpath='{.items[0].metadata.name}')
kubectl logs -f "$POD" | grep -E "\[HBM\]|EOS token|generate_total|Throughput"
```

### Expected output

```
[HBM] after_setup_decode_state  ... used=71.93GB peak=71.93GB limit=94.75GB
[TIME] load_params              ... elapsed=271.1s
[HBM] after_prefill             ... used=72.02GB
[HBM] after_insert              ... used=72.17GB
[HBM] generate_step_0512        ... used=72.30GB peak=72.31GB
[INFO] EOS token (151645) generated at step 767; stopping early.
```

Throughput: **2.8 tok/s** (single sequence, TP=4 EP=2, scan_layers=false).

### HBM allocation (validated 2026-05-12, v10x run)

| Stage | Per TensorCore | Delta |
|---|---|---|
| `init` | 0.00 GB | — |
| `setup_decode_state` | **71.93 GB** | +71.93 GB (weights + KV cache pre-allocated together) |
| `after_prefill` | 72.02 GB | +0.09 GB |
| `after_insert` | 72.17 GB | +0.15 GB |
| `generate_step_0512` | **72.31 GB** | +0.14 GB |

Per chip: ~144.6 GB used / ~192 GiB capacity.
Total across 4 chips (8 JAX devices): **~578.5 GB used / ~768 GiB capacity**.

Dtype breakdown at `after_load_params` (dev=0, 571 shards via `addressable_shards`):

| dtype | HBM | Tensors |
|---|---|---|
| `bfloat16` | **71.928 GB** | 568 |
| `uint32` | < 0.001 GB | 3 |

No `float8_e4m3fn` tensors appear in JAX live arrays — qwix PtqProvider
presents all weights as BF16 to the JAX Python layer.
See [mimo_v2_flash_fp8_dtypes.md](mimo_v2_flash_fp8_dtypes.md) for the full analysis.

---

## 6. Run the PTQ Quantize Job

The quantize job converts the BF16 checkpoint to FP8 PTQ format using qwix
`PtqProvider`.  This is a one-time operation; the FP8 checkpoint already exists
in GCS.  Re-run only if the checkpoint needs to be regenerated.

### Submit

```bash
kubectl apply -f tools/orchestration/mimo_v2_flash_ptq_quantize_job.yaml
```

This submits a 2-host indexed job with a headless Service for stable DNS.

### Monitor

```bash
# Wait for both workers
kubectl get pods -l job-name=mimo-v2-flash-ptq-quantize -w
# Stream worker-0 logs
kubectl logs -f mimo-v2-flash-ptq-quantize-0
```

### Expected completion

Both pods reach `Completed`.  The FP8 checkpoint writes to
`gs://jingnw-mimo-v2-flash-us-central1/mimo-v2-flash-fp8-ptq/0/items`
(441.79 GiB total).

### Key parameters (from YAML)

| Parameter | Value | Reason |
|---|---|---|
| `ici_tensor_parallelism` | 4 | `num_kv_heads=4` must be divisible by TP |
| `ici_expert_parallelism` | 4 | 128 experts per EP group across 16 devices |
| Memory per pod | 750 GiB | D2H of 568 quantized arrays during checkpoint save |
| `async_checkpointing` | false | Required for correct multi-host OCDBT write |

---

## 7. Parallelism Reference

| Job | TP | EP | Total JAX devices | Constraint |
|---|---|---|---|---|
| PTQ quantize | 4 | 4 | 16 | `num_kv_heads=4` must divide TP=4 |
| FP8 demo inference | 4 | 2 | 8 | Same KV head constraint |

For FP8 inference the minimum viable config is TP=4 EP=2 (single 4-chip node).
TP=8 OOMs — `num_kv_heads=4` is not divisible by 8.

---

## 8. Troubleshooting

### Pod stuck in `Pending`

```bash
kubectl describe pod <pod-name> | grep -A10 "Events:"
```

`FailedScheduling` followed by `TriggeredScaleUp` is normal — DWS Flex Start
is provisioning the TPU node.  Wait 2–5 minutes.

If `TriggeredScaleUp` never appears, check `nodeSelector` labels in the YAML
match the actual node pool labels:
```bash
kubectl get nodes --show-labels | grep jingnw-flex
```

### OOMKilled

Increase `resources.requests.memory` and `resources.limits.memory` in the YAML.
The node has ~919 GiB allocatable RAM.  Progressive values that were tried for
the PTQ quantize job: 256Gi → 350Gi → 512Gi → **750Gi** (final).

### `num_kv_heads` divisibility error

```
ValueError: num_kv_heads=4 must be divisible by ici_tensor_parallelism
```

Use `ici_tensor_parallelism=4` (not 8).

### Checkpoint load OOM (demo job)

The Orbax restore buffer alone is ~89.4 GiB (`restore_concurrent_bytes`).
Set `memory: 256Gi` minimum for the demo job.

### `jax.live_arrays()` returns empty on multi-device sharded arrays

When using `_probe_hbm_arrays` with `arr.device()` filtering, sharded global
arrays raise on `.device()` and are silently skipped.  Use the
`addressable_shards` approach instead:

```python
for shard in arr.addressable_shards:
    if shard.device == d:
        nbytes += int(shard.data.size) * arr.dtype.itemsize
```

Note: dev=0 reports accurate totals matching `memory_stats()`; dev=1-7 may
double-count due to EP×TP mesh shard routing.  Trust `memory_stats().bytes_in_use`
as the authoritative per-device HBM figure.

---

## 9. Related Documents

- [mimo_v2_flash_inference_overview.md](mimo_v2_flash_inference_overview.md) — full inference stack comparison
- [mimo_v2_flash_inference_pipeline.md](mimo_v2_flash_inference_pipeline.md) — module-by-module pipeline guide
- [mimo_v2_flash_fp8_dtypes.md](mimo_v2_flash_fp8_dtypes.md) — FP8 weight dtype and HBM dtype analysis
- [mimo_v2_flash_hbm_probes.md](mimo_v2_flash_hbm_probes.md) — HBM probe points and measurement results
- [mimo_v2_flash_env_restore.md](mimo_v2_flash_env_restore.md) — TPU v6e environment restore (SSH-based)
