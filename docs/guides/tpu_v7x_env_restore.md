# MiMo-V2-Flash on TPU v7x — GKE Environment Restore Guide

> **Scope:** This is the active restore path for the TPU v7x GKE environment.
> For the legacy v6e slice guide see [mimo_v2_flash_env_restore.md](mimo_v2_flash_env_restore.md).

This guide recreates the MiMo-V2-Flash inference environment on GKE cluster
`jingnw-tpu7-cluster` with the `tpu-v7x-porting` MaxText branch, from scratch
on a new `jingnw-tpu-op` ops VM.

---

## Fixed Settings

| Setting | Value |
|---|---|
| Project | `tpu-launchpad-playground` |
| GKE cluster | `jingnw-tpu7-cluster` |
| Cluster zone | `us-central1-c` |
| TPU node pool | `jingnw-flex-tpu7` |
| Machine type | `tpu7x-standard-4t` (4 chips, 8 JAX devices, 192 GiB HBM/chip) |
| Autoscaling | `--min-nodes=0 --max-nodes=2` (DWS flex-start) |
| Topology | `2x2x1` |
| JAX image | `us-docker.pkg.dev/cloud-tpu-images/jax-ai-image/tpu:jax0.8.1-rev1` |
| JAX version | 0.8.1 (confirmed working on v7x) |
| libtpu version | 0.0.30 (confirmed working on v7x) |
| MaxText branch | `tpu-v7x-porting` on `github.com/geeningwang/maxtext` |
| Checkpoint (us-central1) | `gs://jingnw-mimo-v2-flash-us-central1/mimo-v2-flash-fixed-ocdbt/checkpoints/0/items` |
| Checkpoint (us-east5, fallback) | `gs://jingnw-mimo-v2-flash-us-east5/mimo-v2-flash-fixed-ocdbt/checkpoints/0/items` |
| Tokenizer | `XiaomiMiMo/MiMo-V2-Flash` |
| Parallelism | `TP=4, EP=2` (8 devices, 4 global KV heads max TP) |

---

## 1. One-Time Setup on `jingnw-tpu-op`

### 1a. Install kubectl and GKE auth plugin

```bash
sudo apt-get update
sudo apt-get install -y kubectl google-cloud-sdk-gke-gcloud-auth-plugin
```

### 1b. Fetch cluster credentials

```bash
gcloud container clusters get-credentials jingnw-tpu7-cluster \
  --zone us-central1-c \
  --project tpu-launchpad-playground
```

### 1c. Verify cluster access

```bash
kubectl get nodes -o wide
# Expected: 2 e2-standard-4 nodes in Ready state (default-pool)
# The tpu7x node pool starts at 0 and is provisioned on demand.

kubectl get nodepool jingnw-flex-tpu7 2>/dev/null || \
  kubectl get nodes -l cloud.google.com/gke-nodepool=jingnw-flex-tpu7
```

### 1d. Clone the MaxText repo (optional, for editing job YAMLs)

```bash
git clone https://github.com/geeningwang/maxtext.git ~/maxtext
cd ~/maxtext
git checkout tpu-v7x-porting
```

---

## 2. Smoke Test: Confirm TPU v7x Hardware

Submit the chip presence report job to validate that the TPU v7x node provisions
correctly and all 8 devices with 94.7 GiB HBM each are accessible:

```bash
kubectl apply -f tools/orchestration/tpu7x_chip_info_job.yaml
kubectl wait --for=condition=complete job/tpu7x-chip-info --timeout=30m
kubectl logs job/tpu7x-chip-info
```

Expected output summary:
```
Global device count  : 8
TPU_0 ... TPU_7      : TPU7x
HBM per device       : total=94.7 GiB  free=94.7 GiB
ALL CHECKS PASSED
```

If you need to re-run the smoke test:
```bash
kubectl delete job tpu7x-chip-info
kubectl apply -f tools/orchestration/tpu7x_chip_info_job.yaml
```

---

## 3. Run MiMo-V2-Flash Inference Demo

### 3a. Submit the demo job

```bash
kubectl apply -f tools/orchestration/mimo_v2_flash_demo_job.yaml
```

### 3b. Monitor progress

```bash
# Watch pod status (Pending → ContainerCreating → Running → Completed)
kubectl get pods -l job-name=mimo-v2-flash-demo-v7x -w

# Stream live logs once the pod is Running
kubectl logs -f job/mimo-v2-flash-demo-v7x

# Check job completion
kubectl get job mimo-v2-flash-demo-v7x
```

### 3c. Get results

```bash
kubectl logs job/mimo-v2-flash-demo-v7x | tail -20
```

Expected output (baseline, BF16 dense, no quantization):
```
Throughput: 1.5 tok/s  [scan_layers=false (dense), no quantization (bfloat16)]
Status:     EOS fired (clean stop)  # or WARNING if max_new_tokens too small
Output:
<model response here>
```

### 3d. Re-run after changes

Kubernetes `Job` specs are immutable. Always delete before reapplying:
```bash
kubectl delete job mimo-v2-flash-demo-v7x
kubectl apply -f tools/orchestration/mimo_v2_flash_demo_job.yaml
```

---

## 4. Run INT8 Quantization Validation

```bash
kubectl apply -f tools/orchestration/mimo_v2_flash_int8_job.yaml
kubectl wait --for=condition=complete job/mimo-v2-flash-int8-v7x --timeout=60m
kubectl logs job/mimo-v2-flash-int8-v7x | tail -20
```

---

## 5. Key GKE Scheduling Contract

All TPU v7x jobs must use these exact node selectors and tolerations:

```yaml
nodeSelector:
  cloud.google.com/gke-nodepool: jingnw-flex-tpu7
  cloud.google.com/gke-tpu-accelerator: tpu7x        # family, NOT machine type
  cloud.google.com/gke-tpu-topology: 2x2x1
tolerations:
- key: google.com/tpu
  operator: Equal
  value: present
  effect: NoSchedule
resources:
  requests:
    google.com/tpu: "4"   # 4 chips = 1 full tpu7x-standard-4t node
  limits:
    google.com/tpu: "4"
```

**Critical:** Use `cloud.google.com/gke-tpu-accelerator: tpu7x` (the accelerator
*family*), NOT `tpu7x-standard-4t` (the machine type). GKE Warden rejects the
machine-type form.

---

## 6. Python Path in the JAX Image

The JAX AI image `jax0.8.1-rev1` installs JAX in `/opt/venv`, NOT
`/usr/local`. Always use:

```bash
/opt/venv/bin/python   # correct
python3                # wrong — no jax
```

In job YAML commands, either:
```yaml
command: ["/opt/venv/bin/python", "-m", "maxtext.inference.decode", ...]
```
or prepend to PATH:
```yaml
export PATH="/opt/venv/bin:${PATH}"
```

---

## 7. Required MaxText Flags for Single-Pod GKE Runs

```
enable_single_controller=true
```

Without this, JAX detects the Kubernetes environment and tries to initialise
a multi-process distributed cluster, which fails because `jax[k8s]` is not
installed in the image. `enable_single_controller=true` skips the distributed
init and runs all 8 devices in a single controller process.

---

## 8. DWS Flex-Start Resource Lifecycle

- The **node pool** (`jingnw-flex-tpu7`) is permanent — no timeout.
- **TPU VM nodes** are provisioned on-demand when a pod is scheduled and
  automatically released (scale to 0) when no pods are running.
- Maximum allocation duration: **7 days** per flex-start request.
- You are billed **only while nodes are running**.
- After a job completes or fails, the autoscaler scales back to 0 automatically.

To check current node count:
```bash
kubectl get nodes -l cloud.google.com/gke-nodepool=jingnw-flex-tpu7
```

---

## 9. Troubleshooting

### Pod stuck in Pending
```bash
kubectl describe pod <pod-name> | grep -A 20 "Events:"
```
- `TriggeredScaleUp` → DWS is provisioning a TPU VM (normal, wait 5–20 min)
- `NotTriggerScaleUp: 1 in backoff after failed scale-up` → autoscaler cooldown; wait ~5 min and resubmit
- `Insufficient google.com/tpu` → another job is using the node; wait or increase `--max-nodes`

### `libtpu_lockfile` error
```
RuntimeError: Unable to initialize backend 'tpu': ABORTED: Internal error when accessing libtpu multi-process lockfile
```
The parent Python process initialised JAX (and acquired the lockfile) before
spawning the MaxText subprocess. Fix: pass `--ici_tensor_parallelism` and
`--ici_expert_parallelism` explicitly so `resolve_parallelism()` does not call
`jax.device_count()` in the parent process.

### `coordinator_address should be defined`
Pass `enable_single_controller=true` to MaxText.

### Job spec immutable error
`Job` specs cannot be patched after creation. Always:
```bash
kubectl delete job <job-name>
kubectl apply -f <job-yaml>
```

### Node evicted during image pull
DWS flex-start nodes can be preempted. If the pod was evicted before the
container started, just resubmit — the next provisioned node is stable.
