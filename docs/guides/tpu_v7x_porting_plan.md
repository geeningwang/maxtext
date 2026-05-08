# TPU v7x Porting Plan — MiMo-V2-Flash on GKE

**Branch:** `tpu-v7x-porting`  
**Date:** 2026-05-07  
**Author:** jingnw

---

## Execution Status

### 2026-05-08 — Task 5 (smoke test) and Task 6 (inference pipeline) complete

**Chip presence report** (`tpu7x-chip-info` job — completed):
- 8× TPU7x devices, each with **94.7 GiB usable HBM** (192 GiB physical / 2 TensorCores per chip)
- All 1 GiB per-device allocations passed (`ALL CHECKS PASSED`)

**MiMo-V2-Flash first inference run** (`mimo-v2-flash-demo-v7x` job — completed):
- Throughput: **1.5 tok/s** (BF16, dense checkpoint, full prefill, no FP8)
- 128 tokens generated in 86.2s (avg 6ms/step after compile)
- Output coherent and correct (math reasoning problem solved step-by-step)
- Status: EOS not fired at 128 tokens — model hit `--max_new_tokens` limit mid-sentence (expected for short limit)

**Bugs fixed during Task 6:**
1. `coordinator_address should be defined` — JAX detects K8s env but `jax[k8s]` not installed; fixed by passing `enable_single_controller=true` to MaxText in both inference and dry-run paths.
2. `libtpu_lockfile` conflict — `resolve_parallelism()` called `jax.device_count()` in the parent process, initialising libtpu before the MaxText subprocess; fixed by skipping JAX init when TP/EP are explicitly passed.
3. Node eviction — flex-start DWS node was preempted during image pull on first attempt; resubmission succeeded immediately on the warm node.

### 2026-05-07 — Initial setup

- Installed missing GKE client packages: `kubectl` and
   `google-cloud-sdk-gke-gcloud-auth-plugin`.
- Fetched cluster credentials for `jingnw-tpu7-cluster` and validated live
   access with `kubectl`.
- Submitted [tools/orchestration/tpu7x_jax_smoke_test.yaml](../../tools/orchestration/tpu7x_jax_smoke_test.yaml),
   which triggered flex-start scale-up on `jingnw-flex-tpu7` and ran on a TPU
   node successfully.
- Verified runtime on the TPU pod:
   - `JAX_VERSION 0.8.1`
   - `JAXLIB_VERSION 0.8.1`
   - `LOCAL_DEVICE_COUNT 8`
   - `DEVICE_COUNT 8`
- Verified the JAX AI image `us-docker.pkg.dev/cloud-tpu-images/jax-ai-image/tpu:jax0.8.1-rev1`
   works on v7x, but it must be invoked with `/opt/venv/bin/python` because the
   default `/usr/local/bin/python3` does not have `jax` on `PATH`.
- Corrected MiMo-V2-Flash parallelism assumption for v7x: on an 8-device
   `tpu7x-standard-4t` host, the valid default is `TP=4, EP=2`, not `TP=8,
   EP=1`, because MiMo has only 4 global-attention KV heads.

---

## 0. Context

The previous environment used a **TPU v6e-32 slice** (`jingnw-node`, `us-east5-b`)
accessed via direct SSH (`gcloud compute tpus tpu-vm ssh --worker=all`).  That
slice has been deleted.

The new environment is a **GKE TPU v7x cluster** (`jingnw-tpu7-cluster`,
`us-central1-c`) with a flex-start (DWS) node pool `jingnw-flex-tpu7` that
provides `tpu7x-standard-4t` nodes (4 chips per node, 192 GiB HBM per chip,
2 logical cores per chip → 8 JAX devices per node).

### Infrastructure delta

| Property            | v6e (old)                        | v7x GKE (new)                            |
|---------------------|----------------------------------|------------------------------------------|
| Hardware            | TPU v6e-32 (8 workers × 4 chips) | TPU v7x-4 (1 node × 4 chips, 8 devices) |
| Access model        | `gcloud tpu-vm ssh --worker=all` | `kubectl` / `xpk` job submission         |
| Execution unit      | TPU VM (bare metal)              | Kubernetes pod (Docker container)        |
| Cluster type        | TPU slice                        | GKE DWS (flex-start)                     |
| Zone                | us-east5-b                       | us-central1-c                            |
| Chips total         | 32                               | 4 (1 node)                               |
| HBM per chip        | ~32 GiB (v6e)                    | 192 GiB (v7x)                            |
| JAX devices         | 32                               | 8 (2 cores/chip × 4 chips)               |
| SparseCore          | No                               | Yes (v7x feature)                        |

---

## 1. Task List

### Task 1 — Cluster credentials and kubectl setup

**Goal:** Connect `jingnw-tpu-op` to `jingnw-tpu7-cluster` so subsequent
`kubectl` commands work.

```bash
sudo apt-get update
sudo apt-get install -y kubectl google-cloud-sdk-gke-gcloud-auth-plugin
gcloud container clusters get-credentials jingnw-tpu7-cluster \
  --zone us-central1-c \
  --project tpu-launchpad-playground
kubectl get nodes -o wide
```

Expected: 1–2 nodes in `Ready` state (default-pool + possibly the flex TPU node
if a workload has already triggered scale-up).

**Files touched:** none (one-time credential setup)

---

### Task 2 — JAX / libtpu version verification for v7x

**Goal:** Confirm that the libtpu and JAX versions pinned in
`dependencies/requirements/generated_requirements/tpu-requirements.txt` support
TPU v7x.

Current pins:
- `jax >= 0.8.1`
- `jaxlib >= 0.8.1`
- `libtpu >= 0.0.30`

**Action items:**
1. Check whether libtpu 0.0.30 ships v7x support, or whether a newer nightly is
   needed.  Reference: `gs://libtpu-builds/` or the `libtpu-nightly` PyPI index.
2. If a newer libtpu is required, update
   `dependencies/requirements/generated_requirements/tpu-requirements.txt` and
   document the verified version in a new `tpu-requirements-v7x.txt` (or update
   the existing file with a comment).
3. Re-verify `jax` and `jaxlib` compatibility matrix for the chosen libtpu.

**Files to update:**
- `dependencies/requirements/generated_requirements/tpu-requirements.txt`
- (possibly) `dependencies/dockerfiles/maxtext_tpu_dependencies.Dockerfile`

---

### Task 3 — Docker image for v7x GKE

**Goal:** Build (or identify) a Docker image that works on `tpu7x-standard-4t`
nodes.

The v6e environment ran `uv pip install` directly on TPU VMs.  GKE pods pull a
container image.  MaxText already has Dockerfiles:
- `dependencies/dockerfiles/clean_py_env.Dockerfile` — slim Python base,
  installs `uv` and nothing else (maxtext installed at pod startup via
  `uv pip install`)
- `dependencies/dockerfiles/maxtext_tpu_dependencies.Dockerfile` — system deps

**Action items:**
1. Decide on image strategy:
   - **Option A (preferred for iteration speed):** Use
     `clean_py_env.Dockerfile`; install Python deps at pod init time via an
     init command.  No rebuild needed when requirements change.
   - **Option B (faster cold start):** Pre-bake `.[tpu]` deps into a versioned
     image and push to `gcr.io/tpu-launchpad-playground/maxtext-tpu:v7x-YYYYMMDD`.
2. Build and push the chosen image:
   ```bash
   docker build --build-arg DEVICE=tpu --build-arg PYTHON_VERSION=3.12 \
     -t gcr.io/tpu-launchpad-playground/maxtext-tpu:v7x-20260507 \
     -f dependencies/dockerfiles/clean_py_env.Dockerfile .
   gcloud auth configure-docker
   docker push gcr.io/tpu-launchpad-playground/maxtext-tpu:v7x-20260507
   ```
3. Verify the image runs on a v7x node (smoke test in Task 5).

**Files to update / create:**
- `dependencies/dockerfiles/` — possibly a new `maxtext_tpu_v7x.Dockerfile`
- `docs/guides/tpu_v7x_env_setup.md` (new, to be created in Task 7)

---

### Task 4 — Model config: tpu7x-4 sharding strategy for MiMo-V2-Flash

**Goal:** Define or verify MaxText model configs for the 4-chip (8-device)
tpu7x-4 topology.

MiMo-V2-Flash on v6e-32 used a 32-device mesh.  On tpu7x-4 we have only 8
devices per node.  The model is 309B params and will not fit in one 4-chip node
in full precision — this task first targets **inference-only** with
appropriate sharding:

| Dimension        | v6e-32 (32 devices)           | tpu7x-4 (8 devices)                 |
|------------------|-------------------------------|-------------------------------------|
| Tensor parallel  | TP=4                          | TP=4                                |
| Expert parallel  | EP=8                          | EP=2                                |
| Data parallel    | DP=1                          | DP=1                                |
| Mesh shape       | (dp=1, fsdp=1, tp=4, ep=8)    | (dp=1, fsdp=1, tp=4, ep=2)          |

> **Correction from initial draft:** MiMo-V2-Flash has only 4 global KV heads,
> so tensor parallelism must not exceed 4. The live demo code now auto-resolves
> to `TP=4, EP=2` on an 8-device `tpu7x-standard-4t` host.

> **Note:** 192 GiB HBM per v7x chip × 4 chips = 768 GiB per node, vs
> ~32 GiB × 32 chips = ~1 TiB on v6e-32.  At 4-chip scale, loading the full
> 309B MoE model in BF16 (~620 GB) is marginal; FP8 weights (~155 GB) should
> fit comfortably.

**Action items:**
1. Check `src/maxtext/configs/` for any existing `tpu7x-4` or `tpu7x` config
   entries.
2. Create `src/maxtext/configs/models/mimo-v2-flash-tpu7x-4.yml` adapting from
   the v6e-32 config.
3. Verify mesh/sharding annotations by running the model's `initialize` in dry-run
   mode (no checkpoint) and checking that all shapes are divisible.

**Files to update / create:**
- `src/maxtext/configs/models/mimo-v2-flash-tpu7x-4.yml` (new)
- `src/maxtext/configs/inference/mimo-v2-flash-decode-tpu7x-4.yml` (new)

---

### Task 5 — Smoke test: JAX device detection on v7x node ✅ COMPLETE

**Goal:** Confirm that `jax.devices()` returns 8 TPU devices on a
`tpu7x-standard-4t` node inside a GKE pod.

**Result (2026-05-07):** Verified. Two jobs confirmed correct hardware:
- `tpu7x_jax_smoke_test.yaml` — `LOCAL_DEVICE_COUNT 8`, `DEVICE_COUNT 8`, JAX 0.8.1
- `tpu7x_chip_info_job.yaml` — 8× TPU7x devices, 94.7 GiB HBM each, all 1 GiB allocations passed

Verified scheduling contract:
- `cloud.google.com/gke-nodepool=jingnw-flex-tpu7`
- `cloud.google.com/gke-tpu-accelerator=tpu7x`
- `cloud.google.com/gke-tpu-topology=2x2x1`
- `google.com/tpu: 4`
- toleration: `google.com/tpu=present:NoSchedule`
- Must use `/opt/venv/bin/python` (not system Python)

---

### Task 6 — Inference pipeline adaptation: GKE job submission ✅ COMPLETE

**Goal:** Replace the `gcloud tpu-vm ssh --worker=all` invocation pattern with
a GKE job that runs the MiMo-V2-Flash demo / benchmark.

**Result (2026-05-08):** End-to-end inference confirmed working.
- Throughput: **1.5 tok/s** (BF16, dense, full prefill — unoptimised baseline)
- Job: `tools/orchestration/mimo_v2_flash_demo_job.yaml`
- Key flags: `--ici_tensor_parallelism 4 --ici_expert_parallelism 2 --enable_single_controller`
- Checkpoint: `gs://jingnw-mimo-v2-flash-us-east5/mimo-v2-flash-fixed-ocdbt/checkpoints/0/items` (cross-region from us-east5, acceptable)
- Bugs fixed: libtpu lockfile conflict (parent-process JAX init), K8s coordinator_address error

**Expected next-step improvements:**
- `scan_layers=true` (stacked checkpoint) → lower memory, faster per-step
- FP8 native matmuls (Task 7)
- GCS bucket in us-central1 (Task 8) to eliminate cross-region latency

On v6e the inference entry point was:
```bash
gcloud compute tpus tpu-vm ssh "$TPU_NAME" --zone "$ZONE" --worker=all \
  --command='cd ~/maxtext && python demos/mimo_v2_flash_demo_jax.py ...'
```

On v7x GKE the equivalent is a single-pod job (all 8 devices in one node):
```bash
kubectl create job mimo-demo --image=<IMAGE> \
  -- python3 /maxtext/demos/mimo_v2_flash_demo_jax.py \
     --checkpoint_path=gs://jingnw-mimo-v2-flash-us-central1/... \
     ...
```

**Action items:**
1. Create a Kubernetes Job manifest `tools/orchestration/mimo_v2_flash_demo_job.yaml`
   that:
   - Sets `resources.limits` for `google.com/tpu: 4` (or `8` depending on
     how v7x cores are exposed in the node pool)
   - Sets `nodeSelector` for the tpu7x node pool
   - Mounts a GCS bucket via gcsfuse sidecar (or passes GCS path directly)
   - Sets required env vars (`JAX_USE_PJRT_C_API_BACKEND=1`, etc.)
2. Verify that the checkpoint in `gs://jingnw-mimo-v2-flash-us-east5/` is
   accessible from `us-central1-c` (cross-region read).  If latency is
   unacceptable, copy the checkpoint to a new
   `gs://jingnw-mimo-v2-flash-us-central1/` bucket.
3. Update the demo script if needed to accept a `--tpu_type tpu7x-4` flag
   or to auto-detect the mesh shape.

**Files to update / create:**
- `tools/orchestration/mimo_v2_flash_demo_job.yaml` (new)
- `tools/orchestration/mimo_v2_flash_benchmark_job.yaml` (new)
- `demos/mimo_v2_flash_demo_jax.py` (possibly update mesh args)

---

### Task 7 — FP8 support validation on v7x

**Goal:** Verify that MiMo-V2-Flash's FP8 checkpoint (dequantized during
conversion) works correctly on v7x hardware.

MiMo-V2-Flash uses FP8-quantized weights at rest; they are dequantized to
BF16 during checkpoint conversion.  This is hardware-agnostic.  However, v7x
supports native FP8 matmuls (`float8_e4m3fn`) which could enable running
the model in true FP8 for extra performance.

**Action items:**
1. Confirm the converted checkpoint in GCS is BF16 (not raw FP8 HF format).
2. Run the smoke test demo in BF16 first (Task 5/6 prerequisite).
3. Evaluate whether native v7x FP8 inference is worth pursuing as a follow-on
   optimization (see `docs/guides/mimo_v2_flash_fp8_dtypes.md`).

---

### Task 8 — GCS bucket for us-central1

**Goal:** Ensure checkpoint data is co-located with the GKE cluster to avoid
cross-region egress costs and latency.

Current checkpoint: `gs://jingnw-mimo-v2-flash-us-east5/...` (us-east5)  
New cluster zone: `us-central1-c`

**Action items:**
1. Create a new regional GCS bucket:
   ```bash
   gcloud storage buckets create gs://jingnw-mimo-v2-flash-us-central1 \
     --location=us-central1 \
     --project=tpu-launchpad-playground
   ```
2. Copy the checkpoint (incremental, OCDBT format):
   ```bash
   gcloud storage cp -r \
     gs://jingnw-mimo-v2-flash-us-east5/mimo-v2-flash-fixed-ocdbt \
     gs://jingnw-mimo-v2-flash-us-central1/
   ```
3. Update all config references from the us-east5 path to the us-central1 path.

---

### Task 9 — New env restore guide for v7x GKE

**Goal:** Write `docs/guides/tpu_v7x_env_setup.md` mirroring the structure of
`docs/guides/mimo_v2_flash_env_restore.md` but for the GKE-based workflow.

Sections to cover:
1. Fixed settings (cluster name, zone, project, node pool, image, checkpoint)
2. One-time setup (kubectl credentials, xpk install, Docker auth)
3. Build / verify the container image
4. Submit and monitor a smoke-test job
5. Run the full inference demo
6. Run the benchmark
7. Read results
8. Troubleshooting (DWS node not scaling up, pod eviction, libtpu errors)

---

### Task 10 — SparseCore investigation (optional, follow-on)

**Goal:** Investigate whether MiMo-V2-Flash's MoE routing can leverage v7x
SparseCore for accelerated expert dispatch.

TPU v7x introduces SparseCore, designed to accelerate sparse operations
including MoE all-to-all.  This is a research/optimization task that should
**not block** the main porting.

**Files relevant:**
- `src/maxtext/utils/accelerator_to_spec_map.py` (already has `tpu7x` entries
  with `chip_config_name: "default"` — SparseCore may need `"sparsecore"`)
- `src/maxtext/layers/` (MoE dispatch and combine kernels)

---

## 2. Dependency Graph

```
Task 1 (cluster creds)
    └─► Task 2 (JAX/libtpu versions)
            └─► Task 3 (Docker image)
                    └─► Task 5 (JAX device smoke test)
                            ├─► Task 4 (model config)
                            │       └─► Task 6 (inference pipeline on GKE)
                            ├─► Task 7 (FP8 validation)          ──► Task 9 (docs)
                            └─► Task 8 (GCS bucket us-central1)  ──► Task 9 (docs)
```

---

## 3. Fixed Settings (v7x)

| Setting                | Value                                                                  |
|------------------------|------------------------------------------------------------------------|
| Project                | `tpu-launchpad-playground`                                             |
| GKE cluster            | `jingnw-tpu7-cluster`                                                  |
| Zone                   | `us-central1-c`                                                        |
| Node pool              | `jingnw-flex-tpu7`                                                     |
| Machine type           | `tpu7x-standard-4t` (4 chips, 8 JAX devices, 192 GiB HBM/chip)        |
| Autoscaling            | 0–1 nodes (DWS flex-start)                                             |
| Ops VM                 | `jingnw-tpu-op` (re-created, Debian 12, e2-small)                      |
| Branch                 | `tpu-v7x-porting`                                                      |
| Source checkpoint      | `gs://jingnw-mimo-v2-flash-us-east5/mimo-v2-flash-fixed-ocdbt/...`    |
| Target checkpoint      | `gs://jingnw-mimo-v2-flash-us-central1/mimo-v2-flash-fixed-ocdbt/...` |
| Tokenizer              | `XiaomiMiMo/MiMo-V2-Flash`                                            |

---

## 4. Open Questions

1. **libtpu version for v7x:** Which minimum libtpu release supports
   `tpu7x-standard-4t`?  Needs verification against the libtpu release notes or
   `gs://libtpu-builds/`.
2. **Resolved:** the Kubernetes TPU resource for `tpu7x-standard-4t` is
   `google.com/tpu: 4`, while JAX reports 8 devices on the host.
3. **DWS node scale-up latency:** Flex-start nodes may take 2–10 minutes to
   provision.  Does the job need an init-container retry loop?
4. **Model fit on 4 chips:** At FP8-converted BF16, MiMo-V2-Flash is ~155 GB.
   With 4 × 192 GiB = 768 GiB HBM available, the model should fit.  Confirm
   with a dry-run shape check before loading the full checkpoint.
5. **Cross-region checkpoint read:** If reading from us-east5 proves acceptable
   (GCS cross-region egress is fast), the GCS copy in Task 8 can be deferred.
