#!/usr/bin/env bash
# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at https://www.apache.org/licenses/LICENSE-2.0

# demos/compare/setup.sh
# ──────────────────────────────────────────────────────────────────────────────
# One-time setup for running the HF CPU reference on TPU worker VMs.
#
# KEY DESIGN:
#   • The HF model (291 GB on GCS) does NOT fit on a single worker's 97 GB disk.
#   • Each worker has 708 GB RAM (~700 GB free) — plenty for the BF16 model.
#   • We use gcsfuse (pre-installed on TPU VMs) to MOUNT the GCS bucket as a
#     local directory so PyTorch can read safetensors directly from GCS without
#     any local download.
#   • CPU-only torch + transformers are installed via system pip3 --user
#     (avoids venv pip-bootstrap issues).
#
# Can run on ALL 8 workers simultaneously:
#   gcloud compute tpus tpu-vm ssh jingnw-node --zone=us-east5-b --worker=all \
#     --internal-ip --command='bash $HOME/maxtext/demos/compare/setup.sh'
#
# After setup, run the reference (per-worker, each writes to its own dir):
#   python3 $HOME/maxtext/demos/compare/hf_reference.py \
#       --model_path /tmp/mimo-hf-gcs \
#       --tokenizer_path $HOME/mimo-tokenizer \
#       --max_new_tokens 16 \
#       --out_dir /tmp/compare_hf
# ──────────────────────────────────────────────────────────────────────────────

set -euo pipefail

GCS_BUCKET="jingnw-mimo-v2-flash-us-east5"
GCS_SUBDIR="hf-model"
MOUNT_DIR="/tmp/mimo-hf-gcs"
TORCH_INDEX="https://download.pytorch.org/whl/cpu"

# ── Step 1: Install CPU-only torch + transformers via system pip3 --user ─────
echo "=== Installing PyTorch (CPU-only) + transformers via pip3 --user ==="
pip3 install --quiet --user torch --index-url "${TORCH_INDEX}"
pip3 install --quiet --user "transformers==4.47.0" accelerate

echo "=== Verifying ==="
python3 -c "import torch; print('torch', torch.__version__)"
python3 -c "import transformers; print('transformers', transformers.__version__)"
# safetensors is already in maxtext_venv, but also install for system python:
pip3 install --quiet --user safetensors
python3 -c "import safetensors; print('safetensors', safetensors.__version__)"

# ── Step 2: Mount GCS bucket with gcsfuse ─────────────────────────────────
echo ""
echo "=== Mounting gs://${GCS_BUCKET}/${GCS_SUBDIR} at ${MOUNT_DIR} ==="

# Unmount if already mounted (ignore errors)
fusermount -u "${MOUNT_DIR}" 2>/dev/null || true

mkdir -p "${MOUNT_DIR}"

# gcsfuse mounts a single bucket; we mount the full bucket then point at the subdir.
# Use --implicit-dirs so safetensors index can enumerate files.
/usr/bin/gcsfuse \
  --implicit-dirs \
  --stat-cache-ttl=60s \
  --type-cache-ttl=60s \
  "${GCS_BUCKET}" "${MOUNT_DIR}"

echo "Mount OK. Listing top-level files in subdir:"
ls "${MOUNT_DIR}/${GCS_SUBDIR}/" | head -10

# Write the actual model path (bucket_mount/subdir) for convenience
echo "${MOUNT_DIR}/${GCS_SUBDIR}" > /tmp/hf_model_path.txt
echo ""
echo "=== Setup complete on $(hostname) ==="
echo "Model path:  ${MOUNT_DIR}/${GCS_SUBDIR}"
echo ""
echo "Run reference:"
echo "  python3 \$HOME/maxtext/demos/compare/hf_reference.py \\"
echo "      --model_path ${MOUNT_DIR}/${GCS_SUBDIR} \\"
echo "      --tokenizer_path \$HOME/mimo-tokenizer \\"
echo "      --max_new_tokens 16 --out_dir /tmp/compare_hf"
