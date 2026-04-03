#!/usr/bin/env bash
# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at https://www.apache.org/licenses/LICENSE-2.0

# demos/compare/setup.sh
# ──────────────────────────────────────────────────────────────────────────────
# One-time setup for running the HF CPU reference on a single TPU worker VM.
#
# Run on WORKER-0 only (it has enough unused RAM when the TPU is idle):
#   gcloud compute tpus tpu-vm ssh jingnw-node --zone=us-east5-b --worker=0 \
#     --internal-ip --command='bash $HOME/maxtext/demos/compare/setup.sh'
#
# What this script does:
#   1. Creates a separate venv ($HOME/compare_venv) for torch+transformers
#      (kept separate from maxtext_venv to avoid JAX/PyTorch conflicts).
#   2. Installs CPU-only PyTorch, transformers, safetensors, accelerate.
#   3. Downloads the HF model weights from GCS to /tmp/mimo-hf-model/.
#      (~320 GB on disk in FP8; plan for this taking 30-60 min from GCS).
#
# After this script finishes, run the reference:
#   source $HOME/compare_venv/bin/activate
#   python3 $HOME/maxtext/demos/compare/hf_reference.py \
#       --model_path /tmp/mimo-hf-model \
#       --tokenizer_path $HOME/mimo-tokenizer \
#       --max_new_tokens 16 \
#       --out_dir /tmp/compare_hf
# ──────────────────────────────────────────────────────────────────────────────

set -euo pipefail

HF_MODEL_GCS="gs://jingnw-mimo-v2-flash-us-east5/hf-model"
LOCAL_MODEL_DIR="/tmp/mimo-hf-model"
VENV_DIR="$HOME/compare_venv"
TORCH_INDEX="https://download.pytorch.org/whl/cpu"

# ── Step 1: Create comparison venv ──────────────────────────────────────────
echo "=== Creating compare_venv at ${VENV_DIR} ==="
if [[ ! -d "${VENV_DIR}" ]]; then
  python3 -m venv "${VENV_DIR}"
fi
source "${VENV_DIR}/bin/activate"

echo "=== Installing PyTorch (CPU-only) + transformers ==="
pip install --quiet --upgrade pip
# CPU-only torch (~200 MB) — much smaller than CUDA builds
pip install --quiet torch --index-url "${TORCH_INDEX}"
pip install --quiet "transformers>=4.40" safetensors accelerate

echo "=== Verifying torch ==="
python3 -c "import torch; print('torch', torch.__version__)"
python3 -c "import transformers; print('transformers', transformers.__version__)"

# ── Step 2: Download HF model weights from GCS ──────────────────────────────
echo ""
echo "=== Downloading HF model from ${HF_MODEL_GCS} to ${LOCAL_MODEL_DIR} ==="
echo "    (This transfers ~320 GB; allow 30-90 minutes.)"
mkdir -p "${LOCAL_MODEL_DIR}"

# Download everything except safetensors first (small config files)
gsutil -m cp \
  "${HF_MODEL_GCS}/config.json" \
  "${HF_MODEL_GCS}/tokenizer.json" \
  "${HF_MODEL_GCS}/tokenizer_config.json" \
  "${HF_MODEL_GCS}/special_tokens_map.json" \
  "${HF_MODEL_GCS}/vocab.json" \
  "${HF_MODEL_GCS}/merges.txt" \
  "${HF_MODEL_GCS}/model.safetensors.index.json" \
  "${LOCAL_MODEL_DIR}/"

# Decompress config.json if gzip-encoded (GCS sometimes uses transparent gzip)
if file "${LOCAL_MODEL_DIR}/config.json" | grep -q gzip; then
  echo "  config.json is gzip-encoded; decompressing..."
  mv "${LOCAL_MODEL_DIR}/config.json" "${LOCAL_MODEL_DIR}/config.json.gz"
  gunzip "${LOCAL_MODEL_DIR}/config.json.gz"
fi

# Download the safetensors weight shards (large — ~320 GB total)
echo "Downloading weight shards (this is the slow part)..."
gsutil -m cp "${HF_MODEL_GCS}/model_*.safetensors" "${LOCAL_MODEL_DIR}/"

# Verify
echo ""
echo "=== Download complete ==="
ls -lh "${LOCAL_MODEL_DIR}/" | head -20
echo "Total disk usage:"
du -sh "${LOCAL_MODEL_DIR}"

echo ""
echo "=== Setup complete ==="
echo "Activate with:  source ${VENV_DIR}/bin/activate"
echo "Then run:       python3 \$HOME/maxtext/demos/compare/hf_reference.py \\"
echo "                    --model_path ${LOCAL_MODEL_DIR} \\"
echo "                    --tokenizer_path \$HOME/mimo-tokenizer \\"
echo "                    --max_new_tokens 16 --out_dir /tmp/compare_hf"
