#!/usr/bin/env bash
# Stream MiMo-V2.5-Pro HF files directly to GCS using curl | gsutil cp -.
# No Python deps required — only curl and gsutil must be available.
#
# Usage:
#   bash tools/dev/upload_mimo_v25pro_hf_to_gcs.sh
#
# Resumable: already-uploaded files are skipped automatically (--skip_existing).

set -euo pipefail

REPO_ID="XiaomiMiMo/MiMo-V2.5-Pro"
BUCKET="jingnw-mimo-v2-5-pro-us-central1"
GCS_PREFIX="hf-weights"
HF_BASE="https://huggingface.co/${REPO_ID}/resolve/main"

# Files to upload: all safetensors shards + essential config/tokenizer files.
# assets/ and README are omitted — not needed for inference.
FILES=(
  config.json
  configuration_mimo_v2.py
  modeling_mimo_v2.py
  model.safetensors.index.json
  tokenizer_config.json
  tokenizer.json
  special_tokens_map.json
  added_tokens.json
  vocab.json
  merges.txt
  model_mtp.safetensors
  model_pp0_ep0_shard0.safetensors
  model_pp0_ep0_shard1.safetensors
  model_pp0_ep1_shard0.safetensors
  model_pp0_ep2_shard0.safetensors
  model_pp0_ep3_shard0.safetensors
  model_pp0_ep4_shard0.safetensors
  model_pp0_ep5_shard0.safetensors
  model_pp0_ep6_shard0.safetensors
  model_pp0_ep7_shard0.safetensors
  model_pp0_ep8_shard0.safetensors
  model_pp0_ep9_shard0.safetensors
  model_pp0_ep10_shard0.safetensors
  model_pp0_ep11_shard0.safetensors
  model_pp0_ep12_shard0.safetensors
  model_pp0_ep13_shard0.safetensors
  model_pp0_ep14_shard0.safetensors
  model_pp0_ep15_shard0.safetensors
  model_pp0_ep16_shard0.safetensors
  model_pp0_ep17_shard0.safetensors
  model_pp0_ep18_shard0.safetensors
  model_pp0_ep19_shard0.safetensors
  model_pp0_ep20_shard0.safetensors
  model_pp0_ep21_shard0.safetensors
  model_pp0_ep22_shard0.safetensors
  model_pp0_ep23_shard0.safetensors
  model_pp0_ep24_shard0.safetensors
  model_pp0_ep25_shard0.safetensors
  model_pp0_ep26_shard0.safetensors
  model_pp0_ep27_shard0.safetensors
  model_pp0_ep28_shard0.safetensors
  model_pp0_ep29_shard0.safetensors
  model_pp0_ep30_shard0.safetensors
  model_pp0_ep31_shard0.safetensors
)

TOTAL=${#FILES[@]}
OK=0
SKIP=0
FAIL=0

echo "Uploading ${TOTAL} files from ${REPO_ID} to gs://${BUCKET}/${GCS_PREFIX}/"
echo ""

for i in "${!FILES[@]}"; do
  FILE="${FILES[$i]}"
  GCS_DEST="gs://${BUCKET}/${GCS_PREFIX}/${FILE}"
  IDX=$((i + 1))

  # Skip if already exists in GCS.
  if gsutil -q stat "${GCS_DEST}" 2>/dev/null; then
    echo "[${IDX}/${TOTAL}] SKIP (exists): ${FILE}"
    SKIP=$((SKIP + 1))
    continue
  fi

  URL="${HF_BASE}/${FILE}"
  echo "[${IDX}/${TOTAL}] Uploading: ${FILE} ..."
  START=$(date +%s)

  if curl -fsSL --retry 3 --retry-delay 5 "${URL}" \
      | gsutil cp - "${GCS_DEST}"; then
    END=$(date +%s)
    echo "[${IDX}/${TOTAL}] OK ($(( END - START ))s): ${FILE}"
    OK=$((OK + 1))
  else
    echo "[${IDX}/${TOTAL}] FAILED: ${FILE}" >&2
    FAIL=$((FAIL + 1))
  fi
done

echo ""
echo "Done: ${OK} uploaded, ${SKIP} skipped, ${FAIL} failed."
echo "GCS path: gs://${BUCKET}/${GCS_PREFIX}/"
[[ ${FAIL} -eq 0 ]] || exit 1
