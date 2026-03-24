#!/usr/bin/env bash
# Qwen3 / Qwen3-VL inference demo using decode.py
#
# Usage:
#   bash run_inference_demo.sh                                          # text-only Qwen3-4B
#   bash run_inference_demo.sh --vl                                     # Qwen3-VL-2B with default images + video
#   bash run_inference_demo.sh --vl --image PATH1 PATH2          # custom images
#   bash run_inference_demo.sh --vl --image PATH1 PATH2 --video PATH  # custom images + video
#   bash run_inference_demo.sh --prompt "…"                             # custom prompt
#
# Prerequisites (first run only):
#   The script will auto-convert HF weights if the checkpoint is not found in GCS.
#   Set HF_TOKEN if the model requires authentication.
#
# Requirements: GCS bucket configured in BASE_OUTPUT_DIR below.

set -euo pipefail

# ── Configuration ────────────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE_OUTPUT_DIR="gs://jingnw_tpu"
TOKENIZER_PATH="src/maxtext/assets/tokenizers/qwen3-tokenizer"
DEFAULT_IMAGE1="tests/assets/image1.jpg"
DEFAULT_IMAGE2="tests/assets/image2.jpg"
DEFAULT_VIDEO="tests/assets/video.mp4"

# Parallelism defaults (auto = use all chips via -1)
# ICI_AR: Autoregressive (batch) parallelism — best for small models on many chips.
#         Each chip holds a full model copy and serves separate batch slots.
#         e.g. 4 chips × per_device_batch_size=1 → 4 concurrent requests.
# ICI_TP: Tensor parallelism — shards weight matrices across chips.
#         Better for large models that don't fit on one chip; adds comm overhead.
# Both can be combined (e.g. ICI_TP=2 ICI_AR=2 on an 8-chip host).
ICI_AR=1           # ici_autoregressive_parallelism (set to -1 to use all chips)
ICI_TP=1           # ici_tensor_parallelism

# Text-only defaults
TEXT_MODEL="qwen3-4b"
TEXT_CKPT="${BASE_OUTPUT_DIR}/qwen3-4b-converted-scanned/0/items"
TEXT_PROMPT="Tell me a short story."
TEXT_MAX_PREFILL=512
TEXT_MAX_TARGET=1024

# Multimodal defaults
VL_MODEL="qwen3-vl-2b"
VL_CKPT="${BASE_OUTPUT_DIR}/qwen3-vl-2b-converted/0/items"
VL_PROMPT="There are two images and a video clip provided. Describe what you see in each image and summarize the main scene in the video."
VL_MAX_PREFILL=1024
VL_MAX_TARGET=1536

# ── Argument parsing ─────────────────────────────────────────────────────────
MODE="text"
IMAGE_PATHS=("${DEFAULT_IMAGE1}" "${DEFAULT_IMAGE2}")  # default: 2 images
VIDEO_PATH="${DEFAULT_VIDEO}"
CUSTOM_PROMPT=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --vl)   MODE="vl"; shift ;;
    --image)
      IMAGE_PATHS=()
      shift
      while [[ $# -gt 0 && "${1:0:2}" != "--" ]]; do
        IMAGE_PATHS+=("$1")
        shift
      done
      ;;
    --video)     VIDEO_PATH="$2"    ; shift 2 ;;
    --prompt)    CUSTOM_PROMPT="$2" ; shift 2 ;;
    --ar)        ICI_AR="$2"        ; shift 2 ;;   # e.g. --ar 4
    --tp)        ICI_TP="$2"        ; shift 2 ;;   # e.g. --tp 4
    *) echo "Unknown option: $1" >&2; exit 1 ;;
  esac
done

# ── Select parameters for chosen mode ────────────────────────────────────────
if [[ "${MODE}" == "vl" ]]; then
  MODEL="${VL_MODEL}"
  CKPT="${VL_CKPT}"
  PROMPT="${CUSTOM_PROMPT:-${VL_PROMPT}}"
  MAX_PREFILL="${VL_MAX_PREFILL}"
  MAX_TARGET="${VL_MAX_TARGET}"
  # Comma-separated image paths; decode.py stacks them as multiple visual entries.
  IMAGE_LIST=$(IFS=, ; echo "${IMAGE_PATHS[*]}")
  EXTRA_ARGS="image_path=\"${IMAGE_LIST}\""
else
  MODEL="${TEXT_MODEL}"
  CKPT="${TEXT_CKPT}"
  PROMPT="${CUSTOM_PROMPT:-${TEXT_PROMPT}}"
  MAX_PREFILL="${TEXT_MAX_PREFILL}"
  MAX_TARGET="${TEXT_MAX_TARGET}"
  EXTRA_ARGS=""
fi

# ── Auto-convert checkpoint if missing ───────────────────────────────────────
PARENT_DIR="${CKPT%/0/items}"   # strip /0/items to get the base directory

if ! gsutil -q stat "${CKPT}/_metadata" 2>/dev/null && \
   ! gsutil ls "${CKPT}/" 2>/dev/null | grep -q "."; then
  echo "Checkpoint not found at ${CKPT}."
  echo "Converting ${MODEL} from HuggingFace — this takes ~3-5 minutes …"
  python src/maxtext/checkpoint_conversion/to_maxtext.py \
    src/maxtext/configs/base.yml \
    model_name="${MODEL}" \
    base_output_directory="${PARENT_DIR%/0}" \
    hardware=cpu \
    skip_jax_distributed_system=True \
    ${HF_TOKEN:+hf_access_token="${HF_TOKEN}"}
  echo "Conversion complete."
fi

# ── Run inference ─────────────────────────────────────────────────────────────
echo ""
echo "Running ${MODEL} inference (mode=${MODE}) …"
echo "  Prompt  : ${PROMPT}"
if [[ "${MODE}" == "vl" ]]; then
  for p in "${IMAGE_PATHS[@]}"; do
    echo "  Image   : ${p}"
  done
  echo "  Video   : ${VIDEO_PATH}"
fi
echo "  Ckpt    : ${CKPT}"
echo "  ICI AR  : ${ICI_AR}  (ici_autoregressive_parallelism; set --ar -1 to use all chips)"
echo "  ICI TP  : ${ICI_TP}  (ici_tensor_parallelism)"
echo "  Visible chips: $(python3 -c 'import jax; print(jax.device_count())' 2>/dev/null || echo '?')"
echo ""

python3 -m maxtext.inference.decode src/maxtext/configs/base.yml \
  model_name="${MODEL}" \
  load_parameters_path="${CKPT}" \
  tokenizer_path="${TOKENIZER_PATH}" \
  ${EXTRA_ARGS:+${EXTRA_ARGS}} \
  prompt="${PROMPT}" \
  ici_autoregressive_parallelism="${ICI_AR}" \
  ici_tensor_parallelism="${ICI_TP}" \
  per_device_batch_size=1 \
  max_prefill_predict_length="${MAX_PREFILL}" \
  max_target_length="${MAX_TARGET}" \
  2>&1 | grep -A200 "^Input"
