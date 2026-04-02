#!/usr/bin/env python3
"""Stream HuggingFace MiMo-V2-Flash model files directly to GCS.

Downloads every safetensors shard + config/tokenizer files from the
HuggingFace Hub and uploads them directly to a GCS bucket using a
streaming pipeline.  No large local disk is required — peak local
disk usage is one HTTP read buffer (~16 MB) at a time.

Usage (run on a VM with gcloud auth and internet access):
  python3 tools/dev/upload_mimo_hf_to_gcs.py \
      --bucket jingnw-mimo-v2-flash-us-east5 \
      --gcs_prefix hf-model \
      --repo_id XiaomiMiMo/MiMo-V2-Flash
"""

import argparse
import sys
import time

import requests
from google.cloud import storage
from huggingface_hub import list_repo_files, hf_hub_url

# Non-safetensors files to always upload (config + tokenizer)
_EXTRA_FILES = {
    "config.json",
    "generation_config.json",
    "model.safetensors.index.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "tokenizer.model",
    "special_tokens_map.json",
    "vocab.json",
    "merges.txt",
}


def stream_file_to_gcs(url: str, blob: storage.Blob, chunk_size: int = 16 * 1024 * 1024) -> None:
    """Stream a URL response directly to a GCS blob (no local disk)."""
    with requests.get(url, stream=True, timeout=300) as resp:
        resp.raise_for_status()
        blob.upload_from_file(resp.raw, content_type="application/octet-stream")


def main():
    parser = argparse.ArgumentParser(
        description="Stream MiMo-V2-Flash HF model files directly to GCS."
    )
    parser.add_argument("--bucket", required=True, help="GCS bucket name (without gs://)")
    parser.add_argument("--gcs_prefix", default="hf-model", help="GCS object prefix")
    parser.add_argument("--repo_id", default="XiaomiMiMo/MiMo-V2-Flash",
                        help="HuggingFace Hub repo id")
    parser.add_argument("--hf_token", default=None,
                        help="HuggingFace token (for private repos)")
    parser.add_argument("--skip_existing", action="store_true", default=True,
                        help="Skip files already present in GCS (resumable)")
    parser.add_argument("--project", default=None, help="GCS project id")
    args = parser.parse_args()

    gcs_prefix = args.gcs_prefix.rstrip("/")
    client = storage.Client(project=args.project)
    bucket = client.bucket(args.bucket)

    print(f"Listing files in {args.repo_id} ...")
    all_files = list(list_repo_files(args.repo_id, token=args.hf_token))
    safetensors = [f for f in all_files if f.endswith(".safetensors")]
    extras = [f for f in all_files if f in _EXTRA_FILES]
    to_upload = extras + safetensors
    print(f"  {len(safetensors)} safetensors shards + {len(extras)} config/tokenizer files")

    ok = 0
    skip = 0
    fail = 0
    for i, filename in enumerate(to_upload):
        gcs_path = f"{gcs_prefix}/{filename}"
        blob = bucket.blob(gcs_path)
        if args.skip_existing and blob.exists():
            print(f"[{i+1}/{len(to_upload)}] SKIP (exists): {filename}")
            skip += 1
            continue

        url = hf_hub_url(args.repo_id, filename)
        print(f"[{i+1}/{len(to_upload)}] Uploading: {filename} ...", end="", flush=True)
        t0 = time.perf_counter()
        try:
            stream_file_to_gcs(url, blob)
            elapsed = time.perf_counter() - t0
            print(f" done ({elapsed:.1f}s)")
            ok += 1
        except Exception as e:  # pylint: disable=broad-except
            print(f" FAILED: {e}", file=sys.stderr)
            fail += 1

    print(f"\nDone: {ok} uploaded, {skip} skipped, {fail} failed.")
    if fail:
        sys.exit(1)


if __name__ == "__main__":
    main()
