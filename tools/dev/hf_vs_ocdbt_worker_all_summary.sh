#!/usr/bin/env bash
set -euo pipefail

NODE_NAME="${1:-jingnw-node}"
ZONE="${ZONE:-us-east5-b}"
INTERVAL_SEC="${INTERVAL_SEC:-30}"
MAX_ROUNDS="${MAX_ROUNDS:-120}"

respawn_ssh() {
  if [[ -z "${SSH_AUTH_SOCK:-}" ]] || ! ssh-add -l >/dev/null 2>&1; then
    eval "$(ssh-agent -s)" >/dev/null
    ssh-add "$HOME/.ssh/google_compute_engine" >/dev/null
  fi
  gcloud compute os-login ssh-keys add --key-file="$HOME/.ssh/google_compute_engine.pub" >/dev/null
}

poll_once() {
  gcloud compute tpus tpu-vm ssh "$NODE_NAME" \
    --worker=all \
    --zone="$ZONE" \
    --internal-ip \
    --command='python3 - <<"PY"
import glob
import json
import socket

files = glob.glob("/tmp/hf_vs_ocdbt_result_w*.json")
if not files:
    print(f"SUM host={socket.gethostname()} status=missing")
else:
    f = files[0]
    obj = json.load(open(f))
    checked = int(obj.get("checked", 0))
    mism = obj.get("mismatches", []) or []
    local_max = 0.0
    local_key = "None"
    for m in mism:
        v = float(m.get("max_abs_diff", 0.0))
        if v > local_max:
            local_max = v
            local_key = m.get("key", "None")
    print(
        f"SUM host={socket.gethostname()} file={f} checked={checked} "
        f"mismatches={len(mism)} local_max_abs={local_max} local_max_key={local_key}"
    )
PY'
}

aggregate_file() {
  local in_file="$1"
  python3 - "$in_file" <<'PY'
import re
import sys

pat = re.compile(r'^SUM host=(\S+) file=(\S+) checked=(\d+) mismatches=(\d+) local_max_abs=([0-9.eE+-]+) local_max_key=(.*)$')
rows = []
for line in open(sys.argv[1]):
    m = pat.match(line.strip())
    if not m:
        continue
    host, file_name, checked, mismatches, local_max, local_key = m.groups()
    rows.append((host, file_name, int(checked), int(mismatches), float(local_max), local_key))

print(f"parsed_workers={len(rows)}")
total_checked = sum(r[2] for r in rows)
total_mismatches = sum(r[3] for r in rows)
global_max_abs = 0.0
global_max_key = "None"
global_max_host = "None"
for host, _file_name, _checked, _mismatches, local_max, local_key in rows:
    if local_max > global_max_abs:
        global_max_abs = local_max
        global_max_key = local_key
        global_max_host = host

print(
    f"SUMMARY total_checked={total_checked} total_mismatches={total_mismatches} "
    f"global_max_abs={global_max_abs} global_max_key={global_max_key} global_max_host={global_max_host}"
)
PY
}

respawn_ssh
TMP_OUT="$(mktemp)"

for round in $(seq 1 "$MAX_ROUNDS"); do
  echo "=== poll_round=${round} ==="
  set +e
  out="$(poll_once 2>&1)"
  rc=$?
  set -e

  if [[ $rc -ne 0 ]]; then
    echo "$out"
    if echo "$out" | grep -q 'return code \[255\]'; then
      echo "SSH 255 detected; respawning key and retrying next round"
      respawn_ssh
    fi
    sleep 10
    continue
  fi

  printf "%s\n" "$out" | tee "$TMP_OUT"

  done_count="$(printf "%s\n" "$out" | grep -c ' file=/tmp/hf_vs_ocdbt_result_w' || true)"
  if [[ "$done_count" -eq 8 ]]; then
    echo "--- aggregate ---"
    aggregate_file "$TMP_OUT"
    exit 0
  fi

  sleep "$INTERVAL_SEC"
done

echo "Timeout: result files not found on all 8 workers within MAX_ROUNDS=$MAX_ROUNDS"
exit 1
