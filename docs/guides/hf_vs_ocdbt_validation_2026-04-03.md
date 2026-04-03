# HF vs OCDBT Direct Validation (2026-04-03)

## Scope

This validation compares original HF safetensor values directly against regenerated zarr3+OCDBT checkpoint values.

- HF source: `gs://jingnw-mimo-v2-flash-us-east5/hf-model`
- OCDBT target: `gs://jingnw-mimo-v2-flash-us-east5/mimo-v2-flash-ocdbt/checkpoints/0/items/`
- TPU workers: 8
- Validation mode: distributed by layer partition (`layer % 8`)

## Final Result

- Workers completed: `8/8`
- Total tensors checked: `568`
- Total mismatches: `0`
- Global max absolute diff: `0.0`

Per-worker checked counts:

- worker 0: 72
- worker 1: 70
- worker 2: 72
- worker 3: 70
- worker 4: 72
- worker 5: 70
- worker 6: 72
- worker 7: 70

Conclusion: direct HF values and OCDBT values are equivalent for all validated tensors.

## Saved Evidence

- Raw per-worker summary:
  - `validation_artifacts/hf_vs_ocdbt_2026-04-03_worker_all_summary.txt`
- Aggregate JSON:
  - `validation_artifacts/hf_vs_ocdbt_2026-04-03_aggregate.json`

## Operational Notes

1. Prefer `--worker=all` for polling/summary commands.
2. Avoid `pkill -f validate_hf_vs_ocdbt_distributed.py` in remote SSH command strings.
3. If `ssh` returns code `255`, respawn ssh-agent and re-add key:

```bash
eval "$(ssh-agent -s)"
ssh-add ~/.ssh/google_compute_engine
gcloud compute os-login ssh-keys add --key-file="$HOME/.ssh/google_compute_engine.pub"
```

## Reusable Summary Command

Use:

```bash
bash tools/dev/hf_vs_ocdbt_worker_all_summary.sh jingnw-node
```

Environment knobs:

- `ZONE` (default: `us-east5-b`)
- `INTERVAL_SEC` (default: `30`)
- `MAX_ROUNDS` (default: `120`)
