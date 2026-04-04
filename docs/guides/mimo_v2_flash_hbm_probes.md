# MaxText HBM Probe Points

`_probe_hbm` / `_probe_hbm_engine` calls in the MaxText/TPU inference pipeline.
**These exist only in the TPU path — the HF Transformers CPU path has no probe points.**

| # | File | Line | Label |
|---|---|---|---|
| 1 | `src/maxtext/inference/decode.py` | 115 | `init` |
| 2 | `src/maxtext/inference/decode.py` | 122 | `after_load_params` |
| 3 | `src/maxtext/inference/decode.py` | 219 | `after_prefill` |
| 4 | `src/maxtext/inference/decode.py` | 227 | `after_insert` |
| 5 | `src/maxtext/inference/decode.py` | 243 | `generate_step_{i:04d}` (one per decode step) |
| 6 | `src/maxtext/inference/maxengine/maxengine.py` | 256 | `before_setup_decode_state` |
| 7 | `src/maxtext/inference/maxengine/maxengine.py` | 260 | `after_setup_decode_state` |
| 8 | `demos/compare/maxtext_reference.py` | 112 | `init` |
| 9 | `demos/compare/maxtext_reference.py` | 128 | `after_load_params` |
| 10 | `demos/compare/maxtext_reference.py` | 167 | `after_prefill` |
| 11 | `demos/compare/maxtext_reference.py` | 175 | `after_insert` |

Row 5 fires once per autoregressive step, so the total number of probe calls is
`7 + max_new_tokens` (for a typical `decode.py` + `maxengine.py` run).
