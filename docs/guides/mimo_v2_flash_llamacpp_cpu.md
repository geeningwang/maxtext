# MiMo-V2-Flash llama.cpp CPU Inference (GGUF)

This guide documents how to run MiMo-V2-Flash inference using
[llama.cpp](https://github.com/ggml-org/llama.cpp) on a CPU-only host via
Q8_0 GGUF quantization.  The setup was validated on **worker-0** of the
`jingnw-node` TPU VM cluster: AMD EPYC 9B14, 180 vCPUs, 708 GB RAM,
`avx512_bf16` / AVX-512 VNNI / VBMI present.

**No patches are required.** llama.cpp HEAD `9c69907` supports
`MiMoV2FlashForCausalLM` natively (`arch = mimo2` in GGUF metadata).

---

## Infrastructure Layout

| Node | Role | Key storage |
|---|---|---|
| worker-0 (`AMD EPYC 9B14`) | 180 vCPUs, 708 GB RAM — llama.cpp server host | `~/llama.cpp` built; NFS client |
| worker-1 (`10.202.0.151`) | HF safetensors weights in tmpfs | `/mnt/mimo-weights` 650 G tmpfs, NFS-exported read-only |
| worker-2 (`10.202.0.29`) | GGUF scratch space + GGUF conversion host | `/mnt/gguf-scratch` 650 G tmpfs, NFS-exported read-write |

### Why llama-server runs on worker-0, not worker-2

Worker-2's 650 G tmpfs is ~47% occupied by the 306 G GGUF itself.  The
remaining ~345 G is not enough for resident model weights (306 G) plus KV
cache and OS overhead. Worker-0 has 583 G free RAM and reads the GGUF over
NFS via `mmap`, so only the pages actively needed for the current batch are
faulted in.

### NFS mounts on worker-0

```
10.202.0.151:/mnt/mimo-weights  →  /mnt/mimo-weights   (ro)   # safetensors
10.202.0.29:/mnt/gguf-scratch   →  /mnt/gguf-scratch   (rw)   # GGUF
```

---

## Source Data

| Artifact | Location | Size |
|---|---|---|
| HF safetensors weights | `/mnt/mimo-weights/` (NFS from worker-1) | 292 G |
| Q8_0 GGUF | `/mnt/gguf-scratch/mimo-v2-flash-Q8_0.gguf` (NFS from worker-2) | 306 G (8.50 BPW) |

---

## Step 1 — GGUF Conversion (worker-2, already done)

The GGUF was produced once on worker-2 using llama.cpp's
`convert_hf_to_gguf.py`, reading the safetensors weights from worker-1 over
NFS.

```bash
# worker-2 — llama.cpp HEAD 9c69907
git clone --depth=1 https://github.com/ggml-org/llama.cpp.git ~/llama.cpp

python3 ~/llama.cpp/convert_hf_to_gguf.py /mnt/mimo-weights \
  --outfile /mnt/gguf-scratch/mimo-v2-flash-Q8_0.gguf \
  --outtype q8_0
```

Runtime: ~35 min @ 159 MB/s.  Output: 306 G at
`/mnt/gguf-scratch/mimo-v2-flash-Q8_0.gguf`.

No build of llama.cpp binaries (`llama-server`, `llama-cli`, etc.) is needed
on worker-2 — `convert_hf_to_gguf.py` is pure Python.

---

## Step 2 — Build llama.cpp on worker-0 (already done)

```bash
# worker-0
git clone --depth=1 https://github.com/ggml-org/llama.cpp.git ~/llama.cpp
# Pinned HEAD: 9c69907

cd ~/llama.cpp

cmake -B build \
  -DGGML_NATIVE=ON \
  -DGGML_AVX512=ON \
  -DGGML_AVX512_VBMI=ON \
  -DGGML_AVX512_VNNI=ON \
  -DGGML_AVX2=ON \
  -DBUILD_SHARED_LIBS=OFF \
  -DCMAKE_BUILD_TYPE=Release

cmake --build build --config Release -j180
# Runtime: ~75 s on 180 cores
```

Key binaries at `~/llama.cpp/build/bin/`:

| Binary | Purpose |
|---|---|
| `llama-server` | OpenAI-compatible HTTP server |
| `llama-cli` | Interactive CLI / single-shot completion |
| `llama-bench` | Throughput benchmarking |
| `llama-gguf` | GGUF file inspection |

**No patches required.** llama.cpp natively supports the `mimo2` architecture
(`arch = mimo2` in the GGUF metadata, mapped from `MiMoV2FlashForCausalLM` in
`convert_hf_to_gguf.py`).

---

## Step 3 — Launch the Server

```bash
~/llama.cpp/build/bin/llama-server \
  --model /mnt/gguf-scratch/mimo-v2-flash-Q8_0.gguf \
  --host 0.0.0.0 --port 8080 \
  --threads 176 \
  --ctx-size 2048 \
  --n-gpu-layers 0
```

**Key flags:**

| Flag | Reason |
|---|---|
| `--threads 176` | Leave a few cores free for OS / NFS; 176 of 180 vCPUs |
| `--ctx-size 2048` | KV cache per slot × 4 default slots; fits in available RAM |
| `--n-gpu-layers 0` | No GPU; all layers run on CPU |

**Startup time:** ~5.5 min (306 G mmap'd over NFS at ~1 GB/s).

The server is ready when the log shows:

```
main: model loaded
main: server is listening on http://0.0.0.0:8080
```

Log file: `/tmp/llama_server.log`

---

## Testing

### Completions endpoint

```bash
curl -s http://localhost:8080/v1/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "mimo-v2-flash",
       "prompt": "What is 1+1? The answer is ",
       "n_predict": 10}'
```

**Observed output (validated 2026-04-04):**

```json
{"choices":[{"text":"2. But what is 0+0?","finish_reason":"length"}],
 "model":"mimo-v2-flash-Q8_0.gguf","system_fingerprint":"b1-9c69907"}
```

Output is **coherent** — contrast with the garbled `葭葭葭…` produced by
SGLang's FP8→BF16 cast path (see
[mimo_v2_flash_sglang_cpu.md](mimo_v2_flash_sglang_cpu.md)).

### Chat endpoint

```bash
curl -s http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "mimo-v2-flash",
       "messages": [{"role": "user", "content": "What is 1+1?"}],
       "max_tokens": 32}'
```

---

## Performance (measured on worker-0)

| Metric | Value |
|---|---|
| Prompt eval | **35.21 tok/s** (28.4 ms/tok, 11 tokens) |
| Generation | **17.17 tok/s** (58.2 ms/tok) |
| Model memory | ~313 GiB resident (mmap, faulted on access) |
| Startup to first token | ~5.5 min (NFS mmap cold load) |

---

## Stopping the Server

```bash
# Find the PID
ps aux | grep llama-server | grep -v grep | awk '{print $2}'

# Graceful stop
kill -TERM <pid>
```

---

## GGUF Model Metadata

Reported by llama.cpp at load time:

| Field | Value |
|---|---|
| Architecture | `mimo2` |
| Model type | `310B.A15B` |
| File type | `Q8_0` (8.50 BPW) |
| File size | 305.68 GiB |
| Training context | 262,144 tokens |
| Layers | 48 |
| Attention heads (Q) | 64 |
| Head dim K | 192 |
| Head dim V | 128 |
| SWA window | 128 tokens |
| Experts total | 256 |
| Experts per token | 8 |
| FFN size | 16,384 |
| Tensor types | f32 (230 tensors), q8_0 (338 tensors) |
