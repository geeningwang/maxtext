# MiMo-V2-Flash SGLang CPU Inference

This guide documents how to run MiMo-V2-Flash inference using
[SGLang](https://github.com/sgl-project/sglang) on a CPU-only host (no GPU,
no TPU).  The setup was validated on **worker-0** of the `jingnw-node` TPU VM
cluster: AMD EPYC 9B14, 180 vCPUs, 708 GB RAM, `avx512_bf16` present,
Intel AMX **not** present.

---

## Infrastructure Layout

| Node | Role | Key storage |
|---|---|---|
| worker-0 | `AMD EPYC 9B14`, 180 vCPUs, 708 GB RAM — SGLang host | NFS client |
| worker-1 (`10.202.0.151`) | HF safetensors weights in tmpfs | `/mnt/mimo-weights` 650 G tmpfs, NFS-exported read-only |
| worker-2 (`10.202.0.29`) | GGUF scratch in tmpfs; llama.cpp build | `/mnt/gguf-scratch` 650 G tmpfs, NFS-exported read-write |

### NFS mounts on worker-0

```
10.202.0.151:/mnt/mimo-weights  →  /mnt/mimo-weights   (ro)
10.202.0.29:/mnt/gguf-scratch   →  /mnt/gguf-scratch   (rw)
```

Weights at `/mnt/mimo-weights/` are read directly over NFS during SGLang
startup — no local copy needed.

---

## Source Data

| Artifact | Location | Size |
|---|---|---|
| HF safetensors weights | `/mnt/mimo-weights/` (NFS from worker-1) | 292 G |
| Q8_0 GGUF (llama.cpp) | `/mnt/gguf-scratch/mimo-v2-flash-Q8_0.gguf` (NFS from worker-2) | 306 G |

The GGUF was produced by llama.cpp `convert_hf_to_gguf.py` on worker-2:

```bash
# worker-2 — llama.cpp HEAD 9c69907, built with AVX-512
python3 ~/llama.cpp/convert_hf_to_gguf.py /mnt/mimo-weights \
  --outfile /mnt/gguf-scratch/mimo-v2-flash-Q8_0.gguf \
  --outtype q8_0
# Runtime: ~35 min @ 159 MB/s
```

---

## SGLang Installation (worker-0)

### 1. Clone SGLang

```bash
git clone --depth=1 https://github.com/sgl-project/sglang.git ~/sglang
# Pinned HEAD: 0f0f004
```

### 2. Install SGLang with CPU torch

```bash
cd ~/sglang
pip3 install -e "python/" \
  --extra-index-url https://download.pytorch.org/whl/cpu
# Installs sglang-0.0.0.dev1+g0f0f004f1, torch 2.9.1+cpu
```

### 3. Build sgl-kernel CPU extension

The CUDA wheel installed by default is unusable without a GPU.
Build the pure-C++ CPU kernel from source instead:

```bash
sudo apt-get install -y cmake libnuma-dev

cd ~/sglang/sgl-kernel
pip3 install -e . --no-build-isolation   # builds csrc/cpu only

# The .so lands in the pip cache, not the editable tree — copy it manually:
SO=$(find ~/.local/lib -name "common_ops.abi3.so" 2>/dev/null | head -1)
cp "$SO" ~/sglang/sgl-kernel/python/sgl_kernel/
```

### 4. Prepare model directory

`/mnt/mimo-weights` contains the safetensors weights but is missing the custom
architecture files (`configuration_mimo_v2_flash.py`,
`modeling_mimo_v2_flash.py`) that `trust_remote_code` needs.  SGLang's bundled
copies are used instead:

```bash
mkdir -p /tmp/sglang_model

# Symlink all NFS weight files
for f in /mnt/mimo-weights/*; do
  ln -sf "$f" "/tmp/sglang_model/$(basename $f)"
done

# Copy architecture .py files from SGLang's auto-downloaded cache
CFG_DIR=$(python3 -c "
import tempfile, transformers
from transformers import AutoConfig
with tempfile.TemporaryDirectory() as d:
    AutoConfig.from_pretrained('/mnt/mimo-weights', trust_remote_code=True, cache_dir=d)
    import glob, os
    print(glob.glob(d+'/**/configuration_mimo_v2_flash.py', recursive=True)[0].rsplit('/',1)[0])
")
cp "$CFG_DIR"/*.py /tmp/sglang_model/
```

---

## Required Patches

SGLang `0f0f004` requires five patches to run on CPU with MiMo-V2-Flash and
`quantization_config: null` (FP8 weights cast to BF16 at load time).

### Patch 1 — `token_dispatcher/__init__.py`: guard flashinfer import

**File**: `~/sglang/python/sglang/srt/layers/moe/token_dispatcher/__init__.py`

Wrap the `FlashinferDispatcher` import in a `try/except` so it degrades
gracefully when `libcudart` is absent:

```python
# Before
from sglang.srt.layers.moe.token_dispatcher.flashinfer import FlashinferDispatcher

# After
try:
    from sglang.srt.layers.moe.token_dispatcher.flashinfer import FlashinferDispatcher
except (ImportError, AssertionError):
    FlashinferDispatcher = None
```

### Patch 2 — `flashinfer/comm/cuda_ipc.py`: guard libcudart load

**File**: `~/.local/lib/python3.10/site-packages/flashinfer/comm/cuda_ipc.py`

```python
# Before
cudart = CudaRTLibrary()

# After
try:
    cudart = CudaRTLibrary()
except (AssertionError, OSError):
    cudart = None
```

### Patch 3 — `mimo_v2_flash.py` expert branch: skip missing scale keys

**File**: `~/sglang/python/sglang/srt/models/mimo_v2_flash.py`

In `load_weights`, inside the `expert_params_mapping` loop, add a guard before
`param = params_dict[name]`:

```python
                name = name.replace(weight_name, param_name)
                if name not in params_dict:
                    break  # scale/quant params absent when quantization_config=null
                param = params_dict[name]
```

### Patch 4 — `mimo_v2_flash.py` stacked branch: skip missing scale keys

Same file, same function, inside the `stacked_params_mapping` loop:

```python
                name = name.replace(weight_name, param_name)
                # Skip loading extra bias for GPTQ models.
                if name.endswith(".bias") and name not in params_dict:
                    continue
                # Skip FP8 scale params absent when quantization_config=null
                if name not in params_dict:
                    break
                param = params_dict[name]
```

### Patch 5 — `torch_native_backend.py`: accept `sinks` kwarg

**File**: `~/sglang/python/sglang/srt/layers/attention/torch_native_backend.py`

`MiMoV2FlashAttention` passes `sinks=self.attention_sink_bias` on every
forward call.  `TorchNativeAttnBackend` does not declare this parameter.
Add `sinks=None` to both methods (value is unused on CPU — it is only applied
in the FlashAttention path):

```python
# forward_extend
def forward_extend(self, q, k, v, layer, forward_batch, save_kv_cache=True, sinks=None):

# forward_decode
def forward_decode(self, q, k, v, layer, forward_batch, save_kv_cache=True, sinks=None):
```

---

## Launching the Server

```bash
# Required env var — without it is_cpu() returns False and vllm is imported
export SGLANG_USE_CPU_ENGINE=1

nohup env SGLANG_USE_CPU_ENGINE=1 PATH="$HOME/.local/bin:$PATH" \
  python3 -m sglang.launch_server \
  --model /tmp/sglang_model \
  --device cpu \
  --dtype bfloat16 \
  --trust-remote-code \
  --host 0.0.0.0 --port 30000 \
  --max-total-tokens 2048 \
  --context-length 1024 \
  --json-model-override-args '{"quantization_config": null}' \
  > /tmp/sglang_server.log 2>&1 &

echo "PID $!"
```

**Key flags:**

| Flag | Reason |
|---|---|
| `--device cpu` | Selects CPU execution path |
| `--dtype bfloat16` | Weights cast to BF16 (AMD EPYC has `avx512_bf16`) |
| `--json-model-override-args '{"quantization_config": null}'` | Disables FP8 quantization; AMD EPYC lacks Intel AMX required by `Fp8LinearMethod` |
| `SGLANG_USE_CPU_ENGINE=1` | Makes `is_cpu()` return `True` inside SGLang |

Weight loading takes ~3–4 minutes over NFS (292 GB @  ~1.5 shards/s).
The server is ready when the log contains:

```
INFO:     Application startup complete.
INFO:     Uvicorn running on http://0.0.0.0:30000
```

---

## Testing

```bash
# Completions endpoint (no chat template required)
curl -s http://localhost:30000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "/tmp/sglang_model",
       "prompt": "What is 1+1? The answer is ",
       "max_tokens": 10}'

# Chat endpoint (requires jinja2>=3.1.0)
pip3 install "jinja2>=3.1.0"
curl -s http://localhost:30000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "/tmp/sglang_model",
       "messages": [{"role": "user", "content": "What is 1+1?"}],
       "max_tokens": 18}'
```

---

## Stopping the Server

```bash
# Find all SGLang processes
ps aux | grep sglang | grep -v grep | awk '{print $2, $11}'

# Graceful stop (replace PIDs with actual values)
kill -TERM <main_pid> <scheduler_pid> <detokenizer_pid>
```

---

## Known Limitations

- **No GPU/TPU acceleration** — all compute is on CPU BF16; throughput is very
  low (suitable for correctness validation only).
- **`sinks` parameter** — MiMo's `attention_sink_bias` is silently dropped by
  the torch-native backend.  This means the attention sink bias is not applied,
  which may affect generation quality.
- **FP8 scale tensors skipped** — `weight_scale_inv` and similar per-tensor
  scales are absent when `quantization_config=null`; all expert and attention
  projection weights are loaded as raw BF16.
- **Observed output** — generation produces garbled Unicode (`葭葭葭…`), matching
  the HF CPU reference run.  This is believed to be a known model/tokenizer
  discrepancy, not an SGLang issue.
