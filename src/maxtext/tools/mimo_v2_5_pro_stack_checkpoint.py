"""Stack MiMo-V2.5-Pro flat-per-layer checkpoint into scan-ready layout.

Reads the existing per-layer OCDBT checkpoint (where each of the 70 decoder
layers has its own params at ``decoder.layers.{i}.*``) and writes a new OCDBT
checkpoint in the eight-phase layout expected by ``scan_layers=True``:

  Phase A  layer 0          → ``decoder.layers_a.*``              (single)
  Phase B  layers 1–6       → ``decoder.layers_b.*``              stacked (6, ...)
  Phase C  layers 7–54      → ``decoder.layers_c.layers_{p}.*``   stacked (6, ...)
                               for p in 0..7 (one per cycle position)
  Phase D  layer 55         → ``decoder.layers_d.*``              (single)
  Phase E  layers 56–61     → ``decoder.layers_e.*``              stacked (6, ...)
  Phase F  layer 62         → ``decoder.layers_f.*``              (single)
  Phase G  layers 63–68     → ``decoder.layers_g.*``              stacked (6, ...)
  Phase H  layer 69         → ``decoder.layers_h.*``              (single)

Phase C covers the regular period-8 cycle (1 GA + 7 SWA) that spans layers 7–54.
Each of the 8 cycle positions is homogeneous across all 6 repetitions:

  layers_c.layers_0  GA+MoE  layers [7, 15, 23, 31, 39, 47]
  layers_c.layers_1  SWA+MoE layers [8, 16, 24, 32, 40, 48]
  layers_c.layers_2  SWA+MoE layers [9, 17, 25, 33, 41, 49]
  layers_c.layers_3  SWA+MoE layers [10,18, 26, 34, 42, 50]
  layers_c.layers_4  SWA+MoE layers [11,19, 27, 35, 43, 51]
  layers_c.layers_5  SWA+MoE layers [12,20, 28, 36, 44, 52]
  layers_c.layers_6  SWA+MoE layers [13,21, 29, 37, 45, 53]
  layers_c.layers_7  SWA+MoE layers [14,22, 30, 38, 46, 54]

Must be run on **all TPU workers simultaneously** with the same parallelism
flags used during inference (``ici_tensor_parallelism=8 ici_expert_parallelism=2``).

Usage (run on all workers via the mimo_v2_5_pro_stack_job.yaml GKE job, or
manually on TPU VMs):

  STACKED_PATH=gs://jingnw-mimo-v2-5-pro-us-central1/mimo-v2-5-pro-fp8-ocdbt-stacked/0/items
  python3 -m maxtext.tools.mimo_v2_5_pro_stack_checkpoint \\
    src/maxtext/configs/base.yml model_name=mimo-v2-5-pro \\
    load_parameters_path=gs://jingnw-mimo-v2-5-pro-us-central1/mimo-v2-5-pro-fp8-ocdbt/0/items \\
    base_output_directory=gs://jingnw-mimo-v2-5-pro-us-central1/ \\
    run_name=mimo_v25pro_stack per_device_batch_size=1 \\
    max_target_length=8192 max_prefill_predict_length=8192 \\
    attention=dot_product scan_layers=false \\
    ici_tensor_parallelism=8 ici_expert_parallelism=2 \\
    async_checkpointing=false

Set the ``STACKED_OUTPUT_PATH`` environment variable to override the output GCS
path (defaults to the path shown in the docstring above).

After the tool completes, verify a couple of tensor shapes:

  python3 -c "
  import orbax.checkpoint as ocp
  from etils import epath
  ckpt = ocp.PyTreeCheckpointer()
  meta = ckpt.metadata(epath.Path('STACKED_PATH').parent.parent)
  for k in sorted(meta.keys())[:20]: print(k)
  "
"""

import os
import socket
import threading
import time
from typing import Sequence

import jax
import jax.numpy as jnp
import orbax.checkpoint as ocp
from absl import app

from maxtext.configs import pyconfig
from maxtext.inference.maxengine import maxengine
from maxtext.utils import max_logging
from maxtext.utils import max_utils


# ---------------------------------------------------------------------------
# Output path — override via STACKED_OUTPUT_PATH env var.
# ---------------------------------------------------------------------------
_STACKED_OUTPUT_PATH = os.environ.get(
    "STACKED_OUTPUT_PATH",
    "gs://jingnw-mimo-v2-5-pro-us-central1/mimo-v2-5-pro-fp8-ocdbt-stacked/0/items",
)


# ---------------------------------------------------------------------------
# MiMo-V2.5-Pro layer groupings — must match the phase layout in decoders.py.
#
#   Phase A : layer 0               (GA + dense MLP, unique)
#   Phase B : layers 1–6            (SWA + MoE, 6 identical)
#   Phase C : layers 7–54           period-8 cycle (1 GA + 7 SWA) × 6 reps
#             cycle pos 0 → GA+MoE   layers  7,15,23,31,39,47
#             cycle pos 1 → SWA+MoE  layers  8,16,24,32,40,48
#             cycle pos 2 → SWA+MoE  layers  9,17,25,33,41,49
#             cycle pos 3 → SWA+MoE  layers 10,18,26,34,42,50
#             cycle pos 4 → SWA+MoE  layers 11,19,27,35,43,51
#             cycle pos 5 → SWA+MoE  layers 12,20,28,36,44,52
#             cycle pos 6 → SWA+MoE  layers 13,21,29,37,45,53
#             cycle pos 7 → SWA+MoE  layers 14,22,30,38,46,54
#   Phase D : layer 55              (GA + MoE, single)
#   Phase E : layers 56–61          (SWA + MoE, 6 identical)
#   Phase F : layer 62              (GA + MoE, single)
#   Phase G : layers 63–68          (SWA + MoE, 6 identical)
#   Phase H : layer 69              (GA + MoE, single)
# ---------------------------------------------------------------------------
_PHASE_B_INDICES = list(range(1, 7))   # [1, 2, 3, 4, 5, 6]

_PHASE_C_POSITIONS = [
    [7,  15, 23, 31, 39, 47],   # pos 0 : GA+MoE
    [8,  16, 24, 32, 40, 48],   # pos 1 : SWA+MoE
    [9,  17, 25, 33, 41, 49],   # pos 2 : SWA+MoE
    [10, 18, 26, 34, 42, 50],   # pos 3 : SWA+MoE
    [11, 19, 27, 35, 43, 51],   # pos 4 : SWA+MoE
    [12, 20, 28, 36, 44, 52],   # pos 5 : SWA+MoE
    [13, 21, 29, 37, 45, 53],   # pos 6 : SWA+MoE
    [14, 22, 30, 38, 46, 54],   # pos 7 : SWA+MoE
]

_PHASE_E_INDICES = list(range(56, 62))  # [56, 57, 58, 59, 60, 61]
_PHASE_G_INDICES = list(range(63, 69))  # [63, 64, 65, 66, 67, 68]


def _probe_hbm(label: str) -> None:
  """Print per-device HBM usage."""
  host = socket.gethostname()
  for d in jax.local_devices():
    try:
      s = d.memory_stats()
      used = s.get("bytes_in_use", 0) / 2**30
      limit = s.get("bytes_limit", 0) / 2**30
      print(f"[HBM] {label:<55s} host={host} dev={d.id} used={used:.2f}GB limit={limit:.2f}GB",
            flush=True)
    except Exception as e:  # pylint: disable=broad-except
      print(f"[HBM] {label:<55s} N/A: {e}", flush=True)


def _start_load_monitor(interval_s: int = 30):
  """Background thread: log HBM fill every interval_s seconds during load_params.

  Returns a threading.Event; call .set() to stop the monitor after load_params
  returns.  HBM usage rising over time confirms that GCS→HBM streaming is
  making progress.
  """
  stop = threading.Event()

  def _loop():
    elapsed = 0
    while not stop.wait(interval_s):
      elapsed += interval_s
      total_used = sum(
          d.memory_stats().get("bytes_in_use", 0)
          for d in jax.local_devices()
          if hasattr(d, "memory_stats")
      ) / 2**30
      total_limit = sum(
          d.memory_stats().get("bytes_limit", 0)
          for d in jax.local_devices()
          if hasattr(d, "memory_stats")
      ) / 2**30
      max_logging.log(
          f"[load_params progress t+{elapsed}s] "
          f"host={socket.gethostname()} "
          f"HBM used={total_used:.1f}GB / {total_limit:.1f}GB "
          f"({100*total_used/max(total_limit,1):.1f}%)"
      )

  t = threading.Thread(target=_loop, daemon=True)
  t.start()
  return stop


def _stack_donated(*pytrees):
  """Stack pytrees along axis 0 with donate_argnums — net HBM change ≈ 0.

  Marks all input pytrees as donated, so XLA reuses their device buffers for
  the stacked output.  Works with multihost sharded jax.Arrays because
  donation operates per-device (no cross-host transfers).

  IMPORTANT: For donation to actually free the source HBM buffers, each leaf
  jax.Array in ``pytrees`` must have refcount == 1 at call time (i.e. no other
  Python name or container references the same array).  The caller is
  responsible for ensuring this by deleting the original params pytree before
  calling this function.
  """
  n = len(pytrees)
  return jax.jit(
      lambda *args: jax.tree_util.tree_map(lambda *xs: jnp.stack(xs, axis=0), *args),
      donate_argnums=tuple(range(n)),
  )(*pytrees)


def _rearrange_layers(flat_layers: dict, decoder_stub: dict,
                      inner_stub: dict, outer_stub: dict) -> dict:
  """Stack flat per-layer params into eight-phase layout using donated stacking.

  ``flat_layers`` must be the ONLY Python container referencing the layer leaf
  jax.Arrays (i.e. the caller must have already deleted the original params
  pytree so that each leaf has refcount == 1).  This allows ``_stack_donated``
  to actually reuse the HBM buffers rather than silently copying them.

  Args:
    flat_layers:  mutable dict mapping str(i) → per-layer param pytree.
    decoder_stub: decoder params except the "layers" key (embed, norm, …).
    inner_stub:   inner params except the "decoder" key.
    outer_stub:   outer params except the "params" key.

  Returns:
    Reconstructed params pytree with eight-phase layout.
  """
  _probe_hbm("before rearrange")

  # ---- Phase A: layer 0 (single, no stacking) ----
  max_logging.log("Stack tool: Phase A — layer 0 (GA+dense, single)")
  layers_a = flat_layers.pop("0")

  # ---- Phase B: layers 1–6 (scan=6) ----
  max_logging.log(f"Stack tool: Phase B — stacking layers {_PHASE_B_INDICES} → (6, ...)")
  _probe_hbm("before Phase B stack")
  phase_b_list = [flat_layers.pop(str(i)) for i in _PHASE_B_INDICES]
  layers_b = _stack_donated(*phase_b_list)
  del phase_b_list
  _probe_hbm("after Phase B stack (inputs donated)")

  # ---- Phase C: 8 cycle positions × 6 reps each ----
  layers_c: dict = {}
  for pos, indices in enumerate(_PHASE_C_POSITIONS):
    max_logging.log(
        f"Stack tool: Phase C pos {pos} — stacking layers {indices} → (6, ...)")
    _probe_hbm(f"before Phase C pos {pos}")
    pos_layers = [flat_layers.pop(str(i)) for i in indices]
    stacked = _stack_donated(*pos_layers)
    del pos_layers
    layers_c[f"layers_{pos}"] = stacked
    _probe_hbm(f"after Phase C pos {pos} (inputs donated)")

  # ---- Phase D: layer 55 (single, no stacking) ----
  max_logging.log("Stack tool: Phase D — layer 55 (GA+MoE, single)")
  layers_d = flat_layers.pop("55")

  # ---- Phase E: layers 56–61 (scan=6) ----
  max_logging.log(f"Stack tool: Phase E — stacking layers {_PHASE_E_INDICES} → (6, ...)")
  _probe_hbm("before Phase E stack")
  phase_e_list = [flat_layers.pop(str(i)) for i in _PHASE_E_INDICES]
  layers_e = _stack_donated(*phase_e_list)
  del phase_e_list
  _probe_hbm("after Phase E stack (inputs donated)")

  # ---- Phase F: layer 62 (single, no stacking) ----
  max_logging.log("Stack tool: Phase F — layer 62 (GA+MoE, single)")
  layers_f = flat_layers.pop("62")

  # ---- Phase G: layers 63–68 (scan=6) ----
  max_logging.log(f"Stack tool: Phase G — stacking layers {_PHASE_G_INDICES} → (6, ...)")
  _probe_hbm("before Phase G stack")
  phase_g_list = [flat_layers.pop(str(i)) for i in _PHASE_G_INDICES]
  layers_g = _stack_donated(*phase_g_list)
  del phase_g_list
  _probe_hbm("after Phase G stack (inputs donated)")

  # ---- Phase H: layer 69 (single, no stacking) ----
  max_logging.log("Stack tool: Phase H — layer 69 (GA+MoE, single)")
  layers_h = flat_layers.pop("69")

  # Sanity check — all 70 source layers consumed.
  if flat_layers:
    raise RuntimeError(
        "Stack tool: unexpected leftover layer keys after rearrangement: "
        f"{sorted(flat_layers.keys())}"
    )

  decoder_stub["layers_a"] = layers_a
  decoder_stub["layers_b"] = layers_b
  decoder_stub["layers_c"] = layers_c
  decoder_stub["layers_d"] = layers_d
  decoder_stub["layers_e"] = layers_e
  decoder_stub["layers_f"] = layers_f
  decoder_stub["layers_g"] = layers_g
  decoder_stub["layers_h"] = layers_h

  inner_stub["decoder"] = decoder_stub
  outer_stub["params"] = inner_stub
  _probe_hbm("after full rearrangement")
  return outer_stub


def main(argv: Sequence[str]) -> None:
  jax.config.update("jax_default_prng_impl", "unsafe_rbg")
  os.environ["TF_CPP_MIN_LOG_LEVEL"] = "0"

  config = pyconfig.initialize(argv)
  max_utils.print_system_information()

  if config.scan_layers:
    raise ValueError(
        "Run with scan_layers=false so that engine.load_params loads the flat "
        "per-layer checkpoint. This tool produces the scan-compatible checkpoint."
    )

  if config.base_num_decoder_layers != 70:
    raise ValueError(
        f"This tool is for MiMo-V2.5-Pro (70 layers). "
        f"Got base_num_decoder_layers={config.base_num_decoder_layers}. "
        "For V2-Flash (48 layers) use mimo_stack_checkpoint.py instead."
    )

  # 1. Load flat per-layer params (same path as a normal scan_layers=false run).
  # A background monitor thread logs HBM fill every 30s so we can track GCS→HBM
  # streaming progress during the otherwise-silent async Orbax restore phase.
  max_logging.log(
      f"Stack tool: loading flat params from {config.load_parameters_path} ...")
  engine = maxengine.MaxEngine(config)
  rng = jax.random.PRNGKey(0)
  _monitor_stop = _start_load_monitor(interval_s=30)
  t0 = time.time()
  params = engine.load_params(rng)
  _monitor_stop.set()
  max_logging.log(
      f"Stack tool: flat params loaded in {time.time()-t0:.1f}s.")

  # 2. Rearrange: stack Phase B, C, E, G; copy Phase A, D, F, H (singles).
  #
  # CRITICAL for donation to work: we must drop ALL references to the original
  # params pytree before calling _rearrange_layers, so that each leaf jax.Array
  # has Python refcount == 1 (only our flat_layers dict holds it).  If `params`
  # (or any alias) is still alive when we call _stack_donated, JAX silently
  # copies instead of donating, causing HBM to double and OOM.
  #
  # Strategy:
  #   1. Shallow-copy each level of the dict tree so we can mutate/reassemble.
  #   2. `del params` in THIS scope to drop the only remaining extra ref chain.
  #   3. CPython's ref-counter immediately frees the original nested dicts,
  #      leaving each leaf jax.Array with refcount == 1 in flat_layers.
  max_logging.log(
      "Stack tool: extracting flat layers for donation-based stacking ...")
  outer_stub = dict(params)
  inner_stub = dict(outer_stub.pop("params"))
  decoder_stub = dict(inner_stub.pop("decoder"))
  flat_layers = dict(decoder_stub.pop("layers"))
  del params  # drop original pytree; leaf arrays now exclusively in flat_layers
  _probe_hbm("after load_params + pre-extract (before donation stacking)")

  max_logging.log("Stack tool: rearranging params into eight-phase scan layout ...")
  stacked = _rearrange_layers(flat_layers, decoder_stub, inner_stub, outer_stub)
  del flat_layers
  max_logging.log("Stack tool: rearrangement complete.")

  # 3. Save stacked checkpoint to new OCDBT+zarr3 path.
  max_logging.log(
      f"Stack tool: saving stacked checkpoint to {_STACKED_OUTPUT_PATH} ...")
  save_ckptr = ocp.Checkpointer(
      ocp.PyTreeCheckpointHandler(
          use_ocdbt=True,
          use_zarr3=True,
          save_concurrent_gb=96,
      )
  )
  save_ckptr.save(
      _STACKED_OUTPUT_PATH,
      args=ocp.args.PyTreeSave({"params": stacked}))
  jax.effects_barrier()
  max_logging.log(
      f"Stack tool: done. Stacked checkpoint at: {_STACKED_OUTPUT_PATH}")


if __name__ == "__main__":
  app.run(main)
