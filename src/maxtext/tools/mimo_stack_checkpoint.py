"""Stack MiMo-V2-Flash flat-per-layer checkpoint into scan-ready layout.

Reads the existing per-layer OCDBT checkpoint (where each of the 48 decoder
layers has its own params at ``decoder.layers.{i}.*``) and writes a new OCDBT
checkpoint in the four-phase layout expected by ``scan_layers=True`` Round 2:

  Phase A  layer 0         → ``decoder.layers_a.*``           (single layer)
  Phase B  layers 1-4      → ``decoder.layers_b.*``           stacked (4, ...)
  Phase C  layers 5-46     → ``decoder.layers_c.layers_{p}.*``  stacked (7, ...)
                             for p in 0..5 (one per in-cycle position)
  Phase D  layer 47        → ``decoder.layers_d.*``           (single layer)

Must be run on **all 8 TPU workers simultaneously** with the same parallelism
flags used during inference (``ici_tensor_parallelism=4 ici_expert_parallelism=8``).

Usage example (run on all 8 workers):

  STACKED_PATH=gs://jingnw-mimo-v2-flash-us-east5/mimo-v2-flash-4phase-stacked/checkpoints/0/items
  gcloud compute tpus tpu-vm ssh jingnw-node --zone=us-east5-a --worker=all \\
    --command="
      source ~/maxtext/maxtext_tpu_venv/bin/activate && cd ~/maxtext &&
      STACKED_OUTPUT_PATH=$STACKED_PATH \\
      python3 -m maxtext.tools.mimo_stack_checkpoint \\
        src/maxtext/configs/base.yml model_name=mimo-v2-flash \\
        load_parameters_path=gs://jingnw-mimo-v2-flash-us-east5/mimo-v2-flash-fixed-ocdbt/checkpoints/0/items \\
        base_output_directory=gs://jingnw-mimo-v2-flash-us-east5/ \\
        run_name=mimo_stack_convert per_device_batch_size=1 \\
        max_target_length=768 max_prefill_predict_length=512 \\
        attention=dot_product scan_layers=false weight_dtype=bfloat16 \\
        ici_tensor_parallelism=4 ici_expert_parallelism=8 async_checkpointing=false \\
        > /tmp/mimo_stack.log 2>&1"

Set the ``STACKED_OUTPUT_PATH`` environment variable to override the output GCS
path (defaults to the path shown in the usage example above).

After the tool completes, verify a couple of tensor shapes:

  python3 -c "
  import orbax.checkpoint as ocp, epath
  ckpt = ocp.PyTreeCheckpointer()
  meta = ckpt.metadata(epath.Path('$STACKED_PATH').parent.parent)
  # look for decoder.layers_b.* entries (shape should start with (4,...))
  for k in sorted(meta.keys())[:10]: print(k)
  "
"""

import os
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
    "gs://jingnw-mimo-v2-flash-us-east5/mimo-v2-flash-4phase-stacked/checkpoints/0/items",
)


# ---------------------------------------------------------------------------
# MiMo layer groupings — must match the constants in mimo_v2_flash.py and the
# decoders.py Round-2 wiring.
#
#   Phase A : layer 0          (unique: global attn + dense MLP)
#   Phase B : layers 1-4       (4× SWA-MoE, all same structure)
#   Phase C : layers 5-46      (7 repetitions of a 6-layer cycle)
#             cycle pos 0 → global-attn MoE  layers  5,11,17,23,29,35,41
#             cycle pos 1 → SWA-MoE          layers  6,12,18,24,30,36,42
#             cycle pos 2 → SWA-MoE          layers  7,13,19,25,31,37,43
#             cycle pos 3 → SWA-MoE          layers  8,14,20,26,32,38,44
#             cycle pos 4 → SWA-MoE          layers  9,15,21,27,33,39,45
#             cycle pos 5 → SWA-MoE          layers 10,16,22,28,34,40,46
#   Phase D : layer 47         (unique: global attn + MoE)
# ---------------------------------------------------------------------------
_PHASE_B_INDICES = list(range(1, 5))  # [1, 2, 3, 4]

_PHASE_C_POSITIONS = [
    [5,  11, 17, 23, 29, 35, 41],   # pos 0 : global-attn MoE
    [6,  12, 18, 24, 30, 36, 42],   # pos 1 : SWA-MoE
    [7,  13, 19, 25, 31, 37, 43],   # pos 2
    [8,  14, 20, 26, 32, 38, 44],   # pos 3
    [9,  15, 21, 27, 33, 39, 45],   # pos 4
    [10, 16, 22, 28, 34, 40, 46],   # pos 5 : SWA-MoE
]


def _rearrange_params(raw_params: dict) -> dict:
  """Rearrange flat per-layer params into the four-phase scan-ready layout.

  Args:
    raw_params: pytree returned by ``engine.load_params(rng)`` when the model
      is configured with ``scan_layers=false``.  Structure::

        {"params": {"decoder": {"layers": {"0": ..., ..., "47": ...},
                                "decoder_norm": ..., ...},
                    "shared_embedding": ...}}

  Returns:
    A new pytree with the same dtype/sharding but the decoder layers renamed
    and stacked::

        {"params": {"decoder": {"layers_a": ...,
                                "layers_b": ...,   # shape (4, ...)
                                "layers_c": {"layers_0": ...,  # shape (7, ...)
                                             ...
                                             "layers_5": ...}, # shape (7, ...)
                                "layers_d": ...,
                                "decoder_norm": ..., ...},
                    "shared_embedding": ...}}

  Memory note: source layer pytrees are popped from the working dict and freed
    (``del``) immediately after stacking so that peak HBM usage stays bounded
    to approximately (initial flat load + one stacked phase) ≈ 21 GiB/chip.
  """
  # Shallow-copy the top-level dicts so we can mutate freely.
  outer = dict(raw_params)
  inner_params = dict(outer.pop("params"))
  decoder = dict(inner_params.pop("decoder"))
  # Pop "layers" from decoder; the rest (decoder_norm, logits_dense, …) are
  # kept unchanged via the final dict merge.
  flat_layers = dict(decoder.pop("layers"))

  # ---- Phase A ----
  max_logging.log("Stack tool: Phase A — layer 0 (unique; no stacking)")
  layers_a = flat_layers.pop("0")

  # ---- Phase B ----
  max_logging.log(f"Stack tool: Phase B — stacking layers {_PHASE_B_INDICES} → (4, ...)")
  phase_b_list = [flat_layers.pop(str(i)) for i in _PHASE_B_INDICES]
  layers_b = jax.tree_util.tree_map(
      lambda *xs: jnp.stack(xs, axis=0),
      *phase_b_list,
  )
  del phase_b_list  # allow HBM reclaim

  # ---- Phase C ----
  layers_c: dict = {}
  for pos, indices in enumerate(_PHASE_C_POSITIONS):
    max_logging.log(f"Stack tool: Phase C pos {pos} — stacking layers {indices} → (7, ...)")
    pos_layers = [flat_layers.pop(str(i)) for i in indices]
    stacked = jax.tree_util.tree_map(
        lambda *xs: jnp.stack(xs, axis=0),
        *pos_layers,
    )
    del pos_layers  # allow HBM reclaim
    layers_c[f"layers_{pos}"] = stacked

  # ---- Phase D ----
  max_logging.log("Stack tool: Phase D — layer 47 (unique; no stacking)")
  layers_d = flat_layers.pop("47")

  # Sanity check — all 48 source layers consumed.
  if flat_layers:
    raise RuntimeError(
        "Stack tool: unexpected leftover layer keys after rearrangement: "
        f"{sorted(flat_layers.keys())}"
    )

  decoder["layers_a"] = layers_a
  decoder["layers_b"] = layers_b
  decoder["layers_c"] = layers_c
  decoder["layers_d"] = layers_d

  inner_params["decoder"] = decoder
  outer["params"] = inner_params
  return outer


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

  # 1. Load flat per-layer params (same path as a normal scan_layers=false run).
  max_logging.log(f"Stack tool: loading flat params from {config.load_parameters_path} ...")
  engine = maxengine.MaxEngine(config)
  rng = jax.random.PRNGKey(0)
  params = engine.load_params(rng)
  max_logging.log("Stack tool: flat params loaded successfully.")

  # 2. Rearrange: stack Phase B and C; rename Phase A and D.
  max_logging.log("Stack tool: rearranging params into four-phase scan layout ...")
  stacked = _rearrange_params(params)
  jax.effects_barrier()  # ensure all stacking XLA ops have completed
  max_logging.log("Stack tool: rearrangement complete.")

  # 3. Save stacked checkpoint to new OCDBT+zarr3 path.
  max_logging.log(f"Stack tool: saving stacked checkpoint to {_STACKED_OUTPUT_PATH} ...")
  save_ckptr = ocp.Checkpointer(
      ocp.PyTreeCheckpointHandler(
          use_ocdbt=True,
          use_zarr3=True,
          save_concurrent_gb=96,
      )
  )
  # Wrap in {"params": ...} to match the layout expected by load_params_from_path
  # (which does ckptr.restore(..., item={"params": abstract}) and returns ["params"]).
  save_ckptr.save(_STACKED_OUTPUT_PATH, args=ocp.args.PyTreeSave({"params": stacked}))
  jax.effects_barrier()
  max_logging.log(f"Stack tool: done. Stacked checkpoint at: {_STACKED_OUTPUT_PATH}")


if __name__ == "__main__":
  app.run(main)
