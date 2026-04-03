"""Convert a zarr2/no-OCDBT checkpoint to zarr3+OCDBT format for faster inference loading.

Run on all 8 TPU workers simultaneously with the same parallelism as inference:

  gcloud compute tpus tpu-vm ssh jingnw-node --worker=all --zone=us-east5-b --internal-ip \\
    --command="cd ~/maxtext && nohup ~/maxtext/maxtext_venv/bin/python3 -u \\
      -m maxtext.tools.convert_checkpoint_to_ocdbt \\
      src/maxtext/configs/base.yml \\
      model_name=mimo-v2-flash \\
      run_name=mimo_ocdbt_convert \\
      load_parameters_path=gs://jingnw-mimo-v2-flash-us-east5/mimo-v2-flash/checkpoints/0/items \\
      base_output_directory=gs://jingnw-mimo-v2-flash-us-east5/ \\
      checkpoint_storage_use_ocdbt=false \\
      checkpoint_storage_use_zarr3=false \\
      ici_tensor_parallelism=4 \\
      ici_expert_parallelism=8 \\
      scan_layers=false \\
      per_device_batch_size=1 \\
      max_target_length=384 \\
      attention=dot_product \\
      dtype=bfloat16 \\
      weight_dtype=bfloat16 \\
      > /tmp/convert.log 2>&1 &"

After conversion, inference can use:
  load_parameters_path=gs://jingnw-mimo-v2-flash-us-east5/mimo-v2-flash-ocdbt/checkpoints/0/items
  checkpoint_storage_use_ocdbt=true
  checkpoint_storage_use_zarr3=true
"""

import os
from typing import Sequence

import jax
import orbax.checkpoint as ocp
from absl import app

from maxtext.configs import pyconfig
from maxtext.inference.maxengine import maxengine
from maxtext.utils import max_utils
from maxtext.utils import max_logging


# Output GCS path for the converted OCDBT checkpoint
_OCDBT_OUTPUT_PATH = "gs://jingnw-mimo-v2-flash-us-east5/mimo-v2-flash-ocdbt/checkpoints/0/items"


def main(argv: Sequence[str]) -> None:
  jax.config.update("jax_default_prng_impl", "unsafe_rbg")
  os.environ["TF_CPP_MIN_LOG_LEVEL"] = "0"

  config = pyconfig.initialize(argv)
  max_utils.print_system_information()

  # 1. Load params using MaxEngine (same path as inference — tested and working)
  max_logging.log(f"Loading params from {config.load_parameters_path} (zarr2, no-OCDBT) ...")
  engine = maxengine.MaxEngine(config)
  rng = jax.random.PRNGKey(0)
  params = engine.load_params(rng)
  max_logging.log("Params loaded successfully.")

  # 2. Save to new OCDBT+zarr3 path. Only process 0 (w-0) needs to save;
  #    but all processes must participate in the collective save for sharded arrays.
  max_logging.log(f"Saving OCDBT+zarr3 checkpoint to {_OCDBT_OUTPUT_PATH} ...")
  save_ckptr = ocp.Checkpointer(
      ocp.PyTreeCheckpointHandler(
          use_ocdbt=True,
          use_zarr3=True,
          save_concurrent_gb=96,
      )
  )
  save_ckptr.save(_OCDBT_OUTPUT_PATH, args=ocp.args.PyTreeSave(params))
  jax.effects_barrier()
  max_logging.log(f"Conversion complete. New checkpoint at: {_OCDBT_OUTPUT_PATH}")


if __name__ == "__main__":
  app.run(main)
