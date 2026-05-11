"""
Copyright 2026 Google LLC
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
     https://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

r"""Offline PTQ quantization: convert BF16 checkpoint → qwix PtqProvider format.

This script loads a BF16 MaxText checkpoint, quantizes the attention-layer
weights to FP8 using qwix's ``quantize_params`` API, and saves the resulting
PTQ-format checkpoint.  The saved checkpoint can be loaded directly by
MaxText with ``PtqProvider`` (``use_qwix_quantization=true`` +
``checkpoint_is_quantized=true``) without requiring any additional runtime
quantization on the TPU.

The script is designed to run on a **CPU-only high-memory machine** (e.g. an
n2-highmem-16 with 128 GiB RAM).  No GPU or TPU is required.

How it works
------------
1. Creates two MaxText models sharing the same architecture:
   - ``bf16_model``: standard model (no quantization), used only to get the
     abstract state structure for BF16 checkpoint loading.
   - ``ptq_model``: PtqProvider-wrapped model.  Its abstract state encodes
     ``WithAux[QArray]`` at every quantized weight position, which tells
     ``qwix.quantize_params`` how to quantize each weight.

2. Loads BF16 weights from ``load_parameters_path`` using the BF16 abstract
   state as the Orbax template (plain JAX arrays, no quantization).

3. Runs ``qwix.quantize_params(bf16_weights, abstract_ptq_params)`` to
   quantize every weight that ``PtqProvider`` intercepts (currently all
   ``dot_general`` operations in the decoder attention blocks).

   MoE expert weights are NOT quantized because ``PtqProvider`` does not
   currently intercept ``gmm``/``ragged_dot``.  They remain in BF16 in the
   saved checkpoint.

4. Saves the quantized params to ``save_quantized_params_path`` using Orbax.
   The saved zarr2 layout for a quantized weight ``kernel`` is:
     kernel/qvalue/  → float8_e4m3fn arrays
     kernel/scale/   → float32 arrays (per-channel scale)

Usage (on CPU highmem node)
---------------------------
  python3 -m maxtext.checkpoint_conversion.standalone_scripts.quantize_params_ptq \
      src/maxtext/configs/base.yml \
      src/maxtext/configs/models/mimo-v2-flash.yml \
      run_name=mimo_ptq_quantize \
      load_parameters_path=gs://jingnw-mimo-v2-flash-us-central1/mimo-v2-flash-fixed-ocdbt/checkpoints/0/items \
      save_quantized_params_path=gs://jingnw-mimo-v2-flash-us-central1/mimo-v2-flash-fp8-ptq/0/items \
      use_qwix_quantization=true \
      quantization=fp8_full \
      checkpoint_is_quantized=false \
      ici_tensor_parallelism=1 \
      ici_expert_parallelism=1 \
      max_target_length=128 \
      per_device_batch_size=1 \
      attention=dot_product \
      enable_single_controller=true \
      checkpoint_storage_use_ocdbt=true \
      checkpoint_storage_use_zarr3=true \
      async_checkpointing=false

After this script succeeds, run inference with:
  use_qwix_quantization=true
  quantization=fp8_full
  checkpoint_is_quantized=true
  load_parameters_path=<save_quantized_params_path>
"""

import os
from typing import Sequence

from absl import app
import jax
import jax.numpy as jnp
import qwix

from maxtext.common.common_types import MODEL_MODE_PREFILL
from maxtext.configs import pyconfig
from maxtext.common import checkpointing
from maxtext.layers import quantizations
from maxtext.utils import max_logging
from maxtext.utils import max_utils
from maxtext.utils import maxtext_utils
from maxtext.utils import model_creation_utils


def main(argv: Sequence[str]) -> None:
  jax.config.update("jax_default_prng_impl", "unsafe_rbg")
  os.environ["TF_CPP_MIN_LOG_LEVEL"] = "0"

  config = pyconfig.initialize(argv)
  _validate_config(config)

  max_utils.print_system_information()
  max_logging.log(f"Starting offline PTQ quantization.")
  max_logging.log(f"  BF16 checkpoint:  {config.load_parameters_path}")
  max_logging.log(f"  PTQ output path:  {config.save_quantized_params_path}")

  rng = jax.random.PRNGKey(0)
  rng, rng_bf16, rng_ptq = jax.random.split(rng, 3)

  devices_array = maxtext_utils.create_device_mesh(config=config)
  mesh = jax.sharding.Mesh(devices_array, config.mesh_axes)

  # ------------------------------------------------------------------
  # Step 1: Build two models — BF16 (for loading) and PTQ (for shapes).
  # ------------------------------------------------------------------
  max_logging.log("Building BF16 model (for checkpoint loading)...")
  quant = quantizations.configure_quantization(config)
  bf16_model = model_creation_utils.get_transformer_model(
      config, mesh, quant, model_mode=MODEL_MODE_PREFILL
  )
  # No qwix wrapping for bf16_model — just vanilla Linen.

  max_logging.log("Building PTQ model (for abstract parameter shapes)...")
  ptq_model = model_creation_utils.get_transformer_model(
      config, mesh, quant, model_mode=MODEL_MODE_PREFILL
  )
  ptq_model = quantizations.maybe_quantize_model(ptq_model, config, ptq=True)

  # ------------------------------------------------------------------
  # Step 2: Load BF16 checkpoint.
  # ------------------------------------------------------------------
  max_logging.log("Loading BF16 checkpoint...")
  bf16_state, _ = maxtext_utils.setup_decode_state(bf16_model, config, rng_bf16, mesh, None)
  # state.params = {'params': {<model params as jax.Arrays>}}
  bf16_weights = bf16_state.params["params"]
  max_logging.log(f"BF16 checkpoint loaded. Top-level keys: {list(bf16_weights.keys())}")

  # ------------------------------------------------------------------
  # Step 3: Get abstract PTQ params (WithAux[QArray] at quantized positions).
  # ------------------------------------------------------------------
  max_logging.log("Computing abstract PTQ parameter shapes via eval_shape...")
  abstract_ptq_state, _, _ = maxtext_utils.get_abstract_state(
      ptq_model, None, config, rng_ptq, mesh, is_training=False
  )
  abstract_ptq_weights = abstract_ptq_state.params["params"]
  max_logging.log("Abstract PTQ shapes computed.")

  # Count how many weights will be quantized.
  flat_abstract = jax.tree_util.tree_leaves_with_path(abstract_ptq_weights)
  num_quantized = sum(
      1 for _, v in jax.tree_util.tree_leaves_with_path(abstract_ptq_weights)
      if _is_with_aux(v)
  )
  max_logging.log(f"Weights to be FP8-quantized: {num_quantized}")

  # ------------------------------------------------------------------
  # Step 4: Quantize weights using qwix.quantize_params.
  # ------------------------------------------------------------------
  max_logging.log("Running qwix.quantize_params (this may take a few minutes on CPU)...")
  quantized_weights = qwix.quantize_params(
      bf16_weights,
      abstract_ptq_weights,
      allow_extra_params=False,
  )
  max_logging.log("Weight quantization complete.")

  # ------------------------------------------------------------------
  # Step 5: Save PTQ checkpoint.
  # ------------------------------------------------------------------
  quantized_params = {"params": quantized_weights}
  max_logging.log(f"Saving PTQ checkpoint to: {config.save_quantized_params_path}")
  checkpointing.save_params_to_path(
      config.save_quantized_params_path,
      quantized_params,
      use_ocdbt=config.checkpoint_storage_use_ocdbt,
      use_zarr3=config.checkpoint_storage_use_zarr3,
  )
  max_logging.log("PTQ checkpoint saved successfully!")
  max_logging.log("")
  max_logging.log("To run inference with this checkpoint, use:")
  max_logging.log(f"  load_parameters_path={config.save_quantized_params_path}")
  max_logging.log("  use_qwix_quantization=true")
  max_logging.log("  quantization=fp8_full")
  max_logging.log("  checkpoint_is_quantized=true")


def _validate_config(config):
  """Validate that the config has the required fields for PTQ quantization."""
  assert config.use_qwix_quantization, (
      "use_qwix_quantization must be true for PTQ quantization."
  )
  assert config.quantization == "fp8_full", (
      f"Only fp8_full quantization is supported for PTQ. Got: {config.quantization}"
  )
  assert config.load_parameters_path, (
      "load_parameters_path must be set (path to BF16 checkpoint)."
  )
  assert config.save_quantized_params_path, (
      "save_quantized_params_path must be set (path for PTQ checkpoint output)."
  )
  assert not config.checkpoint_is_quantized, (
      "checkpoint_is_quantized must be false (we are converting FROM a BF16 checkpoint)."
  )


def _is_with_aux(x):
  """Check if x is a qwix WithAux object (a quantized weight placeholder)."""
  return hasattr(x, "how") and hasattr(x, "array")


if __name__ == "__main__":
  app.run(main)
