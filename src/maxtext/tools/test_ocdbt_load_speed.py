"""Quick load speed test for OCDBT checkpoint vs zarr2 baseline."""

import os
import time
from typing import Sequence

import jax
from absl import app

from maxtext.configs import pyconfig
from maxtext.inference.maxengine import maxengine
from maxtext.utils import max_logging


def main(argv: Sequence[str]) -> None:
  jax.config.update("jax_default_prng_impl", "unsafe_rbg")
  os.environ["TF_CPP_MIN_LOG_LEVEL"] = "0"

  config = pyconfig.initialize(argv)
  rank = jax.process_index()

  max_logging.log(f"[W{rank}] Starting OCDBT load test ...")
  max_logging.log(f"[W{rank}] load_parameters_path = {config.load_parameters_path}")
  max_logging.log(f"[W{rank}] use_ocdbt = {config.checkpoint_storage_use_ocdbt}")
  max_logging.log(f"[W{rank}] use_zarr3 = {config.checkpoint_storage_use_zarr3}")

  t0 = time.time()
  engine = maxengine.MaxEngine(config)
  t1 = time.time()
  max_logging.log(f"[W{rank}] MaxEngine init took {t1 - t0:.2f}s")

  rng = jax.random.PRNGKey(0)
  params = engine.load_params(rng)
  t2 = time.time()

  load_time = t2 - t1
  total_time = t2 - t0

  max_logging.log(f"[W{rank}] Checkpoint load: {load_time:.2f}s")
  max_logging.log(f"[W{rank}] Init + load total: {total_time:.2f}s")

  jax.effects_barrier()

  if rank == 0:
    zarr2_baseline = 490.0
    max_logging.log("=" * 50)
    max_logging.log("LOAD SPEED SUMMARY")
    max_logging.log(f"  Engine init:      {t1 - t0:.2f}s")
    max_logging.log(f"  Checkpoint load:  {load_time:.2f}s")
    max_logging.log(f"  Total:            {total_time:.2f}s")
    max_logging.log(f"  Zarr2 baseline:   {zarr2_baseline:.1f}s")
    max_logging.log(f"  Speedup:          {zarr2_baseline / load_time:.1f}x")
    max_logging.log("=" * 50)


if __name__ == "__main__":
  app.run(main)
