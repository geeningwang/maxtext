"""Validation script for MaxText configurations."""

import os
from maxtext.configs import pyconfig

# Set logging level to reduce noise
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

print("Validating qwen3-vl-2b...")
try:
  config_2b = pyconfig.initialize(
      [
          "",
          "src/maxtext/configs/post_train/sft.yml",
          "model_name=qwen3-vl-2b",
          "run_name=test_2b",
          "packing=False",
          "enable_checkpointing=False",
      ]
  )
  print(
      f"  ✅ Success! Model: {config_2b.model_name}, "
      f"Multimodal: {config_2b.use_multimodal}, "
      f"ViT Hidden Size: {config_2b.hidden_size_for_vit}"
  )
except ValueError as e:
  print(f"  ❌ Failed validation for 2B: {e}")

print("\nValidating qwen3-vl-8b...")
try:
  config_8b = pyconfig.initialize(
      [
          "",
          "src/maxtext/configs/post_train/sft.yml",
          "model_name=qwen3-vl-8b",
          "run_name=test_8b",
          "packing=False",
          "enable_checkpointing=False",
      ]
  )
  print(
      f"  ✅ Success! Model: {config_8b.model_name}, "
      f"Multimodal: {config_8b.use_multimodal}, "
      f"ViT Hidden Size: {config_8b.hidden_size_for_vit}"
  )
except ValueError as e:
  print(f"  ❌ Failed validation for 8B: {e}")
