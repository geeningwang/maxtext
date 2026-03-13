"""Generate golden logits and hidden states for Qwen3-VL full forward pass."""

import json
import os
import torch
from transformers import AutoProcessor, AutoModelForImageTextToText


def generate_golden_logits():
  """Extract Qwen3-VL full model logits and hidden states."""
  model_id = "Qwen/Qwen3-VL-2B-Instruct"
  processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
  # We want hidden states
  model = AutoModelForImageTextToText.from_pretrained(
      model_id, torch_dtype=torch.float32, trust_remote_code=True, output_hidden_states=True
  )

  from PIL import Image  # pylint: disable=import-outside-toplevel

  image_path = "tests/assets/test_image.jpg"
  image = Image.open(image_path).convert("RGB")

  messages = [
      {"role": "user", "content": [{"type": "image", "image": image}, {"type": "text", "text": "Describe this image."}]}
  ]
  text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

  inputs = processor(text=[text], images=[image], padding=True, return_tensors="pt")

  with torch.no_grad():
    outputs = model(**inputs)

  # Grab input tensors used
  input_ids = inputs.input_ids
  pixel_values = inputs.pixel_values
  image_grid_thw = inputs.image_grid_thw

  logits = outputs.logits
  # Just grab the last layer's hidden state
  hidden_states = outputs.hidden_states[-1]

  output_path = "tests/assets/golden_logits/golden_data_qwen3_vl_logits.jsonl"
  os.makedirs(os.path.dirname(output_path), exist_ok=True)

  data = {
      "input_ids": input_ids.cpu().numpy().tolist(),
      "pixel_values": pixel_values.cpu().numpy().tolist(),
      "image_grid_thw": image_grid_thw.cpu().numpy().tolist(),
      "logits": logits.cpu().numpy().tolist(),
      "hidden_states": hidden_states.cpu().numpy().tolist(),
  }

  with open(output_path, "w", encoding="utf-8") as f:
    f.write(json.dumps(data) + "\n")
  print(f"Saved golden logits to {output_path}")


if __name__ == "__main__":
  generate_golden_logits()
