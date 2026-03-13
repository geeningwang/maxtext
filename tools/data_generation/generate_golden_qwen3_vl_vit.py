"""Generate golden image embeddings for Qwen3-VL Vision Encoder."""

import json
import os
import torch
from transformers import AutoProcessor, AutoModelForImageTextToText


def generate_golden_data():
  """Extract Qwen3-VL golden data for tests."""
  model_id = "Qwen/Qwen3-VL-2B-Instruct"
  processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
  model = AutoModelForImageTextToText.from_pretrained(model_id, torch_dtype=torch.float32, trust_remote_code=True)

  from PIL import Image  # pylint: disable=import-outside-toplevel

  image_path = "tests/assets/test_image.jpg"
  image = Image.open(image_path).convert("RGB")

  messages = [
      {"role": "user", "content": [{"type": "image", "image": image}, {"type": "text", "text": "Describe this image."}]}
  ]
  text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

  inputs = processor(text=[text], images=[image], padding=True, return_tensors="pt")

  with torch.no_grad():
    pixel_values = inputs.pixel_values
    image_grid_thw = inputs.image_grid_thw

    # We need the output of the vision encoder (and projector if applicable)
    visual_outputs = model.visual(pixel_values, grid_thw=image_grid_thw)
    if isinstance(visual_outputs, tuple):
      soft_embeddings = visual_outputs[0]
      if len(visual_outputs) > 1:
        deep_features = visual_outputs[1]
      else:
        deep_features = None
    else:
      soft_embeddings = visual_outputs
      deep_features = None

  output_path = "tests/assets/golden_logits/golden_data_qwen3_vl_vit.jsonl"
  os.makedirs(os.path.dirname(output_path), exist_ok=True)

  data = {
      "soft_embeddings": soft_embeddings.cpu().numpy().tolist(),
      "pixel_values": pixel_values.cpu().numpy().tolist(),
      "image_grid_thw": image_grid_thw.cpu().numpy().tolist(),
  }

  if deep_features is not None:
    data["deep_features"] = [f.cpu().numpy().tolist() for f in deep_features]

  with open(output_path, "w", encoding="utf-8") as f:
    f.write(json.dumps(data) + "\n")
  print(f"Saved golden data to {output_path}")


if __name__ == "__main__":
  generate_golden_data()
