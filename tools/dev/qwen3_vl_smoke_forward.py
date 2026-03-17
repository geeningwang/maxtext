import os
import sys
import jax
import jax.numpy as jnp
import numpy as np

sys.path.insert(0, "src")
from maxtext.configs import pyconfig
from maxtext.models import models
from flax import nnx
from maxtext.multimodal.processor import get_bidirectional_mask_vision

config = pyconfig.initialize([
    "", "src/maxtext/configs/post_train/sft.yml", "model_name=qwen3-vl-2b",
    "packing=False", "enable_checkpointing=False", "base_num_decoder_layers=2"
])

print("Instantiating Transformer...")
transformer = models.Transformer(config, jax.sharding.Mesh(jax.devices(), ('x',)), quant=None, rngs=nnx.Rngs(0))
graphdef, state = nnx.split(transformer)

@jax.jit
def forward(state_inner, tks, pos, deep_embeds):
    m_transformer = nnx.merge(graphdef, state_inner)
    bidirectional_mask = get_bidirectional_mask_vision(config, tks)
    out_logits, _, _ = m_transformer.decoder(
        shared_embedding=m_transformer.token_embedder,
        decoder_input_tokens=tks,
        decoder_positions=pos,
        bidirectional_mask=bidirectional_mask,
        deepstack_visual_embeds=deep_embeds,
        deterministic=True,
    )
    return out_logits

# Setup inputs
batch_size = 1
seq_len = 16
tks = jnp.ones((batch_size, seq_len), dtype=jnp.int32) * 151643
pos = jnp.zeros((3, batch_size, seq_len), dtype=jnp.int32)
deep_embeds = [None] * config.base_num_decoder_layers

logits = forward(state, tks, pos, deep_embeds)
print(logits.shape)
