"""Prepare simple synthetic finetuning data for testing."""

from datasets import Dataset

# 1. Define the finetune data directly
raw_data = [
    {
        "messages": [
            {"role": "system", "content": "You are a Qwen3 expert."},
            {"role": "user", "content": "What is the context window of Qwen3?"},
            {
                "role": "assistant",
                "content": "Qwen3 supports a context window of up to 128k tokens by default.",
            },
        ]
    },
    {
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {
                "role": "user",
                "content": "Write a JAX function for matrix multiplication.",
            },
            {
                "role": "assistant",
                "content": "Certainly! In JAX, you use `jax.numpy.matmul` or the "
                "`@` operator: \n\n```python\nimport jax.numpy as jnp\n"
                "def matmul(a, b):\n    return jnp.matmul(a, b)\n```",
            },
        ]
    },
    {
        "messages": [
            {"role": "user", "content": "Explain Qwen3's thinking mode."},
            {
                "role": "assistant",
                "content": "<thought>The user is asking about the new inference-time "
                "scaling/reasoning features in Qwen3.</thought>Qwen3 "
                "introduces a 'Thinking' mode that allows the model to "
                "allocate more compute to complex reasoning steps "
                "before providing a final answer.",
            },
        ]
    },
]

# 2. Create dataset directly from list of dicts
ds = Dataset.from_dict({"messages": [ex["messages"] for ex in raw_data]})

# 3. Save it as a directory
ds.save_to_disk("my_data_conversational")

print("Successfully created 'my_data_conversational' with finetune data.")
print(f"Dataset info:\n{ds}")
