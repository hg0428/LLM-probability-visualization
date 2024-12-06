from text_generation import generate_text
from wrapper import LlamaWrapper, TransformersWrapper
import time

model = LlamaWrapper(
    model_path="/Users/hudsongouge/.ollama/models/blobs/sha256-fa4d41b65761ed565cac6b5f62e35135d050408b033114a128ab308c02b2e83a",
    n_gpu_layers=-1,
    n_ctx=100,
)

prompt = "Hello,"

start = time.time()
tokens = 0
for token, alternatives in generate_text(
    model,
    prompt,
    100,
    truncation_enabled=False,
    randomness_enabled=False,
    penalties_enabled=False,
    dry_enabled=False,
    xtc_enabled=False,
):
    tokens += 1

print(f"Time taken: {time.time() - start}")
print(f"Tokens: {tokens}")
print(f"TPS: {tokens / (time.time() - start)}")
