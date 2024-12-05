from llama_cpp import Llama, llama_get_logits
import torch
from dataclasses import dataclass
from typing import Optional, List
import numpy as np


class LlamaTokenizerWrapper:
    def __init__(self, tokenizer, device: torch.device):
        self.tokenizer = tokenizer
        self.device = device

    def encode(self, text, **kwargs):
        return torch.tensor([self.tokenizer.encode(text)]).to(self.device)

    def decode(self, tokens, **kwargs):
        return self.tokenizer.decode(tokens)


class LlamaWrapper:
    def __init__(self, model_path: str, n_gpu_layers: int = 1, n_ctx: int = 2048):
        self.model = Llama(
            model_path=model_path,
            n_gpu_layers=n_gpu_layers,
            n_ctx=n_ctx,
            verbose=False,
            flash_attn=True,
        )
        # assign device automatically (e.g. CUDA, MPS, CPU)
        self.device = torch.device(
            "mps"
            if torch.backends.mps.is_available()
            else "cuda" if torch.cuda.is_available() else "cpu"
        )
        self._tokenizer = self.model.tokenizer()
        self.tokenizer = LlamaTokenizerWrapper(self._tokenizer, self.device)
        self.eos_token_id = self.model.token_eos()
        self.pad_token_id = self.model.token_eos()

    def __call__(self, input_ids: torch.LongTensor):
        self.model.reset()
        # Directly evaluate tokens
        tokens = input_ids[0].tolist()
        self.model.eval(tokens)

        # Get raw logits pointer
        logits_ptr = llama_get_logits(self.model.ctx)

        # Use numpy for fastest conversion
        return torch.from_numpy(
            np.array([np.ctypeslib.as_array(logits_ptr, shape=(self.model.n_vocab(),))])
        ).to(self.device)

    def tokenize(self, text):
        return self.tokenizer.encode(text)

    def detokenize(self, tokens):
        return self.tokenizer.decode(tokens)

    def to(self, device):
        # No-op as llama.cpp handles device placement internally
        return self


class TransformersWrapper:
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self.device = model.device
        self.eos_token_id = self.tokenizer.eos_token_id
        self.pad_token_id = self.tokenizer.pad_token_id

    def __call__(self, input_tokens):
        outputs = self.model(input_tokens)
        logits = outputs.logits[:, -1, :]
        return logits

    def tokenize(self, text):
        return self.tokenizer.encode(text, return_tensors="pt")

    def detokenize(self, tokens):
        return self.tokenizer.decode([tokens])

    def to(self, device):
        self.model = self.model.to(device)
        self.device = device
        return self
