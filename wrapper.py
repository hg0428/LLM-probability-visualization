from llama_cpp import Llama, llama_get_logits
import llama_cpp
import torch
from dataclasses import dataclass
from typing import Optional, List, Dict, Tuple
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
    def __init__(self, model_path: str, n_gpu_layers: int = -1, n_ctx: int = 8192):
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
        tokens = input_ids[0].tolist()
        self.model.eval(tokens)

        logits_ptr = llama_get_logits(self.model.ctx)
        return torch.from_numpy(
            np.ctypeslib.as_array(logits_ptr, shape=(self.model.n_vocab(),))
        ).unsqueeze(0)

    def reset(self):
        self.model.reset()

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

    def reset(self):
        pass
        # self.model.reset()


class MultiModelWrapper:
    """
    A wrapper that combines multiple models with the same vocabulary and sums their logits
    with specified weights to generate text.
    """

    def __init__(
        self, models_with_weights_and_metadata: List[Tuple[object, float, dict]]
    ):
        """
        Initialize the multi-model wrapper.

        Args:
            models_with_weights_and_metadata: List of tuples containing (model, weight, metadata) triplets
                where metadata is a dict with keys like 'family', 'format', etc.
        """
        if not models_with_weights_and_metadata:
            raise ValueError("At least one model must be provided")

        self.models_with_weights_and_metadata = models_with_weights_and_metadata
        self.primary_model = models_with_weights_and_metadata[0][
            0
        ]  # Use the first model as primary for tokenization

        # Extract just the models and weights for easier access
        self.models_with_weights = [
            (model, weight) for model, weight, _ in models_with_weights_and_metadata
        ]

        # Store model formats for chat formatting
        self.model_formats = {
            id(model): metadata
            for model, _, metadata in models_with_weights_and_metadata
        }

        # Use the primary model's tokenizer and special tokens
        self.device = self.primary_model.device
        self.eos_token_id = self.primary_model.eos_token_id
        self.pad_token_id = self.primary_model.pad_token_id

        # Store primary model metadata
        self.primary_metadata = self.model_formats[id(self.primary_model)]

    def __call__(self, input_ids: torch.LongTensor, chat_messages=None):
        """
        Forward pass that combines models by averaging their probabilities according to weights.

        Args:
            input_ids: Tokenized input for the primary model
            chat_messages: Optional chat messages (not used in this simplified version)
        """
        combined_probs = None
        total_weight = sum(weight for _, weight in self.models_with_weights)

        # Standard processing - use the same input_ids for all models
        for model, weight in self.models_with_weights:
            # Get logits from this model
            model_logits = model(input_ids)

            # Convert logits to probabilities
            model_probs = torch.softmax(model_logits, dim=-1)

            # Normalize the weight
            normalized_weight = weight / total_weight

            # Add weighted probabilities to the combined probabilities
            if combined_probs is None:
                combined_probs = model_probs * normalized_weight
            else:
                combined_probs += model_probs * normalized_weight

        # Convert back to logits for compatibility with the rest of the pipeline
        # Using a small epsilon to avoid log(0)
        epsilon = 1e-10
        combined_logits = torch.log(combined_probs + epsilon)

        return combined_logits

    def reset(self):
        """Reset all models."""
        for model, _ in self.models_with_weights:
            model.reset()

    def tokenize(self, text):
        """Use the primary model's tokenizer."""
        return self.primary_model.tokenize(text)

    def detokenize(self, tokens):
        """Use the primary model's detokenizer."""
        return self.primary_model.detokenize(tokens)

    def to(self, device):
        """Move all models to the specified device."""
        for i, (model, weight) in enumerate(self.models_with_weights):
            self.models_with_weights[i] = (model.to(device), weight)
        self.device = device
        return self
