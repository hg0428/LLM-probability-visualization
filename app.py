from flask import Flask, render_template, request, jsonify, session
from flask_socketio import SocketIO, emit
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    StoppingCriteria,
    StoppingCriteriaList,
)
import torch
import torch.quantization
import string
from chat_templates import format_chat_history
from itertools import count
from uuid import uuid4
import random
import json
from text_generation import benchmark_generation, generate_text
from wrapper import LlamaWrapper, TransformersWrapper, MultiModelWrapper


class StopOnTokens(StoppingCriteria):
    def __init__(self, stop_token_ids):
        self.stop_token_ids = stop_token_ids

    def __call__(self, input_ids, scores, **kwargs):
        for stop_ids in self.stop_token_ids:
            if torch.eq(input_ids[0][-len(stop_ids) :], stop_ids).all():
                return True
        return False


# Available models
MODELS = {
    # "Qwen2.5-0.5B": {"name": "Qwen/Qwen2.5-0.5B-Instruct", "type": "transformers"},
    "Qwen2.5-0.5B": {
        "name": "/Users/hudsongouge/.ollama/models/blobs/sha256-fa4d41b65761ed565cac6b5f62e35135d050408b033114a128ab308c02b2e83a",
        "type": "llama.cpp",
        "family": "qwen2.5",
        "format": "chatml",
    },
    "Qwen2.5-3B": {
        "name": "/Users/hudsongouge/.ollama/models/blobs/sha256-5ee4f07cdb9beadbbb293e85803c569b01bd37ed059d2715faa7bb405f31caa6",
        "type": "llama.cpp",
        "family": "qwen2.5",
        "format": "chatml",
    },
    "Qwen2.5-7B": {
        "name": "/Users/hudsongouge/.cache/lm-studio/models/qwen2.5/qwen2.5-7b-GGUF/qwen2.5-7b.gguf",
        "type": "llama.cpp",
        "family": "qwen2.5",
        "format": "chatml",
    },
    "Qwen2.5-14B": {
        "name": "/Users/hudsongouge/.cache/lm-studio/models/qwen2.5/qwen2.5-14b-GGUF/qwen2.5-14b.gguf",
        "type": "llama.cpp",
        "family": "qwen2.5",
        "format": "chatml",
    },
    "Qwen2.5-32B": {
        "name": "/Users/hudsongouge/.cache/lm-studio/models/qwen2.5/qwen2.5-32b-GGUF/qwen2.5-32b.gguf",
        "type": "llama.cpp",
        "family": "qwen2.5",
        "format": "chatml",
    },
    "Llama3.2-1B": {
        "name": "/Users/hudsongouge/.ollama/models/blobs/sha256-74701a8c35f6c8d9a4b91f3f3497643001d63e0c7a84e085bed452548fa88d45",
        "type": "llama.cpp",
        "family": "llama3",
        "format": "llama",
    },
    "Llama3.2-3B": {
        "name": "/Users/hudsongouge/.ollama/models/blobs/sha256-dde5aa3fc5ffc17176b5e8bdc82f587b24b2678c6c66101bf7da77af9f7ccdff",
        "type": "llama.cpp",
        "family": "llama3",
        "format": "llama",
    },
    "Llama3.1-8B": {
        "name": "/Users/hudsongouge/.ollama/models/blobs/sha256-4f6dc812262ac5e1a74791c2a86310ebba1aa1804fa3cd1c216f5547a620d2f2",
        "type": "llama.cpp",
        "family": "llama3",
        "format": "llama",
    },
    "Chatter-70M": {
        "name": "/Users/hudsongouge/.cache/lm-studio/models/Chatter/chatter-70m/model_q4_k_m.gguf",
        "type": "llama.cpp",
        "family": "llama2",
        "format": "chatter",
    },
    "Granite3-8B": {
        "name": "/Users/hudsongouge/.cache/lm-studio/models/granite3-dense/granite3-dense-8b-GGUF/granite3-dense-8b.gguf",
        "type": "llama.cpp",
        "family": "granite3",
        "format": "granite3",
    },
    "Gemma2-2B": {
        "name": "/Users/hudsongouge/.ollama/models/blobs/sha256-7462734796d67c40ecec2ca98eddf970e171dbb6b370e43fd633ee75b69abe1b",
        "type": "llama.cpp",
        "family": "gemma2",
        "format": "gemma2",
    },
    "Gemma2-9B": {
        "name": "/Users/hudsongouge/.cache/lm-studio/models/gemma2/gemma2-9b-GGUF/gemma2-9b.gguf",
        "type": "llama.cpp",
        "family": "gemma2",
        "format": "gemma2",
    },
    "Gemma2-27B": {
        "name": "/Users/hudsongouge/.cache/lm-studio/models/gemma2/gemma2-27b-GGUF/gemma2-27b.gguf",
        "type": "llama.cpp",
        "family": "gemma2",
        "format": "gemma2",
    },
    "TinyLLM-10M": {
        "name": "/Users/hudsongouge/.cache/lm-studio/models/aimlresearch2023/Tiny-LLM-Q8_0-GGUF/tiny-llm-q8_0.gguf",
        "type": "llama.cpp",
        "family": "llama2",
        "format": "llama2",
    },
    "Llama3.1-8B-Uncensored": {
        "name": "/Users/hudsongouge/.cache/lm-studio/models/Hudson-llama3.1-uncensored/Hudson/llama3.1-uncensored-8b-GGUF/llama3.1-uncensored-8b.gguf",
        "type": "llama.cpp",
        "family": "llama3",
        "format": "llama3",
    },
    "Llama3.2-1B-Math": {
        "name": "/Users/hudsongouge/.cache/lm-studio/models/llama3.2-math/llama3.2-math-1b-GGUF/llama3.2-math-1b.gguf",
        "type": "llama.cpp",
        "family": "llama3",
        "format": "llama3",
    },
    "Llama2-7B": {
        "name": "/Users/hudsongouge/.cache/lm-studio/models/TheBloke/Llama-2-7B-Chat-GGUF/llama-2-7b-chat.Q4_K_S.gguf",
        "type": "llama.cpp",
        "family": "llama2",
        "format": "llama2",
    },
    "Granite3.2-8B": {
        "name": "/Users/hudsongouge/.cache/lm-studio/models/Triangle104/granite-3.2-8b-instruct-preview-Q4_K_M-GGUF/granite-3.2-8b-instruct-preview-q4_k_m.gguf",
        "type": "llama.cpp",
        "family": "granite3",
        "format": "granite3",
    },
}

# Model families for compatibility
MODEL_FAMILIES = {
    "qwen2.5": [
        "Qwen2.5-0.5B",
        "Qwen2.5-3B",
        "Qwen2.5-7B",
        "Qwen2.5-14B",
        "Qwen2.5-32B",
    ],
    "llama3": ["Llama3.2-1B", "Llama3.2-3B", "Llama3.2-1B-Math"],
    "granite3": ["Granite3-dense-8B", "Granite3.2-8B"],
    "llama2": ["Chatter-70M", "Llama2-7B", "TinyLLM-10M"],
    "gemma2": ["Gemma2-2B", "Gemma2-9B", "Gemma2-27B"],
}

app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*")

# Initialize models dict to store loaded models
loaded_models = {}
loaded_tokenizers = {}


def get_model_and_tokenizer(model_spec):
    """Get model and tokenizer based on model specification.

    Args:
        model_spec: Either a string (model name) or a JSON string containing model names and weights
    """
    # Check if this is a multi-model configuration
    if model_spec.startswith("{"):
        try:
            model_config = json.loads(model_spec)
            if not isinstance(model_config, dict) or "models" not in model_config:
                raise ValueError("Invalid multi-model configuration")

            models_with_weights = []
            primary_model_name = None
            model_family = None

            # Load all models
            for model_entry in model_config["models"]:
                model_name = model_entry["name"]
                weight = float(model_entry["weight"])

                if not primary_model_name:
                    primary_model_name = model_name
                    # Store the primary model's family for compatibility checking
                    model_family = MODELS[model_name]["family"]
                else:
                    # Check if this model is from the same family as the primary model
                    current_family = MODELS[model_name]["family"]
                    if current_family != model_family:
                        print(f"Warning: Mixing model families ({model_family} and {current_family}) may cause unexpected results")

                # Load the individual model
                if model_name not in loaded_models:
                    _load_single_model(model_name)

                # Get model metadata
                model_metadata = MODELS[model_name].copy()

                # Add model with weight and metadata
                models_with_weights.append(
                    (loaded_models[model_name], weight, model_metadata)
                )

            # Create a multi-model wrapper
            multi_model = MultiModelWrapper(models_with_weights)
            multi_model_key = model_spec  # Use the JSON string as the key

            loaded_models[multi_model_key] = multi_model
            loaded_tokenizers[multi_model_key] = loaded_tokenizers[primary_model_name]

            tps = benchmark_generation(multi_model, "Hello,")
            print(f"Multi-model configuration\nTPS: {tps}")

            return multi_model, loaded_tokenizers[primary_model_name]

        except (json.JSONDecodeError, KeyError, ValueError) as e:
            raise ValueError(f"Invalid multi-model configuration: {str(e)}")
    else:
        # Single model case
        return _load_single_model(model_spec)


def _load_single_model(model_name):
    """Load a single model by name."""
    if model_name not in MODELS:
        raise ValueError(
            f"Model {model_name} not found. Available models: {list(MODELS.keys())}"
        )

    if model_name in loaded_models:
        return loaded_models[model_name], loaded_tokenizers[model_name]

    model_config = MODELS[model_name]

    if model_config["type"] == "llama.cpp":
        model = LlamaWrapper(
            model_path=model_config["name"],
            n_gpu_layers=model_config.get("n_gpu_layers", -1),
        )
        tokenizer = model.tokenizer
    else:
        # Existing transformers model loading logic
        tokenizer = AutoTokenizer.from_pretrained(model_config["name"])
        model = TransformersWrapper(
            AutoModelForCausalLM.from_pretrained(model_config["name"]), tokenizer
        )

        if torch.cuda.is_available():
            model = model.to("cuda")
        if torch.backends.mps.is_available():
            model = model.to("mps")

    loaded_models[model_name] = model
    loaded_tokenizers[model_name] = tokenizer

    tps = benchmark_generation(model, "Hello,")
    print(f"Model {model_name}\nTPS: {tps}")
    return model, tokenizer


@app.route("/")
def index():
    return render_template(
        "index.html", models=MODELS.keys(), model_families=MODEL_FAMILIES
    )


@socketio.on("connect")
def handle_connect():
    print("Client connected")


@socketio.on("disconnect")
def handle_disconnect():
    print("Client disconnected")


@socketio.on("generate")
def handle_generate(data):
    generation_id = uuid4()
    session["generating"] = generation_id
    mode = data.get("mode", "chat")
    model_name = data.get("model_name", "Qwen2.5-0.5B")
    top_k = data.get("top_k", 0)
    num_show = data.get("num_show", 12)
    temperature = data.get("temperature", 0.7)
    min_p = data.get("min_p", 0)
    max_new_tokens = data.get("max_new_tokens", 100)
    top_p = data.get("top_p", 0.9)
    repetition_penalty = data.get("repetition_penalty", 1.0)
    frequency_penalty = data.get("frequency_penalty", 0.0)
    presence_penalty = data.get("presence_penalty", 0.0)
    repeat_last_n = data.get("repeat_last_n", 64)
    randomness_enabled = data.get("randomness_enabled", True)
    penalties_enabled = data.get("penalties_enabled", True)
    dry_enabled = data.get("dry_enabled", True)
    dry_allowed_length = data.get("dry_allowed_length", 1)
    dry_base = data.get("dry_base", 2)
    dry_multiplier = data.get("dry_multiplier", 3)
    dry_range = data.get("dry_range", 1024)
    xtc_enabled = data.get("xtc_enabled", False)
    xtc_threshold = data.get("xtc_threshold", 0.2)
    xtc_probability = data.get("xtc_probability", 0.5)
    truncation_enabled = data.get("truncation_enabled", True)

    try:
        model, tokenizer = get_model_and_tokenizer(model_name)

        # Get chat history and prompt
        chat_history = data.get("chat_history", []) if mode == "chat" else None

        # Format chat history for display/tokenization
        if mode == "chat":
            # For multi-model configurations, use a consistent format based on the primary model
            if model_name.startswith("{"): 
                try:
                    # Parse the JSON to get the primary model name
                    import json
                    model_config = json.loads(model_name)
                    primary_model_name = model_config["models"][0]["name"]
                    
                    # Get the primary model's format information
                    primary_model_config = MODELS[primary_model_name]
                    model_family = primary_model_config["family"]
                    model_format = primary_model_config["format"]
                    
                    print(f"Using {primary_model_name}'s format ({model_format}) for multi-model chat")
                except (json.JSONDecodeError, KeyError, IndexError) as e:
                    print(f"Error parsing multi-model config: {e}")
                    # Fallback to chatml format which is widely supported
                    model_family = "default"
                    model_format = "chatml"
                    primary_model_name = "unknown"
            else:
                # Single model configuration
                model_config = MODELS[model_name]
                model_family = model_config["family"]
                model_format = model_config["format"]
                primary_model_name = model_name

            prompt = format_chat_history(
                chat_history,
                model,
                primary_model_name,
                model_family,
                model_format,
            )
        else:
            prompt = data.get("prompt", "")
        n = 0
        # Generate text with alternatives
        for token, alternatives in generate_text(
            model,
            prompt,
            max_new_tokens,
            top_k,
            temperature,
            min_p,
            num_show,
            top_p,
            repetition_penalty,
            frequency_penalty,
            presence_penalty,
            repeat_last_n,
            truncation_enabled,
            randomness_enabled,
            penalties_enabled,
            dry_enabled,
            xtc_enabled,
            dry_allowed_length,
            dry_base,
            dry_multiplier,
            dry_range,
            xtc_threshold,
            xtc_probability,
            chat_history,
        ):
            if session["generating"] != generation_id:
                return
            socketio.emit(
                "token",
                {
                    "chosen": token,
                    "options": [[t, float(p), bool(c)] for t, p, c in alternatives],
                    "message_id": len(chat_history) - 1 if mode == "chat" else None,
                    "token_id": n,
                    "mode": mode,
                },
            )
            n += 1
            if session["generating"] != generation_id:
                return
        socketio.emit("end")
        session["generating"] = None

    except Exception as e:
        print(f"Error in generate: {str(e)}")
        socketio.emit("error", {"message": str(e)})
        raise e


@socketio.on("stop")
def stop():
    session["generating"] = None


if __name__ == "__main__":
    # Pre-load the default model
    get_model_and_tokenizer("Qwen2.5-0.5B")
    socketio.run(app, debug=True, port=5002)
