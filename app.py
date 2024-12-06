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
from text_generation import generate_text
from wrapper import LlamaWrapper, TransformersWrapper


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
    },
    "Qwen2.5-3B": {
        "name": "/Users/hudsongouge/.ollama/models/blobs/sha256-5ee4f07cdb9beadbbb293e85803c569b01bd37ed059d2715faa7bb405f31caa6",
        "type": "llama.cpp",
    },
    "Llama3.2-1B": {
        "name": "/Users/hudsongouge/.ollama/models/blobs/sha256-74701a8c35f6c8d9a4b91f3f3497643001d63e0c7a84e085bed452548fa88d45",
        "type": "llama.cpp",
    },
    "Llama3.2-3B": {
        "name": "/Users/hudsongouge/.ollama/models/blobs/sha256-dde5aa3fc5ffc17176b5e8bdc82f587b24b2678c6c66101bf7da77af9f7ccdff",
        "type": "llama.cpp",
    },
}

app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*")

# Initialize models dict to store loaded models
loaded_models = {}
loaded_tokenizers = {}


def get_model_and_tokenizer(model_name):
    """Get model and tokenizer based on model name."""
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

    return model, tokenizer


@app.route("/")
def index():
    return render_template("index.html", models=MODELS.keys())


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

        # Format chat history
        if mode == "chat":
            chat_history = data.get("chat_history", [])
            prompt = format_chat_history(chat_history, model, model_name)
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
