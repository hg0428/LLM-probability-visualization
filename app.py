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
    "Qwen2.5-0.5B": {
        "name": "Qwen/Qwen2.5-0.5B-Instruct",
        "quantization": "dynamic",  # Options: None, "dynamic"
    },
    "GPT2": {"name": "gpt2", "quantization": None},
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

    if model_name not in loaded_models:
        print(f"Loading model: {model_name}")
        model_config = MODELS[model_name]
        quantization = model_config["quantization"]

        # Determine the best available device
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"

        # Load the model with reduced precision
        model = AutoModelForCausalLM.from_pretrained(
            model_config["name"],
            device_map=device,
            torch_dtype=torch.float16,  # Use half-precision
        )

        # Apply additional memory-saving techniques
        if quantization == "dynamic":
            try:
                # Attempt to reduce model size and memory usage
                for param in model.parameters():
                    param.requires_grad = False

                # Clear any unnecessary caches
                torch.cuda.empty_cache()

                print("Memory optimization applied successfully")
            except Exception as e:
                print(f"Warning: Memory optimization failed: {e}")

        # Move to the appropriate device
        if device == "mps":
            model = model.to("mps")

        tokenizer = AutoTokenizer.from_pretrained(model_config["name"])
        loaded_models[model_name] = (model, tokenizer)

    return loaded_models[model_name]


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
    chat_history = data.get("chat_history", [])
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
        formatted_prompt = format_chat_history(chat_history, model_name)
        n = 0
        # Generate text with alternatives
        for token, alternatives in generate_text(
            model,
            tokenizer,
            formatted_prompt,
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
                    "message_id": len(chat_history) - 1,
                    "token_id": n,
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
