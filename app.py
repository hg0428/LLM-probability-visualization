from flask import Flask, render_template, request, jsonify, session
from flask_socketio import SocketIO, emit
from transformers import AutoModelForCausalLM, AutoTokenizer, StoppingCriteria, StoppingCriteriaList
import torch
import torch.quantization
import string
from chat_templates import format_chat_history
from itertools import count
from uuid import uuid4


class StopOnTokens(StoppingCriteria):
    def __init__(self, stop_token_ids):
        self.stop_token_ids = stop_token_ids

    def __call__(self, input_ids, scores, **kwargs):
        for stop_ids in self.stop_token_ids:
            if torch.eq(input_ids[0][-len(stop_ids):], stop_ids).all():
                return True
        return False

# Available models
MODELS = {
    "Qwen2.5-0.5B": {
        "name": "Qwen/Qwen2.5-0.5B-Instruct",
        "quantization": "dynamic"  # Options: None, "dynamic"
    },
    "GPT2": {
        "name": "gpt2",
        "quantization": None
    },
}

app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*")

# Initialize models dict to store loaded models
loaded_models = {}
loaded_tokenizers = {}
sequence_breaker_strings = ["\n", ":", "\"", "*"]

def get_model_and_tokenizer(model_name):
    """Get model and tokenizer based on model name."""
    if model_name not in MODELS:
        raise ValueError(f"Model {model_name} not found. Available models: {list(MODELS.keys())}")
    
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
            torch_dtype=torch.float16  # Use half-precision
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

def calculate_dry_penalty(
    input_ids: torch.LongTensor,
    scores: torch.FloatTensor,
    _range: int = 1024,
    sequence_breakers=None,
    allowed_length=1,
    base=2,
    multiplier=3,
) -> torch.FloatTensor:
    if _range > 0:
        input_ids = input_ids[:, -_range:]

    for input_ids_row, scores_row in zip(input_ids, scores):
        # Raw integer must be extracted here to check for set membership.
        last_token = input_ids_row[-1].item()

        if last_token in sequence_breakers:
            continue

        # Exclude the last token as it always matches.
        match_indices = (input_ids_row[:-1] == last_token).nonzero()

        # Stores the maximum matching sequence length
        # for each token immediately following the sequence in the input.
        match_lengths = {}

        for i in match_indices:
            next_token = input_ids_row[i + 1].item()

            if next_token in sequence_breakers:
                continue

            # We have already found that `last_token` matches at this index,
            # so the match is at least of length 1.
            match_length = 1

            # Extend the match backwards as far as possible.
            while True:
                j = i - match_length
                if j < 0:
                    # Start of input reached.
                    break

                previous_token = input_ids_row[-(match_length + 1)].item()
                if input_ids_row[j] != previous_token:
                    # Start of match reached.
                    break

                if previous_token in sequence_breakers:
                    # Sequence-breaking token reached.
                    break

                match_length += 1

            if next_token in match_lengths:
                match_lengths[next_token] = max(match_length, match_lengths[next_token])
            else:
                match_lengths[next_token] = match_length

        # Apply penalties.
        for token, match_length in match_lengths.items():
            if match_length >= allowed_length:
                penalty = multiplier * base ** (match_length - allowed_length)
                scores_row[token] -= torch.tensor(penalty, dtype=scores_row.dtype, device=scores_row.device)

    return scores


def generate_text(model, tokenizer, prompt, max_new_tokens=0, top_k=50, temperature=0.7, min_p=0, num_show=12, 
                 top_p=0.9, repetition_penalty=1.0, frequency_penalty=0.0, presence_penalty=0.0, repeat_last_n=64, randomness_enabled=True, 
                 penalties_enabled=True, dry_enabled=True, dry_allowed_length=2, dry_base=1.2, dry_multiplier=2, dry_range=512):
    """Generate text and return token alternatives for each position."""
    try:
        sequence_breakers = {tokenizer.encode(f"a{s}")[-1] for s in sequence_breaker_strings}
        # Tokenize input
        input_ids = tokenizer.encode(prompt, return_tensors='pt')
        device = model.device
        input_ids = input_ids.to(device)
        sequence_alternatives = []
        current_tokens = input_ids
        
        # Track token frequencies and presence for penalties
        token_frequencies = {}
        token_presence = set()
        
        # Generate one token at a time
        for i in range(max_new_tokens):
            # Get model outputs and probabilities
            with torch.no_grad():
                outputs = model(current_tokens)
                logits = outputs.logits[:, -1, :]
                # Get original probabilities for displaying alternatives
                orig_probs = torch.softmax(logits, dim=-1)
                num_show_probs, num_show_indices = torch.topk(orig_probs[0], k=min(num_show, len(orig_probs[0])))
                
                # Apply repetition penalty
                if penalties_enabled and repetition_penalty != 1.0:
                    for token_id in set(current_tokens[0][-repeat_last_n:].tolist()):
                        logits[0, token_id] /= repetition_penalty
                
                # Apply frequency penalty
                if penalties_enabled and frequency_penalty > 0:
                    for token_id, freq in token_frequencies.items():
                        logits[0, token_id] -= frequency_penalty * freq
                
                # Apply presence penalty
                if penalties_enabled and presence_penalty != 0:
                    for token_id in token_presence:
                        logits[0, token_id] -= presence_penalty
                
                # Apply DRY penalty
                if dry_enabled:
                    logits = calculate_dry_penalty(
                        current_tokens,
                        logits,
                        _range=dry_range,
                        sequence_breakers=sequence_breakers,
                        allowed_length=dry_allowed_length,
                        base=dry_base,
                        multiplier=dry_multiplier
                    )
                
                # Now apply sampling modifications
                probs = torch.softmax(logits, dim=-1)
                if randomness_enabled:
                    # Apply top-p (nucleus) sampling
                    if 0 < top_p < 1.0:
                        sorted_probs, sorted_indices = torch.sort(probs[0], descending=True)
                        cumsum_probs = torch.cumsum(sorted_probs, dim=-1)
                        # Remove tokens with cumulative probability above the threshold
                        sorted_indices_to_remove = cumsum_probs > top_p
                        # Shift the indices to the right to keep also the first token above the threshold
                        sorted_indices_to_remove[1:] = sorted_indices_to_remove[:-1].clone()
                        sorted_indices_to_remove[0] = False
                        indices_to_remove = sorted_indices[sorted_indices_to_remove]
                        probs[0, indices_to_remove] = 0.0
                        # Renormalize
                        probs = probs / probs.sum(dim=-1, keepdim=True)
                
                    # Get top k probabilities and tokens
                    if top_k == 0:
                        top_k_probs = probs[0]
                        top_k_indices = torch.arange(len(probs[0]), device=device)
                    else:
                        top_k_probs, top_k_indices = torch.topk(probs[0], k=min(top_k, len(probs[0])))
                    
                    # Sample next token using temperature
                    if temperature == 0:
                        # If temperature is 0, equivalent to greedy sampling
                        chosen_idx = torch.argmax(top_k_probs)
                    else:
                        scaled_probs = torch.softmax(torch.log(top_k_probs.clamp(min=1e-10)) / temperature, dim=-1)
                        if min_p > 0:
                            scaled_probs = torch.where(
                                scaled_probs > min_p, scaled_probs, torch.tensor(0.0, device=device)
                            )
                            if scaled_probs.sum() == 0:
                                scaled_probs = torch.ones_like(scaled_probs) / len(scaled_probs)
                        
                        # Ensure probabilities are valid for multinomial sampling
                        scaled_probs = scaled_probs / scaled_probs.sum()  # Renormalize
                        chosen_idx = torch.multinomial(scaled_probs, num_samples=1)[0]
                    
                    next_token = top_k_indices[chosen_idx]
                else:
                    # When randomness is disabled, just pick the most likely token
                    next_token = torch.argmax(probs[0])
            
            # Record alternatives including the chosen token
            alternatives = []
            for prob, idx in zip(num_show_probs, num_show_indices):
                token = tokenizer.decode([idx.item()])
                alternatives.append([
                    token,
                    float(prob.item()),  
                    idx.item() == next_token.item()
                ])
            yield tokenizer.decode([next_token.item()]), alternatives
            if alternatives:
                sequence_alternatives.append(alternatives)
            # Update token frequencies
                if penalties_enabled and frequency_penalty > 0:
                    token_id = next_token.item()
                    token_frequencies[token_id] = token_frequencies.get(token_id, 0) + 1
                if penalties_enabled and presence_penalty != 0:
                    token_id = next_token.item()
                    token_presence.add(token_id)
            # Add token to sequence
            next_token = next_token.to(device)
            current_tokens = torch.cat([current_tokens, next_token.unsqueeze(0).unsqueeze(0)], dim=1)
            
            # Stop if we generate an end token
            if next_token.item() in [tokenizer.eos_token_id, tokenizer.pad_token_id]:
                break
        
        if not sequence_alternatives:
            raise ValueError("No valid token alternatives generated")
        
        return sequence_alternatives
        
    except Exception as e:
        print(f"Error in generate_text: {str(e)}")
        raise e

@app.route('/')
def index():
    return render_template('index.html', models=MODELS.keys())

@socketio.on('connect')
def handle_connect():
    print('Client connected')

@socketio.on('disconnect')
def handle_disconnect():
    print('Client disconnected')

@socketio.on('generate')
def handle_generate(data):
    generation_id = uuid4()
    session["generating"] = generation_id
    chat_history = data.get('chat_history', [])
    model_name = data.get('model_name', 'Qwen2.5-0.5B')
    top_k = data.get('top_k', 0)
    num_show = data.get('num_show', 12)
    temperature = data.get('temperature', 0.7)
    min_p = data.get('min_p', 0)
    max_new_tokens = data.get('max_new_tokens', 100)
    top_p = data.get('top_p', 0.9)
    repetition_penalty = data.get('repetition_penalty', 1.0)
    frequency_penalty = data.get('frequency_penalty', 0.0)
    presence_penalty = data.get('presence_penalty', 0.0)
    repeat_last_n = data.get('repeat_last_n', 64)
    randomness_enabled = data.get('randomness_enabled', True)
    penalties_enabled = data.get("penalties_enabled", True)
    dry_enabled = data.get("dry_enabled", True)
    dry_allowed_length = data.get('dry_allowed_length', 1)
    dry_base = data.get('dry_base', 2)
    dry_multiplier = data.get('dry_multiplier', 3)
    dry_range = data.get('dry_range', 1024)
    
    try:
        model, tokenizer = get_model_and_tokenizer(model_name)
        
        # Format chat history
        formatted_prompt = format_chat_history(chat_history, model_name)
        n=0
        # Generate text with alternatives
        for token, alternatives in generate_text(model, tokenizer, formatted_prompt, max_new_tokens, top_k, temperature, min_p, num_show, top_p, repetition_penalty, frequency_penalty, presence_penalty, repeat_last_n, randomness_enabled, penalties_enabled, dry_enabled, dry_allowed_length, dry_base, dry_multiplier, dry_range):
            if session["generating"] != generation_id:
                return
            socketio.emit('token', {
                'chosen': token,
                'options': [[t, float(p), bool(c)] for t, p, c in alternatives],
                'message_id': len(chat_history)-1,
                "token_id": n
            })
            n+=1
            if session["generating"] != generation_id:
                return
        socketio.emit('end')
        session["generating"] = None
        
    except Exception as e:
        print(f"Error in generate: {str(e)}")
        socketio.emit('error', {'message': str(e)})
        raise e

@socketio.on("stop")
def stop():
    session["generating"] = None


if __name__ == '__main__':
    # Pre-load the default model
    get_model_and_tokenizer('Qwen2.5-0.5B')
    socketio.run(app, debug=True, port=5002)