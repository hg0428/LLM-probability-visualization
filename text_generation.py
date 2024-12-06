import torch
import torch.quantization
import string
from chat_templates import format_chat_history
from itertools import count
from uuid import uuid4
import random

sequence_breaker_strings = ["\n", ":", '"', "*"]


def calculate_dry_penalty(
    input_ids: torch.LongTensor,
    scores: torch.FloatTensor,
    _range: int = 1024,
    sequence_breakers=None,
    allowed_length: int = 1,
    base: float = 2,
    multiplier: float = 3,
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
                scores_row[token] -= torch.tensor(
                    penalty, dtype=scores_row.dtype, device=scores_row.device
                )

    return scores


def generate_text(
    model,
    prompt,
    max_new_tokens=0,
    top_k=50,
    temperature=0.7,
    min_p=0,
    num_show=12,
    top_p=0.9,
    repetition_penalty=1.0,
    frequency_penalty=0.0,
    presence_penalty=0.0,
    repeat_last_n=64,
    truncation_enabled=True,
    randomness_enabled=True,
    penalties_enabled=True,
    dry_enabled=True,
    xtc_enabled=True,
    dry_allowed_length=2,
    dry_base=1.2,
    dry_multiplier=2,
    dry_range=512,
    xtc_threshold=0.1,
    xtc_probability=0.5,
):
    """Generate text and return token alternatives for each position."""
    try:
        if dry_enabled:
            sequence_breakers = {
                model.tokenize(s)[-1] for s in sequence_breaker_strings
            }
        # Tokenize input
        input_ids = model.tokenize(prompt)
        device = model.device
        sequence_alternatives = []
        current_tokens = input_ids

        # Track token frequencies and presence for penalties
        token_frequencies = {}
        token_presence = set()

        # Generate one token at a time
        for i in range(max_new_tokens):
            # Get model outputs and probabilities
            with torch.inference_mode():
                logits = model(current_tokens)

                # Get original probabilities for displaying alternatives
                orig_probs = torch.softmax(logits, dim=-1)
                num_show_probs, num_show_indices = torch.topk(
                    orig_probs[0], k=min(num_show, len(orig_probs[0]))
                )
                del orig_probs

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
                        multiplier=dry_multiplier,
                    )

                # Now apply sampling modifications
                if randomness_enabled and temperature > 1e-7 and temperature != 1.0:
                    logits = logits / temperature
                probs = torch.softmax(logits, dim=-1)
                del logits

                if truncation_enabled:
                    # Apply top-p (nucleus) sampling
                    if 0 < top_p < 1.0:
                        sorted_probs, sorted_indices = torch.sort(
                            probs[0], descending=True
                        )
                        cumsum_probs = torch.cumsum(sorted_probs, dim=-1)
                        # Remove tokens with cumulative probability above the threshold
                        sorted_indices_to_remove = cumsum_probs > top_p
                        # Shift the indices to the right to keep also the first token above the threshold
                        sorted_indices_to_remove[1:] = sorted_indices_to_remove[
                            :-1
                        ].clone()
                        sorted_indices_to_remove[0] = False
                        indices_to_remove = sorted_indices[sorted_indices_to_remove]
                        probs[0, indices_to_remove] = 0.0
                        # Renormalize
                        probs = probs / probs.sum(dim=-1, keepdim=True)

                    # Get top k probabilities and tokens
                    if top_k > 0:
                        top_k_values, top_k_indices = torch.topk(
                            probs[0], k=min(top_k, len(probs[0]))
                        )
                        probs[0] = torch.zeros_like(probs[0])
                        probs[0, top_k_indices] = top_k_values

                    if min_p > 0:
                        probs = torch.where(
                            probs > min_p,
                            probs,
                            torch.tensor(0.0, device=device),
                        )
                    if probs.sum() == 0:
                        probs = torch.ones_like(probs) / len(probs)
                if xtc_enabled and random.random() < xtc_probability:
                    sorted_probs, sorted_indices = torch.sort(probs, descending=True)
                    sorted_indices_to_remove = torch.full_like(
                        sorted_probs, False, dtype=torch.bool
                    )

                    # Mark indices where next probability meets threshold
                    sorted_indices_to_remove[..., :-1] = (
                        sorted_probs[..., 1:] >= xtc_threshold
                    )

                    # Convert back to original indices
                    indices_to_remove = sorted_indices_to_remove.scatter(
                        1, sorted_indices, sorted_indices_to_remove
                    )

                    # Zero out removed probabilities and renormalize
                    filtered_probs = probs.masked_fill(indices_to_remove, 0.0)

                    # Renormalize remaining probabilities
                    norm_factor = filtered_probs.sum(dim=-1, keepdim=True)
                    probs = filtered_probs / norm_factor
                if randomness_enabled:
                    # Ensure probabilities are valid for multinomial sampling
                    probs = probs / probs.sum()  # Renormalize
                    next_token = torch.multinomial(probs, num_samples=1)[0][0]
                else:
                    # When randomness is disabled, just pick the most likely token
                    next_token = torch.argmax(probs[0])

            # Record alternatives including the chosen token
            alternatives = []
            for prob, idx in zip(num_show_probs, num_show_indices):
                token = model.detokenize([idx.item()])
                alternatives.append(
                    [token, float(prob.item()), idx.item() == next_token.item()]
                )
            chosen_token_detokenized = model.detokenize([next_token.item()])
            alternatives.append(
                [chosen_token_detokenized, probs[0][next_token.item()], True]
            )
            # Remove duplicates
            alternatives = [
                alt
                for i, alt in enumerate(alternatives)
                if alt[0] not in [a[0] for a in alternatives[:i]]
            ]
            yield chosen_token_detokenized, alternatives
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
            current_tokens = torch.cat(
                [current_tokens, next_token.unsqueeze(0).unsqueeze(0)], dim=1
            )

            # Stop if we generate an end token
            if next_token.item() in [
                model.eos_token_id,
                model.pad_token_id,
            ]:
                break

        if not sequence_alternatives:
            raise ValueError("No valid token alternatives generated")

        return sequence_alternatives

    except Exception as e:
        print(f"Error in generate_text: {str(e)}")
        raise e
