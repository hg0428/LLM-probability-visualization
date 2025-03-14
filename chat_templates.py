import datetime
import random
from typing import Callable
import re


chatter_message_header_pattern = re.compile(r"\n.+\ at\ \d{4}-\d{2}-\d{2}T")
chatter_message_header_pattern_partial = re.compile(
    r"\n[^\n]*(?: at(?: \d{4}(-\d{2}){0,2})?)?$"
)


def format_chatter_timestamp(dt, offset_hours=5):
    if not isinstance(dt, datetime.datetime):
        dt = datetime.datetime.fromisoformat(dt)
    # Convert to target timezone
    target_tz = datetime.timezone(datetime.timedelta(hours=offset_hours))
    dt_local = dt.astimezone(target_tz)

    # Format output
    return (
        dt_local.strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3]
        + dt_local.strftime("%z")[:3]
        + ":"
        + dt_local.strftime("%z")[3:]
    )


def check_end_of_message(
    next_token_id, content, model, model_format, allow_name=None
) -> bool | None:  # None means maybe
    if model_format == "chatter":
        match = re.search(chatter_message_header_pattern, content)
        if match and not re.search(
            rf"\n{allow_name}(?: at(?: \d{4}(-\d{2}){0,2})?)?$", content
        ):
            return True
        elif re.search(chatter_message_header_pattern_partial, content):
            return None
        return False
        #     content = content[: match.start()]
    else:
        return next_token_id in [
            model.eos_token_id,
            model.pad_token_id,
        ]


def format_chat_history(
    messages,
    model,
    model_name="Qwen2.5-0.5B",
    model_family="qwen2.5",
    model_format="chatml",
    user_role_name="user",
    assistant_role_name="assistant",
    system_role_name="system",
):
    """Format chat history according to model's template."""
    print(f"Formatting chat history for model: {model_name}")
    print(f"Input messages: {messages}")

    if model_format == "chatml":
        formatted = format_chatml_chat(
            messages, user_role_name, assistant_role_name, system_role_name
        )
    elif model_format == "gpt2":
        formatted = format_gpt2_chat(messages)
    elif model_format == "chatter":
        formatted = format_chatter_chat(
            messages,
            user_name=user_role_name,
            assistant_name=assistant_role_name,
            system_name=system_role_name,
        )
    elif model_format == "llama3":
        formatted = format_llama3_chat(
            messages, user_role_name, assistant_role_name, system_role_name
        )
    else:
        formatted = format_default_chat(messages)

    print(f"Formatted chat history: {formatted}")
    return formatted


def format_chatml_chat(messages, user_role_name, assistant_role_name, system_role_name):
    """Format chat history for Qwen models."""
    formatted = ""
    for msg in messages:
        role = msg["role"]
        content = msg["content"]
        if role == "system":
            formatted += f"<|im_start|>{system_role_name}\n{content}"
        elif role == "user":
            formatted += f"<|im_start|>{user_role_name}\n{content}"
        elif role == "assistant":
            formatted += f"<|im_start|>{assistant_role_name}\n{content}"
        if not msg.get("partial", None):
            formatted += "<|im_end|>\n"
    return formatted


def format_gpt2_chat(messages):
    """Format chat history for GPT-2."""
    formatted = ""
    for msg in messages:
        role = msg["role"]
        content = msg["content"]
        if role == "system":
            formatted += f"System: {content}"
        elif role == "user":
            formatted += f"User: {content}"
        elif role == "assistant":
            formatted += f"Assistant: {content}"
        if not msg.get("partial", None):
            formatted += "\n"
    return formatted


def format_llama3_chat(messages, user_role_name, assistant_role_name, system_role_name):
    """Format chat history for Qwen models."""
    formatted = ""
    for msg in messages:
        role = msg["role"]
        content = msg["content"]
        if role == "system":
            formatted += (
                f"<|start_header_id|>{system_role_name}<|end_header_id|>\n\n{content}"
            )
        elif role == "user":
            formatted += (
                f"<|start_header_id|>{user_role_name}<|end_header_id|>\n\n{content}"
            )
        elif role == "assistant":
            formatted += f"<|start_header_id|>{assistant_role_name}<|end_header_id|>\n\n{content}"
        if not msg.get("partial", None):
            formatted += "<|eot_id|>\n"
    return formatted


def format_chatter_chat(
    messages,
    start_time=datetime.datetime.now(),
    offset_hours=5,
    user_name="User",
    assistant_name="Bot",
    system_name="System",
):
    formatted = ""
    offset_seconds = 0
    for msg in messages:
        role = msg["role"]
        content = msg["content"]
        if formatted != "":
            formatted += "\n"
        if role == "system":
            name = system_name
        elif role == "user":
            name = user_name
        elif role == "assistant":
            name = assistant_name
        time = msg.get("time", start_time + datetime.timedelta(seconds=offset_seconds))
        print(time)
        formatted += (
            f"{name} at {format_chatter_timestamp(time, offset_hours)}:\n{content}"
        )
        if not msg.get("partial", None):
            formatted += "\n"
        offset_seconds += random.randint(0, 30)
    return formatted


def format_default_chat(messages):
    """Default chat formatting."""
    formatted = ""
    for msg in messages:
        role = msg["role"]
        content = msg["content"]
        formatted += f"{role.capitalize()}: {content}"
        if not msg.get("partial", None):
            formatted += "\n"
    return formatted
