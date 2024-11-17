def format_chat_history(messages, model_name="Qwen2.5-0.5B"):
    """Format chat history according to model's template."""
    print(f"Formatting chat history for model: {model_name}")
    print(f"Input messages: {messages}")
    
    if model_name.startswith("Qwen"):
        formatted = format_qwen_chat(messages)
    elif model_name.startswith("gpt2"):
        formatted = format_gpt2_chat(messages)
    else:
        formatted = format_default_chat(messages)
    
    print(f"Formatted chat history: {formatted}")
    return formatted

def format_qwen_chat(messages):
    """Format chat history for Qwen models."""
    formatted = ""
    for msg in messages:
        role = msg["role"]
        content = msg["content"]
        if role == "system":
            formatted += f"<|im_start|>system\n{content}"
        elif role == "user":
            formatted += f"<|im_start|>user\n{content}"
        elif role == "assistant":
            formatted += f"<|im_start|>assistant\n{content}"
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
