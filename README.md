# LLM Probability Visualization

## Setup

First, clone into it and install dependencies, Flask, Llama.cpp, flask_socketio, uuid, and PyTorch.

Then, in `app.py` add the models you want to use to the `MODELS` dictionary.

Finally, run the app with `python app.py`.

## TODO: write a proper readme and clean up the project

## TODOs

- Prevent Harmony from restarting with reasoning step after token change
- Allow sending messages as any role (with defaults available depending on chat format).
- Allow editing messages.
- See chat format in completions tab.
