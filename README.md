# LLM Token Explorer

## Overview
An interactive web application that allows exploration of language model token probabilities and generation.

## Features
- Server-side LLM inference
- Interactive token probability visualization
- Token replacement and re-generation

## Setup

### Prerequisites
- Python 3.8+
- pip

### Installation
1. Clone the repository
2. Create a virtual environment
```bash
python3 -m venv venv
source venv/bin/activate
```

3. Install dependencies
```bash
pip install -r requirements.txt
```

### Running the Application
```bash
python app.py
```

Open `http://localhost:5000` in your browser.

## How to Use
1. Enter a prompt in the input field
2. Click "Generate" to see the next token and its probabilities
3. Click on any token in the probability table to replace it and see how the model's output changes

## Model
Currently using GPT-2, but can be easily swapped with other Hugging Face models.

## Technologies
- Flask
- PyTorch
- Hugging Face Transformers
- JavaScript
