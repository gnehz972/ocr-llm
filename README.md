# OCR-LLM: Advanced Image Text Extraction

Extract text tales from images using LLM models([stepfun-ai/GOT-OCR-2.0-hf](https://huggingface.co/stepfun-ai/GOT-OCR-2.0-hf)) or LLM API(Gemini).

## Quick Start

```bash
# Install dependencies
uv sync

# Activate virtual environment
source .venv/bin/activate  # Linux/Mac
# or on Windows: .venv\Scripts\activate

# Set up environment (for Gemini API)
cp .env.example .env
# Edit .env and add your GEMINI_API_KEY=your_api_key_here

# OCR with local LLM model GOT-OCR-2.0
python got_ocr_2_0.py --image document.jpg --output result.txt

# OCR Using Gemini API
python gemini_batch_ocr.py --image document.jpg --output result.txt

# Process entire directory
python got_ocr_2_0.py --di  vc b nrectory ./images --output merged_tables.txt
```

## Environment

For Gemini API usage, create a `.env` file:

```bash
cp .env.example .env
```

Then edit `.env` and add your API key:
```
GEMINI_API_KEY=your_actual_api_key_here
```

Get your Gemini API key from [Google AI Studio](https://aistudio.google.com/app/apikey).

## Requirements

- Python 3.11+
- CPU or GPU (recommended for faster processing for running local LLMs)
- Gemini API key (for `gemini_batch_ocr.py` script)