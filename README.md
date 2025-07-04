# OCR-LLM: Image Text Extraction Scripts

Extract text tables from images using LLM models([stepfun-ai/GOT-OCR-2.0-hf](https://huggingface.co/stepfun-ai/GOT-OCR-2.0-hf)) or LLM API(gemini-2.0-flash-exp).

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
python got_ocr_2_0.py --image ./download_images_test/20250625184111_2608.jpg --output result.md

# OCR Using Gemini API
python gemini_batch_ocr.py --image ./download_images_test/20250625184111_2608.jpg --output result.md

# Process entire directory
python got_ocr_2_0.py --directory ./download_images_test --output result.md
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