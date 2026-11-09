"""Optional OpenRouter vision adapter example.

This demonstrates the multimodal request shape without changing the text RAG
pipeline. The image must be supplied as a data URL or a public URL.
"""
import argparse
import base64
import mimetypes
import os
from pathlib import Path

import requests


def image_data_url(path):
    mime = mimetypes.guess_type(path)[0] or "image/png"
    encoded = base64.b64encode(Path(path).read_bytes()).decode()
    return f"data:{mime};base64,{encoded}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("image")
    parser.add_argument("question")
    parser.add_argument("--model", default="openai/gpt-4o-mini")
    args = parser.parse_args()
    key = os.getenv("OPENROUTER_API_KEY")
    if not key:
        raise SystemExit("Set OPENROUTER_API_KEY before running this example")
    payload = {
        "model": args.model,
        "messages": [{"role": "user", "content": [
            {"type": "text", "text": args.question},
            {"type": "image_url", "image_url": {"url": image_data_url(args.image)}},
        ]}],
    }
    response = requests.post(
        "https://openrouter.ai/api/v1/chat/completions",
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
        json=payload,
        timeout=180,
    )
    response.raise_for_status()
    print(response.json()["choices"][0]["message"]["content"])


if __name__ == "__main__":
    main()
