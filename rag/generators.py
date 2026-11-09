"""Answer generators: local LLM via Ollama, plus a model-free extractive baseline."""
import re

import requests

from .utils import REFUSAL, content_tokens, tokenize

SYSTEM_RAG = (
    "Anda adalah asisten yang menjawab pertanyaan tentang peraturan universitas. "
    "Jawab HANYA berdasarkan konteks yang diberikan. Sebutkan sumber dengan format [nomor]. "
    f"Jika jawabannya tidak ada pada konteks, jawab persis: '{REFUSAL}' "
    "Jawab ringkas dalam Bahasa Indonesia."
)
SYSTEM_NO_RAG = "Jawab pertanyaan berikut secara ringkas dalam Bahasa Indonesia."


def format_context(contexts):
    return "\n\n".join(f"[{i+1}] ({c['doc']}, hlm. {c['page']}) {c['text']}" for i, c in enumerate(contexts))


class OllamaGenerator:
    def __init__(self, model="qwen2.5:3b", url="http://localhost:11434", temperature=0.0, timeout=180):
        self.model, self.url, self.temperature, self.timeout = model, url.rstrip("/"), temperature, timeout

    def chat(self, system, user, json_mode=False):
        payload = {
            "model": self.model, "stream": False,
            "options": {"temperature": self.temperature},
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
        }
        if json_mode:
            payload["format"] = "json"
        r = requests.post(f"{self.url}/api/chat", json=payload, timeout=self.timeout)
        r.raise_for_status()
        return r.json()["message"]["content"].strip()

    def answer(self, question, contexts):
        if contexts:
            user = f"Konteks:\n{format_context(contexts)}\n\nPertanyaan: {question}\nJawaban:"
            return self.chat(SYSTEM_RAG, user)
        return self.chat(SYSTEM_NO_RAG, question)


class ExtractiveGenerator:
    """No-LLM baseline / test double: returns the retrieved sentence most similar to the question."""

    def __init__(self, min_overlap=0.5):
        self.min_overlap = min_overlap

    def answer(self, question, contexts):
        q = set(content_tokens(question))
        best, best_s = None, 0.0
        for i, c in enumerate(contexts):
            for s in re.split(r"(?<=[.!?])\s+", c["text"]):
                st = set(content_tokens(s))
                if not st or not q:
                    continue
                score = len(q & st) / len(q)
                if score > best_s:
                    best, best_s = (s, i + 1), score
        if best is None or best_s < self.min_overlap:
            return REFUSAL
        return f"{best[0]} [{best[1]}]"
