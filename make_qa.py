"""Draft an evaluation QA set from the corpus using the local LLM.

The LLM proposes a question, a reference answer and a VERBATIM supporting quote for randomly
sampled chunks. Quotes that are not found verbatim in the chunk are discarded. The output is a
DRAFT: the team must review/edit every item by hand before using it as ground truth
(also add ~10-20% unanswerable questions with "answerable": false).

  python make_qa.py --docs data/docs --n 60 --out eval/qa_draft.jsonl
"""
import argparse
import json
import random

from rag.generators import OllamaGenerator
from rag.ingest import build_chunks

SYSTEM = (
    "Dari paragraf peraturan berikut, buat SATU pertanyaan faktual yang jawabannya terdapat pada "
    "paragraf. Kembalikan JSON: {\"question\": ..., \"reference_answer\": ..., \"gold_snippet\": ...}. "
    "gold_snippet HARUS berupa kutipan persis (verbatim) dari paragraf yang memuat jawabannya. "
    "Pertanyaan harus dapat dipahami tanpa melihat paragraf. Gunakan Bahasa Indonesia."
)

ap = argparse.ArgumentParser()
ap.add_argument("--docs", default="data/docs")
ap.add_argument("--n", type=int, default=60)
ap.add_argument("--model", default="qwen2.5:7b")
ap.add_argument("--out", default="eval/qa_draft.jsonl")
ap.add_argument("--seed", type=int, default=42)
args = ap.parse_args()

random.seed(args.seed)
chunks = [c for c in build_chunks(args.docs, 800, 100) if len(c["text"]) > 150]
random.shuffle(chunks)
llm = OllamaGenerator(args.model)
n_ok = 0
with open(args.out, "w", encoding="utf-8") as f:
    for c in chunks:
        if n_ok >= args.n:
            break
        try:
            d = json.loads(llm.chat(SYSTEM, c["text"], json_mode=True))
            if d["gold_snippet"] not in c["text"]:
                continue
        except Exception:
            continue
        n_ok += 1
        f.write(json.dumps({"id": f"q{n_ok:03d}", "question": d["question"],
                            "reference_answer": d["reference_answer"],
                            "gold_snippet": d["gold_snippet"], "answerable": True,
                            "source": c["id"]}, ensure_ascii=False) + "\n")
print(f"wrote {n_ok} draft questions to {args.out} -- REVIEW THEM MANUALLY")
