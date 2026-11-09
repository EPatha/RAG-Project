# RAG over University Regulations: Prototype + Evaluation

Local, open-source RAG pipeline (BM25 / dense / hybrid retrieval, Ollama LLM) with a reproducible evaluation harness.

## Materi pembelajaran dan assessment

Materi Generative AI, LLM, OpenRouter, prompt engineering, diffusion,
multimodal, RAG, dan evaluasi tersedia di
[`docs/MATERI_GENERATIVE_AI.md`](docs/MATERI_GENERATIVE_AI.md).
Paket assessment berisi 50 soal dan 2 essay tersedia di
[`eval/assessment_50_2_essay.md`](eval/assessment_50_2_essay.md).

## Layout
```
rag/            ingest.py (PDF/TXT -> chunks), retrievers.py, generators.py, pipeline.py, utils.py
app.py          Gradio demo
make_qa.py      drafts a QA set from the corpus with an LLM (you must review it by hand)
evaluate.py     retrieval ablation + end-to-end generation evaluation
make_report.py  builds evaluation_report.pdf from results/*.csv
data/docs/      PUT YOUR REAL PDFs HERE (data/sample/ is a tiny FICTIONAL corpus for smoke tests)
eval/qa.jsonl   your reviewed QA set (eval/qa_sample.jsonl shows the format)
```

## Setup
```bash
pip install -r requirements.txt
# install Ollama (https://ollama.com), then:
ollama pull qwen2.5:3b        # generator (fast); qwen2.5:7b recommended for make_qa.py / judging
```

## Workflow
```bash
# 1. add documents to data/docs/
# 2. draft questions, then EDIT eval/qa_draft.jsonl -> eval/qa.jsonl
#    (target 50+ questions, add ~10-20% with "answerable": false)
python make_qa.py --docs data/docs --n 60 --model qwen2.5:7b

# 3. retrieval ablation (no LLM needed, fast)
python evaluate.py --docs data/docs --qa eval/qa.jsonl --retrieval-grid --with-rerank

# 4. end-to-end evaluation incl. no-retrieval baseline and LLM judge
python evaluate.py --docs data/docs --qa eval/qa.jsonl --generate \
    --retrievers none bm25 dense hybrid --chunk-size 800 --top-k 4 --judge

# 5. report
python make_report.py --team "Kelompok X" --corpus "Peraturan akademik Univ. Y" --qa eval/qa.jsonl

# 6. demo dengan Ollama lokal
python app.py --docs data/docs --retriever hybrid

# 7. demo dengan OpenRouter (set OPENROUTER_API_KEY terlebih dahulu)
python app.py --docs data/docs --retriever hybrid --llm openrouter \
    --openrouter-model openai/gpt-4o-mini
```
Smoke test without models: `python evaluate.py --docs data/sample --qa eval/qa_sample.jsonl --generate --llm extractive --retrievers bm25`

## 5-minute demo script
1. (0:00-0:45) Architecture in one slide: ingest -> chunk -> BM25 + dense -> RRF -> LLM -> cited answer.
2. (0:45-2:15) Live app: one easy factual question, one multi-passage question, show the source panel.
3. (2:15-3:00) An unanswerable question: the system refuses instead of hallucinating.
4. (3:00-4:30) Evaluation: show the retrieval table and the RAG vs no-retrieval chart from the PDF; state the best configuration.
5. (4:30-5:00) Limitations and next steps.

## Notes
- Dense/hybrid use `intfloat/multilingual-e5-small` (downloaded on first run) and cached embeddings in `.cache/`.
- Relevance for retrieval metrics = chunk contains >= 80% of the gold snippet tokens, so chunk sizes are comparable.
- Fix `--seed`/temperature 0 are already set, so results are repeatable.
