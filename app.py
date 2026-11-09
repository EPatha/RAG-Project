"""Gradio demo: ask questions, see the answer and the retrieved sources.

Run:  python app.py --retriever hybrid --llm ollama
"""
import argparse

import gradio as gr

from rag.pipeline import Config, RAGPipeline

ap = argparse.ArgumentParser()
ap.add_argument("--docs", default="data/docs")
ap.add_argument("--retriever", default="hybrid")
ap.add_argument("--llm", default="ollama", choices=["ollama", "openrouter", "extractive"])
ap.add_argument("--model", default="qwen2.5:3b")
ap.add_argument("--openrouter-model", default="openai/gpt-4o-mini")
ap.add_argument("--chunk-size", type=int, default=800)
ap.add_argument("--top-k", type=int, default=4)
ap.add_argument("--port", type=int, default=7860)
args = ap.parse_args()

pipe = RAGPipeline(Config(docs_dir=args.docs, retriever=args.retriever, llm=args.llm,
                          ollama_model=args.model, openrouter_model=args.openrouter_model,
                          chunk_size=args.chunk_size, top_k=args.top_k))


def ask(question):
    if not question.strip():
        return "", ""
    r = pipe.answer(question)
    src = "\n\n".join(
        f"**[{i+1}] {c['doc']} (hlm. {c['page']})** — score {s:.3f}\n> {c['text']}"
        for i, (c, s) in enumerate(zip(r["contexts"], r["scores"])))
    meta = f"\n\n*retrieval {r['retrieval_s']*1000:.0f} ms · generation {r['generation_s']:.1f} s*"
    return r["answer"] + meta, src


with gr.Blocks(title="RAG Peraturan Akademik") as demo:
    gr.Markdown("# Tanya Peraturan Akademik (RAG)\n"
                f"Retriever: `{args.retriever}` · LLM: `{args.llm}` · {len(pipe.chunks)} chunks")
    q = gr.Textbox(label="Pertanyaan", placeholder="Berapa persen kehadiran minimal untuk ikut UAS?")
    btn = gr.Button("Tanya", variant="primary")
    out = gr.Markdown(label="Jawaban")
    srcs = gr.Markdown(label="Sumber")
    gr.Examples(["Berapa persen kehadiran minimal untuk mengikuti ujian akhir?",
                 "Apa syarat IPK untuk beasiswa prestasi?",
                 "Berapa biaya parkir kendaraan di kampus?"], inputs=q)
    btn.click(ask, q, [out, srcs])
    q.submit(ask, q, [out, srcs])

demo.launch(server_port=args.port)
