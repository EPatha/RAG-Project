"""Build the evaluation report PDF from results/*.csv.

  python make_report.py --results results --qa eval/qa.jsonl --team "Kelompok X" --corpus "Peraturan akademik Universitas Y" --out evaluation_report.pdf
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import Image, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

ap = argparse.ArgumentParser()
ap.add_argument("--results", default="results")
ap.add_argument("--qa", default="eval/qa.jsonl")
ap.add_argument("--team", default="Kelompok ...")
ap.add_argument("--corpus", default="(deskripsi korpus)")
ap.add_argument("--out", default="evaluation_report.pdf")
args = ap.parse_args()

R = Path(args.results)
ss = getSampleStyleSheet()
H1, H2, B = ss["Heading1"], ss["Heading2"], ParagraphStyle("b", parent=ss["BodyText"], leading=14, spaceAfter=6)
SM = ParagraphStyle("sm", parent=B, fontSize=8, leading=10)
story = []


def p(t, s=B):
    story.append(Paragraph(t, s))


def table(df, fmt="{:.3f}", font=8):
    df = df.copy()
    data = [list(df.columns)] + [[fmt.format(v) if isinstance(v, float) else str(v) for v in row]
                                 for row in df.values.tolist()]
    t = Table(data, repeatRows=1)
    t.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#2f4b7c")),
                           ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                           ("FONTSIZE", (0, 0), (-1, -1), font),
                           ("GRID", (0, 0), (-1, -1), 0.3, colors.grey),
                           ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#eef1f6")])]))
    story.append(t)
    story.append(Spacer(1, 8))


def chart(df, index, col_group, value, title, fname):
    piv = df.pivot(index=index, columns=col_group, values=value)
    ax = piv.plot(kind="bar", figsize=(6.2, 3), rot=0)
    ax.set_title(title)
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=7)
    plt.tight_layout()
    path = R / fname
    plt.savefig(path, dpi=150)
    plt.close()
    story.append(Image(str(path), width=14 * cm, height=6.8 * cm))


qa = [json.loads(l) for l in Path(args.qa).read_text(encoding="utf-8").splitlines() if l.strip()]
n_ans = sum(q.get("answerable", True) for q in qa)
rr = pd.read_csv(R / "retrieval_results.csv") if (R / "retrieval_results.csv").exists() else None
gr = pd.read_csv(R / "generation_results.csv") if (R / "generation_results.csv").exists() else None
cfg = json.loads((R / "run_config.json").read_text()) if (R / "run_config.json").exists() else {}

story.append(Paragraph("Evaluation Report: RAG System", ss["Title"]))
p(f"<b>Team:</b> {args.team}<br/><b>Corpus:</b> {args.corpus}")

p("1. System Overview", H1)
p("The prototype is an end-to-end Retrieval-Augmented Generation pipeline: documents (PDF/TXT) are parsed "
  "page by page, split into sentence-aware character chunks with overlap, indexed by a lexical retriever (BM25) "
  "and a dense retriever (multilingual-e5-small, cosine similarity), optionally fused with Reciprocal Rank "
  "Fusion (hybrid), and the top-k chunks are passed to a local LLM (Ollama) with a prompt that requires "
  "grounded answers with [n] source citations and an explicit refusal when the context lacks the answer.")

p("2. Evaluation Methodology", H1)
p("<b>2.1 Test set.</b> "
  f"{len(qa)} questions ({n_ans} answerable, {len(qa) - n_ans} unanswerable). Each answerable item has a reference "
  "answer and a <i>gold snippet</i>, a verbatim passage from the corpus that supports it. Candidate questions were "
  "drafted with an LLM from randomly sampled chunks and then reviewed and corrected manually by the team. "
  "Unanswerable questions probe whether the system refuses instead of hallucinating.")
p("<b>2.2 Relevance judgement.</b> A retrieved chunk is relevant if it contains at least 80% of the unique tokens "
  "of the gold snippet. Because this does not depend on chunk boundaries, different chunk sizes are compared fairly.")
p("<b>2.3 Metrics.</b>")
table(pd.DataFrame([
    ["Hit@k (k=1,3,5)", "Retrieval", "Fraction of questions with at least one relevant chunk in the top k"],
    ["MRR@10", "Retrieval", "Mean reciprocal rank of the first relevant chunk"],
    ["nDCG@10", "Retrieval", "Rank-discounted gain with binary relevance"],
    ["Precision@5", "Retrieval", "Share of relevant chunks in the top 5"],
    ["Token F1 / Recall / ROUGE-L", "Generation", "Overlap between generated and reference answer"],
    ["Lexical faithfulness", "Generation", "Share of answer content tokens present in retrieved context"],
    ["Refusal accuracy", "Generation", "Unanswerable questions correctly refused"],
    ["False-refusal rate", "Generation", "Answerable questions wrongly refused"],
    ["LLM-judge correctness / faithfulness", "Generation", "1-5 scores from a local LLM judge (if enabled)"],
    ["Latency", "Efficiency", "Mean and p95 seconds per query"],
], columns=["Metric", "Stage", "Definition"]), font=7)
p("<b>2.4 Experimental procedure.</b> (a) Retrieval ablation without an LLM: retriever (BM25, dense, hybrid"
  + (", hybrid+rerank" if rr is not None and rr.retriever.astype(str).str.contains("rerank").any() else "")
  + ") x chunk size, retrieving the top 10 chunks per question. (b) End-to-end ablation: no-retrieval baseline "
  "(LLM only) versus RAG with each retriever at a fixed chunk size and top-k. All runs use temperature 0 so they "
  "are repeatable. Retrieval latency is measured on the same machine for all configurations.")
if cfg:
    p(f"<b>Run settings:</b> LLM = {cfg.get('model')} ({cfg.get('llm')}), chunk size = {cfg.get('chunk_size')}, "
      f"top-k = {cfg.get('top_k')}, judge = {cfg.get('judge')}.", SM)

p("3. Results", H1)
if rr is not None and len(rr):
    p("3.1 Retrieval ablation", H2)
    cols = ["retriever", "chunk_size", "n_chunks", "hit@1", "hit@3", "hit@5", "mrr@10", "ndcg@10", "precision@5", "latency_ms"]
    table(rr[cols].round(3))
    chart(rr, "chunk_size", "retriever", "mrr@10", "MRR@10 by chunk size and retriever", "fig_mrr.png")
    best = rr.sort_values(["mrr@10", "hit@3"], ascending=False).iloc[0]
    p(f"The best retrieval configuration was <b>{best.retriever}</b> with chunk size <b>{int(best.chunk_size)}</b> "
      f"(MRR@10 = {best['mrr@10']:.3f}, Hit@3 = {best['hit@3']:.3f}).")
    by_ret = rr.groupby("retriever")["mrr@10"].mean().round(3).to_dict()
    by_cs = rr.groupby("chunk_size")["mrr@10"].mean().round(3).to_dict()
    p(f"Averaged over chunk sizes, MRR@10 per retriever: {by_ret}. Averaged over retrievers, MRR@10 per chunk size: {by_cs}.")
if gr is not None and len(gr):
    p("3.2 End-to-end generation", H2)
    keep = [c for c in ["retriever", "f1", "recall", "rougeL", "faithfulness_lex", "refusal_acc",
                        "false_refusal", "judge_correct", "judge_faith", "latency_mean_s", "latency_p95_s"] if c in gr]
    table(gr[keep].round(3))
    mm = [c for c in ["f1", "recall", "faithfulness_lex", "refusal_acc"] if c in gr]
    long = gr.set_index("retriever")[mm].T
    ax = long.plot(kind="bar", figsize=(6.2, 3), rot=0)
    ax.set_ylim(0, 1.05); ax.set_title("Generation quality by retriever"); ax.legend(fontsize=7)
    plt.tight_layout(); plt.savefig(R / "fig_gen.png", dpi=150); plt.close()
    story.append(Image(str(R / "fig_gen.png"), width=14 * cm, height=6.8 * cm))
    if "none" in set(gr.retriever) and len(gr) > 1:
        base = gr[gr.retriever == "none"].iloc[0]
        rag = gr[gr.retriever != "none"].sort_values("recall", ascending=False).iloc[0]
        p(f"Compared with the no-retrieval baseline, the best RAG configuration ({rag.retriever}) changed token recall "
          f"from {base.recall:.3f} to {rag.recall:.3f} and refusal accuracy from {base.refusal_acc:.3f} to {rag.refusal_acc:.3f}.")

    p("3.3 Error analysis", H2)
    worst = []
    for f in sorted(R.glob("per_question_*.csv")):
        d = pd.read_csv(f)
        d = d[d.answerable == True]
        if "recall" in d:
            for _, r in d.nsmallest(3, "recall").iterrows():
                worst.append([f.stem.replace("per_question_", ""), r["id"], f'{r["recall"]:.2f}', str(r["answer"])[:110]])
    if worst:
        table(pd.DataFrame(worst, columns=["Config", "Question", "Recall", "Answer (truncated)"]), font=7)
    p("<i>Team note: inspect the lowest-scoring questions above and categorise failures "
      "(retrieval miss, chunk split across the answer, wrong refusal, hallucination, verbose answer) in this section.</i>")

p("4. Limitations and Threats to Validity", H1)
p("The test set is small and was partly LLM-drafted, so scores carry sampling noise; n-gram metrics under-reward "
  "correct paraphrases; the local LLM judge is itself a weak model and may be biased; the gold-snippet relevance "
  "rule can miss answers that are spread over several passages. Results should be read as relative comparisons "
  "between configurations rather than absolute quality.")
p("5. Conclusion", H1)
p("<i>Team note: summarise which configuration you recommend and why, citing the tables above.</i>")

SimpleDocTemplate(args.out, pagesize=A4, leftMargin=2 * cm, rightMargin=2 * cm,
                  topMargin=2 * cm, bottomMargin=2 * cm).build(story)
print("wrote", args.out)
