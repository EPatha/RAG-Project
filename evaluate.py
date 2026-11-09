"""Evaluation harness.

Retrieval metrics (answerable questions only):  Hit@1/3/5, MRR@10, nDCG@10, Precision@5
Generation metrics: token F1, token recall, ROUGE-L, lexical faithfulness,
                    refusal accuracy (unanswerable), false-refusal rate (answerable),
                    optional LLM-as-judge (correctness / faithfulness, 1-5), latency.

A retrieved chunk counts as relevant when it contains >= 80% of the unique tokens of the
question's gold_snippet. This is independent of chunk size, so different chunking
configurations can be compared fairly.

Examples
  python evaluate.py --docs data/docs --qa eval/qa.jsonl --retrieval-grid
  python evaluate.py --docs data/docs --qa eval/qa.jsonl --generate --retrievers none bm25 hybrid --judge
"""
import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd

from rag.pipeline import Config, RAGPipeline
from rag.generators import OllamaGenerator
from rag.utils import content_tokens, is_refusal, tokenize

# ---------------------------------------------------------------- relevance


def is_relevant(chunk, gold):
    g = set(tokenize(gold))
    if not g:
        return False
    return len(g & set(tokenize(chunk["text"]))) / len(g) >= 0.8


# ---------------------------------------------------------------- retrieval metrics


def retrieval_metrics(rels, n_rel):
    out = {f"hit@{k}": float(any(rels[:k])) for k in (1, 3, 5)}
    out["precision@5"] = sum(rels[:5]) / 5
    rr = 0.0
    for i, r in enumerate(rels[:10]):
        if r:
            rr = 1.0 / (i + 1)
            break
    out["mrr@10"] = rr
    dcg = sum(r / math.log2(i + 2) for i, r in enumerate(rels[:10]))
    ideal = sum(1 / math.log2(i + 2) for i in range(min(max(n_rel, 1), 10)))
    out["ndcg@10"] = dcg / ideal
    return out


# ---------------------------------------------------------------- generation metrics


def token_f1_recall(pred, ref):
    p, r = tokenize(pred), tokenize(ref)
    if not p or not r:
        return 0.0, 0.0
    common = {}
    rc = {}
    for t in r:
        rc[t] = rc.get(t, 0) + 1
    overlap = 0
    for t in p:
        if rc.get(t, 0) > 0:
            rc[t] -= 1
            overlap += 1
    if overlap == 0:
        return 0.0, 0.0
    prec, rec = overlap / len(p), overlap / len(r)
    return 2 * prec * rec / (prec + rec), rec


def rouge_l(pred, ref):
    a, b = tokenize(pred), tokenize(ref)
    if not a or not b:
        return 0.0
    dp = [[0] * (len(b) + 1) for _ in range(len(a) + 1)]
    for i in range(len(a)):
        for j in range(len(b)):
            dp[i + 1][j + 1] = dp[i][j] + 1 if a[i] == b[j] else max(dp[i][j + 1], dp[i + 1][j])
    lcs = dp[-1][-1]
    if lcs == 0:
        return 0.0
    p, r = lcs / len(a), lcs / len(b)
    return 2 * p * r / (p + r)


def lexical_faithfulness(answer, contexts):
    """Mean over answer sentences of the fraction of content tokens found in the retrieved context."""
    import re
    ctx = set()
    for c in contexts:
        ctx |= set(content_tokens(c["text"]))
    scores = []
    for s in re.split(r"(?<=[.!?])\s+", re.sub(r"\[\d+\]", "", answer)):
        toks = content_tokens(s)
        if toks:
            scores.append(sum(t in ctx for t in toks) / len(toks))
    return float(np.mean(scores)) if scores else 0.0


JUDGE_SYSTEM = (
    "You are a strict evaluator of a question-answering system. Return JSON only: "
    '{"correctness": <1-5>, "faithfulness": <1-5>}. correctness: does the answer agree with the '
    "reference answer (5 = fully, 1 = wrong). faithfulness: is every claim in the answer supported "
    "by the context (5 = fully supported, 1 = mostly unsupported)."
)


def llm_judge(judge, question, reference, answer, contexts):
    ctx = "\n".join(c["text"] for c in contexts) or "(no context)"
    user = f"Question: {question}\nReference answer: {reference}\nContext:\n{ctx}\nSystem answer: {answer}"
    try:
        d = json.loads(judge.chat(JUDGE_SYSTEM, user, json_mode=True))
        return float(d["correctness"]), float(d["faithfulness"])
    except Exception:
        return float("nan"), float("nan")


# ---------------------------------------------------------------- runners


def load_qa(path):
    return [json.loads(l) for l in Path(path).read_text(encoding="utf-8").splitlines() if l.strip()]


def eval_retrieval(pipe, qa):
    rows = []
    for q in qa:
        if not q.get("answerable", True):
            continue
        hits = pipe.retrieve(q["question"], n=10)
        rels = [int(is_relevant(c, q["gold_snippet"])) for c, _ in hits]
        n_rel = sum(is_relevant(c, q["gold_snippet"]) for c in pipe.chunks)
        rows.append({"id": q["id"], **retrieval_metrics(rels, n_rel)})
    return pd.DataFrame(rows)


def eval_generation(pipe, qa, judge=None):
    rows = []
    for q in qa:
        r = pipe.answer(q["question"])
        ans, ans_ok = r["answer"], q.get("answerable", True)
        row = {"id": q["id"], "answerable": ans_ok, "answer": ans,
               "refused": is_refusal(ans), "latency_s": r["total_s"]}
        if ans_ok:
            f1, rec = token_f1_recall(ans, q["reference_answer"])
            row.update(f1=f1, recall=rec, rougeL=rouge_l(ans, q["reference_answer"]),
                       faithfulness_lex=lexical_faithfulness(ans, r["contexts"]) if r["contexts"] else float("nan"))
            hit = any(is_relevant(c, q["gold_snippet"]) for c in r["contexts"])
            row["context_hit"] = float(hit)
            if judge is not None:
                row["judge_correct"], row["judge_faith"] = llm_judge(
                    judge, q["question"], q["reference_answer"], ans, r["contexts"])
        rows.append(row)
    return pd.DataFrame(rows)


def summarize_generation(df):
    a, u = df[df.answerable], df[~df.answerable]
    s = {"f1": a.f1.mean(), "recall": a.recall.mean(), "rougeL": a.rougeL.mean(),
         "faithfulness_lex": a.faithfulness_lex.mean(),
         "false_refusal": a.refused.mean(),
         "refusal_acc": u.refused.mean() if len(u) else float("nan"),
         "latency_mean_s": df.latency_s.mean(), "latency_p95_s": df.latency_s.quantile(0.95)}
    if "judge_correct" in a:
        s["judge_correct"], s["judge_faith"] = a.judge_correct.mean(), a.judge_faith.mean()
    return s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--docs", default="data/docs")
    ap.add_argument("--qa", default="eval/qa.jsonl")
    ap.add_argument("--out", default="results")
    ap.add_argument("--retrieval-grid", action="store_true", help="retrieval ablation (no LLM)")
    ap.add_argument("--chunk-sizes", type=int, nargs="+", default=[400, 800, 1200])
    ap.add_argument("--with-rerank", action="store_true")
    ap.add_argument("--generate", action="store_true", help="end-to-end generation evaluation")
    ap.add_argument("--retrievers", nargs="+", default=["none", "bm25", "dense", "hybrid"])
    ap.add_argument("--chunk-size", type=int, default=800)
    ap.add_argument("--top-k", type=int, default=4)
    ap.add_argument("--llm", default="ollama", choices=["ollama", "extractive"])
    ap.add_argument("--model", default="qwen2.5:3b")
    ap.add_argument("--judge", action="store_true")
    ap.add_argument("--judge-model", default=None, help="defaults to --model")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    qa = load_qa(args.qa)
    print(f"{len(qa)} questions ({sum(q.get('answerable', True) for q in qa)} answerable)")

    if args.retrieval_grid:
        recs = []
        for cs in args.chunk_sizes:
            for ret in ["bm25", "dense", "hybrid"] + (["hybrid+rerank"] if args.with_rerank else []):
                cfg = Config(docs_dir=args.docs, chunk_size=cs, chunk_overlap=cs // 8,
                             retriever=ret.split("+")[0], rerank=ret.endswith("rerank"),
                             llm="extractive", top_k=5)
                try:
                    pipe = RAGPipeline(cfg)
                except Exception as e:  # e.g. dense model unavailable offline
                    print(f"skip {ret}/cs{cs}: {e}")
                    continue
                t0 = time.perf_counter()
                df = eval_retrieval(pipe, qa)
                lat = (time.perf_counter() - t0) / max(len(df), 1)
                rec = {"retriever": ret, "chunk_size": cs, "n_chunks": len(pipe.chunks),
                       "latency_ms": lat * 1000, **df.drop(columns="id").mean().to_dict()}
                recs.append(rec)
                print(rec)
        pd.DataFrame(recs).to_csv(out / "retrieval_results.csv", index=False)

    if args.generate:
        judge = None
        if args.judge:
            judge = OllamaGenerator(args.judge_model or args.model)
        recs = []
        for ret in args.retrievers:
            cfg = Config(docs_dir=args.docs, chunk_size=args.chunk_size,
                         chunk_overlap=args.chunk_size // 8, retriever=ret, top_k=args.top_k,
                         llm=args.llm, ollama_model=args.model)
            try:
                pipe = RAGPipeline(cfg)
            except Exception as e:
                print(f"skip {ret}: {e}")
                continue
            df = eval_generation(pipe, qa, judge)
            df.to_csv(out / f"per_question_{ret}.csv", index=False)
            rec = {"retriever": ret, "llm": args.llm if args.llm != "ollama" else args.model,
                   "chunk_size": args.chunk_size, "top_k": args.top_k, **summarize_generation(df)}
            recs.append(rec)
            print(rec)
        pd.DataFrame(recs).to_csv(out / "generation_results.csv", index=False)

    (out / "run_config.json").write_text(json.dumps(vars(args), indent=2))


if __name__ == "__main__":
    main()
