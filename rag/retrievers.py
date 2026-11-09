"""Retrievers: BM25, dense (sentence-transformers), hybrid (RRF), optional cross-encoder rerank."""
import hashlib
from pathlib import Path

import numpy as np

from .utils import tokenize


class BM25Retriever:
    def __init__(self, chunks):
        from rank_bm25 import BM25Okapi
        self.bm25 = BM25Okapi([tokenize(c["text"]) for c in chunks])

    def search(self, query, n):
        scores = self.bm25.get_scores(tokenize(query))
        idx = np.argsort(-scores)[:n]
        return [(int(i), float(scores[i])) for i in idx]


class DenseRetriever:
    def __init__(self, chunks, model_name="intfloat/multilingual-e5-small", cache_dir=".cache"):
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(model_name)
        self.is_e5 = "e5" in model_name.lower()
        key = hashlib.md5((model_name + "||".join(c["text"] for c in chunks)).encode()).hexdigest()
        path = Path(cache_dir) / f"emb_{key}.npy"
        if path.exists():
            self.emb = np.load(path)
        else:
            texts = [("passage: " + c["text"]) if self.is_e5 else c["text"] for c in chunks]
            self.emb = self.model.encode(texts, normalize_embeddings=True, batch_size=32,
                                         show_progress_bar=False)
            path.parent.mkdir(parents=True, exist_ok=True)
            np.save(path, self.emb)

    def search(self, query, n):
        q = ("query: " + query) if self.is_e5 else query
        qv = self.model.encode([q], normalize_embeddings=True)[0]
        scores = self.emb @ qv
        idx = np.argsort(-scores)[:n]
        return [(int(i), float(scores[i])) for i in idx]


def rrf(rank_lists, k=60):
    """Reciprocal Rank Fusion over lists of (idx, score)."""
    fused = {}
    for lst in rank_lists:
        for rank, (i, _) in enumerate(lst):
            fused[i] = fused.get(i, 0.0) + 1.0 / (k + rank + 1)
    return sorted(fused.items(), key=lambda x: -x[1])


class HybridRetriever:
    def __init__(self, bm25, dense, pool=20):
        self.bm25, self.dense, self.pool = bm25, dense, pool

    def search(self, query, n):
        fused = rrf([self.bm25.search(query, self.pool), self.dense.search(query, self.pool)])
        return [(i, s) for i, s in fused[:n]]


class CrossEncoderReranker:
    def __init__(self, model_name="BAAI/bge-reranker-base"):
        from sentence_transformers import CrossEncoder
        self.model = CrossEncoder(model_name)

    def rerank(self, query, chunks, candidates, n):
        pairs = [(query, chunks[i]["text"]) for i, _ in candidates]
        scores = self.model.predict(pairs)
        order = np.argsort(-scores)[:n]
        return [(candidates[j][0], float(scores[j])) for j in order]
