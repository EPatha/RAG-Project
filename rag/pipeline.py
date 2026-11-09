"""End-to-end RAG pipeline: ingest -> index -> retrieve (-> rerank) -> generate."""
import time
from dataclasses import dataclass, asdict

from .generators import ExtractiveGenerator, OllamaGenerator
from .ingest import build_chunks
from .retrievers import BM25Retriever, CrossEncoderReranker, DenseRetriever, HybridRetriever


@dataclass
class Config:
    docs_dir: str = "data/docs"
    chunk_size: int = 800
    chunk_overlap: int = 100
    retriever: str = "hybrid"      # none | bm25 | dense | hybrid
    top_k: int = 4
    rerank: bool = False
    embed_model: str = "intfloat/multilingual-e5-small"
    llm: str = "ollama"            # ollama | extractive
    ollama_model: str = "qwen2.5:3b"
    ollama_url: str = "http://localhost:11434"

    def name(self):
        r = self.retriever + ("+rerank" if self.rerank else "")
        return f"{r}|cs{self.chunk_size}|k{self.top_k}"

    def to_dict(self):
        return asdict(self)


class RAGPipeline:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.chunks = build_chunks(cfg.docs_dir, cfg.chunk_size, cfg.chunk_overlap)
        if not self.chunks and cfg.retriever != "none":
            raise ValueError(f"No documents found in {cfg.docs_dir}")
        self.retriever = self._build_retriever()
        self.reranker = CrossEncoderReranker() if cfg.rerank else None
        self.generator = (OllamaGenerator(cfg.ollama_model, cfg.ollama_url)
                          if cfg.llm == "ollama" else ExtractiveGenerator())

    def _build_retriever(self):
        r = self.cfg.retriever
        if r == "none":
            return None
        if r == "bm25":
            return BM25Retriever(self.chunks)
        dense = DenseRetriever(self.chunks, self.cfg.embed_model)
        if r == "dense":
            return dense
        if r == "hybrid":
            return HybridRetriever(BM25Retriever(self.chunks), dense)
        raise ValueError(f"unknown retriever {r}")

    def retrieve(self, query, n=None):
        """Return list of (chunk, score), best first."""
        if self.retriever is None:
            return []
        n = n or self.cfg.top_k
        if self.reranker:
            cand = self.retriever.search(query, max(n * 4, 20))
            hits = self.reranker.rerank(query, self.chunks, cand, n)
        else:
            hits = self.retriever.search(query, n)
        return [(self.chunks[i], s) for i, s in hits]

    def answer(self, query):
        t0 = time.perf_counter()
        hits = self.retrieve(query)
        t1 = time.perf_counter()
        ctxs = [c for c, _ in hits]
        ans = self.generator.answer(query, ctxs)
        t2 = time.perf_counter()
        return {"question": query, "answer": ans, "contexts": ctxs,
                "scores": [s for _, s in hits],
                "retrieval_s": t1 - t0, "generation_s": t2 - t1, "total_s": t2 - t0}
