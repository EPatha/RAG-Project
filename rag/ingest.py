"""Document loading and chunking."""
import re
from pathlib import Path


def _clean(text: str) -> str:
    text = text.replace("\r", "")
    text = re.sub(r"-\n(?=\w)", "", text)          # de-hyphenate line breaks
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def load_documents(docs_dir: str):
    """Return a list of {doc, page, text}. Supports .pdf, .txt, .md."""
    docs = []
    for p in sorted(Path(docs_dir).rglob("*")):
        suf = p.suffix.lower()
        if suf == ".pdf":
            from pypdf import PdfReader
            reader = PdfReader(str(p))
            for i, page in enumerate(reader.pages):
                t = _clean(page.extract_text() or "")
                if t:
                    docs.append({"doc": p.name, "page": i + 1, "text": t})
        elif suf in {".txt", ".md"}:
            t = _clean(p.read_text(encoding="utf-8"))
            if t:
                docs.append({"doc": p.name, "page": 1, "text": t})
    return docs


def _units(text: str, size: int):
    """Split into sentence-like units, hard-splitting any unit longer than `size`."""
    out = []
    for u in re.split(r"(?<=[.!?])\s+|\n+", text):
        u = u.strip()
        while len(u) > size:
            out.append(u[:size])
            u = u[size:]
        if u:
            out.append(u)
    return out


def chunk_text(text: str, size: int = 800, overlap: int = 100):
    chunks, cur, n = [], [], 0
    for u in _units(text, size):
        if n + len(u) + 1 > size and cur:
            chunks.append(" ".join(cur))
            keep, m = [], 0
            for x in reversed(cur):
                if m + len(x) > overlap:
                    break
                keep.insert(0, x)
                m += len(x) + 1
            cur, n = keep, m
        cur.append(u)
        n += len(u) + 1
    if cur:
        chunks.append(" ".join(cur))
    return chunks


def build_chunks(docs_dir: str, size: int = 800, overlap: int = 100):
    chunks = []
    for d in load_documents(docs_dir):
        for i, c in enumerate(chunk_text(d["text"], size, overlap)):
            chunks.append({
                "id": f'{d["doc"]}#p{d["page"]}#{i}',
                "doc": d["doc"], "page": d["page"], "text": c,
            })
    return chunks
