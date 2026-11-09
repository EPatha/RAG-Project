import re

REFUSAL = "Informasi tidak ditemukan dalam dokumen."

STOPWORDS = set("""
yang dan di ke dari untuk pada dengan atau adalah itu ini dalam sebagai oleh dapat
akan harus wajib tidak ada juga bagi para setiap secara serta paling lebih telah
sudah suatu satu dua sebuah apa berapa bagaimana kapan siapa apakah jika maka agar
the of and to in is a for on with
""".split())


def tokenize(text: str):
    return re.findall(r"\w+", text.lower())


def content_tokens(text: str):
    return [t for t in tokenize(text) if t not in STOPWORDS]


def is_refusal(answer: str) -> bool:
    a = answer.lower()
    return "tidak ditemukan" in a or "tidak dapat ditemukan" in a or "tidak tersedia dalam dokumen" in a
