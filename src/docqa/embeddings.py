"""Embeddings.

`LocalHashEmbeddings` works fully offline: it hashes word unigrams/bigrams and character
trigrams into a fixed-size vector (the "hashing trick"). It is a lexical model – it won't know
that "car" and "automobile" are related – but it needs no downloads or API keys and is a
reasonable baseline. Set DOCQA_EMBEDDINGS=openai for semantic embeddings.
"""

from __future__ import annotations

import hashlib
import math
import re

from langchain_core.embeddings import Embeddings

_TOKEN = re.compile(r"\w+", re.UNICODE)
_STOP = frozenset(
    "a an the and or of to in on for is are was were be been it this that with as by at from what which who how "
    "do does did can could should would will not no".split()
)


class LocalHashEmbeddings(Embeddings):
    def __init__(self, dim: int = 2048):
        self.dim = dim

    def _features(self, text: str) -> list[str]:
        words = [w for w in _TOKEN.findall(text.lower()) if w not in _STOP]
        feats = list(words)
        feats += [f"{a}_{b}" for a, b in zip(words, words[1:])]
        for w in words:
            padded = f"<{w}>"
            feats += [padded[i : i + 3] for i in range(len(padded) - 2)]
        return feats

    def _embed(self, text: str) -> list[float]:
        vec = [0.0] * self.dim
        for feat in self._features(text):
            digest = hashlib.md5(feat.encode("utf-8")).digest()
            idx = int.from_bytes(digest[:4], "little") % self.dim
            sign = 1.0 if digest[4] & 1 else -1.0
            vec[idx] += sign
        # sublinear scaling + L2 normalisation, so long chunks don't dominate
        vec = [math.copysign(math.log1p(abs(v)), v) for v in vec]
        norm = math.sqrt(sum(v * v for v in vec)) or 1.0
        return [v / norm for v in vec]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._embed(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._embed(text)


def get_embeddings(kind: str) -> Embeddings:
    if kind == "local":
        return LocalHashEmbeddings()
    if kind == "openai":
        try:
            from langchain_openai import OpenAIEmbeddings
        except ImportError as exc:
            raise SystemExit("pip install langchain-openai to use OpenAI embeddings") from exc
        return OpenAIEmbeddings(model="text-embedding-3-small")
    raise ValueError(f"unknown embeddings type {kind!r}")
