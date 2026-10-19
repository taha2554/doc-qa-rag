"""Question answering over the index.

With an LLM configured, retrieved chunks are numbered and the model is asked to answer only
from them and cite the numbers. Without an LLM the most relevant passages are returned as-is
("extractive mode"), so the project still works with no API key at all.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from langchain_core.documents import Document
from langchain_core.language_models import BaseChatModel
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.vectorstores import VectorStore

NO_ANSWER = "I couldn't find this in the documents."

PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You answer questions using only the numbered context passages below.\n"
            "- Cite the passages you used like [1] or [2][3] right after the sentence they support.\n"
            "- If the passages don't contain the answer, reply exactly: \"" + NO_ANSWER + "\"\n"
            "- Answer in the same language as the question. Be concise.\n\n"
            "Context:\n{context}",
        ),
        ("human", "{question}"),
    ]
)


@dataclass
class Source:
    ref: int
    source: str
    page: int | None
    score: float
    excerpt: str


@dataclass
class Answer:
    question: str
    answer: str
    sources: list[Source] = field(default_factory=list)
    mode: str = "llm"


def format_context(docs: list[Document]) -> str:
    parts = []
    for i, doc in enumerate(docs, start=1):
        where = doc.metadata.get("source", "?")
        if doc.metadata.get("page"):
            where += f", page {doc.metadata['page']}"
        parts.append(f"[{i}] ({where})\n{doc.page_content.strip()}")
    return "\n\n".join(parts)


def cited_refs(text: str) -> set[int]:
    return {int(n) for n in re.findall(r"\[(\d+)\]", text)}


class DocQA:
    def __init__(self, store: VectorStore, llm: BaseChatModel | None = None, top_k: int = 4, min_score: float = 0.05):
        self.store = store
        self.llm = llm
        self.top_k = top_k
        self.min_score = min_score
        self.chain = (PROMPT | llm | StrOutputParser()) if llm is not None else None

    def retrieve(self, question: str) -> list[tuple[Document, float]]:
        hits = self.store.similarity_search_with_score(question, k=self.top_k)
        return [(d, s) for d, s in hits if s >= self.min_score]

    def ask(self, question: str) -> Answer:
        question = question.strip()
        if not question:
            raise ValueError("question is empty")

        hits = self.retrieve(question)
        sources = [
            Source(i, d.metadata.get("source", "?"), d.metadata.get("page"), round(float(s), 4), d.page_content.strip()[:300])
            for i, (d, s) in enumerate(hits, start=1)
        ]
        if not hits:
            return Answer(question, NO_ANSWER, [], "llm" if self.chain else "extractive")

        if self.chain is None:
            best = hits[0][0].page_content.strip()
            return Answer(question, best, sources, "extractive")

        text = self.chain.invoke({"context": format_context([d for d, _ in hits]), "question": question}).strip()
        # Only return the sources the model actually cited (fall back to all if it cited none).
        used = cited_refs(text)
        if used:
            sources = [s for s in sources if s.ref in used]
        return Answer(question, text, sources, "llm")


def get_llm(provider: str, model: str) -> BaseChatModel | None:
    if provider == "none":
        return None
    if provider == "anthropic":
        try:
            from langchain_anthropic import ChatAnthropic
        except ImportError as exc:
            raise SystemExit("pip install langchain-anthropic (or `pip install -e .[anthropic]`)") from exc
        return ChatAnthropic(model=model, temperature=0, max_tokens=800)
    if provider == "openai":
        try:
            from langchain_openai import ChatOpenAI
        except ImportError as exc:
            raise SystemExit("pip install langchain-openai (or `pip install -e .[openai]`)") from exc
        return ChatOpenAI(model=model, temperature=0)
    raise ValueError(f"unknown LLM provider {provider!r}")
