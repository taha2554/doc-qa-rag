"""Load documents from a folder, split them into chunks and build the vector index."""

from __future__ import annotations

import hashlib
from pathlib import Path

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter

SUPPORTED = {".pdf", ".md", ".txt"}


def load_file(path: Path) -> list[Document]:
    """One Document per PDF page, or one per text file. `source` and `page` go into metadata for citations."""
    if path.suffix.lower() == ".pdf":
        from pypdf import PdfReader

        reader = PdfReader(str(path))
        return [
            Document(page_content=page.extract_text() or "", metadata={"source": path.name, "page": i + 1})
            for i, page in enumerate(reader.pages)
        ]
    text = path.read_text(encoding="utf-8", errors="replace")
    return [Document(page_content=text, metadata={"source": path.name})]


def load_folder(folder: Path) -> list[Document]:
    docs = []
    for path in sorted(folder.rglob("*")):
        if path.is_file() and path.suffix.lower() in SUPPORTED:
            docs.extend(d for d in load_file(path) if d.page_content.strip())
    return docs


def split(docs: list[Document], chunk_size: int, chunk_overlap: int) -> list[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n## ", "\n### ", "\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(docs)
    for i, chunk in enumerate(chunks):
        chunk.metadata["chunk"] = i
    return chunks


def chunk_id(chunk: Document) -> str:
    # stable ids: re-ingesting unchanged text produces the same id, so it's not duplicated
    key = f"{chunk.metadata.get('source')}|{chunk.metadata.get('page')}|{chunk.page_content}"
    return hashlib.sha1(key.encode("utf-8")).hexdigest()


def build_index(folder: Path, embeddings: Embeddings, chunk_size: int = 500, chunk_overlap: int = 80) -> tuple[InMemoryVectorStore, int, int]:
    docs = load_folder(folder)
    if not docs:
        raise ValueError(f"no {', '.join(sorted(SUPPORTED))} files with text found in {folder}")
    chunks = split(docs, chunk_size, chunk_overlap)
    store = InMemoryVectorStore(embeddings)
    store.add_documents(chunks, ids=[chunk_id(c) for c in chunks])
    return store, len(docs), len(chunks)


def save(store: InMemoryVectorStore, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    store.dump(str(path))


def load(path: Path, embeddings: Embeddings) -> InMemoryVectorStore:
    if not path.exists():
        raise FileNotFoundError(f"index not found at {path} – run `docqa ingest <folder>` first")
    return InMemoryVectorStore.load(str(path), embeddings)
