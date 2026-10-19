# doc-qa-rag

![CI](https://github.com/taha2554/doc-qa-rag/actions/workflows/ci.yml/badge.svg)

Ask questions about your own PDFs, Markdown and text files. A small **retrieval-augmented generation (RAG)** system built with **LangChain**: documents are split into chunks, embedded, stored in a vector index, and the most relevant chunks are given to an LLM which answers with numbered citations.

It also runs **without any API key**: then it returns the most relevant passage directly (extractive mode), which is handy for testing and for CI.

Example with an LLM configured (output will vary slightly by model):

```
$ docqa ask "How many days of annual leave do I get?"

Full-time employees get 20 working days of paid annual leave per year, plus public
holidays; after five years this increases to 25 days [1].

Sources:
  [1] employee_handbook.md
```

## How it works

```
 ingest:  files ──► loaders (pypdf / text) ──► RecursiveCharacterTextSplitter ──► embeddings ──► InMemoryVectorStore ──► index/store.json
 ask:     question ──► embed ──► top-k similar chunks ──► prompt with numbered context ──► LLM ──► answer + cited sources
```

- **Chunking** – `RecursiveCharacterTextSplitter` prefers to split on headings and paragraphs, 500 characters with 80 overlap. Each chunk keeps its file name and PDF page number for citations.
- **Stable chunk ids** – a hash of source + page + text, so re-ingesting the same files doesn't create duplicates.
- **Embeddings** – by default a local hashing-trick embedding (word uni/bigrams + character trigrams). No downloads, no key, deterministic – but lexical only. Switch to OpenAI embeddings with `DOCQA_EMBEDDINGS=openai` for semantic search.
- **Prompt** – the model must answer only from the passages, cite them as `[n]`, and say *"I couldn't find this in the documents."* otherwise. Only the sources the model actually cited are returned.
- **LLM** – Anthropic or OpenAI chat models through LangChain, chosen automatically from whichever API key is set.

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"            # add [anthropic] or [openai] for generated answers
cp .env.example .env               # optional: put your API key here
```

## Usage

```bash
docqa ingest sample_docs/          # or a folder with your own PDFs
docqa ask "What is the limit for client dinners?"
docqa chat                         # interactive loop
docqa serve --port 8000            # HTTP API
```

API:

```bash
curl -s localhost:8000/ask -H 'Content-Type: application/json' \
  -d '{"question": "Are taxis covered?"}'
```

Returns `{"question", "answer", "mode", "sources": [{"ref", "source", "page", "score", "excerpt"}]}`.

`sample_docs/` contains a fictional employee handbook and expenses FAQ to try it out.

## Tests

```bash
pytest
```

15 tests, all offline: retrieval quality on the sample docs (does the right chunk come first?), extractive mode, citation filtering with a fake LLM (`FakeListChatModel`), index save/load, de-duplication, the API and the CLI.

## Configuration

See [`.env.example`](.env.example). API keys are only read from the environment / `.env`, which is git-ignored.

## Ideas for later

- Hybrid search (BM25 + embeddings) and a re-ranker
- Persistent vector DB (Chroma / pgvector) for large collections
- Small evaluation set with expected answers to measure retrieval recall

## License

MIT
