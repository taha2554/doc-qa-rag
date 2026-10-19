"""Command line interface.

    docqa ingest sample_docs/
    docqa ask "How many days of annual leave do employees get?"
    docqa chat
    docqa serve
"""

from __future__ import annotations

import argparse
import sys
import textwrap
from pathlib import Path

from . import ingest
from .config import Settings
from .embeddings import get_embeddings
from .qa import Answer, DocQA, get_llm


def build_qa(settings: Settings) -> DocQA:
    store = ingest.load(settings.index_path, get_embeddings(settings.embeddings))
    return DocQA(store, get_llm(settings.llm_provider, settings.llm_model), top_k=settings.top_k)


def print_answer(ans: Answer) -> None:
    print()
    print(textwrap.fill(ans.answer, 100) if ans.mode == "llm" else ans.answer)
    if ans.mode == "extractive":
        print("\n(no LLM configured – showing the most relevant passage)")
    if ans.sources:
        print("\nSources:")
        for s in ans.sources:
            page = f", p. {s.page}" if s.page else ""
            print(f"  [{s.ref}] {s.source}{page}  (score {s.score:.2f})")
    print()


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="docqa", description="Ask questions about your documents.")
    sub = parser.add_subparsers(dest="command", required=True)

    p_ingest = sub.add_parser("ingest", help="index a folder of .pdf/.md/.txt files")
    p_ingest.add_argument("folder", type=Path)

    p_ask = sub.add_parser("ask", help="ask a single question")
    p_ask.add_argument("question")

    sub.add_parser("chat", help="interactive question loop")

    p_serve = sub.add_parser("serve", help="run the HTTP API")
    p_serve.add_argument("--port", type=int, default=8000)

    args = parser.parse_args(argv)
    settings = Settings.from_env()

    if args.command == "ingest":
        store, n_docs, n_chunks = ingest.build_index(
            args.folder, get_embeddings(settings.embeddings), settings.chunk_size, settings.chunk_overlap
        )
        ingest.save(store, settings.index_path)
        print(f"Indexed {n_docs} documents/pages as {n_chunks} chunks -> {settings.index_path}")

    elif args.command == "ask":
        print_answer(build_qa(settings).ask(args.question))

    elif args.command == "chat":
        qa = build_qa(settings)
        print(f"LLM: {settings.llm_provider} {settings.llm_model}  –  empty line or Ctrl+C to quit")
        try:
            while question := input("? ").strip():
                print_answer(qa.ask(question))
        except (KeyboardInterrupt, EOFError):
            print()

    elif args.command == "serve":
        import uvicorn

        from .api import create_app

        uvicorn.run(create_app(build_qa(settings)), host="127.0.0.1", port=args.port)


if __name__ == "__main__":
    sys.exit(main())
