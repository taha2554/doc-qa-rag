"""Settings come from environment variables (or a .env file). See .env.example."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


def _load_dotenv(path: Path = Path(".env")) -> None:
    """Minimal .env reader so python-dotenv isn't needed. Existing env vars win."""
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


@dataclass(frozen=True)
class Settings:
    index_path: Path
    llm_provider: str  # "anthropic", "openai" or "none"
    llm_model: str
    embeddings: str  # "local" or "openai"
    chunk_size: int
    chunk_overlap: int
    top_k: int

    @classmethod
    def from_env(cls) -> "Settings":
        _load_dotenv()
        provider = os.getenv("DOCQA_LLM_PROVIDER", "").lower()
        if not provider:
            # pick whatever key is available, otherwise run without an LLM
            provider = "anthropic" if os.getenv("ANTHROPIC_API_KEY") else "openai" if os.getenv("OPENAI_API_KEY") else "none"
        default_model = {"anthropic": "claude-haiku-5-5", "openai": "gpt-4o-mini"}.get(provider, "")
        return cls(
            index_path=Path(os.getenv("DOCQA_INDEX_PATH", "index/store.json")),
            llm_provider=provider,
            llm_model=os.getenv("DOCQA_LLM_MODEL", default_model),
            embeddings=os.getenv("DOCQA_EMBEDDINGS", "local").lower(),
            chunk_size=int(os.getenv("DOCQA_CHUNK_SIZE", "500")),
            chunk_overlap=int(os.getenv("DOCQA_CHUNK_OVERLAP", "80")),
            top_k=int(os.getenv("DOCQA_TOP_K", "4")),
        )
