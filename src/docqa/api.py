"""Small FastAPI wrapper: POST /ask {"question": "..."}"""

from __future__ import annotations

from dataclasses import asdict

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from .qa import DocQA


class AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=1000)


def create_app(qa: DocQA) -> FastAPI:
    app = FastAPI(title="Doc QA", version="1.0.0")

    @app.get("/health")
    def health():
        return {"status": "ok", "llm": qa.llm is not None}

    @app.post("/ask")
    def ask(req: AskRequest):
        try:
            return asdict(qa.ask(req.question))
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    return app
