from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from langchain_core.language_models.fake_chat_models import FakeListChatModel

from docqa import ingest
from docqa.api import create_app
from docqa.config import Settings
from docqa.embeddings import LocalHashEmbeddings
from docqa.qa import NO_ANSWER, DocQA, cited_refs, format_context
from docqa.cli import main as cli_main

SAMPLE = "sample_docs"


@pytest.fixture(scope="module")
def store():
    store, n_docs, n_chunks = ingest.build_index(
        Path(SAMPLE), LocalHashEmbeddings(), chunk_size=400, chunk_overlap=50
    )
    assert n_docs == 2 and n_chunks > 5
    return store


def test_local_embeddings_are_normalised_and_deterministic():
    emb = LocalHashEmbeddings(dim=256)
    a, b = emb.embed_query("annual leave days"), emb.embed_query("annual leave days")
    assert a == b
    assert abs(sum(v * v for v in a) - 1) < 1e-6


@pytest.mark.parametrize(
    "question, expected_source, expected_text",
    [
        ("How many days of annual leave do I get?", "employee_handbook.md", "20 working days"),
        ("What is the limit for a client dinner?", "expenses_faq.txt", "750 TL"),
        ("Can I take a taxi home?", "expenses_faq.txt", "22:00"),
        ("How much can I spend on home office equipment?", "employee_handbook.md", "3,000 TL"),
    ],
)
def test_retrieval_finds_the_right_chunk(store, question, expected_source, expected_text):
    qa = DocQA(store, top_k=3)
    top_doc, _ = qa.retrieve(question)[0]
    assert top_doc.metadata["source"] == expected_source
    assert expected_text in top_doc.page_content


def test_extractive_mode_without_llm(store):
    ans = DocQA(store).ask("When are core hours?")
    assert ans.mode == "extractive"
    assert "10:00" in ans.answer
    assert ans.sources[0].source == "employee_handbook.md"


def test_llm_answer_keeps_only_cited_sources(store):
    llm = FakeListChatModel(responses=["You get 20 working days of paid leave [1]."])
    ans = DocQA(store, llm=llm, top_k=4).ask("How many vacation days?")
    assert ans.mode == "llm"
    assert [s.ref for s in ans.sources] == [1]


def test_empty_question_rejected(store):
    with pytest.raises(ValueError):
        DocQA(store).ask("   ")


def test_unrelated_question_returns_no_answer(store):
    qa = DocQA(store, min_score=0.99)
    assert qa.ask("What is the capital of Peru?").answer == NO_ANSWER


def test_helpers():
    assert cited_refs("a [1] b [3][1]") == {1, 3}
    from langchain_core.documents import Document

    ctx = format_context([Document(page_content="text", metadata={"source": "a.pdf", "page": 2})])
    assert ctx.startswith("[1] (a.pdf, page 2)")


def test_index_round_trip(store, tmp_path):
    path = tmp_path / "store.json"
    ingest.save(store, path)
    loaded = ingest.load(path, LocalHashEmbeddings())
    assert loaded.similarity_search("sick leave doctor's note", k=1)[0].metadata["source"] == "employee_handbook.md"


def test_reingest_does_not_duplicate():
    from pathlib import Path

    store, _, n_chunks = ingest.build_index(Path(SAMPLE), LocalHashEmbeddings())
    chunks = ingest.split(ingest.load_folder(Path(SAMPLE)), 800, 120)
    store.add_documents(chunks, ids=[ingest.chunk_id(c) for c in chunks])
    assert len(store.store) == n_chunks


def test_api(store):
    client = TestClient(create_app(DocQA(store)))
    assert client.get("/health").json() == {"status": "ok", "llm": False}
    res = client.post("/ask", json={"question": "How do I report a lost laptop?"})
    assert res.status_code == 200
    assert "24 hours" in res.json()["answer"]
    assert client.post("/ask", json={"question": ""}).status_code == 422


def test_settings_pick_provider_from_keys(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    for key in ["DOCQA_LLM_PROVIDER", "ANTHROPIC_API_KEY", "OPENAI_API_KEY"]:
        monkeypatch.delenv(key, raising=False)
    assert Settings.from_env().llm_provider == "none"
    monkeypatch.setenv("OPENAI_API_KEY", "test")
    assert Settings.from_env().llm_provider == "openai"


def test_cli_ingest_and_ask(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("DOCQA_INDEX_PATH", str(tmp_path / "idx.json"))
    monkeypatch.setenv("DOCQA_LLM_PROVIDER", "none")
    from pathlib import Path

    cli_main(["ingest", str(Path(SAMPLE).resolve())])
    cli_main(["ask", "What happens if I lose a receipt?"])
    out = capsys.readouterr().out
    assert "missing receipt form" in out and "expenses_faq.txt" in out
