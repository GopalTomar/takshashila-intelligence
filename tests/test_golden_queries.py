"""
test_golden_queries.py — golden regression suite (Sections 15, 35, 41).

Drives the CANONICAL service (src.rag_service.answer_query) end-to-end over a
real, in-memory FAISS index with a mocked LLM, and asserts the production
contract for representative Takshashila questions:

  * answerable question  → grounded answer, ≥1 citation, real indexed URL,
    confidence ≠ NONE, and the standard response keys are present;
  * unanswerable question → insufficient-evidence refusal, NO sources, NO
    fabricated citation (the hallucination guard).

The LLM is mocked so the suite is deterministic and offline; the retrieval,
evidence gate, citation verification and grounding logic are all real.
"""

import sys
from pathlib import Path
from unittest.mock import patch

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from src import rag_service, rag_pipeline  # noqa: E402
from tests.conftest import _mock_embed_texts, _mock_embed_query  # noqa: E402


def _grounded_answer_for(context_chunks):
    """A fake LLM reply that cites [Source 1] and reuses the source's own words,
    so citation verification + grounding pass exactly as they would in prod."""
    first = context_chunks[0] if context_chunks else {}
    text = (first.get("text") or "").strip()
    return f"{text} [Source 1]"


@pytest.fixture()
def service_env(synthetic_index):
    """synthetic_index sets up the index + embedding patches; we add the LLM mock
    (and re-assert the embedding patches for the retrieval that happens here)."""
    with patch("src.vector_store.embed_texts", side_effect=_mock_embed_texts), \
         patch("src.vector_store.embed_query", side_effect=_mock_embed_query):
        yield


def _run(query, generate_impl):
    with patch("src.vector_store.embed_texts", side_effect=_mock_embed_texts), \
         patch("src.vector_store.embed_query", side_effect=_mock_embed_query), \
         patch("src.groq_client.generate", side_effect=generate_impl):
        return rag_service.answer_query(query, interface="test", top_k=5)


@pytest.mark.parametrize("query", [
    "Tell me about India's geospatial portals",
    "What does the defence modernisation brief cover?",
    "India economy industrial policy",
])
def test_answerable_queries_are_grounded_with_citations(service_env, query):
    # Build a grounded answer from whatever context the real retriever selected.
    def gen(system_prompt, user_prompt, **kw):
        # Extract the first source block's text from the prompt so the answer
        # genuinely overlaps the retrieved context.
        marker = "[Source 1]"
        idx = user_prompt.find(marker)
        snippet = user_prompt[idx: idx + 400] if idx >= 0 else user_prompt[:400]
        return f"{snippet} [Source 1]"

    resp = _run(query, gen)

    # Standard contract keys present.
    for key in ("request_id", "answer", "grounded", "confidence", "sources",
                "kb_version", "model", "latency_ms"):
        assert key in resp, f"missing contract key {key!r}"

    assert resp["model"] == "openai/gpt-oss-120b"
    assert resp["confidence"] != "NONE"
    assert resp["grounded"] is True
    assert resp["sources"], "a grounded answer must carry at least one citation"
    src0 = resp["sources"][0]
    assert src0["url"].startswith("https://takshashila.org.in/"), \
        "citations must use real indexed URLs, never fabricated ones"
    assert src0["title"]


def test_unanswerable_query_refuses_without_hallucinating(service_env):
    # A topic entirely absent from the corpus, with no salient-token overlap.
    def gen(system_prompt, user_prompt, **kw):
        # Even if the model were asked, refuse — but the evidence gate should
        # short-circuit before generation anyway.
        return "INSUFFICIENT_EVIDENCE"

    resp = _run("nuclear submarine reactor coolant loop schematics", gen)

    assert resp["grounded"] is False
    assert resp["confidence"] == "NONE"
    assert resp["sources"] == [], "a refusal must not show any sources"
    assert rag_pipeline.NO_EVIDENCE_SENTENCE in resp["answer"]
