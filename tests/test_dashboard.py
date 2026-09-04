"""
tests/test_dashboard.py — Dashboard (app.py) verification via Streamlit's
AppTest framework, which actually executes the script headlessly and
surfaces real runtime exceptions — not a mock of Streamlit itself.

Covers:
  - the whole script runs with NO unhandled exception in every state checked
    (empty KB, built KB, chat query -> answer -> citations)
  - the "no KB yet" state doesn't halt tabs after it (regression test for a
    real bug found and fixed this session: st.stop() inside one tab's body
    silently prevented every later tab, including Build & Update, from ever
    rendering — a fresh install had no way to reach the button that builds
    the KB in the first place)
  - a real end-to-end query through the dashboard's chat UI, using the SAME
    synthetic_index fixture as the retrieval tests (real FAISS index, mocked
    embeddings) plus a mocked Groq call, asserting the rendered answer
    actually contains a working citation link to the real source URL

NOT covered here (documented, not silently skipped): real visual/CSS
rendering, real mobile viewport behaviour, and anything requiring an actual
browser — AppTest inspects the Python-side element tree, not pixels.
"""

import sys
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
from streamlit.testing.v1 import AppTest

from tests.conftest import SAMPLE_CHUNKS, _mock_embed_query, _mock_embed_texts

APP_PATH = str(Path(__file__).parent.parent / "app.py")


def test_app_runs_clean_with_no_kb_built(monkeypatch):
    """Fresh-install state: no documents, no FAISS index, no GROQ_API_KEY.
    The whole script must execute with zero unhandled exceptions."""
    from src import config
    monkeypatch.setattr(config, "GROQ_API_KEY", "")
    at = AppTest.from_file(APP_PATH, default_timeout=45)
    at.run()
    assert len(at.exception) == 0, [str(e) for e in at.exception]


def test_no_kb_state_does_not_block_later_tabs(monkeypatch):
    """Regression test for the st.stop()-halts-everything bug: even with no
    index built, the Build & Update and Automation tabs' own buttons must
    still render, or a fresh deployment could never reach the controls that
    build the KB in the first place."""
    from src import config
    monkeypatch.setattr(config, "GROQ_API_KEY", "")
    at = AppTest.from_file(APP_PATH, default_timeout=45)
    at.run()
    assert len(at.exception) == 0
    keys = {b.key for b in at.button}
    for expected in ("btn_incr", "btn_full", "btn_reindex", "btn_val", "auto_run", "auto_refresh"):
        assert expected in keys, f"{expected} missing — a later tab failed to render"


def test_admin_gate_warns_when_unconfigured(monkeypatch):
    from src import config
    monkeypatch.setattr(config, "DASHBOARD_ADMIN_PASSWORD", "")
    at = AppTest.from_file(APP_PATH, default_timeout=45)
    at.run()
    warnings = " ".join(w.value for w in at.warning)
    assert "No admin password configured" in warnings


def test_admin_gate_blocks_action_without_password(monkeypatch):
    """Clicking a destructive admin button without unlocking must NOT run
    the underlying pipeline — it should show the lock message instead."""
    from src import config
    monkeypatch.setattr(config, "DASHBOARD_ADMIN_PASSWORD", "supersecret")
    with patch("scripts.update_knowledge_base.run") as mock_run:
        at = AppTest.from_file(APP_PATH, default_timeout=45)
        at.run()
        btn = next(b for b in at.button if b.key == "btn_incr")
        btn.click().run()
        mock_run.assert_not_called()
    errors = " ".join(e.value for e in at.error)
    assert "Admin password required" in errors


def test_full_query_flow_renders_grounded_answer_with_citation(monkeypatch, tmp_path):
    """
    The real end-to-end check the task asked for: user types a question in
    the dashboard -> retrieval runs against a real (synthetic) FAISS index ->
    a mocked LLM call returns an answer with a [Source N] marker -> the
    rendered chat message contains the linkified citation pointing at the
    exact source URL, and the References list shows it.
    """
    from src import config, vector_store, retriever

    index_dir = tmp_path / "index"
    index_dir.mkdir()
    monkeypatch.setattr(config, "INDEX_DIR", index_dir)
    monkeypatch.setattr(config, "FAISS_INDEX", index_dir / "faiss.index")
    monkeypatch.setattr(config, "METADATA_FILE", index_dir / "metadata.pkl")
    monkeypatch.setattr(vector_store, "METADATA_JSON", index_dir / "metadata.json")
    monkeypatch.setattr(config, "GROQ_API_KEY", "test-key-not-real")

    with patch("src.vector_store.embed_texts", side_effect=_mock_embed_texts), \
         patch("src.vector_store.embed_query", side_effect=_mock_embed_query):
        vector_store.build_index(chunks=[dict(c) for c in SAMPLE_CHUNKS])
    vector_store._INDEX = None
    vector_store._METADATA = []
    retriever._BM25 = None
    retriever._BM25_CHUNKS = []
    retriever._BM25_SIG = None

    geo_chunk = next(c for c in SAMPLE_CHUNKS if c["document_id"] == "doc_geo_1")
    mocked_answer = ("India's geospatial data ecosystem faces portal-access "
                      "challenges [Source 1].")

    with patch("src.vector_store.embed_texts", side_effect=_mock_embed_texts), \
         patch("src.vector_store.embed_query", side_effect=_mock_embed_query), \
         patch("src.embeddings.embed_texts", side_effect=_mock_embed_texts), \
         patch("src.embeddings.embed_query", side_effect=_mock_embed_query), \
         patch("src.embeddings._get_model", return_value=object()), \
         patch("src.groq_client.generate", return_value=mocked_answer):
        at = AppTest.from_file(APP_PATH, default_timeout=45)
        at.run()
        assert len(at.exception) == 0

        # Switch to "Ask Takshashila" and submit a question through the real
        # chat_input widget, exactly as a user would.
        at.tabs[1].chat_input[0].set_value("Tell me about geospatial portals in India").run()
        assert len(at.exception) == 0, [str(e) for e in at.exception]

    # The rendered assistant message must contain a real, linkified citation
    # pointing at the actual source URL — not a bare, unresolved "[Source 1]".
    all_markdown_text = " ".join(m.value for m in at.tabs[1].markdown)
    assert geo_chunk["url"] in all_markdown_text
    assert "[Source 1]" not in all_markdown_text  # should have been linkified to "[1](url)"
