"""
tests/test_staff_handbook_ingestion.py — Proves the Staff Handbook / local
documents feature is actually wired into the ingestion pipeline end-to-end.

Regression test for a real audit finding: config.STAFF_HANDBOOK_FILE and
config.LOCAL_DOCUMENTS_FILE were defined and documented ("drop a JSONL... it
is picked up automatically") but scripts/update_knowledge_base.run() never
actually read them — a handbook file placed there was silently ignored
forever. scripts.update_knowledge_base._load_local_sources() now loads both,
and run() merges them in alongside crawled documents.
"""

import sys
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).parent.parent))

from tests.conftest import _mock_embed_texts, _mock_embed_query


def test_load_local_sources_reads_staff_handbook(tmp_path, monkeypatch):
    from src import config
    from src.utils import save_jsonl

    kb_dir = tmp_path / "kb"
    kb_dir.mkdir()
    handbook_file = kb_dir / "handbook.jsonl"
    save_jsonl(handbook_file, [
        {"title": "Leave Policy", "text": "Staff get 21 days of annual leave per year.",
         "url": "https://commit.takshashila.org.in/staff-handbook#leave-policy"},
        {"title": "AI Use Policy", "text": "Staff must disclose AI-assisted drafts to editors."},
    ])
    monkeypatch.setattr(config, "STAFF_HANDBOOK_FILE", handbook_file)
    monkeypatch.setattr(config, "LOCAL_DOCUMENTS_FILE", tmp_path / "does_not_exist.jsonl")
    monkeypatch.setattr(config, "DOCUMENTS_FILE", tmp_path / "documents.jsonl")

    from scripts.update_knowledge_base import _load_local_sources
    docs = _load_local_sources()

    assert len(docs) == 2
    assert all(d["source"] == "staff_handbook" for d in docs)
    assert all(d["source_name"] == "Takshashila Staff Handbook" for d in docs)
    # The doc with a real URL keeps it; the doc without one gets an
    # auto-generated document_id rather than being dropped or crashing.
    leave = next(d for d in docs if d["title"] == "Leave Policy")
    ai_use = next(d for d in docs if d["title"] == "AI Use Policy")
    assert leave["url"] == "https://commit.takshashila.org.in/staff-handbook#leave-policy"
    assert ai_use.get("document_id")


def test_load_local_sources_skips_unchanged_documents(tmp_path, monkeypatch):
    """Re-submitting an already-merged, byte-identical handbook entry must
    NOT be reported as 'changed' every run (would force a needless reindex
    on every scheduled update)."""
    from src import config
    from src.utils import save_jsonl, content_hash

    kb_dir = tmp_path / "kb"
    kb_dir.mkdir()
    handbook_file = kb_dir / "handbook.jsonl"
    entry = {"document_id": "staff_handbook_leave", "title": "Leave Policy",
              "text": "Staff get 21 days of annual leave per year.", "source": "staff_handbook"}
    save_jsonl(handbook_file, [entry])
    monkeypatch.setattr(config, "STAFF_HANDBOOK_FILE", handbook_file)
    monkeypatch.setattr(config, "LOCAL_DOCUMENTS_FILE", tmp_path / "does_not_exist.jsonl")

    doc_file = tmp_path / "documents.jsonl"
    already_merged = dict(entry)
    already_merged["content_hash"] = content_hash(entry["text"])
    save_jsonl(doc_file, [already_merged])
    monkeypatch.setattr(config, "DOCUMENTS_FILE", doc_file)

    from scripts.update_knowledge_base import _load_local_sources
    docs = _load_local_sources()
    assert docs == []  # unchanged -> nothing to re-merge


def test_load_local_sources_detects_real_change(tmp_path, monkeypatch):
    from src import config
    from src.utils import save_jsonl, content_hash

    kb_dir = tmp_path / "kb"
    kb_dir.mkdir()
    handbook_file = kb_dir / "handbook.jsonl"
    save_jsonl(handbook_file, [
        {"document_id": "staff_handbook_leave", "title": "Leave Policy",
         "text": "Staff get 25 days of annual leave per year (updated).", "source": "staff_handbook"},
    ])
    monkeypatch.setattr(config, "STAFF_HANDBOOK_FILE", handbook_file)
    monkeypatch.setattr(config, "LOCAL_DOCUMENTS_FILE", tmp_path / "does_not_exist.jsonl")

    doc_file = tmp_path / "documents.jsonl"
    save_jsonl(doc_file, [{"document_id": "staff_handbook_leave", "title": "Leave Policy",
                          "text": "Staff get 21 days of annual leave per year.",
                          "content_hash": content_hash("Staff get 21 days of annual leave per year.")}])
    monkeypatch.setattr(config, "DOCUMENTS_FILE", doc_file)

    from scripts.update_knowledge_base import _load_local_sources
    docs = _load_local_sources()
    assert len(docs) == 1
    assert "25 days" in docs[0]["text"]


def test_full_ingestion_incorporates_staff_handbook_end_to_end(monkeypatch, tmp_path):
    """
    The real end-to-end proof: a handbook JSONL on disk -> run() picks it up
    (no crawl needed for it) -> merged into documents.jsonl -> chunked ->
    embedded -> indexed -> retrievable -> its exact URL is what a citation
    would link to.
    """
    from src import config, vector_store

    monkeypatch.setattr(config, "DOCUMENTS_FILE", tmp_path / "documents.jsonl")
    monkeypatch.setattr(config, "CHUNKS_FILE", tmp_path / "chunks.jsonl")
    index_dir = tmp_path / "index"
    index_dir.mkdir()
    monkeypatch.setattr(config, "INDEX_DIR", index_dir)
    monkeypatch.setattr(config, "FAISS_INDEX", index_dir / "faiss.index")
    monkeypatch.setattr(config, "METADATA_FILE", index_dir / "metadata.pkl")
    monkeypatch.setattr(config, "WEBSITE_MANIFEST_FILE", tmp_path / "manifest.json")
    monkeypatch.setattr(config, "COMMIT_KB_USERNAME", "")
    monkeypatch.setattr(config, "COMMIT_KB_PASSWORD", "")
    monkeypatch.setattr(vector_store, "METADATA_JSON", index_dir / "metadata.json")
    from src import incremental_index
    monkeypatch.setattr(incremental_index, "METADATA_JSON", index_dir / "metadata.json")

    kb_dir = tmp_path / "kb"
    kb_dir.mkdir()
    handbook_file = kb_dir / "handbook.jsonl"
    from src.utils import save_jsonl
    save_jsonl(handbook_file, [{
        "title": "Leave Policy",
        "text": "Staff at Takshashila are entitled to 21 days of paid annual "
                "leave, accrued monthly and carried over up to a maximum of "
                "10 days into the following year. " * 3,
        "url": "https://commit.takshashila.org.in/staff-handbook#leave-policy",
    }])
    monkeypatch.setattr(config, "STAFF_HANDBOOK_FILE", handbook_file)
    monkeypatch.setattr(config, "LOCAL_DOCUMENTS_FILE", tmp_path / "does_not_exist.jsonl")

    def fake_crawl_site(site, incremental=True, progress_cb=None):
        from scripts.crawl_engine import CrawlResult
        return CrawlResult(docs=[], removed_ids=[], counts={}, discovered=0)

    with patch("scripts.update_knowledge_base.crawl_site", side_effect=fake_crawl_site), \
         patch("src.vector_store.embed_texts", side_effect=_mock_embed_texts), \
         patch("src.vector_store.embed_query", side_effect=_mock_embed_query), \
         patch("src.embeddings.embed_texts", side_effect=_mock_embed_texts):
        from scripts.update_knowledge_base import run
        summary = run(website=True, commit_kb=True, incremental=True, do_index=True)

    assert summary["per_source"].get("local_sources") == {"staff_handbook": 1}

    from src.utils import load_jsonl
    docs_on_disk = load_jsonl(config.DOCUMENTS_FILE)
    assert any(d.get("source") == "staff_handbook" for d in docs_on_disk)

    vector_store._INDEX = None
    vector_store._METADATA = []
    with patch("src.vector_store.embed_texts", side_effect=_mock_embed_texts), \
         patch("src.vector_store.embed_query", side_effect=_mock_embed_query):
        from src.retriever import retrieve
        results = retrieve("leave policy annual leave days", top_k=3)
    assert results
    assert results[0]["source"] == "staff_handbook"
    assert results[0]["url"] == "https://commit.takshashila.org.in/staff-handbook#leave-policy"

    vector_store._INDEX = None
    vector_store._METADATA = []
