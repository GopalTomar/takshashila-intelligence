"""
tests/test_end_to_end_ingestion.py — Deterministic end-to-end ingestion test.

Simulates the full pipeline scripts/update_knowledge_base.run() drives:
    crawl (mocked — no live network) -> merge into documents.jsonl
    -> chunk -> embed (mocked) -> FAISS index -> validate

This is the offline acceptance test for spec point 29 ("crawl -> extraction
-> chunking -> embedding -> indexing -> ... traceable end-to-end"), scoped to
what's verifiable without live Takshashila/network access: the crawl step
itself is mocked (see docstring on the mock below) since scripts/crawl_engine
hitting the real site is explicitly a live-only concern, documented as such
rather than silently skipped.
"""

import sys
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).parent.parent))

from tests.conftest import _mock_embed_texts, _mock_embed_query


def _fake_crawl_result(docs, removed_ids=None):
    """A stand-in for scripts.crawl_engine.crawl_site()'s return value. Using
    the real CrawlResult dataclass (not a bare dict) keeps this test honest
    about the actual interface update_knowledge_base.run() consumes."""
    from scripts.crawl_engine import CrawlResult
    return CrawlResult(docs=docs, removed_ids=removed_ids or [],
                       counts={"added": len(docs), "updated": 0, "unchanged": 0,
                               "failed": 0, "pdf": 0, "removed": len(removed_ids or [])},
                       discovered=len(docs))


def test_full_ingestion_pipeline_offline(monkeypatch, tmp_path):
    from src import config

    # ── isolate every on-disk artefact under tmp_path ───────────────────────
    monkeypatch.setattr(config, "DOCUMENTS_FILE", tmp_path / "documents.jsonl")
    monkeypatch.setattr(config, "CHUNKS_FILE", tmp_path / "chunks.jsonl")
    index_dir = tmp_path / "index"
    index_dir.mkdir()
    monkeypatch.setattr(config, "INDEX_DIR", index_dir)
    monkeypatch.setattr(config, "FAISS_INDEX", index_dir / "faiss.index")
    monkeypatch.setattr(config, "METADATA_FILE", index_dir / "metadata.pkl")
    monkeypatch.setattr(config, "WEBSITE_MANIFEST_FILE", tmp_path / "manifest.json")
    monkeypatch.setattr(config, "COMMIT_KB_USERNAME", "")  # force commit_kb "skipped" branch
    monkeypatch.setattr(config, "COMMIT_KB_PASSWORD", "")

    from src import vector_store
    monkeypatch.setattr(vector_store, "METADATA_JSON", index_dir / "metadata.json")
    from src import incremental_index
    monkeypatch.setattr(incremental_index, "METADATA_JSON", index_dir / "metadata.json")

    website_docs = [
        {
            "document_id": "website_geo1", "url_hash": "geo1",
            "url": "https://takshashila.org.in/content/publications/geo-portals.html",
            "original_url": "https://takshashila.org.in/content/publications/geo-portals.html",
            "title": "State of India's Geospatial Portals",
            "author": "Dr. Y. Nithiyanandam", "date": "2025-10-24",
            "category": "Geospatial Research", "source": "website",
            "source_name": "Takshashila Website", "source_type": "publication",
            "text": "This publication examines geospatial portal infrastructure and "
                    "policy for defence and civilian mapping data access in India. " * 5,
        },
        {
            "document_id": "website_eco1", "url_hash": "eco1",
            "url": "https://takshashila.org.in/content/publications/economy.html",
            "original_url": "https://takshashila.org.in/content/publications/economy.html",
            "title": "India's Economic Future",
            "author": "Guest Author", "date": "2024-11-10",
            "category": "Economic Policy", "source": "website",
            "source_name": "Takshashila Website", "source_type": "publication",
            "text": "An analysis of India's industrial policy and economic development "
                    "trajectory compared with East Asian peers. " * 5,
        },
    ]

    def fake_crawl_site(site, incremental=True, progress_cb=None):
        # Mimics one real crawl run for whichever source is being asked for.
        if site.source == "website":
            return _fake_crawl_result(website_docs)
        return _fake_crawl_result([])

    with patch("scripts.update_knowledge_base.crawl_site", side_effect=fake_crawl_site), \
         patch("src.vector_store.embed_texts", side_effect=_mock_embed_texts), \
         patch("src.vector_store.embed_query", side_effect=_mock_embed_query), \
         patch("src.embeddings.embed_texts", side_effect=_mock_embed_texts):
        from scripts.update_knowledge_base import run
        summary = run(website=True, commit_kb=True, incremental=True, do_index=True)

    # ── 1. crawl -> merge produced documents.jsonl with both pages ─────────
    from src.utils import load_jsonl
    docs_on_disk = load_jsonl(config.DOCUMENTS_FILE)
    assert {d["document_id"] for d in docs_on_disk} == {"website_geo1", "website_eco1"}
    assert summary["merge"]["added"] == 2
    assert summary["per_source"]["commit_kb"] == {"skipped": "no credentials"}

    # ── 2. chunking actually ran and produced metadata-rich chunks ─────────
    chunks_on_disk = load_jsonl(config.CHUNKS_FILE)
    assert len(chunks_on_disk) >= 2
    assert all(c.get("chunk_id") and c.get("chunk_hash") for c in chunks_on_disk)

    # ── 3. index was built and its vector count matches the chunk count ────
    assert summary["index"]["chunks"] == len(chunks_on_disk)
    assert config.FAISS_INDEX.exists()

    # ── 4. validation ran automatically and found no hard errors ───────────
    assert summary["validation"] is not None
    assert summary["validation"]["ok"], summary["validation"]["errors"]

    # ── 5. retrieval over the freshly built index actually works end-to-end ─
    vector_store._INDEX = None
    vector_store._METADATA = []
    with patch("src.vector_store.embed_texts", side_effect=_mock_embed_texts), \
         patch("src.vector_store.embed_query", side_effect=_mock_embed_query):
        from src.retriever import retrieve
        results = retrieve("geospatial defence mapping", top_k=3)
    assert results
    assert results[0]["document_id"] == "website_geo1"

    # ── 6. a citation-traceable answer path: the top hit's URL is the exact
    #      source page a citation would link to ───────────────────────────
    assert results[0]["url"] == "https://takshashila.org.in/content/publications/geo-portals.html"

    vector_store._INDEX = None
    vector_store._METADATA = []
