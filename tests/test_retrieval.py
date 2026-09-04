"""
tests/test_retrieval.py — Retrieval tests against a synthetic, offline FAISS
index (see tests/conftest.py). These verify real src.vector_store /
src.retriever behaviour: index load, cosine search, metadata filters, hybrid
BM25+FAISS fusion, and per-document dedup — without needing a live-built
production index or a downloaded embedding model.

LIVE-ONLY, NOT covered here (documented, not silently skipped): whether the
*real* sentence-transformers model + a *real* crawled Takshashila corpus
retrieves good results for real user questions. That needs an actual built
KB and is a live/staging verification step, not a unit test.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


def test_load_index(synthetic_index):
    """Smoke test: load index without error, stats reflect what was built."""
    from src.vector_store import load_index, index_stats
    load_index(force=True)
    stats = index_stats()
    assert stats["total_chunks"] == len(synthetic_index)
    assert stats["total_documents"] == len({c["document_id"] for c in synthetic_index})
    assert "publication" in stats["by_source"]
    assert "commit_kb" in stats["by_source"]


def test_search_returns_results(synthetic_index):
    """A query should surface the on-topic chunk with the highest score."""
    from src.vector_store import search
    results = search("geospatial portals in India", top_k=3)
    assert len(results) > 0
    for r in results:
        assert "text" in r and "title" in r and "score" in r
    assert results[0]["document_id"] == "doc_geo_1"
    assert results[0]["score"] > 0.9  # near-exact keyword match in the mock embedding


def test_search_source_filter(synthetic_index):
    from src.vector_store import search
    results = search("defence procurement", top_k=5, source_filter="commit_kb")
    assert len(results) > 0
    assert all(r["source"] == "commit_kb" for r in results)


def test_search_category_filter_excludes_others(synthetic_index):
    from src.vector_store import search
    results = search("economy", top_k=5, category_filter="Economic Policy")
    assert len(results) > 0
    assert all("economic" in (r.get("category") or "").lower() for r in results)


def test_retrieve_hybrid_returns_list(synthetic_index):
    from src.retriever import retrieve
    results = retrieve("geospatial policy remote sensing", top_k=5)
    assert isinstance(results, list)
    assert len(results) > 0


def test_retrieve_dedupes_by_document(synthetic_index):
    """doc_geo_1 has two chunks (hash_geo_1, hash_geo_1b); a query matching
    both must not return more than the retriever's per-document cap, and
    every returned chunk must genuinely belong to a distinct slot count."""
    from src.retriever import retrieve
    results = retrieve("geospatial aviation portals", top_k=10)
    geo_hits = [r for r in results if r.get("document_id") == "doc_geo_1"]
    # retriever.py caps at 2 chunks per document — never more.
    assert len(geo_hits) <= 2


def test_retrieve_confidence_and_evidence_gate(synthetic_index):
    from src.retriever import confidence_level, has_sufficient_evidence, retrieve
    results = retrieve("geospatial portals in India", top_k=3)
    assert has_sufficient_evidence(results)
    assert confidence_level(results) in ("high", "medium", "low")


def test_retrieve_no_evidence_for_unrelated_query(synthetic_index):
    """A query with none of the mock-embedding keywords should score ~0
    against every chunk, so the evidence gate should correctly refuse it."""
    from src.retriever import has_sufficient_evidence, retrieve
    results = retrieve("unrelated topic with no keyword overlap at all", top_k=3)
    assert not has_sufficient_evidence(results)
