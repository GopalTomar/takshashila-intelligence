"""
tests/test_ingestion_consistency.py — Offline tests for the dedup / merge /
validation layer (src/incremental_index.py, scripts/validate_kb.py).

Everything here runs against synthetic documents.jsonl / chunks.jsonl / FAISS
index files under a pytest tmp_path — no live crawl, no real embedding model,
no network. See conftest.py's synthetic_index fixture for the FAISS piece
shared with tests/test_retrieval.py.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils import load_jsonl, save_jsonl


# ── collapse_by_url ───────────────────────────────────────────────────────────

def test_collapse_by_url_merges_www_and_trailing_slash_variants():
    from src.incremental_index import collapse_by_url
    docs = [
        {"document_id": "a", "url": "https://www.takshashila.org.in/blogs/x/",
         "title": "X", "text": "short"},
        {"document_id": "b", "url": "https://takshashila.org.in/blogs/x",
         "title": "X", "author": "Jane", "date": "2025-01-01", "text": "a longer richer body"},
    ]
    kept, dropped = collapse_by_url(docs)
    assert dropped == 1
    assert len(kept) == 1
    # the richer copy (has author + date + longer text) must win
    assert kept[0]["document_id"] == "b"


def test_collapse_by_url_strips_tracking_params():
    from src.incremental_index import collapse_by_url
    docs = [
        {"document_id": "a", "url": "https://takshashila.org.in/blogs/x?utm_source=twitter",
         "title": "X", "text": "one"},
        {"document_id": "b", "url": "https://takshashila.org.in/blogs/x?utm_source=linkedin",
         "title": "X", "text": "two"},
    ]
    kept, dropped = collapse_by_url(docs)
    assert dropped == 1
    assert len(kept) == 1


def test_collapse_by_url_keeps_genuinely_distinct_pages():
    from src.incremental_index import collapse_by_url
    docs = [
        {"document_id": "a", "url": "https://takshashila.org.in/blogs/x", "text": "one"},
        {"document_id": "b", "url": "https://takshashila.org.in/blogs/y", "text": "two"},
    ]
    kept, dropped = collapse_by_url(docs)
    assert dropped == 0
    assert len(kept) == 2


def test_collapse_by_url_prefers_html_over_pdf_twin():
    from src.incremental_index import collapse_by_url
    docs = [
        {"document_id": "a_pdf", "url": "https://takshashila.org.in/report",
         "source_type": "pdf", "pdf_url": "https://takshashila.org.in/report.pdf",
         "title": "Report [PDF]", "text": "pdf body"},
        {"document_id": "a_html", "url": "https://takshashila.org.in/report",
         "title": "Report", "text": "html body"},
    ]
    kept, dropped = collapse_by_url(docs)
    assert dropped == 1
    assert kept[0]["document_id"] == "a_html"


# ── merge_documents ────────────────────────────────────────────────────────

def test_merge_documents_add_update_remove(tmp_path):
    from src.incremental_index import merge_documents
    doc_file = tmp_path / "documents.jsonl"
    save_jsonl(doc_file, [
        {"document_id": "keep_me", "url": "https://takshashila.org.in/a", "text": "unchanged"},
        {"document_id": "will_be_updated", "url": "https://takshashila.org.in/b", "text": "old text"},
        {"document_id": "will_be_removed", "url": "https://takshashila.org.in/c", "text": "gone soon"},
    ])

    summary = merge_documents(
        new_or_changed=[
            {"document_id": "will_be_updated", "url": "https://takshashila.org.in/b", "text": "new text"},
            {"document_id": "brand_new", "url": "https://takshashila.org.in/d", "text": "fresh"},
        ],
        removed_ids=["will_be_removed"],
        documents_file=doc_file,
    )

    assert summary["added"] == 1
    assert summary["updated"] == 1
    assert summary["removed"] == 1

    result = {d["document_id"]: d for d in load_jsonl(doc_file)}
    assert set(result.keys()) == {"keep_me", "will_be_updated", "brand_new"}
    assert "will_be_removed" not in result  # a removed doc must not survive
    assert result["will_be_updated"]["text"] == "new text"  # changed doc must not keep stale text


def test_merge_documents_removal_leaves_no_stale_entry(tmp_path):
    """Direct check against spec point 26: a removed URL must not remain
    retrievable — i.e. must not still be present in documents.jsonl at all."""
    from src.incremental_index import merge_documents
    doc_file = tmp_path / "documents.jsonl"
    save_jsonl(doc_file, [{"document_id": "x", "url": "https://takshashila.org.in/x", "text": "t"}])
    merge_documents(new_or_changed=[], removed_ids=["x"], documents_file=doc_file)
    remaining = load_jsonl(doc_file)
    assert remaining == []


# ── validate_kb.py ────────────────────────────────────────────────────────

def _patch_kb_paths(monkeypatch, tmp_path, docs, chunks):
    from src import config
    doc_file = tmp_path / "documents.jsonl"
    chunk_file = tmp_path / "chunks.jsonl"
    save_jsonl(doc_file, docs)
    save_jsonl(chunk_file, chunks)
    monkeypatch.setattr(config, "DOCUMENTS_FILE", doc_file)
    monkeypatch.setattr(config, "CHUNKS_FILE", chunk_file)
    monkeypatch.setattr(config, "FAISS_INDEX", tmp_path / "nonexistent.index")
    return doc_file, chunk_file


def test_validate_kb_flags_duplicate_urls_after_normalization(monkeypatch, tmp_path):
    """Regression test for the audit finding: validate_kb used to compare RAW
    URL strings, so a www./trailing-slash/utm_source duplicate would NOT be
    reported. It must be reported now."""
    from scripts.validate_kb import validate
    docs = [
        {"document_id": "a", "url": "https://www.takshashila.org.in/blogs/x/",
         "title": "X", "author": "A", "date": "2025-01-01", "text": "body one"},
        {"document_id": "b", "url": "https://takshashila.org.in/blogs/x?utm_source=twitter",
         "title": "X", "author": "A", "date": "2025-01-01", "text": "body two"},
    ]
    _patch_kb_paths(monkeypatch, tmp_path, docs, chunks=[])
    report = validate()
    assert any("duplicate document URL" in e for e in report["errors"])


def test_validate_kb_no_false_positive_on_distinct_urls(monkeypatch, tmp_path):
    from scripts.validate_kb import validate
    docs = [
        {"document_id": "a", "url": "https://takshashila.org.in/blogs/x",
         "title": "X", "author": "A", "date": "2025-01-01", "text": "body one"},
        {"document_id": "b", "url": "https://takshashila.org.in/blogs/y",
         "title": "Y", "author": "A", "date": "2025-01-01", "text": "body two"},
    ]
    _patch_kb_paths(monkeypatch, tmp_path, docs, chunks=[])
    report = validate()
    assert not any("duplicate document URL" in e for e in report["errors"])


def test_validate_kb_flags_orphan_chunks(monkeypatch, tmp_path):
    from scripts.validate_kb import validate
    docs = [{"document_id": "a", "url": "https://takshashila.org.in/a",
             "title": "A", "author": "X", "date": "2025-01-01", "text": "body"}]
    chunks = [{"document_id": "a", "chunk_id": "a_c0", "text": "chunk of a"},
              {"document_id": "does_not_exist", "chunk_id": "z_c0", "text": "orphan chunk"}]
    _patch_kb_paths(monkeypatch, tmp_path, docs, chunks)
    report = validate()
    assert any("orphan chunk" in e for e in report["errors"])


def test_validate_kb_flags_missing_index():
    from scripts.validate_kb import validate
    # FAISS_INDEX left pointing at a real config path that (in this sandbox)
    # has no built index — validate() must report it as an error, not crash.
    report = validate()
    assert any("index" in e.lower() for e in report["errors"]) or report["counts"]["index_vectors"] == -1


def test_validate_kb_passes_on_a_clean_kb(monkeypatch, tmp_path, synthetic_index):
    """End-to-end: a KB built via the synthetic_index fixture (real FAISS
    index + matching chunks) should validate cleanly with no ERRORs."""
    from scripts.validate_kb import validate
    from src import config
    docs = [
        {"document_id": did, "url": f"https://takshashila.org.in/{did}",
         "title": "T", "author": "A", "date": "2025-01-01", "text": "x" * 50}
        for did in {c["document_id"] for c in synthetic_index}
    ]
    doc_file = tmp_path / "documents.jsonl"
    chunk_file = tmp_path / "chunks.jsonl"
    save_jsonl(doc_file, docs)
    save_jsonl(chunk_file, synthetic_index)
    monkeypatch.setattr(config, "DOCUMENTS_FILE", doc_file)
    monkeypatch.setattr(config, "CHUNKS_FILE", chunk_file)
    # FAISS_INDEX/METADATA_FILE are already pointed at the synthetic index's
    # temp dir by the synthetic_index fixture's own monkeypatching.
    report = validate()
    assert report["errors"] == [], report["errors"]
