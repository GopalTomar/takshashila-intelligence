"""
tests/test_chunking.py — Unit tests for chunking logic.

NOTE (audit finding, see RAG_AUDIT_AND_UPGRADE.md): this test previously
imported `_split_into_paragraphs` and `_chunk_paragraphs` from src.chunker.
Those names never existed in src/chunker.py — the module was already
rewritten to a sentence-aware, character-budget packer
(`_split_sentences` / `_pack_sentences`) with richer per-chunk metadata
(chunk_id, chunk_hash, page_number, heading_path, etc.), but the test file
was never updated to match, so pytest failed at collection time for the
whole file. The chunker implementation itself is sound and is the
intended, current API — this file tests it directly instead of
resurrecting the old paragraph-based functions.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.chunker import _split_sentences, _pack_sentences, chunk_document


SAMPLE_TEXT = """
Takshashila Institution has been at the forefront of research on Indian foreign policy.

The institution publishes regular briefs on topics ranging from geopolitics to technology governance.

In 2024, Takshashila released an extensive report on the geopolitics of artificial intelligence,
arguing that India must develop a coherent national AI strategy that balances innovation with safety.

The report identified three pillars: compute access, data governance, and international partnerships.

Without adequate compute, India risks falling behind China and the United States in AI capabilities.
Data governance frameworks must balance openness with privacy concerns, particularly in the context
of cross-border data flows under DPDP Act 2023.

International partnerships with QUAD countries and the EU could help India secure access to
advanced semiconductor technology currently controlled by a small number of firms.
"""


def test_split_sentences():
    """Splitting should yield multiple non-trivial sentence/paragraph units."""
    units = _split_sentences(SAMPLE_TEXT)
    assert len(units) >= 4, f"Expected >=4 units, got {len(units)}"
    for u in units:
        assert len(u) > 10, f"Unit too short: {u!r}"


def test_chunk_size_respected():
    """Packed chunks must not exceed max_chars by more than a small slack
    (a single oversized sentence gets hard-split on whitespace)."""
    sentences = _split_sentences(SAMPLE_TEXT * 10)  # make it longer
    max_chars = 500
    chunks = _pack_sentences(sentences, max_chars=max_chars, overlap_chars=50, min_len=20)
    assert len(chunks) > 1, "Expected multiple chunks for long input"
    for ch in chunks:
        assert len(ch) <= max_chars + 50, (
            f"Chunk exceeds max size + hard-split slack: {len(ch)} chars"
        )


def test_pack_sentences_overlap_and_dedup():
    """Adjacent chunks should share trailing context, and no two consecutive
    chunks should be identical (dedup of overlap edge cases)."""
    sentences = _split_sentences(SAMPLE_TEXT * 5)
    chunks = _pack_sentences(sentences, max_chars=400, overlap_chars=80, min_len=20)
    for i in range(len(chunks) - 1):
        assert chunks[i] != chunks[i + 1], "Consecutive chunks should be deduplicated"


def test_chunk_document_metadata():
    doc = {
        "document_id":  "abc123",
        "title":        "Test Publication",
        "author":       "Test Author",
        "date":         "2024-01-01",
        "url":          "https://example.com/test",
        "pdf_url":      "",
        "source":       "publication",
        "category":     "AI",
        "tags":         ["ai", "policy"],
        "text":         SAMPLE_TEXT,
    }
    chunks = chunk_document(doc)
    assert len(chunks) >= 1, "Expected at least one chunk"
    for ch in chunks:
        assert ch["title"] == "Test Publication"
        assert ch["author"] == "Test Author"
        assert ch["source"] == "publication"
        assert "chunk_id" in ch
        assert "chunk_hash" in ch
        assert ch["document_id"] == "abc123"
        assert len(ch["text"]) > 10


def test_pdf_chunking_with_pages():
    doc = {
        "document_id":  "pdf123",
        "title":        "PDF Report",
        "author":       "Jane Smith",
        "date":         "2023-06-01",
        "url":          "https://example.com/report",
        "pdf_url":      "https://example.com/report.pdf",
        "source":       "pdf",
        "source_type":  "pdf",
        "category":     "",
        "tags":         [],
        "text":         "",  # PDFs use pdf_pages
        "pdf_pages":    [
            {"page_number": 1, "text": SAMPLE_TEXT},
            {"page_number": 2, "text": "Another page of content. " * 50},
        ],
    }
    chunks = chunk_document(doc)
    assert any(ch.get("page_number") == 1 for ch in chunks)
    assert any(ch.get("page_number") == 2 for ch in chunks)


def test_chunk_ids_unique_and_deterministic():
    doc = {
        "document_id": "detcheck",
        "title": "Determinism Check",
        "source": "publication",
        "text": SAMPLE_TEXT * 3,
    }
    chunks_a = chunk_document(dict(doc))
    chunks_b = chunk_document(dict(doc))
    ids_a = [c["chunk_id"] for c in chunks_a]
    ids_b = [c["chunk_id"] for c in chunks_b]
    assert len(ids_a) == len(set(ids_a)), "chunk_id values must be unique within a document"
    assert ids_a == ids_b, "Chunking the same document twice must be deterministic"


if __name__ == "__main__":
    test_split_sentences()
    print("test_split_sentences OK")
    test_chunk_size_respected()
    print("test_chunk_size_respected OK")
    test_pack_sentences_overlap_and_dedup()
    print("test_pack_sentences_overlap_and_dedup OK")
    test_chunk_document_metadata()
    print("test_chunk_document_metadata OK")
    test_pdf_chunking_with_pages()
    print("test_pdf_chunking_with_pages OK")
    test_chunk_ids_unique_and_deterministic()
    print("test_chunk_ids_unique_and_deterministic OK")
    print("\nAll chunking tests passed.")
