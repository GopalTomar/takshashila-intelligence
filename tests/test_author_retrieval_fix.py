"""
tests/test_author_retrieval_fix.py — End-to-end regression test for the exact
reported bug:

    Q1: "Who is Dr Y Nithiyanandam?"                    -> worked
    Q2: "What blogs has Dr Y Nithiyanandam written?"     -> "insufficient evidence"

Root cause (proved in isolation first — see the chat transcript / audit
notes): the evidence gate (has_sufficient_evidence) only trusted FAISS's raw
cosine 'score'. A chunk found ONLY by BM25 — exactly what happens for a
lexically distinctive, name-heavy authorship query that embeds awkwardly —
has no 'score' key at all, so it was silently treated as zero evidence
quality regardless of how precise the actual (author-metadata) match was.

This test builds a REAL FAISS index (via the synthetic_index-style pattern)
with a deliberately adversarial embedding setup: the "who is X" bio chunk
embeds well for query 1, but the "blogs by X" chunk is given a WEAK/absent
embedding match on purpose (simulating the real awkward-phrasing case) while
still being a strong BM25/lexical match — proving the fix works for the
actual mechanism, not just asserting the desired outcome.
"""

import sys
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from src import config


BIO_CHUNK = {
    "document_id": "profile_nithiyanandam", "chunk_id": "profile_nithiyanandam_c0",
    "chunk_hash": "h1", "title": "Dr. Y. Nithiyanandam — Faculty Profile",
    "source": "website", "source_name": "Takshashila Website",
    "category": "Faculty", "author": "", "date": "2023-01-01",
    "url": "https://takshashila.org.in/pages/team/nithiyanandam/",
    "text": "Dr. Y. Nithiyanandam is a senior faculty member leading the "
            "Geospatial programme at Takshashila, with a background in remote sensing.",
}

BLOG_CHUNK_BY_HIM = {
    "document_id": "blog_space_debris", "chunk_id": "blog_space_debris_c0",
    "chunk_hash": "h2", "title": "The Geopolitics of Space Debris",
    "source": "blog", "source_name": "Takshashila Website",
    "category": "Space Policy", "author": "Dr. Y. Nithiyanandam", "date": "2024-05-01",
    "url": "https://takshashila.org.in/pages/blogs/space-debris-geopolitics/",
    "text": "Orbital debris accumulation now threatens satellite infrastructure "
            "essential to modern communication and navigation systems worldwide.",
}

# A document that MERELY MENTIONS him — must never be treated as authored by
# him even though his name appears in the text.
INTERVIEW_CHUNK_MENTIONING_HIM = {
    "document_id": "interview_x", "chunk_id": "interview_x_c0",
    "chunk_hash": "h3", "title": "Interview: The Future of Remote Sensing",
    "source": "blog", "source_name": "Takshashila Website",
    "category": "Interviews", "author": "Guest Author", "date": "2024-02-01",
    "url": "https://takshashila.org.in/pages/blogs/remote-sensing-interview/",
    "text": "In this interview, Dr. Y. Nithiyanandam discusses the future of "
            "remote sensing applications in South Asia.",
}

_DIM = 4
# Bio chunk embeds strongly for "who is" style queries (dim 0).
# The blog chunk deliberately embeds POORLY for the authorship query (all
# zeros) — simulating the real "awkward phrasing" case — so it can ONLY be
# found via BM25, never via FAISS semantic similarity. This is the crux of
# the regression: if the fix only "worked" because the chunk also happened to
# get a good cosine score, it wouldn't actually test the diagnosed mechanism.
_EMBED = {
    BIO_CHUNK["chunk_id"]: [1.0, 0.0, 0.0, 0.0],
    BLOG_CHUNK_BY_HIM["chunk_id"]: [0.0, 0.0, 0.0, 0.0],       # unembeddable on purpose
    INTERVIEW_CHUNK_MENTIONING_HIM["chunk_id"]: [0.0, 0.0, 0.0, 0.0],
}


def _mock_embed_texts(texts, batch_size=64, show_progress=False):
    # texts here are the CHUNK texts at index-build time, matched by content.
    vecs = []
    for t in texts:
        if "senior faculty member" in t:
            vecs.append(_EMBED[BIO_CHUNK["chunk_id"]])
        else:
            vecs.append([0.0, 0.0, 0.0, 0.0])
    arr = np.array(vecs, dtype=np.float32)
    return arr


def _mock_embed_query(query):
    # Every query embeds toward the bio-chunk direction ONLY if it's a "who
    # is" style query — an authorship query ("what blogs...") deliberately
    # embeds as all-zeros (orthogonal to everything), so FAISS alone would
    # find nothing useful for it — forcing this test through the exact BM25
    # only path the real bug involved.
    low = query.lower()
    if "who is" in low:
        return np.array([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32)
    return np.array([[0.0, 0.0, 0.0, 0.0]], dtype=np.float32)


@pytest.fixture()
def author_bug_index(tmp_path, monkeypatch):
    index_dir = tmp_path / "index"
    index_dir.mkdir()
    monkeypatch.setattr(config, "INDEX_DIR", index_dir)
    monkeypatch.setattr(config, "FAISS_INDEX", index_dir / "faiss.index")
    monkeypatch.setattr(config, "METADATA_FILE", index_dir / "metadata.pkl")

    from src import vector_store, retriever
    monkeypatch.setattr(vector_store, "METADATA_JSON", index_dir / "metadata.json")

    chunks = [BIO_CHUNK, BLOG_CHUNK_BY_HIM, INTERVIEW_CHUNK_MENTIONING_HIM]
    with patch("src.vector_store.embed_texts", side_effect=_mock_embed_texts), \
         patch("src.vector_store.embed_query", side_effect=_mock_embed_query):
        vector_store.build_index(chunks=[dict(c) for c in chunks])

    vector_store._INDEX = None
    vector_store._METADATA = []
    retriever._BM25 = None
    retriever._BM25_CHUNKS = []
    retriever._BM25_SIG = None

    with patch("src.vector_store.embed_texts", side_effect=_mock_embed_texts), \
         patch("src.vector_store.embed_query", side_effect=_mock_embed_query):
        yield

    vector_store._INDEX = None
    vector_store._METADATA = []
    retriever._BM25 = None
    retriever._BM25_CHUNKS = []
    retriever._BM25_SIG = None


def test_who_is_query_works(author_bug_index):
    """Q1 from the bug report — sanity check this already worked."""
    from src.retriever import retrieve, has_sufficient_evidence
    chunks = retrieve("Who is Dr Y Nithiyanandam?", top_k=5)
    assert chunks
    assert has_sufficient_evidence(chunks)
    assert chunks[0]["document_id"] == "profile_nithiyanandam"


def test_authorship_query_now_succeeds_via_author_metadata_match(author_bug_index):
    """
    Q2 from the bug report. Before the fix: FAISS finds nothing useful
    (mocked to embed as zero-vectors for this phrasing), so the ONLY way this
    chunk surfaces at all is BM25 on "blogs"/"Nithiyanandam"/"written". Before
    the fix, has_sufficient_evidence() would see best_cosine()==0.0 and
    incorrectly report insufficient evidence. After the fix, the verified
    author-metadata match makes this chunk (and the query) pass the gate.
    """
    from src.retriever import retrieve, has_sufficient_evidence, confidence_level

    chunks = retrieve("What blogs has Dr Y Nithiyanandam written?", top_k=5)
    assert chunks, "retrieval returned nothing at all — BM25 should have found the blog chunk"

    by_him = [c for c in chunks if c["document_id"] == "blog_space_debris"]
    assert by_him, "the chunk he actually authored was not retrieved at all"
    assert by_him[0]["author_verified"] is True

    assert has_sufficient_evidence(chunks), (
        "regression: the exact reported bug — author-matched evidence found "
        "only via BM25 is being rejected by the evidence gate"
    )
    assert confidence_level(chunks) in ("medium", "high")


def test_authored_content_ranks_above_merely_mentioning_content(author_bug_index):
    """Precision guarantee: the interview that merely MENTIONS him in body
    text must not outrank (or be conflated with) the blog he actually wrote."""
    from src.retriever import retrieve

    chunks = retrieve("What blogs has Dr Y Nithiyanandam written?", top_k=5)
    doc_ids_in_order = [c["document_id"] for c in chunks]

    if "interview_x" in doc_ids_in_order and "blog_space_debris" in doc_ids_in_order:
        assert doc_ids_in_order.index("blog_space_debris") < doc_ids_in_order.index("interview_x")

    interview_hits = [c for c in chunks if c["document_id"] == "interview_x"]
    for c in interview_hits:
        assert c["author_verified"] is False, (
            "a document that only MENTIONS the person must never be marked "
            "author-verified just because the name appears in its text"
        )


def test_unrelated_query_is_unaffected_by_the_fix(author_bug_index):
    """The fix must not make the gate more permissive for queries that have
    no author-match signal at all — precision for other queries is preserved."""
    from src.retriever import retrieve, has_sufficient_evidence

    chunks = retrieve("completely unrelated topic with no name in it", top_k=5)
    # No candidate name extracted, no author match possible, and the mock
    # embeddings give this query zero similarity to everything -> still
    # correctly reports insufficient evidence.
    assert not has_sufficient_evidence(chunks)
