"""
tests/conftest.py — Shared fixtures for offline retrieval/index tests.

``synthetic_index`` builds a REAL FAISS index (not a mock of FAISS itself)
from a small, known set of chunks, using a deterministic keyword-bucket
"embedding" function instead of the real sentence-transformers model — so
these tests need no network access and no ~130MB model download, while still
exercising the actual src.vector_store / src.retriever code paths end to end
(index build, save, load, cosine search, hybrid BM25+FAISS fusion, dedup,
metadata filters).

This is what lets tests/test_retrieval.py verify real behaviour instead of
either (a) requiring a live-built index that isn't shipped in this repo, or
(b) being skipped/dismissed because no index exists.
"""

import sys
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from src import config

# Keyword → one-hot dimension. The mock embedding of a text is the (normalized)
# sum of one-hot vectors for every keyword it contains, so unrelated texts are
# orthogonal (cosine ~0) and related texts score high — deterministic and
# meaningful without a real model.
_KEYWORDS = ["geospatial", "economy", "defence", "aviation"]
_DIM = len(_KEYWORDS)


def _mock_embed_texts(texts, batch_size=64, show_progress=False):
    vecs = np.zeros((len(texts), _DIM), dtype=np.float32)
    for i, t in enumerate(texts):
        low = t.lower()
        for j, kw in enumerate(_KEYWORDS):
            if kw in low:
                vecs[i, j] = 1.0
        norm = np.linalg.norm(vecs[i])
        if norm > 0:
            vecs[i] = vecs[i] / norm
        # else: leave as an all-zero vector — genuinely "no keyword match",
        # which correctly yields cosine similarity 0 against every real chunk.
    return vecs


def _mock_embed_query(query):
    return _mock_embed_texts([query])


SAMPLE_CHUNKS = [
    {
        "document_id": "doc_geo_1", "chunk_id": "doc_geo_1_c0", "chunk_index": 0,
        "chunk_hash": "hash_geo_1",
        "title": "State of India's Geospatial Portals", "source": "publication",
        "source_name": "Takshashila Website", "category": "Geospatial Research",
        "author": "Dr. Y. Nithiyanandam", "date": "2025-10-24",
        "url": "https://takshashila.org.in/content/publications/geospatial-portals.html",
        "text": "This report examines the geospatial data ecosystem in India, covering "
                "portal infrastructure and access policy for spatial datasets.",
    },
    {
        "document_id": "doc_eco_1", "chunk_id": "doc_eco_1_c0", "chunk_index": 0,
        "chunk_hash": "hash_eco_1",
        "title": "India's Economic Future", "source": "publication",
        "source_name": "Takshashila Website", "category": "Economic Policy",
        "author": "Guest Author", "date": "2024-11-10",
        "url": "https://takshashila.org.in/content/publications/economic-future.html",
        "text": "An analysis of India's economy and industrial policy, drawing lessons "
                "from South Korea's economic development playbook.",
    },
    {
        "document_id": "doc_def_1", "chunk_id": "doc_def_1_c0", "chunk_index": 0,
        "chunk_hash": "hash_def_1",
        "title": "Defence Modernisation Brief", "source": "commit_kb",
        "source_name": "Commit KB", "category": "Strategic Studies",
        "author": "Staff Analyst", "date": "2025-03-01",
        "url": "https://takshashila.org.in/pages/publications/defence-brief/",
        "text": "This brief covers defence procurement reform and military modernisation "
                "priorities for the Indian armed forces.",
    },
    # A near-duplicate chunk from the SAME document as doc_geo_1, to exercise
    # per-document dedup in the retriever.
    {
        "document_id": "doc_geo_1", "chunk_id": "doc_geo_1_c1", "chunk_index": 1,
        "chunk_hash": "hash_geo_1b",
        "title": "State of India's Geospatial Portals", "source": "publication",
        "source_name": "Takshashila Website", "category": "Geospatial Research",
        "author": "Dr. Y. Nithiyanandam", "date": "2025-10-24",
        "url": "https://takshashila.org.in/content/publications/geospatial-portals.html",
        "text": "Aviation-related geospatial applications, such as airspace mapping, "
                "are discussed as a downstream use case for the same portals.",
    },
]


@pytest.fixture()
def synthetic_index(tmp_path, monkeypatch):
    """Build a real, small FAISS index under a temp dir with mocked embeddings,
    monkeypatch config + module caches to point at it, and yield the chunks
    used to build it. Cleans up the module-level FAISS/BM25 caches afterward
    so this fixture never leaks state into other tests."""
    index_dir = tmp_path / "index"
    index_dir.mkdir()

    monkeypatch.setattr(config, "INDEX_DIR", index_dir)
    monkeypatch.setattr(config, "FAISS_INDEX", index_dir / "faiss.index")
    monkeypatch.setattr(config, "METADATA_FILE", index_dir / "metadata.pkl")
    # Versioning: keep the manifest + snapshots inside the temp dir so tests
    # never touch the real repo's data/index.
    monkeypatch.setattr(config, "INDEX_VERSIONS_DIR", index_dir / "versions")
    monkeypatch.setattr(config, "KB_MANIFEST_FILE", index_dir / "current.json")

    from src import vector_store, retriever
    monkeypatch.setattr(vector_store, "METADATA_JSON", index_dir / "metadata.json")
    vector_store._LOADED_FINGERPRINT = None

    with patch("src.vector_store.embed_texts", side_effect=_mock_embed_texts), \
         patch("src.vector_store.embed_query", side_effect=_mock_embed_query):
        vector_store.build_index(chunks=[dict(c) for c in SAMPLE_CHUNKS])

    # Reset caches so this test's index/BM25 don't leak into other tests.
    vector_store._INDEX = None
    vector_store._METADATA = []
    vector_store._LOADED_FINGERPRINT = None
    retriever._BM25 = None
    retriever._BM25_CHUNKS = []
    retriever._BM25_SIG = None

    with patch("src.vector_store.embed_texts", side_effect=_mock_embed_texts), \
         patch("src.vector_store.embed_query", side_effect=_mock_embed_query):
        yield SAMPLE_CHUNKS

    vector_store._INDEX = None
    vector_store._METADATA = []
    vector_store._LOADED_FINGERPRINT = None
    retriever._BM25 = None
    retriever._BM25_CHUNKS = []
    retriever._BM25_SIG = None
