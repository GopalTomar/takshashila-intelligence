"""
test_bm25_evidence.py — regression for the ATP false-refusal bug (Section 6).

The original hybrid retriever assigned a cosine score of 0 to any BM25-only hit,
so a query whose best evidence was a precise LEXICAL match ("ATP", "All Things
Policy") failed the evidence gate and was refused as "insufficient evidence",
even though the fact was in the knowledge base.

These tests build a real FAISS index in which the query "What is ATP?" has a LOW
dense-cosine similarity to the relevant document but a STRONG lexical (BM25)
match, and assert:
  * the relevant chunk is retrieved and carries a real, non-zero lexical_score,
  * has_sufficient_evidence() returns True on the strength of the lexical match,
  * a purely irrelevant query is still correctly refused (no threshold-lowering
    that would invite hallucination).
"""

import sys
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from src import config, vector_store, retriever  # noqa: E402

# Dense embedding keyed only on generic topic words — deliberately NOT on the
# acronym "atp", so the acronym query is a weak dense match but a strong lexical
# one (exactly the real-world ATP situation).
_DENSE_KEYS = ["policy", "podcast", "economy", "geospatial"]


def _mock_embed_texts(texts, batch_size=64, show_progress=False):
    vecs = np.zeros((len(texts), len(_DENSE_KEYS)), dtype=np.float32)
    for i, t in enumerate(texts):
        low = t.lower()
        for j, kw in enumerate(_DENSE_KEYS):
            if kw in low:
                vecs[i, j] = 1.0
        n = np.linalg.norm(vecs[i])
        if n > 0:
            vecs[i] = vecs[i] / n
    return vecs


def _mock_embed_query(query):
    return _mock_embed_texts([query])


_CHUNKS = [
    {
        "document_id": "atp_1", "chunk_id": "atp_1_c0", "chunk_index": 0,
        "chunk_hash": "hash_atp_1",
        "title": "All Things Policy (ATP) — Podcast", "source": "website",
        "source_name": "Takshashila Website", "category": "Podcasts",
        "url": "https://takshashila.org.in/all-things-policy/",
        # Contains the literal token "ATP" and "All Things Policy" for BM25, but
        # the dense keys ("podcast", "policy") are shared with other docs so the
        # acronym query "what is atp" is not a strong *dense* match.
        "text": "All Things Policy (ATP) is Takshashila's daily public policy "
                "podcast. On ATP, researchers discuss current affairs and policy.",
    },
    {
        "document_id": "eco_1", "chunk_id": "eco_1_c0", "chunk_index": 0,
        "chunk_hash": "hash_eco_1",
        "title": "India's Economy", "source": "website",
        "source_name": "Takshashila Website", "category": "Economics",
        "url": "https://takshashila.org.in/economy/",
        "text": "An analysis of the Indian economy and industrial policy choices.",
    },
    {
        "document_id": "geo_1", "chunk_id": "geo_1_c0", "chunk_index": 0,
        "chunk_hash": "hash_geo_1",
        "title": "Geospatial Portals", "source": "website",
        "source_name": "Takshashila Website", "category": "Geospatial",
        "url": "https://takshashila.org.in/geospatial/",
        "text": "A report on India's geospatial data portals and mapping policy.",
    },
]


@pytest.fixture()
def atp_index(tmp_path, monkeypatch):
    index_dir = tmp_path / "index"
    index_dir.mkdir()
    monkeypatch.setattr(config, "INDEX_DIR", index_dir)
    monkeypatch.setattr(config, "FAISS_INDEX", index_dir / "faiss.index")
    monkeypatch.setattr(config, "METADATA_FILE", index_dir / "metadata.pkl")
    monkeypatch.setattr(config, "INDEX_VERSIONS_DIR", index_dir / "versions")
    monkeypatch.setattr(config, "KB_MANIFEST_FILE", index_dir / "current.json")
    monkeypatch.setattr(vector_store, "METADATA_JSON", index_dir / "metadata.json")
    for mod in (vector_store,):
        mod._INDEX = None
        mod._METADATA = []
        mod._LOADED_FINGERPRINT = None
    retriever._BM25 = retriever._BM25_SIG = None
    retriever._BM25_CHUNKS = []
    with patch("src.vector_store.embed_texts", side_effect=_mock_embed_texts), \
         patch("src.vector_store.embed_query", side_effect=_mock_embed_query):
        vector_store.build_index(chunks=[dict(c) for c in _CHUNKS])
        vector_store._INDEX = None
        vector_store._METADATA = []
        vector_store._LOADED_FINGERPRINT = None
        retriever._BM25 = retriever._BM25_SIG = None
        retriever._BM25_CHUNKS = []
        yield
    vector_store._INDEX = None
    vector_store._METADATA = []
    vector_store._LOADED_FINGERPRINT = None
    retriever._BM25 = retriever._BM25_SIG = None
    retriever._BM25_CHUNKS = []


def test_atp_lexical_match_counts_as_evidence(atp_index):
    with patch("src.vector_store.embed_texts", side_effect=_mock_embed_texts), \
         patch("src.vector_store.embed_query", side_effect=_mock_embed_query):
        chunks = retriever.retrieve("What is ATP?", top_k=5, use_hybrid=True)

    assert chunks, "retrieval returned nothing"
    atp = next((c for c in chunks if c["document_id"] == "atp_1"), None)
    assert atp is not None, "the ATP document was not retrieved"

    # The three signals are kept SEPARATE and the lexical one is real & non-zero.
    assert atp.get("lexical_score", 0.0) > 0.0, "lexical_score should be populated"
    assert retriever.best_lexical(chunks) >= config.LEXICAL_MIN_SCORE

    # The bug: this used to be False because the BM25-only hit was scored 0 cosine.
    assert retriever.has_sufficient_evidence(chunks) is True
    assert retriever.confidence_level(chunks) != "none"


def test_irrelevant_query_still_refused(atp_index):
    """Guardrail: the lexical path must not turn every query into a match."""
    with patch("src.vector_store.embed_texts", side_effect=_mock_embed_texts), \
         patch("src.vector_store.embed_query", side_effect=_mock_embed_query):
        chunks = retriever.retrieve(
            "quantum chromodynamics lattice gauge theory", top_k=5, use_hybrid=True
        )
    # No lexical overlap with the corpus → no strong lexical evidence.
    assert retriever.has_strong_lexical_evidence(chunks) is False
    assert retriever.has_sufficient_evidence(chunks) is False
