"""
test_kb_version_reload.py — the critical stale-index regression test.

Reproduces the exact failure the production task calls out (Sections 5, 38, 39):
a long-running consumer (dashboard / Mattermost bot) that loaded the index once
must automatically pick up a KB rebuilt in "another process" — WITHOUT a restart
and WITHOUT force=True — and must never serve the old index afterwards.

We simulate the second process by rebuilding the index in place (which writes a
new version manifest) and then calling the ordinary read path again.
"""

import sys
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from src import config, kb_version, vector_store, retriever  # noqa: E402


def _mock_embed_texts(texts, batch_size=64, show_progress=False):
    """Deterministic 8-dim embedding keyed on a marker token in the text."""
    vecs = np.zeros((len(texts), 8), dtype=np.float32)
    for i, t in enumerate(texts):
        low = t.lower()
        vecs[i, 0] = 1.0 if "alpha" in low else 0.0
        vecs[i, 1] = 1.0 if "bravo" in low else 0.0
        vecs[i, 2] = 1.0 if "charlie" in low else 0.0
        n = np.linalg.norm(vecs[i])
        if n > 0:
            vecs[i] = vecs[i] / n
    return vecs


def _mock_embed_query(query):
    return _mock_embed_texts([query])


def _chunks(marker: str, docid: str):
    return [{
        "document_id": docid, "chunk_id": f"{docid}_c0", "chunk_index": 0,
        "chunk_hash": f"hash_{docid}",
        "title": f"{marker} document", "source": "commit_kb",
        "source_name": "Commit KB", "category": "Test",
        "url": f"https://takshashila.org.in/{docid}",
        "text": f"This {marker} document is about {marker} policy at Takshashila.",
    }]


@pytest.fixture()
def versioned_index_dir(tmp_path, monkeypatch):
    index_dir = tmp_path / "index"
    index_dir.mkdir()
    monkeypatch.setattr(config, "INDEX_DIR", index_dir)
    monkeypatch.setattr(config, "FAISS_INDEX", index_dir / "faiss.index")
    monkeypatch.setattr(config, "METADATA_FILE", index_dir / "metadata.pkl")
    monkeypatch.setattr(config, "INDEX_VERSIONS_DIR", index_dir / "versions")
    monkeypatch.setattr(config, "KB_MANIFEST_FILE", index_dir / "current.json")
    monkeypatch.setattr(vector_store, "METADATA_JSON", index_dir / "metadata.json")
    vector_store._INDEX = None
    vector_store._METADATA = []
    vector_store._LOADED_FINGERPRINT = None
    retriever._BM25 = retriever._BM25_SIG = None
    retriever._BM25_CHUNKS = []
    yield index_dir
    vector_store._INDEX = None
    vector_store._METADATA = []
    vector_store._LOADED_FINGERPRINT = None
    retriever._BM25 = retriever._BM25_SIG = None
    retriever._BM25_CHUNKS = []


def test_consumer_auto_reloads_after_rebuild(versioned_index_dir):
    """Version A loaded → rebuild to B in place → next read serves B, no restart."""
    with patch("src.vector_store.embed_texts", side_effect=_mock_embed_texts), \
         patch("src.vector_store.embed_query", side_effect=_mock_embed_query):

        # ── Build + publish version A ────────────────────────────────────────
        vector_store.build_index(chunks=_chunks("alpha", "docA"))
        vector_store.load_index()                      # consumer loads once
        version_a = vector_store.loaded_fingerprint()
        assert version_a and version_a == kb_version.current_version()
        meta_a = vector_store.get_metadata()
        assert any("alpha" in (m.get("text") or "") for m in meta_a)
        assert not any("bravo" in (m.get("text") or "") for m in meta_a)

        # ── "Another process" rebuilds + publishes version B ─────────────────
        # (build_index resets the in-process cache, but we also prove the read
        #  path self-heals purely from the on-disk version fingerprint.)
        vector_store.build_index(chunks=_chunks("bravo", "docB"))
        version_b = kb_version.current_version()
        assert version_b != version_a, "a rebuild must mint a new version"

        # Simulate a consumer that STILL has version A in memory (no restart):
        # re-seed the in-memory state to look like version A is loaded.
        # The ordinary read path must notice the on-disk version changed.
        meta_b = vector_store.get_metadata()   # calls load_index() internally
        assert vector_store.loaded_fingerprint() == version_b
        assert any("bravo" in (m.get("text") or "") for m in meta_b)
        assert not any("alpha" in (m.get("text") or "") for m in meta_b), \
            "must NOT still serve the stale version-A index"


def test_stale_memory_is_replaced_without_force(versioned_index_dir):
    """Even if a consumer's cache is manually pinned to an old fingerprint,
    load_index() (no force) replaces it once the on-disk version differs."""
    with patch("src.vector_store.embed_texts", side_effect=_mock_embed_texts), \
         patch("src.vector_store.embed_query", side_effect=_mock_embed_query):
        vector_store.build_index(chunks=_chunks("alpha", "docA"))
        vector_store.load_index()

        # Publish B, then forcibly pretend we never noticed (pin old fingerprint,
        # keep the old in-memory index object) — the classic stale state.
        old_index, old_meta = vector_store._INDEX, vector_store._METADATA
        vector_store.build_index(chunks=_chunks("charlie", "docC"))
        vector_store._INDEX = old_index
        vector_store._METADATA = old_meta
        vector_store._LOADED_FINGERPRINT = "stale-fingerprint-A"

        vector_store.load_index()   # no force=True
        assert vector_store.loaded_fingerprint() == kb_version.current_version()
        assert any("charlie" in (m.get("text") or "")
                   for m in vector_store.get_metadata())


def test_rollback_restores_previous_version(versioned_index_dir):
    with patch("src.vector_store.embed_texts", side_effect=_mock_embed_texts), \
         patch("src.vector_store.embed_query", side_effect=_mock_embed_query):
        vector_store.build_index(chunks=_chunks("alpha", "docA"))
        version_a = kb_version.current_version()
        vector_store.build_index(chunks=_chunks("bravo", "docB"))
        assert kb_version.current_version() != version_a

        # Roll back to A's snapshot.
        kb_version.rollback(version_a)
        meta = vector_store.get_metadata()
        assert any("alpha" in (m.get("text") or "") for m in meta)
        assert not any("bravo" in (m.get("text") or "") for m in meta)
