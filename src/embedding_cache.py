"""
embedding_cache.py — Persistent chunk-embedding cache.

Re-scraping is the expensive part of an update; re-*embedding* is cheap, but on a
large corpus even that adds up when nothing changed. This cache remembers the
embedding vector for every chunk we've ever embedded, keyed by the chunk's
content hash (``chunk_hash``). On an index rebuild we then embed **only the chunks
whose text is new or changed** and reuse cached vectors for everything else.

Because the key is the *content* hash, a chunk whose text is unchanged keeps the
same key across runs and is served from cache; a changed chunk gets a new hash and
is embedded fresh. Deleted chunks simply stop being requested (and are pruned).

Storage: a single compressed ``.npz`` (hashes + matrix) at
``data/index/embedding_cache.npz``. The vectors are the same normalized vectors
``embeddings.embed_texts`` produces, so the FAISS index built from them is
identical to a from-scratch build — only faster.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from src import config
from src.utils import content_hash, get_logger

logger = get_logger("embedding_cache", config.SCRAPE_LOG)

CACHE_FILE = config.INDEX_DIR / "embedding_cache.npz"


class EmbeddingCache:
    """A hash → vector store that fills misses via ``embeddings.embed_texts``."""

    def __init__(self, path: Path = CACHE_FILE):
        self.path = path
        self._store: Dict[str, np.ndarray] = {}
        self._dim: Optional[int] = None
        self._load()

    def _load(self) -> None:
        if not self.path.exists():
            return
        try:
            data = np.load(self.path, allow_pickle=False)
            hashes = data["hashes"]
            matrix = data["vectors"]
            self._store = {str(h): matrix[i] for i, h in enumerate(hashes)}
            if len(matrix):
                self._dim = int(matrix.shape[1])
            logger.info(f"Loaded {len(self._store)} cached embeddings from {self.path.name}")
        except Exception as exc:
            logger.warning(f"Could not load embedding cache ({exc}); starting empty.")
            self._store = {}

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if not self._store:
            return
        hashes = np.array(list(self._store.keys()))
        vectors = np.vstack(list(self._store.values())).astype("float32")
        # numpy appends ".npz" to a savez path without that suffix, so name the
        # temp file with a .npz suffix and then rename it into place.
        tmp = self.path.with_name(self.path.stem + ".tmp.npz")
        np.savez_compressed(tmp, hashes=hashes, vectors=vectors)
        tmp.replace(self.path)
        logger.info(f"Saved {len(self._store)} embeddings to {self.path.name}")

    def _key(self, chunk: Dict) -> str:
        from src.utils import chunk_search_text
        return chunk.get("chunk_hash") or content_hash(chunk_search_text(chunk))

    def embed_chunks(self, chunks: List[Dict], show_progress: bool = True
                     ) -> Tuple[np.ndarray, Dict[str, int]]:
        """
        Return an (N, dim) matrix of embeddings aligned to ``chunks``, embedding
        only cache-misses. Also returns a small stats dict {cached, embedded}.

        Embeds the metadata-aware search text (title/author/date/section/tags +
        body) so metadata questions retrieve the right chunk; the cache key is the
        content hash of that same search text, so a metadata edit correctly
        invalidates the stale vector.
        """
        from src.embeddings import embed_texts     # lazy: heavy model import
        from src.utils import chunk_search_text

        keys = [self._key(ch) for ch in chunks]
        missing_idx = [i for i, k in enumerate(keys) if k not in self._store]
        stats = {"cached": len(chunks) - len(missing_idx), "embedded": len(missing_idx)}

        if missing_idx:
            miss_texts = [chunk_search_text(chunks[i]) for i in missing_idx]
            logger.info(f"Embedding {len(miss_texts)} new/changed chunks "
                        f"({stats['cached']} reused from cache)…")
            new_vecs = embed_texts(miss_texts, show_progress=show_progress)
            if self._dim is None and len(new_vecs):
                self._dim = int(new_vecs.shape[1])
            for j, i in enumerate(missing_idx):
                self._store[keys[i]] = new_vecs[j]
        else:
            logger.info(f"All {len(chunks)} chunks served from embedding cache.")

        matrix = np.vstack([self._store[k] for k in keys]).astype("float32")
        return matrix, stats

    def prune(self, live_hashes: set) -> int:
        """Drop cached vectors whose chunk hash is no longer in use. Returns count."""
        before = len(self._store)
        self._store = {k: v for k, v in self._store.items() if k in live_hashes}
        removed = before - len(self._store)
        if removed:
            logger.info(f"Pruned {removed} stale embeddings from cache.")
        return removed