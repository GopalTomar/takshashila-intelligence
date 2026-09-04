"""
vector_store.py — FAISS index build, save, load, and search.

- Builds a cosine-similarity index (IndexFlatIP over normalized embeddings).
- Saves the FAISS index and the chunk metadata separately (metadata as JSON,
  with a .pkl mirror for backward compatibility).
- Supports full rebuild, loading an existing index, and incremental add.
- Deduplicates chunks by content hash before indexing.
- Emits clear logs for #documents and #chunks indexed.
"""

import json
import pickle
from typing import Dict, List, Optional, Tuple


from src import config
from src.embeddings import embed_texts, embed_query
from src.utils import clean_chunk_metadata, get_logger, load_jsonl

logger = get_logger("vector_store", config.SCRAPE_LOG)

# JSON metadata sidecar (human-readable); .pkl kept for legacy loaders.
METADATA_JSON = config.INDEX_DIR / "metadata.json"


def _atomic_write(path, write_fn) -> None:
    """
    Write via a temp file in the same directory, then ``os.replace`` into place.
    ``os.replace`` is atomic on one filesystem, so a reader (dashboard / bot /
    API in another process) never observes a half-written index or metadata
    file, and a crashed build never leaves a corrupt file where a good one was.
    """
    import os, tempfile
    from pathlib import Path
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), suffix=".tmp")
    os.close(fd)
    try:
        write_fn(tmp)
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)


def atomic_persist_index(index, metadata: List[Dict]) -> None:
    """Persist a FAISS index + metadata atomically to the active on-disk paths."""
    import faiss
    config.INDEX_DIR.mkdir(parents=True, exist_ok=True)
    _atomic_write(config.FAISS_INDEX, lambda tmp: faiss.write_index(index, tmp))
    _atomic_write(config.METADATA_FILE,
                  lambda tmp: pickle.dump(metadata, open(tmp, "wb")))
    _atomic_write(METADATA_JSON,
                  lambda tmp: json.dump(metadata, open(tmp, "w", encoding="utf-8"),
                                        ensure_ascii=False))


def publish_current(metadata: List[Dict], build_duration_s: Optional[float] = None,
                    snapshot: bool = True) -> Dict:
    """
    Compute counts from ``metadata`` and publish a new active KB version (writes
    the manifest last, so consumers flip over only once the index is fully in
    place). Returns the published manifest dict.
    """
    from src import kb_version
    n_docs = len({m.get("document_id") or m.get("doc_id") for m in metadata})
    source_counts: Dict[str, int] = {}
    for m in metadata:
        s = m.get("source", "unknown")
        source_counts[s] = source_counts.get(s, 0) + 1
    manifest = kb_version.build_manifest(
        version=kb_version.new_version(),
        counts={"documents": n_docs, "chunks": len(metadata), "vectors": len(metadata)},
        source_counts=source_counts,
        build_duration_s=build_duration_s,
    )
    return kb_version.publish(manifest, snapshot=snapshot)


# ── Build ──────────────────────────────────────────────────────────────────────

def build_index(progress_cb=None, chunks: Optional[List[Dict]] = None) -> Tuple[int, int]:
    """
    Load chunks (from CHUNKS_FILE unless provided), embed, build a FAISS
    IndexFlatIP, and save index + metadata. Returns (num_chunks, embedding_dim).
    """
    import faiss

    if chunks is None:
        chunks = load_jsonl(config.CHUNKS_FILE)
    if not chunks:
        raise ValueError("No chunks found. Run chunking first.")

    # Deduplicate defensively by chunk_hash.
    seen, unique = set(), []
    for ch in chunks:
        h = ch.get("chunk_hash") or ch.get("chunk_id")
        if h in seen:
            continue
        seen.add(h)
        unique.append(ch)
    if len(unique) != len(chunks):
        logger.info(f"Removed {len(chunks) - len(unique)} duplicate chunks before indexing")
    chunks = unique

    # Final safety net: repair any mojibake so both the embedded text and the
    # persisted metadata are clean (the FAISS index is built from these).
    chunks = [clean_chunk_metadata(ch) for ch in chunks]

    n_docs = len({ch.get("document_id") or ch.get("doc_id") for ch in chunks})
    from src.utils import chunk_search_text
    texts = [chunk_search_text(ch) for ch in chunks]   # metadata header + body

    logger.info(f"Embedding {len(texts)} chunks from {n_docs} documents "
                f"with {config.EMBEDDING_MODEL}…")
    if progress_cb:
        progress_cb(f"Embedding {len(texts)} chunks from {n_docs} documents…")

    embeddings = embed_texts(texts, show_progress=True)
    dim = int(embeddings.shape[1])

    index = faiss.IndexFlatIP(dim)   # cosine via inner product (vectors normalized)
    index.add(embeddings)

    # Metadata parallel to vectors (keep full text for context rendering).
    metadata = [dict(ch) for ch in chunks]

    # Atomic publish: write index + metadata via temp+rename, then flip the
    # version manifest last so no consumer ever reads a half-built index.
    atomic_persist_index(index, metadata)
    manifest = publish_current(metadata)
    logger.info("Build published as KB version %s", manifest.get("version"))

    # Per-source breakdown for the log.
    by_source: Dict[str, int] = {}
    for ch in chunks:
        s = ch.get("source", "unknown")
        by_source[s] = by_source.get(s, 0) + 1
    breakdown = ", ".join(f"{k}={v}" for k, v in sorted(by_source.items()))

    logger.info(f"FAISS index saved: {len(chunks)} vectors (dim={dim}) "
                f"from {n_docs} docs [{breakdown}]")
    if progress_cb:
        progress_cb(f"✓ FAISS index built — {len(chunks)} chunks from {n_docs} docs ({breakdown})")

    # Reset cache so the next search loads fresh data. (load_index() would also
    # self-heal via the new version fingerprint, but resetting here is immediate.)
    global _INDEX, _METADATA, _LOADED_FINGERPRINT
    _INDEX, _METADATA, _LOADED_FINGERPRINT = None, [], None
    return len(chunks), dim


# ── Load ───────────────────────────────────────────────────────────────────────

_INDEX = None
_METADATA: List[Dict] = []
_LOADED_FINGERPRINT: Optional[str] = None   # KB version this in-memory copy reflects


def load_index(force: bool = False):
    """
    Load the FAISS index + metadata into the module-level cache, reloading
    automatically whenever the *active KB version has changed on disk*.

    This is the single mechanism that keeps every long-running consumer
    (dashboard, Mattermost bot, API) from serving a stale index after a rebuild
    happens in another process: each call cheaply reads the current KB
    fingerprint (a tiny manifest file) and only re-reads the heavy index files
    when that fingerprint differs from what is in memory. Because all processes
    read the same manifest, a rebuild anywhere is picked up everywhere with no
    restart. Pass ``force=True`` to reload unconditionally.
    """
    global _INDEX, _METADATA, _LOADED_FINGERPRINT
    from src import kb_version

    disk_fp = kb_version.current_fingerprint()
    if _INDEX is not None and not force and disk_fp == _LOADED_FINGERPRINT:
        return

    import faiss

    if not config.FAISS_INDEX.exists():
        raise FileNotFoundError(
            f"FAISS index not found at {config.FAISS_INDEX}. Run build first."
        )
    new_index = faiss.read_index(str(config.FAISS_INDEX))

    if config.METADATA_FILE.exists():
        with open(config.METADATA_FILE, "rb") as f:
            new_metadata = pickle.load(f)
    elif METADATA_JSON.exists():
        with open(METADATA_JSON, "r", encoding="utf-8") as f:
            new_metadata = json.load(f)
    else:
        raise FileNotFoundError("Index metadata not found.")

    # Swap in the freshly-loaded objects. Replacing the _METADATA *list object*
    # (new identity) is what makes the BM25 index — which keys its cache on
    # id(get_metadata()) — rebuild itself against the new data automatically.
    _INDEX, _METADATA, _LOADED_FINGERPRINT = new_index, new_metadata, disk_fp
    reloaded = "reloaded" if force or _LOADED_FINGERPRINT else "loaded"
    logger.info(f"Index {reloaded}: {_INDEX.ntotal} vectors, "
                f"{len(_METADATA)} metadata records (version={disk_fp})")


def loaded_fingerprint() -> Optional[str]:
    """The KB fingerprint the in-memory index currently reflects (None if unloaded)."""
    return _LOADED_FINGERPRINT


def is_loaded() -> bool:
    """True if the FAISS index + metadata are already in memory."""
    return _INDEX is not None


def ntotal() -> int:
    """Number of vectors in the loaded index (0 if not loaded yet)."""
    return int(_INDEX.ntotal) if _INDEX is not None else 0


def get_metadata() -> List[Dict]:
    """
    Return the in-memory chunk-metadata list (the SINGLE shared copy that backs
    FAISS search). Other components (e.g. the BM25 index) should reuse THIS list
    instead of re-reading chunks.jsonl, so the process keeps exactly one full
    copy of the chunk text in memory rather than two or three.
    """
    load_index()
    return _METADATA


def index_stats() -> Dict:
    """Return basic stats about the loaded index (by source + category)."""
    load_index()
    by_source: Dict[str, int] = {}
    by_category: Dict[str, int] = {}
    doc_ids = set()
    for m in _METADATA:
        s = m.get("source", "unknown")
        by_source[s] = by_source.get(s, 0) + 1
        c = m.get("category", "") or "uncategorized"
        by_category[c] = by_category.get(c, 0) + 1
        doc_ids.add(m.get("document_id") or m.get("doc_id"))
    return {
        "total_chunks":    _INDEX.ntotal if _INDEX else 0,
        "total_documents": len(doc_ids),
        "by_source":       by_source,
        "by_category":     by_category,
    }


# ── Search ─────────────────────────────────────────────────────────────────────

def search(
    query: str,
    top_k: int = config.TOP_K,
    source_filter: Optional[str] = None,
    category_filter: Optional[str] = None,
    author_filter: Optional[str] = None,
    year_filter: Optional[str] = None,
    # legacy alias kept so older callers don't break:
    source_type_filter: Optional[str] = None,
) -> List[Dict]:
    """
    FAISS semantic search with optional metadata filters.
    Returns chunk dicts augmented with 'score' (raw cosine similarity).
    """
    load_index()
    if source_filter is None and source_type_filter is not None:
        source_filter = source_type_filter

    q_vec = embed_query(query)
    k = min(max(top_k * 12, 30), _INDEX.ntotal)
    scores, indices = _INDEX.search(q_vec, k)

    results = []
    for score, idx in zip(scores[0], indices[0]):
        if idx < 0 or idx >= len(_METADATA):
            continue
        meta = _METADATA[idx]

        if source_filter and source_filter != "all":
            ms = (meta.get("source") or meta.get("source_type") or "").lower()
            if ms != source_filter.lower():
                continue
        if category_filter and category_filter != "all":
            mc = (meta.get("category") or "").lower()
            if category_filter.lower() not in mc:
                continue
        if author_filter:
            if author_filter.lower() not in (meta.get("author") or "").lower():
                continue
        if year_filter:
            if not (meta.get("date") or "").startswith(year_filter):
                continue

        results.append({**meta, "score": float(score)})

    return results


def update_index_with_new_chunks(new_chunk_texts: List[str],
                                 new_metadata: List[Dict],
                                 progress_cb=None) -> int:
    """Incrementally add new chunks to an existing FAISS index."""
    import faiss

    load_index()
    # Embed the metadata-aware search text (header + body) to match the rest of
    # the pipeline, so any chunk added this way is retrievable by its metadata too.
    from src.utils import chunk_search_text
    search_texts = [
        chunk_search_text({**(new_metadata[i] if i < len(new_metadata) else {}), "text": t})
        for i, t in enumerate(new_chunk_texts)
    ]
    embeddings = embed_texts(search_texts)
    _INDEX.add(embeddings)
    _METADATA.extend(new_metadata)

    faiss.write_index(_INDEX, str(config.FAISS_INDEX))
    with open(config.METADATA_FILE, "wb") as f:
        pickle.dump(_METADATA, f)
    with open(METADATA_JSON, "w", encoding="utf-8") as f:
        json.dump(_METADATA, f, ensure_ascii=False)

    logger.info(f"Index updated: now {_INDEX.ntotal} vectors")
    if progress_cb:
        progress_cb(f"✓ Index updated — {_INDEX.ntotal} total chunks")
    return _INDEX.ntotal