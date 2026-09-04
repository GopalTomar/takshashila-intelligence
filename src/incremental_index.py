"""
incremental_index.py — Merge document deltas and rebuild the index cheaply.

Two jobs, both designed to keep the *existing* on-disk format byte-compatible so
the retriever, the Streamlit app and the Mattermost bot keep working unchanged:

1. :func:`merge_documents` — apply a crawl delta to
   ``data/processed/documents.jsonl`` **in place**: replace records whose
   ``document_id`` changed, add new ones, and drop removed ones. (The old scraper
   only ever appended, so changed pages were never updated — this fixes that.)

2. :func:`rebuild_index` — re-chunk the unified documents and rebuild the FAISS
   index, but embed **only new/changed chunks** via :class:`EmbeddingCache`. The
   output files (``faiss.index`` + ``metadata.pkl`` + ``metadata.json``) are the
   same ones ``vector_store.build_index`` writes, so nothing downstream changes.

Because ``IndexFlatIP`` can't remove individual vectors, a change/removal triggers
a rebuild — but with the embedding cache that rebuild only pays for the chunks
that actually changed, so an incremental update stays fast end-to-end.
"""

from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Dict, Iterable, List, Optional

from src import config
from src.utils import get_logger, load_jsonl, save_jsonl

logger = get_logger("incremental_index", config.SCRAPE_LOG)

METADATA_JSON = config.INDEX_DIR / "metadata.json"


# ════════════════════════════════════════════════════════════════════════════════
#  Document delta merge
# ════════════════════════════════════════════════════════════════════════════════

def _doc_key(doc: Dict) -> str:
    return doc.get("document_id") or doc.get("id") or doc.get("url_hash") or doc.get("url") or ""


def _norm_url(u: str) -> str:
    """
    Normalise a URL for identity comparison (case, www, trailing slash,
    fragment, tracking query params).

    Delegates to ``src.utils.canonicalize_for_dedup`` — the single shared
    definition of "same page" used everywhere in the pipeline (the crawler's
    frontier dedup in scripts/crawl_engine.py, and this document-collapse
    step). Audit note: this function used to have its own, weaker, separate
    implementation here (lowercased scheme/host and dropped the fragment, but
    did NOT strip ``www.`` or tracking query params like ``?utm_source=``),
    so two documents crawled under e.g. ``www.`` vs. bare-domain variants of
    the same URL would NOT have been recognised as duplicates by this
    "repair" tool even though the crawler itself now treats them as one page.
    """
    from src.utils import canonicalize_for_dedup
    return canonicalize_for_dedup(u) if u else ""


def _doc_richness(doc: Dict) -> tuple:
    """Sort key: prefer the copy with the most metadata / content, then newest."""
    has_author = 1 if (str(doc.get("author") or "").strip() or doc.get("authors")) else 0
    has_date = 1 if str(doc.get("date") or "").strip() else 0
    has_section = 1 if str(doc.get("section") or doc.get("category") or "").strip() else 0
    n_tags = len(doc.get("tags") or [])
    text_len = doc.get("text_length") or len(doc.get("text") or "")
    scraped = str(doc.get("scraped_at") or doc.get("updated_at") or "")
    # non-PDF pages are preferred as the canonical holder of a URL over a "[PDF]" twin
    is_html = 0 if (doc.get("source_type") == "pdf" or doc.get("pdf_url")) else 1
    return (has_author, has_date, has_section, n_tags, is_html, text_len, scraped)


def collapse_by_url(docs: List[Dict]) -> tuple:
    """
    Collapse documents that point at the same (normalised) URL, keeping the
    richest copy of each. Returns (kept_docs, dropped_count).

    A page's identity is its canonical URL when present, else its URL. This
    removes the "same page stored more than once" duplicates that accumulate when
    an older document-id scheme and a newer one both wrote the same page.
    """
    groups: Dict[str, List[Dict]] = {}
    order: List[str] = []
    for d in docs:
        key = _norm_url(d.get("canonical_url") or d.get("url") or d.get("original_url") or "")
        if not key:
            key = _doc_key(d) or f"__noid_{len(order)}"
        if key not in groups:
            groups[key] = []
            order.append(key)
        groups[key].append(d)

    kept: List[Dict] = []
    dropped = 0
    for key in order:
        bucket = groups[key]
        if len(bucket) == 1:
            kept.append(bucket[0])
            continue
        best = max(bucket, key=_doc_richness)
        kept.append(best)
        dropped += len(bucket) - 1
    return kept, dropped


def merge_documents(new_or_changed: List[Dict],
                    removed_ids: Optional[Iterable[str]] = None,
                    documents_file: Path = None) -> Dict[str, int]:
    """
    Merge a crawl delta into ``documents.jsonl`` in place, keyed by document_id.

    * records in ``new_or_changed`` replace any existing record with the same id,
      or are appended if new;
    * any id in ``removed_ids`` is dropped.

    Returns a summary dict {added, updated, removed, total}.
    """
    documents_file = documents_file or config.DOCUMENTS_FILE
    removed_ids = set(removed_ids or [])

    existing = load_jsonl(documents_file)
    by_id: Dict[str, Dict] = {}
    order: List[str] = []
    for d in existing:
        k = _doc_key(d)
        if not k:
            continue
        if k not in by_id:
            order.append(k)
        by_id[k] = d

    added = updated = 0
    for doc in new_or_changed:
        k = _doc_key(doc)
        if not k:
            continue
        if k in by_id:
            updated += 1
        else:
            added += 1
            order.append(k)
        by_id[k] = doc

    removed = 0
    for k in list(removed_ids):
        if k in by_id:
            by_id.pop(k, None)
            removed += 1

    merged = [by_id[k] for k in order if k in by_id]

    # Collapse any same-URL duplicates (e.g. legacy documents written under a
    # different id scheme for a page the new crawler also stored). Keeps the
    # richest copy of each URL so a page is never stored more than once.
    merged, url_dupes = collapse_by_url(merged)

    save_jsonl(documents_file, merged)

    summary = {"added": added, "updated": updated, "removed": removed,
               "url_duplicates_collapsed": url_dupes, "total": len(merged)}
    logger.info(f"documents.jsonl merged: {summary}")
    return summary


# ════════════════════════════════════════════════════════════════════════════════
#  Cached index rebuild
# ════════════════════════════════════════════════════════════════════════════════

def rebuild_index(progress_cb=None, use_cache: bool = True) -> Dict[str, int]:
    """
    Re-chunk ``documents.jsonl`` and rebuild the FAISS index, embedding only
    new/changed chunks when ``use_cache`` is on. Writes the same files as
    ``vector_store.build_index``. Returns {documents, chunks, embedded, cached}.
    """
    import faiss  # lazy: heavy
    from src.chunker import chunk_documents
    from src.utils import clean_chunk_metadata

    docs = load_jsonl(config.DOCUMENTS_FILE)
    if not docs:
        raise ValueError(f"No documents found at {config.DOCUMENTS_FILE}. Crawl first.")

    if progress_cb:
        progress_cb(f"Chunking {len(docs)} documents…")
    chunks = chunk_documents(docs, progress_cb=progress_cb)
    chunks = [clean_chunk_metadata(ch) for ch in chunks]

    # Persist chunks.jsonl too (keeps the classic pipeline artifacts in sync).
    save_jsonl(config.CHUNKS_FILE, chunks)

    if not chunks:
        raise ValueError("No chunks produced from the documents.")

    # ── Embed (cache-aware) ──────────────────────────────────────────────────────
    if use_cache:
        from src.embedding_cache import EmbeddingCache
        cache = EmbeddingCache()
        embeddings, estats = cache.embed_chunks(chunks, show_progress=True)
        live_hashes = {ch.get("chunk_hash") for ch in chunks}
        cache.prune(live_hashes)
        cache.save()
    else:
        from src.embeddings import embed_texts
        from src.utils import chunk_search_text
        embeddings = embed_texts([chunk_search_text(ch) for ch in chunks], show_progress=True)
        estats = {"embedded": len(chunks), "cached": 0}

    dim = int(embeddings.shape[1])

    # ── Build + persist the FAISS index (identical format to build_index) ────────
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings.astype("float32"))

    metadata = [dict(ch) for ch in chunks]

    # Atomic publish (temp+rename, manifest written last) + a new KB version, so
    # every long-running consumer auto-reloads this rebuild with no restart.
    from src import vector_store
    vector_store.atomic_persist_index(index, metadata)
    manifest = vector_store.publish_current(metadata)
    logger.info("Incremental rebuild published as KB version %s", manifest.get("version"))

    # Invalidate the in-process cache so the next search reloads fresh data.
    try:
        vector_store._INDEX, vector_store._METADATA, vector_store._LOADED_FINGERPRINT = None, [], None
    except Exception:
        pass

    n_docs = len({ch.get("document_id") for ch in chunks})
    by_source: Dict[str, int] = {}
    for ch in chunks:
        s = ch.get("source", "unknown")
        by_source[s] = by_source.get(s, 0) + 1
    breakdown = ", ".join(f"{k}={v}" for k, v in sorted(by_source.items()))

    summary = {
        "documents": n_docs, "chunks": len(chunks),
        "embedded": estats["embedded"], "cached": estats["cached"],
    }
    logger.info(f"Index rebuilt: {len(chunks)} vectors (dim={dim}) from {n_docs} docs "
                f"[{breakdown}] — embedded {estats['embedded']}, reused {estats['cached']}.")
    if progress_cb:
        progress_cb(f"✓ Index rebuilt — {len(chunks)} chunks from {n_docs} docs "
                    f"(embedded {estats['embedded']}, reused {estats['cached']}; {breakdown})")
    return summary