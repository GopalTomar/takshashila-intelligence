"""
chunker.py — Sentence-aware, character-based chunking with full metadata.

Each chunk carries:
    document_id, chunk_id, chunk_index, source, source_name,
    title, category, url, text
plus legacy mirror fields (doc_id, source_type, original_url, author, date,
pdf_url, page_number, chunk_hash) so older modules keep working.

Chunk size is configured in CHARACTERS (CHUNK_SIZE ~800–1200) with overlap
(CHUNK_OVERLAP ~150–250). Sentences are kept whole whenever possible.
"""

import re
from typing import Dict, List

from src import config
from src.utils import (
    clean_text, clean_document_metadata, clean_mojibake_text,
    clean_or_drop_bad_lines, get_text_quality_score, CHUNK_QUALITY_MIN,
    content_hash, get_logger, load_jsonl, save_jsonl,
    build_meta_header, chunk_search_text,
)

logger = get_logger("chunker", config.SCRAPE_LOG)


# ── Sentence splitting ──────────────────────────────────────────────────────────

def _split_sentences(text: str) -> List[str]:
    """
    Split text into sentence-like units. Newlines are treated as soft breaks
    (the Commit KB text is line-structured), and we also split on . ! ?
    followed by whitespace.
    """
    # Normalize whitespace but keep single newlines as separators.
    text = re.sub(r"[ \t]+", " ", text)
    # Break on blank lines first, then on sentence punctuation.
    rough = re.split(r"\n+|(?<=[.!?])\s+", text)
    return [s.strip() for s in rough if s and s.strip()]


def _pack_sentences(
    sentences: List[str],
    max_chars: int,
    overlap_chars: int,
    min_len: int,
) -> List[str]:
    """
    Greedily pack sentences into chunks up to max_chars. Adjacent chunks share
    ~overlap_chars worth of trailing sentences so context isn't lost at edges.
    Oversized single sentences are hard-split on whitespace.
    """
    chunks: List[str] = []
    current: List[str] = []
    current_len = 0

    def flush():
        nonlocal current, current_len
        if current:
            joined = " ".join(current).strip()
            if len(joined) >= min_len:
                chunks.append(joined)
        # build overlap tail
        tail: List[str] = []
        tail_len = 0
        for s in reversed(current):
            if tail_len + len(s) + 1 <= overlap_chars:
                tail.insert(0, s)
                tail_len += len(s) + 1
            else:
                break
        current = tail[:]
        current_len = sum(len(s) + 1 for s in current)

    for sent in sentences:
        # Hard-split a sentence that alone exceeds the budget.
        if len(sent) > max_chars:
            if current:
                flush()
            words = sent.split(" ")
            buf = ""
            for w in words:
                if len(buf) + len(w) + 1 > max_chars:
                    if len(buf) >= min_len:
                        chunks.append(buf.strip())
                    buf = w
                else:
                    buf = f"{buf} {w}".strip()
            if buf:
                current.append(buf)
                current_len = len(buf)
            continue

        if current_len + len(sent) + 1 > max_chars and current:
            flush()

        current.append(sent)
        current_len += len(sent) + 1

    if current:
        joined = " ".join(current).strip()
        if len(joined) >= min_len:
            chunks.append(joined)

    # De-dup consecutive identical chunks that can arise from overlap edge cases.
    deduped: List[str] = []
    for c in chunks:
        if not deduped or deduped[-1] != c:
            deduped.append(c)
    return deduped


# ── Document → chunks ────────────────────────────────────────────────────────────

def _base_meta(doc: Dict) -> Dict:
    document_id = doc.get("document_id") or doc.get("id") or doc.get("url_hash") or ""
    source      = doc.get("source") or doc.get("source_type") or "local"
    url         = doc.get("url") or doc.get("original_url") or ""
    authors     = doc.get("authors") or ([doc["author"]] if doc.get("author") else [])
    meta = {
        "document_id":  document_id,
        "source":       source,
        "source_name":  doc.get("source_name") or config.source_display_name(source, source),
        "title":        doc.get("title", "") or "Untitled",
        "subtitle":     doc.get("subtitle", "") or "",
        "category":     doc.get("category", "") or "",
        "section":      doc.get("section", "") or "",
        "url":          url,
        "canonical_url": doc.get("canonical_url", "") or url,
        "document_type": doc.get("document_type", "") or "",
        "language":     doc.get("language", "") or "",
        "page_id":      doc.get("page_id", "") or "",
        "updated_date": doc.get("updated_date", "") or "",
        "breadcrumbs":  doc.get("breadcrumbs", []) or [],
        "authors":      authors,
        # ── legacy mirrors ──
        "doc_id":       document_id,
        "source_type":  source,
        "original_url": url,
        "pdf_url":      doc.get("pdf_url", "") or "",
        "author":       doc.get("author", "") or (authors[0] if authors else ""),
        "date":         doc.get("date", "") or "",
        "tags":         doc.get("tags", []) or [],
    }
    # Precompute the compact metadata header once per document (shared by all its
    # chunks) so metadata is searchable and answerable.
    meta["meta_header"] = build_meta_header(meta)
    return meta


def chunk_document(doc: Dict) -> List[Dict]:
    """Chunk a single (unified) document into metadata-rich chunk dicts."""
    doc  = clean_document_metadata(doc)   # repair mojibake before splitting
    base = _base_meta(doc)
    chunks: List[Dict] = []

    max_chars     = config.CHUNK_SIZE
    overlap_chars = config.CHUNK_OVERLAP
    min_len       = config.CHUNK_MIN_LEN

    # PDF documents may carry per-page text — chunk per page, tagging page_number.
    if doc.get("source_type") == "pdf" and doc.get("pdf_pages"):
        for page_info in doc["pdf_pages"]:
            page_text = clean_mojibake_text(page_info.get("text", ""))
            page_text, _ = clean_or_drop_bad_lines(page_text)  # remove OCR garbage lines
            page_num  = page_info.get("page_number", 0)
            if not page_text:
                continue
            sents = _split_sentences(page_text)
            for idx, ctext in enumerate(_pack_sentences(sents, max_chars, overlap_chars, min_len)):
                if get_text_quality_score(ctext) < CHUNK_QUALITY_MIN:
                    continue  # skip unrecoverable garbage chunk
                ch = {
                    **base,
                    "chunk_id":    f"{base['document_id']}_p{page_num}_c{idx}",
                    "chunk_index": idx,
                    "chunk_order": idx,
                    "heading_path": base.get("section") or base.get("title") or "",
                    "page_number": page_num,
                    "text":        ctext,
                }
                ch["chunk_hash"] = content_hash(chunk_search_text(ch))
                chunks.append(ch)
        return chunks

    text = clean_text(doc.get("text", ""))
    text, _ = clean_or_drop_bad_lines(text)   # remove OCR garbage lines
    if not text:
        return []
    sents = _split_sentences(text)
    for idx, ctext in enumerate(_pack_sentences(sents, max_chars, overlap_chars, min_len)):
        if get_text_quality_score(ctext) < CHUNK_QUALITY_MIN:
            continue  # skip unrecoverable garbage chunk
        ch = {
            **base,
            "chunk_id":    f"{base['document_id']}_c{idx}",
            "chunk_index": idx,
            "chunk_order": idx,
            "heading_path": base.get("section") or base.get("title") or "",
            "page_number": None,
            "text":        ctext,
        }
        ch["chunk_hash"] = content_hash(chunk_search_text(ch))
        chunks.append(ch)
    return chunks


def chunk_documents(docs: List[Dict], progress_cb=None) -> List[Dict]:
    """Chunk a list of unified documents, deduplicating by chunk content hash."""
    all_chunks: List[Dict] = []
    seen = set()
    for i, doc in enumerate(docs):
        if not doc.get("text"):
            continue
        for ch in chunk_document(doc):
            if ch["chunk_hash"] in seen:
                continue
            seen.add(ch["chunk_hash"])
            all_chunks.append(ch)
        if progress_cb and i % 20 == 0:
            progress_cb(f"Chunked {i+1}/{len(docs)} documents ({len(all_chunks)} chunks so far)…")
    return all_chunks


def build_chunks(progress_cb=None) -> int:
    """
    Read unified documents from DOCUMENTS_FILE, chunk them, write CHUNKS_FILE.
    Returns total chunk count. (Kept for compatibility with existing scripts.)
    """
    docs = load_jsonl(config.DOCUMENTS_FILE)
    if not docs:
        logger.warning("No documents found to chunk")
        return 0

    logger.info(f"Chunking {len(docs)} documents…")
    all_chunks = chunk_documents(docs, progress_cb=progress_cb)
    save_jsonl(config.CHUNKS_FILE, all_chunks)
    logger.info(f"Saved {len(all_chunks)} chunks to {config.CHUNKS_FILE}")
    if progress_cb:
        progress_cb(f"✓ Chunking complete — {len(all_chunks)} chunks from {len(docs)} documents")
    return len(all_chunks)