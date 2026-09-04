"""
extractors.py — Text extraction + metadata enrichment.

Two responsibilities:
  1. (NEW) Load and normalize JSONL knowledge-base files (e.g. the Commit KB)
     into a unified document schema the chunker/indexer understands.
  2. (LEGACY) Extract text/metadata from scraped HTML and PDF sources.

The legacy functions are preserved unchanged for backward compatibility.
"""

import json
import re
from pathlib import Path
from typing import Dict, List, Optional

from src import config
from src.utils import (
    clean_text, clean_document_metadata, clean_mojibake_text,
    content_hash, get_logger, now_iso,
)

logger = get_logger("extractors", config.SCRAPE_LOG)


# ════════════════════════════════════════════════════════════════════════════
# NEW — JSONL knowledge-base loading & normalization
# ════════════════════════════════════════════════════════════════════════════

def _safe_text(value) -> str:
    return value if isinstance(value, str) else ("" if value is None else str(value))


def extract_date_from_kb_url(url: str) -> str:
    """
    Commit KB content URLs embed an ISO date, e.g.
    /playbook/2026-06-16-do-not-resolve-... → 2026-06-16
    Returns "" when no date is present (e.g. index pages).
    """
    m = re.search(r"/(\d{4})-(\d{2})-(\d{2})-", url or "")
    if m:
        y, mo, d = m.group(1), m.group(2), m.group(3)
        if 2000 <= int(y) <= 2035 and 1 <= int(mo) <= 12 and 1 <= int(d) <= 31:
            return f"{y}-{mo}-{d}"
    return ""


def normalize_kb_document(
    raw: Dict,
    default_source: str = "commit_kb",
    default_source_name: Optional[str] = None,
    doc_id: Optional[str] = None,
) -> Dict:
    """
    Normalize a raw KB record into the unified document schema used across
    the pipeline.

    Accepts records shaped like the Commit KB export:
        {"source","url","title","category","text","text_length"}
    or already-normalized records that include id/source_name/ingested_at.

    Returns a dict with:
        id / document_id, source, source_name, title, category, url,
        text, text_length, date, ingested_at
    plus legacy mirror fields (original_url, source_type) so older code keeps
    working.
    """
    source = _safe_text(raw.get("source") or default_source).strip() or default_source
    source_name = _safe_text(
        raw.get("source_name")
        or default_source_name
        or config.source_display_name(source, source)
    ).strip()

    url   = _safe_text(raw.get("url") or raw.get("original_url")).strip()
    title = clean_text(_safe_text(raw.get("title"))) or "Untitled"
    text  = clean_text(_safe_text(raw.get("text")))
    category = _safe_text(raw.get("category")).strip().lower()

    # date: prefer explicit field, else parse from a Commit-KB style URL
    date = _safe_text(raw.get("date")).strip() or extract_date_from_kb_url(url)

    document_id = _safe_text(raw.get("id") or doc_id).strip()

    text_length = raw.get("text_length")
    if not isinstance(text_length, int):
        text_length = len(text)

    # Preserve the RECORD's own source_type (e.g. "pdf", "blog", "research")
    # when present — this is what src/chunker.py checks to decide whether to
    # do page-level PDF chunking. Only fall back to the top-level `source`
    # (e.g. "commit_kb", "website") when the record doesn't carry its own.
    record_source_type = _safe_text(raw.get("source_type")).strip() or source

    return clean_document_metadata({
        "id":           document_id,
        "document_id":  document_id,
        "source":       source,
        "source_name":  source_name,
        "title":        title,
        "category":     category,
        "url":          url,
        "text":         text,
        "text_length":  text_length,
        "date":         date,
        "ingested_at":  _safe_text(raw.get("ingested_at")) or now_iso(),
        # ── legacy mirrors (keep older modules/UI working) ──
        "original_url": url,
        "canonical_url": _safe_text(raw.get("canonical_url")) or url,
        "source_type":  record_source_type,
        "author":       _safe_text(raw.get("author")),
        "pdf_url":      _safe_text(raw.get("pdf_url")),
        # Per-page PDF text (Part 2: "preserve page numbers when available"),
        # consumed by src/chunker.py to emit page_number-tagged chunks.
        "pdf_pages":    raw.get("pdf_pages") or [],
        "discovery_method":  _safe_text(raw.get("discovery_method")),
        "extraction_method": _safe_text(raw.get("extraction_method")),
    })


def load_kb_documents(
    path: Path,
    default_source: str = "commit_kb",
    default_source_name: Optional[str] = None,
    id_prefix: Optional[str] = None,
) -> List[Dict]:
    """
    Load a .jsonl knowledge-base file and return normalized documents.

    If records lack an "id", one is generated as f"{id_prefix}_{i:04d}".
    Empty-text records are skipped.
    """
    path = Path(path)
    if not path.exists():
        logger.info(f"KB file not found (skipping): {path}")
        return []

    prefix = id_prefix or default_source
    docs: List[Dict] = []
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            try:
                raw = json.loads(line)
            except json.JSONDecodeError:
                logger.warning(f"Skipping malformed JSON on line {i+1} of {path.name}")
                continue
            doc = normalize_kb_document(
                raw,
                default_source=default_source,
                default_source_name=default_source_name,
                doc_id=f"{prefix}_{len(docs)+1:04d}",
            )
            if not doc["id"]:
                doc["id"] = doc["document_id"] = f"{prefix}_{len(docs)+1:04d}"
            if not doc["text"]:
                continue
            docs.append(doc)

    logger.info(f"Loaded {len(docs)} documents from {path.name} (source='{default_source}')")
    return docs


# ════════════════════════════════════════════════════════════════════════════
# LEGACY — metadata helpers (unchanged behaviour)
# ════════════════════════════════════════════════════════════════════════════

def extract_date_from_url(url: str) -> str:
    """
    Takshashila publication URLs encode date as /YYYYMMDD-title.
    E.g. /content/publications/20260402-Indias-West-Asian-Diplomacy.html → 2026-04-02
    """
    m = re.search(r"/(\d{4})(\d{2})(\d{2})-", url)
    if m:
        y, mo, d = m.group(1), m.group(2), m.group(3)
        if 2000 <= int(y) <= 2035 and 1 <= int(mo) <= 12 and 1 <= int(d) <= 31:
            return f"{y}-{mo}-{d}"
    return ""


def extract_category_from_url(url: str) -> str:
    """Infer a broad topic category from URL path segments."""
    url_lower = url.lower()
    category_map = [
        (r'ai[-_\s]?governance|artificial[-_\s]?intelligence', "AI Governance"),
        (r'geospat|gis|satellite|remote[-_\s]?sensing', "Geospatial"),
        (r'semiconductor|chip|foundry', "Semiconductors"),
        (r'indo[-_\s]?pacific|quad|asean', "Indo-Pacific"),
        (r'china|pla|bri\b', "China Studies"),
        (r'climate|environment|energy|lpg|oil|gas', "Energy & Climate"),
        (r'defence|military|security|strateg', "Defence & Security"),
        (r'space[-_\s]?policy|isro|launch', "Space Policy"),
        (r'economy|trade|wto|finance|fiscal', "Economy & Trade"),
        (r'cyber|digital|data|privacy', "Cyber & Digital"),
        (r'health|pharma|pandemic', "Health"),
        (r'education|school|skill', "Education"),
        (r'nuclear|npt|disarma', "Nuclear Policy"),
        (r'pakistan|border|kashmir', "South Asia"),
    ]
    for pattern, label in category_map:
        if re.search(pattern, url_lower):
            return label
    return ""


def extract_author_from_text(text: str) -> str:
    """Extract author name(s) from document text using common patterns."""
    if not text:
        return ""

    m = re.search(
        r'\bAuthors?\b[\s:]*\n((?:[A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,4}(?:\s*,\s*)?)+)',
        text, re.MULTILINE
    )
    if m:
        author = m.group(1).strip().rstrip(",").strip()
        if 3 < len(author) < 80:
            return author

    m = re.search(r'(?:^|\n)By\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,4})', text, re.MULTILINE)
    if m:
        author = m.group(1).strip()
        if 3 < len(author) < 80:
            return author

    m = re.search(
        r'\n([A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,4})\s*\n'
        r'(?:Takshashila|Research|Analyst|Fellow|Scholar|Professor|Dr\.)',
        text
    )
    if m:
        author = m.group(1).strip()
        if 3 < len(author) < 80:
            return author

    m = re.search(r'\bAuthor[s]?[:\s]+([A-Z][a-z]+(?:[\s,]+[A-Z][a-z]+){1,6})', text)
    if m:
        author = m.group(1).strip().rstrip(",").strip()
        if 3 < len(author) < 120:
            return author

    return ""


def extract_date_from_text(text: str) -> str:
    """Extract date from document text — tries common formats."""
    if not text:
        return ""

    months = (r"(?:January|February|March|April|May|June|July|August|"
              r"September|October|November|December)")

    patterns = [
        rf'\b(\d{{1,2}})\s+{months}\s+(\d{{4}})\b',
        rf'\b{months}\s+(\d{{1,2}}),?\s+(\d{{4}})\b',
        rf'\b{months}\s+(\d{{4}})\b',
        r'\b(\d{4})-(\d{2})-(\d{2})\b',
        rf'Version\s+[\d.]+,\s+{months}\s+(\d{{4}})',
    ]

    for p in patterns:
        m = re.search(p, text[:3000])
        if m:
            from src.utils import parse_date
            raw = m.group(0)
            parsed = parse_date(raw)
            if parsed and re.match(r"\d{4}", parsed):
                return parsed

    return ""


def extract_pdf_text(pdf_path: Path) -> List[Dict]:
    """Extract text page-by-page from a PDF using PyMuPDF."""
    pages = []
    try:
        import fitz
        doc = fitz.open(str(pdf_path))
        for page_num in range(len(doc)):
            text = doc[page_num].get_text("text")
            text = clean_text(text)
            if text:
                pages.append({"page_number": page_num + 1, "text": text})
        doc.close()
        logger.info(f"Extracted {len(pages)} pages from {pdf_path.name}")
    except Exception as exc:
        logger.error(f"PDF extraction failed for {pdf_path}: {exc}")
    return pages


def enrich_documents_with_pdf_text(progress_cb=None) -> int:
    """For every PDF document with empty text, run PDF extraction + backfill metadata."""
    from src.utils import load_jsonl, save_jsonl

    docs = load_jsonl(config.DOCUMENTS_FILE)
    updated = 0

    for doc in docs:
        changed = False

        if doc.get("source_type") == "pdf" and not doc.get("text"):
            pdf_path_str = doc.get("local_pdf_path", "")
            if pdf_path_str:
                pdf_path = Path(pdf_path_str)
                if pdf_path.exists():
                    pages = extract_pdf_text(pdf_path)
                    if pages:
                        for p in pages:
                            p["text"] = clean_mojibake_text(p.get("text", ""))
                        full_text = "\n\n".join(p["text"] for p in pages)
                        full_text = clean_mojibake_text(full_text)
                        doc["text"] = full_text
                        doc["content_hash"] = content_hash(full_text)
                        doc["pdf_pages"] = pages
                        doc["extracted_at"] = now_iso()
                        changed = True
                        if progress_cb:
                            progress_cb(f"Extracted PDF: {doc['title'][:60]}")

        url  = doc.get("original_url", "") or doc.get("pdf_url", "")
        text = doc.get("text", "")

        if not doc.get("date"):
            d = extract_date_from_url(url)
            if not d and text:
                d = extract_date_from_text(text)
            if d:
                doc["date"] = d
                changed = True

        if not doc.get("author") and text:
            a = extract_author_from_text(text)
            if a:
                doc["author"] = a
                changed = True

        if not doc.get("category") and url:
            c = extract_category_from_url(url)
            if c:
                doc["category"] = c
                changed = True

        if changed:
            updated += 1

    if updated:
        docs = [clean_document_metadata(d) for d in docs]
        save_jsonl(config.DOCUMENTS_FILE, docs)
        logger.info(f"Enriched {updated} documents with metadata")
    return updated


def enrich_all_metadata(progress_cb=None) -> int:
    """Standalone enrichment pass over ALL documents (date/author/category)."""
    from src.utils import load_jsonl, save_jsonl

    docs = load_jsonl(config.DOCUMENTS_FILE)
    if not docs:
        return 0
    updated = 0

    for i, doc in enumerate(docs):
        changed = False
        url  = doc.get("original_url", "") or doc.get("url", "") or doc.get("pdf_url", "")
        text = doc.get("text", "")

        if not doc.get("date"):
            d = extract_date_from_url(url) or extract_date_from_kb_url(url)
            if not d and text:
                d = extract_date_from_text(text)
            if d:
                doc["date"] = d
                changed = True

        if not doc.get("author") or doc.get("author") in ("", "Unknown", "Unknown author"):
            a = extract_author_from_text(text)
            if a:
                doc["author"] = a
                changed = True

        # Only infer topic categories for non-KB documents; KB docs already
        # carry curated categories (playbook/decisions/...) we must not clobber.
        if (not doc.get("category")) and url and doc.get("source") not in config.SOURCE_PRIORITY:
            c = extract_category_from_url(url)
            if c:
                doc["category"] = c
                changed = True

        if changed:
            updated += 1
        if progress_cb and i % 100 == 0:
            progress_cb(f"Enriching metadata: {i+1}/{len(docs)} docs…")

    if updated:
        docs = [clean_document_metadata(d) for d in docs]
        save_jsonl(config.DOCUMENTS_FILE, docs)
        logger.info(f"Metadata enrichment: updated {updated}/{len(docs)} documents")
    if progress_cb:
        progress_cb(f"✓ Metadata enriched for {updated} documents")
    return updated