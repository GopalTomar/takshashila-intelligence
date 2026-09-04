#!/usr/bin/env python3
"""
validate_kb.py — Health checks for the knowledge base and index.

Runs a battery of checks over documents.jsonl, chunks.jsonl and the FAISS
index, and reports problems grouped by severity:

  ERROR   — will hurt answers or references (missing URL/title, duplicate URLs,
            index/metadata count mismatch, empty text, orphan chunks).
  WARN    — quality issues worth fixing (missing author/date, over/undersized
            chunks, duplicate chunk hashes, missing embeddings coverage).
  INFO    — coverage statistics.

Usage:
    python scripts/validate_kb.py            # human-readable report, exit 1 on ERROR
    python scripts/validate_kb.py --json     # machine-readable JSON to stdout
    python scripts/validate_kb.py --strict   # exit 1 on WARN too

It is also importable: ``from scripts.validate_kb import validate`` returns the
report dict, so the ingestion pipeline can run it automatically after a build.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import config                                   # noqa: E402
from src.utils import canonicalize_for_dedup, load_jsonl, get_logger  # noqa: E402

logger = get_logger("validate_kb", config.SCRAPE_LOG)

# Chunk-size sanity bounds (characters).
_UNDERSIZE = max(40, config.CHUNK_MIN_LEN)
_OVERSIZE = int(config.CHUNK_SIZE * 2.2)


def _load_index_count() -> int:
    """Number of vectors in the FAISS index (or -1 if unreadable)."""
    try:
        import faiss
        if not config.FAISS_INDEX.exists():
            return -1
        return int(faiss.read_index(str(config.FAISS_INDEX)).ntotal)
    except Exception as exc:
        logger.warning(f"Could not read FAISS index: {exc}")
        return -1


def validate() -> dict:
    """Run all checks and return a structured report dict."""
    errors, warns, infos = [], [], []

    docs = load_jsonl(config.DOCUMENTS_FILE) if config.DOCUMENTS_FILE.exists() else []
    chunks = load_jsonl(config.CHUNKS_FILE) if config.CHUNKS_FILE.exists() else []

    # ── Documents ─────────────────────────────────────────────────────────────
    url_counts = Counter()
    chash_counts = Counter()
    missing_url = missing_title = missing_author = missing_date = empty_text = 0
    missing_url_local = 0
    doc_ids = set()

    for d in docs:
        did = d.get("document_id") or d.get("id") or ""
        doc_ids.add(did)
        url = (d.get("url") or d.get("original_url") or "").strip()
        if url:
            # Audit note: this used to count the RAW url string, so a
            # www./bare-domain pair, a trailing-slash variant, or a URL that
            # only differs by a tracking query param (?utm_source=...) would
            # NOT be flagged as a duplicate even though it's the same page.
            # Canonicalize with the same definition of "same page" the
            # crawler and collapse_by_url() use, so this check actually
            # catches what it claims to.
            url_counts[canonicalize_for_dedup(url)] += 1
        elif (d.get("source") or "").lower() in ("local", "staff_handbook"):
            # Hand-supplied internal sources (Staff Handbook, other local
            # documents) legitimately have no public URL — flag as a
            # warning (worth knowing about) rather than a hard error.
            missing_url_local += 1
        else:
            missing_url += 1
        if not (d.get("title") or "").strip() or (d.get("title") or "").strip().lower() == "untitled":
            missing_title += 1
        if not (str(d.get("author") or "").strip() or d.get("authors")):
            missing_author += 1
        if not str(d.get("date") or "").strip():
            missing_date += 1
        if not (d.get("text") or "").strip():
            empty_text += 1
        ch = d.get("content_hash")
        if ch:
            chash_counts[ch] += 1

    dup_urls = {u: n for u, n in url_counts.items() if n > 1}
    dup_content = {h: n for h, n in chash_counts.items() if n > 1}

    if docs:
        infos.append(f"{len(docs)} documents; {len(url_counts)} distinct URLs.")
    else:
        errors.append("No documents found (documents.jsonl empty or missing).")
    if missing_url:
        errors.append(f"{missing_url} document(s) missing a URL (cannot be linked as a reference).")
    if missing_url_local:
        warns.append(f"{missing_url_local} local/staff-handbook document(s) have no URL "
                     f"(expected for internal sources — citations will show without a link).")
    if dup_urls:
        errors.append(f"{len(dup_urls)} duplicate document URL(s) (same page stored more than once).")
    if empty_text:
        errors.append(f"{empty_text} document(s) have empty text.")
    if missing_title:
        warns.append(f"{missing_title} document(s) missing a meaningful title.")
    if missing_author:
        warns.append(f"{missing_author} document(s) missing author metadata "
                     f"(metadata questions like 'who wrote this' may fail for them).")
    if missing_date:
        warns.append(f"{missing_date} document(s) missing a publication date.")
    if dup_content:
        warns.append(f"{len(dup_content)} group(s) of documents share identical content "
                     f"(possible duplicates / canonical collapse needed).")

    # ── Chunks ────────────────────────────────────────────────────────────────
    if chunks:
        chunk_hashes = Counter(c.get("chunk_hash") or c.get("chunk_id") for c in chunks)
        dup_chunks = sum(n - 1 for n in chunk_hashes.values() if n > 1)
        undersized = sum(1 for c in chunks if len((c.get("text") or "")) < _UNDERSIZE)
        oversized = sum(1 for c in chunks if len((c.get("text") or "")) > _OVERSIZE)
        no_meta_header = sum(1 for c in chunks if not (c.get("meta_header") or "").strip())
        orphans = sum(1 for c in chunks
                      if (c.get("document_id") or c.get("doc_id")) not in doc_ids) if doc_ids else 0
        missing_chunk_title = sum(1 for c in chunks
                                  if not (c.get("title") or "").strip()
                                  or (c.get("title") or "").strip().lower() == "untitled")

        infos.append(f"{len(chunks)} chunks; avg "
                     f"{sum(len(c.get('text') or '') for c in chunks)//max(1,len(chunks))} chars.")
        if dup_chunks:
            warns.append(f"{dup_chunks} duplicate chunk(s) by content hash.")
        if oversized:
            warns.append(f"{oversized} oversized chunk(s) (> {_OVERSIZE} chars).")
        if undersized:
            warns.append(f"{undersized} undersized chunk(s) (< {_UNDERSIZE} chars).")
        if orphans:
            errors.append(f"{orphans} orphan chunk(s) whose document is not in documents.jsonl.")
        if no_meta_header:
            warns.append(f"{no_meta_header} chunk(s) missing a metadata header "
                         f"(rebuild so metadata is searchable).")
        if missing_chunk_title:
            warns.append(f"{missing_chunk_title} chunk(s) missing a title.")
    else:
        warns.append("No chunks found (chunks.jsonl missing) — run a build.")

    # ── Index coverage ────────────────────────────────────────────────────────
    idx_n = _load_index_count()
    if idx_n < 0:
        errors.append("FAISS index missing or unreadable.")
    elif chunks and idx_n != len(chunks):
        errors.append(f"Index/metadata mismatch: {idx_n} vectors vs {len(chunks)} chunks "
                      f"(some chunks have no embedding — rebuild the index).")
    elif chunks:
        infos.append(f"Index has {idx_n} vectors, matching {len(chunks)} chunks.")

    report = {
        "ok": not errors,
        "counts": {"documents": len(docs), "chunks": len(chunks), "index_vectors": idx_n},
        "errors": errors, "warnings": warns, "info": infos,
    }
    return report


def _print_report(report: dict) -> None:
    print("══════════════════════════════════════════════════════════")
    print(" KNOWLEDGE BASE VALIDATION")
    print("══════════════════════════════════════════════════════════")
    c = report["counts"]
    print(f" documents: {c['documents']}   chunks: {c['chunks']}   "
          f"index vectors: {c['index_vectors']}")
    for label, items in (("ERROR", report["errors"]),
                         ("WARN", report["warnings"]),
                         ("INFO", report["info"])):
        for it in items:
            print(f"  [{label}] {it}")
    print("──────────────────────────────────────────────────────────")
    print(" RESULT:", "✅ PASS" if report["ok"] else "❌ FAIL (errors present)")
    print("══════════════════════════════════════════════════════════")


def main() -> int:
    ap = argparse.ArgumentParser(description="Validate the knowledge base and index.")
    ap.add_argument("--json", action="store_true", help="Emit JSON instead of a report.")
    ap.add_argument("--strict", action="store_true", help="Exit non-zero on warnings too.")
    args = ap.parse_args()

    report = validate()
    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=2))
    else:
        _print_report(report)

    if report["errors"]:
        return 1
    if args.strict and report["warnings"]:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())