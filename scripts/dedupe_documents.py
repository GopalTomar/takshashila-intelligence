#!/usr/bin/env python3
"""
dedupe_documents.py — Repair an accumulated documents.jsonl in place.

Collapses documents that point at the same URL (keeping the richest copy), then
rebuilds the FAISS index. Because embeddings are cached by content hash, the
rebuild reuses cached vectors and is fast — it does NOT re-embed the whole KB.

Use this when validation reports "duplicate document URL(s)". These accumulate
when an older document-id scheme (or the legacy build path) stored a page that
the newer crawler also stored under a different id.

    python scripts/dedupe_documents.py            # collapse + cached rebuild + validate
    python scripts/dedupe_documents.py --dry-run  # report what would change, no writes
    python scripts/dedupe_documents.py --no-index # collapse documents.jsonl only

A timestamped backup of documents.jsonl is written next to it before any change.
"""

from __future__ import annotations

import argparse
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import config                                     # noqa: E402
from src.utils import load_jsonl, save_jsonl, get_logger   # noqa: E402
from src.incremental_index import collapse_by_url, rebuild_index  # noqa: E402

logger = get_logger("dedupe_documents", config.SCRAPE_LOG)


def _print(m: str) -> None:
    print(m, flush=True)


def main() -> int:
    ap = argparse.ArgumentParser(description="Collapse duplicate-URL documents and rebuild.")
    ap.add_argument("--dry-run", action="store_true", help="Report only; write nothing.")
    ap.add_argument("--no-index", action="store_true", help="Fix documents.jsonl but skip reindex.")
    args = ap.parse_args()

    path = config.DOCUMENTS_FILE
    if not path.exists():
        _print(f"❌ {path} not found.")
        return 1

    docs = load_jsonl(path)
    kept, dropped = collapse_by_url(docs)

    distinct_urls = len({(d.get("canonical_url") or d.get("url") or "").rstrip("/").lower()
                         for d in kept})
    _print("════════════════════════════════════════════════════════════")
    _print(" DOCUMENT DEDUPLICATION")
    _print("════════════════════════════════════════════════════════════")
    _print(f" before : {len(docs)} documents")
    _print(f" after  : {len(kept)} documents  ({dropped} duplicate-URL copies removed)")
    _print(f" URLs   : {distinct_urls} distinct")

    if args.dry_run:
        _print("\n(dry run — no files written)")
        return 0

    if dropped == 0:
        _print("\nNo duplicate URLs found — nothing to change.")
    else:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        backup = path.with_name(f"documents.backup_{stamp}.jsonl")
        shutil.copy2(path, backup)
        _print(f"\n Backup written: {backup.name}")
        save_jsonl(path, kept)
        _print(f" Rewrote {path.name} with {len(kept)} documents.")

    if not args.no_index and dropped:
        _print("\n── Rebuilding index (reuses cached embeddings) ──────────────")
        summary = rebuild_index(progress_cb=_print, use_cache=True)
        _print(f" Index: {summary['chunks']} chunks from {summary['documents']} docs "
               f"(embedded {summary['embedded']}, reused {summary['cached']}).")

    # Validate the result
    try:
        from scripts.validate_kb import validate
        rep = validate()
        _print("\n── Validation ───────────────────────────────────────────────")
        _print(" result: " + ("✅ PASS" if rep["ok"] else "❌ FAIL"))
        for e in rep["errors"]:
            _print(f"   ERROR: {e}")
        for w in rep["warnings"]:
            _print(f"   WARN:  {w}")
    except Exception as exc:
        logger.warning(f"Validation step failed: {exc}")

    _print("════════════════════════════════════════════════════════════")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
