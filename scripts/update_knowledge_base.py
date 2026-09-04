#!/usr/bin/env python3
"""
update_knowledge_base.py — One command to keep the knowledge base fresh.

Runs an **incremental** crawl of both sources, merges only the new/changed/removed
documents into ``data/processed/documents.jsonl``, then rebuilds the FAISS index
re-embedding **only** the chunks that actually changed. Re-run it any time (a cron
job, a button, by hand) — you never re-scrape or re-embed the whole corpus again.

    # incremental update of everything (website + Commit KB) + reindex
    python scripts/update_knowledge_base.py

    # only one source
    python scripts/update_knowledge_base.py --website-only
    python scripts/update_knowledge_base.py --commit-kb-only

    # ignore state and re-crawl everything (still cached-embeds unchanged chunks)
    python scripts/update_knowledge_base.py --full

    # crawl + merge but skip the reindex (reindex later with build_index.py)
    python scripts/update_knowledge_base.py --no-index

Exit code is non-zero only on a hard failure, so it's safe to schedule.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import config                                   # noqa: E402
from src.utils import get_logger, now_iso, load_jsonl, content_hash  # noqa: E402
from scripts.crawl_engine import (                        # noqa: E402
    website_config, commit_kb_config, crawl_site,
)
from src.incremental_index import merge_documents, rebuild_index  # noqa: E402

logger = get_logger("update_kb", config.SCRAPE_LOG)


def _load_local_sources() -> list:
    """
    Load hand-supplied local sources — the Staff Handbook and any other local
    documents — from data/knowledge_base/, if present.

    Audit finding: config.STAFF_HANDBOOK_FILE and config.LOCAL_DOCUMENTS_FILE
    were defined in src/config.py and documented in README.md ("drop a JSONL
    at data/knowledge_base/takshashila_staff_handbook.jsonl... it is picked
    up automatically as a supporting source") but were never actually read
    anywhere in this pipeline — run() only ever crawled website + commit_kb.
    A handbook file placed there would have been silently ignored forever.

    These files are hand-maintained, not crawled, so there's no "removed"
    concept for them here (delete a document from the file yourself, then
    remove it from documents.jsonl via scripts/dedupe_documents.py or a
    manual edit, same as any other document). To avoid forcing a reindex on
    every single run just because these static files were re-read, only
    documents whose content actually changed (by content_hash) since the
    last merge are returned.
    """
    existing_by_id = {}
    for d in load_jsonl(config.DOCUMENTS_FILE):
        did = d.get("document_id") or d.get("id") or d.get("url_hash") or d.get("url")
        if did:
            existing_by_id[did] = d

    changed = []
    for path, default_source, default_source_name in (
        (config.STAFF_HANDBOOK_FILE, "staff_handbook", "Takshashila Staff Handbook"),
        (config.LOCAL_DOCUMENTS_FILE, "local", "Local Document"),
    ):
        if not path.exists():
            continue
        for i, raw in enumerate(load_jsonl(path)):
            doc = dict(raw)
            doc.setdefault("source", default_source)
            doc.setdefault("source_name", default_source_name)
            if not doc.get("document_id"):
                base = doc.get("url") or doc.get("id") or f"{path.stem}_{i}"
                doc["document_id"] = f"{default_source}_{content_hash(str(base))}"
            text = doc.get("text", "") or ""
            chash = content_hash(text)
            doc["content_hash"] = chash
            prev = existing_by_id.get(doc["document_id"])
            if prev is None or prev.get("content_hash") != chash:
                changed.append(doc)
    return changed


def _print(msg: str) -> None:
    print(msg, flush=True)


def _write_manifest(summary: dict) -> None:
    """Persist a small JSON manifest of the latest run for the UI / audit."""
    try:
        config.LOGS_DIR.mkdir(parents=True, exist_ok=True)
        payload = {"finished_at": now_iso(), **summary}
        config.WEBSITE_MANIFEST_FILE.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    except Exception as exc:                              # never fail a run on this
        logger.warning(f"Could not write run manifest: {exc}")


def _write_ingestion_report(summary: dict) -> None:
    """Write a timestamped, human-readable ingestion + crawl report (md + json)."""
    try:
        config.LOGS_DIR.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        (config.LOGS_DIR / f"ingestion_report_{stamp}.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

        lines = [f"# Ingestion report — {summary.get('mode','?')} run",
                 f"Finished: {now_iso()}  ·  Duration: {summary.get('duration_seconds','?')}s", ""]
        for src, counts in (summary.get("per_source") or {}).items():
            lines.append(f"## {src}")
            if isinstance(counts, dict):
                for k in ("added", "updated", "unchanged", "removed", "failed",
                          "pdf", "skipped_nav", "not_modified", "skipped"):
                    if k in counts:
                        lines.append(f"- {k}: {counts[k]}")
            lines.append("")
        m = summary.get("merge") or {}
        lines += ["## Knowledge base",
                  f"- added: {m.get('added',0)}  ·  updated: {m.get('updated',0)}  ·  "
                  f"removed: {m.get('removed',0)}  ·  total: {m.get('total',0)}", ""]
        if summary.get("index"):
            i = summary["index"]
            lines += ["## Index",
                      f"- documents: {i.get('documents','?')}  ·  chunks: {i.get('chunks','?')}  ·  "
                      f"embedded: {i.get('embedded','?')}  ·  reused: {i.get('cached','?')}", ""]
        v = summary.get("validation")
        if v:
            lines.append("## Validation")
            lines.append(f"- result: {'PASS' if v.get('ok') else 'FAIL'}")
            for e in v.get("errors", []):
                lines.append(f"  - ERROR: {e}")
            for w in v.get("warnings", [])[:10]:
                lines.append(f"  - WARN: {w}")
        (config.LOGS_DIR / f"ingestion_report_{stamp}.md").write_text(
            "\n".join(lines), encoding="utf-8")
    except Exception as exc:
        logger.warning(f"Could not write ingestion report: {exc}")


def run(website: bool = True, commit_kb: bool = True, incremental: bool = True,
        do_index: bool = True, max_pages=None, max_depth=None) -> dict:
    """Crawl the requested sources, merge the delta, and (optionally) reindex."""
    started = time.time()
    all_new_docs = []
    all_removed = []
    per_source = {}

    if website:
        _print("── Website ───────────────────────────────────────────────")
        site = website_config(max_pages=max_pages, max_depth=max_depth)
        res = crawl_site(site, incremental=incremental, progress_cb=_print)
        all_new_docs += res.docs
        all_removed += res.removed_ids
        per_source["website"] = res.counts

    if commit_kb:
        if not (config.COMMIT_KB_USERNAME and config.COMMIT_KB_PASSWORD):
            _print("⚠️  Commit KB credentials not set (COMMIT_KB_USERNAME / "
                   "COMMIT_KB_PASSWORD) — skipping Commit KB.")
            per_source["commit_kb"] = {"skipped": "no credentials"}
        else:
            _print("── Commit KB ─────────────────────────────────────────────")
            site = commit_kb_config(max_pages=max_pages, max_depth=max_depth)
            res = crawl_site(site, incremental=incremental, progress_cb=_print)
            all_new_docs += res.docs
            all_removed += res.removed_ids
            per_source["commit_kb"] = res.counts

    # ── Hand-supplied local sources (Staff Handbook + local documents) ────────────
    local_docs = _load_local_sources()
    if local_docs:
        _print("── Local sources (Staff Handbook / local documents) ────────")
        _print(f"{len(local_docs)} document(s) new/changed")
        all_new_docs += local_docs
        by_source = {}
        for d in local_docs:
            by_source[d["source"]] = by_source.get(d["source"], 0) + 1
        per_source["local_sources"] = by_source

    # ── Merge the delta into documents.jsonl ──────────────────────────────────────
    _print("── Merge ─────────────────────────────────────────────────")
    merge = merge_documents(all_new_docs, removed_ids=all_removed)
    _print(f"documents.jsonl → +{merge['added']} new, ~{merge['updated']} changed, "
           f"-{merge['removed']} removed  (total {merge['total']})")

    from src import kb_version

    # Snapshot the currently-published version BEFORE we rebuild, so a broken
    # rebuild can be rolled back to it (Sections 13/14/32/40: a failed crawl or
    # invalid index must never replace a valid production KB).
    prev_version = kb_version.current_version()

    index_summary = None
    rolled_back = None
    changed = merge["added"] + merge["updated"] + merge["removed"]

    # Safety guard: refuse to reindex a delta that would empty (or all-but-empty)
    # the corpus — the hallmark of a transient source outage returning 0 pages,
    # which would otherwise wipe a good KB. Preserve the existing index instead.
    existing_total = len(load_jsonl(config.DOCUMENTS_FILE)) if config.DOCUMENTS_FILE.exists() else 0
    if do_index and changed and merge["total"] == 0 and existing_total > 0:
        _print("⚠️  Refusing to reindex: the merge would leave 0 documents "
               "(likely a source outage). Keeping the previous index.")
        changed = 0

    if do_index and changed:
        _print("── Reindex (cached embeddings) ───────────────────────────")
        index_summary = rebuild_index(progress_cb=_print, use_cache=True)
    elif do_index:
        _print("No document changes — index left untouched.")

    # ── Validate the resulting KB; roll back to the previous good version on ──────
    #    a hard validation FAILURE so a broken build never stays published.
    validation = None
    if do_index and index_summary:
        try:
            from scripts.validate_kb import validate
            validation = validate()
            status = "✅ PASS" if validation["ok"] else "❌ FAIL"
            _print(f"── Validation: {status} "
                   f"({len(validation['errors'])} errors, {len(validation['warnings'])} warnings)")
            for e in validation["errors"]:
                _print(f"   ERROR: {e}")
            if not validation["ok"] and prev_version and prev_version != "unversioned":
                try:
                    kb_version.rollback(prev_version)
                    rolled_back = prev_version
                    _print(f"⛔ Validation failed — rolled back to previous good "
                           f"version {prev_version}.")
                except Exception as rexc:
                    logger.error(f"Rollback after failed validation did not succeed: {rexc}")
                    _print(f"❗ Validation failed AND rollback failed: {rexc}")
        except Exception as exc:
            logger.warning(f"Validation step failed: {exc}")

    summary = {
        "mode": "full" if not incremental else "incremental",
        "duration_seconds": round(time.time() - started, 1),
        "per_source": per_source,
        "merge": merge,
        "index": index_summary,
        "changed_documents": changed,
        "validation": validation,
        "kb_version": kb_version.current_version(),
        "rolled_back_to": rolled_back,
    }
    _write_manifest(summary)
    _write_ingestion_report(summary)
    _print("── Done ──────────────────────────────────────────────────")
    _print(f"Summary: {summary}")
    return summary


def main() -> int:
    ap = argparse.ArgumentParser(description="Incrementally update the Takshashila knowledge base.")
    ap.add_argument("--website-only", action="store_true", help="Crawl only the public website.")
    ap.add_argument("--commit-kb-only", action="store_true", help="Crawl only the Commit KB.")
    ap.add_argument("--full", "--rescrape-all", dest="full", action="store_true",
                    help="Ignore crawl state and re-crawl everything (still cache-embeds).")
    ap.add_argument("--no-index", action="store_true", help="Crawl + merge but skip the reindex.")
    ap.add_argument("--max-pages", type=int, default=None)
    ap.add_argument("--max-depth", type=int, default=None)
    args = ap.parse_args()

    website = not args.commit_kb_only
    commit_kb = not args.website_only

    try:
        run(website=website, commit_kb=commit_kb, incremental=not args.full,
            do_index=not args.no_index, max_pages=args.max_pages, max_depth=args.max_depth)
        return 0
    except Exception as exc:
        logger.error(f"update_knowledge_base failed: {exc}", exc_info=True)
        _print(f"❌ Failed: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())