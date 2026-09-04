#!/usr/bin/env python3
"""
rescrape_all.py — One command to re-scrape EVERYTHING and rebuild the KB.

Use this for the initial, complete rebuild (or any time you want to force a
from-scratch crawl). It re-crawls every page and subpage of the public website
and the authenticated Commit KB, merges the result into
``data/processed/documents.jsonl``, and rebuilds the FAISS index.

    python scripts/rescrape_all.py                 # full re-crawl + cached reindex
    python scripts/rescrape_all.py --fresh-index   # also re-embed every chunk from scratch
    python scripts/rescrape_all.py --reset-state   # forget crawl history first (truly clean)

After this, keep it fresh automatically with the weekly scheduler:
    python scripts/scheduler.py            # long-running, fires every Monday 18:00
    # or register the Windows task: scripts/setup_windows_task.ps1

Notes
  • Commit KB is only crawled if COMMIT_KB_USERNAME / COMMIT_KB_PASSWORD are set
    in .env; otherwise it's skipped with a warning and the website is still done.
  • The downstream retriever, Streamlit app and Mattermost bot are unaffected —
    this only rewrites documents.jsonl + the index they already read.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import config                                   # noqa: E402
from src.utils import get_logger                          # noqa: E402
from scripts.update_knowledge_base import run             # noqa: E402
from src.incremental_index import rebuild_index           # noqa: E402

logger = get_logger("rescrape_all", config.SCRAPE_LOG)


def _print(msg: str) -> None:
    print(msg, flush=True)


def _reset_state() -> None:
    """Delete per-source crawl state so every page is treated as new."""
    for src in ("website", "commit_kb"):
        p = config.LOGS_DIR / f"{src}_crawl_state.json"
        try:
            if p.exists():
                p.unlink()
                _print(f"  · cleared crawl state: {p.name}")
        except Exception as exc:
            logger.warning(f"Could not clear {p}: {exc}")


def main() -> int:
    ap = argparse.ArgumentParser(description="Full re-scrape of website + Commit KB and rebuild the KB.")
    ap.add_argument("--reset-state", action="store_true",
                    help="Delete crawl-state files first (treat everything as new).")
    ap.add_argument("--fresh-index", action="store_true",
                    help="After crawling, re-embed EVERY chunk from scratch (ignore the embedding cache).")
    ap.add_argument("--website-only", action="store_true", help="Skip the Commit KB.")
    ap.add_argument("--commit-kb-only", action="store_true", help="Skip the public website.")
    ap.add_argument("--max-pages", type=int, default=None)
    ap.add_argument("--max-depth", type=int, default=None)
    args = ap.parse_args()

    website = not args.commit_kb_only
    commit_kb = not args.website_only

    _print("════════════════════════════════════════════════════════════")
    _print(" FULL RE-SCRAPE — website + Commit KB → documents.jsonl → index")
    _print("════════════════════════════════════════════════════════════")

    if args.reset_state:
        _print("── Reset crawl state ─────────────────────────────────────")
        _reset_state()

    try:
        # A full (non-incremental) crawl of both sources, merged + reindexed.
        # do_index is deferred when --fresh-index so we can force a scratch rebuild.
        summary = run(website=website, commit_kb=commit_kb, incremental=False,
                      do_index=not args.fresh_index,
                      max_pages=args.max_pages, max_depth=args.max_depth)

        if args.fresh_index:
            _print("── Fresh index (re-embed every chunk) ────────────────────")
            idx = rebuild_index(progress_cb=_print, use_cache=False)
            summary["index"] = idx

        _print("════════════════════════════════════════════════════════════")
        _print(f" DONE — {summary['merge']['total']} documents in the knowledge base.")
        if summary.get("index"):
            i = summary["index"]
            _print(f"        {i['chunks']:,} chunks embedded across {i['documents']} docs.")
        _print("════════════════════════════════════════════════════════════")
        return 0
    except Exception as exc:
        logger.error(f"rescrape_all failed: {exc}", exc_info=True)
        _print(f"❌ Failed: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())