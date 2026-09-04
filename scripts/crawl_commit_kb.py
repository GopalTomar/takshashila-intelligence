#!/usr/bin/env python3
"""
crawl_commit_kb.py — Incrementally crawl only the authenticated Commit KB.

Requires COMMIT_KB_URL / COMMIT_KB_USERNAME / COMMIT_KB_PASSWORD in .env.

    python scripts/crawl_commit_kb.py               # incremental (default)
    python scripts/crawl_commit_kb.py --full        # re-crawl everything
    python scripts/crawl_commit_kb.py --no-index    # crawl + merge, skip reindex

Thin wrapper around update_knowledge_base.run(website=False, commit_kb=True).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import config  # noqa: E402
from scripts.update_knowledge_base import run  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description="Incrementally crawl the Commit KB.")
    ap.add_argument("--full", action="store_true", help="Ignore state; re-crawl everything.")
    ap.add_argument("--no-index", action="store_true", help="Skip the reindex step.")
    ap.add_argument("--max-pages", type=int, default=None)
    ap.add_argument("--max-depth", type=int, default=None)
    args = ap.parse_args()

    if not (config.COMMIT_KB_USERNAME and config.COMMIT_KB_PASSWORD):
        print("❌ COMMIT_KB_USERNAME / COMMIT_KB_PASSWORD are not set in .env.")
        return 1

    run(website=False, commit_kb=True, incremental=not args.full,
        do_index=not args.no_index, max_pages=args.max_pages, max_depth=args.max_depth)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())