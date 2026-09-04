#!/usr/bin/env python3
"""
crawl_website.py — Incrementally crawl only the public Takshashila website.

    python scripts/crawl_website.py                 # incremental (default)
    python scripts/crawl_website.py --full          # re-crawl everything
    python scripts/crawl_website.py --no-index      # crawl + merge, skip reindex

Thin wrapper around update_knowledge_base.run(website=True, commit_kb=False).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.update_knowledge_base import run  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description="Incrementally crawl the public website.")
    ap.add_argument("--full", action="store_true", help="Ignore state; re-crawl everything.")
    ap.add_argument("--no-index", action="store_true", help="Skip the reindex step.")
    ap.add_argument("--max-pages", type=int, default=None)
    ap.add_argument("--max-depth", type=int, default=None)
    args = ap.parse_args()
    run(website=True, commit_kb=False, incremental=not args.full,
        do_index=not args.no_index, max_pages=args.max_pages, max_depth=args.max_depth)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())