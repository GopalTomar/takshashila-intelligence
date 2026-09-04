#!/usr/bin/env python3
"""
build_index.py — (Re)build the FAISS index from documents.jsonl.

Chunks every unified document and rebuilds the index, embedding only new/changed
chunks via the embedding cache (so even a "full" rebuild is fast after the first
run). Writes the same files the retriever/bot already read
(``faiss.index`` + ``metadata.pkl`` + ``metadata.json``).

    python scripts/build_index.py                 # cached rebuild
    python scripts/build_index.py --no-cache      # embed everything from scratch
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.incremental_index import rebuild_index  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description="Rebuild the FAISS index from documents.jsonl.")
    ap.add_argument("--no-cache", action="store_true", help="Embed all chunks from scratch.")
    args = ap.parse_args()
    summary = rebuild_index(progress_cb=lambda m: print(m, flush=True), use_cache=not args.no_cache)
    print(f"Done: {summary}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())