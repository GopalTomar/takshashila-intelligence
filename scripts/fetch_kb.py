#!/usr/bin/env python3
"""
fetch_kb.py — download the published KB artifact into data/index/.

The always-on backend calls this on boot (and can be scheduled to call it
periodically) so it serves the exact index CI built + validated, without
crawling or embedding itself. Downloads to a temp file, extracts atomically over
the active index, and refuses to overwrite a good local index with a failed
download — a network error never destroys the KB already on disk.

    KB_ARTIFACT_URL=https://github.com/<owner>/<repo>/releases/latest/download/kb-latest.tar.gz \
        python scripts/fetch_kb.py

Exit codes: 0 = fetched (or already current), 2 = no URL configured,
3 = download/extract failed (existing index left intact).
"""
from __future__ import annotations

import os
import sys
import tarfile
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import config  # noqa: E402
from src.utils import get_logger  # noqa: E402

logger = get_logger("fetch_kb", config.SCRAPE_LOG)


def fetch(url: str) -> int:
    import httpx
    config.INDEX_DIR.mkdir(parents=True, exist_ok=True)
    tmp = Path(tempfile.mkdtemp(prefix="kb_fetch_"))
    tarball = tmp / "kb.tar.gz"
    try:
        logger.info("Fetching KB artifact from %s", url)
        with httpx.stream("GET", url, follow_redirects=True, timeout=120) as r:
            r.raise_for_status()
            with open(tarball, "wb") as f:
                for chunk in r.iter_bytes():
                    f.write(chunk)
        # Extract into a staging dir first, then move files into place.
        with tarfile.open(tarball, "r:gz") as tar:
            tar.extractall(tmp, filter="data")
        staged = tmp / "index"
        if not (staged / config.FAISS_INDEX.name).exists():
            raise RuntimeError("artifact did not contain a FAISS index")
        for p in staged.iterdir():
            dest = config.INDEX_DIR / p.name
            os.replace(str(p), str(dest))
        logger.info("KB artifact installed into %s", config.INDEX_DIR)
        print("KB artifact fetched and installed.")
        return 0
    except Exception as exc:
        logger.error("KB fetch failed (%s) — keeping existing index.", exc)
        print(f"KB fetch failed: {exc} (existing index left intact)")
        return 3
    finally:
        import shutil
        shutil.rmtree(tmp, ignore_errors=True)


def main() -> int:
    url = os.getenv("KB_ARTIFACT_URL", "").strip()
    if not url:
        print("KB_ARTIFACT_URL not set — nothing to fetch.")
        return 2
    return fetch(url)


if __name__ == "__main__":
    raise SystemExit(main())
