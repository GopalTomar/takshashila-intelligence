"""
crawl_state.py — Per-source incremental crawl state.

The whole point of incremental crawling is: **fetch a page, and only re-process /
re-index it if its content actually changed.** To do that we remember, per source
(``website``, ``commit_kb``), a small record for every URL we've seen:

    url -> {
        "content_hash":  sha256 of the extracted text last time,
        "etag":          HTTP ETag header (if the server sent one),
        "last_modified": HTTP Last-Modified header (if any),
        "document_id":   the unified-doc id this URL maps to,
        "first_seen":    ISO timestamp,
        "last_seen":     ISO timestamp of the last crawl that saw this URL,
        "last_changed":  ISO timestamp of the last time the content actually changed,
    }

State lives in ``data/logs/<source>_crawl_state.json`` and is loaded on start,
checkpointed during a crawl, and saved at the end. Two cheap wins fall out of it:

1. **Conditional requests** — we can send ``If-None-Match`` / ``If-Modified-Since``
   so a well-behaved server can answer ``304 Not Modified`` and we skip the body
   entirely (no re-download, no re-extract).
2. **Change detection** — even when a server doesn't support conditional GET, we
   compare the fresh content hash to the stored one and skip re-processing when it
   is identical. Only genuinely new or changed pages flow downstream.

It also lets us detect **removals**: any URL in state that the latest crawl did
*not* see again is a candidate for deletion from the knowledge base.

This module has no heavy dependencies and is safe to import anywhere.
"""

from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set

from src import config
from src.utils import get_logger, now_iso

logger = get_logger("crawl_state", config.SCRAPE_LOG)


def _state_path(source: str) -> Path:
    safe = "".join(c if c.isalnum() or c in ("_", "-") else "_" for c in (source or "source"))
    return config.LOGS_DIR / f"{safe}_crawl_state.json"


class CrawlState:
    """Load / query / update the incremental crawl state for one source."""

    def __init__(self, source: str):
        self.source = source
        self.path = _state_path(source)
        self._lock = threading.Lock()
        self.urls: Dict[str, Dict] = self._load()
        # URLs actually seen during the *current* run (for removal detection).
        self._seen_this_run: Set[str] = set()

    # ── persistence ──────────────────────────────────────────────────────────────
    def _load(self) -> Dict[str, Dict]:
        if self.path.exists():
            try:
                data = json.loads(self.path.read_text(encoding="utf-8"))
                if isinstance(data, dict):
                    return data.get("urls", data)  # tolerate either shape
            except Exception as exc:
                logger.warning(f"Could not read crawl state {self.path.name}: {exc}; starting fresh.")
        return {}

    def save(self) -> None:
        with self._lock:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            payload = {"source": self.source, "saved_at": now_iso(), "urls": self.urls}
            tmp = self.path.with_suffix(".tmp")
            tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
            tmp.replace(self.path)  # atomic-ish write

    # ── queries ──────────────────────────────────────────────────────────────────
    def known(self, url: str) -> bool:
        return url in self.urls

    def stored_hash(self, url: str) -> Optional[str]:
        rec = self.urls.get(url)
        return rec.get("content_hash") if rec else None

    def is_unchanged(self, url: str, content_hash: str) -> bool:
        """True when we've seen this URL before and its content hash is identical."""
        return bool(content_hash) and self.stored_hash(url) == content_hash

    def conditional_headers(self, url: str) -> Dict[str, str]:
        """Return If-None-Match / If-Modified-Since headers for a known URL."""
        rec = self.urls.get(url) or {}
        headers: Dict[str, str] = {}
        if rec.get("etag"):
            headers["If-None-Match"] = rec["etag"]
        if rec.get("last_modified"):
            headers["If-Modified-Since"] = rec["last_modified"]
        return headers

    def document_id(self, url: str) -> Optional[str]:
        rec = self.urls.get(url)
        return rec.get("document_id") if rec else None

    # ── updates ──────────────────────────────────────────────────────────────────
    def mark_seen(self, url: str) -> None:
        """Record that this URL was encountered in the current run (even if 304)."""
        self._seen_this_run.add(url)
        rec = self.urls.get(url)
        if rec is not None:
            rec["last_seen"] = now_iso()

    def record(self, url: str, *, content_hash: str, document_id: str,
               etag: str = "", last_modified: str = "", changed: bool = True) -> None:
        """Insert/update the state record for a URL after processing it."""
        with self._lock:
            existing = self.urls.get(url, {})
            ts = now_iso()
            self.urls[url] = {
                "content_hash": content_hash,
                "etag": etag or existing.get("etag", ""),
                "last_modified": last_modified or existing.get("last_modified", ""),
                "document_id": document_id or existing.get("document_id", ""),
                "first_seen": existing.get("first_seen", ts),
                "last_seen": ts,
                "last_changed": ts if changed else existing.get("last_changed", ts),
            }
        self._seen_this_run.add(url)

    def remove(self, url: str) -> None:
        with self._lock:
            self.urls.pop(url, None)

    # ── removal detection ────────────────────────────────────────────────────────
    def unseen_urls(self) -> List[str]:
        """
        URLs known from previous runs that were NOT seen in the current run.
        These are candidates for deletion from the knowledge base (the page 404'd
        or was dropped from discovery). Only meaningful after a *full* discovery
        pass, so callers gate removal on a complete crawl.
        """
        return [u for u in self.urls if u not in self._seen_this_run]

    def document_ids_for(self, urls: Iterable[str]) -> List[str]:
        out = []
        for u in urls:
            did = self.document_id(u)
            if did:
                out.append(did)
        return out

    def begin_run(self) -> None:
        """Reset the per-run 'seen' set at the start of a crawl."""
        self._seen_this_run = set()

    def stats(self) -> Dict[str, int]:
        return {"known_urls": len(self.urls), "seen_this_run": len(self._seen_this_run)}