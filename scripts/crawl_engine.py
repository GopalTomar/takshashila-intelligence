"""
crawl_engine.py — Robust, incremental, RAG-ready crawler.

One engine crawls both the **public website** (``takshashila.org.in``) and the
**authenticated Commit KB**. It borrows the resilient design of the reference
notebook — sitemap + robots discovery, a BFS crawl over internal links with a
thread pool, conditional/HTTP-hash change detection, and PDF handling — but emits
documents in the **existing project schema** (the same shape ``src.scraper``
produced) so the rest of the pipeline (chunker, index, retriever, bot) is
unchanged.

Incremental behaviour (the whole point):
  * every discovered URL is fetched (or answered ``304`` via conditional GET);
  * its content hash is compared to :class:`~src.crawl_state.CrawlState`;
  * **unchanged** pages are skipped (no re-extract, no re-index);
  * **new** and **changed** pages become documents;
  * pages known before but not seen this run are reported as **removed**.

The result is a :class:`CrawlResult` (new/changed docs + removed doc-ids + counts)
that the caller merges into ``documents.jsonl`` and re-indexes.

Only lightweight deps are imported at module load (requests, bs4, trafilatura);
PDF text extraction reuses ``src.extractors.extract_pdf_text`` (PyMuPDF), imported
lazily.
"""

from __future__ import annotations

import re
import threading
import time
import urllib.parse
import urllib.robotparser
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Set, Tuple

import requests
from bs4 import BeautifulSoup

from src import config
from src.crawl_state import CrawlState
from src.utils import (
    canonicalize_for_dedup, clean_document_metadata, clean_text, content_hash,
    get_logger, is_listing_or_landing, looks_like_pdf,
    normalize_url, now_iso, parse_date, url_hash,
)

logger = get_logger("crawl_engine", config.SCRAPE_LOG)


# ════════════════════════════════════════════════════════════════════════════════
#  Site configuration
# ════════════════════════════════════════════════════════════════════════════════

@dataclass
class SiteConfig:
    """Everything the engine needs to crawl one source."""

    base_url: str
    domain: str
    source: str                       # "website" | "commit_kb"
    source_name: str
    # Additional hostnames treated as "internal" alongside ``domain`` — for
    # sites whose content is genuinely split across subdomains (see
    # config.WEBSITE_ADDITIONAL_DOMAINS for a documented, evidence-based
    # example: legion.takshashila.org.in / school.takshashila.org.in carry
    # real institutional content — team, mission, blogs, courses — under a
    # different hostname than the primary takshashila.org.in). Empty by
    # default so behaviour is unchanged unless explicitly opted into.
    additional_domains: Tuple[str, ...] = ()
    sitemap_urls: List[str] = field(default_factory=list)
    listing_urls: List[Tuple[str, str]] = field(default_factory=list)
    # include_patterns: optional KEEP-hint. When non-empty, only pages whose URL
    # matches one of these are kept as documents (discovery still follows all
    # links). Leave EMPTY for full-site coverage.
    include_patterns: List[str] = field(default_factory=list)
    # exclude_patterns: legacy single list. If follow_exclude_patterns /
    # doc_exclude_patterns are not given, this is used for BOTH tiers so older
    # callers keep working unchanged.
    exclude_patterns: List[str] = field(default_factory=list)
    # follow_exclude_patterns: URLs never enqueued/visited (admin/auth/media/…).
    follow_exclude_patterns: List[str] = field(default_factory=list)
    # doc_exclude_patterns: URLs visited (for discovery) but never kept as docs
    # (navigation / archive / search / pagination pages).
    doc_exclude_patterns: List[str] = field(default_factory=list)
    skip_extensions: Tuple[str, ...] = ()
    min_text_len: int = 200
    max_pages: int = 500
    max_depth: int = 4
    include_pdfs: bool = True
    respect_robots: bool = True
    username: str = ""
    password: str = ""
    max_workers: int = 8

    def __post_init__(self):
        # Back-compat: if the two-tier lists weren't supplied, derive them from
        # the single exclude_patterns list (both tiers get the same rules).
        if not self.follow_exclude_patterns:
            self.follow_exclude_patterns = list(self.exclude_patterns)
        if not self.doc_exclude_patterns:
            self.doc_exclude_patterns = list(self.exclude_patterns)


def website_config(max_pages: Optional[int] = None, max_depth: Optional[int] = None,
                   include_pdfs: Optional[bool] = None) -> SiteConfig:
    """
    SiteConfig for the public Takshashila website, tuned for FULL coverage:
      * include_patterns is empty → keep every content page (not just a few
        hard-coded path prefixes);
      * the crawler follows every internal link except the follow-excludes, so
        pages reachable only via category/tag/archive pages are still found;
      * navigation/archive/search/pagination pages are visited but not kept.
    """
    return SiteConfig(
        base_url=config.WEBSITE_BASE_URL,
        domain=config.WEBSITE_DOMAIN,
        source="website",
        source_name="Takshashila Website",
        sitemap_urls=list(config.WEBSITE_SITEMAP_URLS),
        listing_urls=list(config.WEBSITE_LISTING_URLS),
        include_patterns=[],   # keep everything that passes the text filter
        additional_domains=tuple(config.WEBSITE_ADDITIONAL_DOMAINS),
        follow_exclude_patterns=list(config.WEBSITE_FOLLOW_EXCLUDE_PATTERNS),
        doc_exclude_patterns=list(config.WEBSITE_DOC_EXCLUDE_PATTERNS),
        skip_extensions=tuple(config.WEBSITE_SKIP_EXTENSIONS),
        min_text_len=config.WEBSITE_MIN_TEXT_LEN,
        max_pages=config.WEBSITE_MAX_PAGES if max_pages is None else max_pages,
        max_depth=config.WEBSITE_MAX_DEPTH if max_depth is None else max_depth,
        include_pdfs=config.WEBSITE_INCLUDE_PDFS if include_pdfs is None else include_pdfs,
        respect_robots=True,
        max_workers=config.SCRAPE_MAX_WORKERS,
    )


def commit_kb_config(max_pages: Optional[int] = None, max_depth: Optional[int] = None,
                     include_pdfs: Optional[bool] = None) -> SiteConfig:
    """SiteConfig for the authenticated Commit KB, read from src.config + .env."""
    base = config.COMMIT_KB_URL.rstrip("/") + "/"
    domain = urllib.parse.urlparse(base).netloc
    return SiteConfig(
        base_url=base,
        domain=domain,
        source="commit_kb",
        source_name="Commit KB",
        sitemap_urls=[urllib.parse.urljoin(base, "sitemap.xml"),
                      urllib.parse.urljoin(base, "sitemap_index.xml")],
        listing_urls=[],
        include_patterns=[],            # crawl + keep everything internal (KB is small + all relevant)
        # Follow everything except auth/admin/feeds; the KB is a small static
        # Quarto site so full internal following captures every note/decision.
        follow_exclude_patterns=["/wp-admin/", "/wp-login/", "/login", "/logout",
                                 "mailto:", "tel:", "javascript:", "#",
                                 "?share=", "?replytocom=", "/feed/"],
        # Keep every KB page (even the short category landing pages carry useful
        # summaries); the chunker drops any genuinely empty/low-quality text.
        doc_exclude_patterns=[],
        skip_extensions=tuple(config.WEBSITE_SKIP_EXTENSIONS),
        min_text_len=60,                # KB pages are short but meaningful
        max_pages=config.WEBSITE_MAX_PAGES if max_pages is None else max_pages,
        max_depth=8 if max_depth is None else max_depth,
        include_pdfs=config.WEBSITE_INCLUDE_PDFS if include_pdfs is None else include_pdfs,
        respect_robots=False,           # authenticated internal KB
        username=config.COMMIT_KB_USERNAME,
        password=config.COMMIT_KB_PASSWORD,
        max_workers=config.SCRAPE_MAX_WORKERS,
    )


# ════════════════════════════════════════════════════════════════════════════════
#  Crawl result
# ════════════════════════════════════════════════════════════════════════════════

@dataclass
class CrawlResult:
    docs: List[Dict] = field(default_factory=list)          # new + changed unified docs
    removed_ids: List[str] = field(default_factory=list)     # document_ids to drop
    counts: Dict[str, int] = field(default_factory=dict)
    discovered: int = 0


# ════════════════════════════════════════════════════════════════════════════════
#  The engine
# ════════════════════════════════════════════════════════════════════════════════

class CrawlEngine:
    def __init__(self, site: SiteConfig, state: CrawlState,
                 incremental: bool = True, progress_cb: Optional[Callable] = None):
        self.site = site
        self.state = state
        self.incremental = incremental
        self.progress_cb = progress_cb

        self.session = self._build_session()
        self.robots = self._load_robots() if site.respect_robots else None

        self.frontier: deque = deque()
        self.seen: Set[str] = set()
        self.depth: Dict[str, int] = {}
        self.result = CrawlResult()
        self._counts = {"added": 0, "updated": 0, "unchanged": 0,
                        "failed": 0, "pdf": 0, "not_modified": 0, "skipped_nav": 0}
        self._counts_lock = threading.Lock()

    def _bump(self, key: str, n: int = 1) -> None:
        """Thread-safe counter increment (workers update counts concurrently)."""
        with self._counts_lock:
            self._counts[key] = self._counts.get(key, 0) + n

    # ── setup ────────────────────────────────────────────────────────────────────
    def _build_session(self) -> requests.Session:
        from requests.adapters import HTTPAdapter, Retry
        s = requests.Session()
        retries = Retry(total=config.SCRAPE_MAX_RETRIES, backoff_factor=1.0,
                        status_forcelist=[429, 500, 502, 503, 504],
                        allowed_methods=["GET", "HEAD"])
        adapter = HTTPAdapter(max_retries=retries, pool_maxsize=20)
        s.mount("http://", adapter)
        s.mount("https://", adapter)
        s.headers.update({"User-Agent": config.USER_AGENT})
        if self.site.username:
            s.auth = (self.site.username, self.site.password)
        return s

    def _load_robots(self) -> Optional[urllib.robotparser.RobotFileParser]:
        rp = urllib.robotparser.RobotFileParser()
        robots_url = urllib.parse.urljoin(self.site.base_url, "/robots.txt")
        try:
            r = self.session.get(robots_url, timeout=config.SCRAPE_TIMEOUT)
            if r.status_code == 200:
                rp.parse(r.text.splitlines())
                return rp
        except Exception as exc:
            logger.debug(f"robots.txt not loaded: {exc}")
        return None

    def _can_fetch(self, url: str) -> bool:
        if not (self.site.respect_robots and self.robots):
            return True
        try:
            return self.robots.can_fetch(config.USER_AGENT, url)
        except Exception:
            return True

    def _log(self, msg: str) -> None:
        if self.progress_cb:
            self.progress_cb(msg)

    # ── discovery ────────────────────────────────────────────────────────────────
    def _discover_sitemap(self) -> List[str]:
        found: List[str] = []
        seen_sm: Set[str] = set()

        def parse(url: str, depth: int = 0):
            if depth > 4 or url in seen_sm:
                return
            seen_sm.add(url)
            try:
                r = self.session.get(url, timeout=config.SCRAPE_TIMEOUT)
                if r.status_code != 200:
                    return
                root = ET.fromstring(r.content)
                ns = {"sm": "http://www.sitemaps.org/schemas/sitemap/0.9"}
                for sm in root.findall(".//sm:sitemap/sm:loc", ns):
                    if sm.text:
                        parse(sm.text.strip(), depth + 1)
                for loc in root.findall(".//sm:url/sm:loc", ns):
                    if loc.text:
                        found.append(loc.text.strip())
            except Exception as exc:
                logger.debug(f"sitemap parse failed {url}: {exc}")

        for sm in self.site.sitemap_urls:
            parse(sm)
        internal = [u for u in dict.fromkeys(found) if self._is_internal(u)]
        logger.info(f"[{self.site.source}] sitemap discovered {len(internal)} URLs")
        return internal

    # ── url helpers ──────────────────────────────────────────────────────────────
    def _allowed_domains(self) -> Set[str]:
        doms = {self.site.domain.lower().replace("www.", "")}
        doms |= {d.lower().replace("www.", "") for d in self.site.additional_domains if d}
        return doms

    def _is_internal(self, url: str) -> bool:
        try:
            netloc = urllib.parse.urlparse(url).netloc.lower().replace("www.", "")
        except Exception:
            return False
        return netloc in self._allowed_domains()

    def _extension(self, url: str) -> str:
        return Path(urllib.parse.urlparse(url).path).suffix.lower()

    def _should_follow(self, url: str) -> bool:
        """
        Discovery tier — should this internal URL be enqueued and visited?

        We follow *everything* internal so the BFS reaches every page and
        subpage, EXCEPT: skipped asset extensions and the follow-exclude list
        (admin, auth, cart, feeds, JSON APIs, mail/tel/js links, same-page
        fragments). Category / tag / author / pagination pages ARE followed so
        that content reachable only through them is discovered.
        """
        if not self._is_internal(url):
            return False
        ext = self._extension(url)
        if ext in self.site.skip_extensions:
            return False
        low = url.lower()
        if any(p in low for p in self.site.follow_exclude_patterns):
            return False
        return True

    def _should_keep_doc(self, url: str) -> bool:
        """
        Keep tier — should a *visited* page be stored as a KB document?

        Navigation / archive / search / pagination pages are visited (for their
        outbound links) but not kept. When include_patterns is non-empty it acts
        as a keep-whitelist; empty means "keep every content page" (the default
        for full-site coverage). Final acceptance still requires the page to
        pass the min-text-length filter in _build_doc.
        """
        low = url.lower()
        if any(p in low for p in self.site.doc_exclude_patterns):
            return False
        # Aggregate index / section-landing / homepage pages are followed for
        # their links but never stored as documents — so answers cite the
        # specific article, and its reference link opens the exact page. The
        # Commit KB opts out (its short category pages carry real summaries).
        if self.site.source != "commit_kb" and is_listing_or_landing(url):
            return False
        if self.site.include_patterns:
            if not any(p in low for p in self.site.include_patterns):
                return False
        return True

    # ── fetching ─────────────────────────────────────────────────────────────────
    def _fetch(self, url: str) -> Tuple[Optional[requests.Response], str]:
        """
        GET a URL with conditional headers. Returns (response, status) where
        status is one of: ``ok`` (200 body), ``not_modified`` (304),
        ``gone`` (404/410 — a real removal), ``error`` (transient/other).
        """
        headers = self.state.conditional_headers(url) if self.incremental else {}
        try:
            r = self.session.get(url, timeout=config.SCRAPE_TIMEOUT,
                                 headers=headers, allow_redirects=True)
            time.sleep(config.SCRAPE_DELAY)
            if r.status_code == 304:
                return None, "not_modified"
            if r.status_code in (404, 410):
                return None, "gone"
            if r.status_code >= 400:
                logger.warning(f"[{self.site.source}] HTTP {r.status_code} for {url}")
                return None, "error"
            return r, "ok"
        except Exception as exc:
            logger.warning(f"[{self.site.source}] fetch failed {url}: {exc}")
            return None, "error"

    # ── extraction → unified doc ─────────────────────────────────────────────────
    def _extract_main_text(self, html: str, url: str, soup: BeautifulSoup) -> str:
        try:
            import trafilatura
            text = trafilatura.extract(html, include_tables=True, include_links=False,
                                       favor_recall=True, url=url) or ""
        except Exception:
            text = ""
        if len(text.strip()) < 40:
            for sel in ["article", "main", "[class*='content']", "[class*='entry']"]:
                el = soup.select_one(sel)
                if el:
                    text = el.get_text("\n", strip=True)
                    break
            if not text.strip():
                text = soup.get_text("\n", strip=True)
        return clean_text(text)

    # ── rich metadata extraction ─────────────────────────────────────────────────
    @staticmethod
    def _first(*vals) -> str:
        for v in vals:
            if v and str(v).strip():
                return str(v).strip()
        return ""

    def _extract_rich_metadata(self, soup: BeautifulSoup, url: str) -> Dict:
        """
        Pull structured metadata from JSON-LD, OpenGraph, Schema.org microdata and
        <meta> tags. Returns a dict with any of: subtitle, authors, date,
        updated_date, category, section, tags, description, language,
        document_type, breadcrumbs. Robust to malformed JSON-LD.
        """
        import json as _json

        meta: Dict = {"authors": [], "tags": [], "breadcrumbs": []}

        def _meta(name=None, prop=None) -> str:
            if name:
                el = soup.find("meta", attrs={"name": name})
                if el and el.get("content"):
                    return el["content"].strip()
            if prop:
                el = soup.find("meta", attrs={"property": prop})
                if el and el.get("content"):
                    return el["content"].strip()
            return ""

        # ── JSON-LD (most reliable for author/date on article pages) ──────────────
        def _walk_jsonld(node):
            if isinstance(node, list):
                for n in node:
                    _walk_jsonld(n)
                return
            if not isinstance(node, dict):
                return
            if "@graph" in node:
                _walk_jsonld(node["@graph"])
            typ = node.get("@type", "")
            typs = [t.lower() for t in (typ if isinstance(typ, list) else [typ]) if t]

            # BreadcrumbList → ordered trail
            if any("breadcrumb" in t for t in typs):
                items = node.get("itemListElement") or []
                trail = []
                for it in (items if isinstance(items, list) else []):
                    nm = ""
                    if isinstance(it, dict):
                        if isinstance(it.get("item"), dict):
                            nm = it["item"].get("name", "") or it.get("name", "")
                        else:
                            nm = it.get("name", "")
                    if nm:
                        trail.append(str(nm).strip())
                if trail:
                    meta["breadcrumbs"] = trail

            if any(t in ("article", "blogposting", "newsarticle", "report",
                         "techarticle", "scholarlyarticle", "webpage", "creativework")
                   for t in typs):
                a = node.get("author")
                names = []
                if isinstance(a, dict):
                    names = [a.get("name", "")]
                elif isinstance(a, list):
                    names = [x.get("name", "") if isinstance(x, dict) else str(x) for x in a]
                elif isinstance(a, str):
                    names = [a]
                for nm in names:
                    nm = clean_text(nm)
                    if nm and nm not in meta["authors"]:
                        meta["authors"].append(nm)
                meta.setdefault("_headline", node.get("headline") or node.get("name") or "")
                meta.setdefault("_date", parse_date(str(node.get("datePublished", ""))[:10]))
                meta.setdefault("_updated", parse_date(str(node.get("dateModified", ""))[:10]))
                meta.setdefault("_section", node.get("articleSection") or "")
                meta.setdefault("_desc", node.get("description") or "")
                meta.setdefault("_lang", node.get("inLanguage") or "")
                kw = node.get("keywords")
                if isinstance(kw, str):
                    meta["tags"] += [k.strip() for k in re.split(r"[;,]", kw) if k.strip()]
                elif isinstance(kw, list):
                    meta["tags"] += [str(k).strip() for k in kw if str(k).strip()]

        for tag in soup.find_all("script", attrs={"type": "application/ld+json"}):
            raw = tag.string or tag.get_text() or ""
            if not raw.strip():
                continue
            try:
                _walk_jsonld(_json.loads(raw))
            except Exception:
                try:  # tolerate trailing commas
                    _walk_jsonld(_json.loads(re.sub(r",\s*([}\]])", r"\1", raw)))
                except Exception:
                    continue

        # ── OpenGraph / meta fallbacks ────────────────────────────────────────────
        og_author = _meta(prop="article:author") or _meta(name="author")
        if og_author and og_author not in meta["authors"] and not og_author.startswith("http"):
            meta["authors"].append(clean_text(og_author))

        meta["subtitle"]     = self._first(_meta(prop="og:description"),
                                           _meta(name="twitter:description"))
        meta["description"]  = self._first(meta.get("_desc"), _meta(name="description"),
                                           meta.get("subtitle"))
        meta["date"]         = self._first(meta.get("_date"),
                                           parse_date(_meta(prop="article:published_time")[:10]))
        meta["updated_date"] = self._first(meta.get("_updated"),
                                           parse_date(_meta(prop="article:modified_time")[:10]))
        meta["section"]      = self._first(meta.get("_section"), _meta(prop="article:section"))
        html_el = soup.find("html")
        meta["language"]     = self._first(meta.get("_lang"),
                                           (html_el.get("lang") if html_el else ""),
                                           _meta(prop="og:locale"))
        og_type = _meta(prop="og:type")
        meta["document_type"] = "article" if og_type in ("article", "blog") else (og_type or "")

        for t in soup.find_all("meta", attrs={"property": "article:tag"}):
            if t.get("content"):
                meta["tags"].append(clean_text(t["content"]))

        if not meta["breadcrumbs"]:
            bc = soup.select_one("nav[aria-label*='readcrumb'], .breadcrumb, "
                                 ".breadcrumbs, [class*='breadcrumb']")
            if bc:
                parts = [clean_text(a.get_text()) for a in bc.find_all(["a", "span", "li"])]
                meta["breadcrumbs"] = [p for p in parts if p][:8]

        meta["tags"] = list(dict.fromkeys([t for t in meta["tags"] if t]))[:20]
        meta["authors"] = list(dict.fromkeys([a for a in meta["authors"] if a]))
        for k in ("_headline", "_date", "_updated", "_section", "_desc", "_lang"):
            meta.pop(k, None)
        return meta

    def _build_doc(self, url: str, html: str) -> Optional[Dict]:
        # Navigation / archive / search / pagination pages are crawled for their
        # links but never stored as documents. Returning None here does NOT stop
        # discovery — the caller extracts outbound links separately beforehand.
        if not self._should_keep_doc(url):
            self._bump("skipped_nav")
            return None

        soup = BeautifulSoup(html, "lxml")
        text = self._extract_main_text(html, url, soup)
        if len(text) < self.site.min_text_len:
            return None

        title = ""
        og = soup.find("meta", property="og:title")
        if og:
            title = og.get("content", "")
        if not title:
            h1 = soup.find("h1")
            title = clean_text(h1.get_text()) if h1 else ""
        if not title:
            t = soup.find("title")
            title = clean_text(t.get_text()) if t else url

        # A page whose title is only a section or the site name (e.g. "Blogs",
        # "Publications", "Takshashila Institution") is an index/landing page —
        # follow it for links but don't store it as a document (website only).
        if self.site.source != "commit_kb" and is_listing_or_landing(url, title):
            self._bump("skipped_nav")
            return None

        # ── Structured metadata first (JSON-LD / OpenGraph / schema.org) ─────────
        rich = self._extract_rich_metadata(soup, url)

        author = rich["authors"][0] if rich.get("authors") else ""
        if not author:
            for sel in ["span.author", "a[rel='author']", ".post-author",
                        "[class*='author']"]:
                el = soup.select_one(sel)
                if el:
                    author = clean_text(el.get_text())
                    break
        if not author:
            m = soup.find("meta", {"name": "author"})
            author = m.get("content", "") if m else ""

        date = rich.get("date", "")
        if not date:
            for sel in ["time", ".entry-date", ".post-date", ".published"]:
                el = soup.select_one(sel)
                if el:
                    date = parse_date(el.get("datetime", "") or el.get_text())
                    break

        tags = list(rich.get("tags", []))
        tag_els = soup.select("a[rel='tag'], .tags a, .post-tags a")
        for t in tag_els:
            tt = clean_text(t.get_text())
            if tt and tt not in tags:
                tags.append(tt)

        category = rich.get("section", "")
        if not category:
            cat_el = soup.select_one("[class*='category']")
            if cat_el:
                category = clean_text(cat_el.get_text())

        # ── Metadata enrichment fallbacks (URL/text based) ──────────────────────
        try:
            from src.extractors import (
                extract_date_from_url, extract_date_from_text,
                extract_author_from_text, extract_category_from_url,
            )
            if not date:
                date = extract_date_from_url(url) or extract_date_from_text(text)
            if not author:
                author = extract_author_from_text(text)
            if not category:
                category = extract_category_from_url(url)
        except Exception:
            pass

        authors = rich.get("authors", []) or ([author] if author else [])
        section = rich.get("section", "") or category
        breadcrumbs = rich.get("breadcrumbs", [])
        subtitle = rich.get("subtitle", "")
        updated_date = rich.get("updated_date", "")
        language = rich.get("language", "")
        document_type = rich.get("document_type", "")

        canonical = url
        canon = soup.find("link", rel="canonical")
        if canon and canon.get("href"):
            canonical = normalize_url(canon["href"], url)

        pdf_urls = []
        for a in soup.find_all("a", href=True):
            href = normalize_url(a["href"], url)
            # Audit finding: this used to call is_same_domain(href, self.site.domain),
            # which does a raw substring check ("takshashila.org.in" in netloc) —
            # that incorrectly treats ANY hostname containing the domain string as
            # "same domain", including legion.takshashila.org.in/school.takshashila.org.in
            # even when WEBSITE_ADDITIONAL_DOMAINS is deliberately left empty, and even
            # a crafted hostname like "takshashila.org.in.evil.example". Use the same
            # exact-match _is_internal() check the rest of the crawler already uses,
            # so PDF discovery respects the same domain scope as page discovery.
            if looks_like_pdf(href) and self._is_internal(href):
                pdf_urls.append(href)
        pdf_urls = sorted(set(pdf_urls))

        uid = url_hash(url)
        source_type = _infer_source_type(url, self.site.source)
        doc = {
            "document_id":       f"{self.site.source}_{uid}",
            "url_hash":          uid,
            "page_id":           uid,
            "original_url":      url,
            "url":               url,
            "canonical_url":     canonical,
            "title":             title or url,
            "subtitle":          subtitle,
            "author":            author,
            "authors":           authors,
            "date":              date,
            "updated_date":      updated_date,
            "category":          category or source_type,
            "section":           section,
            "tags":              tags,
            "breadcrumbs":       breadcrumbs,
            "language":          language,
            "document_type":     document_type or source_type,
            "source":            self.site.source,
            "source_name":       self.site.source_name,
            "source_type":       source_type,
            "text":              text,
            "text_length":       len(text),
            "content_hash":      content_hash(text),
            "pdf_urls":          pdf_urls,
            "pdf_url":           pdf_urls[0] if pdf_urls else "",
            "discovery_method":  "crawl",
            "extraction_method": "trafilatura",
            "scraped_at":        now_iso(),
            "updated_at":        now_iso(),
        }
        return clean_document_metadata(doc)

    # ── PDF handling ─────────────────────────────────────────────────────────────
    def _process_pdf(self, pdf_url: str, parent: Dict) -> Optional[Dict]:
        uid = url_hash(pdf_url)
        # Skip re-download if content is unchanged (by our stored hash of extracted text).
        dest = config.RAW_PDF_DIR / f"{uid}.pdf"
        try:
            config.RAW_PDF_DIR.mkdir(parents=True, exist_ok=True)
            r = self.session.get(pdf_url, timeout=config.SCRAPE_TIMEOUT,
                                 headers=self.state.conditional_headers(pdf_url) if self.incremental else {})
            if r.status_code == 304 and dest.exists():
                # unchanged; nothing to do (already in the KB from a prior run)
                self.state.mark_seen(pdf_url)
                return None
            if r.status_code >= 400:
                return None
            dest.write_bytes(r.content)
            time.sleep(config.SCRAPE_DELAY)
        except Exception as exc:
            logger.warning(f"PDF download failed {pdf_url}: {exc}")
            return None

        from src.extractors import extract_pdf_text
        pages = extract_pdf_text(dest)
        text = "\n\n".join(p["text"] for p in pages)
        chash = content_hash(text)
        if self.incremental and self.state.is_unchanged(pdf_url, chash):
            self.state.mark_seen(pdf_url)
            self._bump("unchanged")
            return None

        doc_id = f"{self.site.source}_pdf_{uid}"
        doc = {
            "document_id":       doc_id,
            "url_hash":          uid,
            "original_url":      parent.get("original_url", pdf_url),
            "url":               pdf_url,
            "canonical_url":     pdf_url,
            "pdf_url":           pdf_url,
            "title":             (parent.get("title", "") or "Document") + " [PDF]",
            "author":            parent.get("author", ""),
            "date":              parent.get("date", ""),
            "category":          parent.get("category", "") or "pdf",
            "tags":              parent.get("tags", []),
            "source":            self.site.source,
            "source_name":       self.site.source_name,
            "source_type":       "pdf",
            "text":              text,
            "text_length":       len(text),
            "pdf_pages":         pages,
            "content_hash":      chash,
            "local_pdf_path":    str(dest),
            "extraction_method": "pymupdf",
            "scraped_at":        now_iso(),
            "updated_at":        now_iso(),
        }
        et = ""  # ETag captured on the HTML fetch path; PDFs keyed by text hash
        self.state.record(pdf_url, content_hash=chash, document_id=doc_id,
                          etag=et, changed=True)
        self._bump("pdf")
        return clean_document_metadata(doc)

    # ── per-page processing ──────────────────────────────────────────────────────
    def _process(self, url: str) -> Tuple[Optional[Dict], List[str], List[Dict]]:
        """Return (doc_or_None, discovered_internal_links, extra_pdf_docs)."""
        if not self._can_fetch(url):
            self.state.mark_seen(url)   # blocked, but not removed
            return None, [], []

        resp, status = self._fetch(url)

        if status == "not_modified":
            self.state.mark_seen(url)   # exists, unchanged per server
            self._bump("not_modified")
            self._bump("unchanged")
            return None, [], []
        if status == "gone":
            # 404/410 → a genuine removal; do NOT mark seen so it's detected below.
            return None, [], []
        if status == "error" or resp is None:
            # Transient/other error: be conservative, keep the page (mark seen).
            self.state.mark_seen(url)
            self._bump("failed")
            return None, [], []
        if "text/html" not in resp.headers.get("Content-Type", ""):
            self.state.mark_seen(url)
            return None, [], []

        self.state.mark_seen(url)
        html = resp.text
        etag = resp.headers.get("ETag", "")
        last_mod = resp.headers.get("Last-Modified", "")

        # Discover internal links regardless of change status.
        links = self._links_from(html, url)
        # Any PDF linked from a page we can still reach is NOT removed, even if the
        # page itself is unchanged — mark those PDFs as seen so removal detection
        # (HTML-only) never touches them.
        for link in links:
            if looks_like_pdf(link):
                self.state.mark_seen(link)

        doc = self._build_doc(url, html)
        if doc is None:
            return None, links, []

        if self.incremental and self.state.is_unchanged(url, doc["content_hash"]):
            self._bump("unchanged")
            # refresh etag/last-modified without marking a content change
            self.state.record(url, content_hash=doc["content_hash"],
                              document_id=doc["document_id"], etag=etag,
                              last_modified=last_mod, changed=False)
            return None, links, []

        changed = self.state.known(url)
        self._bump("updated" if changed else "added")
        self.state.record(url, content_hash=doc["content_hash"],
                          document_id=doc["document_id"], etag=etag,
                          last_modified=last_mod, changed=True)

        # Cache raw HTML for audit/debug (cheap, matches existing behaviour).
        try:
            config.RAW_HTML_DIR.mkdir(parents=True, exist_ok=True)
            (config.RAW_HTML_DIR / f"{url_hash(url)}.html").write_text(html, encoding="utf-8")
        except Exception:
            pass

        # PDFs linked from this page.
        pdf_docs: List[Dict] = []
        if self.site.include_pdfs:
            for pdf_url in doc.get("pdf_urls", []):
                pd = self._process_pdf(pdf_url, doc)
                if pd:
                    pdf_docs.append(pd)

        return doc, links, pdf_docs

    def _links_from(self, html: str, base: str) -> List[str]:
        soup = BeautifulSoup(html, "lxml")
        out = []
        for a in soup.find_all("a", href=True):
            norm = normalize_url(a["href"], base)
            if norm and self._is_internal(norm):
                out.append(norm)
        return sorted(set(out))

    # ── main crawl loop ──────────────────────────────────────────────────────────
    def crawl(self) -> CrawlResult:
        self.state.begin_run()

        # Seed: base URL + sitemap URLs + listing URLs.
        seeds = [self.site.base_url]
        seeds += [u for u, _ in self.site.listing_urls]
        seeds += self._discover_sitemap()
        # Always re-visit every URL we already know about, so that (a) discovery
        # doesn't depend on re-parsing unchanged pages' links, (b) removal
        # detection is accurate on incremental runs (a page we can't reach this
        # run is genuinely gone, not merely unlinked from a page we skipped), and
        # (c) a full re-crawl can never lose a page that the sitemap forgot to
        # list — every previously known page is re-checked directly.
        for u in self.state.urls:
            if self._extension(u) in ("", ".html", ".htm"):
                seeds.append(u)
        for s in seeds:
            norm = normalize_url(s, self.site.base_url)
            key = canonicalize_for_dedup(norm) if norm else ""
            if norm and key not in self.seen and self._is_internal(norm):
                self.seen.add(key)
                self.depth[norm] = 0
                self.frontier.append(norm)

        self.result.discovered = len(self.frontier)
        self._log(f"[{self.site.source}] seeded {len(self.frontier)} URLs; crawling…")

        processed = 0
        workers = max(2, int(getattr(self.site, "max_workers", 8) or 8))
        with ThreadPoolExecutor(max_workers=workers) as ex:
            while self.frontier and processed < self.site.max_pages:
                batch = []
                while self.frontier and len(batch) < workers and processed + len(batch) < self.site.max_pages:
                    batch.append(self.frontier.popleft())

                futures = {ex.submit(self._process, u): u for u in batch}
                for fut in as_completed(futures):
                    url = futures[fut]
                    processed += 1
                    try:
                        doc, links, pdf_docs = fut.result()
                    except Exception as exc:
                        logger.error(f"[{self.site.source}] error on {url}: {exc}")
                        self._bump("failed")
                        continue

                    if doc:
                        self.result.docs.append(doc)
                    self.result.docs.extend(pdf_docs)

                    # enqueue newly discovered pages within depth budget. We
                    # follow ALL internal links (minus follow-excludes/assets)
                    # so every page and subpage is reached; whether each becomes
                    # a document is decided later by _should_keep_doc.
                    d = self.depth.get(url, 0)
                    if d < self.site.max_depth:
                        for link in links:
                            link_key = canonicalize_for_dedup(link)
                            if link_key in self.seen:
                                continue
                            if not self._should_follow(link):
                                continue
                            self.seen.add(link_key)
                            self.depth[link] = d + 1
                            self.frontier.append(link)

                    if processed % 25 == 0:
                        self._log(f"[{self.site.source}] processed {processed} pages "
                                  f"(+{self._counts['added']} new, ~{self._counts['updated']} changed, "
                                  f"={self._counts['unchanged']} unchanged)")
                self.state.save()  # checkpoint — safe to interrupt/resume

        # Removal detection: URLs known before but not seen this run.
        # Only trust this when we actually completed discovery (didn't hit the cap),
        # and never auto-remove PDFs (their parent page may simply be unchanged).
        removed_ids: List[str] = []
        if self.incremental and processed < self.site.max_pages:
            unseen = self.state.unseen_urls()
            unseen = [u for u in unseen if self._extension(u) not in (".pdf",)]
            candidate_ids = self.state.document_ids_for(unseen)
            removed_ids = [d for d in candidate_ids if "_pdf_" not in d]
            for u in unseen:
                did = self.state.document_id(u)
                if did and "_pdf_" not in did:
                    self.state.remove(u)
        self.result.removed_ids = removed_ids
        self._counts["removed"] = len(removed_ids)

        self.state.save()
        self.result.counts = dict(self._counts)
        logger.info(f"[{self.site.source}] crawl done: {self.result.counts}")
        self._log(f"[{self.site.source}] ✓ crawl done — "
                  f"{self._counts['added']} new, {self._counts['updated']} changed, "
                  f"{self._counts['unchanged']} unchanged, {self._counts['removed']} removed, "
                  f"{self._counts['failed']} failed")
        return self.result


def _infer_source_type(url: str, source: str) -> str:
    low = url.lower()
    for key in ("publication", "blog", "research", "commentary", "report",
                "paper", "brief", "policy", "playbook", "decision"):
        if f"/{key}" in low or f"{key}s/" in low:
            return key
    return "page" if source == "website" else "kb"


def crawl_site(site: SiteConfig, incremental: bool = True,
               progress_cb: Optional[Callable] = None) -> CrawlResult:
    """Convenience wrapper: build state + engine and run one crawl."""
    state = CrawlState(site.source)
    engine = CrawlEngine(site, state, incremental=incremental, progress_cb=progress_cb)
    return engine.crawl()