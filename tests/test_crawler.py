"""
tests/test_crawler.py — Crawl-engine unit tests (no live network).

These exercise the pure/offline logic in scripts/crawl_engine.py: URL scoping
(internal-domain checks, follow vs. keep tiers), URL canonicalization/dedup,
and metadata extraction. Anything that requires hitting the real network
(sitemap fetch, robots.txt fetch, page fetch) is deliberately NOT covered here
— those need a live or mocked HTTP layer and are out of scope for this
sandbox (see chat notes: no network access to takshashila.org.in from here).
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from bs4 import BeautifulSoup

from scripts.crawl_engine import SiteConfig, CrawlEngine
from src.crawl_state import CrawlState
from src.utils import canonicalize_for_dedup


def _engine(tmp_path, **overrides) -> CrawlEngine:
    """Build a CrawlEngine with no network calls (respect_robots=False skips
    the robots.txt fetch that _load_robots() would otherwise make)."""
    site_kwargs = dict(
        base_url="https://takshashila.org.in/",
        domain="takshashila.org.in",
        source="website",
        source_name="Takshashila Website",
        follow_exclude_patterns=["/wp-admin/", "mailto:", "tel:", "javascript:", "#"],
        doc_exclude_patterns=["/tag/", "/category/", "/author/", "/search", "/page/"],
        respect_robots=False,
    )
    site_kwargs.update(overrides)
    site = SiteConfig(**site_kwargs)
    # CrawlState always resolves its file under config.LOGS_DIR; these tests
    # never call .save(), so using a unique per-test source name is enough to
    # avoid clashing with any real state file (nothing is written to disk).
    state = CrawlState(f"test_{site.source}_{id(tmp_path)}")
    return CrawlEngine(site, state, incremental=False)


# ── URL canonicalization / dedup ────────────────────────────────────────────

def test_canonicalize_strips_fragment():
    assert (canonicalize_for_dedup("https://takshashila.org.in/blog/post#section2")
            == canonicalize_for_dedup("https://takshashila.org.in/blog/post"))


def test_canonicalize_strips_tracking_params():
    a = canonicalize_for_dedup("https://takshashila.org.in/blog/post?utm_source=twitter&utm_medium=social")
    b = canonicalize_for_dedup("https://takshashila.org.in/blog/post")
    assert a == b


def test_canonicalize_ignores_param_order():
    a = canonicalize_for_dedup("https://takshashila.org.in/?a=1&b=2")
    b = canonicalize_for_dedup("https://takshashila.org.in/?b=2&a=1")
    assert a == b


def test_canonicalize_collapses_www_and_trailing_slash():
    a = canonicalize_for_dedup("https://www.takshashila.org.in/blog/post/")
    b = canonicalize_for_dedup("https://takshashila.org.in/blog/post")
    assert a == b


def test_canonicalize_keeps_real_query_params():
    """A genuine pagination/filter param must NOT be treated as noise."""
    a = canonicalize_for_dedup("https://takshashila.org.in/blogs/?page=2")
    b = canonicalize_for_dedup("https://takshashila.org.in/blogs/?page=3")
    assert a != b


# ── Internal-domain scoping (multi-domain support) ──────────────────────────

def test_is_internal_primary_domain(tmp_path):
    eng = _engine(tmp_path)
    assert eng._is_internal("https://takshashila.org.in/blogs/some-post")
    assert not eng._is_internal("https://example.com/anything")


def test_is_internal_respects_www(tmp_path):
    eng = _engine(tmp_path)
    assert eng._is_internal("https://www.takshashila.org.in/blogs/some-post")


def test_is_internal_rejects_unlisted_subdomain_by_default(tmp_path):
    """Regression guard for the audit finding: legion.takshashila.org.in and
    school.takshashila.org.in carry real content but must NOT be silently
    treated as internal unless explicitly configured."""
    eng = _engine(tmp_path)
    assert not eng._is_internal("https://legion.takshashila.org.in/team")
    assert not eng._is_internal("https://school.takshashila.org.in/")


def test_is_internal_allows_configured_additional_domain(tmp_path):
    eng = _engine(tmp_path, additional_domains=("legion.takshashila.org.in",))
    assert eng._is_internal("https://legion.takshashila.org.in/team")
    assert not eng._is_internal("https://school.takshashila.org.in/")  # still not opted in


# ── Follow tier vs. keep tier ────────────────────────────────────────────────

def test_should_follow_excludes_admin_and_fragments(tmp_path):
    eng = _engine(tmp_path)
    assert not eng._should_follow("https://takshashila.org.in/wp-admin/edit.php")
    assert not eng._should_follow("https://takshashila.org.in/page#top")
    assert not eng._should_follow("mailto:someone@takshashila.org.in")


def test_should_follow_allows_category_and_pagination_pages(tmp_path):
    """Category/tag/pagination pages must still be FOLLOWED (for discovery)
    even though they are never KEPT as documents — otherwise content only
    linked from a category page would never be discovered."""
    eng = _engine(tmp_path)
    assert eng._should_follow("https://takshashila.org.in/category/ai/")
    assert eng._should_follow("https://takshashila.org.in/blogs/page/3/")


def test_should_keep_doc_excludes_listing_pages(tmp_path):
    eng = _engine(tmp_path)
    assert not eng._should_keep_doc("https://takshashila.org.in/category/ai/")
    assert not eng._should_keep_doc("https://takshashila.org.in/blogs/page/3/")
    assert not eng._should_keep_doc("https://takshashila.org.in/pages/blogs/")


def test_should_keep_doc_allows_real_article(tmp_path):
    eng = _engine(tmp_path)
    assert eng._should_keep_doc("https://takshashila.org.in/content/publications/20251024-example.html")


# ── Rich metadata extraction: regression test for the missing `re` import ──

def test_extract_rich_metadata_handles_string_keywords(tmp_path):
    """
    Before the fix, `_extract_rich_metadata` used `re.split`/`re.sub` without
    `import re` in scripts/crawl_engine.py. Any JSON-LD `keywords` field given
    as a comma-separated STRING (extremely common — Yoast SEO on WordPress
    emits this) raised NameError, silently swallowed by the surrounding
    try/except, discarding author/date/section/tags extracted from that same
    JSON-LD node (and any later @graph nodes in the same script tag).
    """
    html = """
    <html lang="en"><head>
    <script type="application/ld+json">
    {"@type": "Article", "headline": "Test Piece",
     "author": {"@type": "Person", "name": "Jane Researcher"},
     "datePublished": "2025-06-01", "dateModified": "2025-06-02",
     "articleSection": "High-Tech Geopolitics",
     "keywords": "AI, semiconductors, export controls"}
    </script>
    </head><body><h1>Test Piece</h1></body></html>
    """
    eng = CrawlEngine.__new__(CrawlEngine)  # skip __init__ (no network needed)
    soup = BeautifulSoup(html, "lxml")
    meta = eng._extract_rich_metadata(soup, "https://takshashila.org.in/blogs/test-piece")
    assert meta["authors"] == ["Jane Researcher"]
    assert meta["date"] == "2025-06-01"
    assert meta["section"] == "High-Tech Geopolitics"
    assert set(meta["tags"]) == {"AI", "semiconductors", "export controls"}


def test_extract_rich_metadata_tolerates_malformed_jsonld(tmp_path):
    """The trailing-comma repair path (re.sub) must actually work now."""
    html = """
    <html><head>
    <script type="application/ld+json">
    {"@type": "Article", "headline": "Trailing Comma Test",
     "author": {"@type": "Person", "name": "A. Author",},
     "datePublished": "2025-01-01",}
    </script>
    </head><body></body></html>
    """
    eng = CrawlEngine.__new__(CrawlEngine)
    soup = BeautifulSoup(html, "lxml")
    meta = eng._extract_rich_metadata(soup, "https://takshashila.org.in/blogs/x")
    assert meta["authors"] == ["A. Author"]
    assert meta["date"] == "2025-01-01"


def test_pdf_discovery_respects_domain_scope_not_substring_match(tmp_path):
    """
    Regression test: _build_doc()'s PDF-link discovery used to call
    is_same_domain(href, domain), a raw substring check ("takshashila.org.in"
    in netloc) — which would incorrectly treat legion.takshashila.org.in (and
    even a crafted "takshashila.org.in.evil.example") as the same domain as
    takshashila.org.in, even with WEBSITE_ADDITIONAL_DOMAINS left empty. It
    now uses the same exact-match _is_internal() the rest of the crawler uses.
    """
    eng = _engine(tmp_path)  # additional_domains defaults to () — not configured
    html = """
    <html><body>
      <h1>Test Publication</h1>
      <p>Some real body text long enough to pass the minimum length filter for
      this test page, repeated so it clears the threshold reliably. """ + ("Filler text. " * 20) + """</p>
      <a href="https://takshashila.org.in/content/publications/report.pdf">Same domain PDF</a>
      <a href="https://legion.takshashila.org.in/report.pdf">Legion subdomain PDF</a>
      <a href="https://school.takshashila.org.in/report.pdf">School subdomain PDF</a>
      <a href="https://takshashila.org.in.evil.example/report.pdf">Crafted lookalike hostname PDF</a>
    </body></html>
    """
    doc = eng._build_doc("https://takshashila.org.in/content/publications/test-report.html", html)
    assert doc is not None
    assert doc["pdf_urls"] == ["https://takshashila.org.in/content/publications/report.pdf"]


def test_pdf_discovery_includes_additional_domain_when_configured(tmp_path):
    """When legion.takshashila.org.in IS explicitly opted in, its PDFs are
    correctly discovered — the fix must not have broken the opt-in path."""
    eng = _engine(tmp_path, additional_domains=("legion.takshashila.org.in",))
    html = """
    <html><body>
      <h1>Test Publication</h1>
      <p>""" + ("Filler text. " * 20) + """</p>
      <a href="https://takshashila.org.in/content/publications/report.pdf">Same domain PDF</a>
      <a href="https://legion.takshashila.org.in/report.pdf">Legion subdomain PDF</a>
      <a href="https://school.takshashila.org.in/report.pdf">School subdomain PDF (still not opted in)</a>
    </body></html>
    """
    doc = eng._build_doc("https://takshashila.org.in/content/publications/test-report.html", html)
    assert doc is not None
    assert set(doc["pdf_urls"]) == {
        "https://takshashila.org.in/content/publications/report.pdf",
        "https://legion.takshashila.org.in/report.pdf",
    }
