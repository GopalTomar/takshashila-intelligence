"""
tests/test_dashboard_safety_helpers.py — Unit tests for app.py's HTML/Markdown
safety helpers (_esc, _safe_href, _is_http_url, _md_escape) and
src.utils.linkify_citations' scheme validation, isolated from the rest of the
Streamlit script so they run instantly with no AppTest overhead.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
import importlib.util

# app.py isn't a normal importable module (it runs Streamlit calls at import
# time), so pull just the plain functions we need via spec loading would still
# execute the whole script. Instead, exec only the small helper block by
# importing app.py under AppTest's bare-execution context is overkill for pure
# string functions — so these are re-imported directly from the live app.py
# module object AppTest already loads in test_dashboard.py's process. To keep
# this file independent and fast, we duplicate-free import via runpy is also
# heavy; simplest reliable approach: load app.py as a module once (Streamlit
# calls like st.set_page_config are idempotent-safe / no-ops outside a session
# in bare mode) and reuse its functions.
import streamlit as st  # noqa: F401  (ensures streamlit is initialized first)


def _load_app_module():
    spec = importlib.util.spec_from_file_location(
        "app_under_test", str(Path(__file__).parent.parent / "app.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_app = _load_app_module()


def test_esc_neutralizes_html():
    assert _app._esc("<img src=x onerror=alert(1)>") == \
        "&lt;img src=x onerror=alert(1)&gt;"
    assert _app._esc("Tom & Jerry") == "Tom &amp; Jerry"
    assert _app._esc(None) == ""


def test_safe_href_only_allows_http_https():
    assert _app._safe_href("https://example.com/x") == "https://example.com/x"
    assert _app._safe_href("http://example.com") == "http://example.com"
    assert _app._safe_href("javascript:alert(1)") == ""
    assert _app._safe_href("data:text/html,<script>alert(1)</script>") == ""
    assert _app._safe_href("") == ""
    assert _app._safe_href(None) == ""


def test_safe_href_escapes_the_url_itself():
    # A URL containing a quote must not be able to break out of href='...'
    dangerous = "https://example.com/x'onmouseover='alert(1)"
    out = _app._safe_href(dangerous)
    assert "'" not in out or "&#x27;" in out or "&#39;" in out


def test_is_http_url():
    assert _app._is_http_url("https://x.com") is True
    assert _app._is_http_url("http://x.com") is True
    assert _app._is_http_url("javascript:alert(1)") is False
    assert _app._is_http_url("") is False
    assert _app._is_http_url(None) is False


def test_md_escape_neutralizes_markdown_link_syntax():
    dangerous_title = "Report Title](https://evil.example)[Ignore this"
    escaped = _app._md_escape(dangerous_title)
    # The brackets that would otherwise form "](url)[" link syntax must be
    # backslash-escaped, so a markdown renderer treats them as literal
    # characters rather than link delimiters.
    assert "\\]" in escaped
    assert "\\[" in escaped
    assert "evil.example" in escaped  # content preserved, just neutralized


def test_md_escape_handles_none_and_plain_text():
    assert _app._md_escape(None) == ""
    assert _app._md_escape("Ordinary Title") == "Ordinary Title"


def test_linkify_citations_rejects_unsafe_schemes():
    from src.utils import linkify_citations
    sources = [{"url": "javascript:alert(1)"}]
    out = linkify_citations("See the finding [Source 1].", sources)
    # Must NOT produce a markdown link with the javascript: URL.
    assert "javascript:" not in out
    assert "[Source 1]" in out  # left as a bare, non-linked marker


def test_linkify_citations_accepts_http_and_links_correctly():
    from src.utils import linkify_citations
    sources = [{"url": "https://takshashila.org.in/report"}]
    out = linkify_citations("See the finding [Source 1].", sources)
    assert "[[1]](https://takshashila.org.in/report)" in out
