"""
tests/test_dashboard_helpers.py — Unit tests for app.py's pure security/
formatting helpers (_esc, _safe_href, _friendly_pipeline_error).

app.py is a Streamlit script that executes top-to-bottom against a live
ScriptRunContext (st.status(), st.sidebar, st.tabs(), etc.) the moment it's
imported — a bare `import app` fails partway through outside that context,
and a full `streamlit.testing.v1.AppTest` run of the whole 1600-line script
is fragile to depend on for three self-contained pure functions with zero
Streamlit calls in their bodies.

So this file extracts exactly those three function definitions from the
current app.py source via `ast` (never a hand-retyped copy that could drift
from the real implementation) and executes them in an isolated namespace.
This is what actually runs in production — not a reimplementation of it.

A full-script smoke test (does app.py run at all, in what state) lives in
tests/test_dashboard_smoke.py using AppTest, which CAN load the whole app
successfully in bare mode (verified interactively) for the top-level script
run — this file exists specifically because per-function extraction gives
faster, more precise, and more robust assertions for these three functions
than trying to reach them through six tabs' worth of rendered UI state.
"""

import ast
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

APP_PY = Path(__file__).parent.parent / "app.py"
_TARGET_FUNCS = {"_esc", "_safe_href", "_friendly_pipeline_error"}


def _load_helpers():
    """Parse app.py, extract only the named top-level function defs, and exec
    them in a minimal namespace (just `html` — what they actually import)."""
    source = APP_PY.read_text(encoding="utf-8")
    tree = ast.parse(source)
    ns = {"__name__": "app_helpers_under_test"}
    exec("import html as _html_mod", ns)
    found = set()
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name in _TARGET_FUNCS:
            segment = ast.get_source_segment(source, node)
            exec(compile(segment, filename=str(APP_PY), mode="exec"), ns)
            found.add(node.name)
    missing = _TARGET_FUNCS - found
    assert not missing, (
        f"Expected function(s) {missing} not found in app.py — this test file "
        f"is out of sync with the app, or the functions were renamed/removed."
    )
    return ns


_ns = _load_helpers()
_esc = _ns["_esc"]
_safe_href = _ns["_safe_href"]
_friendly_pipeline_error = _ns["_friendly_pipeline_error"]


# ── _esc ──────────────────────────────────────────────────────────────────

def test_esc_escapes_script_tags():
    out = _esc("<script>alert(1)</script>")
    assert "<script>" not in out
    assert "&lt;script&gt;" in out


def test_esc_escapes_event_handler_injection():
    out = _esc('<img src=x onerror=alert(2)>')
    assert "<img" not in out
    assert "&lt;img" in out


def test_esc_escapes_quotes_for_attribute_context():
    out = _esc('Report " onmouseover="alert(1)')
    assert '"' not in out  # would break out of a href='...' attribute otherwise
    assert "&quot;" in out


def test_esc_handles_none_and_non_string():
    assert _esc(None) == ""
    assert _esc(42) == "42"


def test_esc_preserves_legitimate_unicode():
    """Regression guard: a prior implementation elsewhere in the dashboard
    replaced any non-ASCII/non-Devanagari character with a literal quote,
    mangling legitimate international text. Proper escaping must NOT do
    that — accented Latin, CJK, and Arabic script must survive unchanged
    (only HTML-special characters get transformed)."""
    for text in ("café", "François Hollande", "北京大学", "مرحبا", "— “curly quotes”"):
        assert _esc(text) == text  # none of these contain & < > " ' so escape is a no-op


# ── _safe_href ───────────────────────────────────────────────────────────

def test_safe_href_allows_https():
    assert _safe_href("https://takshashila.org.in/blogs/x") == "https://takshashila.org.in/blogs/x"


def test_safe_href_allows_http():
    assert _safe_href("http://example.com") == "http://example.com"


def test_safe_href_blocks_javascript_scheme():
    assert _safe_href("javascript:alert(1)") == ""


def test_safe_href_blocks_data_scheme():
    assert _safe_href("data:text/html,<script>alert(1)</script>") == ""


def test_safe_href_blocks_empty_and_none():
    assert _safe_href("") == ""
    assert _safe_href(None) == ""


def test_safe_href_escapes_quotes_in_otherwise_valid_url():
    """A URL containing a stray quote must not break out of href='...'."""
    out = _safe_href("https://example.com/x\"onmouseover=\"alert(1)")
    assert '"' not in out


# ── _friendly_pipeline_error ─────────────────────────────────────────────

def test_friendly_error_never_leaks_raw_exception_text():
    secret_path = "/home/produser/secret_internal_path/faiss.index"
    exc = FileNotFoundError(f"{secret_path} not found")
    friendly = _friendly_pipeline_error(exc)
    assert secret_path not in friendly
    assert "faiss" not in friendly.lower()


def test_friendly_error_maps_missing_index():
    friendly = _friendly_pipeline_error(FileNotFoundError("index not found at /x/faiss.index"))
    assert "index" in friendly.lower()
    assert "build" in friendly.lower() or "administrator" in friendly.lower()


def test_friendly_error_maps_missing_api_key():
    friendly = _friendly_pipeline_error(ValueError("GROQ_API_KEY is not set. Add it to .env"))
    assert "administrator" in friendly.lower() or "configur" in friendly.lower()
    assert "GROQ_API_KEY" not in friendly


def test_friendly_error_maps_rate_limit():
    friendly = _friendly_pipeline_error(Exception("Error code: 429 - rate limit exceeded"))
    assert "rate" in friendly.lower() or "try again" in friendly.lower()


def test_friendly_error_generic_fallback_is_calm_and_short():
    friendly = _friendly_pipeline_error(RuntimeError("some totally unexpected internal state"))
    assert "totally unexpected internal state" not in friendly
    assert len(friendly) < 200
