"""
tests/test_source_quality.py — Grounded, relevant sources + clickable citations.

Two behaviours:

1. **Only real evidence can be cited.** Author/profile pages, tag & category
   listings, paginated archives and thin nav pages are dropped before the context
   is built, so the model can never cite something like
   "Pranay Kotasthane – Takshashila Institution" as a source for a claim.
2. **Inline [N] markers become links** to the exact document they point at, so a
   reader can click through to the passage the claim came from.

Run:  python tests/test_source_quality.py   ·   pytest tests/test_source_quality.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import src.rag_pipeline as rp                     # noqa: E402
from src.utils import linkify_citations            # noqa: E402


LONG = "x" * 300          # enough text to count as evidence


# ── 1. Source quality filter ─────────────────────────────────────────────────────

def test_author_page_is_low_value():
    ch = {"title": "Pranay Kotasthane \u2013 Takshashila Institution",
          "url": "https://takshashila.org.in/author/pranay-kotasthane",
          "text": LONG, "score": 0.66}
    assert rp._is_low_value_source(ch) is True

def test_tag_category_and_archive_pages_are_low_value():
    for url in ["https://t.org.in/tag/china", "https://t.org.in/category/policy",
                "https://t.org.in/topics/space", "https://t.org.in/archive/2024",
                "https://t.org.in/page/3"]:
        ch = {"title": "Listing", "url": url, "text": LONG, "score": 0.7}
        assert rp._is_low_value_source(ch) is True, url

def test_thin_pages_are_low_value():
    ch = {"title": "Nav", "url": "https://t.org.in/x", "text": "too short", "score": 0.95}
    assert rp._is_low_value_source(ch) is True

def test_real_documents_are_kept():
    for title, url in [
        ("Mediation and Interference [PDF]", "https://t.org.in/content/publications/med.pdf"),
        ("India Must Look Beyond Main Battle Tanks", "https://t.org.in/content/publications/tanks"),
    ]:
        ch = {"title": title, "url": url, "text": LONG, "score": 0.66}
        assert rp._is_low_value_source(ch) is False, title

def test_filter_drops_author_page_and_keeps_documents():
    chunks = [
        {"title": "Pranay Kotasthane \u2013 Takshashila Institution",
         "url": "https://t.org.in/author/pranay", "text": LONG, "score": 0.70},
        {"title": "Mediation and Interference [PDF]",
         "url": "https://t.org.in/pub/med.pdf", "text": LONG, "score": 0.66},
    ]
    kept = rp._filter_evidence_chunks(chunks)
    titles = [c["title"] for c in kept]
    assert "Mediation and Interference [PDF]" in titles
    assert not any("Kotasthane" in t for t in titles)

def test_below_relevance_floor_is_dropped():
    from src import config
    floor = float(config.MIN_SCORE_THRESHOLD)
    chunks = [
        {"title": "Good", "url": "https://t.org.in/pub/a", "text": LONG, "score": floor + 0.2},
        {"title": "Weak", "url": "https://t.org.in/pub/b", "text": LONG, "score": floor - 0.2},
    ]
    kept = rp._filter_evidence_chunks(chunks)
    assert [c["title"] for c in kept] == ["Good"]

def test_filter_never_returns_empty_when_input_nonempty():
    # every chunk is low-value → fall back to originals so a weak answer is still
    # possible (the grounding check downstream decides whether to answer).
    chunks = [{"title": "Nav", "url": "https://t.org.in/tag/x", "text": "s", "score": 0.9}]
    assert rp._filter_evidence_chunks(chunks) == chunks


# ── 2. Clickable citations ───────────────────────────────────────────────────────

def test_markers_become_links():
    sources = [{"url": "https://t.org.in/a"}, {"url": "https://t.org.in/b"}]
    out = linkify_citations("Prioritized [Source 1]. China mediated [2].", sources)
    assert "[[1]](https://t.org.in/a)" in out
    assert "[[2]](https://t.org.in/b)" in out

def test_marker_without_url_is_left_alone():
    out = linkify_citations("No link here [1].", [{"title": "x"}])
    assert out == "No link here [1]."

def test_unknown_marker_is_left_alone():
    out = linkify_citations("Unknown [7].", [{"url": "https://t.org.in/a"}])
    assert out == "Unknown [7]."

def test_linkify_is_idempotent():
    sources = [{"url": "https://t.org.in/a"}]
    once = linkify_citations("Claim [1].", sources)
    assert linkify_citations(once, sources) == once      # no double-wrapping

def test_bot_answer_contains_clickable_citations():
    from integrations import formatting
    result = {
        "answer": "**Answer**\nClaim [Source 1].\n\n**Details**\n- Point [Source 1].",
        "confidence": "high", "retrieval_time": 0.4, "generation_time": 1.0,
        "sources": [{"title": "Mediation and Interference [PDF]",
                     "url": "https://t.org.in/pub/med.pdf",
                     "category": "south asia", "source": "website", "text": "t"}],
    }
    msg = formatting.format_answer("Q?", result, mode="normal")
    assert "[[1]](https://t.org.in/pub/med.pdf)" in msg


if __name__ == "__main__":
    mod = sys.modules[__name__]
    passed = 0
    for name in sorted(dir(mod)):
        if name.startswith("test_"):
            getattr(mod, name)()
            passed += 1
    print(f"OK — {passed} tests passed")