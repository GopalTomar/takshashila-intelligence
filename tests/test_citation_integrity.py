"""
tests/test_citation_integrity.py — Sources shown == sources the answer cites.

Reproduces the reported bug: the same question showed 3, then 2, then 1 source
while the answer only ever cited [1]. The cause was a blind "top-N of what
retrieval returned" source list, plus a bullet cap (MAX_KEY_POINTS) that could
trim the only bullet citing a document — leaving that document orphaned in the
Sources block.

The contract enforced here:
  * every displayed source is cited by the VISIBLE answer text;
  * every visible [N] marker has a matching displayed source (no dangling marker);
  * markers are renumbered 1..N in first-seen order and link to their own document;
  * an uncited retrieved document never appears.

Run:  python tests/test_citation_integrity.py  ·  pytest tests/test_citation_integrity.py
"""

import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from integrations import formatting                       # noqa: E402


def _src(title, url, cat="blog"):
    return {"title": title, "url": url, "category": cat, "source": "website", "text": "t"}


THREE = [
    _src("The Illusion of AI Mapping Intelligence", "https://t.org.in/blog/illusion"),
    _src("China's Mega Dam and the Question India Should Be Asking", "https://t.org.in/blog/dam"),
    _src("Introduction to Geospatial Science and Technology", "https://t.org.in/page/geo", "page"),
]


def _render(answer, sources, mode="normal"):
    result = {"answer": answer, "confidence": "high",
              "retrieval_time": 0.4, "generation_time": 1.5, "sources": sources}
    return formatting.format_answer("Explain the illusion of 3D mapping", result, mode=mode)


def _sources_block(msg):
    return msg.split("### 📚 Sources")[1].split("### 📊")[0] if "### 📚 Sources" in msg else ""


def _visible_markers(msg):
    return sorted({int(n) for n in re.findall(r"\[\[(\d+)\]\]", msg)} |
                  {int(n) for n in re.findall(r"(?<!\[)\[(\d+)\](?!\()", msg)})


# ── The reported bug ─────────────────────────────────────────────────────────────

def test_only_cited_sources_are_shown():
    answer = ("**Answer**\nMisconception about AI [Source 1].\n\n"
              "**Details**\n- LLMs generate maps fast [Source 1].")
    msg = _render(answer, THREE)
    block = _sources_block(msg)
    assert "The Illusion of AI Mapping Intelligence" in block
    assert "Mega Dam" not in block          # retrieved but never cited
    assert "Geospatial Science" not in block
    assert "📚 Sources Used: 1" in msg

def test_trimmed_bullet_does_not_orphan_a_source():
    # 5 bullets; MAX_KEY_POINTS=4 trims the last one, which is the only [Source 2] use.
    answer = ("**Answer**\nMain claim [Source 1].\n\n**Details**\n"
              "- one [Source 1].\n- two [Source 1].\n- three [Source 1].\n"
              "- four [Source 1].\n- five [Source 2].")
    msg = _render(answer, THREE)
    assert "Mega Dam" not in _sources_block(msg)   # its only citation was trimmed away
    assert "📚 Sources Used: 1" in msg

def test_short_mode_uses_only_summary_citations():
    answer = ("**Answer**\nMain claim [Source 1].\n\n**Details**\n- detail [Source 2].")
    msg = _render(answer, THREE, mode="short")
    assert "Mega Dam" not in _sources_block(msg)
    assert "📚 Sources Used: 1" in msg


# ── Renumbering + links stay correct ─────────────────────────────────────────────

def test_noncontiguous_citations_are_renumbered_and_linked():
    answer = ("**Answer**\nA [Source 1]. C [Source 3].\n\n"
              "**Details**\n- from A [Source 1].\n- from C [Source 3].")
    msg = _render(answer, THREE)
    assert "[[1]](https://t.org.in/blog/illusion)" in msg    # source 1 → [1]
    assert "[[2]](https://t.org.in/page/geo)" in msg         # source 3 → [2], own URL
    assert "Mega Dam" not in _sources_block(msg)             # uncited middle source gone
    assert "📚 Sources Used: 2" in msg

def test_no_dangling_marker_without_a_source():
    answer = "**Answer**\nA [Source 1]. C [Source 3].\n\n**Details**\n- x [Source 1]."
    msg = _render(answer, THREE)
    shown = len(re.findall(r"\*\*\d+\.\*\*", _sources_block(msg)))
    assert _visible_markers(msg) == list(range(1, shown + 1))

def test_duplicate_documents_collapse_to_one_source():
    dupes = [_src("Same Doc", "https://x/same"), _src("Same Doc", "https://x/same"),
             _src("Other", "https://x/other")]
    answer = "**Answer**\nA [Source 1]. B [Source 2].\n\n**Details**\n- x [Source 1]."
    msg = _render(answer, dupes)
    assert "📚 Sources Used: 1" in msg           # [1] and [2] are the same document
    assert "Other" not in _sources_block(msg)

def test_answer_without_markers_attributes_to_best_source():
    answer = "**Answer**\nA grounded answer with no markers.\n\n**Details**\n- a point."
    msg = _render(answer, THREE)
    assert "📚 Sources Used: 1" in msg
    assert "The Illusion of AI Mapping Intelligence" in _sources_block(msg)

def test_repeat_runs_are_consistent_for_same_answer():
    answer = "**Answer**\nClaim [Source 1].\n\n**Details**\n- point [Source 1]."
    a, b = _render(answer, THREE), _render(answer, THREE)
    assert _sources_block(a) == _sources_block(b)


# ── select_cited_sources unit level ──────────────────────────────────────────────

def test_select_cited_sources_returns_cited_only():
    shown, ref_map = formatting.select_cited_sources("x [Source 3]. y [1].", THREE)
    titles = [s["title"] for s in shown]
    assert titles == ["Introduction to Geospatial Science and Technology",
                      "The Illusion of AI Mapping Intelligence"]   # first-seen order
    assert ref_map == {3: 1, 1: 2}

def test_select_cited_sources_empty_sources():
    assert formatting.select_cited_sources("[1]", []) == ([], {})

def test_displayed_sources_respects_answer_text():
    assert len(formatting.displayed_sources(THREE, "only [Source 2] here")) == 1
    assert formatting.displayed_sources(THREE, "only [Source 2] here")[0]["title"].startswith("China")


if __name__ == "__main__":
    mod = sys.modules[__name__]
    passed = 0
    for name in sorted(dir(mod)):
        if name.startswith("test_"):
            getattr(mod, name)()
            passed += 1
    print(f"OK — {passed} tests passed")