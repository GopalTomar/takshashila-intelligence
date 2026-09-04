"""
tests/test_mojibake.py — Encoding repair for reference / title text.

Covers the specific bug where an em dash stored as Latin-1-misdecoded UTF-8
("Key Policies â\\x80\\x94 POSH") showed up raw in the dashboard References and the
bot's Related Policies. These forms use raw C1 control bytes (\\x80–\\x9f) rather
than the cp1252 glyphs, so they need their own repair entries — while genuine
accented words (Alcântara, réseau) must be left untouched.

Run:  python tests/test_mojibake.py   ·   pytest tests/test_mojibake.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils import (                                   # noqa: E402
    fix_mojibake_preserve_layout as fx,
    is_true_mojibake_present as detected,
    clean_chunk_metadata,
)


def test_em_and_en_dash_repaired():
    assert fx("Key Policies \u00e2\x80\x94 POSH") == "Key Policies \u2014 POSH"
    assert fx("Report \u00e2\x80\x93 Enhancing") == "Report \u2013 Enhancing"

def test_smart_quotes_repaired():
    assert fx("\u00e2\x80\x9cquoted\u00e2\x80\x9d") == "\u201cquoted\u201d"
    assert fx("India\u00e2\x80\x99s policy") == "India\u2019s policy"
    assert fx("\u00e2\x80\xa6ellipsis") == "\u2026ellipsis"

def test_these_forms_are_detected():
    for bad in ["A \u00e2\x80\x94 B", "A \u00e2\x80\x93 B", "x\u00e2\x80\x99s"]:
        assert detected(bad) is True, f"not detected: {bad!r}"

def test_source_dict_title_is_cleaned():
    src = {"title": "Key Policies \u00e2\x80\x94 POSH, Leave, Laptop",
           "source_name": "Commit KB", "source": "commit_kb", "url": "https://x/p"}
    out = clean_chunk_metadata(src)
    assert out["title"] == "Key Policies \u2014 POSH, Leave, Laptop"
    assert out["url"] == "https://x/p"          # url untouched

def test_valid_accents_preserved():
    for good in ["Alc\u00e2ntara", "r\u00e9seau", "d\u00e9cideurs", "Gr\u00e2ce", "caf\u00e9"]:
        assert detected(good) is False, f"false positive: {good!r}"
        assert fx(good) == good

def test_clean_text_is_idempotent():
    once = fx("Key Policies \u00e2\x80\x94 POSH")
    assert fx(once) == once                      # already-clean text is unchanged


# ── Lone-"â" mojibake (apostrophes, standalone dashes, â + box glyphs) ──────────
def test_lone_a_apostrophe_contractions():
    assert fx("Takshashila\u00e2s core HR") == "Takshashila\u2019s core HR"
    assert fx("doesn\u00e2t matter") == "doesn\u2019t matter"
    assert fx("we\u00e2re ready") == "we\u2019re ready"
    assert fx("they\u00e2ll go") == "they\u2019ll go"

def test_lone_a_standalone_dash():
    assert fx("\u00e2 What We Decided") == "\u2013 What We Decided"
    assert fx("Report \u00e2 Enhancing") == "\u2013".join(["Report ", " Enhancing"])

def test_lone_a_with_replacement_glyphs():
    assert fx("\u00e2\ufffd\ufffd What We Decided") == "\u2013 What We Decided"

def test_lone_a_does_not_touch_real_accents():
    for good in ["Alc\u00e2ntara", "Gr\u00e2ce", "ch\u00e2teau", "\u00e2me"]:
        assert detected(good) is False, f"false positive: {good!r}"
        assert fx(good) == good


# ── Document de-duplication for references (no repeated references) ─────────────
def test_dedupe_collapses_same_document():
    import src.rag_pipeline as rp
    chunks = [
        {"title": "Policy Responses to India LPG Supply Crisis", "source": "website",
         "url": "a.html", "text": "chunk one", "score": 0.80},
        {"title": "Policy Responses to India LPG Supply Crisis", "source": "website",
         "url": "a.html", "text": "chunk two", "score": 0.73},
        {"title": "Policy Responses to India LPG Supply Crisis [PDF]", "source": "website",
         "url": "a.pdf", "text": "pdf chunk", "score": 0.76, "page_number": 3},
        {"title": "Different Doc", "source": "commit_kb", "url": "b", "text": "x", "score": 0.6},
    ]
    out = rp._dedupe_by_document(chunks)
    assert len(out) == 2                                  # 3 LPG variants → 1, + 1 other
    assert "chunk one" in out[0]["text"] and "pdf chunk" in out[0]["text"]  # text merged
    titles = [o["title"] for o in out]
    assert titles.count("Policy Responses to India LPG Supply Crisis") == 1

def test_dedupe_keeps_distinct_documents():
    import src.rag_pipeline as rp
    chunks = [
        {"title": "Doc A", "source": "commit_kb", "text": "a", "score": 0.9},
        {"title": "Doc B", "source": "commit_kb", "text": "b", "score": 0.8},
    ]
    assert len(rp._dedupe_by_document(chunks)) == 2


if __name__ == "__main__":
    mod = sys.modules[__name__]
    passed = 0
    for name in sorted(dir(mod)):
        if name.startswith("test_"):
            getattr(mod, name)()
            passed += 1
    print(f"OK — {passed} tests passed")