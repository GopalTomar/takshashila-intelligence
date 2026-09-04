"""
tests/test_answer_length.py — Response-length control in the RAG pipeline.

Verifies that answer(length=…) / mode=… actually changes the generation prompt
and token budget (short < normal < detailed), that a bot ``mode`` is accepted as
an alias, and that the model is no longer asked to hand-write a Sources list
(the interface renders sources from the verified set). Groq + retrieval are
mocked, so no network or model download is needed.

Run:  python tests/test_answer_length.py   ·   pytest tests/test_answer_length.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src import config                      # noqa: E402
import src.rag_pipeline as rp               # noqa: E402


FAKE_CHUNKS = [
    {"title": "Leave Policy", "category": "policy", "source": "commit_kb",
     "source_name": "Commit KB", "url": "https://x/leave", "text":
     "Staff may take leave with prior approval. Sabbaticals require director sign-off.",
     "score": 0.62},
    {"title": "Handbook", "category": "hr", "source": "handbook",
     "source_name": "Staff Handbook", "url": "https://x/hb", "text":
     "The handbook describes leave categories and approval flow.", "score": 0.55},
]


class Capture:
    """Mock retrieval + groq_client.generate; capture the system prompt + max_tokens."""

    def __init__(self):
        self.system_prompt = None
        self.max_tokens = None
        self._orig = {}

    def __enter__(self):
        self._orig = {
            "retrieve": rp.retrieve,
            "confidence_level": rp.confidence_level,
            "best_cosine": rp.best_cosine,
            "has_sufficient_evidence": rp.has_sufficient_evidence,
            "VERIFY_CITATIONS": config.VERIFY_CITATIONS,
        }
        rp.retrieve = lambda **kw: list(FAKE_CHUNKS)
        rp.confidence_level = lambda chunks: "high"
        rp.best_cosine = lambda chunks: 0.62
        rp.has_sufficient_evidence = lambda chunks: True
        config.VERIFY_CITATIONS = False        # skip grounding math for a clean unit test

        import src.groq_client as gc
        self._orig["generate"] = gc.generate

        def fake_generate(system_prompt, user_prompt, model=None, temperature=0.1,
                          max_tokens=1500):
            self.system_prompt = system_prompt
            self.max_tokens = max_tokens
            return "Staff may take leave with prior approval [Source 1]."
        gc.generate = fake_generate
        self._gc = gc
        return self

    def __exit__(self, *exc):
        rp.retrieve = self._orig["retrieve"]
        rp.confidence_level = self._orig["confidence_level"]
        rp.best_cosine = self._orig["best_cosine"]
        rp.has_sufficient_evidence = self._orig["has_sufficient_evidence"]
        config.VERIFY_CITATIONS = self._orig["VERIFY_CITATIONS"]
        self._gc.generate = self._orig["generate"]


def _run(**kw):
    with Capture() as cap:
        result = rp.answer(query="What is the leave policy?", **kw)
    return result, cap


def test_short_uses_small_budget_and_terse_prompt():
    _, cap = _run(length="short")
    assert cap.max_tokens == rp._LENGTH_GUIDANCE["short"][1]
    assert "2–4 crisp sentences" in cap.system_prompt

def test_detailed_uses_large_budget_and_thorough_prompt():
    _, cap = _run(length="detailed")
    assert cap.max_tokens == rp._LENGTH_GUIDANCE["detailed"][1]
    assert "thorough" in cap.system_prompt.lower()

def test_budgets_increase_short_normal_detailed():
    _, s = _run(length="short")
    _, n = _run(length="normal")
    _, d = _run(length="detailed")
    assert s.max_tokens < n.max_tokens < d.max_tokens

def test_bot_mode_is_accepted_as_alias():
    _, short = _run(mode="short")
    assert short.max_tokens == rp._LENGTH_GUIDANCE["short"][1]
    _, detailed = _run(mode="detailed")
    assert detailed.max_tokens == rp._LENGTH_GUIDANCE["detailed"][1]

def test_no_handwritten_sources_list_requested():
    _, cap = _run(length="normal")
    # The model must NOT be told to emit its own numbered Sources list.
    assert "numbered list: Title" not in cap.system_prompt
    assert "shows the source list automatically" in cap.system_prompt

def test_default_is_normal():
    _, cap = _run()
    assert cap.max_tokens == rp._LENGTH_GUIDANCE["normal"][1]

def test_sources_returned_for_rendering():
    result, _ = _run(length="normal")
    assert result["sources"], "sources should be returned for the UI to render"
    assert result["confidence"] in ("high", "medium", "low", "none")


if __name__ == "__main__":
    mod = sys.modules[__name__]
    passed = 0
    for name in sorted(dir(mod)):
        if name.startswith("test_"):
            getattr(mod, name)()
            passed += 1
    print(f"OK — {passed} tests passed")