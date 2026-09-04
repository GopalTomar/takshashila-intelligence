"""
citations.py — Post-generation citation verification (anti-hallucination).

The LLM is told to answer only from the retrieved context and to cite inline
like [Source N]. This module double-checks that promise BEFORE the answer is
shown, so the displayed references exactly match where the answer came from:

  1. Parse the [Source N] markers the model actually used.
  2. Keep ONLY those sources (dropping retrieved-but-uncited chunks), and
     renumber the answer so [Source 1..M] line up with the shown list.
  3. Ground-check: measure token overlap between the answer text and the cited
     chunks. If the answer cites nothing verifiable AND overlaps the retrieved
     context too little, flag it as ungrounded so the pipeline can refuse
     instead of surfacing a possibly hallucinated answer.

Everything here is deterministic and dependency-free (no extra model calls).
"""

from __future__ import annotations

import re
from typing import Dict, List

# Words ignored when measuring answer↔context overlap.
_STOPWORDS = {
    "the", "a", "an", "and", "or", "but", "if", "then", "else", "for", "of",
    "to", "in", "on", "at", "by", "is", "are", "was", "were", "be", "been",
    "being", "as", "that", "this", "these", "those", "it", "its", "with",
    "from", "into", "about", "which", "who", "whom", "whose", "what", "when",
    "where", "how", "why", "will", "would", "can", "could", "should", "may",
    "might", "must", "not", "no", "do", "does", "did", "has", "have", "had",
    "they", "them", "their", "we", "our", "you", "your", "he", "she", "his",
    "her", "also", "there", "here", "than", "such", "so", "some", "any",
}

_TOKEN_RE = re.compile(r"[a-z0-9]+")
_CITE_RE = re.compile(r"\[\s*Source\s*(\d+)\s*\]", re.IGNORECASE)


def _tokens(text: str) -> List[str]:
    return [t for t in _TOKEN_RE.findall((text or "").lower()) if t not in _STOPWORDS]


def _token_set(text: str) -> set:
    return set(_tokens(text))


def cited_indices(answer_text: str) -> List[int]:
    """The distinct 1-based source numbers the model cited, in first-seen order."""
    seen, out = set(), []
    for m in _CITE_RE.finditer(answer_text or ""):
        n = int(m.group(1))
        if n not in seen:
            seen.add(n)
            out.append(n)
    return out


def _overlap_ratio(text: str, context_tokens: set) -> float:
    toks = _token_set(text)
    if not toks:
        return 0.0
    return len(toks & context_tokens) / len(toks)


def verify(
    answer_text: str,
    used_sources: List[Dict],
    *,
    min_overlap: float = 0.18,
) -> Dict:
    """
    Verify + prune citations for a generated answer.

    Args:
        answer_text  : the raw LLM answer (may contain [Source N] markers).
        used_sources : the chunk dicts passed to the model, in the SAME order the
                       [Source N] numbers refer to (Source 1 == used_sources[0]).
        min_overlap  : minimum answer↔context token-overlap ratio below which an
                       answer that cites nothing verifiable is deemed ungrounded.

    Returns a dict:
        {
          "answer"    : answer text with citations renumbered to the kept sources,
          "sources"   : ONLY the sources actually cited (renumbered order),
          "grounded"  : bool — False means treat as insufficient/ungrounded,
          "overlap"   : float — best answer↔context overlap ratio,
          "cited"     : [int] — original source numbers the model cited,
          "dropped_uncited": int — retrieved sources removed because uncited,
        }
    """
    answer_text = answer_text or ""
    n = len(used_sources)
    context_tokens = set()
    for ch in used_sources:
        context_tokens |= _token_set(ch.get("text", ""))
    overlap = _overlap_ratio(answer_text, context_tokens)

    cited = [i for i in cited_indices(answer_text) if 1 <= i <= n]

    # ── Case A: the model cited specific sources → keep exactly those ──────────
    if cited:
        # Map original 1-based index → new 1-based index (in citation order).
        remap = {orig: new for new, orig in enumerate(cited, start=1)}
        kept_sources = [used_sources[orig - 1] for orig in cited]

        def _sub(m: "re.Match") -> str:
            orig = int(m.group(1))
            return f"[Source {remap[orig]}]" if orig in remap else ""

        new_answer = _CITE_RE.sub(_sub, answer_text)
        new_answer = re.sub(r"[ \t]{2,}", " ", new_answer)
        new_answer = re.sub(r"\s+([.,;:])", r"\1", new_answer).strip()

        return {
            "answer": new_answer,
            "sources": kept_sources,
            "grounded": True,           # cited + verified against the same chunks
            "overlap": overlap,
            "cited": cited,
            "dropped_uncited": n - len(kept_sources),
        }

    # ── Case B: no verifiable citations → fall back to overlap grounding ───────
    grounded = overlap >= min_overlap
    return {
        "answer": answer_text.strip(),
        # With no citations we can't attribute to a specific source; only keep the
        # single best-scoring chunk as a reference when the answer is grounded.
        "sources": used_sources[:1] if grounded and used_sources else [],
        "grounded": grounded,
        "overlap": overlap,
        "cited": [],
        "dropped_uncited": (n - 1) if (grounded and used_sources) else n,
    }