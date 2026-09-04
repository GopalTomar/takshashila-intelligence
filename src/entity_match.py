"""
src/entity_match.py — Deterministic person-name extraction and author-metadata
matching, used to fix a real, diagnosed retrieval gap:

  "Who is Dr Y Nithiyanandam?"              -> works (matches a bio/profile page)
  "What blogs has Dr Y Nithiyanandam written?" -> used to fail

Root cause (reproduced and confirmed, see tests/test_entity_match.py and
tests/test_retrieval_evidence_gate.py): the evidence gate in src/retriever.py
only trusted FAISS's raw cosine score. A chunk found ONLY by BM25 — which is
exactly what happens for a lexically distinctive, name-heavy authorship query
whose phrasing embeds awkwardly — has no 'score' key at all, so
best_cosine() silently treats it as 0.0 "evidence quality" regardless of how
precise the actual match was. Lowering MIN_SCORE_THRESHOLD would not have
fixed this (the chunk's cosine score isn't merely low, it doesn't exist), and
would have made the gate more permissive for every OTHER query too — exactly
the "just lower the threshold" fix this module deliberately avoids.

The fix here is narrow and precision-first, per the explicit requirement that
"a document mentions a person" must never be conflated with "the person
authored it": this module only ever checks the chunk's AUTHOR METADATA field
(populated by the crawler's byline/JSON-LD extraction — see
scripts/crawl_engine.py's _extract_rich_metadata), never the chunk's body
text. A blog post that merely quotes or discusses Nithiyanandam does not
match; only a chunk whose own author field names him does.

No LLM call is used for this base case — deterministic, dependency-free,
and testable in isolation, per "no blind LLM dependency" (see also
src/config.py's PERSON_TITLES for the small, explicit set of honorifics this
strips before comparing names).
"""

from __future__ import annotations

import re
from typing import Dict, List

# Honorifics/titles stripped when extracting or comparing candidate names.
# Deliberately small and explicit rather than guessed — extend if a real,
# observed query uses another title not covered here.
_TITLES = {"dr", "dr.", "prof", "prof.", "professor", "mr", "mr.", "mrs", "mrs.",
           "ms", "ms.", "shri", "smt"}

# Words that are capitalized for reasons OTHER than being a person's name —
# sentence-initial question words, and the institution's own name (which
# never appears in a chunk's *author* field, but excluding it here avoids a
# pointless extraction on nearly every query about Takshashila's own work).
_QUERY_STOPWORDS = {"what", "which", "who", "where", "when", "why", "how",
                    "does", "is", "are", "was", "were", "has", "have",
                    "takshashila", "india", "indian"}

# A candidate name token: capitalized word, optionally with a single trailing
# period (initials like "Y." or "Y" alone), at least 1 letter.
_NAME_TOKEN_RE = re.compile(r"[A-Z][a-zA-Z]*\.?")


def extract_candidate_person_names(query: str) -> List[str]:
    """
    Pull out likely person-name phrases from a natural-language query using a
    simple, deterministic heuristic: runs of 2+ consecutive Title-Case tokens
    (honorifics like "Dr"/"Prof" are allowed to start a run but stripped from
    the returned name).

    "What blogs has Dr Y Nithiyanandam written?" -> ["Y Nithiyanandam"]
    "Who is Dr Y Nithiyanandam?"                 -> ["Y Nithiyanandam"]
    "What has Takshashila published about AI?"   -> [] (single token runs,
                                                          and "AI" alone isn't
                                                          a 2+ token run)

    This is intentionally conservative: it only ever proposes a NAME to look
    up against verified metadata (see author_matches_chunk below) — it never
    asserts that name authored anything by itself. A wrong or spurious
    extraction here has no effect unless it later exactly matches a chunk's
    real author field.
    """
    if not query:
        return []

    words = query.replace("?", " ? ").split()
    runs: List[List[str]] = []
    current: List[str] = []

    for w in words:
        bare = w.strip(",.;:!\"'()")
        is_title_case = bool(_NAME_TOKEN_RE.fullmatch(bare)) and bare[0].isupper()
        is_honorific = bare.lower().rstrip(".") in {t.rstrip(".") for t in _TITLES}
        if is_title_case or is_honorific:
            current.append(bare)
        else:
            if len(current) >= 2:
                runs.append(current)
            current = []
    if len(current) >= 2:
        runs.append(current)

    names = []
    for run in runs:
        cleaned = [t for t in run if t.lower().rstrip(".") not in {t2.rstrip(".") for t2 in _TITLES}]
        if len(cleaned) >= 2:
            names.append(" ".join(cleaned))

    # Fallback: a single, sufficiently distinctive Title-Case token not
    # already covered by a multi-token run above (e.g. "Which articles has
    # Nithiyanandam authored?" — no honorific, no initial, just a surname).
    # Length >= 5 keeps this from firing on short/common capitalized words
    # ("What", "AI", "Takshashila" is 11 chars but is an organization name,
    # not a person — organizations don't appear in a chunk's *author* field
    # in this pipeline, so even a false-positive "candidate" here is
    # harmless: name_matches_author_field only ever compares against real
    # author metadata, never chunk body text).
    covered = {t for run in runs for t in run}
    for w in words:
        bare = w.strip(",.;:!\"'()")
        if (bare not in covered and len(bare) >= 5
                and _NAME_TOKEN_RE.fullmatch(bare) and bare[0].isupper()
                and bare.lower().rstrip(".") not in {t.rstrip(".") for t in _TITLES}
                and bare.lower() not in _QUERY_STOPWORDS):
            names.append(bare)

    return names


def _normalize_name(name: str) -> str:
    """Lowercase, strip honorifics/punctuation, collapse whitespace — used so
    'Dr. Y. Nithiyanandam', 'Y Nithiyanandam', and 'y nithiyanandam' compare
    equal without treating unrelated names as equal."""
    tokens = [t.strip(".,") for t in name.lower().split()]
    tokens = [t for t in tokens if t.rstrip(".") not in {t2.rstrip(".") for t2 in _TITLES}]
    return " ".join(tokens)


def name_matches_author_field(candidate_name: str, author_field) -> bool:
    """
    True if candidate_name plausibly refers to the same person as
    author_field (a string, or a list of strings — chunks store both
    "author" and "authors" depending on source).

    Matches on: exact normalized match, or the candidate's LAST token (surname)
    appearing as a whole word in the author field — covers "Nithiyanandam"
    matching "Dr. Y. Nithiyanandam" without matching unrelated names that
    happen to share a common first name/initial.
    """
    if not candidate_name or not author_field:
        return False

    authors = author_field if isinstance(author_field, list) else [author_field]
    cand_norm = _normalize_name(candidate_name)
    if not cand_norm:
        return False
    cand_tokens = cand_norm.split()
    surname = cand_tokens[-1] if cand_tokens else ""

    for a in authors:
        a_norm = _normalize_name(str(a or ""))
        if not a_norm:
            continue
        if cand_norm == a_norm:
            return True
        a_tokens = set(a_norm.split())
        # Surname match: require it to be a distinctive token (not a common
        # short title-ish word) to avoid weak single-letter/initial matches.
        if len(surname) >= 4 and surname in a_tokens:
            return True
    return False


def chunk_author_matches_query(chunk: Dict, candidate_names: List[str]) -> bool:
    """True if this chunk's own author metadata (never its body text) matches
    any candidate name extracted from the query."""
    if not candidate_names:
        return False
    author_field = chunk.get("author") or chunk.get("authors")
    if not author_field:
        return False
    return any(name_matches_author_field(name, author_field) for name in candidate_names)


# Words that signal the user is asking about what someone WROTE, as opposed
# to who they ARE. Deterministic and explicit rather than guessed — this is
# what keeps the author-match boost (see retriever.retrieve()) from
# hijacking a plain identity query ("Who is Dr X?") toward content that
# merely happens to be authored by a name mentioned in the query, when a
# bio/profile page is the actually-relevant result for that phrasing.
_AUTHORSHIP_INTENT_WORDS = {
    "written", "write", "writes", "wrote", "author", "authored", "authors",
    "publish", "published", "publication", "publications", "blog", "blogs",
    "article", "articles", "paper", "papers", "brief", "briefs", "report",
    "reports", "op-ed", "op-eds", "piece", "pieces", "work", "works",
    "contributed", "contribution", "contributions",
}


def is_authorship_query(query: str) -> bool:
    """True if the query is asking what someone wrote/published, rather than
    e.g. who they are. Simple deterministic keyword check — see
    _AUTHORSHIP_INTENT_WORDS' docstring note above for why this exists."""
    if not query:
        return False
    tokens = re.findall(r"[a-zA-Z-]+", query.lower())
    return any(t in _AUTHORSHIP_INTENT_WORDS for t in tokens)
