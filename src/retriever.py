"""
retriever.py — Hybrid BM25 + FAISS retrieval with source-priority ranking.

Pipeline:
  1. FAISS semantic search (raw cosine 'score' kept on every hit).
  2. BM25 lexical search (optional, improves keyword recall).
  3. Reciprocal Rank Fusion to combine the two rankings.
  4. Source-priority boost: Commit KB > Staff Handbook > everything else.
  5. Per-document dedup, then top-k.

Each returned chunk carries:
  - score      : best raw cosine similarity (used for the evidence gate)
  - rrf_score  : fused, priority-boosted ranking score (used for ordering)
"""

import re
from typing import Dict, List, Optional

from src import config
from src.utils import clean_chunk_metadata, get_logger
from src import vector_store
from src.vector_store import search as faiss_search
from src.entity_match import (
    extract_candidate_person_names, chunk_author_matches_query, is_authorship_query,
)

logger = get_logger("retriever", config.SCRAPE_LOG)

_BM25 = None
_BM25_CHUNKS: List[Dict] = []
_BM25_SIG = None   # (id(metadata_list), len) — used to detect a rebuilt index


def ensure_bm25_ready():
    """
    Lazy-build a BM25 index over the SAME chunk list FAISS already loaded.

    We deliberately reuse ``vector_store.get_metadata()`` rather than re-reading
    chunks.jsonl, so the process keeps a single full copy of the chunk text in
    memory (instead of one for FAISS and another for BM25). The index is rebuilt
    only if the underlying metadata changes (e.g. after a rebuild), which is
    cheap to detect via the list's identity + length.
    """
    global _BM25, _BM25_CHUNKS, _BM25_SIG
    try:
        chunks = vector_store.get_metadata()
    except Exception as exc:
        logger.warning(f"BM25 setup skipped (index not loaded): {exc}")
        return

    sig = (id(chunks), len(chunks))
    if _BM25 is not None and _BM25_SIG == sig:
        return   # already built for this exact metadata list

    if not chunks:
        return
    try:
        from rank_bm25 import BM25Okapi
        from src.utils import chunk_search_text
        tokenised = [chunk_search_text(ch).lower().split() for ch in chunks]
        _BM25 = BM25Okapi(tokenised)
        _BM25_CHUNKS = chunks          # reference, not a copy
        _BM25_SIG = sig
        logger.info(f"BM25 index built over {len(chunks)} chunks (shared metadata)")
    except Exception as exc:
        logger.warning(f"BM25 setup failed (using FAISS-only): {exc}")


# Backwards-compatible alias for any external caller / older imports.
def _ensure_bm25():
    ensure_bm25_ready()


def bm25_search(query: str, top_k: int = 20) -> List[Dict]:
    """
    Return top BM25 results, each carrying BOTH the raw ``bm25_score`` and a
    ``lexical_score`` normalized to [0, 1] against the best hit for this query.

    The normalization matters: BM25's raw scores are unbounded and corpus/query
    dependent, so they can't be compared to a cosine similarity or thresholded
    directly. A relative score (best lexical hit = 1.0) gives the evidence gate a
    stable, query-independent signal — this is what lets a precise lexical match
    ("ATP") count as real evidence even when its dense cosine score is low.
    """
    ensure_bm25_ready()
    if _BM25 is None or not _BM25_CHUNKS:
        return []
    scores = _BM25.get_scores(query.lower().split())
    ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)
    ranked = [(idx, float(s)) for idx, s in ranked[:top_k] if s > 0]
    if not ranked:
        return []
    top_score = ranked[0][1] or 1.0
    return [{**_BM25_CHUNKS[idx],
             "bm25_score": score,
             "lexical_score": round(score / top_score, 4)}
            for idx, score in ranked]


def _cid(ch: Dict, fallback) -> str:
    return ch.get("chunk_id") or ch.get("chunk_hash") or str(fallback)


# ── Lexical (exact-match) evidence via query-term coverage ────────────────────
# A precision-preserving alternative to thresholding a normalized BM25 score
# (which is always 1.0 for the top hit, so it would rubber-stamp every query with
# any token overlap). "Strong lexical evidence" means a single chunk contains
# ALL of the query's *salient* tokens — non-stopword tokens of length ≥ 3 — for a
# short, lookup-style query. That rescues exact-match queries ("What is ATP?",
# "All Things Policy", "GCPP") without loosening the gate for descriptive
# questions or unrelated text (which never have one chunk covering all salients).
_STOPWORDS = {
    "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
    "what", "who", "whom", "whose", "which", "when", "where", "why", "how",
    "does", "do", "did", "of", "in", "on", "to", "for", "with", "and", "or",
    "about", "tell", "me", "us", "explain", "describe", "simple", "terms",
    "please", "give", "list", "behind", "into", "that", "this", "these",
    "those", "it", "its", "there", "here", "no", "not", "any", "some", "you",
    "your", "can", "could", "would", "should", "has", "have", "had", "at",
    "by", "as", "from", "up", "out", "so", "if", "than", "then",
}
_MAX_SALIENT_FOR_LEXICAL = 6   # descriptive queries (>6 salients) skip this path


def _salient_tokens(query: str) -> List[str]:
    toks = re.findall(r"[a-z0-9]+", (query or "").lower())
    return [t for t in toks if len(t) >= 3 and t not in _STOPWORDS]


def _chunk_token_set(ch: Dict) -> set:
    from src.utils import chunk_search_text
    return set(re.findall(r"[a-z0-9]+", chunk_search_text(ch).lower()))


def retrieve(
    query: str,
    top_k: int = config.TOP_K,
    source: Optional[str] = None,
    category: Optional[str] = None,
    author: Optional[str] = None,
    year: Optional[str] = None,
    use_hybrid: bool = True,
    # legacy aliases
    source_type: Optional[str] = None,
) -> List[Dict]:
    """Hybrid retrieval with source-priority ranking and dedup."""
    if source is None and source_type is not None:
        source = source_type

    faiss_results = faiss_search(
        query, top_k=top_k * 3,
        source_filter=source,
        category_filter=category,
        author_filter=author,
        year_filter=year,
    )
    bm25_results = bm25_search(query, top_k=top_k * 3) if use_hybrid else []

    # ── Reciprocal Rank Fusion ─────────────────────────────────────────────
    k_rrf = 60
    fused: Dict[str, float] = {}
    chunk_map: Dict[str, Dict] = {}
    lex_by_cid: Dict[str, float] = {}   # normalized lexical (BM25) score per chunk

    for rank, r in enumerate(faiss_results):
        cid = _cid(r, rank)
        fused[cid] = fused.get(cid, 0.0) + 1.0 / (k_rrf + rank + 1)
        chunk_map[cid] = r   # has cosine 'score'

    for rank, r in enumerate(bm25_results):
        cid = _cid(r, f"b{rank}")
        fused[cid] = fused.get(cid, 0.0) + 1.0 / (k_rrf + rank + 1)
        lex_by_cid[cid] = max(lex_by_cid.get(cid, 0.0), float(r.get("lexical_score", 0.0)))
        if cid not in chunk_map:
            chunk_map[cid] = r   # BM25-only: has no cosine 'score', only lexical

    # ── Source-priority boost ───────────────────────────────────────────────
    for cid, base in fused.items():
        src = (chunk_map[cid].get("source") or chunk_map[cid].get("source_type") or "")
        tier = config.source_priority(src)
        boost = 1.0 + (tier - config.DEFAULT_SOURCE_PRIORITY) * config.SOURCE_PRIORITY_BOOST
        fused[cid] = base * boost

    # ── Author-match boost ──────────────────────────────────────────────────
    # Fixes a real, diagnosed gap: "What blogs has Dr Y Nithiyanandam
    # written?" needs authored-by-this-person content to rank far above
    # content that merely mentions them in passing. This checks the chunk's
    # own AUTHOR METADATA field only — never body text — so a document that
    # quotes or discusses someone is never mistaken for one they wrote (see
    # src/entity_match.py's docstring for the full rationale).
    #
    # Gated on is_authorship_query(): a plain identity question ("Who is Dr
    # X?") should still surface a bio/profile page even if that page has no
    # "author" byline of its own — boosting author-matched content for EVERY
    # query that happens to contain a name would hijack identity queries
    # toward that person's authored content instead of their profile. Only
    # apply the boost when the query itself signals "what did they write."
    # Precompute which chunks fully cover the query's salient tokens (strong
    # exact-match / lexical evidence). Only meaningful for short lookup queries.
    salient = _salient_tokens(query)
    lexical_eligible = bool(salient) and len(salient) <= _MAX_SALIENT_FOR_LEXICAL
    salient_set = set(salient)

    candidate_names = extract_candidate_person_names(query)
    author_matched_cids: set = set()
    if candidate_names and is_authorship_query(query):
        AUTHOR_MATCH_BOOST = 2.5
        for cid in fused:
            if chunk_author_matches_query(chunk_map[cid], candidate_names):
                fused[cid] *= AUTHOR_MATCH_BOOST
                author_matched_cids.add(cid)

    ranked = sorted(fused.items(), key=lambda x: x[1], reverse=True)

    # ── Dedup + evidence-first selection ─────────────────────────────────────
    # At most 2 chunks per source document. Navigational / listing / landing
    # pages are held back (deferred) so they never push a real article out of
    # the top-k; if too few genuine-evidence chunks exist, we backfill from the
    # deferred pool so results are never empty.
    from src.utils import chunk_is_low_value

    doc_count: Dict[str, int] = {}
    taken: set = set()
    final: List[Dict] = []

    def _select(consider_low_value: bool) -> None:
        for cid, rrf_score in ranked:
            if len(final) >= top_k:
                return
            if cid in taken:
                continue
            ch = chunk_map[cid]
            doc_id = ch.get("document_id") or ch.get("doc_id") or cid
            if chunk_is_low_value(ch) != consider_low_value:
                continue          # pass 1 keeps evidence; pass 2 backfills nav/listing
            if doc_count.get(doc_id, 0) >= 2:
                continue
            doc_count[doc_id] = doc_count.get(doc_id, 0) + 1
            taken.add(cid)
            out = {**ch, "rrf_score": rrf_score}
            # Keep the three signals SEPARATE and explicit on every chunk:
            #   score          — dense/semantic cosine similarity (0 if BM25-only)
            #   lexical_score  — normalized BM25 lexical match in [0,1]
            #   rrf_score      — fused, priority-boosted ranking score
            # Critically, we do NOT overwrite a real cosine score with 0 for a
            # BM25-only hit, and we do NOT pretend a lexical hit has a cosine
            # score — the evidence gate reads whichever signal is genuine.
            out.setdefault("score", float(ch.get("score", 0.0)))
            out["lexical_score"] = max(float(out.get("lexical_score", 0.0)),
                                       lex_by_cid.get(cid, 0.0))
            # Exact-match evidence: does THIS chunk contain every salient query
            # token? (Precise substitute for thresholding a normalized BM25 score.)
            out["lexical_verified"] = bool(
                lexical_eligible and salient_set.issubset(_chunk_token_set(ch))
            )
            out["author_verified"] = cid in author_matched_cids
            final.append(clean_chunk_metadata(out))   # safety net if index predates cleaning

    _select(consider_low_value=False)     # genuine evidence first
    if not final:
        _select(consider_low_value=True)  # backfill only if NOTHING else exists

    return final


def best_cosine(chunks: List[Dict]) -> float:
    """Best raw cosine (semantic) similarity among retrieved chunks (0 if none)."""
    return max((float(c.get("score", 0.0)) for c in chunks), default=0.0)


def best_lexical(chunks: List[Dict]) -> float:
    """Best normalized lexical (BM25) score among retrieved chunks (0 if none)."""
    return max((float(c.get("lexical_score", 0.0)) for c in chunks), default=0.0)


def has_strong_lexical_evidence(chunks: List[Dict]) -> bool:
    """
    True when at least one retrieved chunk fully covers the query's salient
    tokens (``lexical_verified``). This is a SEPARATE evidence path from dense
    cosine similarity — it exists so a precise keyword/acronym match ("ATP",
    "GCPP", "All Things Policy") counts as real evidence even when the dense
    embedding for a very short query scores low. It is deliberately a high,
    exact-coverage bar (not a lowered dense threshold, and not a normalized-BM25
    threshold that would rubber-stamp any token overlap), so it raises recall for
    exact-match lookups without making the gate more permissive for descriptive
    or unrelated queries — which never have one chunk covering all salient tokens.
    """
    return any(bool(c.get("lexical_verified")) for c in chunks)


def has_author_verified_evidence(chunks: List[Dict]) -> bool:
    """
    True if at least one retrieved chunk carries a verified author-metadata
    match (set by retrieve()'s author-match boost — see src/entity_match.py).

    This is the fix for a real, diagnosed gap: an authorship query's best
    evidence is sometimes found ONLY by BM25 (strong lexical match on a
    name), never by FAISS's semantic search for that particular phrasing.
    Such a chunk has no 'score' key at all, so best_cosine() treats it as
    0.0 "quality" regardless of how precise the match actually was. Rather
    than lowering MIN_SCORE_THRESHOLD (which would make the gate more
    permissive for every query, including ones with no such verified
    signal), this gives a SEPARATE, narrow, high-precision path to
    sufficiency: the chunk's own author field — never its body text —
    matches a name extracted from the query.
    """
    return any(bool(c.get("author_verified")) for c in chunks)


def confidence_level(chunks: List[Dict]) -> str:
    """Map evidence quality to a confidence tier. A verified author-metadata
    match (see has_author_verified_evidence) is treated as at least "medium"
    confidence even when the raw cosine score is low or absent — it's a
    different, more precise signal than semantic similarity, not a weaker
    version of it."""
    top = best_cosine(chunks)
    if top >= config.CONF_HIGH_THRESHOLD:
        return "high"
    if top >= config.CONF_MEDIUM_THRESHOLD:
        return "medium"
    # A full salient-token cover (exact keyword/acronym match) is a real, precise
    # signal — treated as at least "medium" even when dense cosine is low.
    if has_strong_lexical_evidence(chunks):
        return "medium"
    if top >= config.MIN_SCORE_THRESHOLD:
        return "low"
    if has_author_verified_evidence(chunks):
        return "medium"
    return "none"


def has_sufficient_evidence(chunks: List[Dict]) -> bool:
    """
    True if there is genuine evidence by ANY of the retriever's independent
    signals:
      * dense cosine clears ``MIN_SCORE_THRESHOLD`` (semantic match), OR
      * a strong normalized lexical/BM25 match (exact keyword/acronym match —
        fixes the ATP false-refusal where the best hit was lexical-only), OR
      * a verified author-metadata match (authorship queries).

    Keeping these separate is the whole point: a BM25-only hit is no longer
    silently assigned a cosine score of 0 and discarded.
    """
    return (
        best_cosine(chunks) >= config.MIN_SCORE_THRESHOLD
        or has_strong_lexical_evidence(chunks)
        or has_author_verified_evidence(chunks)
    )