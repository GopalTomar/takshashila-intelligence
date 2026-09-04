"""
rag_pipeline.py — End-to-end RAG pipeline.

- Answers strictly from retrieved chunks (no external knowledge).
- Prioritises the Commit KB; falls back to Staff Handbook / other docs.
- Shows source citations with title, category, source, and URL.
- Returns an honest "insufficient evidence" message when nothing is relevant,
  and never shows misleading sources in that case.
- Reports retrieval confidence: high / medium / low / none.
"""

import re
import time
from typing import Dict, List, Optional, Tuple

from src import config
from src.retriever import (
    retrieve, best_cosine, confidence_level, has_sufficient_evidence,
)
from src.utils import clean_chunk_metadata, fix_mojibake_preserve_layout, get_logger

logger = get_logger("rag_pipeline")

# Exact phrase requested for the "not found" case.
NO_EVIDENCE_SENTENCE = (
    "I do not have sufficient evidence in the Takshashila knowledge base "
    "to answer this confidently."
)

_SYSTEM_BASE = """You are a precise research assistant for the Takshashila \
Institution, an independent public-policy think-tank in India.

Your ONLY source of knowledge is the context passages provided below. These \
passages come from Takshashila's internal knowledge base. The Commit Knowledge \
Base ("commit_kb") is the primary, most up-to-date source; the Staff Handbook \
and other documents are supporting sources.

STRICT RULES:
1. Answer ONLY from the provided context. Use zero outside knowledge.
2. Prefer information from Commit KB sources when they are relevant.
3. Do NOT invent policies, rules, numbers, names, or internal procedures. If \
the context does not clearly support a claim, do not make it.
4. Each source begins with a metadata line (Title / Author / Published / \
Updated / Category / Section / Tags / URL). You MAY use these fields to answer \
metadata questions such as "who wrote this?", "when was it published?", "what \
category is it?" or "what are the tags?", and cite the source they came from. \
Treat the metadata line as trustworthy context, not as outside knowledge.
5. If the context does not contain enough information to answer, respond with \
EXACTLY this single line and nothing else:
   INSUFFICIENT_EVIDENCE
6. Cite sources inline like [Source 1], [Source 2] right after each claim they \
support. Do NOT write a separate "Sources" list at the end — the interface \
shows the source list automatically, so a hand-written one would duplicate it."""

# Per-length FORMAT guidance + a generation token budget. "normal" is the
# balanced default; "short" is terse; "detailed" is comprehensive. All three keep
# the same strict grounding rules above and the inline [Source N] citations.
_LENGTH_GUIDANCE = {
    "short": (
        "Answer in 2–4 crisp sentences. Be direct and specific. Do not add a "
        "separate details section. Put an inline [Source N] after each claim.",
        600,
    ),
    "normal": (
        "**Answer**\n"
        "A direct answer in 2–5 sentences.\n\n"
        "**Details**\n"
        "- 3–6 key points, each ending with its [Source N] citation.",
        1300,
    ),
    "detailed": (
        "Write a thorough, well-structured answer.\n\n"
        "**Answer**\n"
        "A clear 3–6 sentence overview of the answer.\n\n"
        "**Details**\n"
        "- 6–12 specific points, grouped logically (for policy questions cover "
        "short-, medium- and long-term aspects where the context supports them), "
        "each ending with its [Source N] citation.\n\n"
        "Be comprehensive and draw on every relevant passage, but never state "
        "anything the context does not support.",
        2600,
    ),
}


def _normalize_length(length: Optional[str], mode: Optional[str]) -> str:
    """Resolve the requested verbosity. Accepts either ``length`` or a bot ``mode``."""
    val = (mode or length or "normal").lower()
    if val in ("short", "brief"):
        return "short"
    if val in ("detailed", "long", "full"):
        return "detailed"
    return "normal"


def _system_prompt(length: str) -> str:
    guidance, _ = _LENGTH_GUIDANCE.get(length, _LENGTH_GUIDANCE["normal"])
    return f"{_SYSTEM_BASE}\n\nFORMAT:\n{guidance}"


# Backward-compatible module export (some callers import SYSTEM_PROMPT directly).
SYSTEM_PROMPT = _system_prompt("normal")

NO_EVIDENCE_REPLY = (
    f"{NO_EVIDENCE_SENTENCE}\n\n"
    "This topic may not be covered in the indexed Commit KB / Staff Handbook "
    "content yet, or the question may be outside Takshashila's internal "
    "knowledge base.\n\n"
    "**You can try:**\n"
    "- Rephrasing with Takshashila-specific terms (e.g. \"flag system\", "
    "\"meeting rules\", \"core competencies\").\n"
    "- Browsing the indexed pages in the **Browse Sources** tab.\n"
    "- Re-running the Commit KB crawl and rebuilding the index if new pages "
    "were added."
)


# ── Source quality filter ────────────────────────────────────────────────────────
# Author pages, tag/category listings, section-index / landing pages and
# paginated indexes are retrievable text but they are NOT evidence — an answer
# "cited" to a listing/landing page tells the reader nothing about where a claim
# came from and links them to a list instead of the exact article. The single
# source of truth for this test lives in src.utils (shared with the retriever, so
# such pages never even consume a top-k slot); we also drop them here as defence
# in depth before they can reach the model's context.

# Minimum text a chunk must carry to count as evidence (nav pages are thin).
_MIN_EVIDENCE_CHARS = 200


def _is_low_value_source(ch: Dict) -> bool:
    """True when a chunk is a navigational/author/listing/landing page, not evidence."""
    from src.utils import chunk_is_low_value
    return chunk_is_low_value(ch, min_evidence_chars=_MIN_EVIDENCE_CHARS)


def _filter_evidence_chunks(chunks: List[Dict]) -> List[Dict]:
    """
    Keep only chunks that can genuinely support a citation:
      * not a navigational / author / listing page, and
      * at or above the retrieval relevance floor (``MIN_SCORE_THRESHOLD``).

    Never returns an empty list when the input was non-empty *and* something is
    above the floor — if every chunk is filtered out we return the originals so a
    weak-but-real answer is still possible (the grounding check then decides).
    """
    floor = float(getattr(config, "MIN_SCORE_THRESHOLD", 0.0))
    kept, dropped = [], []
    for ch in chunks:
        score = float(ch.get("score", 0.0) or 0.0)
        lexical_ok = bool(ch.get("lexical_verified"))
        author_ok = bool(ch.get("author_verified"))
        if _is_low_value_source(ch):
            dropped.append((ch.get("title"), "low-value page"))
            continue
        # Keep a chunk if ANY genuine signal supports it: a cosine score at/above
        # the floor, an exact salient-token match (lexical_verified), or a
        # verified author match. This stops a real exact-match ("ATP") chunk with
        # a low/zero cosine from being discarded as if it were noise.
        if score and score < floor and not lexical_ok and not author_ok:
            dropped.append((ch.get("title"), f"score {score:.2f} < {floor:.2f}, no exact/author match"))
            continue
        kept.append(ch)

    if dropped:
        logger.info("source filter dropped %d chunk(s): %s", len(dropped),
                    "; ".join(f"{t!r} ({why})" for t, why in dropped[:5]))
    return kept or chunks


def _document_key(ch: Dict) -> str:
    """
    Identity of the *document* a chunk belongs to, so multiple chunks (and the
    .html + .pdf variants of the same publication) collapse to one reference.
    Prefers an explicit document id; otherwise a normalised title + source.
    """
    did = ch.get("document_id") or ch.get("doc_id")
    if did:
        return f"id:{did}"
    title = (ch.get("title") or "").strip()
    title = re.sub(r"\s*\[(?:pdf|html?)\]\s*$", "", title, flags=re.IGNORECASE)
    title = re.sub(r"\s+", " ", title).lower()
    src = (ch.get("source") or "").strip().lower()
    return f"t:{title}|{src}"


def _dedupe_by_document(chunks: List[Dict], max_merged_chars: int = 4000) -> List[Dict]:
    """
    Collapse chunks from the same document into a single source: the first
    (highest-ranked) chunk is the representative, and the text of the other
    same-document chunks is merged into it (de-duplicated, capped). This keeps the
    context rich while ensuring each ``[Source N]`` — and therefore each displayed
    reference — is a distinct document, so references never repeat.
    """
    groups: Dict[str, Dict] = {}
    order: List[str] = []
    for ch in chunks:
        key = _document_key(ch)
        if key not in groups:
            g = dict(ch)
            g["_texts"] = [ch.get("text", "")]
            groups[key] = g
            order.append(key)
        else:
            groups[key]["_texts"].append(ch.get("text", ""))

    merged_out: List[Dict] = []
    for key in order:
        g = groups[key]
        texts, seen, kept = g.pop("_texts", []), set(), []
        for t in texts:
            t = (t or "").strip()
            if t and t not in seen:
                seen.add(t)
                kept.append(t)
        g["text"] = "\n\n".join(kept)[:max_merged_chars]
        merged_out.append(g)
    return merged_out


def _build_context_block(chunks: List[Dict]) -> Tuple[str, List[Dict]]:
    parts, sources = [], []
    budget = config.MAX_CONTEXT_CHARS

    for i, ch in enumerate(chunks):
        title    = ch.get("title", "Untitled")
        category = ch.get("category", "") or "—"
        src_name = ch.get("source_name") or config.source_display_name(
            ch.get("source", ""), ch.get("source", ""))
        url      = ch.get("url") or ch.get("original_url") or ""
        page     = f", page {ch['page_number']}" if ch.get("page_number") else ""

        # Surface the metadata so the model can answer "who wrote/when/what tags"
        # and cite this source for it. Only include fields that are present.
        authors = ch.get("authors")
        author = (", ".join(str(a) for a in authors if a) if isinstance(authors, (list, tuple))
                  else str(ch.get("author") or "")).strip()
        tags = ch.get("tags")
        tags = (", ".join(str(t) for t in tags if t) if isinstance(tags, (list, tuple))
                else str(tags or "")).strip()
        section = str(ch.get("heading_path") or ch.get("section") or "").strip()

        meta_bits = [f"[Source {i+1}] {title}"]
        if author:                    meta_bits.append(f"author: {author}")
        if ch.get("date"):            meta_bits.append(f"published: {ch['date']}")
        if ch.get("updated_date"):    meta_bits.append(f"updated: {ch['updated_date']}")
        meta_bits.append(f"category: {category}")
        if section:                   meta_bits.append(f"section: {section}")
        if tags:                      meta_bits.append(f"tags: {tags}")
        meta_bits.append(f"source: {src_name}{page}")
        meta_bits.append(f"url: {url}")
        header = " | ".join(meta_bits)
        block = f"{header}\n{ch.get('text','')}"

        if len(block) > budget and parts:
            break
        if len(block) > budget:
            block = block[:budget] + "…"
        parts.append(block)
        sources.append(ch)
        budget -= len(block)
        if budget <= 0:
            break

    return "\n\n---\n\n".join(parts), sources


def answer(
    query: str,
    top_k: int = config.TOP_K,
    model: Optional[str] = None,
    temperature: float = config.DEFAULT_TEMP,
    source: Optional[str] = None,
    category: Optional[str] = None,
    author: Optional[str] = None,
    year: Optional[str] = None,
    use_hybrid: bool = True,
    stream_response: bool = False,
    length: str = "normal",
    mode: Optional[str] = None,
    # legacy alias
    source_type: Optional[str] = None,
) -> Dict:
    """
    Full RAG pipeline. Returns:
        answer          : str (or generator if stream_response)
        sources         : list of chunk dicts (empty when no evidence)
        chunks          : all retrieved chunks
        confidence      : 'high' | 'medium' | 'low' | 'none'
        top_score       : best raw cosine similarity
        retrieval_time  : seconds spent retrieving (monotonic)
        generation_time : seconds spent in the LLM call (monotonic)

    ``length`` ("short" | "normal" | "detailed") controls verbosity; a bot
    ``mode`` value is accepted as an alias. Grounding and citation rules are
    identical for every length.
    """
    if source is None and source_type is not None:
        source = source_type

    length = _normalize_length(length, mode)
    system_prompt = _system_prompt(length)
    _, max_tokens = _LENGTH_GUIDANCE.get(length, _LENGTH_GUIDANCE["normal"])

    _t_retr0 = time.perf_counter()
    chunks = retrieve(
        query=query, top_k=top_k,
        source=source, category=category, author=author, year=year,
        use_hybrid=use_hybrid,
    )
    retrieval_time = time.perf_counter() - _t_retr0

    conf = confidence_level(chunks)
    top  = best_cosine(chunks)

    if not has_sufficient_evidence(chunks):
        return {
            "answer":         NO_EVIDENCE_REPLY,
            "sources":        [],
            "chunks":         chunks,
            "confidence":     "none",
            "top_score":      top,
            "retrieval_time": retrieval_time,
            "generation_time": 0.0,
        }

    # Drop navigational/author/listing pages and below-floor hits so the model can
    # only ever cite real evidence, then collapse same-document chunks so each
    # [Source N] is a distinct document.
    context_chunks = _dedupe_by_document(_filter_evidence_chunks(chunks))
    context_text, used_sources = _build_context_block(context_chunks)

    user_prompt = (
        f"Context from the Takshashila knowledge base:\n\n{context_text}\n\n"
        f"---\n\nQuestion: {query}\n\n"
        "Answer strictly from the context above. If it does not contain enough "
        "information, reply with exactly INSUFFICIENT_EVIDENCE."
    )

    from src import groq_client

    def _finish(text: str, generation_time: float) -> Dict:
        text = fix_mojibake_preserve_layout(text)
        if "INSUFFICIENT_EVIDENCE" in text:
            return {
                "answer":          NO_EVIDENCE_REPLY,
                "sources":         [],
                "chunks":          chunks,
                "confidence":      "none",
                "top_score":       top,
                "retrieval_time":  retrieval_time,
                "generation_time": generation_time,
            }

        final_answer, final_sources, final_conf = text, used_sources, conf

        # ── Citation verification (anti-hallucination) ────────────────────────
        # Keep ONLY the sources the answer actually cites (renumbered) and refuse
        # if the answer is ungrounded, so the displayed references exactly match
        # where the answer came from.
        if config.VERIFY_CITATIONS:
            from src.citations import verify
            v = verify(text, used_sources, min_overlap=config.GROUNDING_MIN_OVERLAP)
            if not v["grounded"]:
                logger.warning(
                    f"Ungrounded answer refused (overlap={v['overlap']:.2f} < "
                    f"{config.GROUNDING_MIN_OVERLAP}); query={query!r}"
                )
                return {
                    "answer":          NO_EVIDENCE_REPLY,
                    "sources":         [],
                    "chunks":          chunks,
                    "confidence":      "none",
                    "top_score":       top,
                    "retrieval_time":  retrieval_time,
                    "generation_time": generation_time,
                    "citation_check":  v,
                }
            final_answer = v["answer"]
            final_sources = v["sources"] or used_sources[:1]
            # Confidence is capped by the best-cited source's cosine score, so a
            # weakly-supported citation can't masquerade as high confidence.
            if final_sources:
                best_cited = max(
                    (float(s.get("score", 0.0)) for s in final_sources), default=top
                )
                final_conf = confidence_level(final_sources) if best_cited else final_conf

        return {
            "answer":          final_answer,
            "sources":         [clean_chunk_metadata(s) for s in final_sources],
            "chunks":          chunks,
            "confidence":      final_conf,
            "top_score":       top,
            "retrieval_time":  retrieval_time,
            "generation_time": generation_time,
        }

    if stream_response:
        # Streaming yields tokens live, so citations can't be verified/pruned
        # after the fact. The Mattermost bot uses the NON-streaming path, which
        # is fully verified. Use streaming only for the interactive UI preview.
        #
        # Audit note: this bypasses src.citations.verify() entirely, so a
        # streamed answer can show unverified [Source N] markers. Nothing
        # currently calls answer(stream_response=True) in the dashboard or the
        # Mattermost bot, but if a future feature adds a streaming UI path,
        # loudly say so rather than silently serving unverified citations.
        if config.VERIFY_CITATIONS:
            logger.warning(
                "answer() called with stream_response=True while "
                "VERIFY_CITATIONS is on — citation grounding is NOT applied "
                "to streamed responses; treat displayed sources as unverified."
            )
        gen = groq_client.stream(
            system_prompt=system_prompt, user_prompt=user_prompt,
            model=model, temperature=temperature, max_tokens=max_tokens,
        )
        return {
            "answer":          gen,
            "sources":         [clean_chunk_metadata(s) for s in used_sources],
            "chunks":          chunks,
            "confidence":      conf,
            "top_score":       top,
            "retrieval_time":  retrieval_time,
            "generation_time": 0.0,
        }

    _t_gen0 = time.perf_counter()
    text = groq_client.generate(
        system_prompt=system_prompt, user_prompt=user_prompt,
        model=model, temperature=temperature, max_tokens=max_tokens,
    )
    generation_time = time.perf_counter() - _t_gen0
    return _finish(text, generation_time)