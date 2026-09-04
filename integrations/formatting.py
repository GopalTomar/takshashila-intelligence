"""
formatting.py — Mattermost presentation layer for the Takshashila RAG bot.

This module turns a RAG result dict (produced by the unchanged
``src.rag_pipeline.answer()``) into a clean, professional Mattermost message.
It holds NO retrieval logic and never mutates the pipeline.

Layout produced by ``format_answer``
------------------------------------
    🏛️ **Takshashila Knowledge Assistant**

    ### 📌 Question
    <question>

    ### 🎯 Quick Summary
    <concise answer prose>

    ### 📖 Detailed Answer        (omitted in "short" mode / when absent)
    * key point
    * key point

    ### 📚 Sources                (source "cards", top-3, de-duplicated)
    **1.** 📄 <title>
    🏷️ <category> · [🔗 Open Document](<url>)

    ### 📊 Answer Metadata
    ⚡ Response Time: 1.5s
    🎯 Confidence: 🟢 High
    📚 Sources Used: 3

Public helpers
--------------
* ``format_answer(question, result, mode, response_time)`` — full message.
* ``format_sources(sources)``  → ``(cards_block, ref_map)``  (source cards).
* ``format_confidence(confidence)`` → coloured badge string.
* ``confidence_meta(confidence)``   → "🟢 High" style label for the metrics box.
* ``displayed_sources(sources)``    → the exact source dicts shown (top-3, deduped).
* ``build_previews(sources)``       → citation-preview snippets for those sources.
* ``extract_summary(result)``       → plain-text summary (for Copy Summary).
* ``format_search_results(query, chunks, response_time)`` → no-LLM search list.
* ``parse_mode(text)``              → splits a leading mode keyword off the query.
"""

from __future__ import annotations

import re
from typing import Dict, List, Optional, Tuple

from src.utils import linkify_citations

try:
    from src import config as _config
except Exception:  # pragma: no cover - allows isolated unit testing
    _config = None


# ════════════════════════════════════════════════════════════════════════════
#  Tunables
# ════════════════════════════════════════════════════════════════════════════

MAX_SOURCES = 3          # source cards shown under an answer
MAX_KEY_POINTS = 4       # bullets in the Detailed Answer block
MAX_SEARCH_RESULTS = 8   # documents listed by search mode
PREVIEW_CHARS = 320      # length of a citation-preview snippet

_NO_EVIDENCE_MARKER = (
    "I do not have sufficient evidence in the Takshashila knowledge base"
)


# ════════════════════════════════════════════════════════════════════════════
#  Small text utilities
# ════════════════════════════════════════════════════════════════════════════

def _clean(text: Optional[str]) -> str:
    return (text or "").strip()


def _is_no_evidence(answer_text: str) -> bool:
    return answer_text.lstrip().startswith(_NO_EVIDENCE_MARKER)


def _collapse_ws(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


def _source_name(src: Dict) -> str:
    explicit = _clean(src.get("source_name"))
    if explicit:
        return explicit
    raw = _clean(src.get("source"))
    if _config is not None:
        return _config.source_display_name(raw, raw)
    return raw or "Source"


def _source_title(src: Dict) -> str:
    return _clean(src.get("title")) or "Untitled"


def _source_url(src: Dict) -> str:
    return _clean(src.get("url")) or _clean(src.get("original_url"))


def _source_category(src: Dict) -> str:
    cat = _clean(src.get("category"))
    return "" if cat in ("", "—", "-") else cat


def _source_text(src: Dict) -> str:
    return _clean(src.get("text"))


# ════════════════════════════════════════════════════════════════════════════
#  Parsing the model's self-generated scaffolding apart
# ════════════════════════════════════════════════════════════════════════════
#
# The pipeline prompt makes the LLM emit **Answer** / **Details** / **Sources**
# sections with inline [Source N] citations. We split those apart, keep the
# prose + detail bullets, DROP the model's own source list, and rebuild sources
# ourselves from the structured chunk list so nothing is duplicated or invented.

_SECTION_RE = re.compile(
    r"^\s*(?:\*\*|#+\s*)?(answer|details|key\s*points|sources?|references?)\b[:\*]*\s*$",
    re.IGNORECASE,
)


def _split_sections(answer_text: str) -> Dict[str, str]:
    sections: Dict[str, List[str]] = {"answer": []}
    current = "answer"
    for line in answer_text.splitlines():
        m = _SECTION_RE.match(line)
        if m:
            name = m.group(1).lower().replace(" ", "")
            if name.startswith("source"):
                current = "sources"
            elif name.startswith("reference"):
                current = "references"
            elif name in ("details", "keypoints"):
                current = "details"
            else:
                current = "answer"
            sections.setdefault(current, [])
            continue
        sections.setdefault(current, []).append(line)
    return {k: "\n".join(v).strip() for k, v in sections.items()}


def _extract_bullets(block: str) -> List[str]:
    bullets: List[str] = []
    for line in block.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        m = re.match(r"^(?:[-*•]|\d+[.)])\s+(.*)$", stripped)
        if m and m.group(1).strip():
            bullets.append(m.group(1).strip())
    return bullets


def _renumber_refs(text: str, mapping: Dict[int, int]) -> str:
    """
    Rewrite the model's ``[Source N]``/``[N]`` citations to displayed indices.

    Done in ONE pass: rewriting ``[Source 3]`` to ``[2]`` and then running a second
    ``[N]`` pass would re-map the freshly written ``[2]`` against the original
    numbering and could delete it. A marker with no displayed source (i.e. the
    answer cited something that isn't shown) is removed rather than left dangling.
    """
    def repl(match: "re.Match") -> str:
        shown = mapping.get(int(match.group(1)))
        return f"[{shown}]" if shown else ""
    text = re.sub(r"\[\s*(?:Source\s*)?(\d+)\s*\]", repl, text, flags=re.IGNORECASE)
    text = re.sub(r"[ \t]{2,}", " ", text)
    text = re.sub(r"\s+([.,;:])", r"\1", text)
    return text.strip()


def _strip_refs(text: str) -> str:
    """Remove all inline citations entirely (used for the plain copy summary)."""
    text = re.sub(r"\[\s*(?:Source\s*)?\d+\s*\]", "", text, flags=re.IGNORECASE)
    text = re.sub(r"[ \t]{2,}", " ", text)
    text = re.sub(r"\s+([.,;:])", r"\1", text)
    return text.strip()


# ════════════════════════════════════════════════════════════════════════════
#  Source selection + de-duplication
# ════════════════════════════════════════════════════════════════════════════

def _dedupe_sources(sources: List[Dict]) -> Tuple[List[Dict], Dict[int, int]]:
    """
    De-duplicate by (title, url) and return ``(deduped, original_idx → deduped_idx)``
    so inline citations survive de-duplication (two chunks of one document collapse
    to a single source, and both markers point at it).
    """
    seen: Dict[Tuple[str, str], int] = {}
    deduped: List[Dict] = []
    orig_to_dedupe: Dict[int, int] = {}
    for original_idx, s in enumerate(sources or [], 1):
        key = (_source_title(s).lower(), _source_url(s).lower())
        if key in seen:
            orig_to_dedupe[original_idx] = seen[key]
            continue
        deduped.append(s)
        seen[key] = len(deduped)
        orig_to_dedupe[original_idx] = len(deduped)
    return deduped, orig_to_dedupe


_MARKER_RE = re.compile(r"\[\s*(?:Source\s*)?(\d+)\s*\]", re.IGNORECASE)


def _cited_dedupe_indices(text: str, orig_to_dedupe: Dict[int, int]) -> List[int]:
    """Deduped source indices actually cited in ``text``, in first-seen order."""
    out: List[int] = []
    for m in _MARKER_RE.finditer(text or ""):
        d = orig_to_dedupe.get(int(m.group(1)))
        if d and d not in out:
            out.append(d)
    return out


def select_cited_sources(answer_text: str,
                         sources: List[Dict]) -> Tuple[List[Dict], Dict[int, int]]:
    """
    Choose the sources to display **from the citations the rendered answer actually
    contains** — never a blind "top N" of what retrieval returned.

    This is what makes the source list trustworthy: every displayed source is one
    the answer cited, and every visible ``[N]`` marker has a matching source (so a
    document can't appear that the answer never used, and a marker can't dangle).
    Because the number of displayed sources follows the citations, asking the same
    question twice yields a list that always matches that answer's own reasoning.

    Returns ``(shown_sources, original_idx → displayed_idx)``.
    """
    deduped, orig_to_dedupe = _dedupe_sources(sources or [])
    if not deduped:
        return [], {}

    cited = _cited_dedupe_indices(answer_text, orig_to_dedupe)
    if not cited:
        # The model produced no usable markers; the pipeline has already ground-
        # checked the answer, so attribute it to the single best-matching source
        # rather than listing several the answer never pointed at.
        cited = [1]

    shown = [deduped[d - 1] for d in cited]
    dedupe_to_new = {d: new for new, d in enumerate(cited, 1)}
    ref_map = {orig: dedupe_to_new[d]
               for orig, d in orig_to_dedupe.items() if d in dedupe_to_new}
    return shown, ref_map


def _select_sources(sources: List[Dict]) -> Tuple[List[Dict], Dict[int, int]]:
    """Backward-compatible shim: selection without an answer to inspect."""
    return select_cited_sources("", sources)


def displayed_sources(sources: List[Dict], answer_text: str = "") -> List[Dict]:
    """
    The exact source dicts shown to the user. When ``answer_text`` is supplied the
    list is restricted to the sources that answer cites.
    """
    return select_cited_sources(answer_text, sources or [])[0]


# ════════════════════════════════════════════════════════════════════════════
#  Required helper #1: format_confidence  (+ metadata label)
# ════════════════════════════════════════════════════════════════════════════

_CONF_BADGE = {
    "high": "🟢 High Confidence",
    "medium": "🟡 Medium Confidence",
    "low": "🔴 Low Confidence",
}
_CONF_META = {"high": "🟢 High", "medium": "🟡 Medium", "low": "🔴 Low"}


def format_confidence(confidence: Optional[str]) -> str:
    """Map a confidence tier to a coloured badge (empty for none/unknown)."""
    return _CONF_BADGE.get(_clean(confidence).lower(), "")


def confidence_meta(confidence: Optional[str]) -> str:
    """Compact confidence label for the Answer Metadata box, e.g. '🟢 High'."""
    return _CONF_META.get(_clean(confidence).lower(), "—")


# ════════════════════════════════════════════════════════════════════════════
#  Required helper #2: format_sources  → source CARDS
# ════════════════════════════════════════════════════════════════════════════

def format_sources(sources: List[Dict],
                   answer_text: str = "") -> Tuple[str, Dict[int, int]]:
    """
    Build the Sources block as readable "cards" and return ``(cards, ref_map)``.

    When ``answer_text`` is given, only the sources that answer actually cites are
    shown, so the list can never contain a document the answer never used.

    Each card surfaces only user-safe fields — title, category and a clickable
    "Open Document" link. Chunk ids, document ids, embeddings, cosine / RRF
    scores and every other retrieval internal are deliberately omitted.

        **1.** 📄 Key Policies
        🏷️ Decision · [🔗 Open Document](https://…)
    """
    shown, ref_map = select_cited_sources(answer_text, sources or [])
    if not shown:
        return "", {}

    cards: List[str] = []
    for i, src in enumerate(shown, 1):
        title = _source_title(src)
        category = _source_category(src)
        url = _source_url(src)

        card = [f"**{i}.** 📄 {title}"]
        meta_bits = []
        if category:
            meta_bits.append(f"🏷️ {category}")
        if url:
            meta_bits.append(f"[🔗 Open Document]({url})")
        if meta_bits:
            card.append(" · ".join(meta_bits))
        cards.append("\n".join(card))

    # Blank line between cards for readable separation in Mattermost.
    return "\n\n".join(cards), ref_map


# ════════════════════════════════════════════════════════════════════════════
#  Citation previews  +  plain-text summary
# ════════════════════════════════════════════════════════════════════════════

def build_previews(sources: List[Dict]) -> List[Dict]:
    """
    Snippet previews for the displayed sources (used by the 🔍 Preview buttons).
    Returns ``[{"title", "snippet", "url"}, …]`` in displayed order.
    """
    previews = []
    for src in displayed_sources(sources):
        snippet = _collapse_ws(_source_text(src))
        if len(snippet) > PREVIEW_CHARS:
            snippet = snippet[:PREVIEW_CHARS].rstrip() + "…"
        previews.append({
            "title": _source_title(src),
            "snippet": snippet or "(No excerpt available for this source.)",
            "url": _source_url(src),
        })
    return previews


def extract_summary(result: Dict) -> str:
    """Plain-text concise summary (no markdown headings, no [N] citations)."""
    raw = _clean(result.get("answer"))
    if _is_no_evidence(raw):
        return _collapse_ws(raw)
    body = _split_sections(raw).get("answer", raw)
    return _strip_refs(body) or _collapse_ws(raw)


# ════════════════════════════════════════════════════════════════════════════
#  Detailed Answer (key points)
# ════════════════════════════════════════════════════════════════════════════

def _format_detailed(details_block: str, ref_map: Dict[int, int]) -> str:
    bullets = _extract_bullets(details_block)[:MAX_KEY_POINTS]
    if not bullets:
        return ""
    cleaned = [_renumber_refs(b, ref_map) for b in bullets]
    return "\n".join(f"* {b}" for b in cleaned if b)


# ════════════════════════════════════════════════════════════════════════════
#  Answer Metadata box
# ════════════════════════════════════════════════════════════════════════════

def _metadata_block(response_time: float, confidence: str, n_sources: int,
                    include_confidence: bool = True) -> str:
    lines = ["### 📊 Answer Metadata", "", f"⚡ Response Time: {response_time:.1f}s"]
    if include_confidence:
        lines.append(f"🎯 Confidence: {confidence_meta(confidence)}")
        lines.append(f"📚 Sources Used: {n_sources}")
    return "\n".join(lines)


# ════════════════════════════════════════════════════════════════════════════
#  Required helper #3: format_answer
# ════════════════════════════════════════════════════════════════════════════

def format_answer(
    question: str,
    result: Dict,
    mode: str = "normal",
    response_time: Optional[float] = None,
) -> str:
    """
    Build the final Mattermost markdown message.

    ``mode``:
      * "short"  — Quick Summary only (no Detailed Answer block).
      * "normal" — Quick Summary + Detailed Answer (default).
      * "detailed" — same as normal here (extra depth comes via follow-up buttons).
    """
    raw_answer = _clean(result.get("answer"))
    confidence = _clean(result.get("confidence")) or "none"

    if response_time is None:
        response_time = (
            float(result.get("retrieval_time") or 0.0)
            + float(result.get("generation_time") or 0.0)
        )

    if _is_no_evidence(raw_answer) or confidence == "none":
        return _render_no_evidence(question, raw_answer, response_time)

    sources = result.get("sources") or []
    sections = _split_sections(raw_answer)

    # Render the answer body FIRST (with the model's original numbering), because
    # the Detailed Answer is capped at MAX_KEY_POINTS bullets. Only the citations
    # that survive into the *visible* text may contribute a source — otherwise a
    # trimmed bullet would leave an orphan document in the Sources list.
    summary_raw = sections.get("answer", raw_answer)
    bullets_raw = _extract_bullets(sections.get("details", ""))[:MAX_KEY_POINTS]
    if mode == "short":
        bullets_raw = []
    visible_raw = summary_raw + "\n" + "\n".join(bullets_raw)

    # Sources = exactly what the visible answer cites, renumbered 1..N in order.
    cards_block, ref_map = format_sources(sources, visible_raw)
    shown = displayed_sources(sources, visible_raw)

    summary = _renumber_refs(summary_raw, ref_map)
    detailed = "\n".join(
        f"* {b}" for b in (_renumber_refs(x, ref_map) for x in bullets_raw) if b
    )

    # Make each inline [N] marker a link to that source's document, so a reader can
    # jump straight to where a claim came from. Markers are already renumbered to
    # match `shown`, so the Nth marker maps to shown[N-1]. Markers whose source has
    # no URL are left as plain text rather than becoming a broken link.
    summary = linkify_citations(summary, shown)
    detailed = linkify_citations(detailed, shown)

    lines: List[str] = ["🏛️ **Takshashila Knowledge Assistant**", ""]
    lines += ["### 📌 Question", "", question, ""]
    lines += ["### 🎯 Quick Summary", "", summary or "_No answer was produced._", ""]

    if mode != "short" and detailed:
        lines += ["### 📖 Detailed Answer", "", detailed, ""]

    if cards_block:
        lines += ["### 📚 Sources", "", cards_block, ""]

    lines += [_metadata_block(response_time, confidence, len(shown))]
    return "\n".join(lines).strip()


def _render_no_evidence(question: str, answer_text: str, response_time: float) -> str:
    lines = [
        "🏛️ **Takshashila Knowledge Assistant**", "",
        "### 📌 Question", "", question, "",
        "### 🎯 Quick Summary", "",
        answer_text or "_No answer was produced._", "",
        _metadata_block(response_time, "none", 0, include_confidence=False),
    ]
    return "\n".join(lines).strip()


# ════════════════════════════════════════════════════════════════════════════
#  Search mode (no LLM answer — just matching documents)
# ════════════════════════════════════════════════════════════════════════════

def format_search_results(query: str, chunks: List[Dict],
                          response_time: float = 0.0, kind: str = "search") -> str:
    """
    Render retrieved documents as a clean list, with no generated answer.

    ``kind``:
      * "search"  — the ``/askkb search`` command (broad recall).
      * "related" — the 🔗 Related Policies button. The caller has already filtered
                    to genuinely relevant documents, so the labels reflect that and
                    the empty state is worded as "no closely-related documents".
    """
    related = (kind == "related")
    heading = ("🔗 **Related Policies**" if related
               else "🔍 **Takshashila Knowledge Base — Search**")

    seen, docs = set(), []
    for ch in chunks or []:
        key = (_source_title(ch).lower(), _source_url(ch).lower())
        if key in seen:
            continue
        seen.add(key)
        docs.append(ch)

    if not docs:
        if related:
            return (
                f"{heading}\n\n"
                "This answer did not draw on any specific documents, so there are "
                "no source references to show."
            )
        return (
            f"{heading}\n\n"
            f"No matching documents found for: _{query}_\n\n"
            "Try different or Takshashila-specific keywords."
        )

    listed = docs[:MAX_SEARCH_RESULTS]
    intro = ("The document(s) this answer was drawn from:" if related
             else f"Found **{len(docs)}** matching document(s) for: _{query}_")
    lines = [heading, "", intro, ""]
    for i, ch in enumerate(listed, 1):
        title = _source_title(ch)
        category = _source_category(ch)
        url = _source_url(ch)
        head = f"**{i}.** 📄 {title}"
        if category:
            head += f"  🏷️ {category}"
        lines.append(head)
        if url:
            lines.append(f"[🔗 Open Document]({url})")
        lines.append("")

    if related:
        lines += ["---", "_These are the exact source documents this answer is "
                         "based on — not a broader search._"]
    else:
        lines += ["---", f"⚡ Search Time: {response_time:.1f}s · No AI answer generated (search mode)"]
    return "\n".join(lines).strip()


# ════════════════════════════════════════════════════════════════════════════
#  Length / search mode parsing for the slash-command text
# ════════════════════════════════════════════════════════════════════════════

def parse_mode(text: str) -> Tuple[str, str]:
    """
    Split a leading mode keyword off the slash-command text.

      "short What is POSH?"     → ("short",    "What is POSH?")
      "detailed leave policy"   → ("detailed", "leave policy")
      "search laptop policy"    → ("search",   "laptop policy")
      "What is POSH?"           → ("normal",   "What is POSH?")
    """
    stripped = _clean(text)
    m = re.match(r"^(short|detailed|normal|search)\b\s*(.*)$", stripped, re.IGNORECASE)
    if m and m.group(2).strip():
        return m.group(1).lower(), m.group(2).strip()
    return "normal", stripped