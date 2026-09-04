"""
utils.py — Shared utility helpers for Takshashila RAG
"""

import hashlib
import json
import logging
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional
from urllib.parse import urljoin, urlparse


# ── Logging ────────────────────────────────────────────────────────────────────

def get_logger(name: str, log_file: Optional[Path] = None) -> logging.Logger:
    """Return a logger that writes to stdout and optionally a file."""
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger  # already configured

    logger.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
                            datefmt="%Y-%m-%d %H:%M:%S")

    sh = logging.StreamHandler()
    sh.setFormatter(fmt)
    logger.addHandler(sh)

    if log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(log_file, encoding="utf-8")
        fh.setFormatter(fmt)
        logger.addHandler(fh)

    return logger


# ── Hashing ────────────────────────────────────────────────────────────────────

def url_hash(url: str) -> str:
    return hashlib.sha256(url.strip().encode()).hexdigest()[:16]


def content_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", errors="replace")).hexdigest()[:24]


# ── URL helpers ────────────────────────────────────────────────────────────────

def normalize_url(url: str, base: str = "") -> str:
    url = url.strip()
    if url.startswith("//"):
        url = "https:" + url
    if base and not url.startswith("http"):
        url = urljoin(base, url)
    return url


# Audit finding: is_same_domain(url, domain) — a raw substring check
# ("domain in urlparse(url).netloc") — used to live here. It was only ever
# called from src/scraper.py (a fully unused, unimported duplicate crawler,
# since deleted) and from scripts/crawl_engine.py's PDF-link discovery (now
# fixed to use CrawlEngine._is_internal() instead, an exact hostname-set
# match). Substring matching is a real footgun for domain scoping: it treats
# "legion.takshashila.org.in" — and even a crafted hostname like
# "takshashila.org.in.evil.example" — as "same domain" as
# "takshashila.org.in". Removed rather than left as an unused, misleading
# function future code might reach for.


# Query parameters that never change the underlying page and must be stripped
# before a URL is used as a dedup/identity key — mainly analytics/tracking and
# WordPress comment-reply noise. Anything not in this list is KEPT (some sites
# genuinely use query params for real pagination/filtering, e.g. ?page=2).
_TRACKING_PARAMS = {
    "utm_source", "utm_medium", "utm_campaign", "utm_term", "utm_content",
    "fbclid", "gclid", "gclsrc", "mc_cid", "mc_eid", "ref", "share",
    "replytocom", "amp",
}


def canonicalize_for_dedup(url: str) -> str:
    """
    Collapse cosmetic URL variants that all resolve to the *same* page into one
    identity key, so the crawler's seen-set / frontier / url_hash never treats
    them as distinct pages:

      * scheme + host lowercased, ``www.`` prefix dropped
      * fragment (``#...``) removed — never a distinct page
      * known tracking query params removed (utm_*, fbclid, ref, share, ...);
        remaining params are kept but sorted so ordering doesn't matter
      * trailing slash removed (except for the bare root ``/``)

    This does NOT change what URL is actually fetched or displayed to a user —
    callers use it only as a comparison/hash key (see ``url_hash`` callers in
    scripts/crawl_engine.py), so a page whose canonical form differs from its
    on-the-wire form is still requested exactly as linked.
    """
    try:
        parts = urlparse(url.strip())
    except Exception:
        return url.strip()

    scheme = (parts.scheme or "https").lower()
    netloc = parts.netloc.lower()
    if netloc.startswith("www."):
        netloc = netloc[4:]

    path = parts.path or "/"
    if len(path) > 1 and path.endswith("/"):
        path = path.rstrip("/")

    if parts.query:
        from urllib.parse import parse_qsl, urlencode
        kept = [(k, v) for k, v in parse_qsl(parts.query, keep_blank_values=True)
                if k.lower() not in _TRACKING_PARAMS]
        kept.sort()
        query = urlencode(kept)
    else:
        query = ""

    out = f"{scheme}://{netloc}{path}"
    if query:
        out += f"?{query}"
    return out


# ── Source-quality predicates (single source of truth) ──────────────────────────
# Used by BOTH the retriever (so listing/nav pages never consume a top-k slot)
# and the RAG pipeline (defence in depth), so an answer is always cited to the
# specific article — never to an aggregate index/landing page whose link would
# only send the reader to a list.

# Author / tag / category / paginated-archive URLs.
_NAV_URL_RE = re.compile(
    r"/(?:author|authors|people|team|staff|profile|tag|tags|category|categories|"
    r"topic|topics|archive|archives|search|page)/",
    re.IGNORECASE,
)
_NAV_TITLE_RE = re.compile(
    r"^(?:tag|category|topic|archive|author|search results?)\s*[:\u2013\u2014-]",
    re.IGNORECASE,
)
# "Pranay Kotasthane – Takshashila Institution" — a bare person-name page title.
_PERSON_TITLE_RE = re.compile(
    r"^[A-Z][\w.'-]+(?:\s+[A-Z][\w.'-]+){0,3}\s*[\u2013\u2014-]\s*"
    r"Takshashila(?:\s+Institution)?\s*$"
)
# A section/index URL with nothing after the section name (a listing), e.g.
# /pages/blogs/, /blogs/, /research/ — but NOT /blogs/some-post (an article).
_LISTING_URL_RE = re.compile(
    r"/(?:pages/)?(?:publications?|blogs?|articles?|research|commentary|reports?|"
    r"papers?|briefs?|events?|news|media|content|all[-_]?content|home)/?$",
    re.IGNORECASE,
)
# A title that is *only* a section or the site name (a landing page).
_GENERIC_TITLE_RE = re.compile(
    r"^(?:home|homepage|blogs?|publications?|articles?|research|commentary|"
    r"reports?|papers?|briefs?|events?|news|media|"
    r"takshashila(?:\s+institution)?)$",
    re.IGNORECASE,
)


def is_listing_or_landing(url: str = "", title: str = "") -> bool:
    """True for an aggregate index / section-landing / homepage (not evidence)."""
    u = (url or "").strip()
    if u:
        try:
            path = urlparse(u).path
        except Exception:
            path = u
        if _LISTING_URL_RE.search((path or "/").rstrip("/") + "/"):
            return True
        if path.rstrip("/") in ("", "/index", "/index.html", "/home"):
            return True
    t = (title or "").strip()
    if t and _GENERIC_TITLE_RE.match(t):
        return True
    return False


def is_low_value_source(url: str = "", title: str = "", text_len: int = 9999,
                        min_evidence_chars: int = 200) -> bool:
    """
    True when a page is navigational / author / listing / landing / too-thin —
    i.e. it can't serve as the citation for a specific claim. Callers use this to
    keep such pages out of retrieved evidence so references point to real
    articles with their exact URLs.
    """
    url = (url or "").strip()
    title = (title or "").strip()
    if url and _NAV_URL_RE.search(url):
        return True
    if title and (_NAV_TITLE_RE.search(title) or _PERSON_TITLE_RE.match(title)):
        return True
    if is_listing_or_landing(url, title):
        return True
    if text_len < min_evidence_chars:
        return True
    return False


def chunk_is_low_value(ch: Dict, min_evidence_chars: int = 200) -> bool:
    """Convenience wrapper of is_low_value_source for a chunk/document dict."""
    return is_low_value_source(
        url=(ch.get("url") or ch.get("original_url") or ""),
        title=(ch.get("title") or ""),
        text_len=len((ch.get("text") or "").strip()),
        min_evidence_chars=min_evidence_chars,
    )


def build_meta_header(meta: Dict) -> str:
    """
    Build a compact, human-readable metadata header for a chunk/document.

    This header is prepended to the chunk body when embedding and BM25-indexing
    (via ``chunk_search_text``), so metadata questions — "who wrote this?",
    "when was it published?", "what are the tags?", "which category?" — become
    *retrievable* and the answer's supporting chunk actually carries the author,
    date, section and tags. It is NOT shown to the user (the body is), but the
    same facts are also surfaced in the LLM context header by the pipeline.
    """
    def _fmt_authors(m: Dict) -> str:
        a = m.get("authors")
        if isinstance(a, (list, tuple)) and a:
            return ", ".join(str(x).strip() for x in a if str(x).strip())
        return str(m.get("author") or "").strip()

    def _fmt_tags(m: Dict) -> str:
        t = m.get("tags")
        if isinstance(t, (list, tuple)):
            return ", ".join(str(x).strip() for x in t if str(x).strip())
        return str(t or "").strip()

    lines = []
    title = str(meta.get("title") or "").strip()
    if title and title.lower() != "untitled":
        lines.append(f"Title: {title}")
    if str(meta.get("subtitle") or "").strip():
        lines.append(f"Subtitle: {meta['subtitle'].strip()}")
    authors = _fmt_authors(meta)
    if authors:
        lines.append(f"Author: {authors}")
    if str(meta.get("date") or "").strip():
        lines.append(f"Published: {str(meta['date']).strip()}")
    if str(meta.get("updated_date") or "").strip():
        lines.append(f"Updated: {str(meta['updated_date']).strip()}")
    if str(meta.get("category") or "").strip():
        lines.append(f"Category: {meta['category'].strip()}")
    section = str(meta.get("heading_path") or meta.get("section") or "").strip()
    if section:
        lines.append(f"Section: {section}")
    tags = _fmt_tags(meta)
    if tags:
        lines.append(f"Tags: {tags}")
    if str(meta.get("document_type") or "").strip():
        lines.append(f"Type: {meta['document_type'].strip()}")
    if str(meta.get("source_name") or "").strip():
        lines.append(f"Source: {meta['source_name'].strip()}")
    return "\n".join(lines)


def chunk_search_text(ch: Dict) -> str:
    """
    The text used for EMBEDDING and BM25 (metadata header + chunk body).

    Prefers a precomputed ``meta_header`` on the chunk; otherwise derives it. The
    body (``text``) is what gets displayed to the user — the header only makes the
    metadata searchable so metadata questions resolve to the right document.
    """
    header = (ch.get("meta_header") or "").strip()
    if not header:
        header = build_meta_header(ch)
    body = (ch.get("text") or "").strip()
    return f"{header}\n\n{body}".strip() if header else body


def looks_like_pdf(url: str) -> bool:
    return url.lower().split("?")[0].endswith(".pdf")


# ── Text helpers ───────────────────────────────────────────────────────────────

def clean_text(text: str) -> str:
    """Remove excessive whitespace while preserving paragraph breaks."""
    text = re.sub(r"\r\n", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]+", " ", text)
    return text.strip()


# ── Mojibake / encoding repair + text-quality assessment ─────────────────────────
#
# Website/publication pages and PDF text extraction produced two DISTINCT problems:
#
#   1. TRUE MOJIBAKE — UTF-8 bytes mis-decoded as Windows-1252/Latin-1, e.g.
#      "Indiaâ€™s" (should be "India's"), "JPÂ¥43" (should be "JP¥43"),
#      "â€œquotedâ€" (should be "“quoted”"). This is REPAIRABLE (ftfy / round-trip).
#
#   2. OCR / EXTRACTION GARBAGE — unrecoverable noise from bad PDF text layers, e.g.
#      "®ØÙÚÛÜ" or "Õ6ÖiÀÁ¬É ÅרÙÚÛmÇ×iÜÓ2ÝÞßà". This CANNOT be repaired and must be
#      dropped / quarantined.
#
# Crucially, VALID accented words (Alcântara, Grâce, Duchâtel, Boétie, décideurs,
# réseau) must NOT be treated as either problem. Detection therefore keys on
# SPECIFIC broken byte-pair sequences — never on a lone accented letter.

# Specific two/three-character sequences that only occur in real mojibake.
# (Used both to FIX and to DETECT. A lone "â"/"Ã"/"Â" is intentionally absent.)
_FIX_MAP = {
    # ── smart punctuation (UTF-8 read as cp1252) ──
    "â€™": "\u2019", "â€˜": "\u2018", "â€œ": "\u201c", "â€\x9d": "\u201d",
    "â€”": "\u2014", "â€“": "\u2013", "â€¦": "\u2026", "â€¢": "\u2022",
    "â€\u009d": "\u201d", "â€?": "\u2019", "â€": "\u201d",
    # ── smart punctuation (UTF-8 3-byte E2 80 xx read as Latin-1 → â + C1 byte) ──
    # e.g. the em dash "—" becomes "â\x80\x94". These forms use raw C1 control
    # bytes (\x80–\x9f) rather than the cp1252 printable glyphs above, so they
    # need their own entries or they slip through undetected.
    "â\x80\x93": "\u2013", "â\x80\x94": "\u2014",          # en / em dash
    "â\x80\x98": "\u2018", "â\x80\x99": "\u2019",          # single quotes
    "â\x80\x9a": "\u201a", "â\x80\x9b": "\u2018",
    "â\x80\x9c": "\u201c", "â\x80\x9d": "\u201d",          # double quotes
    "â\x80\x9e": "\u201e", "â\x80\xa6": "\u2026",          # low quote / ellipsis
    "â\x80\xa2": "\u2022", "â\x80\x93 ": "\u2013 ",         # bullet
    "â\x82\xac": "\u20ac", "â\x82\xb9": "\u20b9", "â\x84\xa2": "\u2122",  # € ₹ ™
    # ── currency / trademark ──
    "â‚¹": "\u20b9", "â‚¬": "\u20ac", "â„¢": "\u2122",
    # ── "Â" + symbol (NOT Â+letter, which can be valid) ──
    "Â£": "£", "Â©": "©", "Â®": "®", "Â°": "°", "Â±": "±", "Â½": "½", "Â¼": "¼",
    "Â¾": "¾", "Â·": "·", "Â»": "»", "Â«": "«", "Â¥": "¥", "Â¢": "¢", "Â§": "§",
    "Â¶": "¶", "Â´": "´", "Â¨": "¨", "Â¯": "¯", "Â¸": "¸", "Â¡": "¡", "Â¿": "¿",
    "Â²": "²", "Â³": "³", "Â¹": "¹", "Âµ": "µ", "Âª": "ª", "Âº": "º",
    # ── "Ã" + continuation → accented letters ──
    "Ã©": "é", "Ã¨": "è", "Ã«": "ë", "Ãª": "ê", "Ã¡": "á", "Ã ": "à", "Ã¢": "â",
    "Ã£": "ã", "Ã¥": "å", "Ã¤": "ä", "Ã¦": "æ", "Ã³": "ó", "Ã²": "ò", "Ã´": "ô",
    "Ã¶": "ö", "Ãµ": "õ", "Ã¸": "ø", "Ã±": "ñ", "Ã­": "í", "Ã¬": "ì", "Ã¯": "ï",
    "Ã®": "î", "Ãº": "ú", "Ã¹": "ù", "Ã¼": "ü", "Ã»": "û", "Ã§": "ç", "Ã¿": "ÿ",
    "Ã½": "ý", "Ã€": "À", "Ã‰": "É", "Ã‡": "Ç", "Ã‘": "Ñ", "Ã„": "Ä", "Ã–": "Ö",
    "Ãœ": "Ü", "Ã…": "Å", "Ã†": "Æ", "Ã˜": "Ø", "ÃŸ": "ß",
    # ── non-breaking space mojibake ──
    "Â\xa0": " ", "Â ": " ", "\xa0": " ",
    # ── mis-encoded Unicode replacement char (EF BF BD as cp1252) ──
    "ï¿½": "",
}

# Regex of the same true-mojibake sequences, plus emoji-mojibake and the
# replacement character. This is the AUTHORITATIVE "real mojibake" detector.
_TRUE_MOJIBAKE_RE = re.compile(
    "(" + "|".join(re.escape(k) for k in _FIX_MAP if k.strip() and k not in (" ",)) + ")"
    + r"|ðŸ|\ufffd"
)

# Backwards-compatible marker tuple (kept for any external import).
MOJIBAKE_MARKERS = tuple(k for k in _FIX_MAP if k.strip()) + ("ðŸ", "\ufffd")

# Leftover emoji-mojibake the round-trip could not reconstruct (partial bytes).
_EMOJI_LEFTOVER_RE = re.compile(r"ðŸ[^\sA-Za-z0-9]{0,4}")
# A lone "â" standing in for an en/em dash, e.g. "Report â Enhancing".
_LONE_A_DASH_RE = re.compile(r"(?<=\s)â(?=\s)")

# Deterministic pre-pass for the C1-control punctuation forms (â\x80\x9c, â\x80\x94,
# …). These must always map to the SAME target characters regardless of whether
# ftfy is installed, so both repair paths agree. Applied BEFORE ftfy below.
_C1_PUNCT_FIX = {k: v for k, v in _FIX_MAP.items()
                 if len(k) >= 2 and k[0] == "â" and 0x80 <= ord(k[1]) <= 0x9f}


def _apply_c1_punct(s: str) -> str:
    for bad, good in _C1_PUNCT_FIX.items():
        if bad in s:
            s = s.replace(bad, good)
    return s


# Lone-"â" mojibake: a bare "â" standing in for an apostrophe or dash whose UTF-8
# continuation byte was lost — e.g. "Takshashilaâs" → "Takshashila’s", and
# "â What We Decided" → "– What We Decided". Detected SEPARATELY from the core
# detector so a genuine accented "â" (Alcântara, Grâce, château) is never touched:
# we only match "â" before an English contraction ending, standalone between
# spaces / at a boundary, or immediately followed by replacement/box glyphs.
_LONE_A_MOJIBAKE_RE = re.compile(
    r"â(?=(?:s|t|re|ll|ve|d|m)\b)"           # apostrophe: Takshashilaâs, doesnât, weâre
    r"|(?:(?<=\s)|^)â(?=\s|$)"                # standalone dash: "word â word", "â word"
    r"|â[\ufffd\u2500-\u259f]+"               # â followed by replacement/box characters
)


def _repair_lone_a(s: str) -> str:
    """Deterministically repair the lone-'â' mojibake forms (see the regex above)."""
    s = re.sub(r"â[\ufffd\u2500-\u259f]+", "\u2013", s)          # "â□□" → en dash
    s = re.sub(r"â(?=(?:s|t|re|ll|ve|d|m)\b)", "\u2019", s)      # contraction apostrophe
    s = re.sub(r"(?:(?<=\s)|^)â(?=\s|$)", "\u2013", s)           # standalone en dash
    return s

# Characters that should essentially never appear in this (English/Indian policy)
# corpus — strong signals of extraction garbage. Excludes normal accented Latin,
# Devanagari, and common typographic punctuation.
_HARD_NOISE_RE = re.compile(
    "["
    "\u0000-\u0008\u000b\u000c\u000e-\u001f"   # control chars
    "\u0080-\u009f"                              # C1 controls
    "\u0250-\u02af"                              # IPA extensions
    "\u0300-\u036f"                              # combining marks (stray)
    "\u0370-\u03ff\u0400-\u04ff"                 # Greek / Cyrillic
    "\u0530-\u05ff\u0600-\u06ff"                 # Armenian / Hebrew / Arabic
    "\u2500-\u259f"                              # box drawing / blocks
    "\ue000-\uf8ff"                              # private use
    "\ufffd"                                     # replacement char
    "]"
)

# "Valid" letters we explicitly allow: ASCII, Latin-1 accented letters, Latin
# Extended-A, and Devanagari. Anything alphabetic outside this is "exotic".
def _is_valid_letter(c: str) -> bool:
    o = ord(c)
    return (
        ("a" <= c <= "z") or ("A" <= c <= "Z")
        or (0x00C0 <= o <= 0x00FF and o not in (0x00D7, 0x00F7))  # accented Latin-1
        or (0x0100 <= o <= 0x017F)                                # Latin Extended-A
        or (0x0900 <= o <= 0x097F)                                # Devanagari
    )

_ASCII_VOWEL_RE = re.compile(r"[aeiouAEIOU]")

# A chunk whose quality score is below this is treated as unrecoverable garbage
# and is skipped (and logged) rather than embedded into FAISS.
CHUNK_QUALITY_MIN = 0.5


def _marker_count(s: str) -> int:
    """Count bare high-Latin lead bytes — used only inside the guarded round-trip."""
    return sum(s.count(m) for m in ("â", "Ã", "Â", "ð", "\ufffd"))


def _manual_fix(s: str) -> str:
    """Deterministic mojibake repair used when ftfy is unavailable (and as backup)."""
    for _ in range(3):
        before = s
        for bad, good in _FIX_MAP.items():
            if bad in s:
                s = s.replace(bad, good)
        # Guarded round-trip: reconstruct accents/emoji from a fully-mojibake run.
        # Self-guards because a valid accented letter followed by an ASCII letter
        # is NOT a valid UTF-8 multibyte sequence and raises (so "Alcântara" is safe).
        if _marker_count(s):
            try:
                cand = s.encode("cp1252").decode("utf-8")
                if _marker_count(cand) < _marker_count(s) and "\ufffd" not in cand:
                    s = cand
            except (UnicodeEncodeError, UnicodeDecodeError):
                pass
        if s == before:
            break
    s = _EMOJI_LEFTOVER_RE.sub("", s)
    s = _repair_lone_a(s)
    s = s.replace("\ufffd", "")
    return s


def is_true_mojibake_present(text: Any) -> bool:
    """
    True only for REAL, repairable mojibake (specific broken byte-pair sequences).
    Valid accented words (Alcântara, Grâce, Duchâtel, Boétie) return False.
    """
    if not isinstance(text, str):
        if text is None or isinstance(text, (int, float, bool)):
            return False
        text = str(text)
    return (_TRUE_MOJIBAKE_RE.search(text) is not None
            or _LONE_A_MOJIBAKE_RE.search(text) is not None)


# Backwards-compatible alias. Everything that used the old (over-broad) detector
# now uses the strict one, which fixes the false positives on accented words.
def is_mojibake_present(text: Any) -> bool:
    return is_true_mojibake_present(text)


def _is_garbage_token(tok: str) -> bool:
    """Heuristic: is this whitespace-delimited token unrecoverable OCR garbage?"""
    if not tok:
        return False
    if _HARD_NOISE_RE.search(tok):
        return True
    letters = [c for c in tok if c.isalpha()]
    if not letters:
        return False  # pure punctuation/numbers/symbols tokens are not "garbage"
    n = len(tok)
    non_ascii = sum(1 for c in tok if ord(c) > 0x7F)
    has_vowel = bool(_ASCII_VOWEL_RE.search(tok))
    # 1) A run of 4+ consecutive non-ASCII letters (e.g. "ØÙÚÛÜ", "ÝÞßà").
    run = 0
    for c in tok:
        if c.isalpha() and ord(c) > 0x7F:
            run += 1
            if run >= 4:
                return True
        else:
            run = 0
    # 2) A medium token that is mostly non-ASCII and has no ASCII vowel.
    if n >= 5 and (non_ascii / n) >= 0.4 and not has_vowel:
        return True
    # 3) A longer token dominated by non-ASCII chars (mixed letters/symbols/digits).
    if n >= 6 and (non_ascii / n) >= 0.5:
        return True
    return False


def get_text_quality_score(text: Any) -> float:
    """
    Return a 0.0–1.0 quality score. 1.0 = clean; low = dominated by OCR garbage.
    Valid accented / Devanagari text scores high; symbol/noise runs score low.
    """
    if not isinstance(text, str) or not text.strip():
        return 1.0
    toks = text.split()
    if not toks:
        return 1.0
    total = sum(len(t) for t in toks)
    bad = sum(len(t) for t in toks if _is_garbage_token(t))
    token_score = 1.0 - (bad / total if total else 0.0)
    noise = len(_HARD_NOISE_RE.findall(text))
    noise_penalty = min(1.0, (noise / max(1, len(text))) * 4.0)
    return max(0.0, min(token_score, 1.0 - noise_penalty))


def is_probably_ocr_garbage(text: Any) -> bool:
    """True if a line/snippet is dominated by unrecoverable extraction garbage."""
    if not isinstance(text, str) or not text.strip():
        return False
    return get_text_quality_score(text) < 0.5


def normalize_whitespace(text: str) -> str:
    """Collapse runs of spaces/tabs, trim line ends, and limit blank lines."""
    if not isinstance(text, str):
        return text
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t\u00a0]+", " ", text)
    text = re.sub(r" *\n *", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def clean_mojibake_text(text: Any) -> str:
    """
    Repair true mojibake and normalise whitespace. Safely handles None, numbers,
    lists, dicts and strings. Preserves valid Unicode (— ’ ₹ é â, Devanagari) and
    never mangles URLs (real URLs are ASCII). Does NOT remove OCR garbage — use
    clean_or_drop_bad_lines for that.
    """
    if text is None:
        return ""
    if isinstance(text, (list, tuple)):
        return " ".join(clean_mojibake_text(x) for x in text)
    if isinstance(text, dict):
        return " ".join(clean_mojibake_text(v) for v in text.values())
    if not isinstance(text, str):
        return str(text)

    s = text
    if is_true_mojibake_present(s):
        s = _apply_c1_punct(s)            # deterministic, ftfy-independent
        s = _repair_lone_a(s)             # deterministic lone-"â" repair
        if _TRUE_MOJIBAKE_RE.search(s) is not None:
            try:
                import ftfy
                s = _repair_lone_a(ftfy.fix_text(s))
            except Exception:
                s = _manual_fix(s)
            if _TRUE_MOJIBAKE_RE.search(s) is not None:
                s = _manual_fix(s)
    return normalize_whitespace(s)


def clean_or_drop_bad_lines(text: Any):
    """
    Repair mojibake line-by-line and DROP unrecoverable OCR-garbage lines.
    Returns (clean_text, dropped_lines) where dropped_lines is a list of the
    original garbage lines that were removed (for quarantine logging).
    """
    if not isinstance(text, str) or not text:
        return ("" if text is None else text, [])
    kept, dropped = [], []
    for line in text.split("\n"):
        if not line.strip():
            kept.append(line)
            continue
        fixed = clean_mojibake_text(line) if is_true_mojibake_present(line) else line
        # After repair, decide whether the line is still garbage.
        if is_probably_ocr_garbage(fixed):
            dropped.append(line)
            continue
        kept.append(fixed)
    clean = normalize_whitespace("\n".join(kept))
    return clean, dropped


def fix_mojibake_preserve_layout(text: Any) -> str:
    """Repair mojibake but keep the original whitespace/line layout (for markdown)."""
    if text is None:
        return ""
    if not isinstance(text, str):
        return str(text)
    if not is_true_mojibake_present(text):
        return text
    text = _apply_c1_punct(text)          # deterministic, ftfy-independent
    text = _repair_lone_a(text)           # deterministic lone-"â" repair
    if _TRUE_MOJIBAKE_RE.search(text) is None:
        return text                        # only core mojibake needs ftfy
    try:
        import ftfy
        return _repair_lone_a(ftfy.fix_text(text))
    except Exception:
        return _manual_fix(text)


def clean_url_value(url: Any) -> str:
    """Repair mojibake in a URL without collapsing/normalising its structure."""
    if not url or not isinstance(url, str):
        return "" if url is None else (url if isinstance(url, str) else str(url))
    s = url
    if is_true_mojibake_present(s):
        try:
            import ftfy
            s = ftfy.fix_text(s)
        except Exception:
            s = _manual_fix(s)
    return s.strip()


# Fields that hold human-readable text and should be mojibake-repaired.
_TEXT_FIELDS = (
    "title", "text", "content", "source", "source_name", "category",
    "author", "date", "snippet", "description", "summary", "excerpt",
)
# Fields that hold URLs — cleaned but never whitespace-normalised.
_URL_FIELDS = ("url", "page_url", "original_url", "pdf_url", "source_url", "link")
# Long-form fields that also get blank-line normalisation.
_LONGFORM_FIELDS = ("text", "content", "snippet", "description", "summary", "excerpt")


def _clean_record(rec: Dict[str, Any]) -> Dict[str, Any]:
    """Shared cleaner for document/chunk dicts. Non-text fields pass through."""
    if not isinstance(rec, dict):
        return rec
    out = dict(rec)
    for k, v in out.items():
        if k in _URL_FIELDS:
            out[k] = clean_url_value(v)
        elif k in _TEXT_FIELDS:
            if isinstance(v, str):
                out[k] = clean_mojibake_text(v)
        elif isinstance(v, str):
            if is_true_mojibake_present(v):
                out[k] = clean_mojibake_text(v)
        elif isinstance(v, list):
            out[k] = [clean_mojibake_text(x) if isinstance(x, str) and is_true_mojibake_present(x) else x
                      for x in v]
    for k in _LONGFORM_FIELDS:
        if isinstance(out.get(k), str):
            out[k] = normalize_whitespace(out[k])
    return out


def clean_document_metadata(doc: Dict[str, Any]) -> Dict[str, Any]:
    """Return a copy of a document dict with all text/URL fields mojibake-repaired."""
    return _clean_record(doc)


def clean_chunk_metadata(chunk: Dict[str, Any]) -> Dict[str, Any]:
    """Return a copy of a chunk dict with all text/URL fields mojibake-repaired."""
    return _clean_record(chunk)


def truncate(text: str, max_chars: int = 300) -> str:
    if len(text) <= max_chars:
        return text
    return text[:max_chars].rsplit(" ", 1)[0] + "…"


def word_count(text: str) -> int:
    return len(text.split())


def approx_token_count(text: str) -> int:
    """Rough token estimate: ~0.75 words per token."""
    return int(word_count(text) * 1.33)


# ── Date helpers ───────────────────────────────────────────────────────────────

def parse_date(raw: str) -> str:
    """Try to parse a raw date string into YYYY-MM-DD; return raw on failure."""
    if not raw:
        return ""
    raw = raw.strip()
    fmts = [
        "%Y-%m-%d", "%d %B %Y", "%B %d, %Y", "%d-%m-%Y",
        "%d/%m/%Y", "%B %Y", "%Y",
    ]
    for fmt in fmts:
        try:
            return datetime.strptime(raw, fmt).strftime("%Y-%m-%d")
        except ValueError:
            pass
    return raw


def now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


# ── JSONL helpers ──────────────────────────────────────────────────────────────

def append_jsonl(path: Path, obj: Dict[str, Any]):
    """Append a single JSON object to a .jsonl file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    """Load all records from a .jsonl file."""
    if not path.exists():
        return []
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    return records


def save_jsonl(path: Path, records: List[Dict[str, Any]]):
    """Overwrite a .jsonl file with records."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def summarize_chunks_file(path: Path) -> Dict[str, Any]:
    """
    Stream a large chunks .jsonl file and return ONLY lightweight aggregate
    statistics — never the chunk text itself.

    This is the memory-safe replacement for loading the entire 25k+ chunk list
    into RAM just to show a few counts. It reads one line at a time, so peak
    memory stays tiny and the app scales comfortably to 50k+ chunks.

    Returns a small, picklable dict:
        {
          "total":      int,            # total chunks
          "by_source":  {src: count},   # chunks per source key
          "commit":     int,            # commit_kb chunks
          "avg_chars":  float,          # mean chunk length
          "max_chars":  int,            # longest chunk length
        }
    """
    total = 0
    commit = 0
    sum_chars = 0
    max_chars = 0
    by_source: Dict[str, int] = {}

    if not path.exists():
        return {"total": 0, "by_source": {}, "commit": 0,
                "avg_chars": 0.0, "max_chars": 0}

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                ch = json.loads(line)
            except json.JSONDecodeError:
                continue
            total += 1
            src = (ch.get("source") or ch.get("source_type") or "local").lower()
            by_source[src] = by_source.get(src, 0) + 1
            if src == "commit_kb":
                commit += 1
            n = len(ch.get("text", "") or "")
            sum_chars += n
            if n > max_chars:
                max_chars = n

    return {
        "total":     total,
        "by_source": by_source,
        "commit":    commit,
        "avg_chars": (sum_chars / total) if total else 0.0,
        "max_chars": max_chars,
    }


# ── Retry helper ───────────────────────────────────────────────────────────────

def with_retry(fn, retries: int = 3, backoff: float = 2.0, logger=None):
    """Call fn up to `retries` times with exponential backoff."""
    last_exc = None
    for attempt in range(1, retries + 1):
        try:
            return fn()
        except Exception as exc:
            last_exc = exc
            wait = backoff ** attempt
            if logger:
                logger.warning(f"Attempt {attempt} failed: {exc}. Retrying in {wait:.1f}s…")
            time.sleep(wait)
    raise last_exc 

# ── Clickable citations ─────────────────────────────────────────────────────────
# The model emits inline markers like "[Source 1]" / "[1]". Once the citation
# verifier has renumbered them to match the displayed source list, we can turn each
# marker into a Markdown link straight to that source's URL, so a reader can jump
# to the exact document a claim came from.

# Not followed by "(" → skip markers that are already Markdown links.
_CITATION_MARKER_RE = re.compile(r"\[(?:Source\s+)?(\d{1,2})\](?!\()", re.IGNORECASE)


def _is_http_url(u: str) -> bool:
    """True only for http(s) URLs — the only scheme ever safe to render as a
    clickable link (never javascript:, data:, file:, etc.)."""
    return bool(u) and u.strip().lower().startswith(("http://", "https://"))


def linkify_citations(answer_text: str, sources: list, label: str = "[{n}]") -> str:
    """
    Rewrite inline ``[Source N]`` / ``[N]`` markers into Markdown links pointing at
    ``sources[N-1]``'s URL.

        "…households [Source 2]."  →  "…households [[2]](https://…/lpg)."  (renders as [2])

    A marker whose number has no matching source, whose source has no URL, or
    whose URL isn't http(s), is left exactly as-is (never turned into a broken
    or unsafe link — a non-http(s) scheme such as javascript:/data: must never
    reach a rendered Markdown link). Markers already inside a Markdown link are
    not touched, so calling this twice is safe.
    """
    if not answer_text or not sources:
        return answer_text or ""

    urls = []
    for s in sources:
        u = (s.get("url") or s.get("original_url") or "").strip() if isinstance(s, dict) else ""
        urls.append(u if _is_http_url(u) else "")

    def _sub(m: "re.Match") -> str:
        idx = int(m.group(1))
        if not (1 <= idx <= len(urls)) or not urls[idx - 1]:
            return m.group(0)                      # unknown, URL-less, or unsafe → leave alone
        start = m.start()
        # Already part of "[[1]](url)" or "…](" — don't double-wrap.
        if start > 0 and answer_text[start - 1] == "[":
            return m.group(0)
        return f"[{label.format(n=idx)}]({urls[idx - 1]})"

    return _CITATION_MARKER_RE.sub(_sub, answer_text)