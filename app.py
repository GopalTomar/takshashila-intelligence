"""
app.py — Takshashila Knowledge Base RAG Dashboard (production-grade)

Enterprise dashboard over the upgraded RAG backend. Official Takshashila house
style: Wine #620d3c + Marigold #f1a222 on warm light surfaces, Inter
headings + body (Roboto Mono for metadata), lighthouse mark top-right. Text is
always dark on light / cream-on-maroon for readability — accent gold is used
for fills and large display only, never small body text on white.

Tabs: Home · Ask Takshashila · Document Explorer · Analytics · Build & Update ·
Automation.

Loading / caching:
  • Heavy singletons (FAISS index, embedding model, BM25, chunk metadata) live
    behind @st.cache_resource — loaded once per process, reused across reruns.
  • The small documents.jsonl is loaded once (@st.cache_resource).
  • The large chunks file is summarised by streaming into a tiny cached dict.
  • KB validation, scheduler status and run history are read from disk and cached.
Every backend call is guarded so a missing file never crashes the UI.
"""

import sys, time, re, json, base64, threading, html as _html_mod
from pathlib import Path
from string import Template
from datetime import datetime, timedelta
from collections import Counter

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import pandas as pd
import streamlit as st

from src import config
from src.utils import (
    load_jsonl, now_iso, clean_document_metadata, summarize_chunks_file,
    linkify_citations, truncate, clean_mojibake_text, clean_url_value, get_logger,
)

_app_logger = get_logger("dashboard", config.SCRAPE_LOG)


def _esc(value) -> str:
    """
    HTML-escape any document- or user-derived text before it goes into an
    f-string passed to st.markdown(..., unsafe_allow_html=True).

    Audit finding: scraped document metadata (titles, authors, categories,
    tags, section names) and the user's own typed question were being
    interpolated into raw HTML with NO escaping anywhere in the dashboard.
    A title containing '<', '>', '&' or a user query containing
    '<img onerror=...>' would either break the layout or execute as HTML —
    real scraped web content and free-text user input can both contain these
    characters. Every dynamic (non-hardcoded) string going into an
    unsafe_allow_html block should be wrapped in this.
    """
    return _html_mod.escape(str(value if value is not None else ""), quote=True)


def _safe_href(url: str) -> str:
    """Only ever render http(s) links as clickable hrefs — never javascript:,
    data:, or any other scheme, even if one slipped past upstream filtering."""
    u = (url or "").strip()
    if u.lower().startswith(("http://", "https://")):
        return _esc(u)
    return ""


def _is_http_url(url) -> bool:
    """Boolean form of the same check, for building a plain (non-HTML-escaped)
    Markdown link target — e.g. ``[text](url)`` in a plain st.markdown() call
    where _safe_href()'s HTML-escaped output isn't what's wanted."""
    u = str(url or "").strip()
    return u.lower().startswith(("http://", "https://"))


_MD_ESCAPE_RE = re.compile(r"([\\`*_\[\]])")


def _md_escape(value) -> str:
    """
    Neutralize Markdown-syntax characters in a plain-text value that's about
    to be interpolated into a hand-built Markdown string (not raw HTML).

    Audit finding: scraped titles/author names/section names were being
    dropped straight into f-strings passed to a plain st.markdown() call (no
    unsafe_allow_html, so real HTML injection isn't possible there — Streamlit
    sanitizes that layer) — but Markdown syntax itself was NOT neutralized. A
    title containing "]" or "(" could break the intended [n]/[open] link
    structure of that line, and in the worst case splice together a markdown
    link the title text didn't actually contain (e.g. a title ending in
    "](https://evil.example)[" immediately before a real "[🔗 open](url)"
    link could hijack which URL that link text points to). Escaping the
    handful of characters that are ever markdown-significant here removes
    that ambiguity entirely while leaving normal text untouched.
    """
    return _MD_ESCAPE_RE.sub(r"\\\1", str(value if value is not None else ""))

# ══════════════════════════════════════════════════════════════════════════════
# OFFICIAL BRAND PALETTE  (Takshashila style guide)
# ══════════════════════════════════════════════════════════════════════════════
LLAMA      = "#620d3c"   # primary maroon
LLAMA_D    = "#490a2c"   # darker maroon — gradient end / hover
LLAMA_M    = "#8a2b5c"   # medium maroon — friendlier hero fill
LLAMA_L    = "#f6ebf1"   # very light maroon tint — hover / selected states
MARIGOLD   = "#f1a222"   # accent gold — fills, icons, single highlight, big numbers
MARIGOLD_D = "#c47d0e"   # darker gold — hover on gold
GOLD_TEXT  = "#8a5600"   # readable gold-brown — gold-toned TEXT on light
GOLD_L     = "#fff4dc"   # pale gold — chip background
CREAM      = "#fffbe2"   # pale yellow — alternate surface / text on maroon
WHITE      = "#ffffff"
SURFACE    = "#F7F5F2"   # warm off-white app shell — style-guide "Deep" (never grey)
BORDER     = "#dad6d3"   # hairline — style-guide Ink-20 (rgba(23,20,19,0.16)) over white
TEXT       = "#171413"   # primary body text — style-guide "Ink"
MUTED      = "#8b8988"   # captions / axis / secondary — style-guide Ink-50 over white
OK_G       = "#1f7a4d"
WARN_A     = "#8a5600"
ERR_R      = "#b3261e"
INFO_B     = "#1a4f8c"

CHART_SEQ = [LLAMA, MARIGOLD, LLAMA_M, "#a86a2f", GOLD_TEXT, "#7a2b52",
             "#c98a3a", "#9c4a6e", "#5a0a2d", "#d4a24a", "#804063", "#b8862c"]

SRC_COLOUR = {"commit_kb": LLAMA, "website": MARIGOLD_D, "staff_handbook": GOLD_TEXT,
              "publication": INFO_B, "blog": OK_G, "pdf": "#6a3d8f", "local": MUTED}
SRC_LABEL  = {"commit_kb": "COMMIT KB", "website": "WEBSITE", "staff_handbook": "HANDBOOK",
              "publication": "PUBLICATION", "blog": "BLOG", "pdf": "PDF", "local": "LOCAL"}


def src_colour(s: str) -> str:
    return SRC_COLOUR.get((s or "").lower(), MUTED)


def src_label(s: str) -> str:
    return SRC_LABEL.get((s or "").lower(), (s or "DOC").upper())


st.set_page_config(page_title="Takshashila Knowledge Base RAG", page_icon="🏛️",
                   layout="wide", initial_sidebar_state="expanded")


def lighthouse_svg(size: int = 40) -> str:
    return f"""<svg width="{size}" height="{size}" viewBox="0 0 100 100" fill="none"
        xmlns="http://www.w3.org/2000/svg" style="flex-shrink:0">
      <circle cx="50" cy="50" r="46" fill="{MARIGOLD}"/>
      <path d="M18 42 L50 30 L44 46 Z" fill="#ffe27a" opacity="0.9"/>
      <path d="M82 42 L50 30 L56 46 Z" fill="#ffe27a" opacity="0.55"/>
      <path d="M42 40 h16 v6 h-16 z" fill="{LLAMA}"/>
      <path d="M44 46 h12 l6 34 h-24 z" fill="{LLAMA}"/>
      <rect x="46" y="24" width="8" height="8" rx="1.5" fill="{LLAMA}"/>
      <rect x="47" y="18" width="6" height="7" rx="3" fill="{LLAMA}"/>
      <rect x="48.5" y="55" width="3" height="9" rx="1.5" fill="{MARIGOLD}"/>
      <rect x="43" y="40" width="14" height="3" fill="#ffe27a"/>
    </svg>"""


@st.cache_data(show_spinner=False)
def _logo_base64() -> str:
    """Base64 of assets/takshashila_logo.png, read once and cached — embedding
    it inline (rather than a file:// or relative path) is what makes the logo
    render correctly regardless of working directory or container filesystem
    layout, with no separate static-file server needed."""
    logo_path = Path(__file__).resolve().parent / "assets" / "takshashila_logo.png"
    try:
        return base64.b64encode(logo_path.read_bytes()).decode("ascii")
    except Exception:
        return ""


def logo_img_tag(size: int = 40) -> str:
    """The real Takshashila logo (assets/takshashila_logo.png) as an inline
    <img>. Falls back to the hand-drawn SVG approximation above if the asset
    is ever missing, rather than breaking the header layout."""
    b64 = _logo_base64()
    if not b64:
        return lighthouse_svg(size)
    return (f'<img src="data:image/png;base64,{b64}" width="{size}" height="{size}" '
            f'style="flex-shrink:0;object-fit:contain" alt="Takshashila" />')


# ══════════════════════════════════════════════════════════════════════════════
# GLOBAL CSS
# ══════════════════════════════════════════════════════════════════════════════
_CSS = Template("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Roboto+Mono:wght@400;500&display=swap');

html,body{ font-size:16px; }
.stApp{ background:$SURFACE !important; }
.main .block-container{ padding-top:1.5rem !important; padding-bottom:3rem !important; max-width:1320px !important; }

h1,h2,h3,h4{ font-family:'Inter',sans-serif !important; color:$TEXT !important; }
h2{ font-size:1.55rem !important; font-weight:700 !important; }
h3{ font-size:1.15rem !important; font-weight:600 !important; }
p,li,span,div,label{ font-family:'Inter',sans-serif; }
p,li{ color:$TEXT; line-height:1.6; }

section[data-testid="stSidebar"]{ background:$WHITE !important; border-right:1px solid $BORDER !important;
  box-shadow:none; }
section[data-testid="stSidebar"] *{ color:$TEXT; }

label,.stSelectbox label,.stTextInput label,.stSlider label,.stCheckbox label,.stRadio label,.stMultiSelect label,.stNumberInput label{
  color:$MUTED !important; font-family:'Inter',sans-serif !important; font-size:0.72rem !important;
  font-weight:700 !important; text-transform:uppercase !important; letter-spacing:0.09em !important; }

.stSelectbox > div > div,.stTextInput > div > div > input,.stNumberInput input{
  background:$WHITE !important; border:1.5px solid $BORDER !important; border-radius:0 !important;
  color:$TEXT !important; font-family:'Inter',sans-serif !important; font-size:0.9rem !important; }
.stSelectbox > div > div:hover,.stTextInput > div > div > input:hover{ border-color:$LLAMA !important; }
.stSelectbox > div > div:focus-within,.stTextInput > div > div > input:focus{
  border-color:$LLAMA !important; box-shadow:none; }
.stSelectbox [data-baseweb="select"] span{ color:$TEXT !important; }
.stTextInput input::placeholder{ color:#b0a79c !important; }

[data-baseweb="popover"]{ background:$WHITE !important; border:1.5px solid $BORDER !important;
  border-radius:0 !important; box-shadow:none; }
[data-baseweb="menu"] ul{ background:$WHITE !important; padding:4px !important; }
[data-baseweb="menu"] li{ color:$TEXT !important; font-family:'Inter',sans-serif !important;
  font-size:0.88rem !important; border-radius:0 !important; padding:8px 12px !important; }
[data-baseweb="menu"] li:hover,[data-baseweb="menu"] [aria-selected="true"]{
  background:$LLAMA_L !important; color:$LLAMA !important; }

[data-baseweb="slider"] [role="slider"]{ background:$LLAMA !important; border:2px solid $WHITE !important;
  box-shadow:none; }
[data-baseweb="slider"] [data-testid="stThumbValue"]{ color:$LLAMA !important; font-weight:700 !important; }
[data-testid="stCheckbox"] input:checked + label span:first-child{ background:$LLAMA !important; border-color:$LLAMA !important; }

.stButton > button{ background:$MARIGOLD !important; color:$LLAMA !important; border:none !important;
  border-radius:0 !important; font-family:'Inter',sans-serif !important; font-size:0.88rem !important;
  font-weight:700 !important; padding:0.55rem 1.2rem !important; letter-spacing:0.01em !important;
  box-shadow:none; transition:all .16s !important; }
.stButton > button:hover{ background:$MARIGOLD_D !important; color:$WHITE !important;
  transform:translateY(-1px) !important; box-shadow:none; }
.stButton > button:active{ transform:translateY(0) !important; }
.stDownloadButton > button{ background:$WHITE !important; color:$LLAMA !important;
  border:1.5px solid $LLAMA !important; border-radius:0 !important; font-weight:600 !important;
  font-size:0.85rem !important; padding:0.45rem 1.05rem !important; }
.stDownloadButton > button:hover{ background:$LLAMA !important; color:$CREAM !important; }

.stTabs [data-baseweb="tab-list"]{ background:$WHITE !important; border-bottom:2px solid $BORDER !important;
  gap:0 !important; padding:0 8px !important; border-radius:0 10px 0 0 !important; }
.stTabs [data-baseweb="tab"]{ font-family:'Inter',sans-serif !important; font-size:0.875rem !important;
  font-weight:600 !important; color:$MUTED !important; padding:11px 20px !important;
  border-bottom:3px solid transparent !important; margin-bottom:-2px !important; background:transparent !important; }
.stTabs [data-baseweb="tab"]:hover{ color:$LLAMA !important; background:$LLAMA_L !important; }
.stTabs [aria-selected="true"]{ color:$LLAMA !important; border-bottom:3px solid $MARIGOLD !important; }
.stTabs [data-baseweb="tab-panel"]{ padding:24px 0 0 !important; }

[data-testid="stChatInput"]{ border:2px solid $BORDER !important; border-radius:0 !important;
  background:$WHITE !important; box-shadow:none; margin-top:8px !important; }
[data-testid="stChatInput"]:focus-within{ border-color:$LLAMA !important; box-shadow:none; }
[data-testid="stChatInput"] textarea{ background:$WHITE !important; color:$TEXT !important;
  font-family:'Inter',sans-serif !important; font-size:0.95rem !important; }
[data-testid="stChatInput"] textarea::placeholder{ color:#b0a79c !important; font-style:italic !important; }
[data-testid="stChatInput"] button{ background:$LLAMA !important; border-radius:0 !important; margin:6px !important; }
[data-testid="stChatInput"] button:hover{ background:$LLAMA_D !important; }
[data-testid="stChatInput"] button svg{ fill:$CREAM !important; }

[data-testid="stChatMessage"]{ background:$WHITE !important; border:1px solid $BORDER !important;
  border-radius:0 !important; margin-bottom:10px !important; box-shadow:none; }

details{ background:$WHITE !important; border:1px solid $BORDER !important; border-radius:0 !important;
  margin-bottom:8px !important; overflow:hidden !important; }
details summary{ color:$LLAMA !important; font-weight:600 !important; font-family:'Inter',sans-serif !important;
  font-size:0.87rem !important; padding:11px 16px !important; cursor:pointer !important; }
details summary:hover{ background:$LLAMA_L !important; }
details[open] summary{ border-bottom:1px solid $BORDER !important; }

[data-testid="stAlert"]{ border-radius:0 !important; border-left-width:4px !important; font-family:'Inter',sans-serif !important; }
[data-testid="stProgressBar"] > div > div{ background:linear-gradient(90deg,$LLAMA,$MARIGOLD) !important; border-radius:0 !important; }

[data-testid="stMetric"]{ background:$WHITE !important; border:1px solid $BORDER !important;
  border-radius:0 !important; padding:14px 18px !important; }
[data-testid="stMetricValue"]{ color:$LLAMA !important; font-family:'Inter',sans-serif !important; font-weight:700 !important; }
[data-testid="stMetricLabel"]{ color:$MUTED !important; font-family:'Inter',sans-serif !important; font-size:0.76rem !important; }

[data-testid="stDataFrame"]{ border:1px solid $BORDER !important; border-radius:0 !important; overflow:hidden !important; }
[data-baseweb="tag"]{ background:$LLAMA !important; color:$CREAM !important; border-radius:0 !important; }
hr{ border-color:$BORDER !important; opacity:1 !important; }

.tk-hero,.tk-hero *{ color:rgba(255,251,226,.94) !important; }
.tk-hero h1{ color:$WHITE !important; }
.tk-hero .tk-eyebrow{ color:$MARIGOLD !important; }
.tk-hero strong{ color:$MARIGOLD !important; }

::-webkit-scrollbar{ width:6px; height:6px; }
::-webkit-scrollbar-track{ background:$SURFACE; }
::-webkit-scrollbar-thumb{ background:$BORDER; border-radius:0; }
::-webkit-scrollbar-thumb:hover{ background:$MUTED; }

/* ── Responsive / mobile (audit finding: no breakpoints existed at all) ──────
   Streamlit's own [data-testid="stHorizontalBlock"] columns do NOT stack on
   narrow viewports by default — they shrink in place, which squeezes the
   4-column action row (Markdown/Copy/Bookmark), the recent-searches row, and
   the follow-up-question buttons into unreadably narrow slivers on a phone.
   These rules make Streamlit's own column containers stack vertically below
   a phone-width breakpoint, and tighten spacing/type scale to match. */
@media (max-width: 640px){
  .main .block-container{ padding-left:0.9rem !important; padding-right:0.9rem !important;
    padding-top:1rem !important; }
  h2{ font-size:1.28rem !important; }
  h3{ font-size:1.02rem !important; }
  [data-testid="stHorizontalBlock"]{ flex-direction:column !important; }
  [data-testid="stHorizontalBlock"] > div[data-testid="stColumn"]{ width:100% !important;
    flex:1 1 100% !important; min-width:100% !important; margin-bottom:8px !important; }
  .stTabs [data-baseweb="tab"]{ padding:9px 12px !important; font-size:0.8rem !important; }
  [data-testid="stChatMessage"]{ margin-bottom:8px !important; }
}
</style>
""")
st.markdown(_CSS.substitute(SURFACE=SURFACE, WHITE=WHITE, BORDER=BORDER, TEXT=TEXT, MUTED=MUTED,
            CREAM=CREAM, LLAMA=LLAMA, LLAMA_D=LLAMA_D, LLAMA_L=LLAMA_L, MARIGOLD=MARIGOLD,
            MARIGOLD_D=MARIGOLD_D), unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# CACHED LOADERS
# ══════════════════════════════════════════════════════════════════════════════
@st.cache_resource(show_spinner=False)
def get_docs():
    return [clean_document_metadata(d) for d in load_jsonl(config.DOCUMENTS_FILE)]


@st.cache_data(show_spinner=False)
def get_chunk_summary():
    return summarize_chunks_file(config.CHUNKS_FILE)


@st.cache_resource(show_spinner=False)
def get_rag_resources(kb_fingerprint: str = ""):
    # kb_fingerprint is part of the cache key: when the active KB version changes
    # (a rebuild in another process), Streamlit busts this cache and re-warms
    # against the new index — so the dashboard never serves a stale KB. Retrieval
    # also self-heals independently (vector_store.load_index is version-aware).
    from src import vector_store, embeddings, retriever
    vector_store.load_index()
    embeddings._get_model()
    retriever.ensure_bm25_ready()
    return {"ready": True, "n_vectors": vector_store.ntotal(),
            "kb_version": kb_fingerprint or vector_store.loaded_fingerprint()}


def _kb_fingerprint() -> str:
    try:
        from src import kb_version
        return kb_version.current_fingerprint()
    except Exception:
        return ""


@st.cache_data(show_spinner=False)
def get_validation():
    try:
        from scripts.validate_kb import validate
        return validate()
    except Exception as e:
        return {"ok": None, "counts": {}, "errors": [], "warnings": [], "info": [], "_err": str(e)}


def _safe_json(path):
    try:
        if path and Path(path).exists():
            return json.loads(Path(path).read_text(encoding="utf-8"))
    except Exception:
        pass
    return None


@st.cache_data(show_spinner=False, ttl=30)
def get_scheduler_status():
    return _safe_json(config.SCHEDULER_STATUS)


@st.cache_data(show_spinner=False, ttl=30)
def get_last_manifest():
    return _safe_json(config.WEBSITE_MANIFEST_FILE)


@st.cache_data(show_spinner=False, ttl=30)
def get_run_history(limit: int = 12):
    out = []
    try:
        for f in sorted(config.LOGS_DIR.glob("ingestion_report_*.json"), reverse=True)[:limit]:
            d = _safe_json(f)
            if d:
                d["_file"] = f.name
                out.append(d)
    except Exception:
        pass
    return out


def _clear_all_caches():
    st.cache_data.clear()
    st.cache_resource.clear()
    st.session_state.pop("_kb_ready", None)


docs_all      = get_docs()
chunk_summary = get_chunk_summary()


# ══════════════════════════════════════════════════════════════════════════════
# DERIVED METADATA
# ══════════════════════════════════════════════════════════════════════════════
def _doc_source(d):
    return (d.get("source") or d.get("source_type") or "local").lower()


def _doc_tags(d):
    t = d.get("tags")
    return [str(x).strip() for x in t if str(x).strip()] if isinstance(t, (list, tuple)) else []


def _doc_year(d):
    m = re.match(r"(\d{4})", str(d.get("date") or "").strip())
    return m.group(1) if m else ""


def _unique_sorted(values):
    seen, out = set(), []
    for v in values:
        v = (v or "").strip()
        if v and v.lower() not in ("unknown", "n/a") and v not in seen:
            seen.add(v); out.append(v)
    return sorted(out)


def _src_display(s):
    return config.source_display_name(s, s.replace("_", " ").title())


# Categories / tags / doc-types first (these do NOT depend on authors).
all_categories = _unique_sorted(d.get("category", "") for d in docs_all)
all_tags       = _unique_sorted(t for d in docs_all for t in _doc_tags(d))
all_doctypes   = _unique_sorted((d.get("document_type") or "") for d in docs_all)

# ── Author sanitising ────────────────────────────────────────────────────────────
# Some pages expose a *topic* ("AI", "Governance", "Geopolitics") in their author
# field. Those are not people and must never appear in the Author filter, the
# author charts, or reference by-lines. A value is rejected if it is also a
# category / tag / doc-type, a known topic word, an org name, or simply doesn't
# look like a person's name.
_TOPIC_STOP = {
    "ai", "artificial intelligence", "geopolitics", "governance", "technology",
    "policy", "public policy", "economy", "economics", "security", "defence",
    "defense", "strategy", "science", "health", "education", "climate", "energy",
    "trade", "data", "digital", "cyber", "cybersecurity", "china", "india",
    "indo-pacific", "takshashila", "takshashila institution", "staff", "admin",
    "blog", "blogs", "article", "articles", "publication", "publications",
    "newsletter", "newsletters", "report", "reports", "commentary", "research",
    "opinion", "analysis", "home", "uncategorized", "issue brief",
    "discussion document", "high-tech geopolitics", "podcast", "video", "event",
    "course", "news", "media", "all content", "advanced technology",
}
_BAD_AUTHOR_TERMS = ({c.lower() for c in all_categories} | {t.lower() for t in all_tags}
                     | {dt.lower() for dt in all_doctypes} | _TOPIC_STOP)
_ORG_HINTS = ("institution", "institute", "foundation", " org", "team", "editor",
              "editorial", "admin", "staff", "programme", "program")


def _valid_person(name) -> bool:
    n = (name or "").strip()
    if not (2 <= len(n) <= 40) or not re.search(r"[A-Za-z]", n):
        return False
    low = n.lower()
    if low in _BAD_AUTHOR_TERMS or any(h in low for h in _ORG_HINTS):
        return False
    words = n.split()
    if len(words) > 5:                      # a phrase / sentence, not a name
        return False
    if len(words) == 1:                     # single token → must be a proper name
        return n.isalpha() and n[0].isupper()
    return any(w[:1].isupper() for w in words)


def _doc_authors(d):
    a = d.get("authors")
    raw = ([str(x).strip() for x in a if str(x).strip()]
           if isinstance(a, (list, tuple)) and a
           else ([str(d.get("author")).strip()] if str(d.get("author") or "").strip() else []))
    seen, out = set(), []
    for x in raw:
        if _valid_person(x) and x.lower() not in seen:
            seen.add(x.lower()); out.append(x)
    return out


def _clean_author_str(value) -> str:
    """Filter a raw author value/list down to plausible person names, joined."""
    items = ([str(x).strip() for x in value if str(x).strip()]
             if isinstance(value, (list, tuple))
             else ([str(value).strip()] if str(value or "").strip() else []))
    return ", ".join(x for x in items if _valid_person(x))


# ── Human-readable document "kind" (for composition breakdowns) ──────────────────
_KIND_KEYWORDS = [
    ("PDF", ("pdf",)),
    ("Blog", ("blog",)),
    ("Newsletter", ("newsletter", "anticipating", "gambit", "slugout")),
    ("Publication", ("publication", "discussion document", "issue brief", "monograph",
                     "report", "brief", "working paper")),
    ("Podcast", ("podcast", "puliyabaazi", "all things policy")),
    ("Video", ("video", "youtube")),
    ("Article", ("article", "commentary", "op-ed", "opinion", "in the news")),
    ("Event", ("event", "seminar", "webinar", "workshop", "conference")),
    ("Course", ("course", "gcpp", "pgp", "graduate")),
    ("Playbook", ("playbook",)),
    ("Decision", ("decision",)),
    ("Insight", ("insight",)),
    ("Idea", ("idea",)),
    ("Note", ("note",)),
    ("Research", ("research", "paper", "study")),
]


def _doc_kind(d) -> str:
    src = _doc_source(d)
    if src == "pdf" or d.get("source_type") == "pdf" or d.get("pdf_url"):
        return "PDF"
    hay = " ".join(str(d.get(k, "")) for k in ("document_type", "category", "section")).lower()
    url = (d.get("url") or d.get("original_url") or "").lower()
    for label, keys in _KIND_KEYWORDS:
        if any(k in hay for k in keys) or any(("/" + k) in url for k in keys):
            return label
    return "Commit page" if src == "commit_kb" else "Web page"


all_sources = _unique_sorted(_doc_source(d) for d in docs_all)
all_authors = _unique_sorted(a for d in docs_all for a in _doc_authors(d))
all_years   = _unique_sorted(_doc_year(d) for d in docs_all)
all_sources.sort(key=lambda s: (-config.source_priority(s), s))
src_labels = {"(all sources)": "All sources", **{s: _src_display(s) for s in all_sources}}


def kb_composition():
    """{source: Counter(kind: count)} — 'what is actually in the knowledge base'."""
    comp = {}
    for d in docs_all:
        comp.setdefault(_doc_source(d), Counter())[_doc_kind(d)] += 1
    return comp


def _coverage():
    n = len(docs_all) or 1
    return {"n": n,
            "author": sum(1 for d in docs_all if _doc_authors(d)),
            "date": sum(1 for d in docs_all if str(d.get("date") or "").strip()),
            "url": sum(1 for d in docs_all if d.get("url") or d.get("original_url")),
            "title": sum(1 for d in docs_all if (d.get("title") or "").strip()
                         and (d.get("title") or "").strip().lower() != "untitled"),
            "category": sum(1 for d in docs_all if (d.get("category") or "").strip()),
            "tags": sum(1 for d in docs_all if _doc_tags(d))}


# ══════════════════════════════════════════════════════════════════════════════
# HTML HELPERS
# ══════════════════════════════════════════════════════════════════════════════
def eyebrow(txt, colour=MUTED):
    return (f"<div style='font-size:0.62rem;color:{colour};font-family:Inter,sans-serif;"
            f"text-transform:uppercase;letter-spacing:0.14em;font-weight:700;margin-bottom:10px'>{txt}</div>")


def stat_card(label, value, icon="", colour=LLAMA):
    val = f"{value:,}" if isinstance(value, (int, float)) else str(value)
    st.markdown(
        f"""<div style="background:{WHITE};border:1px solid {BORDER};border-top:3px solid {colour};
            border-radius:0;padding:20px 16px 16px;text-align:center;box-shadow:none">
          <div style="font-size:0.64rem;color:{MUTED};text-transform:uppercase;letter-spacing:0.13em;
                      font-family:Inter,sans-serif;font-weight:700;margin-bottom:9px">{icon} {label}</div>
          <div style="font-size:2.05rem;font-weight:700;color:{colour};font-family:'Inter',sans-serif;
                      line-height:1">{val}</div>
        </div>""", unsafe_allow_html=True)


def confidence_badge(level):
    c = {"high": OK_G, "medium": WARN_A, "low": ERR_R, "none": "#555"}.get(level, "#555")
    return (f"<span style='background:{c};color:#fff;padding:3px 13px;border-radius:0;font-size:0.72rem;"
            f"font-weight:700;letter-spacing:0.08em;font-family:Inter,sans-serif'>{level.upper()}</span>")


def grounding_badge(grounded):
    c, t = (OK_G, "GROUNDED") if grounded else ("#8a6d3b", "NO EVIDENCE")
    return (f"<span style='background:{c}1a;color:{c};padding:3px 11px;border-radius:0;font-size:0.68rem;"
            f"font-weight:700;letter-spacing:0.06em;font-family:Inter,sans-serif;border:1px solid {c}55'>"
            f"{'✓ ' if grounded else '○ '}{t}</span>")


def low_confidence_warning(level):
    if level == "low":
        st.warning("⚠️ **Low confidence.** The retrieved passages are only weakly related to your "
                   "question — treat this answer with caution and verify against the cited sources.")
    elif level == "none":
        st.info("ℹ️ No sufficiently relevant evidence was found in the knowledge base for this question.")


def rich_source_card(ch, index):
    # Truncate the RAW text first, escape afterward — escaping expands
    # characters into multi-char entities (& -> &amp;), so escaping before
    # truncating could cut an entity in half and render a broken literal
    # "&am" instead of "&".
    title    = _esc(truncate(clean_mojibake_text(ch.get("title", "Untitled")) or "Untitled", 118))
    source   = (ch.get("source") or ch.get("source_type") or "").lower()
    src_name = _esc(truncate(clean_mojibake_text(ch.get("source_name", "") or _src_display(source)), 52))
    category = clean_mojibake_text(ch.get("category", "") or "")
    url      = clean_url_value(ch.get("url") or ch.get("original_url") or "")
    date     = _esc(str(ch.get("date") or ""))
    updated  = _esc(str(ch.get("updated_date") or ""))
    page     = ch.get("page_number")
    doctype  = str(ch.get("document_type") or "")
    authors  = ch.get("authors")
    author   = _esc(truncate(_clean_author_str(authors if authors else ch.get("author")), 46))
    tags     = ch.get("tags") if isinstance(ch.get("tags"), (list, tuple)) else []
    section  = _esc(truncate(str(ch.get("heading_path") or ch.get("section") or "").strip(), 42))
    score    = ch.get("score", ch.get("rrf_score", 0)) or 0
    tc       = src_colour(source)
    score_pct = f"{score*100:.1f}%" if score else "—"

    def chip(txt, bg, fg):
        return (f"<span style='background:{bg};color:{fg};font-size:0.62rem;font-weight:700;padding:2px 9px;"
                f"border-radius:0;letter-spacing:.05em;font-family:Inter,sans-serif;display:inline-block'>{_esc(txt)}</span>")

    chips = [chip(src_label(source), tc, "#fff")]
    if category: chips.append(chip(category.upper(), GOLD_L, GOLD_TEXT))
    if doctype and doctype.lower() not in (category.lower(), source): chips.append(chip(doctype.upper(), "#eef1f6", INFO_B))
    chips_html = " ".join(chips)

    meta_bits = []
    if author:  meta_bits.append(f"✍️ {author}")
    if date:    meta_bits.append(f"📅 {date}" + (f" · upd {updated}" if updated and updated != date else ""))
    if section: meta_bits.append(f"🗂 {section}")
    if page:    meta_bits.append(f"📄 p.{_esc(page)}")
    meta_line = " &nbsp;·&nbsp; ".join(meta_bits)

    tag_html = ""
    if tags:
        tag_html = "<div style='margin-top:8px;display:flex;gap:5px;flex-wrap:wrap'>" + " ".join(
            f"<span style='background:{LLAMA_L};color:{LLAMA};font-size:0.6rem;font-weight:600;padding:1px 8px;"
            f"border-radius:0;font-family:Inter,sans-serif'>#{_esc(truncate(str(t), 22))}</span>" for t in tags[:8]) + "</div>"

    link_html = ""
    safe_url = _safe_href(url)
    if safe_url:
        link_html = (f"<a href='{safe_url}' target='_blank' rel='noopener noreferrer' style='color:{tc};font-size:0.76rem;font-weight:600;"
                     f"text-decoration:none;font-family:Inter,sans-serif;border:1.5px solid {tc};padding:3px 11px;"
                     f"border-radius:0;display:inline-block'>🔗 Open exact page →</a>")

    st.markdown(
        f"""<div style="background:{WHITE};border:1px solid {BORDER};border-left:4px solid {tc};
                border-radius:0;padding:15px 19px;margin-bottom:10px;box-shadow:none">
          <div style="display:flex;justify-content:space-between;align-items:flex-start;gap:12px">
            <div style="flex:1;min-width:0">
              <div style="display:flex;align-items:center;gap:7px;margin-bottom:7px;flex-wrap:wrap">
                {chips_html}
                <span style="color:{MUTED};font-size:0.74rem;font-family:Inter,sans-serif">Source [{_esc(index)}]</span>
              </div>
              <div style="font-weight:700;color:{TEXT};font-size:0.94rem;font-family:'Inter',sans-serif;
                          line-height:1.35;margin-bottom:5px">{title}</div>
              <div style="color:{MUTED};font-size:0.77rem;font-family:Inter,sans-serif">
                📚 {src_name}{(' &nbsp;·&nbsp; ' + meta_line) if meta_line else ''}</div>
            </div>
            <div style="text-align:center;min-width:56px;flex-shrink:0">
              <div style="font-size:1.2rem;font-weight:800;color:{tc};font-family:Inter,sans-serif;line-height:1">{score_pct}</div>
              <div style="font-size:0.58rem;color:{MUTED};letter-spacing:.07em;font-family:Inter,sans-serif;
                          text-transform:uppercase">similarity</div>
            </div>
          </div>
          <div style="margin-top:10px;padding-top:10px;border-top:1px solid {BORDER};font-size:0.81rem;
                      color:{MUTED};font-style:italic;line-height:1.65;font-family:Inter,sans-serif">
            {_esc(truncate(clean_mojibake_text(ch.get('text','')), 300))}</div>
          {tag_html}
          <div style="margin-top:10px">{link_html}</div>
        </div>""", unsafe_allow_html=True)


def render_source_cards(chunks):
    if not chunks:
        st.info("No sources retrieved."); return
    for i, ch in enumerate(chunks, 1):
        rich_source_card(ch, i)


def next_scheduled_run():
    day_map = {"mon": 0, "tue": 1, "wed": 2, "thu": 3, "fri": 4, "sat": 5, "sun": 6}
    target = day_map.get(str(config.SCHEDULE_DAY).lower()[:3], 1)
    try:
        from zoneinfo import ZoneInfo
        tz = ZoneInfo(config.SCHEDULE_TIMEZONE)
    except Exception:
        tz = None
    now = datetime.now(tz)
    ahead = (target - now.weekday()) % 7
    cand = now.replace(hour=config.SCHEDULE_HOUR, minute=config.SCHEDULE_MINUTE,
                       second=0, microsecond=0) + timedelta(days=ahead)
    if cand <= now:
        cand += timedelta(days=7)
    return cand


def fmt_dt(s):
    if not s:
        return "—"
    try:
        return datetime.fromisoformat(str(s).replace("Z", "+00:00")).strftime("%d %b %Y, %H:%M UTC")
    except Exception:
        return str(s)


def resolve_followup(user_query: str, history: list, model: str = None) -> str:
    """
    Rewrite a context-dependent follow-up ("summarise the key points") into a
    standalone, on-topic query using the recent conversation — entirely inside the
    dashboard, so it works regardless of which pipeline version is installed and
    without passing any unsupported kwargs to answer(). Never raises.

    Resolution order: the pipeline's own condenser (if present, for consistency) →
    the shared Groq client → a light heuristic that anchors to the last question.
    """
    if not history:
        return user_query
    recent = [h for h in history
              if isinstance(h, dict) and h.get("content")
              and h.get("role") in ("user", "assistant")][-6:]
    if not recent:
        return user_query

    # 1) Reuse the pipeline's condenser when it exists (keeps behaviour identical).
    try:
        from src.rag_pipeline import _condense_query as _pipe_condense
        out = _pipe_condense(user_query, recent, model=model)
        if isinstance(out, str) and out.strip():
            return out.strip()
    except Exception:
        pass

    # 2) Condense directly via the shared Groq client.
    try:
        from src import groq_client
        transcript = "\n".join(
            f"{'User' if h['role'] == 'user' else 'Assistant'}: {str(h['content']).strip()[:600]}"
            for h in recent)
        sys_p = ("You rewrite the user's latest message into a standalone search query using the "
                 "conversation for context. Resolve pronouns and vague references (\"it\", \"this\", "
                 "\"the key points\") to the concrete subject. If the message is already standalone, "
                 "return it unchanged. Output ONLY the query — no preamble, no quotes.")
        usr_p = f"Conversation so far:\n{transcript}\n\nLatest user message: {user_query}\n\nStandalone query:"
        out = groq_client.generate(system_prompt=sys_p, user_prompt=usr_p, model=model,
                                    temperature=0.0, max_tokens=120)
        out = (out or "").strip()
        out = out.splitlines()[0].strip().strip('"').strip("'") if out else ""
        out = re.sub(r"^(standalone (search )?query|query|rewritten question)\s*[:\-]\s*", "",
                     out, flags=re.I).strip()
        if 3 <= len(out) <= 400:
            return out
    except Exception:
        pass

    # 3) Heuristic fallback: anchor to the most recent distinct user question.
    last_user = next((h["content"] for h in reversed(recent)
                      if h["role"] == "user" and h["content"].strip().lower() != user_query.strip().lower()), "")
    return f"{user_query} (regarding: {last_user})" if last_user else user_query


def run_pipeline_live(target_label, fn, *args, **kwargs):
    """Run a backend function in a thread, streaming its _print output to a log box."""
    import inspect
    import scripts.update_knowledge_base as ukb
    logs, holder = [], {}
    orig_print = ukb._print
    cap = lambda msg="": logs.append(str(msg))
    ukb._print = cap
    # If the target itself accepts a progress_cb (e.g. rebuild_index), feed it too
    # so its own progress lines stream to the same log box.
    try:
        if "progress_cb" in inspect.signature(fn).parameters and "progress_cb" not in kwargs:
            kwargs["progress_cb"] = cap
    except (TypeError, ValueError):
        pass

    def work():
        try:
            holder["result"] = fn(*args, **kwargs)
        except Exception as e:
            holder["error"] = e

    lb = st.empty(); pg = st.progress(0.0, target_label)
    t = threading.Thread(target=work, daemon=True); t.start()
    try:
        while t.is_alive():
            lb.code("\n".join(logs[-24:]) or "Starting…", language="log")
            pg.progress(min(0.05 + len(logs) / 80, 0.98), target_label)
            t.join(timeout=0.5)
        t.join()
    finally:
        ukb._print = orig_print
    lb.code("\n".join(logs[-40:]) or "(no output)", language="log")
    pg.progress(1.0, "Done")
    return holder


def _friendly_pipeline_error(exc: Exception) -> str:
    """
    Map a raised exception to a calm, actionable, user-facing message —
    never the raw exception text.

    Audit finding: the previous code did
        f"⚠️ Error after {elapsed:.1f}s: {holder['error']}"
    and rendered that straight into the chat as the assistant's reply. That
    puts whatever the underlying library's exception __str__ happens to be —
    which can include internal file paths, request/response internals, or
    other implementation detail — directly in front of the user. The full
    exception is still logged server-side (see the caller) for debugging;
    only this sanitized summary reaches the UI.
    """
    name = type(exc).__name__
    msg = str(exc).lower()

    if isinstance(exc, FileNotFoundError) or "index not found" in msg or "no such file" in msg:
        return ("The knowledge base index isn't available right now. "
                "An administrator needs to run a build under **🛠️ Build & Update**.")
    if isinstance(exc, ValueError) and ("groq_api_key" in msg or "api key" in msg):
        return ("The answer service isn't configured yet (missing API credentials). "
                "Please contact an administrator.")
    if "rate limit" in msg or "429" in msg:
        return "The answer service is temporarily rate-limited. Please try again in a moment."
    if "timeout" in msg or "timed out" in name.lower():
        return "Generating the answer took too long and timed out. Please try again."
    if "connection" in msg or "connect" in name.lower() or "network" in msg:
        return "Could not reach the answer service — please check connectivity and try again."
    if "authentication" in msg or "unauthorized" in msg or "401" in msg:
        return "The answer service rejected the request due to a configuration issue. Please contact an administrator."
    return "Something went wrong while generating the answer. Please try again in a moment."


def _require_admin_access(tab_label: str) -> bool:
    """
    Gate an admin-only tab (crawl/rebuild/schedule controls) behind a shared
    password when DASHBOARD_ADMIN_PASSWORD is configured. Returns True if the
    caller should render the tab's real controls.

    If no password is configured, the tab stays USABLE (so a single-operator
    deployment isn't locked out by this fix) but shows a persistent, visible
    warning — never a silent "looks secure but isn't" state.
    """
    if not config.DASHBOARD_ADMIN_PASSWORD:
        st.warning(
            "⚠️ **No admin password configured.** Anyone who can open this "
            f"dashboard can use **{tab_label}** — including triggering a full "
            "re-crawl or changing the schedule. Set `DASHBOARD_ADMIN_PASSWORD` "
            "in `.env` to require a password here."
        )
        return True

    if st.session_state.get("_admin_unlocked"):
        return True

    st.info(f"🔒 **{tab_label}** requires the admin password.")
    pw = st.text_input("Admin password", type="password", key=f"_admin_pw_{tab_label}")
    if pw:
        if pw == config.DASHBOARD_ADMIN_PASSWORD:
            st.session_state["_admin_unlocked"] = True
            st.rerun()
        else:
            st.error("Incorrect password.")
    return False


def _admin_pipeline_error(h: dict, action_label: str) -> None:
    """
    Render a pipeline failure the same safe way the chat flow does: a calm,
    actionable message by default, with the real exception available in an
    expander for whoever is actually running the admin tools — never a raw
    exception string dropped straight into st.error() (audit finding: the
    Build & Update / Automation tabs did exactly that, inconsistently with
    the chat flow's already-sanitized _friendly_pipeline_error()).
    """
    exc = h.get("error")
    st.error(f"{action_label} failed: {_friendly_pipeline_error(exc) if isinstance(exc, Exception) else 'Unknown error.'}")
    with st.expander("Technical details"):
        st.code(repr(exc), language="text")


st.session_state.setdefault("messages", [])
st.session_state.setdefault("bookmarks", [])
st.session_state.setdefault("query_log", [])
st.session_state.setdefault("recent_searches", [])
st.session_state.setdefault("dev_mode", False)


# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown(f"""
        <div style="padding:14px 0 14px;border-bottom:2px solid {BORDER}">
          <div style="display:flex;align-items:center;gap:11px">
            {logo_img_tag(42)}
            <div>
              <div style="font-size:0.63rem;color:{MUTED};font-family:Inter,sans-serif;text-transform:uppercase;
                          letter-spacing:0.16em;font-weight:700">Takshashila</div>
              <div style="font-size:0.95rem;font-weight:700;color:{TEXT};font-family:'Inter',sans-serif;
                          line-height:1.15">Knowledge Base RAG</div>
            </div>
          </div>
        </div>""", unsafe_allow_html=True)
    st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)

    groq_ok   = bool(config.GROQ_API_KEY)
    commit_ok = config.COMMIT_KB_JSONL.exists() or any(_doc_source(d) == "commit_kb" for d in docs_all)
    index_ok  = config.FAISS_INDEX.exists()
    val       = get_validation()
    sched     = get_scheduler_status()

    def _dot(ok, t="Ready", f="Not set", warn=False):
        c = WARN_A if warn else (OK_G if ok else ERR_R)
        return f"<span style='color:{c};font-size:0.78rem;font-weight:700;font-family:Inter,sans-serif'>● {t if ok else f}</span>"

    kb_health = "Healthy" if (val.get("ok") is True) else ("Issues" if val.get("ok") is False else "—")
    rows = [("Groq API", _dot(groq_ok, "Connected")),
            ("Commit KB", _dot(commit_ok, "Ingested")),
            ("FAISS Index", _dot(index_ok, "Built")),
            ("KB Health", _dot(val.get("ok") is True, kb_health, kb_health, warn=(val.get("ok") is False)))]
    st.markdown(f"""
        <div style="background:{WHITE};border:1px solid {BORDER};border-radius:0;padding:11px 13px;margin-bottom:14px">
          {eyebrow('⚡ System Status')}
          {''.join(f'''<div style="display:flex;justify-content:space-between;align-items:center;padding:4px 0;
              border-bottom:1px solid {BORDER}"><span style="font-family:Inter,sans-serif;font-size:0.78rem;
              color:{MUTED}">{lbl}</span>{dot}</div>''' for lbl, dot in rows)}
        </div>""", unsafe_allow_html=True)

    def _slbl(icon, text):
        st.markdown(f"<p style='font-size:0.62rem;font-weight:700;text-transform:uppercase;letter-spacing:.13em;"
                    f"color:{MUTED};font-family:Inter,sans-serif;margin:14px 0 8px'>{icon} {text}</p>",
                    unsafe_allow_html=True)

    _slbl("⚙️", "Query Settings")
    model_choice = st.selectbox("Groq Model", config.AVAILABLE_MODELS, index=0, help="LLM used for answer generation")
    top_k       = st.slider("Top-K Results", 3, 15, config.TOP_K)
    temperature = st.slider("Temperature", 0.0, 1.0, config.DEFAULT_TEMP, 0.05)
    use_hybrid  = st.checkbox("Hybrid BM25 + FAISS", value=True)
    _LEN_LABELS = {"Short": "short", "Standard": "normal", "Detailed": "detailed"}
    answer_len_label = st.radio("Response length", list(_LEN_LABELS.keys()), index=1, horizontal=True,
                                help="Short = quick · Standard = balanced · Detailed = comprehensive.")
    answer_length = _LEN_LABELS[answer_len_label]

    _slbl("🔎", "Search Filters")
    sel_source = st.selectbox(f"Source ({len(all_sources)} indexed)", ["(all sources)"] + all_sources, index=0,
                              format_func=lambda s: src_labels.get(s, s),
                              help="Restrict retrieval to one source. Commit KB is the primary source.")
    source_filter = None if sel_source == "(all sources)" else sel_source
    sel_cat = st.selectbox(f"Category ({len(all_categories)} found)", ["(all categories)"] + all_categories, index=0)
    category_filter = None if sel_cat == "(all categories)" else sel_cat

    _slbl("📈", "Index Statistics")
    st.markdown(f"""
        <div style="background:{WHITE};border:1px solid {BORDER};border-radius:0;padding:10px 13px;
                    font-family:Inter,sans-serif;font-size:0.78rem;color:{TEXT}">
          <div style="display:flex;justify-content:space-between;padding:2px 0"><span style="color:{MUTED}">Documents</span><strong>{len(docs_all):,}</strong></div>
          <div style="display:flex;justify-content:space-between;padding:2px 0"><span style="color:{MUTED}">Vector chunks</span><strong>{chunk_summary['total']:,}</strong></div>
          <div style="display:flex;justify-content:space-between;padding:2px 0"><span style="color:{MUTED}">Sources</span><strong>{len(all_sources)}</strong></div>
          <div style="display:flex;justify-content:space-between;padding:2px 0"><span style="color:{MUTED}">Embedding</span><strong style="font-size:0.7rem">{config.EMBEDDING_MODEL.split('/')[-1]}</strong></div>
        </div>""", unsafe_allow_html=True)

    _slbl("🗓️", "Automation")
    nxt = next_scheduled_run()
    last_state = (sched or {}).get("state", "—")
    sc = {"success": OK_G, "error": ERR_R, "running": WARN_A}.get(last_state, MUTED)
    st.markdown(f"""
        <div style="background:{WHITE};border:1px solid {BORDER};border-radius:0;padding:10px 13px;
                    font-family:Inter,sans-serif;font-size:0.76rem;color:{TEXT}">
          <div style="display:flex;justify-content:space-between;padding:2px 0"><span style="color:{MUTED}">Schedule</span><strong>{str(config.SCHEDULE_DAY).title()} {config.SCHEDULE_HOUR:02d}:{config.SCHEDULE_MINUTE:02d}</strong></div>
          <div style="display:flex;justify-content:space-between;padding:2px 0"><span style="color:{MUTED}">Next run</span><strong>{nxt.strftime('%d %b, %H:%M')}</strong></div>
          <div style="display:flex;justify-content:space-between;padding:2px 0"><span style="color:{MUTED}">Last run</span><strong style="color:{sc}">{str(last_state).title() if last_state else '—'}</strong></div>
        </div>""", unsafe_allow_html=True)

    _slbl("🧪", "Quick Settings")
    st.session_state.dev_mode = st.checkbox("Developer mode", value=st.session_state.dev_mode,
                                            help="Show retrieved chunks, scores, timings, grounding.")
    if st.button("♻️ Refresh caches", key="sb_refresh"):
        _clear_all_caches(); st.rerun()


tabs = st.tabs(["🏠  Home", "💬  Ask Takshashila", "📚  Document Explorer",
                "📊  Analytics", "🛠️  Build & Update", "🗓️  Automation"])


# ════════════════════════════════════════════════════════════════════════════
# TAB 1 — HOME
# ════════════════════════════════════════════════════════════════════════════
with tabs[0]:
    st.markdown(f"""
        <div class="tk-hero" style="background:linear-gradient(135deg,{LLAMA_M} 0%,{LLAMA_D} 100%);
                    border-radius:0;padding:38px 44px;margin-bottom:24px;position:relative;overflow:hidden">
          <div style="position:absolute;top:22px;right:26px">{logo_img_tag(52)}</div>
          <div style="position:absolute;top:-40px;right:-40px;width:200px;height:200px;
                      background:rgba(241,162,34,.12);border-radius:0"></div>
          <div style="position:relative">
            <div class="tk-eyebrow" style="font-size:0.68rem;text-transform:uppercase;letter-spacing:0.22em;
                        font-weight:700;margin-bottom:12px">AI-Powered Internal Knowledge Assistant</div>
            <h1 style="margin:0;font-size:2rem;font-family:'Inter',sans-serif;font-weight:700;line-height:1.25;
                       max-width:600px">Takshashila<br>Commit Knowledge Base</h1>
            <p style="margin-top:12px;font-size:0.95rem;max-width:620px;line-height:1.7;font-family:Inter,sans-serif">
              Grounded, cited answers drawn primarily from the living <strong>Commit Knowledge Base</strong>
              and the public website — with metadata-aware retrieval, exact article references, and honest
              "insufficient evidence" when a topic is not covered.</p>
            <div style="margin-top:18px;display:flex;gap:9px;flex-wrap:wrap">
              <span style="background:rgba(241,162,34,.24);color:{MARIGOLD};padding:4px 13px;border-radius:0;
                           font-size:0.72rem;font-weight:600;font-family:Inter,sans-serif;
                           border:1px solid rgba(241,162,34,.45)">🧠 Hybrid FAISS + BM25</span>
              <span style="background:rgba(255,255,255,.15);color:#fff;padding:4px 13px;border-radius:0;
                           font-size:0.72rem;font-weight:600;font-family:Inter,sans-serif">📄 {len(docs_all):,} Documents</span>
              <span style="background:rgba(255,255,255,.15);color:#fff;padding:4px 13px;border-radius:0;
                           font-size:0.72rem;font-weight:600;font-family:Inter,sans-serif">⚡ Exact citations · metadata-aware</span>
            </div>
          </div>
        </div>""", unsafe_allow_html=True)

    commit_docs = sum(1 for d in docs_all if _doc_source(d) == "commit_kb")
    c1, c2, c3, c4, c5 = st.columns(5)
    with c1: stat_card("Documents", len(docs_all), "📄", LLAMA)
    with c2: stat_card("Commit KB Docs", commit_docs, "🏛️", LLAMA_M)
    with c3: stat_card("Vector Chunks", chunk_summary["total"], "🔢", GOLD_TEXT)
    with c4: stat_card("Sources", len(all_sources), "🗂️", INFO_B)
    with c5: stat_card("Categories", len(all_categories), "🏷️", OK_G)

    st.markdown("<div style='height:14px'></div>", unsafe_allow_html=True)
    cov = _coverage()
    d1, d2, d3, d4 = st.columns(4)
    with d1: stat_card("Authors", len(all_authors), "✍️", LLAMA)
    with d2: stat_card("Tags", len(all_tags), "🏷️", GOLD_TEXT)
    with d3: stat_card("Doc Types", len(all_doctypes) or 1, "📑", INFO_B)
    with d4:
        pct = cov["author"] * 100 // cov["n"]
        stat_card("Author Coverage", f"{pct}%", "🧾", OK_G if pct >= 60 else WARN_A)

    # ── Knowledge Base Composition — what is actually scraped from each source ──
    st.markdown("<div style='height:22px'></div>", unsafe_allow_html=True)
    st.markdown(eyebrow("🧭 What's in the Knowledge Base"), unsafe_allow_html=True)
    comp = kb_composition()
    KIND_ICON = {"Blog": "✍️", "Article": "📰", "Publication": "📘", "Newsletter": "✉️",
                 "Podcast": "🎙️", "Video": "🎬", "Event": "📅", "Course": "🎓", "PDF": "📄",
                 "Playbook": "📗", "Decision": "⚖️", "Insight": "💡", "Idea": "🧩", "Note": "🗒️",
                 "Research": "🔬", "Commit page": "🏛️", "Web page": "🌐"}
    comp_cols = st.columns(max(1, len(comp)))
    for ci, s in enumerate(sorted(comp, key=lambda x: (-config.source_priority(x), x))):
        kinds = comp[s]; total = sum(kinds.values()); tc = src_colour(s)
        rows_html = "".join(
            f"<div style='display:flex;justify-content:space-between;align-items:center;padding:4px 0;"
            f"border-bottom:1px solid {BORDER}'>"
            f"<span style='font-family:Inter,sans-serif;font-size:0.82rem;color:{TEXT}'>"
            f"{KIND_ICON.get(k,'•')} {k}</span>"
            f"<span style='font-family:Inter,sans-serif;font-size:0.82rem;font-weight:700;color:{tc}'>{c:,}</span></div>"
            for k, c in kinds.most_common())
        with comp_cols[ci]:
            st.markdown(f"""
                <div style="background:{WHITE};border:1px solid {BORDER};border-top:3px solid {tc};
                            border-radius:0;padding:16px 20px;height:100%">
                  <div style="display:flex;justify-content:space-between;align-items:baseline;margin-bottom:8px">
                    <span style="font-family:'Inter',sans-serif;font-weight:700;font-size:1rem;color:{TEXT}">
                      {_src_display(s)}</span>
                    <span style="font-family:Inter,sans-serif;font-weight:800;font-size:1.1rem;color:{tc}">{total:,}</span>
                  </div>
                  {rows_html}
                </div>""", unsafe_allow_html=True)

    st.markdown("<div style='height:22px'></div>", unsafe_allow_html=True)
    col_l, col_r = st.columns([3, 2])
    SAMPLE_QS = ["What are the meeting rules at Takshashila?",
                 "Who wrote the blog on AI mapping intelligence?",
                 "Which blogs discuss GeoAI?", "What is the flag system?",
                 "When was the accessibility blog published?",
                 "What are the core competencies expected from staff?"]
    with col_l:
        st.markdown(f"""<div style="background:{WHITE};border:1px solid {BORDER};border-radius:0;padding:22px 26px">
              {eyebrow('💡 Try Asking')}""", unsafe_allow_html=True)
        for i, q in enumerate(SAMPLE_QS):
            if st.button(f"›  {q}", key=f"home_q_{i}", width="stretch"):
                st.session_state["_pending_query"] = q
                st.session_state["_go_ask"] = True
                st.rerun()
        st.markdown(f"<p style='font-size:0.77rem;color:{MUTED};font-family:Inter,sans-serif;margin:10px 0 0'>"
                    f"Click any question — it opens in <strong>Ask Takshashila</strong>.</p></div>",
                    unsafe_allow_html=True)
    with col_r:
        src_lines = "".join(
            f"<div style='display:flex;justify-content:space-between;padding:3px 0;font-size:0.8rem;"
            f"font-family:Inter,sans-serif;color:{TEXT}'><span>{_src_display(s)}</span>"
            f"<span style='color:{MUTED}'>{sum(1 for d in docs_all if _doc_source(d)==s)}</span></div>"
            for s in all_sources) or f"<span style='color:{MUTED};font-size:0.82rem'>None yet.</span>"
        chips = "".join(
            f"<span style='background:{GOLD_L};color:{GOLD_TEXT};font-size:0.7rem;font-weight:700;padding:3px 11px;"
            f"border-radius:0;font-family:Inter,sans-serif;display:inline-block;margin:0 6px 6px 0'>{c}</span>"
            for c in all_categories[:24]) or f"<span style='color:{MUTED};font-size:0.82rem'>No categories.</span>"
        st.markdown(f"""<div style="background:{WHITE};border:1px solid {BORDER};border-radius:0;padding:22px 26px">
              {eyebrow('🗂️ Sources Indexed')}{src_lines}
              <div style="height:14px"></div>{eyebrow('🏷️ Categories')}{chips}</div>""", unsafe_allow_html=True)

    if not index_ok:
        st.warning("⚠️ FAISS index not built yet. Open **🛠️ Build & Update** and run an update.")
    if st.session_state.pop("_go_ask", False):
        st.info("Your question is queued — switch to the **💬 Ask Takshashila** tab to see the answer.")


# ════════════════════════════════════════════════════════════════════════════
# TAB 2 — ASK TAKSHASHILA
# ════════════════════════════════════════════════════════════════════════════
with tabs[1]:
    st.markdown(f"""<div style="margin-bottom:14px">
      <h2 style="margin:0 0 4px">Ask the Takshashila Knowledge Base</h2>
      <p style="color:{MUTED};font-family:Inter,sans-serif;font-size:0.87rem;margin:0">
        Grounded, cited answers — content <em>and</em> metadata questions (who wrote it, when,
        which category). Every reference links to the exact article.</p></div>""", unsafe_allow_html=True)

    _ask_blocked = False
    if not config.FAISS_INDEX.exists():
        st.warning("⚠️ FAISS index not found. Go to **🛠️ Build & Update** first.")
        _ask_blocked = True
    elif not config.GROQ_API_KEY:
        st.error("❌ GROQ_API_KEY not set. Add it to `.env`: `GROQ_API_KEY=gsk_...`")
        _ask_blocked = True

    if not _ask_blocked:
        _fp = _kb_fingerprint()
        if not st.session_state.get("_kb_ready") or st.session_state.get("_kb_fp") != _fp:
            with st.status("Preparing the Takshashila knowledge base…", expanded=True) as status:
                status.write("Loading the FAISS index and embedding model…")
                status.write("Building the hybrid BM25 + FAISS retriever…")
                info = get_rag_resources(_fp)
                status.update(label=f"Knowledge base ready — {info['n_vectors']:,} passages indexed.",
                              state="complete", expanded=False)
            st.session_state["_kb_ready"] = True
            st.session_state["_kb_fp"] = _fp
        else:
            get_rag_resources(_fp)

        active = []
        if source_filter:   active.append(f"Source: {_src_display(source_filter)}")
        if category_filter: active.append(f"Category: {category_filter}")
        if active:
            st.caption("Filters → " + "  ·  ".join(active))

        top_row = st.columns([3, 1])
        with top_row[0]:
            if st.session_state.recent_searches:
                st.markdown(f"<span style='font-size:0.66rem;color:{MUTED};text-transform:uppercase;letter-spacing:.12em;"
                            f"font-weight:700;font-family:Inter,sans-serif'>🕘 Recent</span>", unsafe_allow_html=True)
                rq_list = st.session_state.recent_searches[:4]
                rcols = st.columns(len(rq_list) or 1)
                for i, rq in enumerate(rq_list):
                    with rcols[i]:
                        if st.button(truncate(rq, 30), key=f"recent_{i}", width="stretch"):
                            st.session_state["_pending_query"] = rq; st.rerun()
        with top_row[1]:
            if st.session_state.bookmarks:
                _bm_container = (st.popover(f"🔖 Bookmarks ({len(st.session_state.bookmarks)})",
                                            width="stretch")
                                 if hasattr(st, "popover")
                                 else st.expander(f"🔖 Bookmarks ({len(st.session_state.bookmarks)})"))
                with _bm_container:
                    for bi, bm in enumerate(reversed(st.session_state.bookmarks[-12:])):
                        st.markdown(f"**{truncate(bm['query'], 60)}**")
                        st.caption(f"{bm.get('confidence','')} · {fmt_dt(bm.get('ts',''))}")
                        if st.button("Ask again", key=f"bm_again_{bi}"):
                            st.session_state["_pending_query"] = bm["query"]; st.rerun()
                        st.divider()

        def _meta_html(msg):
            bits = [f"<div>Confidence: {confidence_badge(msg.get('confidence','none'))}</div>",
                    f"<div>{grounding_badge(bool(msg.get('grounded')))}</div>"]
            et, rt, gt, ts = (msg.get("elapsed_time"), msg.get("retrieval_time"),
                              msg.get("generation_time"), msg.get("top_score"))
            if et is not None: bits.append(f"<div style='color:{MUTED};font-size:0.74rem'>⏱ <strong>{et:.2f}s</strong> total</div>")
            if rt is not None and gt is not None:
                bits.append(f"<div style='color:{MUTED};font-size:0.72rem'>🔎 {rt:.2f}s retrieval · 🤖 {gt:.2f}s gen</div>")
            if ts is not None: bits.append(f"<div style='color:{MUTED};font-size:0.72rem'>🎯 top match {ts*100:.0f}%</div>")
            n_src = len(msg.get("sources") or [])
            if n_src: bits.append(f"<div style='color:{MUTED};font-size:0.74rem'>📎 <strong>{n_src}</strong> source(s)</div>")
            bits.append(f"<div style='color:{MUTED};font-size:0.74rem'>🗂 {msg.get('model','')}</div>")
            return (f"<div style='display:flex;align-items:center;gap:13px;flex-wrap:wrap;margin-top:10px;padding-top:10px;"
                    f"border-top:1px solid {BORDER};font-family:Inter,sans-serif'>" + "".join(bits) + "</div>")

        def _followups(msg):
            srcs = msg.get("sources") or []; ideas = []
            if srcs:
                s0 = srcs[0]
                au = _clean_author_str(s0.get("authors") or s0.get("author"))
                au = au.split(",")[0].strip() if au else ""
                cat = s0.get("category") or ""
                if au: ideas.append(f"What else has {au} written?")
                if cat: ideas.append(f"What other {cat} content is there?")
            ideas += ["Summarise the key points.", "What are the main recommendations?"]
            seen, out = set(), []
            for x in ideas:
                if x not in seen:
                    seen.add(x); out.append(x)
            return out[:3]

        def _render_assistant_extras(msg, key_suffix):
            # When a follow-up was resolved against the conversation (e.g. "summarise
            # the key points" → "summarise the key points about Gopal Tomar"), show the
            # standalone question that was actually searched, so context retention is visible.
            rq = (msg.get("resolved_query") or "").strip()
            oq = (msg.get("query") or "").strip()
            if rq and oq and rq.lower() != oq.lower():
                st.markdown(
                    f"<div style='background:{LLAMA_L};border:1px solid {BORDER};border-left:3px solid {LLAMA};"
                    f"border-radius:0;padding:7px 13px;margin-bottom:8px;font-family:Inter,sans-serif;"
                    f"font-size:0.78rem;color:{TEXT}'>↳ <span style='color:{MUTED}'>Understood in context as:</span> "
                    f"<strong>{_esc(truncate(rq, 140))}</strong></div>", unsafe_allow_html=True)
            low_confidence_warning(msg.get("confidence", "none"))
            st.markdown(_meta_html(msg), unsafe_allow_html=True)
            sources = msg.get("sources") or []
            if sources:
                ref_lines = ["**References**  \n*(what the inline [Source N] markers point to)*"]
                for i, s in enumerate(sources, start=1):
                    # Plain-text fields (title/author/source name) go through a
                    # Markdown-syntax-safe cleanup: st.markdown() here has
                    # unsafe_allow_html=False (Streamlit's own HTML sanitizer
                    # applies), but a scraped title containing raw "[", "]", "("
                    # or ")" could still break the intended [n] / [open] link
                    # structure of this line, or — combined with a crafted
                    # sequence — splice in a markdown link the title didn't
                    # actually contain. Neutralize those characters in prose text
                    # only (never inside the deliberately-built link syntax).
                    title = _md_escape((s.get("title") or "Untitled").strip())
                    src_name = _md_escape(s.get("source_name")
                                or config.source_display_name(s.get("source", ""), s.get("source", "")))
                    url = s.get("url") or s.get("original_url") or ""
                    au = _md_escape(_clean_author_str(s.get("authors") or s.get("author")))
                    dt = _md_escape(str(s.get("date") or ""))
                    extra = " · ".join(x for x in [au, dt] if x)
                    page = f" · p.{_md_escape(s['page_number'])}" if s.get("page_number") else ""
                    link = f" — [🔗 open]({url})" if _is_http_url(url) else ""
                    ref_lines.append(f"**[{i}]** {title} — *{src_name}*" + (f" · {extra}" if extra else "") + f"{page}{link}")
                st.markdown("  \n".join(ref_lines))
                with st.expander(f"📎 View full source passage(s) ({len(sources)})", expanded=False):
                    render_source_cards(sources)
            else:
                st.caption("No reliable source found for this question.")

            if st.session_state.dev_mode and msg.get("chunks"):
                with st.expander("🧪 Developer — retrieved chunks, scores & grounding", expanded=False):
                    st.caption(f"original query: {msg.get('query')!r}  →  searched: {msg.get('resolved_query')!r}")
                    st.caption(f"top_score={msg.get('top_score')} · retrieval={msg.get('retrieval_time')}s · "
                               f"generation={msg.get('generation_time')}s · grounded={msg.get('grounded')}")
                    rows = [{"#": r, "title": truncate(c.get("title", ""), 40), "source": c.get("source", ""),
                             "score": round(float(c.get("score", 0) or 0), 4),
                             "rrf": round(float(c.get("rrf_score", 0) or 0), 4),
                             "chunk_id": truncate(str(c.get("chunk_id", "")), 26), "url": c.get("url", "")}
                            for r, c in enumerate(msg["chunks"], 1)]
                    st.dataframe(pd.DataFrame(rows), width="stretch",
                                 height=min(340, 60 + 34 * len(rows)),
                                 column_config={"url": st.column_config.LinkColumn("url")})

            ac = st.columns([1.1, 1.1, 1.1, 2])
            md_dl = (f"# Takshashila RAG Answer\n\n**Query:** {msg.get('query','')}\n\n"
                     f"**Confidence:** {msg.get('confidence','none')} · **Time:** {msg.get('elapsed_time',0):.2f}s\n\n"
                     f"{msg.get('content','')}\n\n---\n*Generated {msg.get('timestamp','')} · {msg.get('model','')}*")
            with ac[0]:
                st.download_button("⬇️ Markdown", md_dl, f"answer_{key_suffix}.md", "text/markdown",
                                   key=f"dl_{key_suffix}", width="stretch")
            with ac[1]:
                if st.button("📋 Copy", key=f"copy_{key_suffix}", width="stretch"):
                    st.session_state[f"_show_copy_{key_suffix}"] = True
            with ac[2]:
                if st.button("🔖 Bookmark", key=f"bm_{key_suffix}", width="stretch"):
                    st.session_state.bookmarks.append({"query": msg.get("query", ""), "answer": msg.get("content", ""),
                                                       "sources": sources, "confidence": msg.get("confidence"),
                                                       "ts": now_iso()})
                    (_toast := getattr(st, "toast", None)) and _toast("Bookmarked.")
            if st.session_state.get(f"_show_copy_{key_suffix}"):
                st.code(msg.get("content", ""), language="markdown")

            fu = _followups(msg)
            if fu:
                st.markdown(f"<span style='font-size:0.66rem;color:{MUTED};text-transform:uppercase;letter-spacing:.12em;"
                            f"font-weight:700;font-family:Inter,sans-serif'>💬 Suggested follow-ups</span>",
                            unsafe_allow_html=True)
                fcols = st.columns(len(fu))
                for i, q in enumerate(fu):
                    with fcols[i]:
                        if st.button(truncate(q, 34), key=f"fu_{key_suffix}_{i}", width="stretch"):
                            st.session_state["_pending_query"] = q; st.rerun()

        for i, msg in enumerate(st.session_state.messages):
            with st.chat_message(msg["role"]):
                body = msg["content"]
                if msg["role"] == "assistant" and not msg.get("is_error"):
                    body = linkify_citations(body, msg.get("sources") or [])
                st.markdown(body)
                if msg["role"] == "assistant" and not msg.get("is_error"):
                    _render_assistant_extras(msg, str(i))
                if msg.get("is_error") and st.session_state.dev_mode and msg.get("_raw_error"):
                    with st.expander("🧪 Developer — raw exception (hidden by default)", expanded=False):
                        st.code(msg["_raw_error"], language="text")

        pending = st.session_state.pop("_pending_query", None)
        user_query = st.chat_input("Ask a question about Takshashila…") or pending

        if user_query:
            rs = [user_query] + [x for x in st.session_state.recent_searches if x != user_query]
            st.session_state.recent_searches = rs[:8]
            st.session_state.messages.append({"role": "user", "content": user_query, "timestamp": now_iso()})
            with st.chat_message("user"):
                st.markdown(user_query)

            with st.chat_message("assistant"):
                _STAGES = ["🔍 Searching knowledge base…", "📚 Reading sources…", "🧠 Generating answer…",
                           "✍️ Formatting…", "📨 Delivering…"]
                with st.status(_STAGES[0], expanded=True) as status:
                    # Build the conversation history in the MAIN thread. st.session_state
                    # is bound to the script's run context and is NOT reliably readable
                    # from the worker thread — reading it there can return empty, which
                    # silently disables follow-up context. Prior user/assistant turns
                    # (excluding the current question and error replies) are used to
                    # resolve follow-ups like "summarise the key points" into a
                    # standalone, on-topic query — done here in the dashboard so we
                    # never pass kwargs the installed pipeline may not support.
                    hist = [{"role": m["role"], "content": m["content"]}
                            for m in st.session_state.messages[:-1]
                            if m.get("role") in ("user", "assistant")
                            and m.get("content") and not m.get("is_error")][-6:]

                    t0 = time.perf_counter(); holder = {}

                    def _work():
                        try:
                            from src.rag_pipeline import answer as rag_answer
                            # Resolve the follow-up against the conversation, then run the
                            # normal pipeline on the standalone query. No history kwarg is
                            # passed, so this is compatible with every pipeline version.
                            resolved = resolve_followup(user_query, hist, model_choice)
                            holder["resolved"] = resolved
                            holder["result"] = rag_answer(query=resolved, top_k=top_k, model=model_choice,
                                                          temperature=temperature, source=source_filter,
                                                          category=category_filter, use_hybrid=use_hybrid,
                                                          length=answer_length)
                        except Exception as exc:
                            holder["error"] = exc

                    worker = threading.Thread(target=_work, daemon=True); worker.start()
                    idx = 0
                    while worker.is_alive():
                        status.update(label=_STAGES[min(idx, len(_STAGES) - 1)])
                        worker.join(timeout=0.7); idx += 1
                    worker.join()
                    elapsed = time.perf_counter() - t0

                    if "error" in holder:
                        status.update(label="Something went wrong.", state="error")
                        _app_logger.error(f"rag_answer failed for query {user_query!r}: {holder['error']!r}",
                                          exc_info=holder["error"])
                        friendly = _friendly_pipeline_error(holder["error"])
                        assistant_msg = {"role": "assistant", "content": f"⚠️ {friendly}",
                                         "query": user_query, "sources": [], "confidence": "none", "elapsed_time": elapsed,
                                         "model": model_choice, "timestamp": now_iso(), "is_error": True, "grounded": False,
                                         "_raw_error": repr(holder["error"])}
                    else:
                        result = holder["result"]; conf = result["confidence"]
                        status.update(label=f"✅ Answer ready in {elapsed:.2f}s.", state="complete")
                        assistant_msg = {"role": "assistant", "content": result["answer"], "query": user_query,
                                         "resolved_query": holder.get("resolved") or result.get("resolved_query") or user_query,
                                         "sources": result["sources"], "chunks": result.get("chunks", []),
                                         "confidence": conf, "top_score": result.get("top_score"),
                                         "grounded": bool(result.get("sources")) and conf != "none",
                                         "elapsed_time": elapsed, "retrieval_time": result.get("retrieval_time"),
                                         "generation_time": result.get("generation_time"), "model": model_choice,
                                         "length": answer_length, "timestamp": now_iso(), "is_error": False}

            st.session_state.messages.append(assistant_msg)
            st.session_state.query_log.append({
                "query": user_query, "confidence": assistant_msg.get("confidence"),
                "grounded": assistant_msg.get("grounded", False), "elapsed": assistant_msg.get("elapsed_time"),
                "retrieval": assistant_msg.get("retrieval_time"), "generation": assistant_msg.get("generation_time"),
                "n_sources": len(assistant_msg.get("sources") or []),
                "top_category": (assistant_msg.get("sources") or [{}])[0].get("category", "") if assistant_msg.get("sources") else "",
                "top_title": (assistant_msg.get("sources") or [{}])[0].get("title", "") if assistant_msg.get("sources") else "",
                "ts": now_iso()})
            st.rerun()

        if st.session_state.messages:
            cc = st.columns([1, 1, 4])
            with cc[0]:
                if st.button("🗑️ Clear chat", key="clr"):
                    st.session_state.messages = []; st.rerun()
            with cc[1]:
                convo = "\n\n---\n\n".join(f"**{m['role'].title()}:** {m['content']}" for m in st.session_state.messages)
                st.download_button("⬇️ Export chat", f"# Takshashila conversation\n\n{convo}",
                                   "conversation.md", "text/markdown", key="dl_convo")


# ════════════════════════════════════════════════════════════════════════════
# TAB 3 — DOCUMENT EXPLORER
# ════════════════════════════════════════════════════════════════════════════
with tabs[2]:
    st.markdown(f"""<div style="margin-bottom:18px">
      <h2 style="margin:0 0 5px">Document Explorer</h2>
      <p style="color:{MUTED};font-family:Inter,sans-serif;font-size:0.88rem;margin:0">
        Search and filter all {len(docs_all):,} documents by source, author, category, tag, year,
        type, and full text.</p></div>""", unsafe_allow_html=True)

    if not docs_all:
        st.info("No documents indexed yet. Run an update in **🛠️ Build & Update** first.")
    else:
        df = pd.DataFrame([{
            "Title": d.get("title", "") or "Untitled", "Source": _src_display(_doc_source(d)),
            "_src": _doc_source(d), "Author": ", ".join(_doc_authors(d)), "Category": d.get("category", "") or "",
            "Type": d.get("document_type", "") or "", "Year": _doc_year(d), "Tags": ", ".join(_doc_tags(d)),
            "Chars": len(d.get("text", "") or ""), "Date": str(d.get("date") or ""),
            "URL": d.get("url") or d.get("original_url") or "", "_text": d.get("text", "") or "",
        } for d in docs_all])

        r1 = st.columns([3, 2, 2])
        with r1[0]: sq = st.text_input("Full-text search", "", placeholder="title or body text…")
        with r1[1]: sel_s = st.selectbox("Source", ["(all)"] + all_sources, key="ex_src",
                                         format_func=lambda s: src_labels.get(s, s))
        with r1[2]: sel_a = st.selectbox("Author", ["(all)"] + all_authors, key="ex_auth")
        r2 = st.columns([2, 2, 2, 2])
        with r2[0]: sel_c = st.selectbox("Category", ["(all)"] + all_categories, key="ex_cat")
        with r2[1]: sel_t = st.selectbox("Tag", ["(all)"] + all_tags, key="ex_tag")
        with r2[2]: sel_y = st.selectbox("Year", ["(all)"] + all_years, key="ex_year")
        with r2[3]: sel_dt = st.selectbox("Type", ["(all)"] + all_doctypes, key="ex_type")

        filt = df.copy()
        if sq:
            filt = filt[filt["Title"].str.contains(sq, case=False, na=False)
                        | filt["_text"].str.contains(sq, case=False, na=False)]
        if sel_s != "(all)":  filt = filt[filt["_src"] == sel_s]
        if sel_a != "(all)":  filt = filt[filt["Author"].str.contains(re.escape(sel_a), case=False, na=False)]
        if sel_c != "(all)":  filt = filt[filt["Category"] == sel_c]
        if sel_t != "(all)":  filt = filt[filt["Tags"].str.contains(re.escape(sel_t), case=False, na=False)]
        if sel_y != "(all)":  filt = filt[filt["Year"] == sel_y]
        if sel_dt != "(all)": filt = filt[filt["Type"] == sel_dt]

        PAGE = 100
        total = len(filt); n_pages = max(1, (total + PAGE - 1) // PAGE)
        pc = st.columns([1, 5])
        with pc[0]: page = st.number_input("Page", 1, n_pages, 1, 1, key="ex_page")
        start, end = (int(page) - 1) * PAGE, (int(page) - 1) * PAGE + PAGE
        st.markdown(f"<p style='font-size:0.81rem;color:{MUTED};font-family:Inter,sans-serif'>Showing "
                    f"<strong>{start+1 if total else 0}–{min(end,total)}</strong> of <strong>{total}</strong> "
                    f"matching (of {len(df)} total) · page {int(page)}/{n_pages}</p>", unsafe_allow_html=True)

        show = filt.iloc[start:end].drop(columns=["_text", "_src"])
        st.dataframe(show, width="stretch", height=340,
                     column_config={"URL": st.column_config.LinkColumn("Page URL")})
        st.download_button("⬇️ Download CSV (filtered)", filt.drop(columns=["_text", "_src"]).to_csv(index=False),
                           "takshashila_documents.csv", "text/csv")

        st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)
        st.markdown(eyebrow("🔍 Document Preview"), unsafe_allow_html=True)
        titles = filt.iloc[start:end]["Title"].tolist()
        if titles:
            sel = st.selectbox("Select a document (current page)", titles, key="ex_preview")
            row = filt[filt["Title"] == sel].iloc[0]
            tc = src_colour(row["_src"])
            # Audit fix: this used to strip every character outside ASCII +
            # Devanagari with a literal quote mark — mangling perfectly valid
            # text (café, em dashes, curly quotes, French/Arabic/Chinese names
            # in citations) into garbage, while ALSO not actually protecting
            # against HTML injection (it didn't touch '<', '>', '&'). Proper
            # HTML-escaping preserves all legitimate Unicode and is safe.
            clean_title = _esc(truncate(str(row["Title"]), 200))
            meta = " &nbsp;·&nbsp; ".join(x for x in [
                f"✍️ {_esc(row['Author'])}" if row["Author"] else "",
                f"📅 {_esc(row['Date'])}" if row["Date"] else "",
                f"🏷️ {_esc(row['Category'])}" if row["Category"] else "",
                f"📑 {_esc(row['Type'])}" if row["Type"] else ""] if x)
            doc_match = next((d for d in docs_all if (d.get('title') or 'Untitled') == sel), {})
            tags_html = "".join(
                f"<span style='background:{LLAMA_L};color:{LLAMA};font-size:0.62rem;font-weight:600;padding:1px 9px;"
                f"border-radius:0;margin:0 5px 5px 0;display:inline-block'>#{_esc(truncate(str(t), 30))}</span>"
                for t in _doc_tags(doc_match))
            safe_doc_url = _safe_href(row.get('URL', ''))
            open_link = (f"<a href='{safe_doc_url}' target='_blank' rel='noopener noreferrer' style='color:{tc};font-size:0.81rem;font-weight:600;"
                         f"text-decoration:none;border:1.5px solid {tc};padding:3px 11px;border-radius:0'>🔗 Open page</a>") if safe_doc_url else ""
            st.markdown(f"""
                <div style="background:{WHITE};border:1px solid {BORDER};border-left:4px solid {tc};
                            border-radius:0;padding:20px 26px;margin-top:8px">
                  <div style="display:flex;gap:7px;align-items:center;margin-bottom:9px;flex-wrap:wrap">
                    <span style="background:{tc};color:#fff;font-size:0.6rem;font-weight:700;padding:2px 9px;
                                 border-radius:0;letter-spacing:.1em">{_esc(src_label(row['_src']))}</span>
                    <span style="color:{MUTED};font-size:0.79rem;font-family:Inter,sans-serif">{row['Chars']:,} chars</span>
                  </div>
                  <div style="font-family:'Inter',sans-serif;font-size:1.05rem;font-weight:700;color:{TEXT};
                              margin-bottom:6px">{clean_title}</div>
                  <div style="color:{MUTED};font-size:0.82rem;font-family:Inter,sans-serif;margin-bottom:8px">
                    📚 {_esc(row['Source'])}{(' &nbsp;·&nbsp; ' + meta) if meta else ''}</div>
                  <div style="margin-bottom:10px">{tags_html}</div>{open_link}
                </div>""", unsafe_allow_html=True)
            if row.get("_text"):
                with st.expander("View extracted text (first 1500 chars)"):
                    clean_prev = _esc(str(row["_text"])[:1500])
                    st.markdown(f"<div style='font-family:Inter,sans-serif;font-size:0.83rem;color:{TEXT};"
                                f"line-height:1.7;white-space:pre-wrap'>{clean_prev}</div>", unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════════════════
# TAB 4 — ANALYTICS
# ════════════════════════════════════════════════════════════════════════════
with tabs[3]:
    st.markdown(f"""<div style="margin-bottom:18px">
      <h2 style="margin:0 0 5px">Knowledge Base Analytics</h2>
      <p style="color:{MUTED};font-family:Inter,sans-serif;font-size:0.88rem;margin:0">
        Coverage, composition and this session's retrieval performance.</p></div>""", unsafe_allow_html=True)

    if not docs_all:
        st.info("No documents to analyse yet.")
    else:
        import plotly.express as px
        import plotly.graph_objects as go
        BL = dict(plot_bgcolor=WHITE, paper_bgcolor=WHITE,
                  font=dict(family="Inter,sans-serif", size=12, color=TEXT),
                  margin=dict(t=48, b=18, l=18, r=18),
                  hoverlabel=dict(bgcolor=WHITE, bordercolor=BORDER,
                                  font=dict(family="Inter,sans-serif", size=12, color=TEXT)))
        df_a = pd.DataFrame([{"source": _src_display(_doc_source(d)),
                              "category": (d.get("category") or "uncategorized").strip() or "uncategorized",
                              "year": _doc_year(d) or "unknown"} for d in docs_all])

        r1a, r1b = st.columns(2)
        with r1a:
            sdf = df_a.groupby("source").size().reset_index(name="count").sort_values("count")
            fig = px.bar(sdf, x="count", y="source", orientation="h", title="Documents by Source",
                         color="source", color_discrete_sequence=CHART_SEQ)
            fig.update_layout(**BL, showlegend=False, xaxis=dict(gridcolor="#EDE7E0"),
                              yaxis=dict(gridcolor="#EDE7E0", title=""))
            st.plotly_chart(fig, width="stretch")
        with r1b:
            cdf = df_a.groupby("category").size().reset_index(name="count").sort_values("count", ascending=False).head(12)
            fig = go.Figure(go.Pie(labels=cdf["category"], values=cdf["count"], hole=0.56,
                                   marker=dict(colors=CHART_SEQ, line=dict(color=WHITE, width=3)),
                                   textposition="outside", textfont=dict(size=11, family="Inter,sans-serif")))
            fig.update_layout(**BL, title="Documents by Category",
                              annotations=[dict(text=f"<b>{len(docs_all)}</b><br>docs", x=0.5, y=0.5, showarrow=False,
                                                font=dict(size=15, color=TEXT, family="Inter,sans-serif"))])
            st.plotly_chart(fig, width="stretch")

        # Composition: source → type (what is actually scraped from where)
        comp = kb_composition()
        comp_rows = [{"source": _src_display(s), "type": k, "count": c}
                     for s, kinds in comp.items() for k, c in kinds.items()]
        if comp_rows:
            comp_df = pd.DataFrame(comp_rows)
            fig = px.sunburst(comp_df, path=["source", "type"], values="count",
                              title="Knowledge Base Composition (source → type)",
                              color="source", color_discrete_sequence=CHART_SEQ)
            fig.update_traces(insidetextorientation="radial",
                              hovertemplate="<b>%{label}</b><br>%{value} documents<extra></extra>")
            fig.update_layout(**BL, height=430)
            st.plotly_chart(fig, width="stretch")

        r2a, r2b = st.columns(2)
        with r2a:
            tl = df_a[df_a["year"] != "unknown"].groupby("year").size().reset_index(name="count").sort_values("year")
            if len(tl):
                fig = px.bar(tl, x="year", y="count", title="Document Timeline (by year)")
                fig.update_traces(marker_color=LLAMA)
                fig.update_layout(**BL, xaxis=dict(title="", gridcolor="#EDE7E0"),
                                  yaxis=dict(title="Documents", gridcolor="#EDE7E0"))
                st.plotly_chart(fig, width="stretch")
            else:
                st.info("No dated documents to plot a timeline yet.")
        with r2b:
            top_auth = Counter(a for d in docs_all for a in _doc_authors(d)).most_common(12)
            if top_auth:
                adf = pd.DataFrame(top_auth, columns=["author", "count"]).sort_values("count")
                fig = px.bar(adf, x="count", y="author", orientation="h", title="Top Authors")
                fig.update_traces(marker_color=MARIGOLD_D)
                fig.update_layout(**BL, xaxis=dict(title="Documents", gridcolor="#EDE7E0"),
                                  yaxis=dict(title="", gridcolor="#EDE7E0"))
                st.plotly_chart(fig, width="stretch")
            else:
                st.info("No author metadata to chart yet.")

        top_tags = Counter(t for d in docs_all for t in _doc_tags(d)).most_common(20)
        if top_tags:
            tdf = pd.DataFrame(top_tags, columns=["tag", "count"]).sort_values("count")
            fig = px.bar(tdf, x="count", y="tag", orientation="h", title="Top Tags")
            fig.update_traces(marker_color=LLAMA_M)
            fig.update_layout(**BL, height=460, xaxis=dict(title="Documents", gridcolor="#EDE7E0"),
                              yaxis=dict(title="", gridcolor="#EDE7E0"))
            st.plotly_chart(fig, width="stretch")

        st.markdown(eyebrow("🩺 Knowledge Base Health"), unsafe_allow_html=True)
        cov = _coverage()
        h1, h2, h3, h4 = st.columns(4)
        with h1: st.metric("Total chunks", f"{chunk_summary['total']:,}")
        with h2: st.metric("Avg chunk size", f"~{chunk_summary['avg_chars']:.0f} chars")
        with h3: st.metric("Validation errors", len(val.get("errors", [])))
        with h4: st.metric("Validation warnings", len(val.get("warnings", [])))

        cvals = [("Title", cov["title"]), ("URL", cov["url"]), ("Author", cov["author"]),
                 ("Date", cov["date"]), ("Category", cov["category"]), ("Tags", cov["tags"])]
        cov_df = pd.DataFrame([{"field": k, "pct": round(v * 100 / cov["n"], 1)} for k, v in cvals])
        fig = px.bar(cov_df, x="pct", y="field", orientation="h", title="Metadata Coverage (%)",
                     range_x=[0, 100], color="pct",
                     color_continuous_scale=[[0, "#f0d5c0"], [0.6, MARIGOLD], [1, LLAMA]])
        fig.update_layout(**BL, coloraxis_showscale=False, xaxis=dict(gridcolor="#EDE7E0"),
                          yaxis=dict(title="", gridcolor="#EDE7E0"))
        st.plotly_chart(fig, width="stretch")

        if val.get("errors"):
            for e in val["errors"]: st.error(f"❌ {e}")
        for w in val.get("warnings", []): st.warning(f"⚠️ {w}")

        st.markdown(eyebrow("⚡ This Session — Retrieval Performance"), unsafe_allow_html=True)
        qlog = st.session_state.query_log
        if not qlog:
            st.info("Ask a few questions to populate session analytics.")
        else:
            qdf = pd.DataFrame(qlog)
            s1, s2, s3, s4 = st.columns(4)
            with s1: st.metric("Queries", len(qdf))
            with s2: st.metric("Avg response", f"{qdf['elapsed'].dropna().mean():.2f}s" if qdf['elapsed'].notna().any() else "—")
            with s3:
                grounded = int(qdf["grounded"].sum()) if "grounded" in qdf else 0
                st.metric("Grounded", f"{grounded}/{len(qdf)}")
            with s4:
                st.metric("No-evidence", int((~qdf["grounded"]).sum()) if "grounded" in qdf else 0)

            cperf = st.columns(2)
            with cperf[0]:
                conf_df = qdf["confidence"].value_counts().reset_index()
                conf_df.columns = ["confidence", "count"]
                fig = px.bar(conf_df, x="confidence", y="count", title="Confidence Distribution",
                             color="confidence",
                             color_discrete_map={"high": OK_G, "medium": WARN_A, "low": ERR_R, "none": "#888"})
                fig.update_layout(**BL, showlegend=False, xaxis=dict(title="", gridcolor="#EDE7E0"),
                                  yaxis=dict(title="", gridcolor="#EDE7E0"))
                st.plotly_chart(fig, width="stretch")
            with cperf[1]:
                if qdf["elapsed"].notna().any():
                    fig = px.line(qdf.reset_index(), x="index", y="elapsed", title="Response Time (per query)",
                                  markers=True, color_discrete_sequence=[LLAMA])
                    fig.update_layout(**BL, xaxis=dict(title="query #", gridcolor="#EDE7E0"),
                                      yaxis=dict(title="seconds", gridcolor="#EDE7E0"))
                    st.plotly_chart(fig, width="stretch")

            mc = st.columns(2)
            with mc[0]:
                st.markdown(eyebrow("🗂️ Most-hit Categories (session)"), unsafe_allow_html=True)
                cats = Counter(x.get("top_category", "") for x in qlog if x.get("top_category"))
                if cats:
                    for c, n in cats.most_common(6):
                        st.markdown(f"<div style='display:flex;justify-content:space-between;font-size:0.82rem;"
                                    f"font-family:Inter,sans-serif;padding:2px 0'><span>{_esc(c)}</span><strong>{n}</strong></div>",
                                    unsafe_allow_html=True)
                else:
                    st.caption("—")
            with mc[1]:
                st.markdown(eyebrow("🚫 Recent No-evidence Queries"), unsafe_allow_html=True)
                fails = [x for x in qlog if not x.get("grounded")][-6:]
                if fails:
                    for x in reversed(fails):
                        st.markdown(f"<div style='font-size:0.82rem;font-family:Inter,sans-serif;color:{MUTED};"
                                    f"padding:2px 0'>• {_esc(truncate(x.get('query',''), 60))}</div>", unsafe_allow_html=True)
                else:
                    st.caption("None — every query found evidence. 🎉")


# ════════════════════════════════════════════════════════════════════════════
# TAB 5 — BUILD & UPDATE
# ════════════════════════════════════════════════════════════════════════════
with tabs[4]:
    st.markdown(f"""<div style="margin-bottom:16px">
      <h2 style="margin:0 0 5px">Build & Update Knowledge Base</h2>
      <p style="color:{MUTED};font-family:Inter,sans-serif;font-size:0.88rem;margin:0">
        Crawl the website + Commit KB, merge, chunk, embed, index and validate — with live logs.
        Incremental runs re-embed only changed chunks.</p></div>""", unsafe_allow_html=True)

    _admin_ok = _require_admin_access("Build & Update")

    def _shdr(num, title, desc, col=LLAMA):
        st.markdown(f"""
            <div style="background:{WHITE};border:1px solid {BORDER};border-radius:0;padding:15px 18px 11px;margin-bottom:10px">
              <div style="display:flex;align-items:center;gap:9px;margin-bottom:5px">
                <div style="background:{col};color:#fff;width:25px;height:25px;border-radius:0;display:flex;
                            align-items:center;justify-content:center;font-weight:800;font-size:0.8rem">{num}</div>
                <div style="font-weight:700;color:{TEXT};font-family:'Inter',sans-serif;font-size:0.95rem">{title}</div>
              </div>
              <p style="color:{MUTED};font-size:0.81rem;margin:0;font-family:Inter,sans-serif;line-height:1.5">{desc}</p>
            </div>""", unsafe_allow_html=True)

    ca, cb = st.columns(2)
    with ca:
        _shdr("1", "Incremental Update", "Crawl website + Commit KB, fetch only new/changed pages, merge, "
              "re-embed changed chunks, rebuild the index, then validate.")
        _clicked_incr = st.button("🔄  Run incremental update", key="btn_incr", width="stretch")
        if _clicked_incr and not _admin_ok:
            st.error("🔒 Admin password required to run this — enter it above.")
        elif _clicked_incr:
            from scripts.update_knowledge_base import run as ukb_run
            h = run_pipeline_live("Incremental update…", ukb_run, website=True, commit_kb=True,
                                  incremental=True, do_index=True)
            if "error" in h:
                _admin_pipeline_error(h, "Update")
            else:
                s = h["result"]; m = s.get("merge", {}); v = s.get("validation") or {}
                st.success(f"✓ +{m.get('added',0)} new · ~{m.get('updated',0)} changed · "
                           f"-{m.get('removed',0)} removed · {m.get('total',0)} total")
                (st.success if v.get("ok") else st.warning)(
                    f"Validation: {'PASS' if v.get('ok') else 'issues'} — "
                    f"{len(v.get('errors',[]))} errors, {len(v.get('warnings',[]))} warnings")
                _clear_all_caches()
    with cb:
        _shdr("2", "Full Re-scrape", "Re-crawl everything from scratch (ignores crawl state). Re-embeds only "
              "where content changed (cache-backed).", col=MARIGOLD_D)
        _clicked_full = st.button("🌐  Full re-scrape + rebuild", key="btn_full", width="stretch")
        if _clicked_full and not _admin_ok:
            st.error("🔒 Admin password required to run this — enter it above.")
        elif _clicked_full:
            from scripts.update_knowledge_base import run as ukb_run
            h = run_pipeline_live("Full re-scrape…", ukb_run, website=True, commit_kb=True,
                                  incremental=False, do_index=True)
            if "error" in h:
                _admin_pipeline_error(h, "Re-scrape")
            else:
                s = h["result"]; m = s.get("merge", {})
                st.success(f"✓ Rebuilt — {m.get('total',0)} documents in the knowledge base.")
                _clear_all_caches()

    cc, cd = st.columns(2)
    with cc:
        _shdr("3", "Rebuild Index Only", "Re-chunk documents.jsonl and rebuild FAISS from the embedding cache "
              "(no crawl). Fast — use after manual edits.", col=INFO_B)
        _clicked_reindex = st.button("⚡  Rebuild index", key="btn_reindex", width="stretch")
        if _clicked_reindex and not _admin_ok:
            st.error("🔒 Admin password required to run this — enter it above.")
        elif _clicked_reindex:
            from src.incremental_index import rebuild_index
            h = run_pipeline_live("Rebuilding index…", rebuild_index, use_cache=True)
            if "error" in h:
                _admin_pipeline_error(h, "Rebuild")
            else:
                s = h["result"]
                st.success(f"✓ {s.get('chunks',0):,} chunks from {s.get('documents',0)} docs "
                           f"(embedded {s.get('embedded',0)}, reused {s.get('cached',0)})")
                _clear_all_caches()
    with cd:
        _shdr("4", "Validate Knowledge Base", "Check for missing metadata, duplicate URLs, orphan/oversized "
              "chunks and index drift.", col=OK_G)
        if st.button("🩺  Run validation", key="btn_val", width="stretch"):
            get_validation.clear()
            v = get_validation()
            (st.success if v.get("ok") else st.error)(
                f"{'✅ PASS' if v.get('ok') else '❌ FAIL'} — {len(v.get('errors',[]))} errors, "
                f"{len(v.get('warnings',[]))} warnings")
            for e in v.get("errors", []): st.error(f"❌ {e}")
            for w in v.get("warnings", []): st.warning(f"⚠️ {w}")
            for i in v.get("info", []): st.caption(f"ℹ️ {i}")

    man = get_last_manifest()
    if man:
        st.markdown("<div style='height:6px'></div>", unsafe_allow_html=True)
        st.markdown(eyebrow("📋 Last Run Summary"), unsafe_allow_html=True)
        mm = man.get("merge", {}); mi = man.get("index", {}) or {}
        st.markdown(f"""<div style="background:{WHITE};border:1px solid {BORDER};border-left:4px solid {MARIGOLD};
                        border-radius:0;padding:14px 20px;font-family:Inter,sans-serif;font-size:0.83rem;color:{TEXT}">
              <strong>{man.get('mode','?').title()} run</strong> · {fmt_dt(man.get('finished_at',''))} ·
              {man.get('duration_seconds','?')}s<br>
              Merge: +{mm.get('added',0)} / ~{mm.get('updated',0)} / -{mm.get('removed',0)} · {mm.get('total',0)} docs
              &nbsp;·&nbsp; Index: {mi.get('chunks','?')} chunks (embedded {mi.get('embedded','?')},
              reused {mi.get('cached','?')})</div>""", unsafe_allow_html=True)

    st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
    st.markdown(f"""
        <div style="background:{CREAM};border:1px solid {BORDER};border-left:4px solid {MARIGOLD};
                    border-radius:0;padding:16px 22px">
          {eyebrow('🔐 Terminal equivalents (need .env credentials)', GOLD_TEXT)}
          <pre style="background:{WHITE};border:1px solid {BORDER};border-radius:0;padding:12px 14px;
                      font-size:0.78rem;color:{TEXT};overflow:auto;margin:0">python scripts/rescrape_all.py --reset-state   # one-time full rebuild
python scripts/update_knowledge_base.py         # incremental (website + Commit KB)
python scripts/build_index.py                   # rebuild index from documents.jsonl
python scripts/validate_kb.py                   # KB health check
python scripts/dedupe_documents.py              # collapse duplicate-URL documents</pre>
        </div>""", unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════════════════
# TAB 6 — AUTOMATION
# ════════════════════════════════════════════════════════════════════════════
with tabs[5]:
    st.markdown(f"""<div style="margin-bottom:16px">
      <h2 style="margin:0 0 5px">Automation</h2>
      <p style="color:{MUTED};font-family:Inter,sans-serif;font-size:0.88rem;margin:0">
        Weekly self-refresh — every {str(config.SCHEDULE_DAY).title()} at {config.SCHEDULE_HOUR:02d}:{config.SCHEDULE_MINUTE:02d}
        {config.SCHEDULE_TIMEZONE}.</p></div>""", unsafe_allow_html=True)

    _admin_ok = _require_admin_access("Automation")

    sched = get_scheduler_status()
    nxt = next_scheduled_run()
    last_state = (sched or {}).get("state", "—")
    last_run = (sched or {}).get("finished_at") or (sched or {}).get("started_at")
    sc = {"success": OK_G, "error": ERR_R, "running": WARN_A}.get(last_state, MUTED)

    a1, a2, a3 = st.columns(3)
    with a1:
        st.markdown(f"""<div style="background:linear-gradient(135deg,{LLAMA_M},{LLAMA_D});border-radius:0;
                        padding:20px 22px;color:#fff">
              <div style="font-size:0.62rem;text-transform:uppercase;letter-spacing:.14em;font-weight:700;
                          color:{MARIGOLD};font-family:Inter,sans-serif">Next scheduled run</div>
              <div style="font-size:1.35rem;font-weight:700;font-family:'Inter',sans-serif;margin-top:6px">{nxt.strftime('%a, %d %b')}</div>
              <div style="font-size:0.9rem;color:{CREAM};font-family:Inter,sans-serif">{nxt.strftime('%H:%M')} {config.SCHEDULE_TIMEZONE}</div>
            </div>""", unsafe_allow_html=True)
    with a2:
        st.markdown(f"""<div style="background:{WHITE};border:1px solid {BORDER};border-top:3px solid {sc};
                        border-radius:0;padding:20px 22px">
              <div style="font-size:0.62rem;text-transform:uppercase;letter-spacing:.14em;font-weight:700;
                          color:{MUTED};font-family:Inter,sans-serif">Last run</div>
              <div style="font-size:1.3rem;font-weight:700;font-family:'Inter',sans-serif;color:{sc};margin-top:6px">{str(last_state).title()}</div>
              <div style="font-size:0.82rem;color:{MUTED};font-family:Inter,sans-serif">{fmt_dt(last_run)}</div>
            </div>""", unsafe_allow_html=True)
    with a3:
        dur = (sched or {}).get("duration_seconds")
        chg = ((sched or {}).get("summary") or {}).get("changed_documents", "—")
        st.markdown(f"""<div style="background:{WHITE};border:1px solid {BORDER};border-top:3px solid {MARIGOLD};
                        border-radius:0;padding:20px 22px">
              <div style="font-size:0.62rem;text-transform:uppercase;letter-spacing:.14em;font-weight:700;
                          color:{MUTED};font-family:Inter,sans-serif">Last run detail</div>
              <div style="font-size:1.3rem;font-weight:700;font-family:'Inter',sans-serif;color:{LLAMA};margin-top:6px">{chg} changed</div>
              <div style="font-size:0.82rem;color:{MUTED};font-family:Inter,sans-serif">
                {f'{dur:.0f}s' if isinstance(dur,(int,float)) else '—'} duration</div>
            </div>""", unsafe_allow_html=True)

    st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)
    mcol = st.columns([1, 1, 3])
    with mcol[0]:
        _clicked_auto = st.button("▶️ Run update now", key="auto_run", width="stretch")
        if _clicked_auto and not _admin_ok:
            st.error("🔒 Admin password required to run this — enter it above.")
        elif _clicked_auto:
            from scripts.update_knowledge_base import run as ukb_run
            h = run_pipeline_live("Manual run…", ukb_run, website=True, commit_kb=True, incremental=True, do_index=True)
            if "error" in h:
                _admin_pipeline_error(h, "Run")
            else:
                st.success("✓ Update complete.")
                get_scheduler_status.clear(); get_run_history.clear(); _clear_all_caches()
    with mcol[1]:
        if st.button("↻ Refresh status", key="auto_refresh", width="stretch"):
            get_scheduler_status.clear(); get_run_history.clear(); st.rerun()

    st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)
    st.markdown(eyebrow("🗂️ Run History"), unsafe_allow_html=True)
    hist = get_run_history()
    if hist:
        hrows = []
        for r in hist:
            m = r.get("merge", {}); i = r.get("index", {}) or {}; v = r.get("validation") or {}
            hrows.append({"Finished": fmt_dt(r.get("finished_at", "")), "Mode": r.get("mode", ""),
                          "Added": m.get("added", 0), "Updated": m.get("updated", 0), "Removed": m.get("removed", 0),
                          "Chunks": i.get("chunks", "—"),
                          "Validation": "PASS" if v.get("ok") else ("FAIL" if v.get("ok") is False else "—"),
                          "Duration(s)": r.get("duration_seconds", "—")})
        st.dataframe(pd.DataFrame(hrows), width="stretch", height=300)
    else:
        st.info("No ingestion reports yet. They appear after the first scheduled or manual run "
                "(`data/logs/ingestion_report_*.json`).")

    st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
    st.markdown(f"""
        <div style="background:{CREAM};border:1px solid {BORDER};border-left:4px solid {MARIGOLD};
                    border-radius:0;padding:16px 22px">
          {eyebrow('🖥️ Set up the weekly job', GOLD_TEXT)}
          <p style="font-size:0.83rem;color:{TEXT};font-family:Inter,sans-serif;line-height:1.6;margin:0 0 8px">
            Windows (recommended — survives reboots):</p>
          <pre style="background:{WHITE};border:1px solid {BORDER};border-radius:0;padding:12px 14px;
                      font-size:0.78rem;color:{TEXT};overflow:auto;margin:0 0 10px">powershell -ExecutionPolicy Bypass -File scripts\\setup_windows_task.ps1</pre>
          <p style="font-size:0.83rem;color:{TEXT};font-family:Inter,sans-serif;line-height:1.6;margin:0 0 8px">
            Cross-platform (long-running scheduler):</p>
          <pre style="background:{WHITE};border:1px solid {BORDER};border-radius:0;padding:12px 14px;
                      font-size:0.78rem;color:{TEXT};overflow:auto;margin:0">python scripts/scheduler.py            # blocks; fires weekly
python scripts/scheduler.py --run-now  # run now, then keep schedule
python scripts/scheduler.py --status   # last run result</pre>
        </div>""", unsafe_allow_html=True)