"""
config.py — Central configuration for Takshashila RAG
Loads from .env; provides typed constants for the whole project.

Updated: adds Commit KB (authenticated) settings, a dedicated
data/knowledge_base/ area, source-priority ranking, and tunable
retrieval thresholds.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()


def _env_float(name: str, default: float) -> float:
    """
    os.getenv(name, default) only falls back to ``default`` when the variable
    is completely UNSET. If it's present in .env but left blank (a very easy
    mistake when filling in a template — e.g. ``SOURCE_PRIORITY_BOOST=`` with
    nothing after the ``=``), os.getenv returns ``""``, and ``float("")``
    crashes the whole app at import time with a bare ValueError and no hint
    about which setting caused it. Treat a blank/whitespace-only value the
    same as "not set", and fall back to the default on any other malformed
    value too (e.g. a stray typo) rather than crashing startup over one
    tuning knob.
    """
    raw = os.getenv(name, "").strip()
    if not raw:
        return default
    try:
        return float(raw)
    except ValueError:
        print(f"[config] Warning: {name}={raw!r} is not a valid number — using default {default}.")
        return default


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name, "").strip()
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError:
        print(f"[config] Warning: {name}={raw!r} is not a valid whole number — using default {default}.")
        return default

# ── Paths ──────────────────────────────────────────────────────────────────────
ROOT_DIR        = Path(__file__).parent.parent
DATA_DIR        = ROOT_DIR / "data"
RAW_HTML_DIR    = DATA_DIR / "raw" / "html"
RAW_PDF_DIR     = DATA_DIR / "raw" / "pdfs"
PROCESSED_DIR   = DATA_DIR / "processed"
INDEX_DIR       = DATA_DIR / "index"
LOGS_DIR        = DATA_DIR / "logs"

# Dedicated knowledge-base area (RAG-ready, normalized documents)
KB_DIR          = DATA_DIR / "knowledge_base"

DOCUMENTS_FILE  = PROCESSED_DIR / "documents.jsonl"   # unified docs used to build the index
CHUNKS_FILE     = PROCESSED_DIR / "chunks.jsonl"
FAISS_INDEX     = INDEX_DIR / "faiss.index"
METADATA_FILE   = INDEX_DIR / "metadata.pkl"
SCRAPE_LOG      = LOGS_DIR / "scrape.log"
FAILED_CSV      = LOGS_DIR / "failed_urls.csv"

# ── Versioned knowledge base (atomic publishing + stale-index detection) ─────────
# The active index is described by a small manifest written atomically after a
# build. Every consumer (dashboard, Mattermost bot, API) reads this manifest and
# reloads its in-memory index automatically when the fingerprint changes — so a
# rebuild in one process is picked up everywhere with NO restart. Versioned
# builds live under INDEX_VERSIONS_DIR; KB_MANIFEST_FILE is the "current" pointer.
INDEX_VERSIONS_DIR = INDEX_DIR / "versions"
KB_MANIFEST_FILE   = INDEX_DIR / "current.json"   # {version, index_hash, built_at, counts, model}

# ── Commit KB (primary source) ──────────────────────────────────────────────────
# Raw crawl output (produced by scripts/scrape_commit_kb.py)
COMMIT_KB_CRAWL_DIR  = ROOT_DIR / "commit_kb_clean_crawled"
COMMIT_KB_RAW_JSONL  = COMMIT_KB_CRAWL_DIR / "rag_documents.jsonl"

# RAG-ready Commit KB files (produced by scripts/save_commit_kb_to_rag.py)
COMMIT_KB_JSONL      = KB_DIR / "takshashila_commit_kb.jsonl"
COMMIT_KB_INDEX      = KB_DIR / "takshashila_commit_kb_index.json"
COMMIT_KB_METADATA   = KB_DIR / "takshashila_commit_kb_metadata.json"

# Public Takshashila website (publications + blogs) as a first-class KB source.
# Raw crawl output is data/processed/documents.jsonl (produced by src.scraper);
# the RAG-ready, cleaned website KB (produced by scripts/save_website_to_rag.py):
WEBSITE_JSONL     = KB_DIR / "takshashila_website.jsonl"
WEBSITE_INDEX     = KB_DIR / "takshashila_website_index.json"
WEBSITE_METADATA  = KB_DIR / "takshashila_website_metadata.json"

# Optional secondary local sources (folded into the index if present)
LOCAL_DOCUMENTS_FILE = KB_DIR / "local_documents.jsonl"        # secondary
STAFF_HANDBOOK_FILE  = KB_DIR / "takshashila_staff_handbook.jsonl"  # optional supporting

# ── Commit KB crawl settings (read from .env; never hardcode credentials) ────────
COMMIT_KB_URL      = os.getenv("COMMIT_KB_URL", "https://commit.takshashila.org.in/")
COMMIT_KB_USERNAME = os.getenv("COMMIT_KB_USERNAME", "")
COMMIT_KB_PASSWORD = os.getenv("COMMIT_KB_PASSWORD", "")

# ── Legacy public-website sources (kept for backward compatibility) ──────────────
PUBLICATIONS_URL = "https://takshashila.org.in/pages/publications/"
BLOGS_URL        = "https://takshashila.org.in/pages/blogs/"

RSS_FEEDS = [
    "https://takshashila.org.in/feed/",
    "https://takshashila.org.in/category/publications/feed/",
    "https://takshashila.org.in/category/blogs/feed/",
]

USER_AGENT = (
    "TakshashilaRAG-Research-Bot/1.0 "
    "(Academic research tool; contact: research@example.com)"
)

# ── Scraper ────────────────────────────────────────────────────────────────────
SCRAPE_DELAY       = _env_float("SCRAPE_DELAY", 1.0)
SCRAPE_TIMEOUT     = _env_int("SCRAPE_TIMEOUT", 30)
SCRAPE_MAX_RETRIES = _env_int("SCRAPE_MAX_RETRIES", 3)

# ── Public website — full-site discovery (sitemap + listings + crawl fallback) ──
WEBSITE_BASE_URL = os.getenv("WEBSITE_BASE_URL", "https://takshashila.org.in/").rstrip("/") + "/"
WEBSITE_DOMAIN   = "takshashila.org.in"

# ── Additional internal subdomains (audit finding) ──────────────────────────────
# The crawler treats a link as "internal" (worth following/keeping) only if its
# hostname is in WEBSITE_DOMAIN + this list. A live check (Sept 2026) found real,
# current institutional content served from hostnames OTHER than the bare
# takshashila.org.in domain — e.g. legion.takshashila.org.in (team, mission,
# blogs, commentary, milestones, contact) and school.takshashila.org.in (Policy
# School / GCPP / Expert Capsule Course programme pages). Because _is_internal()
# previously did an exact-string match against a single hardcoded domain, links
# to those hostnames were silently never followed, regardless of crawl depth —
# an entire subdomain's worth of content excluded with no error or log line.
#
# Left EMPTY by default (no behaviour change without an explicit decision) since
# it's unclear from the outside whether legion.* is a live, currently-maintained
# mirror/predecessor site or content Takshashila wants included — that's a call
# for whoever owns the site, not something to assume here. Set via env, e.g.:
#   WEBSITE_ADDITIONAL_DOMAINS=legion.takshashila.org.in,school.takshashila.org.in
WEBSITE_ADDITIONAL_DOMAINS = [
    d.strip() for d in os.getenv("WEBSITE_ADDITIONAL_DOMAINS", "").split(",") if d.strip()
]

# Candidate sitemap URLs to try (WordPress/Yoast-style sitemap index is checked
# first; each entry may itself be a sitemap index that is expanded recursively).
WEBSITE_SITEMAP_URLS = [
    "https://takshashila.org.in/sitemap.xml",
    "https://takshashila.org.in/sitemap_index.xml",
    "https://takshashila.org.in/wp-sitemap.xml",
]

# Known listing pages to paginate (beyond publications/blogs), tried politely —
# a 404 on any of these is skipped silently, so it's safe to over-list here.
WEBSITE_LISTING_URLS = [
    ("https://takshashila.org.in/pages/publications/", "publication"),
    ("https://takshashila.org.in/pages/blogs/", "blog"),
    ("https://takshashila.org.in/research/", "research"),
    ("https://takshashila.org.in/commentary/", "commentary"),
    ("https://takshashila.org.in/reports/", "report"),
    ("https://takshashila.org.in/papers/", "paper"),
    ("https://takshashila.org.in/briefs/", "brief"),
]

# URL-path substrings that mark a page as "useful content".
#
# IMPORTANT (full-site coverage): these are now used ONLY as an *optional*
# keep-hint, NOT as a discovery gate. The crawler follows EVERY internal link
# (minus the follow-excludes below) so that every page and subpage is visited;
# whether a visited page is *kept* as a knowledge-base document is decided by
# WEBSITE_DOC_EXCLUDE_PATTERNS + the minimum-text-length filter. Leaving this
# list EMPTY means "keep every content page that passes the text filter", which
# is what you want for a complete crawl. It is kept here (populated) only for
# backward compatibility / documentation; the website config below passes an
# empty keep-filter on purpose.
WEBSITE_INCLUDE_PATH_PATTERNS = [
    "/content/", "/publications/", "/blogs/", "/articles/", "/research/",
    "/commentary/", "/policy/", "/papers/", "/reports/", "/briefs/",
]

# ── Two-tier crawl filtering (the key to "scrape every page and subpage") ───────
#
# 1) FOLLOW-exclude — URLs the crawler must NEVER enqueue/visit at all
#    (admin, auth, cart, feeds, JSON APIs, mail/tel links, same-page fragments,
#    share/reply query strings). Everything else internal IS followed, so the
#    BFS reaches the whole site — including pages only linked from category,
#    tag, author or paginated archive pages.
WEBSITE_FOLLOW_EXCLUDE_PATTERNS = [
    "/wp-admin/", "/wp-login", "/wp-json/", "/xmlrpc.php",
    "/cart/", "/checkout/", "/my-account/", "/logout", "/register",
    "mailto:", "tel:", "javascript:", "#", "?share=", "?replytocom=",
    "/comments/feed/", "/trackback/",
]

# 2) DOC-exclude — URLs that ARE followed (for discovery) but must NEVER be
#    kept as a knowledge-base document (navigation, archives, search, feeds,
#    bare pagination shells). Their outbound links are still crawled so we
#    reach the real content behind them.
WEBSITE_DOC_EXCLUDE_PATTERNS = [
    "/tag/", "/tags/", "/category/", "/categories/", "/author/", "/authors/",
    "/search", "/page/", "/feed/", "/wp-json/", "/wp-admin/",
]

# Back-compat alias (older code referenced WEBSITE_EXCLUDE_PATH_PATTERNS).
WEBSITE_EXCLUDE_PATH_PATTERNS = WEBSITE_DOC_EXCLUDE_PATTERNS

# Non-document file extensions to never fetch as a "page" during crawl fallback
# (still allowed as linked PDFs, handled separately).
WEBSITE_SKIP_EXTENSIONS = (
    ".jpg", ".jpeg", ".png", ".gif", ".svg", ".webp", ".ico", ".css", ".js",
    ".woff", ".woff2", ".ttf", ".eot", ".mp4", ".mp3", ".zip", ".xml",
)

# Minimum extracted text length for a crawled page to be kept as a document
# (filters out empty/near-empty navigation & chrome pages).
WEBSITE_MIN_TEXT_LEN = _env_int("WEBSITE_MIN_TEXT_LEN", 200)

# Crawl safety caps (overridable via CLI flags / env). Raised so a first
# full crawl genuinely reaches every page/subpage of the site rather than
# stopping at an artificially low cap. The incremental weekly run is cheap
# regardless of these because unchanged pages are skipped.
WEBSITE_MAX_PAGES  = _env_int("WEBSITE_MAX_PAGES", 5000)
WEBSITE_MAX_DEPTH  = _env_int("WEBSITE_MAX_DEPTH", 8)
WEBSITE_INCLUDE_PDFS = os.getenv("WEBSITE_INCLUDE_PDFS", "true").lower() in ("1", "true", "yes", "on")

# Concurrency for the crawl thread pool. Kept modest to stay polite to the
# origin server; combined with SCRAPE_DELAY it bounds the request rate.
SCRAPE_MAX_WORKERS = _env_int("SCRAPE_MAX_WORKERS", 8)

# Manifest of the most recent website scrape run (what was scraped / skipped /
# failed / updated), written fresh on every run of scripts/update_knowledge_base.py.
WEBSITE_MANIFEST_FILE = LOGS_DIR / "website_scrape_manifest.json"

# ── Embeddings ─────────────────────────────────────────────────────────────────
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5")
EMBEDDING_DIM   = 384   # bge-small-en-v1.5; update if model changes

# ── Chunking ───────────────────────────────────────────────────────────────────
# Chunk sizes are expressed in CHARACTERS (the new chunker is character-based,
# which matches the 800–1200 char / 150–250 overlap guidance).
CHUNK_SIZE    = _env_int("CHUNK_SIZE", 1200)     # target chars per chunk
CHUNK_OVERLAP = _env_int("CHUNK_OVERLAP", 150)   # overlap chars
CHUNK_MIN_LEN = _env_int("CHUNK_MIN_LEN", 200)   # drop tiny fragments

# ── Groq ───────────────────────────────────────────────────────────────────────
GROQ_API_KEY  = os.getenv("GROQ_API_KEY", "")

# ── Dashboard admin gate (audit finding) ─────────────────────────────────────
# The Streamlit dashboard had NO authentication anywhere: the "Build & Update"
# and "Automation" tabs — which trigger a full site re-crawl, rebuild the
# FAISS index, or change the weekly schedule — were reachable by anyone who
# could load the dashboard URL, with no login of any kind.
#
# Left EMPTY by default so a single-operator local/private deployment (the
# common case) keeps working exactly as before with no new lockout risk —
# app.py shows a persistent, visible warning on those tabs when this is unset
# rather than silently pretending they're protected. Set this to require a
# shared password before those tabs' controls are usable:
#   DASHBOARD_ADMIN_PASSWORD=some-long-random-value
DASHBOARD_ADMIN_PASSWORD = os.getenv("DASHBOARD_ADMIN_PASSWORD", "")

# ── LLM model (single authoritative definition) ───────────────────────────────
# There is exactly ONE supported model. The dashboard, RAG engine, Mattermost
# bot, API and tests all read GROQ_MODEL / AVAILABLE_MODELS from here, so there
# can never be a dashboard/bot model mismatch. If the configured model is not in
# AVAILABLE_MODELS the LLM client raises a clear operational error rather than
# silently falling back to a different model (see src/groq_client.py).
AVAILABLE_MODELS = [
    "openai/gpt-oss-120b",
]
DEFAULT_MODEL = AVAILABLE_MODELS[0]
GROQ_MODEL    = os.getenv("GROQ_MODEL", DEFAULT_MODEL)

# ── RAG retrieval ────────────────────────────────────────────────────────────────
TOP_K             = _env_int("TOP_K", 5)
DEFAULT_TEMP      = 0.1
MAX_CONTEXT_CHARS = 12000   # rough cap before sending to Groq

# Minimum raw cosine similarity for a chunk to count as "evidence".
# Below this, the pipeline returns an honest "insufficient evidence" reply.
MIN_SCORE_THRESHOLD = _env_float("MIN_SCORE_THRESHOLD", 0.35)

# Confidence tiers, based on the best raw cosine similarity among hits.
CONF_HIGH_THRESHOLD   = _env_float("CONF_HIGH_THRESHOLD", 0.62)
CONF_MEDIUM_THRESHOLD = _env_float("CONF_MEDIUM_THRESHOLD", 0.48)
# (low = MIN_SCORE_THRESHOLD .. CONF_MEDIUM_THRESHOLD ; none = below MIN)

# ── Lexical (BM25) evidence gate ──────────────────────────────────────────────
# A hybrid retriever must NOT treat a strong lexical (BM25) match as if it had a
# cosine score of 0. A short acronym query ("What is ATP?") is often matched
# precisely by BM25 lexically while scoring low on dense cosine similarity —
# treating that as "no evidence" is the root cause of the ATP false-refusal bug.
# Each chunk therefore carries an independent normalized lexical_score in [0,1]
# (best BM25 hit for the query = 1.0). A chunk clears the evidence gate when its
# cosine clears MIN_SCORE_THRESHOLD *or* its normalized lexical score clears
# LEXICAL_MIN_SCORE. This raises recall for exact-match queries without lowering
# the dense threshold for everything else (which would invite hallucination).
LEXICAL_MIN_SCORE  = _env_float("LEXICAL_MIN_SCORE", 0.55)
# Normalized lexical score at/above which a lexical-only hit counts as "medium"
# confidence (a precise keyword match is a real, if different, signal).
LEXICAL_CONF_MEDIUM = _env_float("LEXICAL_CONF_MEDIUM", 0.75)

# ── Citation verification (anti-hallucination) ──────────────────────────────────
# After the LLM answers, verify its inline [Source N] citations against the
# retrieved context and return ONLY the sources actually used. This guarantees
# the displayed references exactly match where the answer came from.
VERIFY_CITATIONS = os.getenv("VERIFY_CITATIONS", "true").lower() in ("1", "true", "yes", "on")
# If the model cites nothing verifiable and no answer sentence overlaps the
# retrieved context above this token-overlap ratio, treat the answer as
# ungrounded and return the honest "insufficient evidence" reply instead.
GROUNDING_MIN_OVERLAP = _env_float("GROUNDING_MIN_OVERLAP", 0.18)

# ── Source priority ──────────────────────────────────────────────────────────────
# Higher number = higher priority. Used to boost ranking when scores are close.
SOURCE_PRIORITY = {
    "commit_kb":      3,   # primary, living knowledge base
    "staff_handbook": 2,   # secondary supporting source
    "website":        2,   # public Takshashila website (publications + blogs)
}
DEFAULT_SOURCE_PRIORITY = 1   # everything else (pdfs, local, legacy)

# Fractional boost applied per priority tier above the base tier.
# e.g. commit_kb (tier 3) → fused_score * (1 + (3-1)*0.12) = *1.24
SOURCE_PRIORITY_BOOST = _env_float("SOURCE_PRIORITY_BOOST", 0.12)

# Human-readable names for known sources (used in citations / UI).
SOURCE_DISPLAY_NAMES = {
    "commit_kb":      "Commit — Takshashila Knowledge Base",
    "staff_handbook": "Takshashila Staff Handbook",
    "website":        "Takshashila Website",
    "publication":    "Takshashila Publication",
    "blog":           "Takshashila Blog",
    "pdf":            "Takshashila PDF",
    "local":          "Local Document",
}


def source_priority(source: str) -> int:
    """Return the priority tier for a source string."""
    return SOURCE_PRIORITY.get((source or "").lower(), DEFAULT_SOURCE_PRIORITY)


def source_display_name(source: str, fallback: str = "") -> str:
    """Return a human-readable name for a source key."""
    return SOURCE_DISPLAY_NAMES.get((source or "").lower(), fallback or source or "Source")


# ── Automated refresh schedule (scripts/scheduler.py) ───────────────────────────
# The knowledge base refreshes itself on a weekly cron. Defaults to every
# Monday at 18:00 India time. All four are overridable from .env.
#   SCHEDULE_DAY       cron day_of_week: mon,tue,wed,thu,fri,sat,sun (or 0-6)
#   SCHEDULE_HOUR      0-23   (local to SCHEDULE_TIMEZONE)
#   SCHEDULE_MINUTE    0-59
#   SCHEDULE_TIMEZONE  IANA tz name, e.g. Asia/Kolkata
# Default: every Monday at 18:00 IST — one incremental update a week.
SCHEDULE_DAY       = os.getenv("SCHEDULE_DAY", "mon")
SCHEDULE_HOUR      = _env_int("SCHEDULE_HOUR", 18)
SCHEDULE_MINUTE    = _env_int("SCHEDULE_MINUTE", 0)
SCHEDULE_TIMEZONE  = os.getenv("SCHEDULE_TIMEZONE", "Asia/Kolkata")

# Scheduler bookkeeping files.
SCHEDULER_LOG      = LOGS_DIR / "scheduler.log"
SCHEDULER_LOCK     = LOGS_DIR / "scheduler.lock"        # single-instance guard
SCHEDULER_STATUS   = LOGS_DIR / "scheduler_status.json" # last-run result, for the UI


# ── Ensure dirs exist ──────────────────────────────────────────────────────────
def ensure_dirs():
    for d in [RAW_HTML_DIR, RAW_PDF_DIR, PROCESSED_DIR, INDEX_DIR,
              INDEX_VERSIONS_DIR, LOGS_DIR, KB_DIR]:
        d.mkdir(parents=True, exist_ok=True)

ensure_dirs()