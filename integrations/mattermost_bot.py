"""
mattermost_bot.py — Mattermost slash-command bot for the Takshashila RAG.

Flow
----
1. A user types:  /askkb What is the leave policy?
2. Mattermost POSTs the slash command to  /mattermost/ask
3. The endpoint validates the token + allowlist, returns an INSTANT ephemeral
   acknowledgement, then runs the work in a background task.
4. The background task calls the EXISTING `src.rag_pipeline.answer()` (unchanged)
   and posts a clean, formatted reply back to the channel with interactive
   buttons (More Details, Related Policies, Export Markdown, and 👍/👎 feedback).
5. Button clicks call back to  /mattermost/action  (feedback is also accepted on
   the legacy  /mattermost/feedback  route).

Design notes
------------
* The RAG pipeline in `src/` is reused as-is and never modified. Search mode
  (used by "Related Policies") calls the existing public
  `src.retriever.retrieve()` — a read-only retrieval call, no LLM generation.
* Secrets are never logged or sent to Mattermost.
* Only stdlib + httpx + fastapi are used.

Run locally:
    uvicorn integrations.mattermost_bot:app --host 0.0.0.0 --port 8000
"""

from __future__ import annotations

import hashlib
import hmac
import secrets
import json
import logging
import os
import re
import textwrap
import threading
import time
import traceback
from collections import OrderedDict
from logging.handlers import RotatingFileHandler
from typing import List, Optional

import httpx
from dotenv import load_dotenv
from fastapi import BackgroundTasks, Body, FastAPI, Form, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse

load_dotenv()

import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src import config  # noqa: E402

# Presentation + feedback live in their own modules; neither touches the pipeline.
from integrations import formatting  # noqa: E402
from integrations.feedback import record_feedback  # noqa: E402

# Delivery layer — completely separate from retrieval. The command parser decides
# WHERE an answer goes; the response router dispatches to a destination handler;
# mattermost_api holds the low-level REST lookups. None of these touch the RAG
# pipeline. (See integrations/response_router.py.)
from integrations import command_parser, help_text, mattermost_api, response_router  # noqa: E402
from integrations.destination_handlers.base import (  # noqa: E402
    DeliveryResult, Requester, ResponsePayload,
)


# ════════════════════════════════════════════════════════════════════════════════
#  Configuration (from environment)
# ════════════════════════════════════════════════════════════════════════════════

MATTERMOST_URL = os.getenv("MATTERMOST_URL", "").rstrip("/")
MATTERMOST_BOT_TOKEN = os.getenv("MATTERMOST_BOT_TOKEN", "").strip()
MATTERMOST_SLASH_TOKEN = os.getenv("MATTERMOST_SLASH_TOKEN", "").strip()

RAG_TOP_K = config._env_int("MATTERMOST_RAG_TOP_K", 5)
RAG_TEMPERATURE = config._env_float("MATTERMOST_RAG_TEMPERATURE", 0.1)
GROQ_MODEL = os.getenv("GROQ_MODEL") or config.GROQ_MODEL

# Search mode retrieves more candidates than a normal answer (no LLM call).
SEARCH_TOP_K = config._env_int("MATTERMOST_SEARCH_TOP_K", 10)

# ── CORS for the static (GitHub Pages) dashboard ──────────────────────────────
# The public dashboard is a static site (GitHub Pages) that calls this backend's
# /api/query over HTTPS. Browsers enforce CORS for those cross-origin requests,
# so the frontend origin(s) must be allow-listed here. Never use "*" together
# with credentials; list the exact GitHub Pages / custom-domain origins instead.
#   API_ALLOWED_ORIGINS=https://gopaltomar.github.io,https://kb.takshashila.org.in
API_ALLOWED_ORIGINS = [
    o.strip() for o in os.getenv("API_ALLOWED_ORIGINS", "").split(",") if o.strip()
]
# Optional shared secret for /api/query (public dashboard callers). If set, the
# request must carry it as the "X-API-Key" header. Leave empty for an
# unauthenticated read-only query API (answers are already grounded + public).
API_KEY = os.getenv("API_KEY", "").strip()

MAX_MESSAGE_CHARS = config._env_int("MATTERMOST_MAX_MESSAGE_CHARS", 12000)
WARM_RAG_ON_STARTUP = os.getenv("MATTERMOST_WARM_RAG_ON_STARTUP", "true").lower() in (
    "1", "true", "yes", "on",
)

# Public base URL of THIS bot service — the callback target for interactive
# buttons. If unset, buttons are omitted (everything else still works).
PUBLIC_BASE_URL = os.getenv("MATTERMOST_BOT_PUBLIC_URL", "").rstrip("/")
_BUTTON_SWITCH = os.getenv(
    "MATTERMOST_ENABLE_BUTTONS",
    os.getenv("MATTERMOST_ENABLE_FEEDBACK", "true"),  # back-compat with old var
).lower() in ("1", "true", "yes", "on")
ENABLE_BUTTONS = bool(PUBLIC_BASE_URL) and _BUTTON_SWITCH
ACTION_URL = f"{PUBLIC_BASE_URL}/mattermost/action" if PUBLIC_BASE_URL else ""

# ── Answer visibility ───────────────────────────────────────────────────────────
# "private" (default): each person's answer is delivered ONLY to them (ephemeral),
#   right inside the channel — so a shared channel never fills up with everyone's
#   Q&A. A user can still share a specific answer with `/askkb public <question>`.
# "public": answers are posted into the channel for everyone (the older behaviour).
#   A user can keep a single answer to themselves with `/askkb private <question>`.
ANSWER_VISIBILITY = os.getenv("MATTERMOST_ANSWER_VISIBILITY", "private").strip().lower()
if ANSWER_VISIBILITY not in ("private", "public"):
    ANSWER_VISIBILITY = "private"

# How a PRIVATE answer reaches the asker:
#   "dm"        (default): a direct message from the bot — fully featured
#               (working 👍/👎 feedback, private Markdown export, delete, etc.).
#   "ephemeral": shown only to the requester inside the channel. Note: Mattermost
#               ephemeral posts can't run interactive buttons or carry file
#               downloads, so those features are unavailable in this mode.
PRIVATE_DELIVERY = os.getenv("MATTERMOST_PRIVATE_DELIVERY", "dm").strip().lower()
if PRIVATE_DELIVERY not in ("dm", "ephemeral"):
    PRIVATE_DELIVERY = "dm"

# Private replies: when on (default), each answer is visible ONLY to the person
# who asked — typed answers come back as an ephemeral in-channel message, voice
# answers as a direct message — so the bot never spams the whole channel.
PRIVATE_REPLIES = os.getenv("MATTERMOST_PRIVATE_REPLIES", "true").lower() in (
    "1", "true", "yes", "on",
)

# ── Voice input ─────────────────────────────────────────────────────────────────
# `/askkb voice` returns a link to a browser page (served at /voice) that
# transcribes speech client-side (Web Speech API) and posts the question +
# answer back into the channel. Requires a public URL + bot token. Links are
# signed + short-lived so only people who ran the command can use them.
ENABLE_VOICE = bool(PUBLIC_BASE_URL) and os.getenv(
    "MATTERMOST_ENABLE_VOICE", "true"
).lower() in ("1", "true", "yes", "on")
VOICE_TTL = config._env_int("MATTERMOST_VOICE_TTL_SECONDS", 1800)  # link validity
_VOICE_SECRET = (
    os.getenv("MATTERMOST_VOICE_SECRET", "")
    or MATTERMOST_SLASH_TOKEN
    or secrets.token_hex(32)
).encode("utf-8")
if not (os.getenv("MATTERMOST_VOICE_SECRET") or MATTERMOST_SLASH_TOKEN):
    # Audit finding: this used to fall back to a hardcoded literal string
    # ("takshashila-voice-fallback-secret") baked into the source. Anyone who
    # had read this codebase would know that key and could forge a valid,
    # signed voice token for any channel_id/user_id within VOICE_TTL, without
    # ever running /askkb voice. A random-per-process secret closes that: the
    # only real-world cost is that any in-flight voice link becomes invalid
    # across a process restart, which is minor given the short default TTL —
    # voice input is disabled entirely, or MATTERMOST_VOICE_SECRET should be
    # set explicitly, for a deployment where that matters.
    logging.getLogger("mattermost_bot").warning(
        "MATTERMOST_VOICE_SECRET and MATTERMOST_SLASH_TOKEN are both unset — "
        "using a random per-process voice-link signing secret. Voice links "
        "will stop working after any restart. Set MATTERMOST_VOICE_SECRET "
        "for a stable secret across restarts."
    )


def _csv_set(raw: str) -> set:
    return {item.strip() for item in (raw or "").split(",") if item.strip()}


# ── Live progress indicator ("typing" stages) ───────────────────────────────────
# While RAG runs, the bot posts one status message and edits it through a few
# stages ("🔍 Searching…" → "🧠 Generating…" → …), then replaces it with the
# finished answer. This only applies to answers delivered to a real channel (your
# DM with the bot, or a public channel) via the bot token; ephemeral answers and
# external --user/--channel/--group shares are unaffected. Requires the bot token.
ENABLE_PROGRESS = os.getenv("MATTERMOST_ENABLE_PROGRESS", "true").lower() in (
    "1", "true", "yes", "on",
)
PROGRESS_INTERVAL = config._env_float("MATTERMOST_PROGRESS_INTERVAL", 1.1)  # seconds/stage
PROGRESS_STAGES_RAG = [
    "🔍 Searching Knowledge Base…",
    "📚 Reading Policies…",
    "🧠 Generating Answer…",
    "✍️ Formatting…",
    "📨 Delivering…",
]
PROGRESS_STAGES_SEARCH = [
    "🔍 Searching Knowledge Base…",
    "✍️ Formatting…",
    "📨 Delivering…",
]


ALLOWED_TEAM_IDS = _csv_set(os.getenv("MATTERMOST_ALLOWED_TEAM_IDS", ""))
ALLOWED_CHANNEL_IDS = _csv_set(os.getenv("MATTERMOST_ALLOWED_CHANNEL_IDS", ""))

# ── Enterprise routing (--user / --channel / --group destinations) ───────────────
# When on (default), an answer can be delivered to another user's DM, a channel or
# a group DM. Requires the bot token (same one DM delivery already uses). The
# default `/askkb <question>` and `/askkb --me …` behaviour is unaffected either way.
ENABLE_ROUTING = os.getenv("MATTERMOST_ENABLE_ROUTING", "true").lower() in (
    "1", "true", "yes", "on",
)

# Group direct messages (--group / -g / 👥 Share to Group). Enabled by default;
# set MATTERMOST_ENABLE_GROUP=false to hide the group option on deployments that
# only use channels + one-to-one direct messages.
ENABLE_GROUP_DESTINATION = os.getenv("MATTERMOST_ENABLE_GROUP", "true").lower() in (
    "1", "true", "yes", "on",
)

# ── Share buttons (bonus) ────────────────────────────────────────────────────────
# The 👤/📢/👥 "Share …" buttons + ⬇️ Download PDF beneath an answer. They reuse the
# already-generated answer (no second RAG run) and open a Mattermost dialog to
# collect the target. Needs the public URL (for the dialog callback) + bot token.
ENABLE_SHARE_BUTTONS = (
    ENABLE_ROUTING
    and bool(PUBLIC_BASE_URL)
    and bool(MATTERMOST_BOT_TOKEN)
    and os.getenv("MATTERMOST_ENABLE_SHARE_BUTTONS", "true").lower() in (
        "1", "true", "yes", "on",
    )
)
DIALOG_URL = f"{PUBLIC_BASE_URL}/mattermost/dialog" if PUBLIC_BASE_URL else ""

# ── PDF export (bonus) ───────────────────────────────────────────────────────────
# ⬇️ Download PDF renders the already-generated answer to a PDF (via pymupdf,
# already a project dependency) and uploads it — no second RAG run. Requires the
# bot token (to upload the file) and pymupdf to be importable.
try:
    import fitz as _fitz  # PyMuPDF
    _HAVE_PYMUPDF = True
except Exception:  # pragma: no cover - pymupdf is a listed dependency
    _fitz = None
    _HAVE_PYMUPDF = False

ENABLE_PDF_EXPORT = (
    _HAVE_PYMUPDF
    and bool(MATTERMOST_BOT_TOKEN)
    and os.getenv("MATTERMOST_ENABLE_PDF_EXPORT", "true").lower() in (
        "1", "true", "yes", "on",
    )
)

USAGE_HINT = (
    "Please ask a question, for example:\n\n"
    "`/askkb What is the leave policy?`\n\n"
    "By default only **you** see the answer in this channel. You can also:\n"
    "• `/askkb public <question>` — share the answer with the whole channel\n"
    "• `/askkb short <question>` — a brief answer\n"
    "• `/askkb detailed <question>` — a fuller answer\n"
    "• `/askkb search <keywords>` — list matching documents (no AI answer)\n"
    "• `/askkb voice` — 🎤 ask by speaking instead of typing\n\n"
    "**Send the answer somewhere specific:**\n"
    "• `/askkb --me <question>` — privately to you (default)\n"
    "• `/askkb --user abhishek.k <question>` — to a colleague's direct messages\n"
    "• `/askkb --channel research <question>` — into a channel"
    + ("\n• `/askkb --group abhishek.k,nithiya,amit <question>` — to a group message"
       if ENABLE_GROUP_DESTINATION else "")
)
ACK_MESSAGE = "🔍 Searching Takshashila Knowledge Base..."
ACK_SEARCH = "🔍 Searching documents in the Takshashila Knowledge Base..."
GENERIC_ERROR_MESSAGE = (
    "I faced an error while searching the knowledge base. "
    "Please try again or contact the admin."
)


# ════════════════════════════════════════════════════════════════════════════════
#  Logging  →  data/logs/mattermost_bot.log
# ════════════════════════════════════════════════════════════════════════════════

LOG_FILE = config.LOGS_DIR / "mattermost_bot.log"


def _build_logger() -> logging.Logger:
    log = logging.getLogger("mattermost_bot")
    if log.handlers:
        return log
    log.setLevel(logging.INFO)
    fmt = logging.Formatter(
        "%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    sh = logging.StreamHandler()
    sh.setFormatter(fmt)
    log.addHandler(sh)
    try:
        LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
        fh = RotatingFileHandler(LOG_FILE, maxBytes=2_000_000, backupCount=3, encoding="utf-8")
        fh.setFormatter(fmt)
        log.addHandler(fh)
    except Exception as exc:
        log.warning(f"Could not attach file logger at {LOG_FILE}: {exc}")
    return log


logger = _build_logger()


# ════════════════════════════════════════════════════════════════════════════════
#  RAG warm-up
# ════════════════════════════════════════════════════════════════════════════════

_warm_lock = threading.Lock()
_embed_model_loaded = False


def warm_rag_resources() -> None:
    """
    Ensure RAG resources are loaded AND reflect the CURRENT knowledge-base
    version (thread-safe).

    The embedding model is a one-time heavy load (it never changes at runtime).
    The FAISS index + BM25 are refreshed to the active KB version on every call:
    ``vector_store.load_index()`` is version-aware, so it re-reads the heavy index
    files only when a rebuild happened in another process, and BM25 rebuilds
    itself against the new metadata automatically. This is what lets the bot pick
    up a KB rebuild with NO restart.

    (Note: every answer also goes through ``vector_store.search`` → the same
    version-aware ``load_index``, so freshness holds even for callers that never
    invoke this function.)
    """
    global _embed_model_loaded
    with _warm_lock:
        from src import vector_store, embeddings, retriever
        if not _embed_model_loaded:
            embeddings._get_model()
            _embed_model_loaded = True
        vector_store.load_index()       # version-aware: reloads only on change
        retriever.ensure_bm25_ready()   # rebuilds only when the metadata changed
        try:
            logger.info(f"RAG resources current — {vector_store.ntotal()} vectors "
                        f"(KB version={vector_store.loaded_fingerprint()})")
        except Exception:
            logger.info("RAG resources current.")


# ════════════════════════════════════════════════════════════════════════════════
#  Answer cache (powers export / copy / preview button callbacks)
# ════════════════════════════════════════════════════════════════════════════════
#
# Keyed by the Mattermost post id we get back when we create the answer post.
# Bounded + thread-safe. On a cache miss (e.g. after a restart) the action
# handlers fall back to regenerating from the question carried in the button
# context, so callbacks stay functional without this cache.

_CACHE_MAX = 300
_ANSWER_CACHE: "OrderedDict[str, dict]" = OrderedDict()
_cache_lock = threading.Lock()


def _cache_put(post_id: str, payload: dict) -> None:
    if not post_id:
        return
    with _cache_lock:
        _ANSWER_CACHE[post_id] = payload
        _ANSWER_CACHE.move_to_end(post_id)
        while len(_ANSWER_CACHE) > _CACHE_MAX:
            _ANSWER_CACHE.popitem(last=False)


def _cache_get(post_id: str) -> Optional[dict]:
    with _cache_lock:
        return _ANSWER_CACHE.get(post_id)


# ════════════════════════════════════════════════════════════════════════════════
#  Message building (presentation lives in formatting.py)
# ════════════════════════════════════════════════════════════════════════════════

def build_message(question: str, result: dict, mode: str = "normal",
                  response_time: Optional[float] = None) -> str:
    """Render the clean Mattermost message, then enforce the length budget."""
    return _truncate(
        formatting.format_answer(question, result, mode=mode, response_time=response_time)
    )


def format_answer(question: str, result: dict) -> str:
    """Backwards-compatible alias."""
    return build_message(question, result)


def _truncate(message: str) -> str:
    if len(message) <= MAX_MESSAGE_CHARS:
        return message
    notice = "\n\n_Answer truncated because it exceeded Mattermost message length._"
    cut = MAX_MESSAGE_CHARS - len(notice)
    return message[: max(cut, 0)].rstrip() + notice


# ════════════════════════════════════════════════════════════════════════════════
#  Interactive button attachments
# ════════════════════════════════════════════════════════════════════════════════

def _sign_action_context(ctx: dict) -> str:
    """HMAC-sign a button's context dict (everything except the signature
    itself), using the same secret/approach as _voice_token. Canonical JSON
    (sorted keys) makes the signature independent of dict insertion order."""
    canonical = json.dumps({k: v for k, v in ctx.items() if k != "_sig"},
                           sort_keys=True, separators=(",", ":"))
    return hmac.new(_VOICE_SECRET, canonical.encode("utf-8"), hashlib.sha256).hexdigest()[:32]


def _action_context_ok(ctx: dict) -> bool:
    sig = ctx.get("_sig")
    if not sig:
        return False
    return hmac.compare_digest(sig, _sign_action_context(ctx))


def _action(name: str, action: str, **context) -> dict:
    """
    One interactive button that calls back to /mattermost/action.

    Audit finding: /mattermost/action had NO authentication of any kind —
    unlike /mattermost/ask (slash-token check) and /voice (HMAC-signed,
    time-limited token), any internet client that could reach this endpoint
    (it's on the same public-facing FastAPI app, reachable for the same
    reason the voice feature needs to be) could POST an arbitrary context and
    trigger delete_all_confirm, export_pdf, more_details, etc. with fabricated
    values — no interaction with Mattermost's UI required. Every button's
    context is now HMAC-signed at build time with the same secret used for
    voice tokens, and /mattermost/action verifies that signature before
    dispatching to any action branch.
    """
    ctx = {"action": action}
    ctx.update(context)
    ctx["_sig"] = _sign_action_context(ctx)
    return {"name": name, "integration": {"url": ACTION_URL, "context": ctx}}


def _button_groups(question: str) -> List[dict]:
    """
    The non-feedback button groups shown beneath an answer:
      • 📌 Suggested Follow-Ups — More Details / Related Policies
      • 📤 Export                — Export Markdown / Download PDF
    Kept separate so the feedback group can be swapped for a confirmation card
    (after a 👍 / 👎 click) while these stay clickable.
    """
    q = (question or "")[:400]
    export_actions = [_action("📄 Export Markdown", "export_markdown", question=q)]
    # Download PDF reuses the already-generated answer (no second RAG run).
    if ENABLE_PDF_EXPORT:
        export_actions.append(_action("⬇️ Download PDF", "export_pdf", question=q))
    return [
        {
            "title": "📌 Suggested Follow-Ups",
            "color": "#1c75bc",
            "actions": [
                _action("🔗 Related Policies", "related", question=q),
            ],
        },
        {
            "title": "📤 Export",
            "color": "#27ae60",
            "actions": export_actions,
        },
    ]


def _share_group(question: str) -> dict:
    """
    Bonus 'Share this answer' controls. Each button opens a Mattermost dialog to
    collect a target and then re-delivers the ALREADY-GENERATED answer via the
    response router — the RAG pipeline is never re-run.

    Only User + Channel are shown by default; the group option appears only when
    group direct messages are explicitly enabled (MATTERMOST_ENABLE_GROUP).
    """
    q = (question or "")[:400]
    actions = [
        _action("👤 Share to User", "share_user", question=q),
        _action("📢 Share to Channel", "share_channel", question=q),
    ]
    if ENABLE_GROUP_DESTINATION:
        actions.append(_action("👥 Share to Group", "share_group", question=q))
    return {
        "title": "📨 Share this answer",
        "color": "#8e44ad",
        "actions": actions,
    }


def _share_tail(question: str) -> List[dict]:
    """Share controls, included only when share buttons are enabled."""
    return [_share_group(question)] if ENABLE_SHARE_BUTTONS else []


def _feedback_group(question: str) -> dict:
    """The 👍 / 👎 attachment block."""
    q = (question or "")[:400]
    return {
        "title": "Was this helpful?",
        "actions": [
            _action("👍 Helpful", "feedback", verdict="helpful", question=q),
            _action("👎 Not Helpful", "feedback", verdict="not_helpful", question=q),
        ],
    }


def _manage_group(question: str) -> dict:
    """
    The 🗑️ delete controls shown beneath an answer.

      • Delete this response — removes just this post.
      • Delete all           — removes every bot response in this channel
                               (asks for confirmation first).

    Deletion uses the Mattermost REST API, so it is only offered when a bot
    token is configured.
    """
    q = (question or "")[:400]
    return {
        "title": "🗑️ Manage this response",
        "color": "#c0392b",
        "actions": [
            _action("🗑️ Delete this response", "delete_one"),
            _action("🧹 Delete all", "delete_all", question=q),
        ],
    }


def _confirm_delete_group(question: str) -> dict:
    """Replaces the manage group with a Yes/Cancel confirmation for 'Delete all'."""
    q = (question or "")[:400]
    return {
        "title": "⚠️ Delete ALL bot responses in this channel?",
        "color": "#c0392b",
        "actions": [
            _action("✅ Yes, delete all", "delete_all_confirm"),
            _action("✖️ Cancel", "delete_cancel", question=q),
        ],
    }


def _manage_tail(question: str) -> List[dict]:
    """Delete controls, included only when deletion is actually possible."""
    return [_manage_group(question)] if (MATTERMOST_BOT_TOKEN and MATTERMOST_URL) else []


def _answer_attachments(question: str) -> List[dict]:
    """The full, fresh button set beneath an answer (in display order)."""
    return (
        _button_groups(question)
        + _share_tail(question)
        + [_feedback_group(question)]
        + _manage_tail(question)
    )


def _feedback_attachment(question: str, mode: str = "normal") -> dict:
    """Full interactive button set rendered beneath a fresh answer."""
    return {"attachments": _answer_attachments(question)}


def _confirmation_card(verdict: str) -> dict:
    """
    A non-interactive 'card' that REPLACES the 👍/👎 buttons after a click,
    giving a visible, professional transition right inside the post.
    """
    if verdict == "helpful":
        return {
            "color": "#2ecc71",
            "title": "🎉 Hurray!",
            "text": "Thanks for the thumbs up — glad this hit the mark! 🙌",
        }
    return {
        "color": "#e67e22",
        "title": "🙏 Thanks for the honest feedback",
        "text": "Logged it — we're on it, and answers will keep getting better. 🔧",
    }


# Private 'popup' lines shown only to the person who clicked.
FEEDBACK_POPUP = {
    "helpful": "🎉 Woohoo! Thanks for the 👍 — your feedback keeps the assistant sharp. 🚀",
    "not_helpful": "📝 Got it — thanks for flagging this. We'll use it to improve. 💪",
}


# ════════════════════════════════════════════════════════════════════════════════
#  Posting back to Mattermost
# ════════════════════════════════════════════════════════════════════════════════

def post_to_channel(channel_id: str, message: str, response_url: Optional[str],
                    props: Optional[dict] = None) -> Optional[str]:
    """
    Deliver `message` to Mattermost. Returns the created post id when the
    bot-token REST path is used (needed for button callbacks), else None.

    Interactive buttons require the bot-token path. The response_url fallback
    (only available for slash commands) can also carry attachments, but voice
    answers have no response_url, so for them the bot token is mandatory.
    """
    n_groups = len((props or {}).get("attachments", []) or [])

    if MATTERMOST_BOT_TOKEN and MATTERMOST_URL and channel_id:
        url = f"{MATTERMOST_URL}/api/v4/posts"
        headers = {"Authorization": f"Bearer {MATTERMOST_BOT_TOKEN}"}
        payload = {"channel_id": channel_id, "message": message}
        if props:
            payload["props"] = props
        try:
            with httpx.Client(timeout=20.0) as client:
                resp = client.post(url, headers=headers, json=payload)
            if resp.status_code in (200, 201):
                pid = None
                try:
                    pid = resp.json().get("id")
                except Exception:
                    pass
                logger.info(
                    f"posted via bot-token  channel={channel_id!r}  post_id={pid!r}  "
                    f"button_groups={n_groups}  msg_chars={len(message)}"
                )
                return pid
            # Surface the server's reason — this is what to read when buttons vanish.
            body_preview = ""
            try:
                body_preview = resp.text[:300]
            except Exception:
                pass
            logger.error(
                f"Bot-token post failed (HTTP {resp.status_code}) channel={channel_id!r} "
                f"button_groups={n_groups}; body={body_preview!r}; "
                f"{'trying response_url' if response_url else 'NO response_url fallback'}."
            )
        except Exception as exc:
            logger.error(f"Bot-token post raised {type(exc).__name__}; "
                         f"{'trying response_url' if response_url else 'no fallback'}.")

    if response_url:
        try:
            body = {"response_type": "in_channel", "text": message}
            if props and props.get("attachments"):
                body["attachments"] = props["attachments"]
            with httpx.Client(timeout=20.0) as client:
                resp = client.post(response_url, json=body)
            if resp.status_code == 200:
                logger.info(f"posted via response_url  button_groups={n_groups}")
                return None
            logger.error(f"response_url post failed (HTTP {resp.status_code}).")
        except Exception as exc:
            logger.error(f"response_url post raised {type(exc).__name__}.")
        return None

    logger.error(
        "No delivery channel available: set MATTERMOST_BOT_TOKEN (+ MATTERMOST_URL) "
        "or ensure Mattermost sends a response_url. (Voice answers REQUIRE the bot token.)"
    )
    return None


def _patch_post(post_id: str, message: str, props: Optional[dict] = None) -> bool:
    """
    Edit an existing bot post in place (``PUT /api/v4/posts/{id}/patch``).

    Used to animate the live progress indicator and to replace the status message
    with the finished answer. Returns True on success. Requires the bot token.
    """
    if not (MATTERMOST_BOT_TOKEN and MATTERMOST_URL and post_id):
        return False
    url = f"{MATTERMOST_URL}/api/v4/posts/{post_id}/patch"
    headers = {"Authorization": f"Bearer {MATTERMOST_BOT_TOKEN}"}
    body: dict = {"message": message}
    if props is not None:
        body["props"] = props            # {} clears buttons; a dict sets them
    try:
        with httpx.Client(timeout=20.0) as client:
            resp = client.put(url, headers=headers, json=body)
        if resp.status_code in (200, 201):
            return True
        logger.error(f"Patch post failed (HTTP {resp.status_code}) post_id={post_id!r}.")
    except Exception as exc:
        logger.error(f"Patch post raised {type(exc).__name__} for post_id={post_id!r}.")
    return False


class _ProgressIndicator:
    """
    A single status post that animates through ``stages`` while RAG runs, then is
    edited into the finished answer by :meth:`finalize`.

    Non-invasive: it runs the animation on a daemon thread and never touches the
    RAG pipeline. If anything fails (no bot token, a failed post/patch), it
    degrades silently to a normal single post so behaviour is never worse than
    before. Only used for answers delivered to a real channel via the bot token.
    """

    def __init__(self, channel_id: str, stages: List[str]):
        self.channel_id = channel_id
        self.stages = stages or ["🔍 Searching Knowledge Base…"]
        self.post_id: Optional[str] = None
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._thread: Optional[threading.Thread] = None

    def start(self) -> bool:
        """Post the first stage as a real bot message and begin animating."""
        self.post_id = post_to_channel(self.channel_id, self.stages[0], None, props=None)
        if not self.post_id:
            return False
        self._thread = threading.Thread(target=self._animate, daemon=True)
        self._thread.start()
        return True

    def _animate(self) -> None:
        i = 1
        while i < len(self.stages):
            if self._stop.wait(PROGRESS_INTERVAL):
                return                       # finalized/cancelled during the wait
            with self._lock:
                if self._stop.is_set():
                    return
                _patch_post(self.post_id, self.stages[i])
            i += 1
        # Hold on the last stage until finalize() replaces it.

    def finalize(self, message: str, props: Optional[dict] = None) -> Optional[str]:
        """Stop animating and edit the status post into the final ``message``."""
        with self._lock:
            self._stop.set()
        if self._thread:
            self._thread.join(timeout=PROGRESS_INTERVAL + 1.0)
        # props=None would leave the status post with no attachments; pass {} to
        # explicitly clear, or the real button attachments when present.
        patch_props = props if props is not None else {}
        if self.post_id and _patch_post(self.post_id, message, patch_props):
            return self.post_id
        # Patch failed → fall back to a fresh post so the answer is never lost.
        logger.warning("Progress finalize patch failed; posting the answer fresh.")
        if self.post_id:
            _delete_post(self.post_id)
        return post_to_channel(self.channel_id, message, None, props=props)

    def cancel(self) -> None:
        """Abort the indicator and remove the status post (used on fallback)."""
        with self._lock:
            self._stop.set()
        if self._thread:
            self._thread.join(timeout=PROGRESS_INTERVAL + 1.0)
        if self.post_id:
            _delete_post(self.post_id)
            self.post_id = None


def _upload_markdown_and_post(channel_id: str, filename: str,
                              md_content: str, message: str) -> bool:
    """Upload `md_content` as a .md file and attach it to a new channel post."""
    if not (MATTERMOST_BOT_TOKEN and MATTERMOST_URL and channel_id):
        return False
    headers = {"Authorization": f"Bearer {MATTERMOST_BOT_TOKEN}"}
    try:
        with httpx.Client(timeout=30.0) as client:
            files = {"files": (filename, md_content.encode("utf-8"), "text/markdown")}
            data = {"channel_id": channel_id}
            up = client.post(f"{MATTERMOST_URL}/api/v4/files",
                             headers=headers, data=data, files=files)
            if up.status_code not in (200, 201):
                logger.error(f"File upload failed (HTTP {up.status_code}).")
                return False
            infos = up.json().get("file_infos") or []
            file_ids = [fi["id"] for fi in infos if fi.get("id")]
            if not file_ids:
                return False
            post = client.post(
                f"{MATTERMOST_URL}/api/v4/posts",
                headers=headers,
                json={"channel_id": channel_id, "message": message, "file_ids": file_ids},
            )
            return post.status_code in (200, 201)
    except Exception as exc:
        logger.error(f"Markdown export raised {type(exc).__name__}.")
        return False


# ── Post deletion (powers the 🗑️ Delete buttons) ────────────────────────────────


def _auth_headers() -> dict:
    return {"Authorization": f"Bearer {MATTERMOST_BOT_TOKEN}"}


def _get_bot_user_id() -> Optional[str]:
    """
    Resolve + cache this bot's own user id (needed to find its own posts).

    Delegates to ``mattermost_api.get_bot_user_id`` so there is a single
    implementation shared with the routing layer.
    """
    return mattermost_api.get_bot_user_id()


def _delete_post(post_id: str) -> bool:
    """Delete a single post (the bot may delete its own posts)."""
    if not (MATTERMOST_BOT_TOKEN and MATTERMOST_URL and post_id):
        return False
    try:
        with httpx.Client(timeout=15.0) as client:
            r = client.delete(f"{MATTERMOST_URL}/api/v4/posts/{post_id}", headers=_auth_headers())
        return r.status_code == 200
    except Exception as exc:
        logger.error(f"Delete post raised {type(exc).__name__}.")
        return False


def _post_ephemeral_via_url(response_url: str, message: str,
                            props: Optional[dict] = None) -> bool:
    """
    Deliver an answer visible ONLY to the requesting user, in-channel, using the
    slash command's response_url. This is how private mode keeps a shared channel
    free of everyone's Q&A.
    """
    if not response_url:
        return False
    n_groups = len((props or {}).get("attachments", []) or [])
    body = {"response_type": "ephemeral", "text": message}
    if props and props.get("attachments"):
        body["attachments"] = props["attachments"]
    try:
        with httpx.Client(timeout=20.0) as client:
            resp = client.post(response_url, json=body)
        ok = resp.status_code == 200
        (logger.info if ok else logger.error)(
            f"posted ephemeral via response_url  ok={ok}  button_groups={n_groups}  "
            f"msg_chars={len(message)}"
        )
        return ok
    except Exception as exc:
        logger.error(f"Ephemeral response_url post raised {type(exc).__name__}.")
        return False


def _get_or_create_dm_channel(user_id: str) -> Optional[str]:
    """
    Return the bot↔user direct-message channel id (created if needed).

    Delegates to ``mattermost_api.get_or_create_dm_channel`` so DM opening has a
    single implementation shared with the routing layer.
    """
    return mattermost_api.get_or_create_dm_channel(user_id)


def _delete_all_bot_posts(channel_id: str, max_scan: int = 200) -> int:
    """
    Delete every post authored by THIS bot in the given channel (most recent
    `max_scan` posts). Returns the number deleted. Only the bot's own posts are
    removed; user messages and system posts are left untouched.
    """
    bot_id = _get_bot_user_id()
    if not (bot_id and MATTERMOST_URL and channel_id):
        return 0
    deleted = 0
    try:
        with httpx.Client(timeout=25.0) as client:
            r = client.get(
                f"{MATTERMOST_URL}/api/v4/channels/{channel_id}/posts",
                headers=_auth_headers(), params={"per_page": max_scan},
            )
            if r.status_code != 200:
                logger.error(f"List channel posts failed (HTTP {r.status_code}).")
                return 0
            posts = (r.json() or {}).get("posts", {}) or {}
            for pid, post in posts.items():
                if post.get("user_id") == bot_id and not post.get("type"):
                    dr = client.delete(f"{MATTERMOST_URL}/api/v4/posts/{pid}",
                                       headers=_auth_headers())
                    if dr.status_code == 200:
                        deleted += 1
                        _ANSWER_CACHE.pop(pid, None)
    except Exception as exc:
        logger.error(f"Delete-all raised {type(exc).__name__}.")
    return deleted


# ════════════════════════════════════════════════════════════════════════════════
#  Background task: run RAG (or search) and post the result
# ════════════════════════════════════════════════════════════════════════════════

def run_rag_and_reply(
    question: str,
    channel_id: str,
    response_url: Optional[str],
    user_name: str,
    user_id: str,
    channel_name: str,
    mode: str = "normal",
    voice: bool = False,
    visibility: str = "private",
    dm_user_id: str = "",
    destination: "Optional[command_parser.Destination]" = None,
    team_id: str = "",
) -> None:
    """
    Heavy lifting, executed off the request path so Mattermost never times out.

    Retrieval and delivery are separated: this function builds the answer once,
    then either delivers it via the existing private/public logic (default), or —
    when ``destination`` is an explicit external target (``--user`` / ``--channel``
    / ``--group``) — hands the rendered answer to the response router and confirms
    to the requester. The RAG pipeline is identical in every case.

    Delivery (when ``destination`` is None or ``me``):
      * visibility == "public"  → posted into the channel for everyone.
      * visibility == "private" → only the requester sees it:
          - typed questions  → ephemeral in-channel (via response_url),
          - voice questions  → a private DM from the bot (no response_url exists),
          falling back to a normal post only if neither private channel is usable.
    """
    confidence = "none"
    retrieval_time = generation_time = 0.0
    error_note = ""
    _t_start = time.perf_counter()
    voice_tag = "🎤 *Asked by voice*\n\n" if voice else ""
    private = (visibility == "private")
    progress = None                      # live "typing" indicator (self-answers only)
    # Delivery target for a self-answer (resolved up front so the progress
    # indicator appears in the right place while RAG runs). External destinations
    # ignore these — the response router handles their delivery.
    target_channel = channel_id
    post_response_url = response_url
    ephemeral_only = False

    try:
        warm_rag_resources()

        # ── Resolve the self-answer delivery target + start a live indicator ───
        # (Only for the default self-answer — external --user/--channel/--group
        # shares are delivered by the response router below and are untouched.)
        self_answer = destination is None or not destination.is_external
        if self_answer:
            if private:
                use_dm = bool(dm_user_id) and (PRIVATE_DELIVERY == "dm" or not response_url)
                if use_dm:
                    dm = _get_or_create_dm_channel(dm_user_id)
                    if dm:
                        target_channel = dm
                        post_response_url = None      # DM is a real channel → bot-token post
                    elif response_url:
                        ephemeral_only = True         # couldn't open a DM → ephemeral fallback
                    # else: no DM + no response_url → falls through to a channel post
                else:
                    ephemeral_only = bool(response_url)

            # A live progress indicator needs a real channel to edit a post in, so
            # it runs for DM/public delivery but not for ephemeral (which can't be
            # edited). Degrades silently if the status post can't be created.
            if ENABLE_PROGRESS and not ephemeral_only and MATTERMOST_BOT_TOKEN:
                stages = PROGRESS_STAGES_SEARCH if mode == "search" else PROGRESS_STAGES_RAG
                indicator = _ProgressIndicator(target_channel, stages)
                if indicator.start():
                    progress = indicator

        # ── Build the message (search = no LLM; otherwise a full RAG answer) ───
        # answer_sources holds the EXACT grounding documents the answer was built
        # from (after citation verification). These — and only these — are what
        # the 🔗 Related Policies button later shows, so references are never a
        # fresh, potentially-irrelevant KB search.
        answer_sources: List[dict] = []
        if mode == "search":
            from src.retriever import retrieve
            chunks = retrieve(query=question, top_k=SEARCH_TOP_K, use_hybrid=True)
            retrieval_time = time.perf_counter() - _t_start
            base = formatting.format_search_results(question, chunks, retrieval_time)
            has_answer = False
        else:
            from src.rag_pipeline import answer as rag_answer
            result = rag_answer(
                query=question, top_k=RAG_TOP_K, model=GROQ_MODEL,
                temperature=RAG_TEMPERATURE, source=None, category=None, use_hybrid=True,
                mode=mode,
            )
            confidence = result.get("confidence", "none")
            retrieval_time = float(result.get("retrieval_time") or 0.0)
            generation_time = float(result.get("generation_time") or 0.0)
            response_time = time.perf_counter() - _t_start
            base = build_message(question, result, mode=mode, response_time=response_time)
            has_answer = confidence != "none"
            answer_sources = formatting.displayed_sources(
                result.get("sources") or [], result.get("answer") or "")

        message = _truncate(voice_tag + base)

        # ── External destination: hand off to the response router ──────────────
        # Retrieval is complete. If the user targeted another user / channel /
        # group, delivery is the router's job — the retrieval engine above never
        # knew (or needed to know) where this answer would go.
        if destination is not None and destination.is_external:
            routed_props = (
                _feedback_attachment(question, mode=mode)
                if (ENABLE_BUTTONS and has_answer) else None
            )
            payload = ResponsePayload(
                question=question,
                message=message,
                props=routed_props,
                metadata={
                    "mode": mode,
                    "confidence": confidence,
                    "retrieval_time": retrieval_time,
                    "generation_time": generation_time,
                },
                citations=answer_sources,
                mode=mode,
            )
            requester = Requester(
                user_id=user_id, user_name=user_name,
                team_id=team_id, channel_id=channel_id,
            )
            delivery = response_router.send_response(
                destination, payload, requester,
                allowed_channel_ids=(ALLOWED_CHANNEL_IDS or None),
            )
            note = delivery.confirmation if delivery.ok else f"⚠️ {delivery.error}"
            # Confirm privately to the requester. Slash commands carry a
            # response_url; button/dialog-driven shares fall back to an ephemeral.
            if response_url:
                _post_ephemeral_via_url(response_url, note)
            elif user_id and channel_id:
                mattermost_api.post_ephemeral(channel_id, user_id, note)
            return

        # ── Decide the delivery target ─────────────────────────────────────────
        # (The target was already resolved above so the progress indicator could
        # start in the right place; here we simply deliver into it.)

        # ── Ephemeral delivery (in-channel, only the requester sees it) ────────
        if ephemeral_only:
            props = None
            if ENABLE_BUTTONS and has_answer:
                props = {"attachments": [_feedback_group(question)]}
            _post_ephemeral_via_url(response_url, message, props)
            return

        # ── Full post (channel for public, DM for private) with all buttons ───
        props = None
        if ENABLE_BUTTONS and has_answer:
            props = _feedback_attachment(question, mode=mode)

        # If a live progress indicator is running, edit its status post into the
        # finished answer; otherwise post fresh. Either way we get the post id.
        if progress is not None:
            post_id = progress.finalize(message, props)
            progress = None
        else:
            post_id = post_to_channel(target_channel, message, post_response_url, props=props)

        # Cache so Export Markdown, the feedback transition and 🔗 Related
        # Policies can act instantly (and rebuild the post's buttons) without
        # re-running the pipeline. `sources` are the EXACT grounding documents
        # this answer was drawn from — what Related Policies shows.
        if post_id:
            _cache_put(post_id, {
                "question": question,
                "mode": mode,
                "message": message,
                "sources": answer_sources,
            })

    except Exception:
        error_note = "exception"
        logger.error("RAG task failed:\n" + traceback.format_exc())
        try:
            # If a live indicator is running, edit it into the error message so it
            # doesn't hang on "Delivering…"; otherwise post the error normally.
            if progress is not None:
                progress.finalize(GENERIC_ERROR_MESSAGE, {})
                progress = None
            else:
                post_to_channel(channel_id, GENERIC_ERROR_MESSAGE, response_url)
        except Exception:
            logger.error("Also failed to post the error message to Mattermost.")

    finally:
        logger.info(
            "request_handled  "
            f"user_name={user_name!r}  user_id={user_id!r}  "
            f"channel_name={channel_name!r}  mode={mode}  "
            f"confidence={confidence}  "
            f"retrieval_time={retrieval_time:.2f}s  generation_time={generation_time:.2f}s  "
            f"error={error_note or 'none'}  question={question!r}"
        )


# ════════════════════════════════════════════════════════════════════════════════
#  Helpers used by the action callbacks (cache-miss fallbacks)
# ════════════════════════════════════════════════════════════════════════════════

def _regen_for_callback(question: str) -> dict:
    """Re-run the pipeline for a question (used when the cache has no entry)."""
    warm_rag_resources()
    from src.rag_pipeline import answer as rag_answer
    return rag_answer(
        query=question, top_k=RAG_TOP_K, model=GROQ_MODEL,
        temperature=RAG_TEMPERATURE, source=None, category=None, use_hybrid=True,
    )


def _related_task(channel_id: str, question: str, post_id: str) -> None:
    """
    Post the 🔗 Related Policies list: the EXACT documents this answer was drawn
    from. It reuses the grounding sources cached with the answer (no re-search,
    so no unrelated documents can appear). On a genuine cache miss the answer is
    regenerated and its own grounding sources are used — still never a broad
    keyword search.
    """
    cached = _cache_get(post_id)
    if cached is not None and "sources" in cached:
        sources = cached.get("sources") or []
    else:
        try:
            result = _regen_for_callback(question)
            sources = formatting.displayed_sources(
                result.get("sources") or [], result.get("answer") or "")
        except Exception:
            logger.error("Related regeneration failed:\n" + traceback.format_exc())
            return
    message = _truncate(
        formatting.format_search_results(question, sources, 0.0, kind="related")
    )
    post_to_channel(channel_id, message, None, props=None)


def _export_markdown_task(channel_id: str, question: str, md_content: Optional[str]) -> None:
    if md_content is None:
        try:
            result = _regen_for_callback(question)
            md_content = build_message(question, result, mode="normal")
        except Exception:
            logger.error("Export regeneration failed:\n" + traceback.format_exc())
            return
    safe = "".join(c if c.isalnum() else "_" for c in (question or "answer"))[:40] or "answer"
    _upload_markdown_and_post(
        channel_id, f"takshashila_{safe}.md", md_content,
        "📄 **Exported answer** (Markdown) — download below.",
    )


def _markdown_to_pdf_bytes(markdown_text: str) -> Optional[bytes]:
    """
    Render answer markdown to a simple, clean multi-page PDF using pymupdf
    (already a project dependency). Markdown markers are lightly stripped so the
    text reads well; this is a portable export, not a full markdown renderer.
    Returns the PDF bytes, or ``None`` if pymupdf is unavailable.
    """
    if not (_HAVE_PYMUPDF and _fitz):
        return None

    # Light markdown → plain text (headings, bullets, bold/italics, links).
    lines: List[str] = []
    for raw in (markdown_text or "").splitlines():
        line = raw.rstrip()
        line = re.sub(r"^#{1,6}\s*", "", line)                       # headings
        line = re.sub(r"^\s*[*\-]\s+", "• ", line)                    # bullets
        line = re.sub(r"\*\*(.+?)\*\*", r"\1", line)                  # bold
        line = re.sub(r"(?<!\*)\*(?!\*)(.+?)\*(?!\*)", r"\1", line)   # italics
        line = re.sub(r"\[([^\]]+)\]\(([^)]+)\)", r"\1 (\2)", line)    # links
        lines.append(line)
    body = "\n".join(lines).strip() or "No content."

    try:
        doc = _fitz.open()
        margin, font_size, leading = 54, 11, 15
        page = doc.new_page()
        width, height = page.rect.width, page.rect.height
        y = margin
        # Wrap text to the page width using a rough character budget.
        max_chars = int((width - 2 * margin) / (font_size * 0.5))
        for para in body.split("\n"):
            wrapped = textwrap.wrap(para, width=max_chars) or [""]
            for chunk in wrapped:
                if y > height - margin:
                    page = doc.new_page()
                    y = margin
                page.insert_text((margin, y), chunk, fontsize=font_size, fontname="helv")
                y += leading
        pdf_bytes = doc.tobytes()
        doc.close()
        return pdf_bytes
    except Exception:
        logger.error("PDF render failed:\n" + traceback.format_exc())
        return None


def _export_pdf_task(channel_id: str, question: str, md_content: Optional[str]) -> None:
    """Render the (cached or regenerated) answer to a PDF and upload it."""
    if md_content is None:
        try:
            result = _regen_for_callback(question)
            md_content = build_message(question, result, mode="normal")
        except Exception:
            logger.error("PDF export regeneration failed:\n" + traceback.format_exc())
            return
    pdf_bytes = _markdown_to_pdf_bytes(md_content)
    if not pdf_bytes:
        logger.error("PDF export produced no bytes (pymupdf unavailable?).")
        return
    safe = "".join(c if c.isalnum() else "_" for c in (question or "answer"))[:40] or "answer"
    file_ids = mattermost_api.upload_file(
        channel_id, f"takshashila_{safe}.pdf", pdf_bytes, "application/pdf"
    )
    if file_ids:
        mattermost_api.create_post(
            channel_id, "⬇️ **Exported answer** (PDF) — download below.", file_ids=file_ids
        )


# ════════════════════════════════════════════════════════════════════════════════
#  Voice input — signed short-lived links + a self-contained recorder page
# ════════════════════════════════════════════════════════════════════════════════

def _voice_token(channel_id: str, user_id: str = "") -> str:
    """Signed, time-limited token authorising a voice answer for this user+channel."""
    exp = int(time.time()) + VOICE_TTL
    sig = hmac.new(_VOICE_SECRET, f"{channel_id}:{user_id}:{exp}".encode("utf-8"),
                   hashlib.sha256).hexdigest()[:32]
    return f"{exp}.{sig}"


def _voice_token_ok(channel_id: str, user_id: str, token: str) -> bool:
    try:
        exp_str, sig = token.split(".", 1)
        exp = int(exp_str)
    except (ValueError, AttributeError):
        return False
    if exp < int(time.time()):
        return False
    expected = hmac.new(_VOICE_SECRET, f"{channel_id}:{user_id}:{exp}".encode("utf-8"),
                        hashlib.sha256).hexdigest()[:32]
    return hmac.compare_digest(sig, expected)


# Self-contained recorder page. Speech-to-text runs entirely in the browser via
# the Web Speech API (no audio leaves the device, no API keys). On submit it
# POSTs the transcript to /mattermost/voice-ask, which posts the answer back
# into the channel. channel id + token are read from the query string by JS.
_VOICE_PAGE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Takshashila Knowledge Assistant — Voice</title>
<style>
  :root { --brand:#1c75bc; --ink:#1f2733; --muted:#6b7280; --line:#e5e7eb; --ok:#27ae60; --err:#c0392b; }
  * { box-sizing: border-box; }
  body { margin:0; font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,Helvetica,Arial,sans-serif;
         background:#f4f6f9; color:var(--ink); display:flex; min-height:100vh; align-items:center; justify-content:center; padding:20px; }
  .card { width:100%; max-width:560px; background:#fff; border:1px solid var(--line); border-radius:16px;
          box-shadow:0 10px 30px rgba(16,24,40,.08); padding:28px; }
  h1 { font-size:20px; margin:0 0 4px; } .sub { color:var(--muted); font-size:13px; margin:0 0 20px; }
  .mic-wrap { display:flex; flex-direction:column; align-items:center; gap:14px; margin:8px 0 18px; }
  .mic { width:96px; height:96px; border-radius:50%; border:none; cursor:pointer; font-size:38px; color:#fff;
         background:var(--brand); transition:transform .12s ease, background .2s ease; box-shadow:0 6px 18px rgba(28,117,188,.35); }
  .mic:hover { transform:scale(1.04); }
  .mic.recording { background:var(--err); animation:pulse 1.3s infinite; }
  @keyframes pulse { 0%{box-shadow:0 0 0 0 rgba(192,57,43,.45);} 70%{box-shadow:0 0 0 18px rgba(192,57,43,0);} 100%{box-shadow:0 0 0 0 rgba(192,57,43,0);} }
  .hint { font-size:13px; color:var(--muted); }
  textarea { width:100%; min-height:96px; resize:vertical; padding:12px 14px; border:1px solid var(--line);
             border-radius:10px; font-size:15px; line-height:1.5; color:var(--ink); }
  .row { display:flex; gap:10px; margin-top:14px; }
  button.action { flex:1; padding:12px 14px; border-radius:10px; border:1px solid var(--line); background:#fff;
                  font-size:14px; font-weight:600; cursor:pointer; }
  button.primary { background:var(--brand); color:#fff; border-color:var(--brand); }
  button.primary:disabled { opacity:.5; cursor:not-allowed; }
  .status { margin-top:14px; font-size:14px; min-height:20px; }
  .status.ok { color:var(--ok); } .status.err { color:var(--err); }
  .foot { margin-top:18px; font-size:12px; color:var(--muted); text-align:center; }
</style>
</head>
<body>
  <div class="card">
    <h1>🏛️ Takshashila Knowledge Assistant</h1>
    <p class="sub">🎤 Speak your question — we'll transcribe it and send the answer to your Mattermost channel.</p>

    <div class="mic-wrap">
      <button id="mic" class="mic" title="Start / stop recording">🎤</button>
      <div class="hint" id="hint">Tap the mic and start speaking</div>
    </div>

    <textarea id="transcript" placeholder="Your question will appear here as you speak… (you can also edit it)"></textarea>

    <div class="row">
      <button class="action" id="clear">Clear</button>
      <button class="action primary" id="send" disabled>Send to Knowledge Base</button>
    </div>

    <div class="status" id="status"></div>
    <div class="foot">Speech is transcribed on your device. Works best in Chrome, Edge or Brave.</div>
  </div>

<script>
(function () {
  var params = new URLSearchParams(location.search);
  var channelId = params.get("c") || "";
  var userId = params.get("u") || "";
  var token = params.get("t") || "";

  var micBtn = document.getElementById("mic");
  var hint = document.getElementById("hint");
  var ta = document.getElementById("transcript");
  var sendBtn = document.getElementById("send");
  var clearBtn = document.getElementById("clear");
  var statusEl = document.getElementById("status");

  function setStatus(msg, cls) { statusEl.textContent = msg || ""; statusEl.className = "status " + (cls || ""); }
  function refreshSend() { sendBtn.disabled = ta.value.trim().length === 0; }

  ta.addEventListener("input", refreshSend);
  clearBtn.addEventListener("click", function () { ta.value = ""; refreshSend(); setStatus(""); });

  if (!channelId || !token) {
    setStatus("This voice link is missing its session details. Run /askkb voice again.", "err");
    micBtn.disabled = true; return;
  }

  var SR = window.SpeechRecognition || window.webkitSpeechRecognition;
  var recognizer = null, recording = false, finalText = "";

  if (!SR) {
    hint.textContent = "Speech recognition isn't supported in this browser — you can type the question instead.";
    micBtn.disabled = true;
  } else {
    recognizer = new SR();
    recognizer.continuous = true;
    recognizer.interimResults = true;
    recognizer.lang = "en-IN";

    recognizer.onresult = function (e) {
      var interim = "";
      for (var i = e.resultIndex; i < e.results.length; i++) {
        var t = e.results[i][0].transcript;
        if (e.results[i].isFinal) { finalText += t + " "; } else { interim += t; }
      }
      ta.value = (finalText + interim).replace(/\\s+/g, " ").trimStart();
      refreshSend();
    };
    recognizer.onerror = function (e) { setStatus("Microphone error: " + e.error, "err"); stop(); };
    recognizer.onend = function () { if (recording) { try { recognizer.start(); } catch (x) {} } };
  }

  function start() {
    finalText = ta.value ? ta.value + " " : "";
    try { recognizer.start(); } catch (x) {}
    recording = true; micBtn.classList.add("recording");
    hint.textContent = "Listening… tap again to stop"; setStatus("");
  }
  function stop() {
    recording = false; micBtn.classList.remove("recording");
    hint.textContent = "Tap the mic to start speaking again";
    if (recognizer) { try { recognizer.stop(); } catch (x) {} }
  }
  micBtn.addEventListener("click", function () { if (!recognizer) return; recording ? stop() : start(); });

  sendBtn.addEventListener("click", function () {
    var q = ta.value.trim();
    if (!q) return;
    stop(); sendBtn.disabled = true; setStatus("Sending to the knowledge base…");
    fetch("/mattermost/voice-ask", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ channel_id: channelId, user_id: userId, token: token, question: q })
    }).then(function (r) { return r.json().then(function (d) { return { ok: r.ok, d: d }; }); })
      .then(function (res) {
        if (res.ok && res.d.ok) {
          setStatus("✅ Sent! The answer will appear in your Mattermost channel shortly.", "ok");
        } else {
          setStatus("⚠️ " + ((res.d && res.d.error) || "Could not send the question."), "err");
          refreshSend();
        }
      }).catch(function () { setStatus("⚠️ Network error — please try again.", "err"); refreshSend(); });
  });

  refreshSend();
})();
</script>
</body>
</html>"""


# ════════════════════════════════════════════════════════════════════════════════
#  FastAPI app
# ════════════════════════════════════════════════════════════════════════════════

from contextlib import asynccontextmanager


def _maybe_fetch_kb_on_boot() -> None:
    """
    Optionally sync the published KB artifact before serving (production model:
    CI builds + validates the index and publishes it as a release asset; the
    always-on backend downloads that exact version). Runs when KB_ARTIFACT_URL is
    set AND either there is no local index yet or KB_FETCH_ON_BOOT is truthy. A
    failed fetch never destroys an existing local index (see scripts/fetch_kb.py).
    """
    url = os.getenv("KB_ARTIFACT_URL", "").strip()
    if not url:
        return
    force = os.getenv("KB_FETCH_ON_BOOT", "").lower() in ("1", "true", "yes", "on")
    if not force and config.FAISS_INDEX.exists():
        return
    try:
        from scripts.fetch_kb import fetch
        fetch(url)
    except Exception as exc:
        logger.warning(f"KB artifact fetch on boot failed ({exc}); using local index if present.")


def _start_kb_refresher() -> None:
    """
    Optional background poller: periodically re-fetch the published KB artifact so
    a CI-published rebuild reaches this always-on backend with NO restart. After
    fetch_kb swaps the files in, the next query's version-aware load_index picks
    up the new version automatically. Enabled by KB_ARTIFACT_URL +
    KB_REFRESH_INTERVAL_MIN (minutes); disabled if either is unset.
    """
    url = os.getenv("KB_ARTIFACT_URL", "").strip()
    interval_min = config._env_int("KB_REFRESH_INTERVAL_MIN", 0)
    if not url or interval_min <= 0:
        return

    def _loop():
        from scripts.fetch_kb import fetch
        while True:
            time.sleep(interval_min * 60)
            try:
                fetch(url)  # never destroys a good index on failure
            except Exception as exc:
                logger.warning(f"Scheduled KB refresh failed: {exc}")

    threading.Thread(target=_loop, name="kb-refresher", daemon=True).start()
    logger.info(f"KB refresher started (every {interval_min} min from artifact URL).")


def _run_startup_checks() -> None:
    _maybe_fetch_kb_on_boot()
    _start_kb_refresher()
    if not MATTERMOST_SLASH_TOKEN:
        logger.warning("MATTERMOST_SLASH_TOKEN is not set — every slash request will be rejected (403).")
    if not MATTERMOST_BOT_TOKEN:
        logger.warning("MATTERMOST_BOT_TOKEN is not set — will rely on response_url fallback.")
    if not ENABLE_BUTTONS:
        logger.warning("Interactive buttons disabled (set MATTERMOST_BOT_PUBLIC_URL to enable).")
    logger.info(f"answer visibility = {ANSWER_VISIBILITY!r} "
                f"({'private/ephemeral by default' if ANSWER_VISIBILITY == 'private' else 'public channel posts'}).")
    logger.info(
        f"routing enabled = {ENABLE_ROUTING} "
        f"(--user/--channel{'/--group' if ENABLE_GROUP_DESTINATION else ''}); "
        f"group destinations = {ENABLE_GROUP_DESTINATION}; "
        f"share buttons = {ENABLE_SHARE_BUTTONS}; pdf export = {ENABLE_PDF_EXPORT}."
    )
    if WARM_RAG_ON_STARTUP:
        try:
            warm_rag_resources()
        except Exception as exc:
            logger.error(f"Startup RAG warm-up failed ({type(exc).__name__}); will retry lazily.")


@asynccontextmanager
async def _lifespan(_app: FastAPI):
    # Audit finding: @app.on_event("startup") is deprecated in current FastAPI
    # (removal after the on_event deprecation window, which — as of this
    # audit — has already passed per FastAPI's own warning). The lifespan
    # context manager is the supported replacement; startup logic runs before
    # the yield, with an equivalent place for shutdown logic after it if ever
    # needed.
    _run_startup_checks()
    yield


app = FastAPI(title="Takshashila Intelligence RAG API", version="3.0.0", lifespan=_lifespan)

if API_ALLOWED_ORIGINS:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=API_ALLOWED_ORIGINS,
        allow_credentials=False,
        allow_methods=["GET", "POST", "OPTIONS"],
        allow_headers=["Content-Type", "X-API-Key"],
    )


def _ephemeral(text: str) -> JSONResponse:
    return JSONResponse({"response_type": "ephemeral", "text": text})


def _ephemeral_dismissable(text: str) -> JSONResponse:
    """
    Like :func:`_ephemeral`, but attaches a single 🗑️ Dismiss button so the user
    can clear long help/examples cards after reading them. The button only appears
    when the public URL is configured (so the callback can reach the bot);
    otherwise it degrades to a plain ephemeral message.
    """
    body = {"response_type": "ephemeral", "text": text}
    if ACTION_URL:
        body["attachments"] = [{"actions": [_action("🗑️ Dismiss", "dismiss")]}]
    return JSONResponse(body)


def _routing_ack(destination: "command_parser.Destination") -> str:
    """A short, destination-aware acknowledgement shown while routing runs."""
    if destination.kind == "user" and destination.usernames:
        return (f"Working on it — I'll send the answer to **@{destination.usernames[0]}** "
                "and confirm here.")
    if destination.kind == "channel" and destination.channel_name:
        return (f"Working on it — I'll post the answer in **~{destination.channel_name}** "
                "and confirm here.")
    if destination.kind == "group" and destination.usernames:
        who = ", ".join(f"@{u}" for u in destination.usernames)
        return f"Working on it — I'll send the answer to a group message with {who}."
    return "Working on it — I'll deliver the answer and confirm here."


@app.get("/health")
def health() -> dict:
    """Liveness: the process is up. Never touches the index, so it stays fast."""
    return {"status": "ok", "service": "takshashila-intelligence-rag"}


@app.get("/ready")
def ready() -> JSONResponse:
    """
    Readiness: the service can actually answer (index present + LLM configured).
    Returns 503 until it can, so a load balancer / platform health check doesn't
    route traffic to a not-yet-ready instance.
    """
    from src import rag_service
    st = rag_service.status()
    is_ready = bool(st.get("ready")) and bool(st.get("groq_configured"))
    st["status"] = "ready" if is_ready else "not_ready"
    return JSONResponse(st, status_code=200 if is_ready else 503)


@app.get("/rag/status")
def rag_status() -> dict:
    """Full, secret-free operational status: KB version, counts, models, etc."""
    from src import rag_service
    return rag_service.status()


def _check_api_key(x_api_key: str) -> None:
    if API_KEY and not hmac.compare_digest(x_api_key or "", API_KEY):
        raise HTTPException(status_code=401, detail="Invalid or missing API key.")


@app.post("/api/query")
async def api_query(payload: dict = Body(...), x_api_key: str = Header("")):
    """
    Canonical query endpoint consumed by the static dashboard (and any other
    client). Same shared RAG engine and same response contract the Mattermost
    path uses, so answers do not depend on which interface asked.

        Request : {"query": "...", "conversation_id": "...", "interface": "dashboard", "length": "normal"}
        Response: {"answer", "grounded", "confidence", "sources", "kb_version", "latency_ms", ...}
    """
    _check_api_key(x_api_key)
    query = (payload.get("query") or "").strip()
    if not query:
        raise HTTPException(status_code=400, detail="Missing 'query'.")
    if len(query) > 2000:
        raise HTTPException(status_code=413, detail="Query too long (max 2000 chars).")
    from src import rag_service
    try:
        return rag_service.answer_query(
            query,
            interface=(payload.get("interface") or "dashboard"),
            conversation_id=payload.get("conversation_id"),
            length=(payload.get("length") or "normal"),
            top_k=RAG_TOP_K,
            model=GROQ_MODEL,
        )
    except Exception as exc:  # clean user-facing error; details stay in logs
        logger.exception("api_query failed: %s", exc)
        raise HTTPException(status_code=502, detail="RAG service error. Please try again.")


@app.post("/mattermost/ask")
async def mattermost_ask(
    background_tasks: BackgroundTasks,
    token: str = Form(""),
    text: str = Form(""),
    user_name: str = Form(""),
    user_id: str = Form(""),
    channel_id: str = Form(""),
    channel_name: str = Form(""),
    team_id: str = Form(""),
    team_domain: str = Form(""),
    response_url: str = Form(""),
):
    """Handle the /askkb slash command."""
    if not MATTERMOST_SLASH_TOKEN or token != MATTERMOST_SLASH_TOKEN:
        logger.warning(f"Rejected slash request: invalid token (user_name={user_name!r}).")
        raise HTTPException(status_code=403, detail="Invalid slash command token.")

    if ALLOWED_TEAM_IDS and team_id not in ALLOWED_TEAM_IDS:
        return _ephemeral("This command is not enabled for this team.")
    if ALLOWED_CHANNEL_IDS and channel_id not in ALLOWED_CHANNEL_IDS:
        return _ephemeral("This command is not enabled for this channel.")

    raw_text = (text or "").strip()

    # ── Parse into ONE normalized command object ────────────────────────────────
    # command_parser handles long flags + short aliases (-m/-u/-c/-g, -s/-d/-f/-v)
    # in any order, plus the standalone help/examples commands and the empty case.
    # The remaining fields (destination, mode, visibility, voice, question) come
    # out already normalized, so nothing below re-parses the text.
    parsed = command_parser.parse_command(raw_text)

    # Standalone commands never return an error card.
    if parsed.command == "empty":
        return _ephemeral(help_text.format_landing())
    if parsed.command == "help":
        return _ephemeral_dismissable(help_text.format_help(ENABLE_GROUP_DESTINATION, ENABLE_VOICE))
    if parsed.command == "examples":
        return _ephemeral_dismissable(help_text.format_examples(ENABLE_GROUP_DESTINATION, ENABLE_VOICE))

    # Destination (routing can be disabled globally, in which case we stay private).
    destination = parsed.destination if ENABLE_ROUTING else command_parser.Destination("me")
    if ENABLE_ROUTING and parsed.error:
        return _ephemeral(parsed.error)

    # Group messages are opt-in; guide the user to the supported destinations.
    if destination.kind == "group" and not ENABLE_GROUP_DESTINATION:
        return _ephemeral(
            "Group messages aren't enabled here. You can send the answer to a "
            "single person with `-u <username>` (`--user`) or to a channel with "
            "`-c <channel>` (`--channel`) instead."
        )

    # Visibility: parser reports default/public/private; fall back to the configured default.
    visibility = ANSWER_VISIBILITY
    if parsed.visibility == "public":
        visibility = "public"
    elif parsed.visibility == "private":
        visibility = "private"

    mode = parsed.mode
    question = parsed.question

    # ── Voice input: return a signed link to the in-browser recorder page ───────
    # Voice is only meaningful for the default self-delivery.
    if destination.kind == "me" and parsed.voice:
        if not ENABLE_VOICE:
            return _ephemeral("Voice input needs the bot's public URL to be configured "
                              "(set MATTERMOST_BOT_PUBLIC_URL).")
        if not (MATTERMOST_BOT_TOKEN and MATTERMOST_URL):
            return _ephemeral("Voice input needs the bot token to be configured so the "
                              "answer can be sent back to you.")
        link = (f"{PUBLIC_BASE_URL}/voice?c={channel_id}&u={user_id}"
                f"&t={_voice_token(channel_id, user_id)}")
        where = ("as a private message from the bot" if ANSWER_VISIBILITY == "private"
                 else "right here in this channel")
        return _ephemeral(
            "🎤 **Voice input**\n\n"
            "Tap the link below, allow microphone access, speak your question, then press "
            f"**Send to Knowledge Base** — the answer will be sent {where}.\n\n"
            f"👉 {link}\n\n"
            f"_The link is valid for {VOICE_TTL // 60} minutes and works best in Chrome, Edge or Brave._"
        )

    # No question after the modifiers → show the friendly landing, not an error.
    if not question:
        return _ephemeral(help_text.format_landing())

    # ── External destination (--user / --channel / --group) ─────────────────────
    # Retrieval is scheduled exactly as normal; only the delivery target differs.
    if destination.is_external:
        if not (MATTERMOST_BOT_TOKEN and MATTERMOST_URL):
            return _ephemeral(
                "Sending to another user, channel or group needs the bot token to be "
                "configured. Ask an admin to set `MATTERMOST_BOT_TOKEN`."
            )
        background_tasks.add_task(
            run_rag_and_reply,
            question=question, channel_id=channel_id, response_url=response_url or None,
            user_name=user_name, user_id=user_id, channel_name=channel_name, mode=mode,
            visibility=visibility, dm_user_id="",
            destination=destination, team_id=team_id,
        )
        return _ephemeral("🔎 " + _routing_ack(destination))

    background_tasks.add_task(
        run_rag_and_reply,
        question=question, channel_id=channel_id, response_url=response_url or None,
        user_name=user_name, user_id=user_id, channel_name=channel_name, mode=mode,
        visibility=visibility,
        dm_user_id=(user_id if visibility == "private" else ""),
        destination=destination, team_id=team_id,
    )
    if visibility == "private":
        if PRIVATE_DELIVERY == "dm":
            return _ephemeral(
                "🔒 Working on it — I'll send your answer to your **direct messages** "
                "with me, so only you can see it (with feedback + download buttons)."
            )
        return _ephemeral("🔒 " + (ACK_SEARCH if mode == "search" else ACK_MESSAGE)
                          + "\n\n_Only you will see the answer in this channel._")
    return _ephemeral(ACK_SEARCH if mode == "search" else ACK_MESSAGE)


@app.get("/voice")
def voice_page() -> HTMLResponse:
    """Serve the in-browser voice recorder page."""
    return HTMLResponse(content=_VOICE_PAGE)


@app.post("/mattermost/voice-ask")
async def voice_ask(background_tasks: BackgroundTasks, payload: dict):
    """Accept a transcribed question from the voice page and answer the asker."""
    channel_id = (payload.get("channel_id") or "").strip()
    user_id = (payload.get("user_id") or "").strip()
    token = (payload.get("token") or "").strip()
    question = (payload.get("question") or "").strip()

    if not (channel_id and token and _voice_token_ok(channel_id, user_id, token)):
        return JSONResponse(
            {"ok": False, "error": "This voice session has expired. Run /askkb voice again."},
            status_code=403,
        )
    if not question:
        return JSONResponse({"ok": False, "error": "No speech was captured."}, status_code=400)
    if not (MATTERMOST_BOT_TOKEN and MATTERMOST_URL):
        return JSONResponse(
            {"ok": False, "error": "Voice answers need the bot token configured."},
            status_code=500,
        )

    mode, q = formatting.parse_mode(question)
    # Private voice answers are DM'd to the asker (there's no response_url for an
    # ephemeral reply); public voice answers go to the channel.
    dm_user_id = user_id if ANSWER_VISIBILITY == "private" else ""
    background_tasks.add_task(
        run_rag_and_reply,
        question=q, channel_id=channel_id, response_url=None,
        user_name="voice", user_id=user_id, channel_name="", mode=mode, voice=True,
        visibility=ANSWER_VISIBILITY, dm_user_id=dm_user_id,
    )
    logger.info(f"voice_question_received  channel={channel_id!r}  user={user_id!r}  "
                f"mode={mode}  visibility={ANSWER_VISIBILITY}  question={q!r}")
    return JSONResponse({"ok": True})


# ════════════════════════════════════════════════════════════════════════════════
#  Interactive button callbacks
# ════════════════════════════════════════════════════════════════════════════════

def _handle_feedback(payload: dict, context: dict) -> JSONResponse:
    """
    Record the vote and respond with an interactive transition:
      * `update`        → swaps the 👍/👎 buttons in the post for a confirmation
                          card (the other button groups are rebuilt so they stay
                          usable). Requires the post to be in our cache.
      * `ephemeral_text`→ a private 'popup' line shown only to the clicker.
    """
    verdict = (context.get("verdict") or "").strip()
    question = (context.get("question") or "").strip()
    if verdict not in ("helpful", "not_helpful"):
        return JSONResponse({"ephemeral_text": "Unrecognised feedback action."})

    record_feedback(
        verdict=verdict, question=question,
        user_name=payload.get("user_name", ""), user_id=payload.get("user_id", ""),
        channel_id=payload.get("channel_id", ""), post_id=payload.get("post_id", ""),
    )
    logger.info(f"feedback_recorded  verdict={verdict}  question={question!r}")

    popup = FEEDBACK_POPUP[verdict]
    cached = _cache_get(payload.get("post_id", ""))
    if cached:
        # Rebuild the post: keep the follow-up + export + delete buttons, replace
        # the feedback buttons with a confirmation card → a clean transition.
        attachments = (
            _button_groups(cached["question"])
            + [_confirmation_card(verdict)]
            + _manage_tail(cached["question"])
        )
        return JSONResponse({
            "update": {"message": cached["message"], "props": {"attachments": attachments}},
            "ephemeral_text": popup,
        })
    # Cache miss (e.g. after a restart): still give the clicker the popup.
    return JSONResponse({"ephemeral_text": popup})


# ── Share buttons → dialog → router (bonus) ──────────────────────────────────────

_SHARE_KIND = {"share_user": "user", "share_channel": "channel", "share_group": "group"}

_SHARE_DIALOG = {
    "user": {
        "title": "Share to a user",
        "element": {
            "display_name": "Username", "name": "target", "type": "text",
            "placeholder": "abhishek.k",
            "help_text": "The user who should receive this answer (no @ needed).",
        },
    },
    "channel": {
        "title": "Share to a channel",
        "element": {
            "display_name": "Channel", "name": "target", "type": "text",
            "placeholder": "research",
            "help_text": "The channel to post this answer in. The bot must be a member.",
        },
    },
    "group": {
        "title": "Share to a group",
        "element": {
            "display_name": "Usernames", "name": "target", "type": "text",
            "placeholder": "abhishek.k,nithiya,amit",
            "help_text": "Comma-separated usernames for the group message.",
        },
    },
}


def _handle_share_button(action: str, payload: dict, question: str,
                         post_id: str) -> JSONResponse:
    """Open a Mattermost dialog collecting the share target (no RAG re-run)."""
    kind = _SHARE_KIND.get(action, "")
    if not (kind and ENABLE_SHARE_BUTTONS):
        return JSONResponse({"ephemeral_text": "Sharing isn't enabled on this deployment."})
    if kind == "group" and not ENABLE_GROUP_DESTINATION:
        return JSONResponse({"ephemeral_text": "Group messages aren't enabled here. Use Share to User or Share to Channel."})
    trigger_id = payload.get("trigger_id", "")
    if not trigger_id:
        return JSONResponse({"ephemeral_text": "Couldn't open the share dialog (missing trigger)."})

    spec = _SHARE_DIALOG[kind]
    state = {"p": post_id, "q": (question or "")[:400]}
    state["_sig"] = _sign_action_context(state)
    dialog = {
        "callback_id": kind,
        "title": spec["title"],
        "submit_label": "Share",
        "elements": [spec["element"]],
        # Carry the post id (to reuse the cached answer) + a fallback question.
        # Signed for the same reason button contexts are (see _action) — this
        # round-trips through the client via the dialog submission, so it must
        # not be trustable just because it looks like our own JSON.
        "state": json.dumps(state),
    }
    if not mattermost_api.open_dialog(trigger_id, DIALOG_URL, dialog):
        return JSONResponse({"ephemeral_text": "Couldn't open the share dialog. Please try again."})
    return JSONResponse({})


def _deliver_cached_answer(destination: "command_parser.Destination", post_id: str,
                           question: str, user_id: str, user_name: str,
                           channel_id: str, team_id: str) -> DeliveryResult:
    """
    Deliver an ALREADY-GENERATED answer (from the cache, or regenerated on a cache
    miss) to ``destination`` via the response router. Never re-runs RAG on the
    cache-hit path — it reuses the exact message the user is looking at.
    """
    cached = _cache_get(post_id)
    if cached:
        message = cached["message"]
        q = cached.get("question") or question
        mode = cached.get("mode", "normal")
    else:
        if not question:
            return DeliveryResult(False, error="This answer is no longer available. Please re-run the query.")
        try:
            result = _regen_for_callback(question)
            message = build_message(question, result, mode="normal")
            q, mode = question, "normal"
        except Exception:
            logger.error("Share regeneration failed:\n" + traceback.format_exc())
            return DeliveryResult(False, error="Couldn't rebuild the answer to share. Please try again.")

    props = _feedback_attachment(q, mode=mode) if ENABLE_BUTTONS else None
    payload = ResponsePayload(question=q, message=message, props=props, mode=mode)
    requester = Requester(user_id=user_id, user_name=user_name,
                          team_id=team_id, channel_id=channel_id)
    return response_router.send_response(
        destination, payload, requester, allowed_channel_ids=(ALLOWED_CHANNEL_IDS or None)
    )


@app.post("/mattermost/action")
async def mattermost_action(background_tasks: BackgroundTasks, payload: dict):
    """Route interactive-button clicks (follow-ups, export, share, feedback)."""
    context = payload.get("context") or {}

    if not _action_context_ok(context):
        logger.warning(
            f"Rejected /mattermost/action request: missing/invalid context "
            f"signature (user_id={payload.get('user_id')!r}, "
            f"channel_id={payload.get('channel_id')!r})."
        )
        return JSONResponse(
            {"ephemeral_text": "⚠️ This button has expired or is no longer valid. "
                                "Please run `/askkb` again."},
            status_code=200,  # Mattermost expects 200 for a graceful ephemeral message
        )

    action = (context.get("action") or "").strip()
    question = (context.get("question") or "").strip()
    channel_id = payload.get("channel_id", "")
    post_id = payload.get("post_id", "")
    user_id = payload.get("user_id", "")
    user_name = payload.get("user_name", "")

    # ── Feedback (👍 / 👎) — interactive transition ─────────────────────────────
    if action == "feedback":
        return _handle_feedback(payload, context)

    # ── Dismiss an ephemeral help/examples card ─────────────────────────────────
    if action == "dismiss":
        return JSONResponse({
            "update": {"message": "🗑️ _Dismissed._", "props": {"attachments": []}},
            "ephemeral_text": "Dismissed.",
        })

    # ── Export markdown (file upload, in background) ────────────────────────────
    if action == "export_markdown":
        if not (MATTERMOST_BOT_TOKEN and MATTERMOST_URL):
            return JSONResponse({"ephemeral_text": "Export needs the bot token to be configured."})
        cached = _cache_get(post_id)
        md = cached["message"] if cached else None
        background_tasks.add_task(_export_markdown_task, channel_id, question, md)
        return JSONResponse({"ephemeral_text": "📄 Preparing your Markdown export…"})

    # ── Export PDF (file upload, in background) ─────────────────────────────────
    if action == "export_pdf":
        if not ENABLE_PDF_EXPORT:
            return JSONResponse({"ephemeral_text": "PDF export isn't available on this deployment."})
        cached = _cache_get(post_id)
        md = cached["message"] if cached else None
        background_tasks.add_task(_export_pdf_task, channel_id, question, md)
        return JSONResponse({"ephemeral_text": "⬇️ Preparing your PDF export…"})

    # ── Share buttons: open a dialog to collect the target (no RAG re-run) ──────
    if action in ("share_user", "share_channel", "share_group"):
        return _handle_share_button(action, payload, question, post_id)

    # ── Related Policies: show the exact documents THIS answer was drawn from ────
    # (reuses the answer's grounding sources — never a fresh KB search).
    if action == "related":
        background_tasks.add_task(_related_task, channel_id, question, post_id)
        return JSONResponse({"ephemeral_text": "🔗 Fetching the documents this answer is based on…"})

    # ── More details: re-run the pipeline in a fuller mode ──────────────────────
    if action == "more_details":
        background_tasks.add_task(
            run_rag_and_reply,
            question=question, channel_id=channel_id, response_url=None,
            user_name=user_name, user_id=user_id, channel_name="", mode="detailed",
        )
        return JSONResponse({"ephemeral_text": "📖 Fetching more details…"})

    # ── Delete this single response ─────────────────────────────────────────────
    if action == "delete_one":
        if not (MATTERMOST_BOT_TOKEN and MATTERMOST_URL):
            return JSONResponse({"ephemeral_text": "Deletion needs the bot token to be configured."})
        ok = _delete_post(post_id)
        _ANSWER_CACHE.pop(post_id, None)
        return JSONResponse({"ephemeral_text":
            "🗑️ This response was deleted." if ok
            else "Couldn't delete this response (check the bot's permissions)."})

    # ── Delete all: step 1 — ask for confirmation (in-post transition) ──────────
    if action == "delete_all":
        if not (MATTERMOST_BOT_TOKEN and MATTERMOST_URL):
            return JSONResponse({"ephemeral_text": "Deletion needs the bot token to be configured."})
        cached = _cache_get(post_id)
        if cached:
            attachments = (
                _button_groups(cached["question"])
                + [_feedback_group(cached["question"]), _confirm_delete_group(cached["question"])]
            )
            return JSONResponse({
                "update": {"message": cached["message"], "props": {"attachments": attachments}},
                "ephemeral_text": "⚠️ Please confirm: this will remove every bot response in this channel.",
            })
        # No cached post to transform → fall back to deleting immediately.
        n = _delete_all_bot_posts(channel_id)
        return JSONResponse({"ephemeral_text": f"🧹 Deleted {n} bot response(s) in this channel."})

    # ── Delete all: step 2 — confirmed ──────────────────────────────────────────
    if action == "delete_all_confirm":
        if not (MATTERMOST_BOT_TOKEN and MATTERMOST_URL):
            return JSONResponse({"ephemeral_text": "Deletion needs the bot token to be configured."})
        n = _delete_all_bot_posts(channel_id)
        return JSONResponse({"ephemeral_text": f"🧹 Deleted {n} bot response(s) in this channel."})

    # ── Delete all: cancelled — restore the original buttons ────────────────────
    if action == "delete_cancel":
        cached = _cache_get(post_id)
        if cached:
            return JSONResponse({
                "update": {"message": cached["message"],
                           "props": {"attachments": _answer_attachments(cached["question"])}},
                "ephemeral_text": "✖️ Deletion cancelled.",
            })
        return JSONResponse({"ephemeral_text": "✖️ Deletion cancelled."})

    return JSONResponse({"ephemeral_text": "Unknown action."})


@app.post("/mattermost/feedback")
async def mattermost_feedback(payload: dict):
    """Legacy feedback route — kept so older posts' buttons keep working."""
    return _handle_feedback(payload, payload.get("context") or {})


@app.post("/mattermost/dialog")
async def mattermost_dialog(payload: dict):
    """
    Handle a Share-dialog submission: reuse the already-generated answer and
    deliver it to the chosen user / channel / group via the response router.

    On a resolution failure (unknown user/channel, etc.) the error is returned on
    the dialog's ``target`` field so the user can correct it without losing the
    dialog. On success a private ephemeral confirmation is posted to the sharer.
    """
    if payload.get("cancelled"):
        return JSONResponse({})

    kind = (payload.get("callback_id") or "").strip()
    if kind not in ("user", "channel", "group"):
        return JSONResponse({"error": "Unknown share action."})
    if kind == "group" and not ENABLE_GROUP_DESTINATION:
        return JSONResponse({"errors": {"target": "Group messages aren't enabled here."}})

    submission = payload.get("submission") or {}
    target = (submission.get("target") or "").strip()
    if not target:
        return JSONResponse({"errors": {"target": "Please enter a target."}})

    try:
        state = json.loads(payload.get("state") or "{}")
    except Exception:
        state = {}
    if not _action_context_ok(state):
        logger.warning(f"Rejected /mattermost/dialog submission: missing/invalid state signature "
                       f"(user_id={payload.get('user_id')!r}).")
        return JSONResponse({"errors": {"target": "This share dialog has expired. Please try again."}})
    post_id = state.get("p", "")
    question = state.get("q", "")

    user_id = payload.get("user_id", "")
    user_name = payload.get("user_name", "")
    channel_id = payload.get("channel_id", "")
    team_id = payload.get("team_id", "")

    destination = command_parser.build_destination(kind, target)
    delivery = _deliver_cached_answer(
        destination, post_id, question, user_id, user_name, channel_id, team_id
    )

    if not delivery.ok:
        # Show the reason inline on the field so the dialog stays open for a retry.
        return JSONResponse({"errors": {"target": delivery.error}})

    if channel_id and user_id:
        mattermost_api.post_ephemeral(channel_id, user_id, delivery.confirmation)
    logger.info(f"shared via dialog  kind={kind}  target={target!r}  post_id={delivery.post_id!r}")
    return JSONResponse({})


# Allow `python integrations/mattermost_bot.py` as a convenience (uvicorn preferred).
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "integrations.mattermost_bot:app",
        host=os.getenv("MATTERMOST_BOT_HOST", "0.0.0.0"),
        port=config._env_int("MATTERMOST_BOT_PORT", 8000),
        reload=False,
    )