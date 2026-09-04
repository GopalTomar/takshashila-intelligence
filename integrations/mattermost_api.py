"""
mattermost_api.py — Thin, stateless Mattermost REST helpers for the router.

This module is the single place the delivery layer performs the *new* Mattermost
lookups that enterprise routing needs (resolve a username, resolve a channel by
name, check the bot's channel membership, open/reuse a group DM, open interactive
dialogs, post ephemeral confirmations, upload files). It reads the same
environment variables the bot already uses and holds **no** RAG or FastAPI code,
so it can be imported from anywhere without pulling in heavy dependencies and
without risking a circular import.

Reuse over duplication
----------------------
The bot's existing ``mattermost_bot._get_bot_user_id`` and
``mattermost_bot._get_or_create_dm_channel`` delegate to the functions here so
there is exactly one implementation of each Mattermost call in the project.

Every function is defensive: on any HTTP/transport error it logs and returns a
falsy value (``None`` / ``False`` / ``[]``) instead of raising, so a delivery
attempt can fail gracefully with a friendly message rather than crashing a
background task.
"""

from __future__ import annotations

import logging
import os
import threading
from typing import Dict, List, Optional

import httpx
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger("mattermost_bot")  # share the bot's logger namespace

# ── Configuration (identical variable names to the rest of the bot) ──────────────
MATTERMOST_URL = os.getenv("MATTERMOST_URL", "").rstrip("/")
MATTERMOST_BOT_TOKEN = os.getenv("MATTERMOST_BOT_TOKEN", "").strip()

_HTTP_TIMEOUT = float(os.getenv("MATTERMOST_HTTP_TIMEOUT", "20"))

# Mattermost requires a group DM to have between 3 and 8 members (the bot counts).
GROUP_DM_MIN_MEMBERS = 3
GROUP_DM_MAX_MEMBERS = 8


def is_configured() -> bool:
    """True when we have both a base URL and a bot token (REST calls possible)."""
    return bool(MATTERMOST_URL and MATTERMOST_BOT_TOKEN)


def _auth_headers() -> Dict[str, str]:
    return {"Authorization": f"Bearer {MATTERMOST_BOT_TOKEN}"}


# ════════════════════════════════════════════════════════════════════════════════
#  Bot identity (cached)
# ════════════════════════════════════════════════════════════════════════════════

_bot_user_id: Optional[str] = None
_bot_id_lock = threading.Lock()


def get_bot_user_id() -> Optional[str]:
    """Resolve + cache this bot's own user id. Returns ``None`` if unavailable."""
    global _bot_user_id
    if _bot_user_id:
        return _bot_user_id
    if not is_configured():
        return None
    with _bot_id_lock:
        if _bot_user_id:
            return _bot_user_id
        try:
            with httpx.Client(timeout=_HTTP_TIMEOUT) as client:
                r = client.get(f"{MATTERMOST_URL}/api/v4/users/me", headers=_auth_headers())
            if r.status_code == 200:
                _bot_user_id = r.json().get("id")
            else:
                logger.error(f"Resolve bot user id failed (HTTP {r.status_code}).")
        except Exception as exc:
            logger.error(f"Resolve bot user id raised {type(exc).__name__}.")
    return _bot_user_id


# ════════════════════════════════════════════════════════════════════════════════
#  Lookups
# ════════════════════════════════════════════════════════════════════════════════

def find_user_by_username(username: str) -> Optional[Dict]:
    """
    Look up a Mattermost user by their ``@username`` (the ``@`` is optional).
    Returns the user object (``id``, ``username``, …) or ``None`` if not found.
    """
    name = (username or "").strip().lstrip("@").strip()
    if not (name and is_configured()):
        return None
    try:
        with httpx.Client(timeout=_HTTP_TIMEOUT) as client:
            r = client.get(f"{MATTERMOST_URL}/api/v4/users/username/{name}",
                           headers=_auth_headers())
        if r.status_code == 200:
            return r.json()
        if r.status_code != 404:
            logger.error(f"User lookup for {name!r} failed (HTTP {r.status_code}).")
    except Exception as exc:
        logger.error(f"User lookup raised {type(exc).__name__}.")
    return None


def find_channel_on_team(team_id: str, channel_name: str) -> Optional[Dict]:
    """
    Resolve a channel on a given team by its name/slug (e.g. ``research``).

    Tries the exact-name endpoint first, then falls back to the team channel
    search (which also matches display names), so both ``--channel research`` and
    a friendlier display name work. Returns the channel object or ``None``.
    """
    name = (channel_name or "").strip().lstrip("~").strip()
    if not (name and team_id and is_configured()):
        return None

    # 1) Exact URL-name match — the common case.
    try:
        with httpx.Client(timeout=_HTTP_TIMEOUT) as client:
            r = client.get(
                f"{MATTERMOST_URL}/api/v4/teams/{team_id}/channels/name/{name}",
                headers=_auth_headers(),
            )
        if r.status_code == 200:
            return r.json()
        if r.status_code not in (404, 403):
            logger.error(f"Channel name lookup for {name!r} failed (HTTP {r.status_code}).")
    except Exception as exc:
        logger.error(f"Channel name lookup raised {type(exc).__name__}.")

    # 2) Fallback: search by term (matches display names + partials).
    try:
        with httpx.Client(timeout=_HTTP_TIMEOUT) as client:
            r = client.post(
                f"{MATTERMOST_URL}/api/v4/teams/{team_id}/channels/search",
                headers=_auth_headers(), json={"term": name},
            )
        if r.status_code == 200:
            results = r.json() or []
            lowered = name.lower()
            # Prefer an exact name / display-name match, else the first result.
            for ch in results:
                if lowered in (ch.get("name", "").lower(), ch.get("display_name", "").lower()):
                    return ch
            if results:
                return results[0]
    except Exception as exc:
        logger.error(f"Channel search raised {type(exc).__name__}.")
    return None


def bot_in_channel(channel_id: str) -> bool:
    """True when this bot is a member of ``channel_id`` (so it may post there)."""
    bot_id = get_bot_user_id()
    if not (bot_id and channel_id and is_configured()):
        return False
    try:
        with httpx.Client(timeout=_HTTP_TIMEOUT) as client:
            r = client.get(
                f"{MATTERMOST_URL}/api/v4/channels/{channel_id}/members/{bot_id}",
                headers=_auth_headers(),
            )
        return r.status_code == 200
    except Exception as exc:
        logger.error(f"Channel membership check raised {type(exc).__name__}.")
        return False


# ════════════════════════════════════════════════════════════════════════════════
#  Direct + group message channels
# ════════════════════════════════════════════════════════════════════════════════

def get_or_create_dm_channel(user_id: str) -> Optional[str]:
    """Return the bot↔user direct-message channel id (created if needed)."""
    bot_id = get_bot_user_id()
    if not (bot_id and user_id and is_configured()):
        return None
    try:
        with httpx.Client(timeout=_HTTP_TIMEOUT) as client:
            r = client.post(f"{MATTERMOST_URL}/api/v4/channels/direct",
                            headers=_auth_headers(), json=[bot_id, user_id])
        if r.status_code in (200, 201):
            return r.json().get("id")
        logger.error(f"Create DM channel failed (HTTP {r.status_code}).")
    except Exception as exc:
        logger.error(f"Create DM channel raised {type(exc).__name__}.")
    return None


def create_group_channel(user_ids: List[str]) -> Optional[Dict]:
    """
    Create (or reuse) a group direct-message channel for the bot + ``user_ids``
    and return the **full channel object** (validated), or ``None`` on failure.

    Mattermost's ``/api/v4/channels/group`` endpoint is idempotent: posting the
    same member set returns the existing channel, so this doubles as "get".
    We validate the HTTP status, that the response has an ``id``, and that the
    returned channel is actually of type ``"G"`` (group) — so a malformed or
    unexpected response can never be mistaken for success.
    """
    bot_id = get_bot_user_id()
    if not bot_id:
        logger.error("Group DM: could not resolve the bot's own user id (check bot token).")
        return None
    if not is_configured():
        logger.error("Group DM: bot token / URL not configured.")
        return None

    members: List[str] = [bot_id]
    for uid in user_ids:
        if uid and uid not in members:
            members.append(uid)

    if len(members) < GROUP_DM_MIN_MEMBERS:
        logger.error(
            f"Group DM needs at least {GROUP_DM_MIN_MEMBERS} members "
            f"(bot + {GROUP_DM_MIN_MEMBERS - 1} users); got {len(members)} → {members}."
        )
        return None
    if len(members) > GROUP_DM_MAX_MEMBERS:
        logger.error(
            f"Group DM supports at most {GROUP_DM_MAX_MEMBERS} members; got {len(members)}."
        )
        return None

    try:
        with httpx.Client(timeout=_HTTP_TIMEOUT) as client:
            r = client.post(f"{MATTERMOST_URL}/api/v4/channels/group",
                            headers=_auth_headers(), json=members)
        if r.status_code in (200, 201):
            channel = r.json()
            cid = channel.get("id")
            ctype = channel.get("type")
            if not cid:
                logger.error(f"Group DM create returned no channel id: {str(channel)[:200]!r}")
                return None
            if ctype and ctype != "G":
                logger.error(f"Group DM create returned unexpected channel type {ctype!r} (id={cid}).")
                return None
            logger.info(f"Group DM ready: channel_id={cid!r} members={members}")
            return channel
        logger.error(
            f"Create group DM failed (HTTP {r.status_code}); body={_short(r)}; members={members}."
        )
    except Exception as exc:
        logger.error(f"Create group DM raised {type(exc).__name__}: {exc}")
    return None


def get_or_create_group_channel(user_ids: List[str]) -> Optional[str]:
    """Backward-compatible wrapper: return just the group channel **id** (or None)."""
    channel = create_group_channel(user_ids)
    return channel.get("id") if channel else None


def get_channel_members(channel_id: str) -> List[str]:
    """Return the list of member user-ids for ``channel_id`` (empty on failure)."""
    if not (channel_id and is_configured()):
        return []
    ids: List[str] = []
    try:
        with httpx.Client(timeout=_HTTP_TIMEOUT) as client:
            r = client.get(
                f"{MATTERMOST_URL}/api/v4/channels/{channel_id}/members",
                headers=_auth_headers(), params={"page": 0, "per_page": 200},
            )
        if r.status_code == 200:
            ids = [m.get("user_id") for m in r.json() if m.get("user_id")]
        else:
            logger.error(f"List channel members failed (HTTP {r.status_code}); body={_short(r)}.")
    except Exception as exc:
        logger.error(f"List channel members raised {type(exc).__name__}: {exc}")
    return ids


def show_group_channel_to_members(channel_id: str, user_ids: List[str]) -> int:
    """
    Make a group DM visible in each member's sidebar by setting the
    ``group_channel_show`` preference to ``"true"`` for every user.

    A freshly, programmatically-created GM is otherwise hidden in a member's
    sidebar until they've interacted with it, which is why a bot can post to the
    channel yet the recipients never notice. Setting this preference surfaces it.

    Best-effort: setting another user's preferences requires the bot to have the
    ``edit_other_users`` permission. If it doesn't, Mattermost returns 403 and we
    log a clear hint but do **not** fail the delivery (the post still exists and
    will surface as unread on most Mattermost versions).

    Returns the number of users for whom the preference was set successfully.
    """
    if not (channel_id and is_configured()):
        return 0
    ok = 0
    forbidden = False
    for uid in user_ids:
        if not uid:
            continue
        pref = [{
            "user_id": uid,
            "category": "group_channel_show",
            "name": channel_id,
            "value": "true",
        }]
        try:
            with httpx.Client(timeout=_HTTP_TIMEOUT) as client:
                r = client.put(
                    f"{MATTERMOST_URL}/api/v4/users/{uid}/preferences",
                    headers=_auth_headers(), json=pref,
                )
            if r.status_code == 200:
                ok += 1
            elif r.status_code == 403:
                forbidden = True
            else:
                logger.error(f"Show GM for user {uid} failed (HTTP {r.status_code}); body={_short(r)}.")
        except Exception as exc:
            logger.error(f"Show GM for user {uid} raised {type(exc).__name__}: {exc}")
    if forbidden:
        logger.warning(
            "Could not set group_channel_show for one or more members (HTTP 403). "
            "The bot lacks the 'edit_other_users' permission, so the group DM may stay "
            "collapsed in recipients' sidebars until they open it. Grant the bot that "
            "permission (System Console → bot's role) for reliable group-DM visibility."
        )
    logger.info(f"group_channel_show set for {ok}/{len(user_ids)} member(s) on channel {channel_id!r}.")
    return ok


def _short(resp) -> str:
    """A short, safe preview of an httpx response body for logs."""
    try:
        return resp.text[:300]
    except Exception:
        return "<unreadable>"


# ════════════════════════════════════════════════════════════════════════════════
#  Posting helpers used by the router / dialog flow
# ════════════════════════════════════════════════════════════════════════════════

def post_ephemeral(channel_id: str, user_id: str, message: str) -> bool:
    """
    Send a message visible only to ``user_id`` inside ``channel_id`` (used for
    confirmations after a button/dialog-driven share, where no ``response_url``
    exists). Best-effort; returns success.
    """
    if not (channel_id and user_id and is_configured()):
        return False
    try:
        with httpx.Client(timeout=_HTTP_TIMEOUT) as client:
            r = client.post(
                f"{MATTERMOST_URL}/api/v4/posts/ephemeral",
                headers=_auth_headers(),
                json={"user_id": user_id, "post": {"channel_id": channel_id, "message": message}},
            )
        return r.status_code in (200, 201)
    except Exception as exc:
        logger.error(f"Ephemeral post raised {type(exc).__name__}.")
        return False


def open_dialog(trigger_id: str, url: str, dialog: Dict) -> bool:
    """
    Open a Mattermost interactive dialog (used by the Share buttons to collect a
    target username / channel / group without re-running the RAG pipeline).
    """
    if not (trigger_id and url and is_configured()):
        return False
    try:
        with httpx.Client(timeout=_HTTP_TIMEOUT) as client:
            r = client.post(
                f"{MATTERMOST_URL}/api/v4/actions/dialogs/open",
                headers=_auth_headers(),
                json={"trigger_id": trigger_id, "url": url, "dialog": dialog},
            )
        ok = r.status_code in (200, 201)
        if not ok:
            logger.error(f"Open dialog failed (HTTP {r.status_code}).")
        return ok
    except Exception as exc:
        logger.error(f"Open dialog raised {type(exc).__name__}.")
        return False


def upload_file(channel_id: str, filename: str, content: bytes, mime: str) -> List[str]:
    """Upload a single file to ``channel_id``; return the created file id(s)."""
    if not (channel_id and is_configured()):
        return []
    try:
        with httpx.Client(timeout=max(_HTTP_TIMEOUT, 30.0)) as client:
            files = {"files": (filename, content, mime)}
            data = {"channel_id": channel_id}
            r = client.post(f"{MATTERMOST_URL}/api/v4/files",
                            headers=_auth_headers(), data=data, files=files)
        if r.status_code in (200, 201):
            infos = r.json().get("file_infos") or []
            return [fi["id"] for fi in infos if fi.get("id")]
        logger.error(f"File upload failed (HTTP {r.status_code}).")
    except Exception as exc:
        logger.error(f"File upload raised {type(exc).__name__}.")
    return []


def create_post(channel_id: str, message: str,
                file_ids: Optional[List[str]] = None,
                props: Optional[Dict] = None) -> Optional[str]:
    """Create a post in ``channel_id`` (optionally with files/props); return its id."""
    if not (channel_id and is_configured()):
        return None
    payload: Dict = {"channel_id": channel_id, "message": message}
    if file_ids:
        payload["file_ids"] = file_ids
    if props:
        payload["props"] = props
    try:
        with httpx.Client(timeout=_HTTP_TIMEOUT) as client:
            r = client.post(f"{MATTERMOST_URL}/api/v4/posts",
                            headers=_auth_headers(), json=payload)
        if r.status_code in (200, 201):
            return r.json().get("id")
        logger.error(f"Create post failed (HTTP {r.status_code}).")
    except Exception as exc:
        logger.error(f"Create post raised {type(exc).__name__}.")
    return None