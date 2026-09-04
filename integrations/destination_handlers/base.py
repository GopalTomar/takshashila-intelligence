"""
base.py — Shared contracts + the one reusable delivery primitive for handlers.

Everything a destination handler needs that is *not* specific to a single target
lives here:

* :class:`Requester`        — who ran the command (for attribution + confirmations).
* :class:`ResponsePayload`  — the already-rendered answer to deliver (retrieval is
                              done; this layer only moves bytes to a destination).
* :class:`DeliveryResult`   — success/failure + the confirmation/error text to show
                              the requester.
* :func:`build_shared_header` — the "Shared by …" attribution banner prepended when
                              an answer is sent somewhere other than the asker's DM.
* :func:`deliver`           — post a payload to an already-resolved channel id,
                              **reusing the bot's existing** ``post_to_channel`` and
                              answer cache so delivered posts keep working buttons.

Keeping delivery in one primitive means there is no duplicated Mattermost posting
logic across the four handlers.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional


# ════════════════════════════════════════════════════════════════════════════════
#  Contracts
# ════════════════════════════════════════════════════════════════════════════════

@dataclass
class Requester:
    """The person who ran ``/askkb`` (or clicked a Share button)."""

    user_id: str
    user_name: str
    team_id: str = ""
    channel_id: str = ""          # channel the command was invoked from


@dataclass
class ResponsePayload:
    """
    A fully-rendered answer, ready to deliver. The retrieval engine produced
    ``answer``/``sources``/``metadata``; the bot turned that into ``message`` +
    ``props`` (buttons). This layer never regenerates any of it — it only routes.
    """

    question: str
    message: str                              # delivery-ready markdown
    props: Optional[Dict] = None              # interactive button attachments
    metadata: Optional[Dict] = None           # confidence, timings, mode, …
    citations: Optional[List[Dict]] = None     # displayed sources (for reference)
    mode: str = "normal"


@dataclass
class DeliveryResult:
    """Outcome of a delivery attempt, plus the text to show the requester."""

    ok: bool
    confirmation: str = ""        # ephemeral success text for the requester
    error: str = ""               # ephemeral failure text for the requester
    target_channel_id: str = ""
    post_id: str = ""


# ════════════════════════════════════════════════════════════════════════════════
#  Attribution header
# ════════════════════════════════════════════════════════════════════════════════

def build_shared_header(requester: Requester, question: str,
                        mention: str = "") -> str:
    """
    A compact attribution banner prepended to answers shared to another user,
    channel or group. If ``mention`` is given (e.g. ``@abhishek.k``) it is placed
    first, on its own line, so the target reliably receives a notification.

        @abhishek.k
        > 📣 **Shared via Takshashila Knowledge Assistant**
        > Shared by **@gopal.tomar** · Original question: _<question>_
    """
    q = (question or "").strip().replace("\n", " ")
    if len(q) > 300:
        q = q[:300].rstrip() + "…"
    by = requester.user_name or "a colleague"
    lines: List[str] = []
    if mention:
        lines.append(mention)
    lines.append("> 📣 **Shared via Takshashila Knowledge Assistant**")
    lines.append(f"> Shared by **@{by}** · Original question: _{q}_")
    return "\n".join(lines)


# ════════════════════════════════════════════════════════════════════════════════
#  The single reusable delivery primitive
# ════════════════════════════════════════════════════════════════════════════════

def deliver(channel_id: str, payload: ResponsePayload,
            header: str = "") -> Optional[str]:
    """
    Post ``payload`` to an already-resolved ``channel_id`` and return the created
    post id (or ``None``).

    Reuses the bot's existing ``post_to_channel`` (bot-token REST path) so the
    delivered post carries the same interactive buttons as a normal answer, and
    caches it via the bot's answer cache so those buttons work immediately. The
    import is done lazily to avoid any import cycle with the FastAPI app module.
    """
    # Lazy import keeps this package free of a hard dependency on the FastAPI app.
    from integrations import mattermost_bot as bot

    body = f"{header}\n\n{payload.message}" if header else payload.message
    body = bot._truncate(body)

    post_id = bot.post_to_channel(channel_id, body, response_url=None, props=payload.props)

    if post_id:
        bot._cache_put(post_id, {
            "question": payload.question,
            "mode": payload.mode,
            "message": body,
            # Grounding sources of the shared answer, so 🔗 Related Policies works
            # on the delivered post too (never a fresh search).
            "sources": payload.citations or [],
        })
    return post_id