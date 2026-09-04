"""
response_router.py — Decide *where* an already-generated answer goes.

This is the seam the whole feature is built around: the retrieval engine returns
``answer`` / ``sources`` / ``citations`` / ``metadata`` and knows **nothing** about
delivery; this router takes the rendered payload plus a :class:`Destination` and
dispatches to exactly one destination handler:

    me      → dm_handler.send_to_my_dm
    user    → user_handler.send_to_user_dm
    channel → channel_handler.send_to_channel
    group   → group_handler.send_to_group_dm

Every handler returns a uniform :class:`DeliveryResult`, so the caller (the bot's
background task, or the Share-button dialog handler) can surface a consistent
confirmation / error to the requester without knowing which target was used.

Adding a new destination later means adding a handler + one line in
:data:`_DISPATCH` — nothing else in the codebase changes.
"""

from __future__ import annotations

import logging
from typing import Callable, Optional, Set

from integrations.command_parser import Destination
from integrations.destination_handlers import (
    channel_handler, dm_handler, group_handler, user_handler,
)
from integrations.destination_handlers.base import (
    DeliveryResult, Requester, ResponsePayload,
)

logger = logging.getLogger("mattermost_bot")


# ── Dispatch table: destination kind → handler callable ──────────────────────────
# Kept as a table (not an if/elif chain) so destinations stay data-driven and
# easy to extend.
_DISPATCH: dict = {
    "me": dm_handler.send_to_my_dm,
    "user": user_handler.send_to_user_dm,
    "group": group_handler.send_to_group_dm,
    # "channel" is handled separately because it also takes the allowlist.
}


def send_response(destination: Destination, payload: ResponsePayload,
                  requester: Requester,
                  allowed_channel_ids: Optional[Set[str]] = None) -> DeliveryResult:
    """
    Route ``payload`` to ``destination`` and return the delivery outcome.

    ``allowed_channel_ids`` (optional) is forwarded to the channel handler so the
    same allowlist the slash command enforces also applies to shared channel posts.
    """
    kind = (destination.kind or "me").lower()
    logger.info(f"routing answer  kind={kind}  target={destination.raw_target!r}")

    try:
        if kind == "channel":
            return channel_handler.send_to_channel(
                destination, payload, requester, allowed_channel_ids=allowed_channel_ids
            )

        handler: Optional[Callable] = _DISPATCH.get(kind)
        if handler is None:
            return DeliveryResult(False, error=f"Unknown destination: {kind!r}.")
        return handler(destination, payload, requester)

    except Exception as exc:  # a handler must never crash the background task
        logger.error(f"Delivery to {kind!r} raised {type(exc).__name__}.")
        return DeliveryResult(
            False,
            error="Something went wrong while delivering the answer. Please try again.",
        )
