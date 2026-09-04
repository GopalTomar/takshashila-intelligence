"""
dm_handler.py — Deliver an answer to the requester's own direct messages.

This is the default destination (``/askkb <question>`` and ``/askkb --me …``). In
normal operation the bot's existing private-DM path handles this case directly;
this handler exists so the router can service an explicit ``--me`` and so all four
destinations share one uniform interface.
"""

from __future__ import annotations

import logging

from integrations import mattermost_api
from integrations.command_parser import Destination
from integrations.destination_handlers.base import (
    DeliveryResult, Requester, ResponsePayload, deliver,
)

logger = logging.getLogger("mattermost_bot")


def send_to_my_dm(destination: Destination, payload: ResponsePayload,
                  requester: Requester) -> DeliveryResult:
    """Deliver ``payload`` privately to the requester (bot↔requester DM)."""
    if not mattermost_api.is_configured():
        return DeliveryResult(False, error="The bot token is not configured, so I can't open a direct message.")
    if not requester.user_id:
        return DeliveryResult(False, error="I couldn't determine who to send this to.")

    dm_channel = mattermost_api.get_or_create_dm_channel(requester.user_id)
    if not dm_channel:
        return DeliveryResult(False, error="I couldn't open a direct message with you. Please try again.")

    # No attribution header — it's the requester's own private answer.
    post_id = deliver(dm_channel, payload)
    if not post_id:
        return DeliveryResult(False, error="I couldn't deliver the answer to your direct messages.")

    logger.info(f"delivered to self DM  user_id={requester.user_id!r}  post_id={post_id!r}")
    return DeliveryResult(
        True,
        confirmation="🔒 Sent the answer to your **direct messages** with me.",
        target_channel_id=dm_channel,
        post_id=post_id,
    )
