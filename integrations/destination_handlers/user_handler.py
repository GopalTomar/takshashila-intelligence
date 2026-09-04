"""
user_handler.py — Deliver an answer to another user's direct messages.

    /askkb --user abhishek.k Leave policy

Flow: resolve the username → open/reuse the bot↔user DM → post the answer (with an
attribution header and an @mention so they're notified) → confirm to the requester.
"""

from __future__ import annotations

import logging

from integrations import mattermost_api
from integrations.command_parser import Destination
from integrations.destination_handlers.base import (
    DeliveryResult, Requester, ResponsePayload, build_shared_header, deliver,
)

logger = logging.getLogger("mattermost_bot")


def send_to_user_dm(destination: Destination, payload: ResponsePayload,
                    requester: Requester) -> DeliveryResult:
    """Deliver ``payload`` to another user's DM with the bot."""
    if not mattermost_api.is_configured():
        return DeliveryResult(False, error="The bot token is not configured, so I can't send direct messages.")

    username = destination.usernames[0] if destination.usernames else ""
    if not username:
        return DeliveryResult(False, error="Please provide a username, e.g. `--user abhishek.k`.")

    user = mattermost_api.find_user_by_username(username)
    if not user or not user.get("id"):
        return DeliveryResult(False, error=f'User "{username}" not found.')

    dm_channel = mattermost_api.get_or_create_dm_channel(user["id"])
    if not dm_channel:
        return DeliveryResult(False, error=f'I couldn\'t open a direct message with @{username}.')

    header = build_shared_header(requester, payload.question, mention=f"@{username}")
    post_id = deliver(dm_channel, payload, header=header)
    if not post_id:
        return DeliveryResult(False, error=f'I couldn\'t deliver the answer to @{username}.')

    logger.info(f"delivered to user DM  target={username!r}  post_id={post_id!r}")
    return DeliveryResult(
        True,
        confirmation=f"✅ Successfully sent to **@{username}**",
        target_channel_id=dm_channel,
        post_id=post_id,
    )