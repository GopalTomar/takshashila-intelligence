"""
channel_handler.py — Post an answer into a channel.

    /askkb --channel research Explain this paper

Flow: resolve the channel by name on the requester's team → verify the bot is a
member (so it may post) → optionally honour the channel allowlist → post the
answer with an attribution header → confirm to the requester.
"""

from __future__ import annotations

import logging
from typing import Optional, Set

from integrations import mattermost_api
from integrations.command_parser import Destination
from integrations.destination_handlers.base import (
    DeliveryResult, Requester, ResponsePayload, build_shared_header, deliver,
)

logger = logging.getLogger("mattermost_bot")


def send_to_channel(destination: Destination, payload: ResponsePayload,
                    requester: Requester,
                    allowed_channel_ids: Optional[Set[str]] = None) -> DeliveryResult:
    """Deliver ``payload`` into a named channel on the requester's team."""
    if not mattermost_api.is_configured():
        return DeliveryResult(False, error="The bot token is not configured, so I can't post to channels.")

    channel_name = destination.channel_name
    if not channel_name:
        return DeliveryResult(False, error="Please provide a channel, e.g. `--channel research`.")
    if not requester.team_id:
        return DeliveryResult(False, error="I couldn't determine which team this channel belongs to.")

    channel = mattermost_api.find_channel_on_team(requester.team_id, channel_name)
    if not channel or not channel.get("id"):
        return DeliveryResult(False, error=f'Channel "{channel_name}" not found.')
    channel_id = channel["id"]

    # Respect the same channel allowlist the slash command already enforces.
    if allowed_channel_ids and channel_id not in allowed_channel_ids:
        return DeliveryResult(False, error=f'Posting to "{channel_name}" is not enabled for this bot.')

    if not mattermost_api.bot_in_channel(channel_id):
        return DeliveryResult(
            False,
            error=f'Bot is not a member of "{channel_name}". Add the bot to that channel and try again.',
        )

    header = build_shared_header(requester, payload.question)
    post_id = deliver(channel_id, payload, header=header)
    if not post_id:
        return DeliveryResult(False, error=f'I couldn\'t post the answer in "{channel_name}".')

    logger.info(f"delivered to channel  name={channel_name!r}  id={channel_id!r}  post_id={post_id!r}")
    display = channel.get("name") or channel_name
    return DeliveryResult(
        True,
        confirmation=f"✅ Posted in ~{display}",
        target_channel_id=channel_id,
        post_id=post_id,
    )