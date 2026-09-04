"""
group_handler.py — Deliver an answer to a group direct message, end-to-end.

    /askkb -g abhishek.k,nithiya Explain leave policy      (also: --group)

The flow validates **every** step and never reports success unless the message
was actually posted into a group channel that contains the intended people:

1. resolve every username → real, active user id (report any that don't exist);
2. drop the bot's own account and de-duplicate;
3. include the **requester** in the group (it's their conversation) — this also
   means a single recipient is enough (bot + requester + 1 recipient = 3, the
   Mattermost minimum);
4. create / reuse the group channel and validate the response;
5. verify the channel actually contains the requester + recipients;
6. post the answer and validate the created post;
7. surface the group DM in each member's sidebar (best-effort);
8. only then confirm — otherwise return a meaningful error.
"""

from __future__ import annotations

import logging
from typing import List, Tuple

from integrations import mattermost_api
from integrations.command_parser import Destination
from integrations.destination_handlers.base import (
    DeliveryResult, Requester, ResponsePayload, build_shared_header, deliver,
)

logger = logging.getLogger("mattermost_bot")


def _resolve_users(usernames: Tuple[str, ...], bot_id: str
                   ) -> Tuple[List[str], List[str], List[str]]:
    """
    Resolve names → user ids for a group DM.

    Returns ``(resolved_ids, resolved_names, missing_names)``:
      * de-duplicated by id;
      * the bot's own account is skipped (it's added as a member automatically);
      * a name that doesn't resolve to a user is reported in ``missing_names``.
    """
    resolved_ids: List[str] = []
    resolved_names: List[str] = []
    missing: List[str] = []
    for name in usernames:
        user = mattermost_api.find_user_by_username(name)
        if user and user.get("id"):
            uid = user["id"]
            if uid == bot_id:
                continue                          # bot added automatically
            if uid in resolved_ids:
                continue                          # duplicate user → skip silently
            resolved_ids.append(uid)
            resolved_names.append(user.get("username") or name)
        else:
            missing.append(name)
    return resolved_ids, resolved_names, missing


def send_to_group_dm(destination: Destination, payload: ResponsePayload,
                     requester: Requester) -> DeliveryResult:
    """Deliver ``payload`` to a group DM (bot + requester + named users)."""
    if not mattermost_api.is_configured():
        return DeliveryResult(False, error="The bot token is not configured, so I can't create group messages.")

    usernames = destination.usernames
    if not usernames:
        return DeliveryResult(False, error="Please list users, e.g. `-g abhishek.k,nithiya`.")

    bot_id = mattermost_api.get_bot_user_id()
    if not bot_id:
        logger.error("Group DM: bot user id could not be resolved.")
        return DeliveryResult(False, error="I couldn't identify the bot account. Check the bot token.")

    # 1–2) Resolve + validate every username.
    recipient_ids, recipient_names, missing = _resolve_users(usernames, bot_id)
    if missing:
        names = ", ".join(f'"{m}"' for m in missing)
        return DeliveryResult(
            False,
            error=f"Unable to create the group message — these users were not found: {names}.",
        )
    if not recipient_ids:
        return DeliveryResult(
            False,
            error="No valid recipients were found for the group message.",
        )

    # 3) Include the requester so it's a real group conversation (dedupe).
    member_ids: List[str] = list(recipient_ids)
    if requester.user_id and requester.user_id != bot_id and requester.user_id not in member_ids:
        member_ids.append(requester.user_id)

    # Mattermost group DMs allow 3–8 total members (bot + these). Guard clearly.
    if len(member_ids) < (mattermost_api.GROUP_DM_MIN_MEMBERS - 1):
        return DeliveryResult(
            False,
            error="A group message needs at least one other person besides you. "
                  "For a single person, use `-u <username>` (`--user`) instead.",
        )
    if len(member_ids) > (mattermost_api.GROUP_DM_MAX_MEMBERS - 1):
        allowed = mattermost_api.GROUP_DM_MAX_MEMBERS - 2   # minus bot, minus requester
        return DeliveryResult(
            False,
            error=f"A group message supports at most {allowed} other users. "
                  "Please share with fewer people or use a channel.",
        )

    # 4) Create / reuse the group channel — validated (status, id, type "G").
    channel = mattermost_api.create_group_channel(member_ids)
    if not channel or not channel.get("id"):
        return DeliveryResult(
            False,
            error="I couldn't create the group message (Mattermost rejected the request). "
                  "Please check the bot's permissions and try again.",
        )
    channel_id = channel["id"]

    # 5) Verify the intended people are actually members before claiming success.
    members = set(mattermost_api.get_channel_members(channel_id))
    if members:                                    # only enforce if the list was retrievable
        expected = set(member_ids) | {bot_id}
        missing_ids = [m for m in expected if m not in members]
        if missing_ids:
            logger.error(
                f"Group DM {channel_id!r} is missing expected members {missing_ids} "
                f"(got {sorted(members)})."
            )
            return DeliveryResult(
                False,
                error="I created the group message but couldn't confirm everyone was added. "
                      "Please try again.",
            )

    # 6) Post the answer and validate the created post id.
    header = build_shared_header(requester, payload.question)
    post_id = deliver(channel_id, payload, header=header)
    if not post_id:
        return DeliveryResult(
            False,
            error="I created the group message but the answer failed to post. "
                  "Please check the bot's permissions and try again.",
        )

    # 7) Surface the GM in each member's sidebar (best-effort; logs a hint on 403).
    mattermost_api.show_group_channel_to_members(channel_id, member_ids)

    participants = len(member_ids)                 # humans in the group (recipients + requester)
    who = ", ".join(f"@{n}" for n in recipient_names)
    logger.info(f"delivered to group DM  channel={channel_id!r}  recipients={recipient_names}  "
                f"requester={requester.user_name!r}  post_id={post_id!r}")
    return DeliveryResult(
        True,
        confirmation=f"✅ Shared with {participants} participants ({who}).",
        target_channel_id=channel_id,
        post_id=post_id,
    )