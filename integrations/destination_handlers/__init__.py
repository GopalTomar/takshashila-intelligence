"""
destination_handlers — one small module per delivery target.

Each handler exposes a single function that takes a parsed
``Destination`` + a rendered ``ResponsePayload`` + the ``Requester`` and returns a
``DeliveryResult``:

    dm_handler.send_to_my_dm         → the requester's own DM (default behaviour)
    user_handler.send_to_user_dm     → another user's DM
    channel_handler.send_to_channel  → a channel
    group_handler.send_to_group_dm   → a group DM

Handlers only decide/resolve a target channel and then reuse the shared
``base.deliver`` primitive (which in turn reuses the bot's existing
``post_to_channel``). They never touch the RAG pipeline.
"""
