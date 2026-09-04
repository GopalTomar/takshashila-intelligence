"""
tests/test_group_delivery.py — End-to-end group-DM delivery logic (mocked API).

These tests exercise ``group_handler.send_to_group_dm`` without a live Mattermost
by stubbing ``mattermost_api`` and the ``deliver`` post step. They prove the
handler only reports success when the message was actually posted into a group
channel containing the intended people, and returns a meaningful error otherwise.

Run:  pytest tests/test_group_delivery.py   ·   python tests/test_group_delivery.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from integrations.command_parser import Destination                      # noqa: E402
from integrations.destination_handlers import base, group_handler        # noqa: E402
from integrations.destination_handlers.base import Requester, ResponsePayload  # noqa: E402
from integrations import mattermost_api                                   # noqa: E402


BOT_ID = "bot_id"
USERS = {                       # username → user id
    "lakshmi": "u_lakshmi",
    "nithiya": "u_nithiya",
    "abhishek.k": "u_abhishek",
}
REQUESTER = Requester(user_id="u_gopal", user_name="gopaltomar", channel_id="chan1")
PAYLOAD = ResponsePayload(question="What is the LPG crisis?", message="Here is the answer.")


class FakeState:
    """Installs configurable fakes onto mattermost_api + group_handler.deliver."""

    def __init__(self, *, members=None, create_ok=True, deliver_id="post1",
                 users=None, show_ok=2):
        self.users = users if users is not None else USERS
        self.create_ok = create_ok
        self.deliver_id = deliver_id
        self.members = members            # what get_channel_members returns
        self.show_ok = show_ok
        self.created_with = None
        self.posted_channel = None
        self._orig = {}

    def __enter__(self):
        m = mattermost_api
        self._orig = {
            "is_configured": m.is_configured,
            "get_bot_user_id": m.get_bot_user_id,
            "find_user_by_username": m.find_user_by_username,
            "create_group_channel": m.create_group_channel,
            "get_channel_members": m.get_channel_members,
            "show_group_channel_to_members": m.show_group_channel_to_members,
            "deliver": group_handler.deliver,
        }
        m.is_configured = lambda: True
        m.get_bot_user_id = lambda: BOT_ID
        m.find_user_by_username = lambda name: (
            {"id": self.users[name], "username": name} if name in self.users else None
        )

        def _create(member_ids):
            self.created_with = list(member_ids)
            return {"id": "gm1", "type": "G"} if self.create_ok else None
        m.create_group_channel = _create

        def _members(cid):
            if self.members is not None:
                return list(self.members)
            # default: everyone we were asked to create + the bot
            return list(self.created_with or []) + [BOT_ID]
        m.get_channel_members = _members
        m.show_group_channel_to_members = lambda cid, ids: self.show_ok

        def _deliver(channel_id, payload, header=""):
            self.posted_channel = channel_id
            return self.deliver_id
        group_handler.deliver = _deliver
        return self

    def __exit__(self, *exc):
        m = mattermost_api
        m.is_configured = self._orig["is_configured"]
        m.get_bot_user_id = self._orig["get_bot_user_id"]
        m.find_user_by_username = self._orig["find_user_by_username"]
        m.create_group_channel = self._orig["create_group_channel"]
        m.get_channel_members = self._orig["get_channel_members"]
        m.show_group_channel_to_members = self._orig["show_group_channel_to_members"]
        group_handler.deliver = self._orig["deliver"]


def _send(usernames, **kwargs):
    dest = Destination("group", usernames=tuple(usernames))
    with FakeState(**kwargs) as st:
        result = group_handler.send_to_group_dm(dest, PAYLOAD, REQUESTER)
    return result, st


# ── Success path ─────────────────────────────────────────────────────────────────

def test_group_success_includes_requester_and_posts():
    result, st = _send(["lakshmi", "nithiya"])
    assert result.ok is True
    assert result.post_id == "post1"
    assert result.target_channel_id == "gm1"
    # requester is included as a member; bot is added by the API layer
    assert set(st.created_with) == {"u_lakshmi", "u_nithiya", "u_gopal"}
    assert st.posted_channel == "gm1"
    # 3 human participants: 2 recipients + requester
    assert "3 participants" in result.confirmation

def test_group_single_recipient_now_valid_with_requester():
    # bot + requester + 1 recipient = 3 (the minimum) → allowed
    result, st = _send(["lakshmi"])
    assert result.ok is True
    assert set(st.created_with) == {"u_lakshmi", "u_gopal"}


# ── Failure paths → meaningful errors, never false success ───────────────────────

def test_missing_user_errors_and_does_not_post():
    result, st = _send(["lakshmi", "ghost"])
    assert result.ok is False
    assert "ghost" in result.error
    assert st.posted_channel is None            # nothing posted

def test_channel_creation_failure_errors():
    result, st = _send(["lakshmi", "nithiya"], create_ok=False)
    assert result.ok is False
    assert "couldn't create" in result.error.lower()
    assert st.posted_channel is None

def test_membership_verification_failure_errors():
    # channel reports members WITHOUT nithiya → must not claim success
    result, st = _send(["lakshmi", "nithiya"], members=["u_lakshmi", "u_gopal", BOT_ID])
    assert result.ok is False
    assert "couldn't confirm everyone" in result.error.lower()

def test_post_failure_errors():
    result, st = _send(["lakshmi", "nithiya"], deliver_id="")
    assert result.ok is False
    assert "failed to post" in result.error.lower()


# ── Edge cases: bot in list, duplicates, too many ────────────────────────────────

def test_bot_and_duplicates_are_ignored():
    users = dict(USERS, thebot="bot_id")        # a username that maps to the bot
    dest = Destination("group", usernames=("lakshmi", "lakshmi", "thebot", "nithiya"))
    with FakeState(users=users) as st:
        result = group_handler.send_to_group_dm(dest, PAYLOAD, REQUESTER)
    assert result.ok is True
    # bot excluded, duplicate collapsed → recipients {lakshmi, nithiya} + requester
    assert set(st.created_with) == {"u_lakshmi", "u_nithiya", "u_gopal"}

def test_too_many_users_errors():
    users = {f"u{i}": f"id{i}" for i in range(8)}
    dest = Destination("group", usernames=tuple(users))
    with FakeState(users=users) as st:
        result = group_handler.send_to_group_dm(dest, PAYLOAD, REQUESTER)
    assert result.ok is False
    assert "at most" in result.error.lower()

def test_no_usernames_errors():
    dest = Destination("group", usernames=())
    with FakeState() as st:
        result = group_handler.send_to_group_dm(dest, PAYLOAD, REQUESTER)
    assert result.ok is False


if __name__ == "__main__":
    mod = sys.modules[__name__]
    passed = 0
    for name in sorted(dir(mod)):
        if name.startswith("test_"):
            getattr(mod, name)()
            passed += 1
    print(f"OK — {passed} tests passed")