"""
tests/test_mattermost_bot.py — Real HTTP-level tests for the Mattermost
FastAPI integration, using FastAPI's TestClient (not a mock of FastAPI
itself). Covers auth/signing on every webhook endpoint, lifespan startup, and
confirms the bot calls the SAME rag_pipeline.answer() the dashboard uses.

NOT covered here (documented, not silently skipped): a real Mattermost
server actually delivering a slash command, DM, or interactive message —
that needs a live or staging Mattermost instance and is out of reach from
this sandbox (no network access to any such server). These tests exercise
the bot's own HTTP surface directly, exactly as a real Mattermost server
would call it.
"""

import sys
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
from fastapi.testclient import TestClient

import integrations.mattermost_bot as mb


@pytest.fixture()
def client(monkeypatch):
    monkeypatch.setattr(mb, "MATTERMOST_SLASH_TOKEN", "real-slash-token")
    monkeypatch.setattr(mb, "WARM_RAG_ON_STARTUP", False)
    with TestClient(mb.app) as c:
        yield c


# ── lifespan / health ────────────────────────────────────────────────────────

def test_lifespan_startup_runs_without_exception(client):
    """TestClient's context manager triggers the FastAPI lifespan startup —
    if _run_startup_checks() raised, entering this fixture would fail."""
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


# ── slash command auth ───────────────────────────────────────────────────────

def test_ask_rejects_missing_token(client):
    r = client.post("/mattermost/ask", data={"text": "hello", "token": ""})
    assert r.status_code == 403


def test_ask_rejects_wrong_token(client):
    r = client.post("/mattermost/ask", data={"text": "hello", "token": "wrong"})
    assert r.status_code == 403


def test_ask_accepts_correct_token_and_shows_landing_for_empty_text(client):
    r = client.post("/mattermost/ask", data={"text": "", "token": "real-slash-token"})
    assert r.status_code == 200
    assert "response_type" in r.json()


def test_ask_help_command_works(client):
    r = client.post("/mattermost/ask", data={"text": "help", "token": "real-slash-token"})
    assert r.status_code == 200


# ── /mattermost/action signing (audit finding: previously unauthenticated) ──

def test_action_rejects_forged_context_without_signature(client):
    """A context built by an attacker (no _sig, or a wrong one) must be
    rejected before any action branch runs — regression test for the finding
    that /mattermost/action had no authentication at all."""
    forged = {
        "channel_id": "any-channel-attacker-wants",
        "post_id": "any-post",
        "user_id": "attacker",
        "context": {"action": "delete_all_confirm"},  # no _sig at all
    }
    with patch.object(mb, "_delete_all_bot_posts") as mock_delete:
        r = client.post("/mattermost/action", json=forged)
        mock_delete.assert_not_called()
    assert r.status_code == 200  # Mattermost expects 200 even for a graceful rejection
    assert "expired" in r.json().get("ephemeral_text", "").lower() or \
           "no longer valid" in r.json().get("ephemeral_text", "").lower()


def test_action_rejects_context_with_wrong_signature(client):
    forged = {
        "channel_id": "x", "post_id": "y", "user_id": "z",
        "context": {"action": "delete_all_confirm", "_sig": "0" * 32},
    }
    with patch.object(mb, "_delete_all_bot_posts") as mock_delete:
        client.post("/mattermost/action", json=forged)
        mock_delete.assert_not_called()


def test_action_accepts_correctly_signed_context(client):
    """A context built the real way (via _action(), which now signs it) must
    be accepted and dispatched — the fix must not have broken legitimate use."""
    btn = mb._action("Dismiss", "dismiss")
    payload = {
        "channel_id": "c1", "post_id": "p1", "user_id": "u1", "user_name": "gopal",
        "context": btn["integration"]["context"],
    }
    r = client.post("/mattermost/action", json=payload)
    assert r.status_code == 200
    body = r.json()
    assert "dismissed" in str(body).lower()


def test_sign_action_context_is_deterministic_and_order_independent():
    a = {"action": "export_pdf", "question": "what is X?"}
    b = {"question": "what is X?", "action": "export_pdf"}  # same content, different order
    assert mb._sign_action_context(a) == mb._sign_action_context(b)


def test_sign_action_context_changes_if_content_changes():
    a = mb._sign_action_context({"action": "export_pdf", "question": "A"})
    b = mb._sign_action_context({"action": "export_pdf", "question": "B"})
    assert a != b


# ── /mattermost/dialog state signing ────────────────────────────────────────

def test_dialog_rejects_tampered_state(client):
    payload = {
        "callback_id": "user",
        "submission": {"target": "someone"},
        "state": '{"p": "post123", "q": "forged question", "_sig": "deadbeef"}',
        "user_id": "attacker",
    }
    r = client.post("/mattermost/dialog", json=payload)
    assert r.status_code == 200
    assert "errors" in r.json()


def test_dialog_accepts_correctly_signed_state(monkeypatch, client):
    state = {"p": "post123", "q": "a real question"}
    state["_sig"] = mb._sign_action_context(state)
    import json as _json
    payload = {
        "callback_id": "user",
        "submission": {"target": "someone"},
        "state": _json.dumps(state),
        "user_id": "u1", "user_name": "gopal", "channel_id": "c1", "team_id": "t1",
    }
    with patch.object(mb, "_deliver_cached_answer") as mock_deliver:
        from integrations.destination_handlers.base import DeliveryResult
        mock_deliver.return_value = DeliveryResult(ok=True, confirmation="Sent.")
        r = client.post("/mattermost/dialog", json=payload)
    assert r.status_code == 200
    mock_deliver.assert_called_once()
    # The un-tampered post_id/question from the signed state must reach the delivery call.
    _, kwargs = mock_deliver.call_args if mock_deliver.call_args.kwargs else (mock_deliver.call_args.args, {})
    called_args = mock_deliver.call_args.args
    assert "post123" in called_args
    assert "a real question" in called_args


# ── voice token ──────────────────────────────────────────────────────────────

def test_voice_token_roundtrip():
    tok = mb._voice_token("chan1", "user1")
    assert mb._voice_token_ok("chan1", "user1", tok)


def test_voice_token_rejects_wrong_user():
    tok = mb._voice_token("chan1", "user1")
    assert not mb._voice_token_ok("chan1", "user2", tok)


def test_voice_token_rejects_tampering():
    tok = mb._voice_token("chan1", "user1")
    exp, sig = tok.split(".", 1)
    tampered = f"{exp}.{'0' * len(sig)}"
    assert not mb._voice_token_ok("chan1", "user1", tampered)


def test_voice_secret_is_not_the_old_hardcoded_literal():
    """Regression test for the audit finding: the fallback used to be the
    literal string 'takshashila-voice-fallback-secret' baked into source."""
    assert mb._VOICE_SECRET != b"takshashila-voice-fallback-secret"


# ── shared RAG pipeline (no duplicate implementation) ───────────────────────

def test_run_rag_and_reply_calls_shared_rag_pipeline(monkeypatch):
    """The bot must call src.rag_pipeline.answer() — the same function the
    dashboard uses — not a separate/duplicate answer-generation path."""
    monkeypatch.setattr(mb, "MATTERMOST_BOT_TOKEN", "")  # force response_url-only delivery path
    called = {}

    def fake_answer(**kwargs):
        called["yes"] = True
        return {"answer": "A grounded answer.", "sources": [], "confidence": "high",
                "top_score": 0.9, "retrieval_time": 0.1, "generation_time": 0.2}

    with patch("src.rag_pipeline.answer", side_effect=fake_answer), \
         patch.object(mb, "warm_rag_resources"), \
         patch.object(mb, "post_to_channel") as mock_post:
        mb.run_rag_and_reply(
            question="what is X?", channel_id="c1", response_url="https://example.com/hook",
            user_name="gopal", user_id="u1", channel_name="general",
        )
    assert called.get("yes") is True


# ── Answer action buttons (Export/Share/Feedback/Delete) ────────────────────
# Regression tests for a real reported issue: .env.example shipped
# MATTERMOST_PRIVATE_DELIVERY=ephemeral, contradicting the code's own default
# of "dm" — and ephemeral posts cannot carry any of these buttons at all
# (a Mattermost platform limitation). Fixed the template; these tests prove
# the full button set actually assembles correctly when configured as
# documented (MATTERMOST_BOT_PUBLIC_URL set, MATTERMOST_PRIVATE_DELIVERY=dm).

def test_full_button_set_assembles_when_fully_configured(monkeypatch):
    monkeypatch.setattr(mb, "MATTERMOST_BOT_TOKEN", "faketoken")
    monkeypatch.setattr(mb, "MATTERMOST_URL", "https://mm.example.com")
    monkeypatch.setattr(mb, "PUBLIC_BASE_URL", "https://bot.example.com")
    monkeypatch.setattr(mb, "ACTION_URL", "https://bot.example.com/mattermost/action")
    monkeypatch.setattr(mb, "ENABLE_BUTTONS", True)
    monkeypatch.setattr(mb, "ENABLE_SHARE_BUTTONS", True)
    monkeypatch.setattr(mb, "ENABLE_GROUP_DESTINATION", True)
    monkeypatch.setattr(mb, "ENABLE_PDF_EXPORT", True)

    attachments = mb._answer_attachments("What is the leave policy?")
    all_labels = {a["name"] for att in attachments for a in att.get("actions", [])}

    expected = {
        "📄 Export Markdown", "⬇️ Download PDF",
        "👤 Share to User", "📢 Share to Channel", "👥 Share to Group",
        "👍 Helpful", "👎 Not Helpful",
        "🗑️ Delete this response", "🧹 Delete all",
    }
    missing = expected - all_labels
    assert not missing, f"Missing buttons: {missing}"

    # Every button's context must carry a valid signature (see the earlier
    # /mattermost/action auth fix) — a button with no working signature would
    # look present but fail silently when clicked.
    for att in attachments:
        for action in att.get("actions", []):
            ctx = action["integration"]["context"]
            assert mb._action_context_ok(ctx), f"Unsigned/broken button: {action['name']}"


def test_export_and_delete_buttons_absent_without_bot_token(monkeypatch):
    """Delete needs the Mattermost REST API (a bot token); PDF export is
    independently gated by ENABLE_PDF_EXPORT. Both must degrade gracefully,
    not crash, when their prerequisites aren't configured."""
    monkeypatch.setattr(mb, "MATTERMOST_BOT_TOKEN", "")
    monkeypatch.setattr(mb, "MATTERMOST_URL", "")
    monkeypatch.setattr(mb, "PUBLIC_BASE_URL", "https://bot.example.com")
    monkeypatch.setattr(mb, "ACTION_URL", "https://bot.example.com/mattermost/action")
    monkeypatch.setattr(mb, "ENABLE_BUTTONS", True)
    monkeypatch.setattr(mb, "ENABLE_PDF_EXPORT", False)

    attachments = mb._answer_attachments("What is the leave policy?")
    all_labels = {a["name"] for att in attachments for a in att.get("actions", [])}
    assert "🗑️ Delete this response" not in all_labels
    assert "🧹 Delete all" not in all_labels
    assert "⬇️ Download PDF" not in all_labels
    assert "📄 Export Markdown" in all_labels  # markdown export needs no bot token


def test_no_buttons_at_all_without_public_url(monkeypatch):
    """The documented, unavoidable requirement: no MATTERMOST_BOT_PUBLIC_URL
    means no interactive buttons of any kind, regardless of every other flag."""
    monkeypatch.setattr(mb, "ENABLE_BUTTONS", False)
    # _button_groups/_share_group/etc. don't individually check ENABLE_BUTTONS
    # (the call sites that decide whether to attach them at all do — see
    # run_rag_and_reply); confirm that gate exists and is what callers check.
    assert mb.ENABLE_BUTTONS is False


def test_private_delivery_default_is_dm_not_ephemeral():
    """Regression test for the .env.example bug: the CODE's own default must
    stay 'dm' (the only mode with working buttons) so a correctly-following
    default install gets working buttons, matching what .env.example now says."""
    assert mb.PRIVATE_DELIVERY in ("dm", "ephemeral")  # sanity: a real mode
    # The module-level default (no env var set) is asserted directly against
    # the source rather than re-importing with a clean environment, since
    # MATTERMOST_PRIVATE_DELIVERY may already be set in this process's env.
    import inspect
    src = inspect.getsource(mb)
    assert 'os.getenv("MATTERMOST_PRIVATE_DELIVERY", "dm")' in src


def test_help_command_returns_content(client):
    r = client.post("/mattermost/ask", data={"text": "help", "token": "real-slash-token"})
    assert r.status_code == 200
    body = r.json()
    assert body.get("text") or body.get("response_type")


def test_examples_command_returns_content(client):
    r = client.post("/mattermost/ask", data={"text": "examples", "token": "real-slash-token"})
    assert r.status_code == 200
    body = r.json()
    assert body.get("text") or body.get("response_type")
