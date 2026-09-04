"""
tests/test_progress_indicator.py — Live "typing" progress indicator.

Exercises _ProgressIndicator without a live Mattermost by stubbing the bot's
post_to_channel / _patch_post / _delete_post. Verifies the status post is created,
animates through stages, and is edited into the final answer — and that a failed
final patch falls back to a fresh post so the answer is never lost.

Run:  python tests/test_progress_indicator.py   ·   pytest tests/test_progress_indicator.py
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import integrations.mattermost_bot as bot   # noqa: E402


class Recorder:
    """Stub post_to_channel / _patch_post / _delete_post and record calls."""

    def __init__(self, *, first_post_id="status1", patch_ok=True):
        self.first_post_id = first_post_id
        self.patch_ok = patch_ok
        self.posts = []        # (channel, message, props)
        self.patches = []      # (post_id, message, props)
        self.deletes = []      # post_id
        self._orig = {}

    def __enter__(self):
        self._orig = {
            "post_to_channel": bot.post_to_channel,
            "_patch_post": bot._patch_post,
            "_delete_post": bot._delete_post,
            "PROGRESS_INTERVAL": bot.PROGRESS_INTERVAL,
        }

        def post_to_channel(channel_id, message, response_url, props=None):
            self.posts.append((channel_id, message, props))
            # first call = the status post; later calls = fallback posts
            return self.first_post_id if len(self.posts) == 1 else "fallback1"

        def _patch_post(post_id, message, props=None):
            self.patches.append((post_id, message, props))
            return self.patch_ok

        def _delete_post(post_id):
            self.deletes.append(post_id)
            return True

        bot.post_to_channel = post_to_channel
        bot._patch_post = _patch_post
        bot._delete_post = _delete_post
        bot.PROGRESS_INTERVAL = 0.02      # fast animation for the test
        return self

    def __exit__(self, *exc):
        bot.post_to_channel = self._orig["post_to_channel"]
        bot._patch_post = self._orig["_patch_post"]
        bot._delete_post = self._orig["_delete_post"]
        bot.PROGRESS_INTERVAL = self._orig["PROGRESS_INTERVAL"]


STAGES = ["🔍 Searching…", "📚 Reading…", "🧠 Generating…", "✍️ Formatting…", "📨 Delivering…"]


def test_start_posts_first_stage():
    with Recorder() as rec:
        ind = bot._ProgressIndicator("chan1", STAGES)
        assert ind.start() is True
        assert ind.post_id == "status1"
        assert rec.posts[0] == ("chan1", STAGES[0], None)
        ind.finalize("done", {})


def test_animation_advances_through_stages():
    with Recorder() as rec:
        ind = bot._ProgressIndicator("chan1", STAGES)
        ind.start()
        time.sleep(0.02 * (len(STAGES) + 2))     # let it walk the stages
        ind.finalize("FINAL ANSWER", {"attachments": [{"actions": []}]})
        # it patched at least a couple of intermediate stages…
        stage_msgs = [m for (_pid, m, _p) in rec.patches]
        assert STAGES[1] in stage_msgs
        # …and the LAST patch is the final answer with its props
        assert rec.patches[-1][0] == "status1"
        assert rec.patches[-1][1] == "FINAL ANSWER"
        assert rec.patches[-1][2] == {"attachments": [{"actions": []}]}


def test_finalize_edits_same_post_no_new_post():
    with Recorder() as rec:
        ind = bot._ProgressIndicator("chan1", STAGES)
        ind.start()
        pid = ind.finalize("answer", {})
        assert pid == "status1"
        assert len(rec.posts) == 1               # only the status post; no extra post
        assert rec.patches[-1][1] == "answer"


def test_finalize_fallback_when_patch_fails():
    with Recorder(patch_ok=False) as rec:
        ind = bot._ProgressIndicator("chan1", STAGES)
        ind.start()
        pid = ind.finalize("answer", {"attachments": []})
        # patch failed → status post deleted + answer posted fresh
        assert "status1" in rec.deletes
        assert pid == "fallback1"
        assert rec.posts[-1][1] == "answer"


def test_start_returns_false_when_post_fails():
    with Recorder(first_post_id=None) as rec:
        ind = bot._ProgressIndicator("chan1", STAGES)
        assert ind.start() is False              # no status post → caller skips streaming


def test_cancel_deletes_status_post():
    with Recorder() as rec:
        ind = bot._ProgressIndicator("chan1", STAGES)
        ind.start()
        ind.cancel()
        assert "status1" in rec.deletes
        assert ind.post_id is None


if __name__ == "__main__":
    mod = sys.modules[__name__]
    passed = 0
    for name in sorted(dir(mod)):
        if name.startswith("test_"):
            getattr(mod, name)()
            passed += 1
    print(f"OK — {passed} tests passed")