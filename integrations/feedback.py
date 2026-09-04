"""
feedback.py — Append-only feedback log for the Takshashila Mattermost bot.

When a user taps 👍 / 👎 under an answer, the bot records a single structured,
secret-free line in ``data/logs/mattermost_feedback.jsonl``. Leadership can tail
or aggregate this file to see how helpful the assistant is over time.

Only non-sensitive fields are stored: a timestamp, the verdict, the (truncated)
question, the user-visible name/id Mattermost already exposes, and the channel.
No tokens, no document content, no embeddings.
"""

from __future__ import annotations

import json
import threading
import time
from pathlib import Path

try:
    from src import config as _config
    _LOG_PATH = _config.LOGS_DIR / "mattermost_feedback.jsonl"
except Exception:  # pragma: no cover
    _LOG_PATH = Path("data/logs/mattermost_feedback.jsonl")

_lock = threading.Lock()


def record_feedback(
    verdict: str,
    question: str,
    user_name: str = "",
    user_id: str = "",
    channel_id: str = "",
    post_id: str = "",
) -> None:
    """Append one feedback event. Never raises — feedback must not break the bot."""
    entry = {
        "ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "verdict": verdict,                       # "helpful" | "not_helpful"
        "question": (question or "")[:500],
        "user_name": user_name,
        "user_id": user_id,
        "channel_id": channel_id,
        "post_id": post_id,
    }
    try:
        with _lock:
            _LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
            with open(_LOG_PATH, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(entry, ensure_ascii=False) + "\n")
    except Exception:
        # Best-effort only; the caller logs failures via the main logger.
        pass


def feedback_log_path() -> Path:
    return _LOG_PATH