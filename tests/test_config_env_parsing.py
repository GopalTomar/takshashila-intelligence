"""
tests/test_config_env_parsing.py — Regression tests for a real crash a user
hit: every numeric setting in src/config.py and integrations/mattermost_bot.py
used `float(os.getenv(NAME, "default"))` / `int(os.getenv(NAME, "default"))`.
os.getenv's default only applies when a variable is completely UNSET — if
it's present in .env but left blank (e.g. `SOURCE_PRIORITY_BOOST=` with
nothing after the `=`, an easy mistake filling in a template), os.getenv
returns "", and float("")/int("") raises ValueError, crashing the whole app
at import time with no clue which setting caused it.

src.config._env_float / _env_int now treat a blank/whitespace-only value the
same as "not set" (and fall back to the default on any other malformed value
too, rather than crashing).
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import _env_float, _env_int


def test_env_float_blank_falls_back_to_default(monkeypatch):
    monkeypatch.setenv("SOME_FLOAT_SETTING", "")
    assert _env_float("SOME_FLOAT_SETTING", 0.12) == 0.12


def test_env_float_whitespace_only_falls_back_to_default(monkeypatch):
    monkeypatch.setenv("SOME_FLOAT_SETTING", "   ")
    assert _env_float("SOME_FLOAT_SETTING", 0.12) == 0.12


def test_env_float_unset_falls_back_to_default(monkeypatch):
    monkeypatch.delenv("SOME_FLOAT_SETTING", raising=False)
    assert _env_float("SOME_FLOAT_SETTING", 0.12) == 0.12


def test_env_float_real_value_is_used(monkeypatch):
    monkeypatch.setenv("SOME_FLOAT_SETTING", "0.25")
    assert _env_float("SOME_FLOAT_SETTING", 0.12) == 0.25


def test_env_float_malformed_value_falls_back_instead_of_crashing(monkeypatch, capsys):
    monkeypatch.setenv("SOME_FLOAT_SETTING", "not-a-number")
    assert _env_float("SOME_FLOAT_SETTING", 0.12) == 0.12


def test_env_int_blank_falls_back_to_default(monkeypatch):
    monkeypatch.setenv("SOME_INT_SETTING", "")
    assert _env_int("SOME_INT_SETTING", 18) == 18


def test_env_int_real_value_is_used(monkeypatch):
    monkeypatch.setenv("SOME_INT_SETTING", "9")
    assert _env_int("SOME_INT_SETTING", 18) == 9


def test_config_module_imports_cleanly_with_every_numeric_var_blank(monkeypatch):
    """The exact real-world scenario: a user's .env has every numeric
    setting present but blank (e.g. copied from a template and never
    filled in). Reloading src.config under these conditions must not raise."""
    blank_numeric_vars = [
        "SCRAPE_DELAY", "SCRAPE_TIMEOUT", "SCRAPE_MAX_RETRIES",
        "WEBSITE_MIN_TEXT_LEN", "WEBSITE_MAX_PAGES", "WEBSITE_MAX_DEPTH",
        "SCRAPE_MAX_WORKERS", "CHUNK_SIZE", "CHUNK_OVERLAP", "CHUNK_MIN_LEN",
        "TOP_K", "MIN_SCORE_THRESHOLD", "CONF_HIGH_THRESHOLD",
        "CONF_MEDIUM_THRESHOLD", "GROUNDING_MIN_OVERLAP",
        "SOURCE_PRIORITY_BOOST", "SCHEDULE_HOUR", "SCHEDULE_MINUTE",
    ]
    for name in blank_numeric_vars:
        monkeypatch.setenv(name, "")

    import importlib
    import src.config as config
    importlib.reload(config)
    try:
        assert config.SOURCE_PRIORITY_BOOST == 0.12
        assert config.SCHEDULE_HOUR == 18
        assert config.CHUNK_SIZE == 1200
    finally:
        importlib.reload(config)  # restore normal env for any tests that follow
