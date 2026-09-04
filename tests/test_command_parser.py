"""
tests/test_command_parser.py — Normalized /askkb command parsing.

Pure-Python tests (no index, no network) for the upgraded parser: long flags and
short aliases, order-independent modifiers, the help/examples/empty commands,
group routing, and graceful errors. Run with:

    pytest tests/test_command_parser.py
    python  tests/test_command_parser.py      # also works without pytest
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from integrations.command_parser import (  # noqa: E402
    Destination, autocomplete_spec, build_destination, dequote, parse_command,
)


# ── Defaults + backward compatibility ────────────────────────────────────────────

def test_no_flag_defaults_to_me():
    r = parse_command("What is the leave policy?")
    assert r.command == "ask" and r.error is None
    assert r.destination.kind == "me" and r.destination.is_external is False
    assert r.question == "What is the leave policy?"
    assert r.text == r.question                      # backward-compat alias

def test_explicit_me_long_and_short_equivalent():
    long = parse_command("--me Leave policy")
    short = parse_command("-m Leave policy")
    assert long.destination.kind == short.destination.kind == "me"
    assert long.question == short.question == "Leave policy"

def test_legacy_user_channel_group_still_work():
    assert parse_command("--user abhishek.k Leave policy").destination.usernames == ("abhishek.k",)
    assert parse_command("--channel research Explain").destination.channel_name == "research"
    assert parse_command("--group a,b Leave policy").destination.usernames == ("a", "b")

def test_legacy_modes_still_work():
    assert parse_command("short What is POSH?").mode == "short"
    assert parse_command("detailed leave policy").mode == "detailed"
    assert parse_command("search laptop policy").mode == "search"
    assert parse_command("search laptop policy").question == "laptop policy"


# ── Feature 1 + 6: long flags and short aliases are equivalent ───────────────────

def test_destination_aliases_match_long_forms():
    for long, short, kind in [("--user", "-u", "user"), ("--channel", "-c", "channel"),
                              ("--group", "-g", "group"), ("--me", "-m", "me")]:
        target = "pranay" if kind in ("user", "group") else ("research" if kind == "channel" else "")
        a = parse_command(f"{long} {target} What is LPG crisis?".strip())
        b = parse_command(f"{short} {target} What is LPG crisis?".strip())
        assert a.destination.kind == b.destination.kind == kind
        assert a.question == b.question == "What is LPG crisis?"

def test_mode_aliases_match_long_forms():
    assert parse_command("-s What is POSH?").mode == "short"
    assert parse_command("-d leave policy").mode == "detailed"
    assert parse_command("-f laptop policy").mode == "search"

def test_short_alias_examples_from_spec():
    assert parse_command("-u pranay.kotasthane What is LPG crisis?").destination.usernames == ("pranay.kotasthane",)
    assert parse_command("-c chowk-discussions What is LPG crisis?").destination.channel_name == "chowk-discussions"
    assert parse_command("-g pranay,sowmya What is LPG crisis?").destination.usernames == ("pranay", "sowmya")


# ── Feature 7: order of modifiers does not matter ────────────────────────────────

def test_mixed_ordering_is_normalized():
    variants = [
        "short -u pranay question",
        "-u pranay short question",
        "-s -u pranay question",
        "-u pranay -s question",
    ]
    for v in variants:
        r = parse_command(v)
        assert r.destination.kind == "user"
        assert r.destination.usernames == ("pranay",)
        assert r.mode == "short"
        assert r.question == "question"

def test_mixed_ordering_channel_detailed():
    for v in ["detailed -c chowk question", "-d -c chowk question", "-c chowk -d question"]:
        r = parse_command(v)
        assert r.destination.kind == "channel" and r.destination.channel_name == "chowk"
        assert r.mode == "detailed" and r.question == "question"


# ── Feature 3/4/5: help, examples, empty ─────────────────────────────────────────

def test_help_command_and_alias():
    assert parse_command("help").command == "help"
    assert parse_command("-h").command == "help"

def test_examples_command_and_alias():
    assert parse_command("examples").command == "examples"
    assert parse_command("-e").command == "examples"

def test_empty_is_landing_not_error():
    for txt in ["", "   ", None]:
        r = parse_command(txt)
        assert r.command == "empty"
        assert r.error is None                       # never an error


# ── Visibility + voice ───────────────────────────────────────────────────────────

def test_visibility_public_private():
    assert parse_command("public What is POSH?").visibility == "public"
    assert parse_command("private What is POSH?").visibility == "private"
    assert parse_command("What is POSH?").visibility == "default"

def test_voice_flag_and_alias():
    assert parse_command("voice").voice is True
    assert parse_command("-v").voice is True
    assert parse_command("What is POSH?").voice is False


# ── Feature 2: group routing + validation ────────────────────────────────────────

def test_group_multiple_users_deduped():
    r = parse_command("-g a,b,a,c Leave policy")
    assert r.destination.kind == "group"
    assert r.destination.usernames == ("a", "b", "c")     # deduped, order kept
    assert r.question == "Leave policy"

def test_group_tolerates_spaces_after_commas():
    r = parse_command("--group a, b, c Explain leave policy")
    assert r.destination.usernames == ("a", "b", "c")
    assert r.question == "Explain leave policy"

def test_group_at_prefix_stripped():
    r = parse_command("-g @a,@b Leave policy")
    assert r.destination.usernames == ("a", "b")


# ── Missing arguments / malformed flags → friendly error, default to me ──────────

def test_user_without_username_errors():
    r = parse_command("--user")
    assert r.error is not None and "username" in r.error.lower()

def test_channel_without_name_errors():
    r = parse_command("-c")
    assert r.error is not None and "channel" in r.error.lower()

def test_group_without_users_errors():
    r = parse_command("--group")
    assert r.error is not None


# ── Helpers ──────────────────────────────────────────────────────────────────────

def test_dequote():
    assert dequote('"Leave policy"') == "Leave policy"
    assert dequote("'Leave policy'") == "Leave policy"
    assert dequote("Leave policy") == "Leave policy"

def test_quoted_question_is_unwrapped():
    r = parse_command('-u abhishek.k "What is the leave policy?"')
    assert r.question == "What is the leave policy?"

def test_build_destination_matches_parser():
    assert build_destination("user", "@abhishek.k").usernames == ("abhishek.k",)
    assert build_destination("channel", "~research").channel_name == "research"
    assert build_destination("group", "a, b ,c").usernames == ("a", "b", "c")

def test_extra_whitespace_ignored():
    r = parse_command("   -u   abhishek.k    Leave   policy  ")
    assert r.destination.usernames == ("abhishek.k",)
    assert r.question == "Leave policy"


# ── Feature 8: autocomplete spec includes aliases ────────────────────────────────

def test_autocomplete_spec_has_commands_modes_destinations():
    spec = autocomplete_spec()
    assert {"commands", "modes", "destinations"} <= set(spec)
    dest_aliases = {d["alias"] for d in spec["destinations"]}
    assert {"-u", "-c", "-g", "-m"} <= dest_aliases
    mode_aliases = {m["alias"] for m in spec["modes"]}
    assert {"-s", "-d", "-f"} <= mode_aliases


if __name__ == "__main__":
    mod = sys.modules[__name__]
    passed = 0
    for name in sorted(dir(mod)):
        if name.startswith("test_"):
            getattr(mod, name)()
            passed += 1
    print(f"OK — {passed} tests passed")