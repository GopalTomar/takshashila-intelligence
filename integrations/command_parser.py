"""
command_parser.py — Normalized parsing for the ``/askkb`` slash command.

This module turns the raw slash-command text into **one normalized command
object** describing *what* the user wants and *where* the answer should go. It
answers, in a single pass:

* **command**   — ``ask`` (the default), ``help``, ``examples`` or ``empty``;
* **destination** — ``me`` (default) / ``user`` / ``channel`` / ``group``;
* **mode**      — ``normal`` / ``short`` / ``detailed`` / ``search``;
* **visibility**— ``default`` / ``public`` / ``private``;
* **voice**     — whether the user asked to speak the question;
* **question**  — the remaining text.

Two design guarantees:

1. **Long flags and short aliases are equivalent.** ``--user`` and ``-u`` produce
   identical output; the same for ``--channel``/``-c``, ``--group``/``-g``,
   ``--me``/``-m``, and for the modes/commands (``short``/``-s``,
   ``detailed``/``-d``, ``search``/``-f``, ``voice``/``-v``, ``help``/``-h``,
   ``examples``/``-e``). Everything is normalized through one registry, so there
   is no duplicated parsing logic.
2. **Order of modifiers does not matter.** ``short -u pranay Q``,
   ``-u pranay short Q``, ``-s -u pranay Q`` and ``-u pranay -s Q`` all parse to
   the same destination, mode and question. Modifiers are consumed from the front
   in any order until the first plain question word; the rest is the question.

The parser performs **no** network / Mattermost lookup — resolving a username or
channel name to an id is the delivery layer's job (``response_router`` and the
``destination_handlers``). This keeps parsing pure and unit-testable.

Backward compatibility: :class:`ParseResult` still exposes ``destination``,
``text`` and ``error`` exactly as before (``text`` equals the parsed question), so
older call sites keep working. New call sites use the richer fields.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

# ════════════════════════════════════════════════════════════════════════════════
#  Registries (single source of truth — used by the parser AND autocomplete)
# ════════════════════════════════════════════════════════════════════════════════

# Destination flags → destination kind. Long form + short alias both map here.
DESTINATION_FLAGS: Dict[str, str] = {
    "--me": "me",       "-m": "me",
    "--user": "user",   "-u": "user",
    "--channel": "channel", "-c": "channel",
    "--group": "group", "-g": "group",
}

# Mode keywords → canonical mode. Long form + short alias.
MODE_ALIASES: Dict[str, str] = {
    "short": "short",     "-s": "short",
    "detailed": "detailed", "-d": "detailed",
    "search": "search",   "-f": "search",
    "normal": "normal",
}

# Standalone commands (terminal — they don't take a question).
HELP_TOKENS = {"help", "-h"}
EXAMPLES_TOKENS = {"examples", "-e"}

# Voice + visibility modifiers.
VOICE_TOKENS = {"voice", "-v", "🎤"}
VISIBILITY_TOKENS: Dict[str, str] = {"public": "public", "private": "private"}

# Human-friendly hints per destination flag (surfaced to autocomplete).
_FLAG_HINTS: Dict[str, Tuple[str, str]] = {
    "--me": ("", "send the answer privately to yourself (default)"),
    "--user": ("[username]", "send the answer to another user's direct messages"),
    "--channel": ("[channel]", "post the answer in a channel"),
    "--group": ("[user1,user2,…]", "send the answer to a group direct message"),
}


# ════════════════════════════════════════════════════════════════════════════════
#  Data model
# ════════════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class Destination:
    """Where a generated answer should be delivered."""

    kind: str                       # "me" | "user" | "channel" | "group"
    usernames: Tuple[str, ...] = ()
    channel_name: str = ""
    raw_target: str = ""

    @property
    def is_external(self) -> bool:
        """True when the answer leaves the requester's own DM (user/channel/group)."""
        return self.kind in ("user", "channel", "group")


@dataclass
class ParseResult:
    """The normalized command object produced by :func:`parse_command`."""

    destination: Destination
    text: str                        # remaining text == the question (backward compat)
    error: Optional[str] = None      # user-facing message when parsing failed
    command: str = "ask"             # "ask" | "help" | "examples" | "empty"
    mode: str = "normal"             # "normal" | "short" | "detailed" | "search"
    visibility: str = "default"      # "default" | "public" | "private"
    voice: bool = False
    question: str = ""

    @property
    def is_command(self) -> bool:
        """True for the standalone help/examples/empty landing responses."""
        return self.command in ("help", "examples", "empty")


class CommandParseError(ValueError):
    """Raised internally for malformed flags; surfaced as ``ParseResult.error``."""


# ════════════════════════════════════════════════════════════════════════════════
#  Small text helpers
# ════════════════════════════════════════════════════════════════════════════════

_QUOTE_PAIRS = (('"', '"'), ("'", "'"), ("“", "”"), ("‘", "’"))


def dequote(text: str) -> str:
    """Strip a single pair of matching surrounding quotes, if present."""
    s = (text or "").strip()
    for open_q, close_q in _QUOTE_PAIRS:
        if len(s) >= 2 and s[0] == open_q and s[-1] == close_q:
            return s[1:-1].strip()
    return s


def _split_first_token(text: str) -> Tuple[str, str]:
    """Return ``(first_whitespace_delimited_token, rest)``."""
    parts = text.strip().split(None, 1)
    if not parts:
        return "", ""
    if len(parts) == 1:
        return parts[0], ""
    return parts[0], parts[1]


def _consume_group_target(text: str) -> Tuple[str, str]:
    """
    Pull a ``--group`` comma-list target off the front of ``text``, tolerating
    spaces after commas::

        "a, b, c Explain leave policy"  → ("a, b, c", "Explain leave policy")
        "a,b,c Explain leave policy"    → ("a,b,c",   "Explain leave policy")
    """
    target, rest = _split_first_token(text)
    while target.endswith(",") and rest:
        nxt, rest = _split_first_token(rest)
        if not nxt:
            break
        target = f"{target}{nxt}"
    return target, rest


def _clean_usernames(raw: str) -> List[str]:
    """Split a comma list into de-duplicated, ``@``-stripped, non-empty names."""
    seen, names = set(), []
    for piece in (raw or "").split(","):
        name = piece.strip().lstrip("@").strip()
        if name and name.lower() not in seen:
            seen.add(name.lower())
            names.append(name)
    return names


# ════════════════════════════════════════════════════════════════════════════════
#  Public API
# ════════════════════════════════════════════════════════════════════════════════

def parse_command(text: str) -> ParseResult:
    """
    Parse ``text`` into a normalized :class:`ParseResult`.

    * Empty input → ``command="empty"`` (the caller shows a friendly landing card,
      never an error).
    * A leading ``help`` / ``-h`` or ``examples`` / ``-e`` → the matching command.
    * Otherwise ``command="ask"`` with destination, mode, visibility, voice and
      question all normalized. Long flags and short aliases are equivalent, and
      the order of modifiers does not matter.

    On a malformed destination flag (e.g. ``--user`` with no username) the
    ``destination`` falls back to ``me`` and ``error`` carries a friendly message.
    """
    stripped = (text or "").strip()
    if not stripped:
        return ParseResult(Destination("me"), "", command="empty", question="")

    tokens = stripped.split()
    first = tokens[0].lower()

    # ── Standalone commands ───────────────────────────────────────────────────────
    if first in HELP_TOKENS:
        return ParseResult(Destination("me"), "", command="help")
    if first in EXAMPLES_TOKENS:
        return ParseResult(Destination("me"), "", command="examples")

    # ── Walk tokens: consume leading modifiers (any order), collect the question ──
    dest_kind: Optional[str] = None
    dest_target: str = ""
    mode = "normal"
    visibility = "default"
    voice = False
    error: Optional[str] = None
    q_parts: List[str] = []
    consuming = True

    i = 0
    while i < len(tokens):
        low = tokens[i].lower()

        if consuming and low in DESTINATION_FLAGS:
            kind = DESTINATION_FLAGS[low]
            if kind == "me":
                dest_kind = "me"
                i += 1
                continue
            if kind in ("user", "channel"):
                if i + 1 < len(tokens):
                    dest_kind = kind
                    dest_target = tokens[i + 1]
                    i += 2
                    continue
                # flag with no target → record a friendly error, keep scanning.
                error = (
                    "Please provide a username, e.g. `/askkb -u abhishek.k Leave policy`."
                    if kind == "user" else
                    "Please provide a channel, e.g. `/askkb -c research Explain this`."
                )
                dest_kind = kind
                i += 1
                continue
            if kind == "group":
                remaining = " ".join(tokens[i + 1:])
                target, rest = _consume_group_target(remaining)
                dest_kind = "group"
                dest_target = target
                tokens = rest.split()      # keep scanning the remainder for mode/question
                i = 0
                continue

        if consuming and low in MODE_ALIASES:
            mode = MODE_ALIASES[low]
            i += 1
            continue
        if consuming and low in VISIBILITY_TOKENS:
            visibility = VISIBILITY_TOKENS[low]
            i += 1
            continue
        if consuming and low in VOICE_TOKENS:
            voice = True
            i += 1
            continue

        # First non-modifier token → everything from here is the question.
        consuming = False
        q_parts.append(tokens[i])
        i += 1

    question = dequote(" ".join(q_parts))

    # ── Build + validate the destination ─────────────────────────────────────────
    destination = Destination("me")
    if dest_kind == "user":
        names = _clean_usernames(dest_target)
        if not names and not error:
            error = "Please provide a username, e.g. `/askkb -u abhishek.k Leave policy`."
        destination = Destination("user", usernames=tuple(names[:1]), raw_target=dest_target)
    elif dest_kind == "channel":
        channel = (dest_target or "").strip().lstrip("~").strip()
        if not channel and not error:
            error = "Please provide a channel, e.g. `/askkb -c research Explain this`."
        destination = Destination("channel", channel_name=channel, raw_target=dest_target)
    elif dest_kind == "group":
        names = _clean_usernames(dest_target)
        if not names and not error:
            error = ("Please list at least one user, e.g. "
                     "`/askkb -g abhishek.k,nithiya Leave policy`.")
        destination = Destination("group", usernames=tuple(names), raw_target=dest_target)

    return ParseResult(
        destination=destination,
        text=question,
        error=error,
        command="ask",
        mode=mode,
        visibility=visibility,
        voice=voice,
        question=question,
    )


def build_destination(kind: str, target: str) -> Destination:
    """
    Construct a :class:`Destination` from a ``kind`` + free-text ``target``.

    Used by the Share-button dialog flow, where the user types the target into a
    Mattermost dialog rather than as a slash-command flag. Reuses the same
    normalisation (``@``/``~`` stripping, comma splitting) as the flag parser.
    """
    kind = (kind or "").lower()
    if kind == "user":
        names = _clean_usernames(target)
        return Destination("user", usernames=(names[0],) if names else (), raw_target=target)
    if kind == "channel":
        channel = (target or "").strip().lstrip("~").strip()
        return Destination("channel", channel_name=channel, raw_target=target)
    if kind == "group":
        names = _clean_usernames(target)
        return Destination("group", usernames=tuple(names), raw_target=target)
    return Destination("me")


def autocomplete_spec() -> Dict[str, List[Dict[str, str]]]:
    """
    Machine-readable description of every command, mode and destination flag —
    long form **and** short alias — ready to wire into a Mattermost autocomplete
    definition. Derived entirely from the registries above, so adding a flag or
    alias in one place updates the autocomplete automatically.
    """
    destinations: List[Dict[str, str]] = []
    for flag, kind in DESTINATION_FLAGS.items():
        if flag.startswith("--"):                 # list long form; attach its alias
            alias = next((a for a, k in DESTINATION_FLAGS.items()
                          if k == kind and not a.startswith("--")), "")
            hint, help_text = _FLAG_HINTS.get(flag, ("", ""))
            destinations.append({"flag": flag, "alias": alias, "kind": kind,
                                 "hint": hint, "help": help_text})

    modes = [
        {"keyword": "short", "alias": "-s", "help": "a brief answer"},
        {"keyword": "detailed", "alias": "-d", "help": "a fuller answer"},
        {"keyword": "search", "alias": "-f", "help": "list matching documents (no AI answer)"},
    ]
    commands = [
        {"keyword": "help", "alias": "-h", "help": "show the full command guide"},
        {"keyword": "examples", "alias": "-e", "help": "show practical examples"},
        {"keyword": "voice", "alias": "-v", "help": "ask by speaking instead of typing"},
    ]
    return {"commands": commands, "modes": modes, "destinations": destinations}