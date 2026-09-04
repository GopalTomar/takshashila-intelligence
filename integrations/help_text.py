"""
help_text.py — Presentation for the bot's ``help`` / ``examples`` / landing cards.

Kept separate from ``command_parser`` (which only *parses*) and from
``formatting`` (which renders *answers*). These three functions return finished
Markdown strings the bot posts verbatim, so the copy lives in one place and can
evolve without touching parsing, routing or delivery.

Group examples are shown only when group messaging is enabled, so the help never
advertises a destination the deployment has turned off.
"""

from __future__ import annotations

BRAND = "🏛️ **Takshashila Knowledge Assistant**"


def format_landing() -> str:
    """Friendly response to a bare ``/askkb`` (no question) — never an error."""
    return (
        "🤔 **What would you like to know?**\n\n"
        "Ask me anything from Takshashila's knowledge base — policies, research, "
        "publications and internal docs.\n\n"
        "**Examples**\n"
        "• `/askkb What are the core job expectations for Takshashila staff members?`\n"
        "• `/askkb short Explain the illusion of 3D mapping`\n"
        "• `/askkb detailed Explain India's defence policy`\n"
        "• `/askkb What is the opportunity hidden in the West Asia crisis?`\n\n"
        "Need more commands?\n"
        "➡️ `/askkb help`  ·  see practical examples with `/askkb examples`"
    )


def format_help(enable_group: bool = False, enable_voice: bool = True) -> str:
    """The full, product-style command guide (``/askkb help`` or ``/askkb -h``)."""
    group_cmd = (
        "\n| `--group` | `-g` | `/askkb -g abhishek.k,nithiya <q>` | Send to a group message |"
        if enable_group else ""
    )
    voice_line = "• `/askkb voice` — 🎤 speak your question instead of typing\n" if enable_voice else ""

    return (
        f"{BRAND} — **Command Guide**\n"
        "_Your enterprise assistant for grounded answers from the Takshashila knowledge base._\n\n"
        "———————————————————————————————\n\n"
        "### 💬 The basics\n"
        "Just ask a question. By default the answer is shown **privately to you**.\n"
        "```\n/askkb What is the leave policy?\n```\n\n"
        "### ⚙️ Modes\n"
        "| Mode | Alias | Example | What it does |\n"
        "|---|---|---|---|\n"
        "| `short` | `-s` | `/askkb -s What is POSH?` | A brief, to-the-point answer |\n"
        "| `detailed` | `-d` | `/askkb -d Leave policy` | A fuller answer with more points |\n"
        "| `search` | `-f` | `/askkb -f laptop policy` | List matching documents (no AI answer) |\n\n"
        + ("### 🎤 Voice\n" + voice_line + "\n" if enable_voice else "") +
        "### 📤 Sharing an answer (destinations)\n"
        "Send the answer somewhere specific. Long flag or short alias — both work.\n\n"
        "| Destination | Alias | Example | Where it goes |\n"
        "|---|---|---|---|\n"
        "| `--me` | `-m` | `/askkb -m <q>` | Privately to you (default) |\n"
        "| `--user` | `-u` | `/askkb -u pranay.kotasthane <q>` | A colleague's direct messages |\n"
        "| `--channel` | `-c` | `/askkb -c chowk-discussions <q>` | Into a channel (bot must be a member) |"
        + group_cmd + "\n\n"
        "You can also tap the **👤 Share to User** / **📢 Share to Channel** buttons "
        "under any answer to re-send it without asking again.\n\n"
        "### 🔀 Mix & match — order doesn't matter\n"
        "```\n"
        "/askkb -s -u pranay.kotasthane What is the LPG crisis?\n"
        "/askkb -u pranay.kotasthane -s What is the LPG crisis?\n"
        "/askkb detailed -c research Explain India's Act East policy\n"
        "```\n\n"
        "### 🧭 More\n"
        "• `/askkb examples` (`-e`) — practical, copy-pasteable examples\n"
        "• `/askkb help` (`-h`) — this guide\n\n"
        "———————————————————————————————\n"
        "_Answers are grounded in retrieved documents and cited. If the knowledge base "
        "doesn't have enough evidence, the assistant says so rather than guessing._"
    )


def format_examples(enable_group: bool = False, enable_voice: bool = True) -> str:
    """Practical, sectioned examples (``/askkb examples`` or ``/askkb -e``)."""
    sections = [
        ("📚 General questions", [
            "/askkb What are the core job expectations for Takshashila staff members?",
            "/askkb What is Takshashila's approach to AI use?",
            "/askkb What is the first-week checklist for new staff?",
        ]),
        ("⚡ Quick summary", [
            "/askkb short Explain the illusion of 3D mapping",
            "/askkb -s What is POSH?",
        ]),
        ("📖 Detailed answers", [
            "/askkb detailed Explain India's defence policy",
            "/askkb -d Summarise the National Geospatial Policy 2022",
        ]),
        ("📂 Search (documents only)", [
            "/askkb search laptop policy",
            "/askkb -f meeting rules",
        ]),
        ("🏛 Policy questions", [
            "/askkb What does the leave policy say about sabbaticals?",
            "/askkb What are the rules for sharing documents for review?",
        ]),
        ("🌏 Geopolitics & economics", [
            "/askkb What is the opportunity hidden in the West Asia crisis?",
            "/askkb detailed How should India respond to Chinese chip export controls?",
        ]),
        ("🔬 Research", [
            "/askkb What has Takshashila published on drone warfare?",
            "/askkb search economic security",
        ]),
        ("👤 Share to a user", [
            "/askkb -u pranay.kotasthane What is the LPG crisis?",
            "/askkb --user abhishek.k Leave policy",
        ]),
        ("📢 Share to a channel", [
            "/askkb -c chowk-discussions What is the LPG crisis?",
            "/askkb --channel research Explain India's Act East policy",
        ]),
    ]
    if enable_group:
        sections.append(("👥 Group sharing", [
            "/askkb -g pranay.kotasthane,sowmya What is the LPG crisis?",
            "/askkb --group abhishek.k,nithiya Leave policy",
        ]))
    if enable_voice:
        sections.append(("🎤 Voice", [
            "/askkb voice",
            "/askkb -v",
        ]))

    out = [f"{BRAND} — **Examples**", ""]
    for title, items in sections:
        out.append(f"**{title}**")
        out.extend(f"• `{ex}`" for ex in items)
        out.append("")
    out.append("_Tip: mix a mode and a destination in any order — "
               "`/askkb -s -u pranay.kotasthane <question>`._")
    return "\n".join(out)