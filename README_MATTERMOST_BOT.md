# Takshashila Mattermost RAG Bot

A production-ready Mattermost slash-command bot that answers questions from the
**existing** Takshashila RAG knowledge base — and now delivers those answers
wherever they're needed: your own direct messages, a colleague's DM, or a
channel.

A user in Mattermost types:

```
/askkb What is the leave policy?
```

…and the bot replies **privately** with an answer, a confidence level, timing,
and citations — all produced by the **existing** `src.rag_pipeline.answer()`.
The Streamlit app is untouched and keeps working exactly as before.

The bot has since grown from a personal DM assistant into an **enterprise
knowledge assistant**: the same answer can be routed to another user or a
channel, either from the slash command (`--user`, `--channel`) or by tapping a
**Share** button under any answer — without ever re-running the RAG pipeline.

---

## Table of contents

1. [How it works](#how-it-works)
2. [What's new (enterprise routing)](#whats-new-enterprise-routing)
3. [Files in this integration](#files-in-this-integration)
4. [Command syntax](#command-syntax)
5. [Response format & UX](#response-format--ux)
6. [Answer privacy](#-answer-privacy-no-channel-spam)
7. [Enterprise response routing](#enterprise-response-routing)
8. [Share buttons, PDF & Related Policies](#share-buttons-pdf--related-policies)
9. [Voice input](#-voice-input)
10. [Environment variables](#environment-variables)
11. [Endpoints](#endpoints)
12. [Setup (install → Mattermost → run)](#setup)
13. [Testing everything](#testing-everything)
14. [Production deployment](#production-deployment)
15. [Troubleshooting](#troubleshooting)
16. [Final checklist](#final-checklist)

---

## How it works

```
Mattermost user types  /askkb [--dest] <question>
        │
        ▼
Mattermost POSTs the slash command  ──►  FastAPI  /mattermost/ask
        │
        ▼
FastAPI validates MATTERMOST_SLASH_TOKEN  (403 if invalid)
        │
        ├─ command_parser splits off any --me / --user / --channel / --group flag
        ├─ returns an INSTANT ephemeral ack  ("Searching the knowledge base…")
        │
        ▼ (FastAPI BackgroundTask)
calls existing  src.rag_pipeline.answer()   ← RETRIEVAL (never knows the destination)
        │
        ▼
RAG retrieves evidence (FAISS + BM25) and generates via Groq
        │
        ▼
Bot formats answer + confidence + sources  →  ResponsePayload
        │
        ▼
Default (me):   post privately to the asker's DM     (existing behaviour)
External:       response_router → destination handler → post to user/channel   ← DELIVERY
        │
        ▼
Requester gets a private confirmation ("✅ Sent to @…")
```

**Retrieval and delivery are fully separated.** The retrieval engine returns
`answer / sources / citations / metadata` and knows nothing about where the
answer goes; a separate **response router** decides the destination. The
existing RAG pipeline in `src/` is **reused, not rewritten**.

---

## What's new (enterprise routing)

| Feature | Command / action | Notes |
|---|---|---|
| Private answer to you (default) | `/askkb <question>` or `/askkb --me <question>` | Unchanged from before — DM to the asker. |
| Send to another user | `/askkb --user abhishek.k <question>` | DMs the answer to that user, with an @mention + "Shared by" header. |
| Post to a channel | `/askkb --channel research <question>` | Bot must be a member of the channel. |
| Send to a group DM | `/askkb --group a,b,c <question>` | **Off by default** — enable with `MATTERMOST_ENABLE_GROUP=true`. |
| Share an existing answer | 👤 **Share to User** / 📢 **Share to Channel** buttons | Opens a dialog; **reuses the cached answer** (no second RAG run). |
| Download PDF | ⬇️ **Download PDF** button | Renders the answer to PDF via `pymupdf`. |
| Related Policies | 🔗 **Related Policies** button | Now shows **only the documents the answer was drawn from** (see below). |

**Backward compatibility:** `/askkb <question>` and `/askkb --me <question>`
behave exactly as before (private DM to the asker with all buttons). Nothing in
the default path changed.

---

## Files in this integration

```
integrations/
├── __init__.py
├── mattermost_bot.py              ← FastAPI bot: transport, endpoints, buttons, wiring
├── formatting.py                  ← clean Mattermost message builder (presentation)
├── help_text.py                   ← NEW  help / examples / landing card copy (UI)
├── feedback.py                    ← append-only 👍/👎 feedback log
│
│   ── delivery layer (separate from retrieval) ──
├── command_parser.py              ← NEW  splits --me/--user/--channel/--group off the text
├── mattermost_api.py              ← NEW  low-level Mattermost REST lookups (single source of truth)
├── response_router.py             ← NEW  decides WHERE an answer goes → dispatch
└── destination_handlers/          ← NEW  one small module per target
    ├── __init__.py
    ├── base.py                    ←   DeliveryResult / ResponsePayload / Requester + deliver()
    ├── dm_handler.py              ←   send_to_my_dm     (the requester's own DM)
    ├── user_handler.py            ←   send_to_user_dm   (another user's DM)
    ├── channel_handler.py         ←   send_to_channel   (a channel)
    └── group_handler.py           ←   send_to_group_dm  (a group DM, opt-in)

tests/
└── test_command_parser.py         ← NEW  15 unit tests for the destination parser

README_MATTERMOST_BOT.md           ← this file
.env.example / env.example         ← env templates (copy .env.example to .env)
Dockerfile.mattermost              ← optional containerised deployment
requirements.txt                   ← FastAPI/uvicorn/httpx/python-multipart/pymupdf
```

The RAG pipeline in `src/` is **not modified** — parsing, routing, delivery and
presentation live entirely in `integrations/`.

### Why the delivery layer lives in `integrations/`, not `src/`

`src/` is the retrieval engine, and the whole point of this design is that it
stays ignorant of delivery. Putting routing code in `src/` would contradict that
separation (and risk an import cycle), so the parser, router and handlers sit
next to the Mattermost transport they use. The module names still mirror a clean
`command_parser → response_router → destination_handlers/*` layout.

---

## Command syntax

Every command has a **long form and a short alias** — both produce identical
results, and **the order of modifiers doesn't matter**.

```
/askkb <question>                              default → private DM to you
/askkb -m <question>   (--me)                  explicit → private DM to you
/askkb -u pranay.kotasthane <question>  (--user)    → that user's direct messages
/askkb -c chowk-discussions <question>  (--channel) → the channel (bot must be a member)
/askkb -g pranay,sowmya <question>      (--group)   → a group message  (only if enabled)
```

Modes and the standalone commands have aliases too:

```
/askkb -s What is POSH?     (short)      Quick Summary only
/askkb -d leave policy      (detailed)   answer + detailed points
/askkb -f laptop policy     (search)     lists matching documents, NO AI answer
/askkb -v                   (voice)      🎤 speak instead of typing
/askkb -h                   (help)       the full command guide
/askkb -e                   (examples)   practical, copy-pasteable examples
```

### Alias reference

| Long | Alias | Meaning |
|---|---|---|
| `--me` | `-m` | send privately to you (default) |
| `--user <name>` | `-u` | send to a user's DMs |
| `--channel <name>` | `-c` | post in a channel |
| `--group <a,b,…>` | `-g` | send to a group message (opt-in) |
| `short` | `-s` | brief answer |
| `detailed` | `-d` | fuller answer |
| `search` | `-f` | documents only, no AI answer |
| `voice` | `-v` | ask by speaking |
| `help` | `-h` | command guide |
| `examples` | `-e` | example gallery |

### Order-independent — all of these are equivalent

```
/askkb -s -u pranay.kotasthane What is the LPG crisis?
/askkb -u pranay.kotasthane -s What is the LPG crisis?
/askkb short -u pranay.kotasthane What is the LPG crisis?
/askkb detailed -c research Explain India's Act East policy
```

The parser consumes any run of leading modifiers (destination, mode, `public`/
`private`, `voice`) in any order, then treats the rest as the question.

### `help`, `examples`, and the empty command

* `/askkb help` (`-h`) → a formatted command guide.
* `/askkb examples` (`-e`) → a sectioned gallery of practical examples.
* `/askkb` with **no question** → a friendly landing card with example commands —
  **never an error**.

The `help` and `examples` cards are private (only you see them) and carry a
**🗑️ Dismiss** button so you can clear them once you're done reading.

The parser is forgiving: it tolerates extra whitespace, an optional `@`/`~`
prefix (`-u @abhishek.k`, `-c ~research`), spaces after commas in a group list
(`-g a, b, c`), and questions wrapped in quotes. A missing target returns a
friendly message, e.g. `Please provide a username, e.g. /askkb -u abhishek.k
Leave policy`.

> **Autocomplete-ready.** Every flag, alias, mode and command lives in one
> registry in `command_parser` and is exposed via
> `command_parser.autocomplete_spec()` (which returns `commands`, `modes` and
> `destinations`, each with its alias), so a Mattermost autocomplete definition
> can enumerate them without any code changes. Nothing is hard-coded per
> deployment.

---

## Backward compatibility

Every pre-existing command still works **exactly** as before — the aliases and
order-independence are purely additive:

```
/askkb <question>            /askkb short <question>       /askkb detailed <question>
/askkb search <keywords>     /askkb voice                  /askkb public <question>
/askkb --user <u> <q>        /askkb --channel <c> <q>      /askkb --group <u,u> <q>
```

The default (no flag) is still a private answer to you, with all buttons. The
only behavioural changes are improvements: a bare `/askkb` now shows a landing
card instead of a usage error, and confirmations are crisper (see below).

---

## Response format & UX

Every answer is rendered from one clean template (no duplicated headings, no raw
chunk/vector metadata, no debug dump):

```
🏛️ Takshashila Knowledge Assistant

### 📌 Question
<the user's question>

### 🎯 Quick Summary
<concise answer, with inline [1] citations>

### 📖 Detailed Answer
* key point [1]
* key point [2]

### 📚 Sources
**1.** 📄 Key Policies
🏷️ Decision · [🔗 Open Document](https://…)

### 📊 Answer Metadata
⚡ Response Time: 1.5s
🎯 Confidence: 🟢 High
📚 Sources Used: 3
```

Interactive buttons (rendered when `MATTERMOST_BOT_PUBLIC_URL` is set) appear
beneath the answer in labelled groups:

* **📌 Suggested Follow-Ups** — `🔗 Related Policies` (see the fix below).
* **📤 Export** — `📄 Export Markdown` uploads a `.md`; `⬇️ Download PDF`
  uploads a `.pdf` (when PDF export is enabled). Both reuse the cached answer.
* **📨 Share this answer** — `👤 Share to User` / `📢 Share to Channel`
  (and `👥 Share to Group` only when group is enabled). Each opens a dialog and
  re-delivers the existing answer — no second RAG run.
* **Was this helpful?** — `👍 Helpful` / `👎 Not Helpful`. A click is logged to
  `data/logs/mattermost_feedback.jsonl` and swaps the buttons in-place for a
  confirmation card; the clicker also gets a short private popup.
* **🗑️ Manage this response** — `🗑️ Delete this response` removes just that
  post; `🧹 Delete all` removes every bot response in the channel after a
  Yes/Cancel confirmation. The bot only ever deletes its own posts.

Behaviour notes:

* **Confidence** shows as 🟢 High / 🟡 Medium / 🔴 Low. A no-evidence reply
  shows no confidence and no sources (never invents any).
* **Top-3 sources only**, de-duplicated by title + URL. Inline `[Source N]`
  citations from the model are renumbered to match the displayed cards.
* The bot **caches each answer by post id** (question, mode, message **and its
  grounding sources**) so export / share / Related Policies are instant; on a
  cache miss it regenerates from the question carried in the button context.
* Interactive buttons and file export require `MATTERMOST_BOT_TOKEN`; without it
  the bot still answers via the `response_url` fallback (plain text only).

---

## 🔒 Answer privacy (no channel spam)

By default the bot answers **privately**: each person's answer is shown only to
them, so a shared channel never fills up with everyone's questions.

Controls:

* `/askkb <question>` → private to you (default).
* `/askkb public <question>` → deliberately share that answer with the channel.
* `/askkb private <question>` → keep an answer private even when the default is public.
* `MATTERMOST_ANSWER_VISIBILITY=public` flips the default for a channel.

How a private answer is delivered (`MATTERMOST_PRIVATE_DELIVERY`):

* `dm` (default) — a **direct message** from the bot, so **all** buttons work
  (feedback, export, share, delete). Requires the bot token.
* `ephemeral` — shown only to the requester inside the channel. Mattermost
  ephemeral posts **cannot run interactive buttons or carry file downloads**, so
  those features are unavailable in this mode (which is why `dm` is the default).

---

## Enterprise response routing

The routing layer is what turns a single answer into something you can send
anywhere. It is deliberately split into three responsibilities:

```
command_parser        →  WHAT destination did the user ask for?
                         (Destination: me | user | channel | group)

response_router       →  dispatch that Destination to the right handler

destination_handlers  →  HOW to reach each target (resolve id, then post)
     dm_handler.send_to_my_dm
     user_handler.send_to_user_dm
     channel_handler.send_to_channel
     group_handler.send_to_group_dm
```

Every handler returns a uniform `DeliveryResult(ok, confirmation, error, …)`, so
the caller can show a consistent confirmation or error to the requester without
knowing which target was used. The handlers **reuse** the bot's existing
`post_to_channel` (and answer cache), and the low-level Mattermost lookups live
in one place (`mattermost_api.py`) — no duplicated API calls.

### What each destination does

* **`--me` / default** → opens (or reuses) the bot↔you DM and posts there.
  This is the untouched original path.
* **`--user <username>`** → resolves the username, opens the bot↔user DM, posts
  the answer with an attribution header and an **@mention** so they're notified,
  then confirms `✅ Sent the answer to @username.`
* **`--channel <name>`** → resolves the channel on your team, verifies the bot
  is a member, honours the channel allowlist, posts the answer with a header,
  then confirms `✅ Posted the answer in ~channel.`
* **`--group <a,b,c>`** → resolves every user, opens/reuses a group DM
  (bot + users), posts the answer, then confirms. **Opt-in** via
  `MATTERMOST_ENABLE_GROUP=true`.

### Shared-answer header

When an answer is sent somewhere other than your own DM, it's prefixed with a
compact attribution banner (and, for a user, an @mention so they're notified):

```
@abhishek.k
> 📣 Shared via Takshashila Knowledge Assistant
> Shared by @gopal.tomar · Original question: _Leave policy_

🏛️ Takshashila Knowledge Assistant
### 📌 Question
…
```

### Permission checks & graceful errors

Nothing is sent blindly. Each destination validates first and fails gracefully
with a message only the requester sees:

| Situation | Message |
|---|---|
| Unknown username | `User "abc" not found.` |
| Unknown channel | `Channel "xyz" not found.` |
| Bot not in channel | `Bot is not a member of "xyz". Add the bot to that channel and try again.` |
| Group member missing | `Unable to create group DM because one or more users do not exist: "abc".` |
| Group disabled | `Group messages aren't enabled here. …use --user or --channel instead.` |
| No target given | `Please provide a username, e.g. /askkb --user abhishek.k Leave policy.` |

> **A note on channel permissions.** The bot posts as *itself*, so it enforces
> bot-membership plus your channel allowlist (`MATTERMOST_ALLOWED_CHANNEL_IDS`).
> Per-user posting rights would require the requester's own token, which the bot
> doesn't hold.

---

## Share buttons, PDF & Related Policies

### Share buttons (reuse the answer — no second RAG run)

Under each answer, **👤 Share to User** and **📢 Share to Channel** open a small
Mattermost **dialog** where you type the target. On submit, Mattermost POSTs to
`/mattermost/dialog`; the bot pulls the **already-generated answer from its
cache** and hands it to the same response router. If the target can't be
resolved, the error is shown **inline on the dialog field** so you can correct it
without losing your place. A private ephemeral confirmation is posted on success.

Requirements: `MATTERMOST_BOT_PUBLIC_URL` (the dialog callback target) and
`MATTERMOST_BOT_TOKEN`. The 👥 **Share to Group** button appears only when
`MATTERMOST_ENABLE_GROUP=true`.

### ⬇️ Download PDF

Renders the current answer to a clean, multi-page PDF using **pymupdf** (already
a project dependency) and uploads it to the conversation. It reuses the cached
answer — no second RAG run. Toggle with `MATTERMOST_ENABLE_PDF_EXPORT`
(disabled automatically if `pymupdf` isn't importable).

### 🔗 Related Policies — now grounded, never a broad search

**Previous behaviour (removed):** the button ran a fresh keyword/semantic search
over the whole knowledge base, which surfaced documents that merely *mentioned*
a word in the question — so "Leave policy" could list unrelated papers on defence
or supply chains. A relevance-score floor was tried and didn't help, because
dense embeddings score many same-domain documents above any reasonable threshold.

**Current behaviour:** Related Policies shows **only the exact documents this
answer was drawn from** — the grounding sources the citation-verifier already
kept (the same ones in the answer's **Sources** section). There is **no fresh
retrieval**, so an irrelevant document cannot appear. Those grounding sources are
cached with each answer; on a cache miss the answer is regenerated and *its own*
grounding sources are used — still never a broad search.

```
🔗 Related Policies

The document(s) this answer was drawn from:

**1.** 📄 Sowmya Prabhakar – Takshashila Institution  🏷️ article
[🔗 Open Document](…)

**2.** 📄 Key Policies — POSH, Leave, Laptop, Confidentiality, Skill Development  🏷️ decisions
[🔗 Open Document](…)

---
_These are the exact source documents this answer is based on — not a broader search._
```

---

## 🎤 Voice input

`/askkb voice` returns a private link to a recorder page served at `/voice`. The
user allows mic access, speaks, and the question is transcribed **in the
browser** (Web Speech API — no audio uploaded, no extra keys). Pressing **Send to
Knowledge Base** posts the transcribed question and answer (tagged
`🎤 Asked by voice`). Links are HMAC-signed, expire after
`MATTERMOST_VOICE_TTL_SECONDS` (default 30 min), and are bound to the channel.
Works in Chrome, Edge and Brave. Voice is recognised only for the default
self-delivery (not combined with an external destination flag).

---

## Environment variables

Copy the template and fill in the values:

```bash
cp .env.example .env            # Linux/macOS
Copy-Item .env.example .env      # Windows PowerShell
```

**Required** (unchanged from before):

```
GROQ_API_KEY=...                       # your existing Groq key
MATTERMOST_URL=https://matter.takshashila.org.in
MATTERMOST_BOT_TOKEN=...               # Bot Account access token
MATTERMOST_SLASH_TOKEN=...             # Slash Command token
```

**Routing & UX — all optional, all have defaults:**

| Variable | Default | Purpose |
|---|---|---|
| `MATTERMOST_ENABLE_ROUTING` | `true` | Enables `--user` / `--channel` (and `--group` when that's on). |
| `MATTERMOST_ENABLE_GROUP` | `true` | Enables the `--group` / `-g` command **and** the 👥 Share to Group button. Set `false` to hide it. |
| `MATTERMOST_ENABLE_SHARE_BUTTONS` | `true` | Share buttons under answers. Needs `MATTERMOST_BOT_PUBLIC_URL`. |
| `MATTERMOST_ENABLE_PDF_EXPORT` | `true` | ⬇️ Download PDF button. Needs `pymupdf` + bot token. |
| `MATTERMOST_BOT_PUBLIC_URL` | — | Public URL of this bot; enables all interactive buttons + dialogs + voice. |
| `MATTERMOST_ANSWER_VISIBILITY` | `private` | `private` (default) or `public`. |
| `MATTERMOST_PRIVATE_DELIVERY` | `dm` | `dm` (full buttons) or `ephemeral`. |
| `MATTERMOST_ALLOWED_TEAM_IDS` | — | Comma-separated allowlist (empty = all). |
| `MATTERMOST_ALLOWED_CHANNEL_IDS` | — | Comma-separated allowlist (also applied to `--channel`). |
| `MATTERMOST_RAG_TOP_K` | `5` | Retrieval depth for the bot. |
| `MATTERMOST_RAG_TEMPERATURE` | `0.1` | Generation temperature. |
| `MATTERMOST_MAX_MESSAGE_CHARS` | `12000` | Truncation guard. |
| `MATTERMOST_WARM_RAG_ON_STARTUP` | `true` | Warm FAISS + embeddings + BM25 at boot. |

> **No new *required* variables.** Routing reuses your existing
> `MATTERMOST_BOT_TOKEN`; the team id needed for channel lookups comes from the
> slash-command payload. If your `.env` is untouched, group stays hidden and
> Related Policies shows only grounding sources — exactly the intended defaults.
> You'd only edit `.env` to turn something *off* or to re-enable group.

---

## Endpoints

| Method | Path | Purpose |
|---|---|---|
| `GET` | `/health` | Liveness check. |
| `POST` | `/mattermost/ask` | The `/askkb` slash command (parses destination, schedules RAG). |
| `POST` | `/mattermost/action` | Interactive button clicks (follow-ups, export, share, feedback, delete). |
| `POST` | `/mattermost/dialog` | **NEW** — Share-dialog submissions (re-delivers the cached answer). |
| `POST` | `/mattermost/feedback` | Legacy feedback route (older posts). |
| `GET` | `/voice` | The in-browser voice recorder page. |
| `POST` | `/mattermost/voice-ask` | Accepts a transcribed voice question. |

---

## Setup

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

Adds `fastapi`, `uvicorn[standard]`, `httpx`, `python-multipart`; `pymupdf` and
`python-dotenv` were already present.

### 2. Configure environment

See [Environment variables](#environment-variables) above.

### 3. Run the bot locally

```bash
uvicorn integrations.mattermost_bot:app --host 0.0.0.0 --port 8000
```

Verify health:

```bash
curl http://localhost:8000/health
# {"status":"ok","service":"takshashila-mattermost-rag-bot"}
```

On startup the log prints the active configuration, e.g.:

```
routing enabled = True (--user/--channel); group destinations = False;
share buttons = True; pdf export = True.
```

### 4. Create a Mattermost Bot Account

1. **System Console → Integrations → Bot Accounts** → ensure creation is enabled.
2. **Add Bot Account** (e.g. `takshashila-kb-bot`, role Member), **copy the token**.
3. Put it in `.env` as `MATTERMOST_BOT_TOKEN=...`.
4. **Add the bot to any channel** you'll post to (`--channel`, public answers).
   The bot can only post to channels it belongs to.

### 5. Create the Slash Command

1. **Main Menu → Integrations → Slash Commands → Add Slash Command.**
2. Trigger word `askkb`, Request URL `https://your-bot-server.com/mattermost/ask`,
   method `POST`, autocomplete on, hint `[--user|--channel] [question]`.
3. **Save**, copy the token into `.env` as `MATTERMOST_SLASH_TOKEN=...`.
4. Restart the bot.

> ⚠️ Don't use `localhost` as the Request URL unless Mattermost and the bot run
> on the same machine — the Mattermost **server** makes the request. Use the
> bot's LAN IP or a public HTTPS domain. The same public URL goes in
> `MATTERMOST_BOT_PUBLIC_URL` so buttons, dialogs and voice work.

---

## Testing everything

Run the offline parser tests (no server needed):

```bash
python tests/test_command_parser.py       # → OK — 15 tests passed
```

Then, inside Mattermost, work through these:

**Backward compatibility**

```
/askkb What is the leave policy?     → private DM to you, all buttons
/askkb --me Leave policy             → same
/askkb short / detailed / search / public / voice   → all unchanged
```

**New destinations** (use real usernames/channels; add the bot to the channel first)

```
/askkb --user abhishek.k Leave policy          → DM to @abhishek.k + "✅ Sent…"
/askkb --channel research Explain the rules     → posts in ~research + "✅ Posted…"
/askkb --group a,b,c Leave policy               → only if MATTERMOST_ENABLE_GROUP=true
```

**Graceful errors** (each returns a private message, never a crash)

```
/askkb --user ghostuser Hi        → User "ghostuser" not found.
/askkb --channel nope Hi          → Channel "nope" not found.
/askkb --channel <not-joined> Hi  → Bot is not a member of "…".
/askkb --user                     → Please provide a username, e.g. …
```

**Buttons** (need `MATTERMOST_BOT_PUBLIC_URL`)

```
👤 Share to User / 📢 Share to Channel → dialog opens → deliver cached answer
⬇️ Download PDF → a .pdf file appears     📄 Export Markdown → a .md file
👍 / 👎 → feedback card swaps in
🔗 Related Policies → shows ONLY the answer's source documents (not a search)
```

---

## Production deployment

- Run the bot on an internal VM the Mattermost server can reach.
- Put **HTTPS** in front via Nginx (Mattermost prefers HTTPS).
- Keep `.env` secure (chmod 600; `.gitignore` already blocks it).
- Run it as a long-lived service via **systemd** or **Docker**.
- Ensure `data/index/faiss.index` (+ metadata) exists and `GROQ_API_KEY` is set.

### systemd (`/etc/systemd/system/takshashila-mm-bot.service`)

```ini
[Unit]
Description=Takshashila Mattermost RAG Bot
After=network.target

[Service]
WorkingDirectory=/opt/takshashila-rag
EnvironmentFile=/opt/takshashila-rag/.env
ExecStart=/opt/takshashila-rag/.venv/bin/uvicorn integrations.mattermost_bot:app --host 0.0.0.0 --port 8000
Restart=on-failure
User=takshashila

[Install]
WantedBy=multi-user.target
```

```bash
sudo systemctl daemon-reload
sudo systemctl enable --now takshashila-mm-bot
sudo journalctl -u takshashila-mm-bot -f
```

### Nginx reverse proxy

```nginx
location /mattermost/ {
    proxy_pass http://127.0.0.1:8000;
    proxy_set_header Host $host;
    proxy_set_header X-Real-IP $remote_addr;
    proxy_read_timeout 60s;
}
location = /voice { proxy_pass http://127.0.0.1:8000; }
```

### Docker

```bash
docker build -f Dockerfile.mattermost -t takshashila-mm-bot .
docker run -d --name takshashila-mm-bot \
  -p 8000:8000 --env-file .env \
  -v "$(pwd)/data:/app/data" \
  takshashila-mm-bot
```

Or run this alongside the dashboard, sharing the same knowledge base, via
`docker compose up -d --build` (see `docker-compose.yml` at the repo root —
both services mount the same `./data` volume, so there is one knowledge base,
not two).

---

## Troubleshooting

**Invalid token (HTTP 403)** — the request `token` doesn't match
`MATTERMOST_SLASH_TOKEN`. Copy the slash token into `.env` and restart.

**Ack appears but no answer** — 1) is `MATTERMOST_BOT_TOKEN` set/valid?
2) is the **bot a member of the channel**? 3) check
`data/logs/mattermost_bot.log` for `Bot-token post failed`. 4) the `response_url`
fallback expires after minutes; the bot-token path is robust.

**`--channel` says the bot isn't a member** — add the bot account to that channel
(**Add Members**), then retry. The bot can only post where it belongs.

**`--user` / `--group` says a user isn't found** — the username must match the
Mattermost `@username` exactly (the `@` is optional). Deactivated users won't
resolve.

**`--group` / `-g` says group isn't enabled** — group is **on by default** now;
this only appears if `MATTERMOST_ENABLE_GROUP=false` is set. Remove it (or set
`true`) and restart.

**Group message is confirmed but recipients don't see it** — the answer is posted
into the group channel, but a *programmatically-created* group DM can stay
collapsed in a member's sidebar until it's surfaced. The bot sets each member's
`group_channel_show` preference to make it appear, which requires the bot account
to have the **`edit_other_users`** permission. If the bot lacks it, the log shows
`Could not set group_channel_show … (HTTP 403)`; grant that permission to the
bot's role (System Console → User Management → Permissions, or the bot's assigned
scheme) and the group DM will appear reliably. The message is delivered either
way — this only affects sidebar visibility. The handler now verifies channel
membership and validates the post before confirming, so a genuine failure returns
a clear error instead of a false "Shared with…".

**Share buttons / PDF don't appear** — set `MATTERMOST_BOT_PUBLIC_URL` (buttons
+ dialogs) and ensure the bot token is set. PDF also needs `pymupdf` importable.
The startup log line shows `share buttons = …` and `pdf export = …`.

**Related Policies shows nothing** — that answer cited no documents (e.g. an
"insufficient evidence" reply), so there are no source references to show. This
is expected and honest.

**FAISS index missing / `GROQ_API_KEY` missing** — build the KB first and set the
key (same as the Streamlit app). The user sees only a generic error; details stay
in the log.

**`localhost` not reachable** — the Mattermost **server** calls your Request URL,
not your browser. Use the bot server's LAN IP or a public HTTPS domain.

**`ModuleNotFoundError: No module named 'src'`** — run from the **project root**
and launch as a module path (`uvicorn integrations.mattermost_bot:app …`).

---

## Final checklist

```
[ ] RAG works in Streamlit
[ ] FAISS index exists in data/index/
[ ] GROQ_API_KEY is set
[ ] MATTERMOST_URL / MATTERMOST_BOT_TOKEN / MATTERMOST_SLASH_TOKEN set
[ ] Bot account added to target channels (for --channel / public answers)
[ ] MATTERMOST_BOT_PUBLIC_URL set (for buttons, dialogs, voice)
[ ] Slash command points to /mattermost/ask
[ ] /health works; /askkb test works
[ ] tests/test_command_parser.py passes (15 tests)
[ ] --user, --channel routing + confirmations work
[ ] Related Policies shows only the answer's grounding sources
```



