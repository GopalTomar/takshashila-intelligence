# Takshashila Intelligence

A production, end-to-end **retrieval-augmented generation (RAG)** assistant that
answers questions about Takshashila **only** from indexed Takshashila sources —
the public website, the authenticated Commit Knowledge Base, and the Staff
Handbook — with **inline citations** and an honest "insufficient evidence" reply
when the knowledge base does not support an answer. It never invents facts,
names, dates, links, or citations.

The **same** RAG engine serves both interfaces:

- a **static dashboard** (GitHub Pages) at `frontend/`, and
- a **Mattermost bot** (`/askkb`)

so the two can never diverge in their answers.

---

## 1. Real architecture (why the pieces are where they are)

```
                       ┌─────────────────────────────────────────────┐
                       │                GitHub repo                   │
                       │  GopalTomar/takshashila-intelligence         │
                       └───────────────┬─────────────────────────────┘
                                       │
        ┌──────────────────────────────┼───────────────────────────────┐
        │ GitHub Actions (CI/CD + scheduled ingestion — never your laptop)│
        │  • ci.yml          lint/compile + offline tests + secret scan   │
        │  • kb-pipeline.yml crawl → build → validate → publish artifact   │
        │  • kb-rollback.yml restore a previous good KB version            │
        │  • deploy-pages.yml publish the static dashboard                 │
        └───────────────┬───────────────────────────────┬────────────────┘
                        │ publishes kb-latest.tar.gz     │ deploys frontend/
                        ▼ (GitHub Release asset)          ▼
        ┌───────────────────────────────┐     ┌─────────────────────────────┐
        │  Always-on backend (Render)   │     │  GitHub Pages (static)      │
        │  FastAPI — one canonical RAG  │◀────│  dashboard calls /api/query │
        │  service, HTTPS, health checks│ HTTPS│  (browser → backend, CORS)  │
        │  fetches kb-latest on boot +  │     └─────────────────────────────┘
        │  on a timer → auto-reload     │
        └──────────────┬────────────────┘
                       │ same /api/query + same src.rag_pipeline.answer()
                       ▼
             ┌──────────────────────┐
             │  Mattermost /askkb   │  →  private DM / channel answers + citations
             └──────────────────────┘
```

**Why GitHub Pages does not host the backend.** GitHub Pages is *static* hosting
— it serves HTML/CSS/JS only and cannot run Python, FastAPI, or Streamlit. So the
browser dashboard is static and calls a **separate always-on backend** over HTTPS.
The backend is the one place the RAG engine, FAISS index, and the LLM call live.

**Zero local dependency.** Nothing above needs your laptop. GitHub Actions builds
the KB on a schedule; Render runs the backend 24/7 with a stable HTTPS URL,
health checks and auto-restart; the dashboard is on GitHub Pages; Mattermost
calls the backend directly. Your machine is only for development and admin.

### Data flow

```
SOURCE (website / Commit KB / Staff Handbook)
  → CRAWLER (scripts/crawl_engine.py: sitemap + BFS, retries, backoff, rate-limit)
  → DOCUMENT NORMALIZATION + de-dup (canonical URL)
  → CHUNKING (src/chunker.py, 1200 chars / 150 overlap)
  → EMBEDDINGS (BAAI/bge-small-en-v1.5) ─┬→ FAISS vector index
                                          └→ BM25 lexical index
  → VALIDATION (scripts/validate_kb.py)
  → ATOMIC PUBLISH + VERSION MANIFEST (src/kb_version.py)
  → PRODUCTION RAG SERVICE (src/rag_service.py)
       ├→ GitHub Pages dashboard (/api/query)
       └→ Mattermost bot (/askkb → /api/query engine)
```

---

## 2. The one canonical RAG engine

There is a **single** retrieval + answer pipeline. Interfaces are thin adapters:

| Layer | File | Role |
|---|---|---|
| Canonical service | `src/rag_service.py` | request id, version-aware load, structured logging, one response contract |
| Engine | `src/rag_pipeline.py` | prompt, evidence gate, context build, grounding, citation verify |
| Retrieval | `src/retriever.py` | hybrid FAISS + BM25 (RRF), source-priority boost, dedup |
| Vector store | `src/vector_store.py` | FAISS build/search, **version-aware auto-reload**, atomic persist |
| Versioning | `src/kb_version.py` | manifest, fingerprint, atomic publish, snapshots, rollback |
| Config | `src/config.py` | **single source of truth** for model, thresholds, chunking, sources |

The dashboard (`app.py`), the Mattermost bot (`integrations/mattermost_bot.py`)
and the HTTP API (`POST /api/query`) all call the same engine — there is no
second retriever and no per-interface RAG logic.

### Frontend/backend contract (`POST /api/query`)

```jsonc
// request
{ "query": "What is the ATP podcast?", "interface": "dashboard", "conversation_id": null, "length": "normal" }
// response
{
  "request_id": "…", "answer": "…", "grounded": true, "confidence": "HIGH",
  "sources": [{ "title": "…", "url": "https://takshashila.org.in/…", "source_type": "website", "snippet": "…" }],
  "kb_version": "20260904T…-abcd1234", "model": "openai/gpt-oss-120b",
  "latency_ms": 812, "retrieval_ms": 40, "generation_ms": 760
}
```

Operational endpoints (no secrets exposed): `GET /health` (liveness),
`GET /ready` (503 until index + LLM configured), `GET /rag/status` (KB version,
counts, models, last build).

---

## 3. Model configuration (single, authoritative)

```python
AVAILABLE_MODELS = ["openai/gpt-oss-120b"]   # src/config.py
GROQ_MODEL       = "openai/gpt-oss-120b"
```

There is exactly one supported model, shared by dashboard, engine, bot, API and
tests. Requesting any other model raises a clear operational error
(`ModelUnavailableError`) — there is **no silent fallback**.

---

## 4. Knowledge-base versioning, atomic publish & auto-reload (the stale-index fix)

The original design loaded the index once per process and served it forever, so
a KB rebuilt elsewhere left the dashboard and bot stuck on a stale index until a
manual restart. That is fixed:

- **Atomic publish** — new index files are written to temp files and
  `os.replace`d into place; the version manifest (`data/index/current.json`) is
  written **last**. No consumer ever reads a half-built index; a crashed build
  never replaces a good one.
- **Version fingerprint** — every consumer cheaply reads the manifest version on
  each request and reloads the heavy index **only when it changes**
  (`vector_store.load_index` is version-aware). Because every process reads the
  same manifest, a rebuild is picked up **everywhere, with no restart**.
- **Snapshots + rollback** — each published version is snapshotted under
  `data/index/versions/<version>/`; `kb_version.rollback()` restores a previous
  good version. A failed validation auto-rolls-back (see `update_knowledge_base.py`).

Verified by `tests/test_kb_version_reload.py` (stale-index auto-reload, no
force, no restart; plus rollback).

---

## 5. Retrieval & the ATP fix (hybrid evidence, strict grounding)

Signals are kept **separate** on every chunk — dense cosine `score`, normalized
`lexical_score` (BM25), fused `rrf_score`. A BM25-only hit is **no longer**
assigned a cosine score of 0. A chunk clears the evidence gate when **any** real
signal supports it:

1. dense cosine ≥ `MIN_SCORE_THRESHOLD`, or
2. an **exact salient-token match** (`lexical_verified` — a chunk that contains
   every non-stopword token of a short lookup query, e.g. "ATP", "All Things
   Policy", "GCPP"), or
3. a verified author-metadata match (authorship queries).

This rescues acronym/exact-match questions **without** lowering the dense
threshold (which would invite hallucination): an unrelated query still finds no
chunk covering all its salient tokens and is correctly refused. Verified by
`tests/test_bm25_evidence.py`.

Grounding stays strict: the LLM answers only from retrieved context, citations
are verified against that context after generation, and ungrounded answers are
replaced with an honest insufficient-evidence reply
(`tests/test_golden_queries.py`, `tests/test_citation_integrity.py`).

---

## 6. Deployment

### 6a. Backend (always-on RAG API) — Render

`render.yaml` is a ready blueprint (Docker, HTTPS, health check `/health`,
auto-deploy from GitHub, persistent disk for the index, restart on crash).

**One-time manual steps (require a Render account — cannot be done for you):**

1. Push this repo to GitHub (done).
2. Render → **New → Blueprint** → connect this repo → **Apply**.
3. In the service **Environment** tab set the secrets marked `sync: false`:
   `GROQ_API_KEY`, `API_ALLOWED_ORIGINS` (your Pages origin), and the
   `MATTERMOST_*` / `COMMIT_KB_*` values you use.
4. (Optional, closes the auto-update loop) set `KB_ARTIFACT_URL` to
   `https://github.com/GopalTomar/takshashila-intelligence/releases/latest/download/kb-latest.tar.gz`
   and `KB_REFRESH_INTERVAL_MIN=30`.
5. Verify: `https://<service>.onrender.com/health` → 200, `/ready` → 200,
   `/rag/status` shows the KB version.

Any Docker-capable platform (Railway, Fly.io, Cloud Run) works too — the image
honors `$PORT`. **Do not** use ngrok, localhost, or a dev tunnel in production.

### 6b. Dashboard — GitHub Pages

`.github/workflows/deploy-pages.yml` publishes `frontend/` to Pages.

**One-time:** repo **Settings → Pages → Source = GitHub Actions**. Then edit the
`DEFAULT_API` constant in `frontend/index.html` (or append `?api=<backend-url>`)
to point at your backend, and set `API_ALLOWED_ORIGINS` on the backend to the
Pages origin (e.g. `https://gopaltomar.github.io`).

### 6c. Local / self-hosted — Docker Compose

```bash
cp .env.example .env      # fill in real values
docker compose up -d --build
# dashboard :8501 · backend/API :8000 · scheduler runs the weekly refresh
```

---

## 7. Automated KB refresh (crawler schedule)

`.github/workflows/kb-pipeline.yml`:

| Trigger | Schedule (IST) | Cron (UTC) | Mode |
|---|---|---|---|
| Daily incremental | 02:00 | `30 20 * * *` | incremental crawl + reindex changed |
| Weekly reconciliation | Sun 03:00 | `30 21 * * 6` | full re-crawl + rebuild |
| Manual | on demand | `workflow_dispatch` | incremental or full |

Each run crawls → merges the delta → rebuilds (embedding only changed chunks) →
**validates** → packages `kb-latest.tar.gz` → publishes it as the `kb-latest`
release asset. The backend fetches that asset (boot + timer) and auto-reloads.

**Failure safety (a failed crawl never destroys the KB):** Commit KB is skipped
if credentials are absent; a delta that would empty the corpus is refused; a
build that fails validation is auto-rolled-back to the previous good version;
network errors keep the existing index intact.

---

## 8. Mattermost setup (`/askkb`)

1. Create a **bot account** (or personal access token) → `MATTERMOST_BOT_TOKEN`.
2. **Integrations → Slash Commands → Add**:
   - Command trigger: `askkb`
   - Request URL: `https://<backend>/mattermost/ask`
   - Request method: **POST**
   - Copy the generated token → `MATTERMOST_SLASH_TOKEN`.
3. Set `MATTERMOST_BOT_PUBLIC_URL=https://<backend>` (required for the
   interactive buttons — export, share, feedback, delete — to call back).
4. `MATTERMOST_PRIVATE_DELIVERY=dm` for private answers with working buttons.

Security: slash requests are rejected unless the token matches; button callbacks
are HMAC-signed and verified; team/channel allow-lists are supported; tokens and
payloads are never logged. Full guide: [`README_MATTERMOST_BOT.md`](README_MATTERMOST_BOT.md).

---

## 9. Configuration & secrets

All tunables live in `src/config.py` and are overridable via `.env` — see
[`.env.example`](.env.example) (placeholders only; never commit real secrets).
Secrets belong in **GitHub Actions Secrets** and the **Render environment**, not
in git, the frontend, logs, or Docker image layers. Key values:

```
GROQ_MODEL=openai/gpt-oss-120b   EMBEDDING_MODEL=BAAI/bge-small-en-v1.5
TOP_K=5  MIN_SCORE_THRESHOLD=0.35  CONF_HIGH_THRESHOLD=0.62  CONF_MEDIUM_THRESHOLD=0.48
GROUNDING_MIN_OVERLAP=0.18  VERIFY_CITATIONS=true  LEXICAL_MIN_SCORE=0.55
CHUNK_SIZE=1200  CHUNK_OVERLAP=150  CHUNK_MIN_LEN=200
```

---

## 10. Testing

```bash
pip install numpy faiss-cpu rank-bm25 python-dotenv pytest fastapi httpx \
            python-multipart pydantic beautifulsoup4 lxml ftfy chardet
python -m pytest tests/ --ignore=tests/test_dashboard.py \
  --ignore=tests/test_dashboard_safety_helpers.py --ignore=tests/test_dashboard_helpers.py
```

The offline suite mocks the embedding model, so it needs no model download or
network. Streamlit-dashboard tests additionally require `streamlit`.

Key regression tests: `test_kb_version_reload.py` (stale index),
`test_bm25_evidence.py` (ATP), `test_golden_queries.py` (grounded answers +
refusals), `test_api_contract.py` (HTTP contract), `test_citation_integrity.py`.

---

## 11. Troubleshooting

| Symptom | Cause / fix |
|---|---|
| `/ready` returns 503 | No index (`/rag/status` → `kb_version: unversioned`) or `GROQ_API_KEY` unset. Run the KB pipeline / set the key. |
| Dashboard "Backend unreachable" | Wrong API base URL, or `API_ALLOWED_ORIGINS` doesn't include the Pages origin. |
| `ModelUnavailableError` | `GROQ_MODEL` is not `openai/gpt-oss-120b`. |
| Answers seem stale after a rebuild | Should not happen (version-aware reload). Confirm the new `kb_version` on `/rag/status`; if using the artifact model, ensure `KB_ARTIFACT_URL`/refresh are set. |
| Slash command 403 | `MATTERMOST_SLASH_TOKEN` missing or mismatched. |

See `SETUP_KB_UPDATE.md` for the ingestion/scheduler details and
`RAG_AUDIT_AND_UPGRADE.md` for the original audit notes.
