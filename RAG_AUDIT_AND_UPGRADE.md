# Takshashila RAG — Production Audit & Upgrade

This pass makes the pipeline answer **content questions and metadata questions**
("who wrote this?", "when was it published?", "what tags?", "which category?")
with **exact article citations**, while honestly saying *insufficient evidence*
only when the answer truly isn't in the knowledge base. It keeps the existing
architecture and the dashboard/bot untouched — every change is inside the
ingestion + retrieval layers they already depend on.

---

## 1. Summary of improvements

**The headline problem** (your screenshot): *"Who wrote this blog?"* returned
*insufficient evidence* even though the blog and its author exist. Root cause:
only the chunk **body** was embedded and BM25-indexed — **author/date/title/
tags/section were never searchable**, and the LLM context header omitted them.
Two things were needed: (a) capture that metadata during crawling, and (b) make
it searchable and visible to the model.

Delivered:

1. **Metadata is now searchable and answerable.** Every chunk gets a compact
   metadata header (Title / Author / Published / Updated / Category / Section /
   Tags / Type / Source) that is embedded + BM25-indexed and shown in the LLM
   context. Metadata questions now retrieve the right article and the model can
   answer + cite them. The body shown to the user is unchanged.
2. **Rich metadata extraction** from JSON-LD, OpenGraph, Schema.org and `<meta>`
   tags: authors, published/updated dates, section, tags, breadcrumbs, canonical
   URL, language, description, document type, page id — with URL/text fallbacks.
   Robust to malformed JSON-LD.
3. **Exact references.** Aggregate index / section-landing / homepage pages are
   detected by a single shared predicate and never cited (retriever skips them,
   pipeline filters them, crawler follows-but-doesn't-store them). Answers cite
   the specific article's URL.
4. **Full-site crawl coverage** (from the prior pass, retained): follows every
   internal link so blogs/publications/research/events/etc. and pages reachable
   only via nav/pagination are all discovered.
5. **Validation** (`scripts/validate_kb.py`): detects missing URLs/titles/authors/
   dates, duplicate URLs, duplicate/oversized/undersized/orphan chunks, and
   index↔metadata mismatches. Runs automatically after every build.
6. **Reports**: each run writes a timestamped `ingestion_report_*.md/.json`
   (added/updated/removed/failed per source, index stats, validation result)
   plus a machine-readable manifest.
7. **Automation** (from the prior pass, retained): weekly Tuesday 09:00 refresh
   (Windows Task Scheduler or APScheduler), incremental + hash-skip + report.
8. **Tests**: `scripts/test_metadata_qa.py` (metadata + content QA + exact refs +
   validation) alongside the existing `test_incremental_pipeline.py`.

---

## 2. Files modified / added

| File | Change |
|------|--------|
| `src/utils.py` | **+** `build_meta_header`, `chunk_search_text` (metadata-aware embed/BM25 text); shared source-quality predicate already added earlier. |
| `src/chunker.py` | Rich chunk metadata (subtitle, authors, updated_date, section, heading_path, canonical_url, language, document_type, page_id, chunk_order); builds `meta_header`; hashes on the search text. |
| `src/embedding_cache.py` | Embeds + keys on `chunk_search_text` (metadata edits correctly invalidate the cache). |
| `src/incremental_index.py` | Embeds `chunk_search_text`. |
| `src/vector_store.py` | `build_index` + legacy append embed `chunk_search_text`. |
| `src/retriever.py` | BM25 over `chunk_search_text`; evidence-first selection (listing/nav pages never take a slot; backfill only when nothing else exists). |
| `src/rag_pipeline.py` | Context header now includes author/date/updated/section/tags; system prompt permits metadata answers; evidence filter uses the shared predicate. |
| `scripts/crawl_engine.py` | `_extract_rich_metadata` (JSON-LD/OG/schema/breadcrumbs); documents carry all rich fields; listing/landing pages followed-but-not-stored. |
| `scripts/update_knowledge_base.py` | Runs validation after build; writes ingestion report + manifest. |
| `scripts/validate_kb.py` | **New** — KB/index health checks (importable + CLI). |
| `scripts/test_metadata_qa.py` | **New** — metadata QA + validation test. |

Earlier passes (also included): `scripts/scheduler.py`, `scripts/rescrape_all.py`,
`scripts/run_update.bat`, `scripts/setup_windows_task.ps1`, `requirements.txt`.

---

## 3. Why each change was made

- **Metadata header in the embedded text** — the only reliable way to make
  "who/when/what-tags/which-category" retrievable *and* answerable without a
  separate metadata store; it puts the facts in both the vector space and the
  model's context.
- **Hash on search text** — so a metadata correction (e.g. a fixed author)
  invalidates the stale embedding and two identical bodies with different
  metadata stay distinct.
- **Rich extraction from JSON-LD/OG** — article pages expose author/date most
  reliably there; HTML-selector scraping alone missed them (the cause of the
  screenshot failure).
- **Listing/landing exclusion via one shared predicate** — guarantees the three
  layers agree, so references always resolve to the exact article.
- **Validation + reports** — make silent data problems (missing authors, dup
  URLs, index drift) visible before they degrade answers.

---

## 4. Expected impact on retrieval quality

- Metadata questions: from *insufficient evidence* → answered with the correct
  author/date/tags and the exact article citation.
- Content questions: same-or-better recall (metadata terms add signal), and
  references now point to the specific article instead of a listing/homepage.
- Fewer false "insufficient evidence" results, because the searchable surface of
  each chunk now includes title/section/author/tags.
- No change to latency of note (header adds a few tokens per chunk; embeddings
  are cached).

---

## 5. Remaining limitations (honest)

- **Chunking is still character/sentence-based**, not fully heading-structured.
  Section is attributed at the document level (from breadcrumbs/JSON-LD), not a
  per-paragraph heading path. True structural (heading-aware) chunking from HTML
  is a further improvement.
- **No cross-encoder reranker.** Retrieval is hybrid RRF + source-priority +
  evidence-first. A cross-encoder (e.g. `bge-reranker`) would sharpen ordering
  at some latency/dependency cost; it's a clean drop-in next step.
- **JS-rendered pages**: extraction is HTTP + trafilatura (no headless browser).
  If some Takshashila pages render content client-side, those need a JS renderer.
- **Author/date depend on the site exposing them** (JSON-LD/OG/meta or URL
  patterns). Pages with none will still lack those fields — validation flags them.
- **Embedding model unchanged** (`bge-small-en-v1.5`). It's a solid,
  cost-free default; `bge-base`/`bge-large` or an API embedder would raise
  ceiling accuracy at higher cost. Changing it requires a full re-embed and
  updating `EMBEDDING_DIM`.
- First build after this upgrade **re-embeds everything once** (chunk hashes now
  include metadata) — expected, one-time.

---

## 6. Production-readiness checklist

Before relying on it in production:

- [ ] `pip install -r requirements.txt` (adds APScheduler + tzdata).
- [ ] Set `COMMIT_KB_USERNAME` / `COMMIT_KB_PASSWORD` / `GROQ_API_KEY` in `.env`.
- [ ] One-time full rebuild: `python scripts/rescrape_all.py --reset-state`.
- [ ] `python scripts/validate_kb.py` → resolve any ERRORs (WARNs are advisory).
- [ ] Spot-check metadata questions in the app ("who wrote X", "when was X
      published", "which blog discusses Y") and confirm references open the exact
      article.
- [ ] Register the weekly job:
      `powershell -ExecutionPolicy Bypass -File scripts\setup_windows_task.ps1`
      (or run `python scripts/scheduler.py`).
- [ ] Confirm `data/logs/ingestion_report_*.md` is produced after a run.
- [ ] Run tests: `python scripts/test_incremental_pipeline.py` and
      `python scripts/test_metadata_qa.py`.
- [ ] Decide on optional upgrades: cross-encoder reranker, larger embedding
      model, heading-aware chunking, JS rendering (only if pages need it).

---

## What the weekly automation does (unchanged, Tuesday 09:00)

Crawl website → crawl Commit KB → detect new/updated pages (hash-skip
unchanged) → merge delta → re-embed only changed chunks → rebuild FAISS + BM25 →
**validate** → write crawl + ingestion reports. No manual step required.

