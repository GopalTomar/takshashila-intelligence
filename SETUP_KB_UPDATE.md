# Knowledge Base — Robust Re-scrape + Weekly Automation

This update makes the crawl reach **every page and subpage** of the public
website and the Commit KB, keeps the chunk/embed/index step intact, and adds a
**self-maintaining weekly refresh (every Monday 18:00 IST)**. The retriever,
Streamlit dashboard and Mattermost bot are untouched — they read the same
`data/index/faiss.index` + `metadata.pkl` as before.

## Files in this update

| File | Change |
|------|--------|
| `src/config.py` | Two-tier crawl filters (follow-all vs keep), higher caps (5000 pages / depth 8), worker count, weekly schedule settings. |
| `scripts/crawl_engine.py` | Follows **every** internal link so nothing is missed; nav/archive pages are crawled but not stored; richer date/author/category metadata; thread-safe counters. |
| `scripts/update_knowledge_base.py` | Writes a run manifest, clearer summary, `--rescrape-all` alias. |
| `scripts/scheduler.py` | **New.** APScheduler weekly refresh + `--once`/`--run-now`/`--status`, single-instance lock. |
| `scripts/rescrape_all.py` | **New.** One command to re-scrape everything and rebuild. |
| `scripts/run_update.bat` | **New.** Windows runner for Task Scheduler. |
| `scripts/setup_windows_task.ps1` | **New.** Registers the weekly Windows task. |
| `requirements.txt` | Adds `APScheduler` + `tzdata` (Windows). |

## 1. Install the one new dependency

```powershell
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## 2. Set credentials in `.env` (Commit KB is skipped without them)

```
COMMIT_KB_URL=https://commit.takshashila.org.in/
COMMIT_KB_USERNAME=your_username
COMMIT_KB_PASSWORD=your_password
GROQ_API_KEY=gsk_...
```

## 3. One-time full re-scrape + rebuild (do this once now)

```powershell
python scripts\rescrape_all.py --reset-state
```

- `--reset-state` forgets prior crawl history so every page is re-fetched.
- Add `--fresh-index` to also re-embed every chunk from scratch.
- Website only / Commit KB only: `--website-only` / `--commit-kb-only`.

When it finishes you'll have a complete `data/processed/documents.jsonl` and a
rebuilt index. Nothing else in the app changes.

## 4. Automate the weekly refresh (Monday 18:00)

**Recommended on Windows — Task Scheduler (survives reboots, no window open):**

```powershell
powershell -ExecutionPolicy Bypass -File scripts\setup_windows_task.ps1
# change the time/day if you like:
# ... setup_windows_task.ps1 -Day Monday -At 18:00
# run it on demand to test:  Start-ScheduledTask -TaskName "TakshashilaKB-WeeklyUpdate"
# remove it:                 ... setup_windows_task.ps1 -Remove
```

**Cross-platform alternative — long-running scheduler process:**

```powershell
python scripts\scheduler.py            # blocks; fires every Monday 18:00 IST
python scripts\scheduler.py --run-now  # update now, then keep the schedule
python scripts\scheduler.py --status   # show the last run result
```

Each scheduled run is **incremental**: only new/changed/removed pages are
crawled and only changed chunks are re-embedded, so it's fast and safe to leave
alone. Logs go to `data/logs/weekly_update.log` (Task Scheduler) or
`data/logs/scheduler.log` (long-running mode); the last result is in
`data/logs/scheduler_status.json`.

## Manual commands (unchanged, still work)

```powershell
python scripts\update_knowledge_base.py                 # incremental, both sources
python scripts\update_knowledge_base.py --website-only
python scripts\update_knowledge_base.py --commit-kb-only
python scripts\update_knowledge_base.py --full          # = --rescrape-all
python scripts\build_index.py                           # rebuild index from documents.jsonl
```

## How full coverage works (why nothing is missed now)

The crawler now separates **discovery** from **keeping**:

- **Follow** every internal link except admin/auth/cart/feeds/JSON/mail/`#`
  fragments — so pages linked only from category, tag, author or paginated
  archive pages (and pages a stale sitemap forgot) are still reached.
- **Keep** a visited page as a KB document only if it isn't a nav/archive/search/
  pagination page **and** it passes the minimum-text-length filter.

Tuning (all optional, via `.env`): `WEBSITE_MAX_PAGES`, `WEBSITE_MAX_DEPTH`,
`WEBSITE_MIN_TEXT_LEN`, `SCRAPE_MAX_WORKERS`, `SCRAPE_DELAY`, and the schedule
knobs `SCHEDULE_DAY`, `SCHEDULE_HOUR`, `SCHEDULE_MINUTE`, `SCHEDULE_TIMEZONE`.
