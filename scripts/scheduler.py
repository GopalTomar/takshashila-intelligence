#!/usr/bin/env python3
"""
scheduler.py — Keep the Takshashila knowledge base fresh, automatically.

Runs an **incremental** update of both sources (public website + Commit KB) on a
weekly cron. The default is **every Monday at 18:00 India time** — only the
new/changed/removed pages are crawled, merged into ``documents.jsonl`` and
re-embedded, so a scheduled run is cheap and safe to leave unattended.

Two ways to use it
──────────────────
1. **Long-running process (cross-platform).** Start it once and leave it
   running (e.g. in a terminal, tmux, a service, or a container). It blocks and
   fires the job on schedule:

       python scripts/scheduler.py                 # run forever on the cron
       python scripts/scheduler.py --run-now       # update immediately, then keep scheduling
       python scripts/scheduler.py --status        # print the last run result and exit

2. **One-shot for an external scheduler (recommended on Windows).** Let Windows
   Task Scheduler (or cron) own the timing and just call the update once:

       python scripts/scheduler.py --once          # do one incremental update and exit

   See ``scripts/setup_windows_task.ps1`` — it registers a Task Scheduler entry
   that calls ``--once`` every Monday 18:00 and survives reboots.

Schedule + timezone are configurable in ``.env``:

    SCHEDULE_DAY=tue          # mon,tue,wed,thu,fri,sat,sun (or 0-6)
    SCHEDULE_HOUR=9           # 0-23, local to SCHEDULE_TIMEZONE
    SCHEDULE_MINUTE=0
    SCHEDULE_TIMEZONE=Asia/Kolkata

A single-instance lock (``data/logs/scheduler.lock``) prevents two runs from
overlapping, and every run's result is written to
``data/logs/scheduler_status.json`` for the dashboard / audit.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import config                                   # noqa: E402
from src.utils import get_logger, now_iso                # noqa: E402
from scripts.update_knowledge_base import run            # noqa: E402

logger = get_logger("scheduler", config.SCHEDULER_LOG)


# ════════════════════════════════════════════════════════════════════════════
#  Single-instance lock (so overlapping runs can't corrupt state)
# ════════════════════════════════════════════════════════════════════════════

_LOCK_STALE_SECONDS = 6 * 60 * 60   # a lock older than 6h is assumed dead


def _acquire_lock() -> bool:
    lock = config.SCHEDULER_LOCK
    try:
        if lock.exists():
            age = time.time() - lock.stat().st_mtime
            if age < _LOCK_STALE_SECONDS:
                logger.warning(f"Another update appears to be running "
                               f"(lock age {int(age)}s) — skipping this run.")
                return False
            logger.warning("Found a stale lock; overriding it.")
        lock.parent.mkdir(parents=True, exist_ok=True)
        lock.write_text(json.dumps({"pid": os.getpid(), "at": now_iso()}),
                        encoding="utf-8")
        return True
    except Exception as exc:
        logger.warning(f"Could not create lock ({exc}); proceeding without it.")
        return True


def _release_lock() -> None:
    try:
        config.SCHEDULER_LOCK.unlink(missing_ok=True)
    except Exception:
        pass


def _write_status(status: dict) -> None:
    try:
        config.SCHEDULER_STATUS.parent.mkdir(parents=True, exist_ok=True)
        config.SCHEDULER_STATUS.write_text(
            json.dumps(status, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    except Exception as exc:
        logger.warning(f"Could not write scheduler status: {exc}")


# ════════════════════════════════════════════════════════════════════════════
#  The job
# ════════════════════════════════════════════════════════════════════════════

def do_update(incremental: bool = True) -> dict:
    """Run one guarded incremental (or full) update of both sources + reindex."""
    kind = "incremental" if incremental else "full"
    if not _acquire_lock():
        return {"ok": False, "skipped": True, "reason": "locked"}

    started = time.time()
    logger.info(f"▶ Starting {kind} knowledge-base update…")
    _write_status({"state": "running", "kind": kind, "started_at": now_iso()})
    try:
        summary = run(website=True, commit_kb=True,
                      incremental=incremental, do_index=True)
        status = {
            "ok": True,
            "state": "success",
            "kind": kind,
            "started_at": datetime.utcfromtimestamp(started).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "finished_at": now_iso(),
            "duration_seconds": round(time.time() - started, 1),
            "summary": summary,
        }
        _write_status(status)
        logger.info(f"✓ Update complete in {status['duration_seconds']}s — "
                    f"{summary.get('changed_documents', 0)} documents changed.")
        return status
    except Exception as exc:
        logger.error(f"✗ Update failed: {exc}", exc_info=True)
        status = {"ok": False, "state": "error", "kind": kind,
                  "finished_at": now_iso(), "error": str(exc)}
        _write_status(status)
        return status
    finally:
        _release_lock()


# ════════════════════════════════════════════════════════════════════════════
#  Scheduling (long-running mode)
# ════════════════════════════════════════════════════════════════════════════

def _make_scheduler():
    from apscheduler.schedulers.blocking import BlockingScheduler
    from apscheduler.triggers.cron import CronTrigger

    try:
        # zoneinfo needs the 'tzdata' package on Windows (see requirements.txt).
        from zoneinfo import ZoneInfo
        tz = ZoneInfo(config.SCHEDULE_TIMEZONE)
    except Exception as exc:
        logger.warning(f"Timezone '{config.SCHEDULE_TIMEZONE}' unavailable "
                       f"({exc}); falling back to system local time.")
        tz = None

    scheduler = BlockingScheduler(timezone=tz) if tz else BlockingScheduler()
    trigger = CronTrigger(
        day_of_week=config.SCHEDULE_DAY,
        hour=config.SCHEDULE_HOUR,
        minute=config.SCHEDULE_MINUTE,
        timezone=tz,
    )
    scheduler.add_job(do_update, trigger, id="weekly_kb_update",
                      max_instances=1, coalesce=True, misfire_grace_time=3600)
    return scheduler, trigger


def serve(run_now: bool = False) -> int:
    if run_now:
        logger.info("Running an update immediately (--run-now)…")
        do_update(incremental=True)

    scheduler, trigger = _make_scheduler()
    when = (f"{config.SCHEDULE_DAY} at "
            f"{config.SCHEDULE_HOUR:02d}:{config.SCHEDULE_MINUTE:02d} "
            f"{config.SCHEDULE_TIMEZONE}")
    logger.info(f"Scheduler started — weekly knowledge-base update every {when}.")
    print(f"✓ Scheduler running. Weekly update every {when}.")
    print("  Leave this process running. Press Ctrl+C to stop.")
    try:
        scheduler.start()
    except (KeyboardInterrupt, SystemExit):
        print("\nScheduler stopped.")
        logger.info("Scheduler stopped by user.")
    return 0


# ════════════════════════════════════════════════════════════════════════════
#  CLI
# ════════════════════════════════════════════════════════════════════════════

def main() -> int:
    ap = argparse.ArgumentParser(description="Automated weekly knowledge-base refresh.")
    ap.add_argument("--once", action="store_true",
                    help="Run one incremental update and exit (for cron / Task Scheduler).")
    ap.add_argument("--full", action="store_true",
                    help="With --once: re-crawl everything instead of incremental.")
    ap.add_argument("--run-now", action="store_true",
                    help="Long-running mode: update immediately, then keep the weekly schedule.")
    ap.add_argument("--status", action="store_true",
                    help="Print the last run's status and exit.")
    args = ap.parse_args()

    if args.status:
        if config.SCHEDULER_STATUS.exists():
            print(config.SCHEDULER_STATUS.read_text(encoding="utf-8"))
        else:
            print("No scheduler status recorded yet.")
        return 0

    if args.once:
        result = do_update(incremental=not args.full)
        return 0 if result.get("ok") else 1

    return serve(run_now=args.run_now)


if __name__ == "__main__":
    raise SystemExit(main())