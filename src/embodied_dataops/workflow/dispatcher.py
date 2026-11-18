"""Transactional outbox dispatcher and recovery loop (SPEC 3.1, 6.3.2).

Every second the dispatcher reads unpublished ``outbox_events`` rows with row
locks, ``XADD``s them to ``stream:tasks:{cpu|vision|ros2}`` and marks
``published_at``.  The task row itself was committed *before* the outbox row was
visible, so a Redis outage only delays a wakeup: the row stays unpublished and is
retried by the next cycle.  Re-publishing a duplicate wakeup is safe because the
database lease, not Redis, decides who runs a task.

The dispatcher process also runs the recovery duties assigned to it by SPEC 3.1
("超时任务扫描、延迟重试恢复、失联 Worker 回收"): every
``reaper_interval_seconds`` (10s) it executes
:func:`embodied_dataops.workflow.reaper.reap_once`, which expires dead leases,
promotes due retries, re-notifies lost wakeups and reclaims stale Redis PEL
entries.  The reaper module is separately runnable for ops/tests.
"""

from __future__ import annotations

import logging
import signal
import threading
import time

from sqlalchemy.orm import Session

from embodied_dataops.config import get_settings
from embodied_dataops.db.session import session_scope
from embodied_dataops.logging_setup import configure_logging, log_event
from embodied_dataops.observability.metrics import OUTBOX_PUBLISH_FAILURES
from embodied_dataops.operations.gap_analysis import process_queued_gap_analyses
from embodied_dataops.workflow import queue
from embodied_dataops.workflow.outbox import (
    ALL_STREAMS,
    mark_published,
    pending_count,
    take_pending,
)
from embodied_dataops.workflow.reaper import reap_once

logger = logging.getLogger(__name__)


def dispatch_once(session: Session, redis_client, *, limit: int = 100) -> dict[str, int]:
    """Publish all currently unpublished outbox events that fit in one batch.

    Returns counters; on a Redis or commit failure the offending row (and every
    row after it) stays unpublished for the next cycle.
    """
    events = take_pending(session, limit=limit)
    published = 0
    failed = 0
    for event in events:
        payload = dict(event.payload_json or {})
        topic = event.topic
        try:
            queue.publish_task(redis_client, topic, payload)
        except Exception:
            OUTBOX_PUBLISH_FAILURES.inc()
            failed += 1
            _safe_rollback(session)
            log_event(
                logger,
                "outbox.publish_failed",
                level=logging.WARNING,
                task_id=payload.get("task_id"),
                error_code="REDIS_UNAVAILABLE",
            )
            break
        try:
            mark_published(event)
            session.commit()
        except Exception:
            # XADD succeeded but recording it failed: the event stays unpublished
            # and will be re-published; a duplicate hint is harmless (AT-13).
            OUTBOX_PUBLISH_FAILURES.inc()
            failed += 1
            _safe_rollback(session)
            log_event(
                logger,
                "outbox.mark_published_failed",
                level=logging.WARNING,
                task_id=payload.get("task_id"),
                error_code="OUTBOX_MARK_FAILED",
            )
            break
        published += 1
        log_event(logger, "outbox.published", task_id=payload.get("task_id"))
    return {
        "published": published,
        "failed": failed,
        "unpublished": pending_count(session),
    }


def _safe_rollback(session: Session) -> None:
    try:
        session.rollback()
    except Exception:  # pragma: no cover - the connection itself is broken
        logger.warning("outbox.rollback_failed", exc_info=True)


def run_forever(*, stop_event: threading.Event | None = None, redis_client=None, limit: int = 100) -> None:
    """Publish outbox rows every 1s and run lease recovery every 10s."""
    settings = get_settings()
    stop = stop_event or threading.Event()
    interval = settings.dispatcher_interval_seconds
    reap_interval = settings.reaper_interval_seconds
    last_reap = 0.0
    while not stop.is_set():
        try:
            with session_scope() as session:
                dispatch_once(session, redis_client, limit=limit)
        except Exception:  # pragma: no cover - keep the dispatcher alive
            logger.exception("outbox.dispatch_cycle_failed")

        # DO-A02 C3/C4: complete any queued gap analysis (deterministic, bounded).
        try:
            with session_scope() as session:
                processed = process_queued_gap_analyses(session)
                if processed:
                    logger.info("gap_analysis.cycle processed=%d", processed)
        except Exception:  # pragma: no cover - retried on the next dispatcher cycle
            logger.exception("gap_analysis.cycle_failed")

        now_monotonic = time.monotonic()
        if now_monotonic - last_reap >= reap_interval:
            last_reap = now_monotonic
            try:
                with session_scope() as session:
                    reap_once(session, redis_client=redis_client, commit=False)
            except Exception:  # pragma: no cover - recovery retried next cycle
                logger.exception("reaper.cycle_failed")
        stop.wait(interval)


def main() -> int:
    settings = get_settings()
    configure_logging("dispatcher", settings.log_level)
    redis_client = queue.create_client()
    try:
        for stream in ALL_STREAMS:
            queue.ensure_group(redis_client, stream)
    except Exception:  # pragma: no cover - Redis may be slow/unavailable at boot
        logger.warning("dispatcher.redis_not_ready; PostgreSQL outbox keeps events", exc_info=True)
    stop = threading.Event()

    def _stop(signum, _frame):  # pragma: no cover - signal handling
        logger.info("dispatcher.signal received=%s", signum)
        stop.set()

    signal.signal(signal.SIGTERM, _stop)
    signal.signal(signal.SIGINT, _stop)
    logger.info(
        "dispatcher.started interval_seconds=%s",
        settings.dispatcher_interval_seconds,
        extra={"event": "dispatcher.started"},
    )
    run_forever(stop_event=stop, redis_client=redis_client)
    return 0


if __name__ == "__main__":  # pragma: no cover - module entry point
    raise SystemExit(main())
