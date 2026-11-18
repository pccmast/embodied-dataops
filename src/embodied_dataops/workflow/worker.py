"""Redis Streams worker entry point.

Usage::

    python -m embodied_dataops.workflow.worker --pool cpu --concurrency 2
    python -m embodied_dataops.workflow.worker --pool vision --concurrency 1

Each consumer thread ``XREADGROUP``s a *wakeup hint* and then asks PostgreSQL for
the fenced lease.  The stream entry is ``XACK``ed as soon as the lease
transaction is committed - including invalid/duplicate hints - because Redis is
not the completion authority (SPEC 6.3.4).  Execution itself is delegated to
``runner.execute_task_by_id``: heartbeat thread, cooperative cancellation, step
timeout and CAS completion all live there.

SIGTERM/SIGINT stops claiming immediately; in-flight tasks get up to 20 seconds to
finish, after which the process exits and the reaper recovers their leases.
"""

from __future__ import annotations

import argparse
import logging
import signal
import threading
import time
import uuid
from collections.abc import Iterable

from embodied_dataops.config import get_settings
from embodied_dataops.db.session import get_session_factory
from embodied_dataops.logging_setup import configure_logging
from embodied_dataops.observability.metrics import ACTIVE_WORKERS
from embodied_dataops.storage.base import get_storage
from embodied_dataops.workflow import queue
from embodied_dataops.workflow.outbox import TOPIC_BY_POOL
from embodied_dataops.workflow.runner import ALL_OUTCOMES, execute_task_by_id

logger = logging.getLogger(__name__)

#: Grace period for in-flight tasks after SIGTERM (SPEC 6.2: at most 20s).
SHUTDOWN_GRACE_SECONDS = 20.0

#: Database-side concurrency ceilings from the specification (SPEC 3.1):
#: cpu worker total 2, vision worker total 1.
DEFAULT_CONCURRENCY = {"cpu": 2, "vision": 1, "ros2": 1}


def _ack(client, stream: str, entry_id: str) -> bool:
    try:
        queue.ack(client, stream, entry_id)
        return True
    except Exception:  # pragma: no cover - Redis transient failure
        logger.warning("worker.ack_failed stream=%s entry=%s", stream, entry_id, exc_info=True)
        return False


def _consumer_loop(
    *,
    client,
    stream: str,
    consumer: str,
    pool: str,
    worker_id: str,
    stop_event: threading.Event,
    session_factory,
    storage,
) -> None:
    logger.info("worker.consumer_started consumer=%s stream=%s", consumer, stream)
    while not stop_event.is_set():
        try:
            messages = queue.read_tasks(client, stream, consumer, count=1, block_ms=1000)
        except Exception:  # pragma: no cover - Redis unavailable
            logger.warning("worker.read_failed consumer=%s", consumer, exc_info=True)
            stop_event.wait(1.0)
            continue
        for entry_id, payload in messages:
            if stop_event.is_set():
                # Do not start new work after SIGTERM; leave the hint unacked so
                # XAUTOCLAIM/reaper can re-notify after the DB state is recovered.
                break
            task_id = payload.get("task_id")
            ack_state = {"acked": False}

            def _ack_once(
                _claimed_task_id: str | None = None,
                entry: str = entry_id,
                _state: dict = ack_state,
            ) -> None:
                # The runner calls on_claimed(task_id); the positional argument is
                # informational only - the stream entry id is what gets XACKed.
                if not _state["acked"]:
                    _state["acked"] = True
                    _ack(client, stream, entry)

            try:
                if not task_id:
                    _ack_once()
                    continue
                outcome = execute_task_by_id(
                    str(task_id),
                    worker_id,
                    session_factory=session_factory,
                    storage=storage,
                    pool=pool,
                    on_claimed=_ack_once,
                )
            except Exception:  # pragma: no cover - defensive: never kill the consumer
                logger.exception("worker.execution_error task_id=%s", task_id)
                outcome = "error"
            finally:
                # Invalid/duplicate hints are acknowledged even when no claim happened.
                _ack_once()
            logger.info(
                "worker.task_consumed task_id=%s outcome=%s consumer=%s",
                task_id,
                outcome,
                consumer,
                extra={
                    "event": "worker.task_consumed",
                    "extra_fields": {"task_id": task_id, "worker_id": worker_id},
                },
            )
            if outcome not in ALL_OUTCOMES and outcome != "error":  # pragma: no cover
                logger.warning("worker.unexpected_outcome outcome=%r", outcome)
    logger.info("worker.consumer_stopped consumer=%s", consumer)


def run_worker(
    pool: str,
    concurrency: int,
    *,
    redis_client=None,
    session_factory=None,
    stop_event: threading.Event | None = None,
    storage=None,
    worker_id: str | None = None,
    shutdown_grace_seconds: float = SHUTDOWN_GRACE_SECONDS,
) -> bool:
    """Start ``concurrency`` consumers for ``pool`` and block until stopped.

    Returns ``True`` when every consumer thread finished inside the grace period.
    """
    settings = get_settings()
    client = redis_client or queue.create_client()
    factory = session_factory or get_session_factory()
    backend = storage or get_storage()
    stream = TOPIC_BY_POOL[pool]
    # Tolerate a slow/restarting Redis at boot; read_tasks() re-ensures the group.
    queue.wait_for(client, [stream], timeout_seconds=30.0)
    worker_id = worker_id or settings.worker_id or f"{pool}-{uuid.uuid4().hex[:8]}"
    stop = stop_event or threading.Event()
    consumer_count = max(1, int(concurrency))

    threads: list[threading.Thread] = []
    for index in range(consumer_count):
        consumer = f"{worker_id}:{pool}:{index}"
        thread = threading.Thread(
            target=_consumer_loop,
            kwargs=dict(
                client=client,
                stream=stream,
                consumer=consumer,
                pool=pool,
                worker_id=worker_id,
                stop_event=stop,
                session_factory=factory,
                storage=backend,
            ),
            name=consumer,
            daemon=True,
        )
        thread.start()
        threads.append(thread)

    ACTIVE_WORKERS.labels(pool=pool).set(consumer_count)
    logger.info(
        "worker.started pool=%s worker_id=%s concurrency=%d stream=%s",
        pool,
        worker_id,
        consumer_count,
        stream,
        extra={"event": "worker.started", "extra_fields": {"worker_id": worker_id}},
    )
    try:
        while not stop.is_set():
            if not any(thread.is_alive() for thread in threads):
                break
            stop.wait(0.5)
        deadline = time.monotonic() + max(0.0, float(shutdown_grace_seconds))
        for thread in threads:
            remaining = deadline - time.monotonic()
            thread.join(timeout=max(0.0, remaining))
        alive = [thread.name for thread in threads if thread.is_alive()]
        if alive:
            logger.warning("worker.shutdown_incomplete threads=%s", alive)
        else:
            logger.info("worker.stopped pool=%s", pool)
        return not alive
    finally:
        ACTIVE_WORKERS.labels(pool=pool).set(0)


def _install_signal_handlers(stop: threading.Event) -> None:
    def _stop(signum, _frame):  # pragma: no cover - signal path
        logger.info(
            "worker.signal received=%s",
            signum,
            extra={"event": "worker.signal", "extra_fields": {"signum": signum}},
        )
        stop.set()

    signal.signal(signal.SIGTERM, _stop)
    signal.signal(signal.SIGINT, _stop)


def main(argv: Iterable[str] | None = None) -> int:
    settings = get_settings()
    parser = argparse.ArgumentParser(
        prog="python -m embodied_dataops.workflow.worker",
        description="Embodied DataOps task worker (PostgreSQL truth source + Redis wakeups)",
    )
    parser.add_argument("--pool", choices=sorted(TOPIC_BY_POOL), default=settings.worker_pool)
    parser.add_argument(
        "--concurrency",
        type=int,
        default=None,
        help="consumer threads; defaults to cpu=2, vision=1, ros2=1",
    )
    parser.add_argument("--worker-id", default=None, help="stable worker identity override")
    args = parser.parse_args(list(argv) if argv is not None else None)
    concurrency = args.concurrency or DEFAULT_CONCURRENCY.get(args.pool, 1)

    configure_logging(f"worker-{args.pool}", settings.log_level)
    stop = threading.Event()
    _install_signal_handlers(stop)
    client = queue.create_client()
    ok = run_worker(
        args.pool,
        concurrency,
        redis_client=client,
        stop_event=stop,
        worker_id=args.worker_id,
    )
    return 0 if ok else 1


if __name__ == "__main__":  # pragma: no cover - module entry point
    raise SystemExit(main())
