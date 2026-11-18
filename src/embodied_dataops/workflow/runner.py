"""Single-task execution engine shared by workers and the synchronous test driver.

``execute_task_by_id`` performs one full step: claim the task in PostgreSQL
(fenced lease), run the handler **inside the task's database transaction**, then
commit with the CAS or record a classified failure.  ``run_until_idle`` drives the
same machinery without Redis, which is what integration/failure tests use.

Cancellation/timeout design
---------------------------
The engine cooperates with handlers through ``HandlerContext.cancel_check``:

* a daemon heartbeat thread renews the 60s lease every 15s and aborts on a stale
  token (lease lost);
* the cancel check polls ``tasks.cancel_requested``, the lease token and the
  per-step timeout (SPEC 6.1);
* after the per-step timeout the heartbeat **stops renewing**, so even a handler
  that ignores cancellation cannot hold the lease forever: the reaper expires it
  within one lease interval and the task is retried/failed.

Hard pre-emption of a non-cooperative, CPU-bound handler is intentionally not
attempted in-process (Python cannot kill a thread safely); the persisted lease is
the recovery mechanism.
"""

from __future__ import annotations

import logging
import threading
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path

from sqlalchemy import select
from sqlalchemy.orm import Session, sessionmaker

from embodied_dataops.config import get_settings
from embodied_dataops.db.models import Task
from embodied_dataops.db.session import get_session_factory
from embodied_dataops.domain.enums import STEP_TIMEOUT_SECONDS, Step, TaskState
from embodied_dataops.errors import (
    ApiError,
    CancelledError,
    ErrorCode,
    RetryableError,
    TerminalTaskError,
)
from embodied_dataops.logging_setup import log_context, log_event
from embodied_dataops.storage.base import StorageBackend, get_storage
from embodied_dataops.workflow.claim import claim_next_task, claim_task_by_id
from embodied_dataops.workflow.completion import (
    cancel_running_task,
    complete_task,
    fail_task,
    heartbeat,
)
from embodied_dataops.workflow.handlers import get_handler
from embodied_dataops.workflow.handlers.base import HandlerContext, HandlerResult
from embodied_dataops.workflow.outbox import utcnow

logger = logging.getLogger(__name__)

OUTCOME_SUCCEEDED = "succeeded"
OUTCOME_RETRIED = "retried"
OUTCOME_FAILED = "failed"
OUTCOME_CANCELLED = "cancelled"
OUTCOME_SKIPPED = "skipped"
OUTCOME_LEASE_LOST = "lease_lost"

ALL_OUTCOMES = (
    OUTCOME_SUCCEEDED,
    OUTCOME_RETRIED,
    OUTCOME_FAILED,
    OUTCOME_CANCELLED,
    OUTCOME_SKIPPED,
    OUTCOME_LEASE_LOST,
)


@dataclass
class _ExecutionState:
    """Shared state between the handler thread and its heartbeat watchdog."""

    task_id: str
    token: str
    deadline: datetime | None
    stop_event: threading.Event = field(default_factory=threading.Event)
    cancel_event: threading.Event = field(default_factory=threading.Event)
    reason: str | None = None
    _lock: threading.Lock = field(default_factory=threading.Lock)

    def trip(self, reason: str) -> None:
        with self._lock:
            if self.reason is None:
                self.reason = reason
        self.cancel_event.set()

    def current_reason(self) -> str | None:
        with self._lock:
            return self.reason


def _make_cancel_check(
    factory: sessionmaker[Session], state: _ExecutionState, task_id: str, token: str
) -> Callable[[], bool]:
    def check() -> bool:
        if state.cancel_event.is_set():
            return True
        now = utcnow()
        if state.deadline is not None and now >= state.deadline:
            state.trip("timeout")
            return True
        try:
            with factory() as session:
                row = session.execute(
                    select(
                        Task.state,
                        Task.cancel_requested,
                        Task.lease_token,
                        Task.lease_until,
                    ).where(Task.id == task_id)
                ).first()
        except Exception:  # pragma: no cover - DB hiccup while polling
            return state.cancel_event.is_set()
        if row is None:
            state.trip("lease_lost")
            return True
        db_state, cancel_requested, db_token, lease_until = row
        if db_state != TaskState.RUNNING.value or db_token != token:
            state.trip("lease_lost")
            return True
        if cancel_requested:
            state.trip("cancelled")
            return True
        if lease_until is not None and lease_until < now:
            state.trip("lease_lost")
            return True
        return False

    return check


def _start_heartbeat(
    factory: sessionmaker[Session], state: _ExecutionState, interval_seconds: float
) -> threading.Thread:
    def loop() -> None:
        while not state.stop_event.wait(max(0.01, float(interval_seconds))):
            if state.cancel_event.is_set():
                return
            now = utcnow()
            if state.deadline is not None and now >= state.deadline:
                # Stop renewing: the lease expires and the reaper recovers the task.
                state.trip("timeout")
                return
            try:
                with factory() as session:
                    renewed = heartbeat(session, state.task_id, state.token)
            except Exception:  # pragma: no cover - transient DB error
                logger.warning("task.heartbeat_failed task_id=%s", state.task_id, exc_info=True)
                continue
            if not renewed:
                state.trip("lease_lost")
                return

    thread = threading.Thread(target=loop, name=f"heartbeat-{state.task_id}", daemon=True)
    thread.start()
    return thread


def execute_task_by_id(
    task_id: str,
    worker_id: str,
    *,
    session_factory: sessionmaker[Session] | None = None,
    storage: StorageBackend | None = None,
    pool: str | None = None,
    on_claimed: Callable[[str], None] | None = None,
    now: datetime | None = None,
) -> str:
    """Claim and execute one specific task; returns one of ``ALL_OUTCOMES``."""
    factory = session_factory or get_session_factory()
    backend = storage or get_storage()
    claim_session = factory()
    try:
        task = claim_task_by_id(claim_session, task_id, worker_id, pool=pool, now=now)
        if task is None:
            return OUTCOME_SKIPPED
        token = task.lease_token
        step = Step(task.step)
    finally:
        claim_session.close()
    if token is None:  # pragma: no cover - claim always sets a token
        return OUTCOME_SKIPPED
    if on_claimed is not None:
        on_claimed(task_id)
    return _execute_claimed(factory, backend, task_id, token, step, worker_id, now)


def _execute_claimed(
    factory: sessionmaker[Session],
    storage: StorageBackend,
    task_id: str,
    token: str,
    step: Step,
    worker_id: str,
    now: datetime | None,
) -> str:
    settings = get_settings()
    started = now or utcnow()
    timeout_seconds = STEP_TIMEOUT_SECONDS.get(step, 600)
    state = _ExecutionState(
        task_id=task_id,
        token=token,
        deadline=started + timedelta(seconds=timeout_seconds),
    )
    cancel_check = _make_cancel_check(factory, state, task_id, token)
    heartbeat_thread = _start_heartbeat(factory, state, settings.heartbeat_seconds) if settings.heartbeat_seconds > 0 else None

    session = factory()
    outcome = OUTCOME_LEASE_LOST
    try:
        with log_context(task_id=task_id, worker_id=worker_id):
            task = session.get(Task, task_id, populate_existing=True)
            if task is None or task.state != TaskState.RUNNING.value or task.lease_token != token:
                return OUTCOME_LEASE_LOST
            workspace = Path(settings.scratch_dir) / task_id
            workspace.mkdir(parents=True, exist_ok=True)
            ctx = HandlerContext(
                session=session,
                task=task,
                settings=settings,
                storage=storage,
                workspace=workspace,
                cancel_check=cancel_check,
            )
            try:
                handler = get_handler(step)
            except Exception as exc:
                session.rollback()
                log_event(logger, "task.handler_unavailable", task_id=task_id, error_code="HANDLER_UNAVAILABLE")
                return _record_failure(session, task_id, token, exc, retryable=True, reason="handler_unavailable")

            try:
                result = handler(ctx)
            except CancelledError as exc:
                outcome = _record_cancellation(session, state, task_id, token, exc)
            except RetryableError as exc:
                outcome = _record_failure(session, task_id, token, exc, retryable=True)
            except (TerminalTaskError, ApiError) as exc:
                outcome = _record_failure(session, task_id, token, exc, retryable=False)
            except Exception as exc:  # noqa: BLE001 - unexpected handler crash
                logger.exception("task.handler_crashed task_id=%s", task_id)
                outcome = _record_failure(
                    session, task_id, token, exc, retryable=True, reason="handler_exception"
                )
            else:
                outcome = _record_success(session, state, task_id, token, result, storage, now)

            if outcome == OUTCOME_SUCCEEDED:
                log_event(logger, "task.execution_succeeded", task_id=task_id, worker_id=worker_id)
            elif outcome in (OUTCOME_FAILED, OUTCOME_RETRIED):
                log_event(logger, "task.execution_failed", task_id=task_id, error_code=state.current_reason())
            return outcome
    finally:
        state.stop_event.set()
        if heartbeat_thread is not None:
            heartbeat_thread.join(timeout=2.0)
        session.close()


def _record_success(
    session: Session,
    state: _ExecutionState,
    task_id: str,
    token: str,
    result: HandlerResult | None,
    storage: StorageBackend,
    now: datetime | None,
) -> str:
    reason = state.current_reason()
    if reason == "cancelled":
        session.rollback()
        cancelled = cancel_running_task(session, task_id, token, now=now or utcnow())
        return OUTCOME_CANCELLED if cancelled else OUTCOME_LEASE_LOST
    if reason == "lease_lost":
        session.rollback()
        return OUTCOME_LEASE_LOST
    if reason == "timeout" or (state.deadline is not None and utcnow() > state.deadline):
        session.rollback()
        exc = RetryableError(ErrorCode.TIMEOUT, f"step exceeded its wall-clock budget ({state.deadline})")
        return _record_failure(session, task_id, token, exc, retryable=True, reason="timeout")
    if result is not None and not isinstance(result, HandlerResult):
        session.rollback()
        exc = TerminalTaskError(
            ErrorCode.INTERNAL_ERROR,
            f"handler returned {type(result)!r}, expected HandlerResult",
        )
        return _record_failure(session, task_id, token, exc, retryable=False)
    committed = complete_task(session, task_id, token, result, now=now)
    return OUTCOME_SUCCEEDED if committed else OUTCOME_LEASE_LOST


def _record_failure(
    session: Session,
    task_id: str,
    token: str,
    error: BaseException,
    *,
    retryable: bool,
    reason: str | None = None,
) -> str:
    # Discard any partial rows the handler created before the failure.
    session.rollback()
    new_state = fail_task(session, task_id, token, error, retryable=retryable, reason=reason)
    if new_state is None:
        return OUTCOME_LEASE_LOST
    return OUTCOME_RETRIED if new_state == TaskState.RETRY_WAIT.value else OUTCOME_FAILED


def _record_cancellation(
    session: Session, state: _ExecutionState, task_id: str, token: str, error: BaseException
) -> str:
    session.rollback()
    reason = state.current_reason()
    if reason == "timeout":
        return _record_failure(session, task_id, token, error, retryable=True, reason="timeout")
    if reason == "lease_lost":
        return OUTCOME_LEASE_LOST
    cancelled = cancel_running_task(session, task_id, token, now=utcnow())
    return OUTCOME_CANCELLED if cancelled else OUTCOME_LEASE_LOST


# ---------------------------------------------------------------------------
# Synchronous driver (tests / offline demo, no Redis)
# ---------------------------------------------------------------------------
def run_until_idle(
    session_factory: sessionmaker[Session] | None = None,
    pools: Iterable[str] = ("cpu", "vision", "ros2"),
    *,
    max_iterations: int = 100,
    worker_id: str = "test-runner",
    storage: StorageBackend | None = None,
    now: datetime | None = None,
) -> dict[str, int]:
    """Execute claimable tasks pool by pool until no task can be claimed.

    The driver deliberately does **not** advance time: retry-wait rows become
    claimable only when a test moves ``next_run_at`` into the past or the reaper
    promotes them, which keeps timing assertions exact and fast.
    """
    factory = session_factory or get_session_factory()
    backend = storage or get_storage()
    counts: dict[str, int] = {name: 0 for name in ALL_OUTCOMES}
    counts["iterations"] = 0
    iterations = 0
    while iterations < max_iterations:
        progressed = False
        for pool in pools:
            while iterations < max_iterations:
                claim_session = factory()
                try:
                    task = claim_next_task(claim_session, pool, worker_id, now=now)
                    if task is None:
                        break
                    task_id = task.id
                    token = task.lease_token
                    step = Step(task.step)
                finally:
                    claim_session.close()
                progressed = True
                iterations += 1
                counts["iterations"] = iterations
                if token is None:  # pragma: no cover
                    counts[OUTCOME_SKIPPED] += 1
                    continue
                outcome = _execute_claimed(factory, backend, task_id, token, step, worker_id, now)
                counts[outcome] += 1
        if not progressed:
            break
    return counts


__all__ = [
    "ALL_OUTCOMES",
    "OUTCOME_SUCCEEDED",
    "OUTCOME_RETRIED",
    "OUTCOME_FAILED",
    "OUTCOME_CANCELLED",
    "OUTCOME_SKIPPED",
    "OUTCOME_LEASE_LOST",
    "execute_task_by_id",
    "run_until_idle",
]
