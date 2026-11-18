"""Model-call audit durability and atomic budget reservation (C4, C5).

Two properties are pinned here without a database:

* **C4** -- a failed remote call leaves durable evidence.  The audit writer is a
  separate session, and ``perform_inference`` asks it to persist after *every*
  attempt, so a ``TerminalTaskError`` (which makes the engine roll the task
  transaction back) cannot erase the record of a call that may already be billed.
* **C5** -- admission happens *after* the reservation row is written and under a
  ``FOR UPDATE`` lock on the run, so two workers cannot both read the same balance.
"""

from __future__ import annotations

from decimal import Decimal

import pytest

from embodied_dataops.ai.providers import (
    ERROR_BUDGET_EXCEEDED,
    ERROR_PROVIDER_UNAVAILABLE,
    InferenceRequest,
    KeyframeImage,
    ProviderError,
    ReplayProvider,
)
from embodied_dataops.ai.service import (
    COST_NOT_BILLED,
    COST_UNKNOWN_UPSTREAM_CHARGE,
    MAX_RECORDED_ATTEMPTS,
    BudgetState,
    budget_state,
    perform_inference,
    record_attempt,
    reserve_budget,
    settle_inference,
)
from embodied_dataops.config import Settings
from embodied_dataops.db.models import ModelCallAttempt, ModelInference

pytestmark = pytest.mark.no_db

JPEG = b"\xff\xd8\xff\xe0" + b"\x00" * 64


class _FakeSession:
    """Just enough Session surface for the audit/budget code paths."""

    def __init__(self, *, committed: float = 0.0, reserved: float = 0.0) -> None:
        self.committed = committed
        self.reserved = reserved
        self.added: list[object] = []
        self.statements: list[object] = []
        self.flushes = 0
        self.commits = 0
        self._scalar_calls = 0

    def execute(self, statement, *_args, **_kwargs):
        self.statements.append(statement)
        return None

    def scalar(self, _statement, *_args, **_kwargs):
        self._scalar_calls += 1
        return self.committed if self._scalar_calls % 2 == 1 else self.reserved

    def add(self, row) -> None:
        self.added.append(row)

    def flush(self) -> None:
        self.flushes += 1

    def commit(self) -> None:
        self.commits += 1


def _settings(**overrides) -> Settings:
    base = {
        "vlm_provider": "replay",
        "vlm_api_key": "",
        "vlm_budget_usd_per_run": 2.0,
        "vlm_max_attempts": 3,
    }
    base.update(overrides)
    return Settings(**base)


def _inference(**overrides) -> ModelInference:
    values = {
        "id": "inf-1",
        "run_id": "run-1",
        "task_id": "task-1",
        "clip_input_id": "clip-1",
        "provider": "replay",
        "model_name": "replay",
        "mode": "REPLAY",
        "prompt_version": "ego-action-prompt/2.1",
        "prompt_sha256": "a" * 64,
        "request_sha256": "b" * 64,
        "status": "QUEUED",
    }
    values.update(overrides)
    return ModelInference(**values)


def _request() -> InferenceRequest:
    return InferenceRequest(
        clip_id="clip-1",
        clip_start_ns=0,
        clip_end_ns=8_000_000_000,
        keyframes=(KeyframeImage.from_bytes(0, JPEG),),
    )


# ---------------------------------------------------------------------------
# C5: reservation under the run lock
# ---------------------------------------------------------------------------


def test_reservation_is_written_before_admission_and_locks_the_run():
    session = _FakeSession(committed=0.0, reserved=0.0)
    inference = _inference()

    admitted, state = reserve_budget(session, inference=inference, settings=_settings())

    assert admitted is True
    assert inference.status == "RUNNING"
    assert inference.estimated_cost_usd is not None
    assert state.estimate_usd == pytest.approx(float(inference.estimated_cost_usd))
    lock = session.statements[0]
    assert getattr(lock, "_for_update_arg", None) is not None


def test_a_reservation_that_would_exceed_the_budget_is_refused_and_released():
    session = _FakeSession(committed=0.09, reserved=0.0)
    inference = _inference()

    admitted, _state = reserve_budget(
        session, inference=inference, settings=_settings(vlm_budget_usd_per_run=0.05)
    )

    assert admitted is False
    assert inference.status == "FAILED"
    assert inference.error_code == ERROR_BUDGET_EXCEEDED
    # the reservation must be released, otherwise the refusal itself eats the budget
    assert inference.estimated_cost_usd is None


def test_a_second_worker_sees_the_first_reservation():
    """Both used to read "balance is enough" because nothing was written first."""
    settings = _settings(vlm_budget_usd_per_run=0.015)
    first_inference = _inference(id="inf-a")
    first = _FakeSession(committed=0.0, reserved=0.0)
    assert reserve_budget(first, inference=first_inference, settings=settings)[0] is True
    reserved_by_first = float(first_inference.estimated_cost_usd)

    # The second worker's read now includes the first worker's committed reservation.
    second = _FakeSession(committed=0.0, reserved=reserved_by_first)
    admitted, state = reserve_budget(second, inference=_inference(id="inf-b"), settings=settings)

    assert admitted is False
    assert state.running_reserved_usd == pytest.approx(reserved_by_first)


def test_admission_uses_committed_plus_other_running_reservations():
    """Admission reads the balance *before* writing this call's own reservation."""
    room = BudgetState(limit_usd=0.02, committed_usd=0.0, running_reserved_usd=0.01, estimate_usd=0.01)
    assert room.admitted() is True

    full = BudgetState(limit_usd=0.015, committed_usd=0.0, running_reserved_usd=0.01, estimate_usd=0.01)
    assert full.admitted() is False


def test_budget_state_sums_committed_and_running_rows():
    session = _FakeSession(committed=0.25, reserved=0.5)
    state = budget_state(session, "run-1", _settings())
    assert state.committed_usd == pytest.approx(0.25)
    assert state.running_reserved_usd == pytest.approx(0.5)


def test_settlement_writes_the_final_cost_and_status():
    session = _FakeSession()
    inference = _inference()
    settle_inference(
        session, inference=inference, status="FAILED", error_code="X", cost_usd=0.03
    )
    assert inference.status == "FAILED"
    assert inference.estimated_cost_usd == Decimal("0.03")
    assert inference.finished_at is not None


# ---------------------------------------------------------------------------
# C4: attempts survive a failing call
# ---------------------------------------------------------------------------


def test_every_attempt_is_persisted_before_the_task_transaction_can_roll_back():
    session = _FakeSession()
    commits: list[int] = []
    provider = ReplayProvider(fail_with=ProviderError(ERROR_PROVIDER_UNAVAILABLE, "down", retryable=True))

    outcome = perform_inference(
        session,
        inference=_inference(),
        provider=provider,
        request=_request(),
        settings=_settings(),
        clip_start_ns=0,
        clip_end_ns=8_000_000_000,
        sent_positions=[0],
        sleep=lambda _seconds: None,
        admitted=True,
        persist=lambda: commits.append(len(session.added)),
    )

    assert outcome.succeeded is False
    attempts = [row for row in session.added if isinstance(row, ModelCallAttempt)]
    assert len(attempts) == 3
    assert len(commits) == 3  # one durable write per real HTTP attempt
    assert all(row.cost_state == COST_UNKNOWN_UPSTREAM_CHARGE for row in attempts)


def test_a_transport_failure_is_recorded_as_an_unknown_upstream_charge():
    session = _FakeSession()
    provider = ReplayProvider(fail_with=ProviderError("VLM_PROVIDER_UNAVAILABLE", "timeout", retryable=False))

    outcome = perform_inference(
        session,
        inference=_inference(),
        provider=provider,
        request=_request(),
        settings=_settings(),
        clip_start_ns=0,
        clip_end_ns=8_000_000_000,
        sent_positions=[0],
        sleep=lambda _seconds: None,
        admitted=True,
    )

    assert outcome.cost_state == COST_UNKNOWN_UPSTREAM_CHARGE
    assert outcome.error_code == "VLM_PROVIDER_UNAVAILABLE"


def test_a_locally_rejected_call_is_not_billed():
    session = _FakeSession()
    provider = ReplayProvider(
        fail_with=ProviderError("VLM_IMAGE_TOO_LARGE", "too big", retryable=False, status=422)
    )

    outcome = perform_inference(
        session,
        inference=_inference(),
        provider=provider,
        request=_request(),
        settings=_settings(),
        clip_start_ns=0,
        clip_end_ns=8_000_000_000,
        sent_positions=[0],
        sleep=lambda _seconds: None,
        admitted=True,
    )
    assert outcome.cost_state == COST_NOT_BILLED


def test_recorded_attempts_cap_a_task_level_retry():
    """Attempt rows are durable now, so a retry must not start a fourth billed call."""
    session = _FakeSession()
    provider = ReplayProvider()

    outcome = perform_inference(
        session,
        inference=_inference(),
        provider=provider,
        request=_request(),
        settings=_settings(),
        clip_start_ns=0,
        clip_end_ns=8_000_000_000,
        sent_positions=[0],
        sleep=lambda _seconds: None,
        admitted=True,
        attempt_offset=MAX_RECORDED_ATTEMPTS,
    )

    assert outcome.succeeded is False
    assert outcome.attempts == 0
    assert not [row for row in session.added if isinstance(row, ModelCallAttempt)]


def test_record_attempt_defaults_an_http_200_to_a_known_cost():
    session = _FakeSession()
    row = record_attempt(
        session,
        inference_id="inf-1",
        attempt_no=1,
        started_at=None,
        response_status=200,
        estimated_cost_usd=0.02,
    )
    assert row.cost_state == "KNOWN"
