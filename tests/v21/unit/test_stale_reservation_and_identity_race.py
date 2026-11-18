"""Stale reservations and the infererence-identity race (C4/C5 follow-up).

Two failure windows survive a correct reservation design, and both are fixed here:

* a worker killed **between reserving and settling** leaves ``model_inferences``
  ``RUNNING`` forever, so its estimate is counted as reserved for the rest of the run;
* a stale worker that outlives its lease can race a healthy one on the inference
  identity insert.  The unique key keeps the data right, but the loser used to crash
  a task that had nothing wrong with it.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from decimal import Decimal

import pytest
from sqlalchemy.exc import IntegrityError

from embodied_dataops.workflow.handlers.vlm_preannotate_clip import _get_or_create_inference
from embodied_dataops.workflow.reaper import _release_stale_inferences, _stale_inference_statement

pytestmark = pytest.mark.no_db

NOW = datetime(2026, 10, 30, 12, 0, 0, tzinfo=UTC)


class _Inference:
    def __init__(self, inference_id: str, *, status: str = "RUNNING", created_at=None) -> None:
        self.id = inference_id
        self.run_id = "run-1"
        self.status = status
        self.error_code = None
        self.finished_at = None
        self.estimated_cost_usd = Decimal("0.01")
        self.created_at = created_at or (NOW - timedelta(hours=2))


class _ReaperSession:
    """Returns the rows the (tested separately) statement would have selected."""

    def __init__(self, rows: list[_Inference], billed: dict[str, float | None]) -> None:
        self.rows = rows
        self.billed = billed
        self._scalar_calls = 0

    def scalars(self, _statement):
        return list(self.rows)

    def scalar(self, _statement):
        row = self.rows[self._scalar_calls]
        self._scalar_calls += 1
        return self.billed.get(row.id, 0)


# ---------------------------------------------------------------------------
# stale reservation release
# ---------------------------------------------------------------------------


def test_a_stale_reservation_is_settled_and_releases_its_estimate():
    inference = _Inference("inf-1")
    session = _ReaperSession([inference], {"inf-1": None})

    released = _release_stale_inferences(session, NOW, older_than_seconds=1800)

    assert released == 1
    assert inference.status == "FAILED"
    assert inference.error_code == "STALE_RESERVATION"
    assert inference.finished_at == NOW
    # Nothing was ever billed, so the reservation must disappear entirely.
    assert inference.estimated_cost_usd is None


def test_a_stale_reservation_keeps_a_cost_that_was_actually_recorded():
    inference = _Inference("inf-1")
    session = _ReaperSession([inference], {"inf-1": 0.03})

    _release_stale_inferences(session, NOW, older_than_seconds=1800)

    assert inference.status == "FAILED"
    assert inference.estimated_cost_usd == Decimal("0.03")


def test_the_reaped_set_only_contains_running_rows_older_than_the_threshold():
    from sqlalchemy.dialects import postgresql

    sql = str(
        _stale_inference_statement(NOW, older_than_seconds=1800).compile(
            dialect=postgresql.dialect(), compile_kwargs={"literal_binds": True}
        )
    )
    assert "model_inferences.status = 'RUNNING'" in sql
    assert "model_inferences.created_at <" in sql
    # 1800 s before NOW -> a live 60 s call can never be in the result set.
    assert "2026-10-30 11:30:00" in sql
    # PostgreSQL only: the generic compiler drops SKIP LOCKED, so compiling with the
    # real dialect is the only way this assertion means anything.
    assert "FOR UPDATE SKIP LOCKED" in sql


def test_a_long_but_live_call_is_not_reaped():
    """The threshold must stay above the VLM_PREANNOTATE_CLIP wall clock (900 s)."""
    from embodied_dataops.config import Settings
    from embodied_dataops.domain.enums import STEP_TIMEOUT_SECONDS, Step

    assert Settings().vlm_stale_inference_seconds > STEP_TIMEOUT_SECONDS[Step.VLM_PREANNOTATE_CLIP]


# ---------------------------------------------------------------------------
# inference identity race
# ---------------------------------------------------------------------------


class _Clip:
    id = "clip-row-1"


class _Run:
    id = "run-1"


class _RaceSession:
    """First SELECT misses, INSERT violates the unique key, second SELECT finds the row."""

    def __init__(self, winner: _Inference) -> None:
        self.winner = winner
        self.selects = 0
        self.added: list[object] = []
        self.flushes = 0
        self.rollbacks = 0

    def scalars(self, _statement):
        self.selects += 1
        return _Result(self.winner if self.selects > 1 else None)

    def add(self, row) -> None:
        self.added.append(row)

    def flush(self) -> None:
        self.flushes += 1
        if self.flushes == 1:
            raise IntegrityError("INSERT INTO model_inferences", {}, Exception("duplicate key"))

    def rollback(self) -> None:
        self.rollbacks += 1

    def scalar(self, _statement):
        return 2  # two attempts already recorded by the winning worker


class _Result:
    def __init__(self, value) -> None:
        self._value = value

    def first(self):
        return self._value


def test_losing_the_identity_race_reuses_the_committed_row():
    winner = _Inference("inf-winner")
    session = _RaceSession(winner)

    inference, prior_attempts = _get_or_create_inference(
        session,
        run=_Run(),
        task_id="task-1",
        clip=_Clip(),
        provider_name="replay",
        model_name="replay",
        mode="REPLAY",
        prompt_sha256="a" * 64,
        prompt_version="ego-action-prompt/2.1",
        request_sha256="b" * 64,
    )

    assert inference is winner
    assert session.rollbacks == 1
    # The attempt counter is taken from the committed row, so the 3-attempt ceiling
    # cannot be reset by a worker that lost the race.
    assert prior_attempts == 2
