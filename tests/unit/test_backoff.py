"""Deterministic retry backoff (SPEC 6.2, AT-14).

The schedule is 5s after the first failed attempt and 20s after the second, with
a deterministic ±10% jitter derived from ``hash(task_id)`` so tests and operators
can reproduce it exactly.
"""

from __future__ import annotations

import re

import pytest

from embodied_dataops.workflow.completion import (
    BACKOFF_JITTER_FRACTION,
    backoff_delay_seconds,
    jitter_unit,
)

pytestmark = pytest.mark.unit

TASK_ID = "0192f0aa-1111-7abc-8def-0123456789ab"
UUID_RE = re.compile(r"^[0-9a-f-]{36}$")


def test_first_retry_is_five_seconds_with_ten_percent_jitter():
    for offset in range(20):
        task_id = f"{TASK_ID[:-2]}{offset:02d}"
        delay = backoff_delay_seconds(task_id, 1)
        assert 5.0 * (1 - BACKOFF_JITTER_FRACTION) <= delay <= 5.0 * (1 + BACKOFF_JITTER_FRACTION)


def test_second_retry_is_twenty_seconds_with_ten_percent_jitter():
    for offset in range(20):
        task_id = f"{TASK_ID[:-2]}{offset:02d}"
        delay = backoff_delay_seconds(task_id, 2)
        assert 20.0 * (1 - BACKOFF_JITTER_FRACTION) <= delay <= 20.0 * (1 + BACKOFF_JITTER_FRACTION)


def test_backoff_is_deterministic_for_same_task_and_attempt():
    first = backoff_delay_seconds(TASK_ID, 1)
    second = backoff_delay_seconds(TASK_ID, 1)
    assert first == second
    assert backoff_delay_seconds(TASK_ID, 2) == backoff_delay_seconds(TASK_ID, 2)


def test_jitter_varies_between_tasks_and_between_attempts():
    first_attempt = {backoff_delay_seconds(f"task-{i}", 1) for i in range(20)}
    assert len(first_attempt) > 1, "jitter must not be a constant"
    # attempts have independent jitter, so the ratio is usually not exactly 4x
    ratios = {round(backoff_delay_seconds(f"task-{i}", 2) / backoff_delay_seconds(f"task-{i}", 1), 9) for i in range(20)}
    assert len(ratios) > 1


def test_jitter_unit_is_in_unit_interval():
    for i in range(50):
        value = jitter_unit(f"task-{i}", (i % 3) + 1)
        assert 0.0 <= value < 1.0


def test_attempts_beyond_schedule_use_the_last_base_delay():
    # Only two backoff steps exist; further attempts (defensive path) reuse 20s.
    delay = backoff_delay_seconds(TASK_ID, 7)
    assert 18.0 <= delay <= 22.0


def test_settings_schedule_is_honoured():
    from embodied_dataops.config import Settings

    custom = Settings(retry_backoff_seconds=(1.0, 2.0))
    assert 0.9 <= backoff_delay_seconds(TASK_ID, 1, settings=custom) <= 1.1
    assert 1.8 <= backoff_delay_seconds(TASK_ID, 2, settings=custom) <= 2.2
