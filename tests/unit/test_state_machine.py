"""Only the transitions listed in SPEC 6.2 are legal."""

from __future__ import annotations

import pytest

from embodied_dataops.domain.enums import (
    DatasetVersionState,
    EpisodeStatus,
    ImportJobState,
    RunState,
    TaskState,
)
from embodied_dataops.domain.state_machines import (
    DATASET_VERSION_TRANSITIONS,
    EPISODE_STATUS_TRANSITIONS,
    IMPORT_JOB_TRANSITIONS,
    TASK_TRANSITIONS,
    TERMINAL_TASK_STATES,
    assert_run_transition,
    assert_task_transition,
    can_transition,
)
from embodied_dataops.errors import ApiError, ErrorCode

pytestmark = pytest.mark.unit


@pytest.mark.parametrize(
    ("current", "target"),
    [
        (TaskState.PENDING, TaskState.RUNNING),
        (TaskState.PENDING, TaskState.CANCELLED),
        (TaskState.RUNNING, TaskState.SUCCEEDED),
        (TaskState.RUNNING, TaskState.RETRY_WAIT),
        (TaskState.RUNNING, TaskState.FAILED),
        (TaskState.RUNNING, TaskState.CANCELLED),
        (TaskState.RETRY_WAIT, TaskState.PENDING),
        (TaskState.RETRY_WAIT, TaskState.CANCELLED),
        (TaskState.RETRY_WAIT, TaskState.FAILED),
    ],
)
def test_legal_task_transitions_are_accepted(current: TaskState, target: TaskState):
    assert_task_transition(current, target)  # must not raise
    assert can_transition(TASK_TRANSITIONS, current, target)


@pytest.mark.parametrize(
    ("current", "target"),
    [
        (TaskState.RUNNING, TaskState.PENDING),  # cannot go back without a retry
        (TaskState.PENDING, TaskState.SUCCEEDED),  # must claim first
        (TaskState.PENDING, TaskState.RETRY_WAIT),  # only RUNNING can fail
        (TaskState.RETRY_WAIT, TaskState.RUNNING),  # must be promoted to PENDING first
        (TaskState.SUCCEEDED, TaskState.RUNNING),
        (TaskState.SUCCEEDED, TaskState.FAILED),
        (TaskState.FAILED, TaskState.RUNNING),
        (TaskState.FAILED, TaskState.SUCCEEDED),
        (TaskState.CANCELLED, TaskState.PENDING),
        (TaskState.CANCELLED, TaskState.RUNNING),
    ],
)
def test_illegal_task_transitions_are_rejected(current: TaskState, target: TaskState):
    with pytest.raises(ApiError) as excinfo:
        assert_task_transition(current, target)
    assert excinfo.value.code is ErrorCode.TASK_ALREADY_TERMINAL
    assert excinfo.value.details["from"] == current.value
    assert excinfo.value.details["to"] == target.value


def test_terminal_task_states_have_no_successors():
    for state in TERMINAL_TASK_STATES:
        assert TASK_TRANSITIONS[state] == frozenset()
    assert TERMINAL_TASK_STATES == {
        TaskState.SUCCEEDED,
        TaskState.FAILED,
        TaskState.CANCELLED,
    }


@pytest.mark.parametrize(
    ("current", "target"),
    [
        (RunState.QUEUED, RunState.RUNNING),
        (RunState.QUEUED, RunState.CANCELLED),
        (RunState.QUEUED, RunState.FAILED),
        (RunState.RUNNING, RunState.SUCCEEDED),
        (RunState.RUNNING, RunState.FAILED),
        (RunState.RUNNING, RunState.CANCELLED),
    ],
)
def test_legal_run_transitions(current: RunState, target: RunState):
    assert_run_transition(current, target)


@pytest.mark.parametrize(
    ("current", "target"),
    [
        (RunState.RUNNING, RunState.QUEUED),
        (RunState.SUCCEEDED, RunState.RUNNING),
        (RunState.FAILED, RunState.RUNNING),
        (RunState.CANCELLED, RunState.RUNNING),
    ],
)
def test_illegal_run_transitions_raise(current: RunState, target: RunState):
    with pytest.raises(ApiError) as excinfo:
        assert_run_transition(current, target)
    assert excinfo.value.code is ErrorCode.RUN_ALREADY_TERMINAL


def test_import_job_machine_matches_spec():
    assert IMPORT_JOB_TRANSITIONS[ImportJobState.QUEUED] == frozenset(
        {ImportJobState.RUNNING, ImportJobState.FAILED}
    )
    assert IMPORT_JOB_TRANSITIONS[ImportJobState.RUNNING] == frozenset(
        {ImportJobState.SUCCEEDED, ImportJobState.FAILED}
    )
    assert IMPORT_JOB_TRANSITIONS[ImportJobState.SUCCEEDED] == frozenset()


def test_episode_machine_matches_spec():
    assert EPISODE_STATUS_TRANSITIONS[EpisodeStatus.PROCESSING] == frozenset(
        {EpisodeStatus.READY_FOR_REVIEW, EpisodeStatus.QUALITY_FAILED}
    )
    assert EpisodeStatus.READY_FOR_REVIEW in EPISODE_STATUS_TRANSITIONS[EpisodeStatus.PROCESSING]
    assert EpisodeStatus.READY_FOR_REVIEW not in EPISODE_STATUS_TRANSITIONS[EpisodeStatus.IMPORTED]


def test_dataset_version_published_is_immutable():
    assert DATASET_VERSION_TRANSITIONS[DatasetVersionState.PUBLISHED] == frozenset()
    assert can_transition(
        DATASET_VERSION_TRANSITIONS, DatasetVersionState.READY, DatasetVersionState.PUBLISHED
    )
