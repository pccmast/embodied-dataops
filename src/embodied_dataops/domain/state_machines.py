"""Task and run state machines (SPEC 6.2).  Only the listed transitions are legal."""

from __future__ import annotations

from embodied_dataops.domain.enums import (
    DatasetVersionState,
    EpisodeStatus,
    ImportJobState,
    RunState,
    TaskState,
)
from embodied_dataops.errors import ApiError, ErrorCode

TASK_TRANSITIONS: dict[TaskState, frozenset[TaskState]] = {
    TaskState.PENDING: frozenset({TaskState.RUNNING, TaskState.CANCELLED, TaskState.FAILED}),
    TaskState.RUNNING: frozenset(
        {TaskState.SUCCEEDED, TaskState.RETRY_WAIT, TaskState.FAILED, TaskState.CANCELLED}
    ),
    TaskState.RETRY_WAIT: frozenset({TaskState.PENDING, TaskState.CANCELLED, TaskState.FAILED}),
    TaskState.SUCCEEDED: frozenset(),
    TaskState.FAILED: frozenset(),
    TaskState.CANCELLED: frozenset(),
}

RUN_TRANSITIONS: dict[RunState, frozenset[RunState]] = {
    RunState.QUEUED: frozenset({RunState.RUNNING, RunState.CANCELLED, RunState.FAILED}),
    RunState.RUNNING: frozenset({RunState.SUCCEEDED, RunState.FAILED, RunState.CANCELLED}),
    RunState.SUCCEEDED: frozenset(),
    RunState.FAILED: frozenset(),
    RunState.CANCELLED: frozenset(),
}

IMPORT_JOB_TRANSITIONS: dict[ImportJobState, frozenset[ImportJobState]] = {
    ImportJobState.QUEUED: frozenset({ImportJobState.RUNNING, ImportJobState.FAILED}),
    ImportJobState.RUNNING: frozenset({ImportJobState.SUCCEEDED, ImportJobState.FAILED}),
    ImportJobState.SUCCEEDED: frozenset(),
    ImportJobState.FAILED: frozenset(),
}

DATASET_VERSION_TRANSITIONS: dict[DatasetVersionState, frozenset[DatasetVersionState]] = {
    DatasetVersionState.BUILDING: frozenset({DatasetVersionState.READY, DatasetVersionState.FAILED}),
    DatasetVersionState.READY: frozenset({DatasetVersionState.PUBLISHED, DatasetVersionState.FAILED}),
    DatasetVersionState.PUBLISHED: frozenset(),
    DatasetVersionState.FAILED: frozenset(),
}

EPISODE_STATUS_TRANSITIONS: dict[EpisodeStatus, frozenset[EpisodeStatus]] = {
    EpisodeStatus.IMPORTED: frozenset({EpisodeStatus.PROCESSING, EpisodeStatus.QUALITY_FAILED}),
    EpisodeStatus.PROCESSING: frozenset(
        {EpisodeStatus.READY_FOR_REVIEW, EpisodeStatus.QUALITY_FAILED}
    ),
    EpisodeStatus.READY_FOR_REVIEW: frozenset({EpisodeStatus.PROCESSING, EpisodeStatus.QUALITY_FAILED}),
    EpisodeStatus.QUALITY_FAILED: frozenset({EpisodeStatus.PROCESSING}),
}

TERMINAL_TASK_STATES = frozenset({TaskState.SUCCEEDED, TaskState.FAILED, TaskState.CANCELLED})


def can_transition(machine: dict, current, target) -> bool:
    return target in machine.get(current, frozenset())


def assert_task_transition(current: TaskState, target: TaskState) -> None:
    if not can_transition(TASK_TRANSITIONS, current, target):
        raise ApiError(
            ErrorCode.TASK_ALREADY_TERMINAL,
            f"Illegal task transition {current.value} -> {target.value}",
            details={"from": current.value, "to": target.value},
        )


def assert_run_transition(current: RunState, target: RunState) -> None:
    if not can_transition(RUN_TRANSITIONS, current, target):
        raise ApiError(
            ErrorCode.RUN_ALREADY_TERMINAL,
            f"Illegal run transition {current.value} -> {target.value}",
            details={"from": current.value, "to": target.value},
        )
