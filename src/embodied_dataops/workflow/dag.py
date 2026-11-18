"""DAG construction for import jobs and episode pipeline runs (SPEC 6.1).

Creating tasks, dependencies, state transitions and outbox rows happens in the
caller's PostgreSQL transaction.  No Redis call is made here.
"""

from __future__ import annotations

import hashlib
from collections.abc import Sequence

from sqlalchemy import select
from sqlalchemy.orm import Session

from embodied_dataops import PIPELINE_VERSION
from embodied_dataops.canonical.contract import canonical_json_bytes
from embodied_dataops.config import get_settings
from embodied_dataops.db.models import (
    Episode,
    ImportJob,
    PipelineRun,
    SourceImport,
    Task,
    TaskDependency,
)
from embodied_dataops.domain.enums import (
    PIPELINE_DAG,
    PIPELINE_DAG_BY_VERSION,
    STEP_POOL,
    EpisodeStatus,
    ImportJobState,
    RunState,
    Step,
    TaskState,
)
from embodied_dataops.ids import new_id
from embodied_dataops.workflow.outbox import enqueue_task_wakeup, utcnow

IMPORT_DAG: dict[Step, tuple[Step, ...]] = {
    Step.VALIDATE_SOURCE: (),
    Step.IMPORT_EPISODES: (Step.VALIDATE_SOURCE,),
}


def input_digest(step: Step, input_sha256: Sequence[str], config: dict | None = None) -> str:
    """sha256(pipeline_version + step + sorted input asset sha256 + config JSON)."""
    payload = {
        "pipeline_version": PIPELINE_VERSION,
        "step": step.value,
        "inputs": sorted(input_sha256),
        "config": config or {},
    }
    return hashlib.sha256(canonical_json_bytes(payload)).hexdigest()


#: Execution order used to materialise tasks.  Steps absent from a DAG are skipped, so
#: the V1/V2 order stays exactly as it was while the 2.1 DAG adds its branch in place.
_TOPOLOGICAL_ORDER: tuple[Step, ...] = (
    Step.NORMALIZE_EPISODE,
    Step.QUALITY_CHECK,
    Step.DETECT_OBJECTS,
    Step.EXTRACT_KEYFRAMES,
    Step.BUILD_PREVIEW,
    Step.MERGE_CANDIDATE_ANNOTATIONS,
    Step.VALIDATE_AI_OUTPUT,
    Step.READY_FOR_REVIEW,
)

#: Steps that are created at run time by their parent task, not by the DAG builder: the
#: clip count is unknown until EXTRACT_KEYFRAMES has read the canonical duration.
FAN_OUT_STEPS: tuple[Step, ...] = (Step.VLM_PREANNOTATE_CLIP,)


def _topological_steps(dag: dict[Step, tuple[Step, ...]] | None = None) -> list[Step]:
    active = dag if dag is not None else PIPELINE_DAG
    return [step for step in _TOPOLOGICAL_ORDER if step in active]


def downstream_closure(
    step: Step, dag: dict[Step, tuple[Step, ...]] | None = None
) -> list[Step]:
    """The step itself plus every transitive downstream step, in DAG order.

    ``dag`` defaults to the V2 graph so existing callers are unaffected; a 2.1 retry
    passes its own graph so the AI branch is not silently dropped.
    """
    active = dag if dag is not None else PIPELINE_DAG
    include = {step}
    changed = True
    while changed:
        changed = False
        for candidate, deps in active.items():
            if candidate not in include and any(dep in include for dep in deps):
                include.add(candidate)
                changed = True
    return [s for s in _topological_steps(active) if s in include]


def create_import_job_tasks(session: Session, source_import: SourceImport) -> ImportJob:
    """Create the import-level job and its VALIDATE_SOURCE/IMPORT_EPISODES tasks."""
    settings = get_settings()
    job = ImportJob(
        id=new_id(),
        import_id=source_import.id,
        state=ImportJobState.QUEUED.value,
        created_at=utcnow(),
    )
    session.add(job)
    session.flush()

    tasks: dict[Step, Task] = {}
    for step, deps in IMPORT_DAG.items():
        task = Task(
            id=new_id(),
            import_job_id=job.id,
            step=step.value,
            pool=STEP_POOL[step],
            state=TaskState.PENDING.value,
            attempt=0,
            max_attempts=settings.max_attempts,
            next_run_at=utcnow(),
            input_digest=input_digest(step, [source_import.sha256 or ""], {"import_id": source_import.id}),
        )
        session.add(task)
        session.flush()
        tasks[step] = task
        for dep in deps:
            session.add(TaskDependency(task_id=task.id, depends_on_task_id=tasks[dep].id))
        if not deps:
            enqueue_task_wakeup(session, task.id, task.pool)
    return job


def create_pipeline_run(
    session: Session,
    episode: Episode,
    *,
    requested_by: str | None = None,
    pipeline_version: str = PIPELINE_VERSION,
    from_step: Step | None = None,
    config: dict | None = None,
) -> PipelineRun:
    """Create a PipelineRun plus its task DAG.

    ``from_step`` supports POST /tasks/{id}/retry: a *new* run starts at the
    failed step and re-runs its downstream closure; the old run is untouched.
    """
    settings = get_settings()
    dag = PIPELINE_DAG_BY_VERSION.get(pipeline_version, PIPELINE_DAG)
    steps = [
        step
        for step in (
            downstream_closure(from_step, dag) if from_step else _topological_steps(dag)
        )
        if step in dag and step not in FAN_OUT_STEPS
    ]
    run = PipelineRun(
        id=new_id(),
        episode_id=episode.id,
        pipeline_version=pipeline_version,
        state=RunState.QUEUED.value,
        requested_by=requested_by,
        created_at=utcnow(),
    )
    session.add(run)
    session.flush()

    tasks: dict[Step, Task] = {}
    for step in steps:
        task = Task(
            id=new_id(),
            run_id=run.id,
            step=step.value,
            pool=STEP_POOL[step],
            state=TaskState.PENDING.value,
            attempt=0,
            max_attempts=settings.max_attempts,
            next_run_at=utcnow(),
            input_digest=input_digest(step, [episode.canonical_sha256 or "", episode.id], config),
        )
        session.add(task)
        session.flush()
        tasks[step] = task
        for dep in dag[step]:
            if dep in tasks:
                session.add(TaskDependency(task_id=task.id, depends_on_task_id=tasks[dep].id))
        # Only root steps are woken here; everything else is advanced by the completion
        # engine once its dependencies have actually succeeded.
        if not [dep for dep in dag[step] if dep in tasks]:
            enqueue_task_wakeup(session, task.id, task.pool)
    episode.status = EpisodeStatus.PROCESSING.value
    episode.updated_at = utcnow()
    return run


def tasks_for_run(session: Session, run_id: str) -> list[Task]:
    stmt = select(Task).where(Task.run_id == run_id).order_by(Task.created_at)
    return list(session.scalars(stmt))


def task_dependencies(session: Session, task_id: str) -> list[str]:
    stmt = select(TaskDependency.depends_on_task_id).where(TaskDependency.task_id == task_id)
    return [row for row in session.scalars(stmt)]


def downstream_task_ids(session: Session, task_id: str) -> list[str]:
    """All tasks transitively depending on ``task_id``."""
    found: set[str] = set()
    frontier = [task_id]
    while frontier:
        current = frontier.pop()
        stmt = select(TaskDependency.task_id).where(TaskDependency.depends_on_task_id == current)
        for child in session.scalars(stmt):
            if child not in found:
                found.add(child)
                frontier.append(child)
    return sorted(found)


def dependencies_satisfied(session: Session, task: Task) -> bool:
    deps = task_dependencies(session, task.id)
    if not deps:
        return True
    stmt = select(Task.state).where(Task.id.in_(deps))
    states = list(session.scalars(stmt))
    return len(states) == len(deps) and all(state == TaskState.SUCCEEDED.value for state in states)
