"""PostgreSQL-backed proofs for the AI audit/reservation path (C4, C5).

Everything here needs a real server: a single SQLite connection cannot show that a
``FOR UPDATE`` lock serialises two workers, that a unique index turns a losing insert
into a violation instead of a duplicate, or that a reaper sees another transaction's
committed rows.

Run with ``TEST_DATABASE_URL`` pointing at a PostgreSQL database.
"""

from __future__ import annotations

import threading
from datetime import timedelta
from types import SimpleNamespace

import pytest
from sqlalchemy import select
from sqlalchemy.orm import Session

from embodied_dataops.ai.service import budget_state, reserve_budget
from embodied_dataops.config import Settings
from embodied_dataops.db.models import (
    AIClipInput,
    Asset,
    Episode,
    ModelInference,
    PipelineRun,
    SourceImport,
    Task,
)
from embodied_dataops.ids import new_id
from embodied_dataops.workflow.handlers import vlm_preannotate_clip as clip_module
from embodied_dataops.workflow.handlers.vlm_preannotate_clip import _get_or_create_inference
from embodied_dataops.workflow.outbox import utcnow
from embodied_dataops.workflow.reaper import reap_once

pytestmark = [pytest.mark.integration, pytest.mark.concurrency]

CLIP_ID = "c" * 64


def _seed(db: Session) -> tuple[PipelineRun, AIClipInput, Task]:
    """SourceImport -> Episode -> PipelineRun -> Asset -> AIClipInput -> Task."""
    source = SourceImport(
        id=new_id(),
        source_type="EGO_MP4",
        upload_state="READY",
        data_origin="synthetic",
        license="CC-BY-4.0",
        filename="fixture.mp4",
        size_bytes=0,
        sha256="a" * 64,
        mapping_json={},
        created_at=utcnow(),
    )
    db.add(source)
    db.flush()
    episode = Episode(
        id=new_id(),
        source_import_id=source.id,
        source_episode_index=0,
        episode_type="ego_video_episode",
        task_label="open the drawer",
        duration_ns=8_000_000_000,
        status="READY_FOR_REVIEW",
        canonical_version=1,
        canonical_sha256="b" * 64,
        quality_status="PASS",
        sample_count=0,
        data_origin="synthetic",
        canonical_schema_version="1.0",
        created_at=utcnow(),
    )
    db.add(episode)
    db.flush()
    run = PipelineRun(
        id=new_id(),
        episode_id=episode.id,
        pipeline_version="2.1.0",
        state="RUNNING",
        created_at=utcnow(),
    )
    db.add(run)
    db.flush()
    asset = Asset(
        id=new_id(),
        sha256="d" * 64,
        byte_size=1,
        media_type="application/json",
        object_key=f"test/{new_id()}",
        origin="derived",
        logical_name="frame-manifest",
        meta_json={},
        created_at=utcnow(),
    )
    db.add(asset)
    db.flush()
    clip = AIClipInput(
        id=new_id(),
        episode_id=episode.id,
        clip_id=CLIP_ID,
        start_ns=0,
        end_ns=8_000_000_000,
        source_asset_sha256="e" * 64,
        preprocessing_sha256="f" * 64,
        frame_manifest_asset_id=asset.id,
        frame_count=8,
        created_at=utcnow(),
    )
    db.add(clip)
    db.flush()
    task = Task(
        id=new_id(),
        run_id=run.id,
        step="VLM_PREANNOTATE_CLIP",
        unit_key=CLIP_ID,
        pool="vision",
        state="RUNNING",
        attempt=1,
        max_attempts=3,
        business_revision=1,
        output_json={},
        created_at=utcnow(),
        updated_at=utcnow(),
    )
    db.add(task)
    db.flush()
    return run, clip, task


def _inference(db: Session, *, run: PipelineRun, clip: AIClipInput, task: Task, model: str):
    row = ModelInference(
        id=new_id(),
        run_id=run.id,
        task_id=task.id,
        clip_input_id=clip.id,
        provider="replay",
        model_name=model,
        mode="REPLAY",
        prompt_version="ego-action-prompt/2.1",
        prompt_sha256="1" * 64,
        request_sha256="2" * 64,
        status="QUEUED",
    )
    db.add(row)
    db.flush()
    return row


# ---------------------------------------------------------------------------
# C5: two workers, one budget
# ---------------------------------------------------------------------------
def test_budget_admission_lets_exactly_one_worker_through(db, session_factory):
    run, clip, task = _seed(db)
    first = _inference(db, run=run, clip=clip, task=task, model="replay-a")
    second = _inference(db, run=run, clip=clip, task=task, model="replay-b")
    db.commit()
    first_id, second_id = first.id, second.id

    # 0.015 USD with a 0.01 estimate admits one call and refuses the second.
    settings = Settings(vlm_budget_usd_per_run=0.015, vlm_max_attempts=3)
    results: dict[str, object] = {}
    start = threading.Barrier(2)

    def worker(name: str, inference_id: str) -> None:
        session = session_factory()
        try:
            start.wait(timeout=10)
            inference = session.get(ModelInference, inference_id)
            admitted, state = reserve_budget(session, inference=inference, settings=settings)
            session.commit()
            results[name] = (admitted, state.running_reserved_usd)
        except Exception as exc:  # noqa: BLE001 - surfaced by the assertion below
            session.rollback()
            results[name] = exc
        finally:
            session.close()

    threads = [
        threading.Thread(target=worker, args=("a", first_id)),
        threading.Thread(target=worker, args=("b", second_id)),
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=30)
        assert not thread.is_alive(), "a worker blocked on the run lock past the timeout"

    admitted_flags = sorted(bool(results[key][0]) for key in ("a", "b") if not isinstance(results[key], Exception))
    assert admitted_flags == [False, True], results

    db.expire_all()
    refused = [row for row in (db.get(ModelInference, first_id), db.get(ModelInference, second_id)) if row.status == "FAILED"]
    accepted = [row for row in (db.get(ModelInference, first_id), db.get(ModelInference, second_id)) if row.status == "RUNNING"]
    assert len(refused) == 1 and len(accepted) == 1
    # The refusal must not leave a reservation behind.
    assert refused[0].estimated_cost_usd is None
    assert accepted[0].estimated_cost_usd is not None


def test_a_committed_reservation_is_visible_to_the_next_admission(db):
    run, clip, task = _seed(db)
    first = _inference(db, run=run, clip=clip, task=task, model="replay-a")
    db.commit()

    settings = Settings(vlm_budget_usd_per_run=0.015, vlm_max_attempts=3)
    admitted, _state = reserve_budget(db, inference=first, settings=settings)
    assert admitted is True
    db.commit()

    second = _inference(db, run=run, clip=clip, task=task, model="replay-b")
    db.commit()
    admitted_second, state = reserve_budget(db, inference=second, settings=settings)
    assert admitted_second is False
    # Exactly one call's worth is reserved, and the reader sees it.
    assert float(state.running_reserved_usd) == pytest.approx(
        float(first.estimated_cost_usd)
    )


# ---------------------------------------------------------------------------
# C5 follow-up: a killed worker must not hold the budget forever
# ---------------------------------------------------------------------------
def test_stale_running_inference_is_released_and_stops_counting(db, session_factory):
    run, clip, task = _seed(db)
    stale = _inference(db, run=run, clip=clip, task=task, model="replay-a")
    stale.status = "RUNNING"
    stale.estimated_cost_usd = 0.01
    stale.created_at = utcnow() - timedelta(hours=2)
    db.commit()

    settings = Settings(vlm_budget_usd_per_run=1.0)
    before = budget_state(db, run.id, settings)
    assert float(before.running_reserved_usd) == pytest.approx(0.01)

    now = utcnow()
    stats = reap_once(db, now=now, redis_client=None, streams=())
    assert stats["released_inferences"] == 1

    db.expire_all()
    row = db.get(ModelInference, stale.id)
    assert row.status == "FAILED"
    assert row.error_code == "STALE_RESERVATION"
    assert row.estimated_cost_usd is None

    after = budget_state(db, run.id, settings)
    assert float(after.running_reserved_usd) == pytest.approx(0.0)


def test_a_live_inference_is_not_reaped(db):
    run, clip, task = _seed(db)
    fresh = _inference(db, run=run, clip=clip, task=task, model="replay-a")
    fresh.status = "RUNNING"
    fresh.estimated_cost_usd = 0.01
    db.commit()

    stats = reap_once(db, now=utcnow(), redis_client=None, streams=())
    assert stats["released_inferences"] == 0
    db.expire_all()
    assert db.get(ModelInference, fresh.id).status == "RUNNING"


# ---------------------------------------------------------------------------
# C4 follow-up: losing the identity race must not fail the task
# ---------------------------------------------------------------------------
def test_losing_the_identity_race_reuses_the_committed_row(
    db, session_factory, monkeypatch
):
    run, clip, task = _seed(db)
    db.commit()
    run_ref = SimpleNamespace(id=run.id)
    clip_ref = SimpleNamespace(id=clip.id)

    winner_committed = threading.Event()
    loser_missed = threading.Event()
    real_find = clip_module._find_inference
    loser_thread_name = "loser"

    def patched_find(session, **kwargs):
        row = real_find(session, **kwargs)
        if row is None and threading.current_thread().name == loser_thread_name:
            # The loser has looked and seen nothing: hold it until the winner commits,
            # which is exactly the interleaving the unique index has to survive.
            loser_missed.set()
            winner_committed.wait(timeout=10)
        return row

    monkeypatch.setattr(clip_module, "_find_inference", patched_find)

    def call(session: Session) -> tuple[str, int]:
        inference, prior = _get_or_create_inference(
            session,
            run=run_ref,
            task_id=task.id,
            clip=clip_ref,
            provider_name="replay",
            model_name="replay",
            mode="REPLAY",
            prompt_sha256="9" * 64,
            prompt_version="ego-action-prompt/2.1",
            request_sha256="8" * 64,
        )
        session.commit()
        return inference.id, prior

    outcome: dict[str, object] = {}

    def loser() -> None:
        session = session_factory()
        try:
            outcome["loser"] = call(session)
        except Exception as exc:  # noqa: BLE001 - the assertion below reports it
            session.rollback()
            outcome["loser"] = exc
        finally:
            session.close()

    thread = threading.Thread(target=loser, name=loser_thread_name)
    thread.start()
    assert loser_missed.wait(timeout=10), "the loser never reached the insert phase"

    winner_session = session_factory()
    try:
        winner_id, _ = call(winner_session)
    finally:
        winner_session.close()
    winner_committed.set()
    thread.join(timeout=30)
    assert not thread.is_alive()

    assert not isinstance(outcome["loser"], Exception), outcome["loser"]
    loser_id, prior_attempts = outcome["loser"]
    assert loser_id == winner_id, "both workers must end up on one inference row"
    assert prior_attempts == 0

    rows = list(db.scalars(select(ModelInference).where(ModelInference.run_id == run.id)))
    assert len(rows) == 1
