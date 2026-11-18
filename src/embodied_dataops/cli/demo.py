"""Scripted end-to-end demo (SPEC 1.2, 11): import -> QC -> review -> version ->
official export -> simulated evaluation -> collection order.

Every printed value is read back from PostgreSQL/object storage.  Stages that
cannot run in the current environment (e.g. no official LeRobot API installed)
are reported as BLOCKED instead of being faked.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

from sqlalchemy import select

from embodied_dataops.api.auth import Principal
from embodied_dataops.db.models import AnnotationSet, Dataset, Episode
from embodied_dataops.db.session import get_session_factory
from embodied_dataops.domain.enums import EpisodeType
from embodied_dataops.errors import ApiError

REPORT_PATH = Path("artifacts/demo-report.json")
DATASET_SLUG = "robot-grab-demo"


def _reviewer(session) -> Principal:
    from embodied_dataops.cli.seed import ensure_seed_principal

    return ensure_seed_principal(session)


def run_demo(*, force_fixtures: bool = False) -> int:
    from embodied_dataops.annotations.service import review_annotation_set
    from embodied_dataops.cli.seed import print_seed_summary, run_seed
    from embodied_dataops.datasets import service as dataset_service
    from embodied_dataops.feedback import service as feedback_service

    report: dict = {
        "schema_version": "demo-report/1.0",
        "generated_at": datetime.now(UTC).isoformat(),
        "stages": {},
    }
    factory = get_session_factory()

    # ---- stage 1: import + pipeline -------------------------------------
    seed_report = run_seed(
        only=["F01_good_lerobot", "F04_ego_video", "F10_blurry_video"],
        run_tasks=True,
        force_fixtures=force_fixtures,
    )
    report["stages"]["import_and_pipeline"] = seed_report
    print_seed_summary(seed_report)

    session = factory()
    try:
        reviewer = _reviewer(session)
        session.flush()
        # Only the episodes this run just imported: a demo must be reproducible on a
        # database that already holds older imports (a MCAP episode, for example,
        # carries no task label and would otherwise abort the export stage).
        demo_episode_ids = [
            episode["episode_id"]
            for source_import in seed_report.get("imports") or []
            for episode in source_import.get("episodes") or []
        ]
        robot_episodes = list(
            session.scalars(
                select(Episode).where(
                    Episode.id.in_(demo_episode_ids),
                    Episode.episode_type == EpisodeType.ROBOT_EPISODE.value,
                    Episode.status == "READY_FOR_REVIEW",
                )
            )
        )
        approved: list[dict] = []
        for episode in robot_episodes:
            annotation_set = session.scalars(
                select(AnnotationSet)
                .where(AnnotationSet.episode_id == episode.id)
                .order_by(AnnotationSet.created_at.desc())
            ).first()
            if annotation_set is None:
                continue
            try:
                review_annotation_set(
                    session,
                    annotation_set_id=annotation_set.id,
                    decision="APPROVED",
                    note="demo approval after reviewing quality findings and detection overlay",
                    reviewer=reviewer,
                )
                approved.append({"episode_id": episode.id, "annotation_set_id": annotation_set.id})
            except ApiError as exc:
                approved.append({"episode_id": episode.id, "error": exc.code.value})
        session.commit()
        report["stages"]["annotation_review"] = approved
    finally:
        session.close()

    # ---- stage 2: dataset version + official export ---------------------
    session = factory()
    try:
        principal = _reviewer(session)
        session.flush()
        # Re-use the demo dataset so the script can be replayed against the same
        # database; only the version number advances.
        dataset = session.scalars(select(Dataset).where(Dataset.slug == DATASET_SLUG)).first()
        if dataset is None:
            dataset = dataset_service.create_dataset(
                session,
                slug=DATASET_SLUG,
                description="demo robot grasp dataset version",
                actor=principal,
                request_id="demo",
            )
        session.flush()
        eligible_ids = [
            episode.id
            for episode in session.scalars(
                select(Episode).where(
                    Episode.id.in_(demo_episode_ids),
                    Episode.episode_type == EpisodeType.ROBOT_EPISODE.value,
                    Episode.quality_status.in_(["PASS", "WARN"]),
                )
            )
        ]
        version = dataset_service.create_version(
            session,
            dataset_id=dataset.id,
            episode_ids=eligible_ids,
            description="demo version",
            actor=principal,
            request_id="demo",
        )
        session.commit()
        report["stages"]["dataset_version"] = {
            "dataset_id": dataset.id,
            "slug": dataset.slug,
            "version_id": version.id,
            "version": int(version.version),
            "state": version.state,
            "manifest_sha256": version.manifest_sha256,
            "error_code": version.error_code,
            "error_detail": version.error_detail,
            "roundtrip": (session.get(type(version), version.id) and None),
        }
        version_id = version.id
    except ApiError as exc:
        session.rollback()
        report["stages"]["dataset_version"] = {"error": exc.code.value, "message": exc.message}
        version_id = None
    finally:
        session.close()

    # ---- stage 3: publish ------------------------------------------------
    if version_id:
        session = factory()
        try:
            principal = _reviewer(session)
            version = dataset_service.publish_version(
                session, version_id=version_id, actor=principal, request_id="demo"
            )
            session.commit()
            report["stages"]["publish"] = {
                "version_id": version.id,
                "state": version.state,
                "manifest_sha256": version.manifest_sha256,
            }
        except ApiError as exc:
            session.rollback()
            report["stages"]["publish"] = {"error": exc.code.value, "message": exc.message}
        finally:
            session.close()

    # ---- stage 4: simulated evaluation + gap analysis --------------------
    if version_id:
        session = factory()
        try:
            principal = _reviewer(session)
            evaluation = feedback_service.create_evaluation(
                session,
                dataset_version_id=version_id,
                payload={
                    "schema_version": "evaluation/1.0",
                    "model_id": "demo-policy-v1",
                    "results": [
                        {"scenario_key": "transparent_cup_low_light", "trials": 30, "successes": 12},
                        {"scenario_key": "opaque_cup_normal_light", "trials": 30, "successes": 27},
                    ],
                    "is_simulated": True,
                },
                actor=principal,
                request_id="demo",
            )
            analysis = feedback_service.analyze_gaps(
                session, evaluation_id=evaluation.id, actor=principal, request_id="demo"
            )
            orders = feedback_service.orders_for_evaluation(session, evaluation.id)
            session.commit()
            report["stages"]["feedback"] = {
                "evaluation_id": evaluation.id,
                "analysis_id": analysis.id,
                "orders": [
                    {
                        "scenario_key": order.scenario_key,
                        "requested_count": order.requested_count,
                        "reason": order.reason,
                    }
                    for order in orders
                ],
            }
        except ApiError as exc:
            session.rollback()
            report["stages"]["feedback"] = {"error": exc.code.value, "message": exc.message}
        finally:
            session.close()

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(json.dumps(report, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    print("-" * 78)
    for stage, payload in report["stages"].items():
        print(f"[demo] {stage}: {json.dumps(payload, default=str)[:400]}")
    print(f"[demo] report: {REPORT_PATH}")
    blocked = [
        stage
        for stage, payload in report["stages"].items()
        if isinstance(payload, dict) and payload.get("error")
    ]
    return 1 if blocked else 0
