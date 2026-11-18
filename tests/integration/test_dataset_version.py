"""Integration tests for dataset versions (AT-15, AT-16, AT-17, AT-18, AT-19).

These tests use the real PostgreSQL test database and the local object-storage
backend from ``tests/conftest.py``.  The happy-path export tests require the
official ``lerobot==0.4.4`` package; without it they skip with an explicit reason
(never ``xfail``) so the blocked acceptance item is visible in the report.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq
import pytest
from sqlalchemy import select

from embodied_dataops import SAMPLE_INTERVAL_NS
from embodied_dataops.api.auth import Principal, generate_api_key
from embodied_dataops.canonical.contract import (
    ROBOT_JOINT_NAMES,
    ROBOT_PROFILE,
    CanonicalMetadata,
    SampleRow,
    canonical_digest,
    canonical_keys,
    rows_to_robot_table,
)
from embodied_dataops.datasets import lerobot_writer
from embodied_dataops.datasets.manifest import manifest_bytes
from embodied_dataops.datasets.service import (
    create_dataset,
    create_version,
    get_version,
    modify_version_membership,
    publish_version,
)
from embodied_dataops.db.models import (
    AnnotationReview,
    AnnotationSet,
    ApiKey,
    Asset,
    Dataset,
    DatasetVersion,
    Episode,
    PipelineRun,
    QualityReport,
    SourceImport,
)
from embodied_dataops.domain.enums import Role
from embodied_dataops.errors import ApiError, ErrorCode
from embodied_dataops.feedback.service import (
    analyze_gaps,
    create_evaluation,
    list_collection_orders,
    orders_for_evaluation,
)
from embodied_dataops.ids import new_id
from embodied_dataops.storage.base import LocalStorage, sha256_file

pytestmark = pytest.mark.integration

pytest.importorskip("sqlalchemy")

FRAME_COUNT = 10


@pytest.fixture(scope="module")
def official_lerobot() -> None:
    if not lerobot_writer.lerobot_importable():
        pytest.skip(
            "official lerobot==0.4.4 is not importable in this environment; "
            "install the `lerobot` dependency extra to exercise the export/round-trip tests"
        )


@pytest.fixture()
def actor(db) -> Principal:
    raw_key, record = generate_api_key(Role.OPERATOR, name="fixture-operator")
    db.add(record)
    db.commit()
    return Principal(key_id=record.id, role=Role.OPERATOR, name=record.name)


def _write_video(path: Path, frame_count: int = FRAME_COUNT) -> Path:
    import cv2
    import numpy as np

    path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(path), cv2.VideoWriter_fourcc(*"mp4v"), 10.0, (640, 480)
    )
    assert writer.isOpened(), "OpenCV could not open a VideoWriter for the fixture"
    rng = np.random.default_rng(20261029)
    for index in range(frame_count):
        frame = np.full((480, 640, 3), (index * 11) % 255, dtype=np.uint8)
        frame[:, :, 2] = rng.integers(0, 128, size=(480, 640), dtype=np.uint8)
        writer.write(frame)
    writer.release()
    return path


def _canonical_episode(
    db,
    storage: LocalStorage,
    tmp_path: Path,
    *,
    episode_id: str,
    quality_status: str,
    annotation_status: str,
    note: str,
    license_value: str,
    episode_type: str = "robot_episode",
    task_label: str = "pick up the transparent cup",
    frame_count: int = FRAME_COUNT,
    profile_id: str | None = "arm6_joint_position_v1",
    profile_revision: int | None = 1,
) -> dict[str, Any]:
    source_bytes = f"source:{episode_id}".encode()
    source_sha = hashlib.sha256(source_bytes).hexdigest()
    source_asset = Asset(
        id=new_id(),
        sha256=source_sha,
        byte_size=len(source_bytes),
        media_type="application/gzip",
        object_key=f"raw/sha256/{source_sha}",
        origin="raw",
        logical_name="fixture-source.tar.gz",
        meta_json={"data_origin": "synthetic"},
    )
    db.add(source_asset)
    db.flush()
    source = SourceImport(
        id=new_id(),
        source_type="LEROBOT_V3",
        upload_state="READY",
        asset_id=source_asset.id,
        data_origin="synthetic",
        license=license_value,
        filename="fixture-source.tar.gz",
        size_bytes=len(source_bytes),
        sha256=source_sha,
        mapping_json={},
    )
    db.add(source)
    db.flush()
    episode = Episode(
        id=episode_id,
        source_import_id=source.id,
        source_episode_index=0,
        episode_type=episode_type,
        task_label=task_label,
        duration_ns=(frame_count - 1) * SAMPLE_INTERVAL_NS,
        status="READY_FOR_REVIEW",
        canonical_version=1,
        quality_status=quality_status,
        sample_count=frame_count if episode_type == "robot_episode" else 0,
        data_origin="synthetic",
        # A publishable robot episode states the profile revision behind its numbers;
        # publishing refuses a version whose profile evidence is missing (fail-closed).
        profile_id=profile_id,
        profile_revision=profile_revision,
    )
    db.add(episode)
    db.flush()

    run = PipelineRun(
        id=new_id(),
        episode_id=episode.id,
        pipeline_version="1.0.0",
        state="SUCCEEDED",
    )
    db.add(run)
    db.flush()
    report = QualityReport(
        id=new_id(),
        episode_id=episode.id,
        run_id=run.id,
        policy_version="qc-policy/1.0",
        status=quality_status,
        summary_json={"verdict": quality_status, "findings": []},
    )
    db.add(report)

    annotation = AnnotationSet(
        id=new_id(),
        episode_id=episode.id,
        run_id=run.id,
        model_id="yolo11n",
        model_sha256="d" * 64,
        status=annotation_status,
        summary_json={"frames": 2, "detections": 3},
    )
    db.add(annotation)
    db.flush()
    api_key = db.scalars(select(ApiKey).limit(1)).first()
    if api_key is None:
        raw_key, api_key = generate_api_key(Role.REVIEWER, name="fixture-reviewer")
        db.add(api_key)
        db.flush()
    if annotation_status in {"APPROVED", "REJECTED"}:
        db.add(
            AnnotationReview(
                id=new_id(),
                annotation_set_id=annotation.id,
                reviewer_key_id=api_key.id,
                decision=annotation_status,
                note=note,
            )
        )

    canonical_sha = "e" * 64
    if episode_type == "robot_episode":
        keys = canonical_keys(episode.id, 1)
        video_path = _write_video(tmp_path / f"{episode.id}.mp4", frame_count)
        storage.put_file(keys["video"], video_path, content_type="video/mp4")
        rows = [
            SampleRow(
                sample_index=index,
                t_ns=index * SAMPLE_INTERVAL_NS,
                frame_index=index,
                state=[0.25 * index + joint for joint in range(6)],
                action=[-0.5 * index + joint for joint in range(6)],
                state_source_t_ns=index * SAMPLE_INTERVAL_NS,
                action_source_t_ns=index * SAMPLE_INTERVAL_NS,
            )
            for index in range(frame_count)
        ]
        samples_path = tmp_path / f"{episode.id}.parquet"
        pq.write_table(rows_to_robot_table(rows), samples_path)
        storage.put_file(keys["samples"], samples_path, content_type="application/octet-stream")
        metadata = CanonicalMetadata(
            episode_id=episode.id,
            episode_type="robot_episode",
            data_origin="synthetic",
            source_type="LEROBOT_V3",
            source_asset_sha256=source_sha,
            source_episode_index=0,
            duration_ns=(frame_count - 1) * SAMPLE_INTERVAL_NS,
            clock_domain="lerobot_episode_relative",
            joint_names=ROBOT_JOINT_NAMES,
            state_dim=6,
            action_dim=6,
            task_label=task_label,
            robot_profile=ROBOT_PROFILE,
            source_license=license_value,
            sample_count=frame_count,
        )
        storage.put_bytes(keys["metadata"], metadata.canonical_json(), content_type="application/json")
        canonical_sha = canonical_digest(
            metadata,
            sha256_file(samples_path),
            sha256_file(video_path),
        )
        episode.canonical_sha256 = canonical_sha
    db.flush()
    return {
        "episode": episode,
        "quality_report": report,
        "annotation_set": annotation,
        "source": source,
        "canonical_sha256": canonical_sha,
    }


def _make_dataset(db, actor: Principal) -> Dataset:
    return create_dataset(
        db,
        slug=f"robot-grasp-{new_id()[:8]}",
        description="fixture dataset",
        actor=actor,
        request_id="req-fixture",
    )


def _episode(
    db,
    storage,
    tmp_path,
    *,
    quality_status="PASS",
    annotation_status="APPROVED",
    note="Reviewed and approved",
    license_value="CC-BY-4.0",
    episode_type="robot_episode",
    profile_id: str | None = "arm6_joint_position_v1",
    profile_revision: int | None = 1,
) -> str:
    spec = _canonical_episode(
        db,
        storage,
        tmp_path,
        episode_id=new_id(),
        quality_status=quality_status,
        annotation_status=annotation_status,
        note=note,
        license_value=license_value,
        episode_type=episode_type,
        profile_id=profile_id,
        profile_revision=profile_revision,
    )
    return spec["episode"].id


# ---------------------------------------------------------------------------
# dataset creation
# ---------------------------------------------------------------------------
def test_create_dataset_and_duplicate_slug_conflict(db, storage, actor) -> None:
    dataset = _make_dataset(db, actor)
    assert dataset.slug.startswith("robot-grasp-")
    with pytest.raises(ApiError) as excinfo:
        create_dataset(db, slug=dataset.slug, description="dupe", actor=actor, request_id="r2")
    assert excinfo.value.code is ErrorCode.CONFLICT
    assert excinfo.value.status_code == 409


# ---------------------------------------------------------------------------
# AT-15: QC FAIL cannot enter a publishable version
# ---------------------------------------------------------------------------
def test_qc_fail_episode_is_rejected_with_409(db, storage, actor, tmp_path) -> None:
    dataset = _make_dataset(db, actor)
    episode_id = _episode(db, storage, tmp_path, quality_status="FAIL", annotation_status="PENDING_REVIEW", note="")
    with pytest.raises(ApiError) as excinfo:
        create_version(
            db,
            dataset_id=dataset.id,
            episode_ids=[episode_id],
            description="should fail",
            actor=actor,
            request_id="req-qc",
        )
    assert excinfo.value.code is ErrorCode.QC_STATUS_NOT_ELIGIBLE
    assert excinfo.value.status_code == 409
    assert db.scalars(select(DatasetVersion)).all() == []


# ---------------------------------------------------------------------------
# AT-16: unapproved annotation cannot enter a version
# ---------------------------------------------------------------------------
def test_unapproved_annotation_is_rejected_with_409(db, storage, actor, tmp_path) -> None:
    dataset = _make_dataset(db, actor)
    episode_id = _episode(db, storage, tmp_path, annotation_status="PENDING_REVIEW", note="")
    with pytest.raises(ApiError) as excinfo:
        create_version(
            db, dataset_id=dataset.id, episode_ids=[episode_id], actor=actor, request_id="req-ann"
        )
    assert excinfo.value.code is ErrorCode.ANNOTATION_NOT_APPROVED
    assert excinfo.value.status_code == 409
    assert db.scalars(select(DatasetVersion)).all() == []


def test_warn_release_requires_long_review_note(db, storage, actor, tmp_path) -> None:
    dataset = _make_dataset(db, actor)
    episode_id = _episode(
        db,
        storage,
        tmp_path,
        quality_status="WARN",
        annotation_status="APPROVED",
        note="short",
    )
    with pytest.raises(ApiError) as excinfo:
        create_version(db, dataset_id=dataset.id, episode_ids=[episode_id], actor=actor)
    assert excinfo.value.code is ErrorCode.ANNOTATION_NOT_APPROVED
    assert excinfo.value.status_code == 409


def test_ego_episode_is_incompatible(db, storage, actor, tmp_path) -> None:
    dataset = _make_dataset(db, actor)
    episode_id = _episode(
        db,
        storage,
        tmp_path,
        episode_type="ego_video_episode",
        annotation_status="APPROVED",
    )
    with pytest.raises(ApiError) as excinfo:
        create_version(db, dataset_id=dataset.id, episode_ids=[episode_id], actor=actor)
    assert excinfo.value.code is ErrorCode.INCOMPATIBLE_EPISODE
    assert excinfo.value.status_code == 409


def test_unknown_license_is_rejected(db, storage, actor, tmp_path) -> None:
    dataset = _make_dataset(db, actor)
    episode_id = _episode(db, storage, tmp_path, annotation_status="APPROVED", license_value="")
    with pytest.raises(ApiError) as excinfo:
        create_version(db, dataset_id=dataset.id, episode_ids=[episode_id], actor=actor)
    assert excinfo.value.code is ErrorCode.LICENSE_UNKNOWN
    assert excinfo.value.status_code == 409


def test_empty_membership_is_rejected(db, storage, actor) -> None:
    dataset = _make_dataset(db, actor)
    with pytest.raises(ApiError) as excinfo:
        create_version(db, dataset_id=dataset.id, episode_ids=[], actor=actor)
    assert excinfo.value.code is ErrorCode.EMPTY_VERSION


def test_missing_dataset_is_not_found(db, storage, actor) -> None:
    with pytest.raises(ApiError) as excinfo:
        create_version(db, dataset_id=new_id(), episode_ids=[new_id()], actor=actor)
    assert excinfo.value.code is ErrorCode.NOT_FOUND


# ---------------------------------------------------------------------------
# AT-17 (through the service): BUILDING -> READY with verified round-trip
# ---------------------------------------------------------------------------
def test_create_version_builds_ready_version_via_official_roundtrip(
    official_lerobot, db, storage, actor, tmp_path
) -> None:
    dataset = _make_dataset(db, actor)
    first = _episode(db, storage, tmp_path, annotation_status="APPROVED")
    second = _episode(db, storage, tmp_path, annotation_status="APPROVED")

    version = create_version(
        db,
        dataset_id=dataset.id,
        episode_ids=[second, first],  # order is normalised by episode_id
        description="v1",
        actor=actor,
        request_id="req-v1",
    )
    assert version.state == "READY"
    assert version.error_code is None
    assert version.manifest_sha256 and len(version.manifest_sha256) == 64
    assert version.manifest_asset_id
    assert version.export_asset_id

    detail = get_version(db, dataset.id, version.version)
    assert detail["state"] == "READY"
    assert detail["manifest"]["schema_version"] == "dataset-manifest/1.0"
    assert [entry["episode_id"] for entry in detail["manifest"]["episodes"]] == sorted([first, second])
    assert detail["export_roundtrip"]["episode_count"] == 2
    assert detail["export_roundtrip"]["frame_count"] == 2 * FRAME_COUNT
    assert detail["export_roundtrip"]["max_abs_error"] <= 1e-5
    assert detail["export_roundtrip"]["max_timestamp_error"] <= 1e-4
    assert detail["export_roundtrip"]["package_version"] == "0.4.4"
    assert detail["export_url"]
    assert detail["manifest_url"]
    assert detail["manifest_sha256"] == version.manifest_sha256


def test_publish_requires_ready_then_becomes_immutable(
    official_lerobot, db, storage, actor, tmp_path
) -> None:
    dataset = _make_dataset(db, actor)
    episode_id = _episode(db, storage, tmp_path, annotation_status="APPROVED")
    version = create_version(
        db, dataset_id=dataset.id, episode_ids=[episode_id], actor=actor, request_id="req-pub"
    )
    assert version.state == "READY"
    published = publish_version(db, version_id=version.id, actor=actor, request_id="req-pub")
    assert published.state == "PUBLISHED"
    assert published.published_at is not None

    # Second publish is refused because PUBLISHED is terminal.
    with pytest.raises(ApiError) as excinfo:
        publish_version(db, version_id=version.id, actor=actor)
    assert excinfo.value.code is ErrorCode.VERSION_IMMUTABLE
    assert excinfo.value.status_code == 409


def test_publish_reverifies_manifest_sha_and_refuses_tampering(
    official_lerobot, db, storage, actor, tmp_path
) -> None:
    dataset = _make_dataset(db, actor)
    episode_id = _episode(db, storage, tmp_path, annotation_status="APPROVED")
    version = create_version(
        db, dataset_id=dataset.id, episode_ids=[episode_id], actor=actor, request_id="req-tamper"
    )
    manifest_asset = db.get(Asset, version.manifest_asset_id)
    assert manifest_asset is not None
    storage.put_bytes(manifest_asset.object_key, b'{"tampered":true}')
    with pytest.raises(ApiError) as excinfo:
        publish_version(db, version_id=version.id, actor=actor)
    assert excinfo.value.code is ErrorCode.VERSION_NOT_READY
    assert version.state == "READY"


# ---------------------------------------------------------------------------
# AT-18: PUBLISHED membership changes are refused and the manifest hash is stable
# ---------------------------------------------------------------------------
def test_publish_refuses_manifest_without_publish_level(
    official_lerobot, db, storage, actor, tmp_path
) -> None:
    """A manifest that does not freeze a publish level must not publish (fail-closed).

    ``dataset-manifest/1.0`` used to omit ``profile`` entirely, and the publish gate
    read that as "nothing to check" and skipped ``assert_publishable``.
    """
    dataset = _make_dataset(db, actor)
    episode_id = _episode(db, storage, tmp_path, annotation_status="APPROVED")
    version = create_version(
        db, dataset_id=dataset.id, episode_ids=[episode_id], actor=actor, request_id="req-level"
    )
    assert version.state == "READY"

    manifest_asset = db.get(Asset, version.manifest_asset_id)
    assert manifest_asset is not None
    frozen = json.loads(storage.get_bytes(manifest_asset.object_key))
    assert frozen["profile"]["publish_level"] == "SEMANTIC_VERIFIED"

    legacy = {key: value for key, value in frozen.items() if key != "profile"}
    body = manifest_bytes(legacy)
    new_sha = hashlib.sha256(body).hexdigest()
    storage.put_bytes(manifest_asset.object_key, body)
    manifest_asset.sha256 = new_sha
    manifest_asset.byte_size = len(body)
    version.manifest_sha256 = new_sha
    db.flush()

    with pytest.raises(ApiError) as excinfo:
        publish_version(db, version_id=version.id, actor=actor, request_id="req-level")
    assert excinfo.value.code is ErrorCode.PUBLISH_LEVEL_CONFLICT
    assert version.state == "READY"


def test_format_verified_profile_cannot_publish_as_robot_dataset(
    official_lerobot, db, storage, actor, tmp_path
) -> None:
    """utokyo-style evidence (FORMAT_VERIFIED) may not ship as a training dataset."""
    dataset = _make_dataset(db, actor)
    episode_id = _episode(
        db,
        storage,
        tmp_path,
        annotation_status="APPROVED",
        profile_id="utokyo_xarm_pick_and_place_v1",
        profile_revision=1,
    )
    version = create_version(
        db, dataset_id=dataset.id, episode_ids=[episode_id], actor=actor, request_id="req-fv"
    )
    manifest_asset = db.get(Asset, version.manifest_asset_id)
    manifest = json.loads(storage.get_bytes(manifest_asset.object_key))
    assert manifest["profile"]["publish_level"] == "FORMAT_VERIFIED"

    with pytest.raises(ApiError) as excinfo:
        publish_version(db, version_id=version.id, actor=actor, request_id="req-fv")
    assert excinfo.value.code is ErrorCode.PUBLISH_LEVEL_CONFLICT


def test_published_version_membership_is_immutable(
    official_lerobot, db, storage, actor, tmp_path
) -> None:
    dataset = _make_dataset(db, actor)
    first = _episode(db, storage, tmp_path, annotation_status="APPROVED")
    second = _episode(db, storage, tmp_path, annotation_status="APPROVED")
    version = create_version(
        db, dataset_id=dataset.id, episode_ids=[first], actor=actor, request_id="req-imm"
    )
    publish_version(db, version_id=version.id, actor=actor, request_id="req-imm")
    hash_before = version.manifest_sha256
    manifest_asset = db.get(Asset, version.manifest_asset_id)
    bytes_before = storage.get_bytes(manifest_asset.object_key)

    with pytest.raises(ApiError) as excinfo:
        modify_version_membership(
            db,
            version_id=version.id,
            add_episode_ids=[second],
            actor=actor,
            request_id="req-imm-add",
        )
    assert excinfo.value.code is ErrorCode.VERSION_IMMUTABLE
    assert excinfo.value.status_code == 409

    db.refresh(version)
    assert version.state == "PUBLISHED"
    assert version.manifest_sha256 == hash_before
    assert storage.get_bytes(manifest_asset.object_key) == bytes_before
    assert hashlib.sha256(bytes_before).hexdigest() == hash_before


def test_second_version_increments_number(
    official_lerobot, db, storage, actor, tmp_path
) -> None:
    dataset = _make_dataset(db, actor)
    first = _episode(db, storage, tmp_path, annotation_status="APPROVED")
    second = _episode(db, storage, tmp_path, annotation_status="APPROVED")
    v1 = create_version(db, dataset_id=dataset.id, episode_ids=[first], actor=actor, request_id="v1")
    v2 = create_version(db, dataset_id=dataset.id, episode_ids=[second], actor=actor, request_id="v2")
    assert (v1.version, v2.version) == (1, 2)
    with pytest.raises(ApiError) as excinfo:
        get_version(db, dataset.id, 99)
    assert excinfo.value.code is ErrorCode.NOT_FOUND


# ---------------------------------------------------------------------------
# AT-19: F09 feedback analysis is idempotent and produces exactly one order
# ---------------------------------------------------------------------------
F09_PAYLOAD = {
    "schema_version": "evaluation/1.0",
    "model_id": "demo-policy-v1",
    "results": [
        {"scenario_key": "transparent_cup_low_light", "trials": 30, "successes": 12},
        {"scenario_key": "opaque_cup_normal_light", "trials": 30, "successes": 27},
    ],
    "is_simulated": True,
}


def _ready_dataset_version(db, storage, actor, tmp_path) -> DatasetVersion:
    """A minimal READY version row (feedback tests do not need the exported bytes)."""
    dataset = _make_dataset(db, actor)
    _episode(db, storage, tmp_path, annotation_status="APPROVED")
    version = DatasetVersion(
        id=new_id(),
        dataset_id=dataset.id,
        version=1,
        state="READY",
        description="feedback fixture",
        created_by=actor.key_id,
        manifest_sha256="f" * 64,
    )
    db.add(version)
    db.flush()
    return version


def test_f09_gap_analysis_is_idempotent(db, storage, actor, tmp_path) -> None:
    version = _ready_dataset_version(db, storage, actor, tmp_path)
    report = create_evaluation(
        db,
        dataset_version_id=version.id,
        payload=F09_PAYLOAD,
        actor=actor,
        request_id="req-eval-create",
    )
    first = analyze_gaps(
        db,
        evaluation_id=report.id,
        success_rate_threshold=0.8,
        min_trials=20,
        actor=actor,
        request_id="req-gap-1",
    )
    second = analyze_gaps(
        db,
        evaluation_id=report.id,
        success_rate_threshold=0.8,
        min_trials=20,
        actor=actor,
        request_id="req-gap-2",
    )
    assert first.id == second.id
    assert first.state == "SUCCEEDED"
    assert first.summary_json["order_count"] == 1

    orders = orders_for_evaluation(db, report.id)
    assert len(orders) == 1
    order = orders[0]
    assert order.scenario_key == "transparent_cup_low_light"
    assert order.requested_count == 36
    assert "40.0%" in order.reason and "80.0%" in order.reason
    assert order.state == "PROPOSED"  # V2 lifecycle (SUPERSEDED_BY_DO2-28)

    listed = list_collection_orders(db, limit=20, cursor=None, filters={})
    assert len(listed["data"]) == 1
    assert listed["data"][0]["requested_count"] == 36


def test_evaluation_invalid_scenario_key_returns_422(db, storage, actor, tmp_path) -> None:
    version = _ready_dataset_version(db, storage, actor, tmp_path)
    with pytest.raises(ApiError) as excinfo:
        create_evaluation(
            db,
            dataset_version_id=version.id,
            payload={"schema_version": "evaluation/1.0", "results": [{"scenario_key": "Bad-Key", "trials": 20, "successes": 1}], "is_simulated": True},
            actor=actor,
        )
    assert excinfo.value.code is ErrorCode.SCENARIO_KEY_INVALID
    assert excinfo.value.status_code == 422


def test_evaluation_for_missing_dataset_version_is_404(db, storage, actor) -> None:
    with pytest.raises(ApiError) as excinfo:
        create_evaluation(
            db,
            dataset_version_id=new_id(),
            payload=F09_PAYLOAD,
            actor=actor,
        )
    assert excinfo.value.code is ErrorCode.NOT_FOUND


# ---------------------------------------------------------------------------
# HTTP acceptance (AT-15 / AT-18 through the real FastAPI routes)
# ---------------------------------------------------------------------------
def _assert_error_envelope(response, code: str, status_code: int) -> None:
    assert response.status_code == status_code, response.text
    body = response.json()
    assert body["error"]["code"] == code, body
    assert body["request_id"]


def test_http_qc_fail_version_request_returns_409(
    db, storage, client, operator_headers, tmp_path
) -> None:
    created = client.post(
        "/api/v1/datasets",
        json={"slug": "http-qc-fail", "description": "AT-15"},
        headers=operator_headers,
    )
    assert created.status_code == 201, created.text
    dataset_id = created.json()["data"]["dataset_id"]
    episode_id = _episode(db, storage, tmp_path, quality_status="FAIL", annotation_status="PENDING_REVIEW", note="")
    db.commit()

    response = client.post(
        f"/api/v1/datasets/{dataset_id}/versions",
        json={"episode_ids": [episode_id], "description": ""},
        headers=operator_headers,
    )
    _assert_error_envelope(response, "QC_STATUS_NOT_ELIGIBLE", 409)


def test_http_duplicate_dataset_slug_returns_409(db, storage, client, operator_headers) -> None:
    headers = {**operator_headers, "Idempotency-Key": "dup-slug-key"}
    first = client.post(
        "/api/v1/datasets", json={"slug": "http-dup", "description": "a"}, headers=headers
    )
    assert first.status_code == 201, first.text
    second = client.post(
        "/api/v1/datasets",
        json={"slug": "http-dup", "description": "b"},
        headers={**operator_headers, "Idempotency-Key": "dup-slug-key-2"},
    )
    _assert_error_envelope(second, "CONFLICT", 409)


def test_http_publish_then_membership_mutation_returns_409(
    official_lerobot, db, storage, client, operator_headers, reviewer_headers, tmp_path
) -> None:
    created = client.post(
        "/api/v1/datasets",
        json={"slug": "http-immutable", "description": "AT-18"},
        headers=operator_headers,
    )
    assert created.status_code == 201, created.text
    dataset_id = created.json()["data"]["dataset_id"]
    first = _episode(db, storage, tmp_path, annotation_status="APPROVED")
    second = _episode(db, storage, tmp_path, annotation_status="APPROVED")
    db.commit()

    version_response = client.post(
        f"/api/v1/datasets/{dataset_id}/versions",
        json={"episode_ids": [first], "description": "v1"},
        headers=operator_headers,
    )
    assert version_response.status_code == 202, version_response.text
    version_body = version_response.json()["data"]
    version_id = version_body["version_id"]
    assert version_body["state"] == "READY"

    published = client.post(
        f"/api/v1/dataset-versions/{version_id}/publish", headers=reviewer_headers
    )
    assert published.status_code == 200, published.text
    assert published.json()["data"]["state"] == "PUBLISHED"
    manifest_hash = published.json()["data"]["manifest_sha256"]

    mutated = client.post(
        f"/api/v1/datasets/{dataset_id}/versions/1/membership",
        json={"add_episode_ids": [second], "remove_episode_ids": []},
        headers=operator_headers,
    )
    _assert_error_envelope(mutated, "VERSION_IMMUTABLE", 409)

    detail = client.get(
        f"/api/v1/datasets/{dataset_id}/versions/1", headers=operator_headers
    )
    assert detail.status_code == 200, detail.text
    assert detail.json()["data"]["state"] == "PUBLISHED"
    assert detail.json()["data"]["manifest_sha256"] == manifest_hash
    assert [entry["episode_id"] for entry in detail.json()["data"]["manifest"]["episodes"]] == [first]
