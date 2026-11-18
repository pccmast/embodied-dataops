"""DO2-17 acceptance: a PUBLISHED dataset version never drifts with new revisions.

Flow:

1. canonical-episode/2.0 Episode + approved ``annotation-revision/2.0`` revision 2
   (created by reviewer A, approved by reviewer B) -> ``create_version`` binds
   ``dataset_version_episodes.annotation_revision_id`` and freezes a
   ``dataset-manifest/2.0`` whose per-episode ``annotation_revision_id`` matches;
2. publish the version, capturing manifest bytes/SHA, canonical digest and the
   bound revision content SHA;
3. create + approve a NEW revision (3) on the same annotation set;
4. assert the PUBLISHED version row, manifest bytes/digest, canonical digest and
   bound revision are byte-identical (no drift), the explicit
   ``approved_revision_id`` pointer moved to revision 3 only, and that a new
   dataset version binds revision 3 (old versions keep the old approval).
"""

from __future__ import annotations

import hashlib
import importlib.util
import tempfile
from pathlib import Path

import pytest
from _annotation_fixtures import (
    detection,
    make_principal,
    revision_frame,
    seed_annotation_set,
)
from _canonical_helpers import prealigned_source

from embodied_dataops.annotations.revisions import (
    create_annotation_revision,
    decide_annotation_revision,
    load_revision_document,
    resolve_approved_revision_id,
)
from embodied_dataops.canonical.contract import canonical_json_bytes
from embodied_dataops.canonical.v2.build import build_canonical_v2
from embodied_dataops.canonical.v2.contract import canonical_sha256_v2, canonical_v2_keys
from embodied_dataops.canonical.v2.model import PreviewArtifact
from embodied_dataops.canonical.v2.preview import encode_camera_preview
from embodied_dataops.datasets import service
from embodied_dataops.datasets.manifest_v2 import parse_manifest_v2
from embodied_dataops.db.models import (
    AnnotationRevision,
    AnnotationSet,
    Asset,
    DatasetVersion,
    DatasetVersionEpisode,
    Episode,
    PipelineRun,
    QualityReport,
    SourceImport,
)
from embodied_dataops.ids import new_id
from embodied_dataops.storage.base import get_storage, sha256_file

pytestmark = pytest.mark.integration

ROWS = 12
SOURCE_SHA = "9" * 64


def _seed_v2_episode(
    session, storage, *, profile_id: str = "arm6_joint_position_v1"
) -> tuple[Episode, PipelineRun, AnnotationSet]:
    """Canonical-episode/2.0 Episode with a real approved AnnotationSet + artifact."""
    source = prealigned_source(profile_id=profile_id, cameras=("front",), rows=ROWS)
    source.episode_id = f"episode-{new_id()[:8]}"
    storage.put_bytes("raw/snapshot/source.bin", b"synthetic", content_type="application/octet-stream")
    raw_asset = Asset(
        id=new_id(),
        sha256=SOURCE_SHA,
        byte_size=9,
        media_type="application/octet-stream",
        object_key="raw/snapshot/source.bin",
        origin="raw",
        logical_name="raw source",
        meta_json={},
    )
    source_import = SourceImport(
        id=new_id(),
        source_type="LEROBOT_V3",
        upload_state="READY",
        asset_id=raw_asset.id,
        data_origin="synthetic",
        license="CC-BY-4.0",
        filename="snapshot.tar.gz",
        size_bytes=9,
        sha256=SOURCE_SHA,
        mapping_json={"profile_id": profile_id, "profile_revision": 1},
    )
    episode = Episode(
        id=source.episode_id,
        source_import_id=source_import.id,
        source_episode_index=0,
        episode_type="robot_episode",
        task_label="synthetic snapshot task",
        duration_ns=ROWS * 100_000_000,
        status="READY_FOR_REVIEW",
        canonical_version=1,
        quality_status="PASS",
        sample_count=ROWS,
        data_origin="synthetic",
        profile_id=profile_id,
        profile_revision=1,
        canonical_schema_version="2.0",
    )
    run = PipelineRun(id=new_id(), episode_id=episode.id, pipeline_version="2.0.0", state="SUCCEEDED")
    quality = QualityReport(
        id=new_id(),
        episode_id=episode.id,
        run_id=run.id,
        policy_version="qc-policy/2.0",
        status="PASS",
        summary_json={"status": "PASS"},
        schema_version="quality-report/2.0",
    )
    session.add_all([raw_asset, source_import])
    session.flush()
    session.add_all([episode, run])
    session.flush()
    session.add(quality)
    session.flush()

    keys = canonical_v2_keys(episode.id, 1)
    tmp_dir = Path(tempfile.mkdtemp(prefix="v2-snapshot-seed-"))
    previews: list[PreviewArtifact] = []
    preview_bytes: dict[str, bytes] = {}
    for camera in source.cameras:
        destination = tmp_dir / f"{camera.name}.mp4"
        encode_camera_preview(camera, destination=destination, origin_ns=0)
        data = destination.read_bytes()
        preview_bytes[camera.name] = data
        previews.append(
            PreviewArtifact(
                camera_name=camera.name,
                asset_key=keys["preview"].format(camera=camera.name),
                asset_sha256=sha256_file(destination),
                fps=10,
                width=640,
                height=480,
                frame_count=ROWS,
            )
        )
    bundle = build_canonical_v2(source, canonical_version=1, previews=previews)
    assert bundle.publishable is True, bundle.blocking_reasons
    artifact_keys = {
        "metadata": keys["metadata"],
        "mapping_report": keys["mapping_report"],
        "native_state": keys["native_state"],
        "native_action": keys["native_action"],
        "aligned_samples": keys["aligned_samples"],
    }
    native_hashes: dict[str, str] = {}
    for name, data in bundle.artifacts.items():
        if name.startswith("native_camera:"):
            camera = name.split(":", 1)[1]
            key = keys["native_camera"].format(camera=camera)
            native_hashes[name] = hashlib.sha256(data).hexdigest()
        else:
            key = artifact_keys[name]
            if name in {"native_state", "native_action"}:
                native_hashes[name] = hashlib.sha256(data).hexdigest()
        storage.put_bytes(key, data, content_type="application/octet-stream")
    for camera, data in preview_bytes.items():
        storage.put_bytes(keys["preview"].format(camera=camera), data, content_type="video/mp4")
    metadata_sha = hashlib.sha256(bundle.artifacts["metadata"]).hexdigest()
    aligned_sha = hashlib.sha256(bundle.artifacts["aligned_samples"]).hexdigest()
    episode.canonical_sha256 = canonical_sha256_v2(metadata_sha, aligned_sha, native_hashes)
    session.flush()

    annotation_set, _payload = seed_annotation_set(session, storage, episode, run, status="APPROVED")
    session.flush()
    return episode, run, annotation_set


def _first_revision(session, storage, annotation_set: AnnotationSet, reviewer_a, reviewer_b):
    created = create_annotation_revision(
        session,
        storage=storage,
        annotation_set_id=annotation_set.id,
        expected_revision_no=1,
        base_revision_no=1,
        frames=[
            revision_frame(
                0,
                detections=[
                    detection(
                        "human-0-000",
                        class_id=0,
                        class_name="person",
                        score=0.91,
                        xyxy=[20.0, 30.0, 170.0, 300.0],
                    )
                ],
            )
        ],
        note="first human correction before the dataset version is built",
        actor=reviewer_a,
    )
    session.flush()
    decide_annotation_revision(
        session,
        revision_id=created.revision.id,
        decision="APPROVED",
        note="reviewer B verified the corrected detection before publishing",
        reviewer=reviewer_b,
    )
    session.flush()
    return created.revision


@pytest.mark.skipif(
    importlib.util.find_spec("lerobot") is None,
    reason="publishing freezes a manifest built by the official LeRobot Writer",
)
def test_published_version_keeps_revision_pointer_manifest_and_digest(db, storage) -> None:
    reviewer_a, _raw_a = make_principal(db, "reviewer", name="snapshot-reviewer-a")
    reviewer_b, _raw_b = make_principal(db, "reviewer", name="snapshot-reviewer-b")
    episode, run, annotation_set = _seed_v2_episode(db, storage)
    db.commit()

    revision_2 = _first_revision(db, storage, annotation_set, reviewer_a, reviewer_b)
    db.commit()
    assert revision_2.revision_no == 2
    assert resolve_approved_revision_id(db, annotation_set.id) == revision_2.id
    assert db.get(AnnotationSet, annotation_set.id).status == "APPROVED"

    dataset = service.create_dataset(db, slug="v2-published-snapshot-demo")
    version = service.create_version(db, dataset_id=dataset.id, episode_ids=[episode.id])
    db.commit()
    assert version.state == "READY", (version.error_code, version.error_detail)

    snapshot_row = db.get(DatasetVersionEpisode, (version.id, episode.id))
    assert snapshot_row is not None
    assert snapshot_row.annotation_revision_id == revision_2.id
    canonical_digest = snapshot_row.canonical_digest
    assert len(canonical_digest) == 64
    manifest_sha = version.manifest_sha256
    manifest_asset = db.get(Asset, version.manifest_asset_id)
    manifest = parse_manifest_v2(get_storage().get_bytes(manifest_asset.object_key))
    assert manifest["schema_version"] == "dataset-manifest/2.0"
    assert manifest["episodes"][0]["annotation_revision_id"] == revision_2.id
    assert manifest["annotation_revision_id"] == revision_2.id
    revision_2_digest = revision_2.content_sha256
    revision_2_asset = db.get(Asset, revision_2.artifact_asset_id)
    assert revision_2_asset is not None
    revision_2_bytes = storage.get_bytes(revision_2_asset.object_key)

    published = service.publish_version(db, version_id=version.id)
    db.commit()
    assert published.state == "PUBLISHED"

    # A NEW revision (3) is created and explicitly approved after publication.
    revision_3 = create_annotation_revision(
        db,
        storage=storage,
        annotation_set_id=annotation_set.id,
        expected_revision_no=2,
        base_revision_no=2,
        frames=[revision_frame(10, detections=[])],
        note="post-publication correction of frame 10",
        actor=reviewer_a,
    )
    db.flush()
    decide_annotation_revision(
        db,
        revision_id=revision_3.revision.id,
        decision="APPROVED",
        note="reviewer B approved the post-publication correction",
        reviewer=reviewer_b,
    )
    db.commit()
    db.expire_all()

    # The explicit pointer moved (new content is the latest approved) ...
    assert resolve_approved_revision_id(db, annotation_set.id) == revision_3.revision.id
    assert revision_3.revision.revision_no == 3

    # ... but the PUBLISHED version is frozen: row, manifest and digests unchanged.
    frozen_row = db.get(DatasetVersionEpisode, (version.id, episode.id))
    assert frozen_row is not None
    assert frozen_row.annotation_revision_id == revision_2.id
    assert frozen_row.canonical_digest == canonical_digest
    frozen_version = db.get(DatasetVersion, version.id)
    assert frozen_version.state == "PUBLISHED"
    assert frozen_version.manifest_sha256 == manifest_sha
    frozen_manifest_bytes = get_storage().get_bytes(
        db.get(Asset, frozen_version.manifest_asset_id).object_key
    )
    assert frozen_manifest_bytes == get_storage().get_bytes(manifest_asset.object_key)
    frozen_manifest = parse_manifest_v2(frozen_manifest_bytes)
    assert frozen_manifest["episodes"][0]["annotation_revision_id"] == revision_2.id
    assert frozen_manifest["annotation_revision_id"] == revision_2.id

    # The bound revision content is immutable (same SHA and same stored bytes).
    frozen_revision = db.get(AnnotationRevision, revision_2.id)
    assert frozen_revision.state == "APPROVED"
    assert frozen_revision.content_sha256 == revision_2_digest
    assert storage.get_bytes(revision_2_asset.object_key) == revision_2_bytes
    document = load_revision_document(db, storage, frozen_revision)
    assert hashlib.sha256(canonical_json_bytes(document)).hexdigest() == revision_2_digest

    # The version read model exposes the frozen binding, not the latest approval.
    snapshot = service.get_version(db, version.dataset_id, int(version.version))
    assert snapshot["manifest_sha256"] == manifest_sha
    assert snapshot["state"] == "PUBLISHED"
    assert snapshot["episodes"][0]["annotation_revision_id"] == revision_2.id

    # A NEW dataset version binds the newly approved revision; the old one stays.
    next_version = service.create_version(db, dataset_id=dataset.id, episode_ids=[episode.id])
    db.commit()
    assert next_version.state == "READY", (next_version.error_code, next_version.error_detail)
    assert next_version.version == version.version + 1
    next_row = db.get(DatasetVersionEpisode, (next_version.id, episode.id))
    assert next_row is not None
    assert next_row.annotation_revision_id == revision_3.revision.id
    db.expire_all()
    still_frozen = db.get(DatasetVersionEpisode, (version.id, episode.id))
    assert still_frozen.annotation_revision_id == revision_2.id
    assert still_frozen.canonical_digest == canonical_digest
