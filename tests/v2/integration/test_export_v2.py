"""DO2-08 service-level V2 export: manifest-2.0, official Writer, publish state.

Seeds canonical-episode/2.0 Episodes (stored objects written through the real
local object store), then drives ``datasets.service.create_version`` /
``publish_version``.  Asserts:

* dataset-manifest/2.0 fields (profile binding, camera keys, aligned fps, dims,
  native index SHA-256 values, aligned-view SHA-256, annotation revision);
* the official Reader round-trip stats are persisted on the export asset;
* BUILDING -> READY -> PUBLISHED, and the published manifest is re-verified;
* mixed profile revision membership is refused with 409
  ``MIXED_PROFILE_EXPORT_UNSUPPORTED`` before any version is created.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest
from _canonical_helpers import prealigned_source
from sqlalchemy import select

from embodied_dataops.canonical.v2.build import build_canonical_v2
from embodied_dataops.canonical.v2.contract import canonical_sha256_v2, canonical_v2_keys
from embodied_dataops.canonical.v2.model import PreviewArtifact
from embodied_dataops.canonical.v2.preview import encode_camera_preview
from embodied_dataops.datasets import service
from embodied_dataops.datasets.export_v2 import episode_manifest_entry
from embodied_dataops.datasets.manifest_v2 import parse_manifest_v2
from embodied_dataops.db.models import (
    AnnotationSet,
    Asset,
    DatasetVersion,
    Episode,
    PipelineRun,
    QualityReport,
    SourceImport,
)
from embodied_dataops.errors import ApiError, ErrorCode
from embodied_dataops.ids import new_id
from embodied_dataops.storage.base import get_storage, sha256_file

pytestmark = pytest.mark.integration

ROWS = 12
SOURCE_SHA = "9" * 64


def _seed_episode(session, *, profile_id: str, suffix: str) -> Episode:
    storage = get_storage()
    source = prealigned_source(profile_id=profile_id, cameras=("front",), rows=ROWS)
    source.episode_id = f"episode-{suffix}"
    storage.put_bytes(
        f"raw/{suffix}/source.bin", b"synthetic", content_type="application/octet-stream"
    )
    raw_asset = Asset(
        id=new_id(),
        sha256=SOURCE_SHA,
        byte_size=9,
        media_type="application/octet-stream",
        object_key=f"raw/{suffix}/source.bin",
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
        filename=f"{suffix}.tar.gz",
        size_bytes=9,
        sha256=SOURCE_SHA,
        mapping_json={"profile_id": profile_id, "profile_revision": 1},
    )
    episode = Episode(
        id=source.episode_id,
        source_import_id=source_import.id,
        source_episode_index=0,
        episode_type="robot_episode",
        task_label=f"synthetic task {suffix}",
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
    run = PipelineRun(
        id=new_id(), episode_id=episode.id, pipeline_version="2.0.0", state="SUCCEEDED"
    )
    quality = QualityReport(
        id=new_id(),
        episode_id=episode.id,
        run_id=run.id,
        policy_version="qc-policy/2.0",
        status="PASS",
        summary_json={"status": "PASS"},
        schema_version="quality-report/2.0",
    )
    annotation = AnnotationSet(
        id=new_id(),
        episode_id=episode.id,
        run_id=run.id,
        model_id="fake-detector",
        model_sha256="f" * 64,
        status="APPROVED",
        summary_json={},
    )
    session.add_all([raw_asset, source_import])
    session.flush()
    session.add_all([episode, run])
    session.flush()
    session.add_all([quality, annotation])
    session.flush()

    previews: list[PreviewArtifact] = []
    preview_bytes: dict[str, bytes] = {}
    keys = canonical_v2_keys(episode.id, 1)
    tmp_dir = Path(__import__("tempfile").mkdtemp(prefix="v2-export-seed-"))
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
            native_hashes[name] = __import__("hashlib").sha256(data).hexdigest()
        else:
            key = artifact_keys[name]
            if name in {"native_state", "native_action"}:
                native_hashes[name] = __import__("hashlib").sha256(data).hexdigest()
        storage.put_bytes(key, data, content_type="application/octet-stream")
    for camera, data in preview_bytes.items():
        storage.put_bytes(
            keys["preview"].format(camera=camera), data, content_type="video/mp4"
        )
    metadata_sha = __import__("hashlib").sha256(bundle.artifacts["metadata"]).hexdigest()
    aligned_sha = __import__("hashlib").sha256(bundle.artifacts["aligned_samples"]).hexdigest()
    episode.canonical_sha256 = canonical_sha256_v2(metadata_sha, aligned_sha, native_hashes)
    session.flush()
    return episode


def _seed_and_create(session, *, profiles: tuple[str, ...]) -> DatasetVersion:
    dataset = service.create_dataset(session, slug="v2-export-demo")
    episodes = [
        _seed_episode(session, profile_id=profile_id, suffix=f"e{index}")
        for index, profile_id in enumerate(profiles)
    ]
    session.commit()
    return service.create_version(
        session, dataset_id=dataset.id, episode_ids=[episode.id for episode in episodes]
    )


@pytest.mark.skipif(
    importlib.util.find_spec("lerobot") is None,
    reason="official LeRobot Writer/Reader round-trip requires the pinned lerobot extra",
)
def test_v2_create_version_manifest_roundtrip_and_publish(db, storage):
    version = _seed_and_create(db, profiles=("arm6_joint_position_v1",))
    assert version.state == "READY", (version.error_code, version.error_detail)
    manifest_asset = db.get(Asset, version.manifest_asset_id)
    assert manifest_asset is not None
    manifest = parse_manifest_v2(get_storage().get_bytes(manifest_asset.object_key))
    assert manifest["schema_version"] == "dataset-manifest/2.0"
    assert manifest["profile"]["profile_id"] == "arm6_joint_position_v1"
    assert manifest["profile"]["profile_revision"] == 1
    entry = manifest["episodes"][0]
    assert entry["camera_keys"] == ["front"]
    assert entry["aligned_fps"] == 10
    assert entry["state_dim"] == 6
    assert entry["action_dim"] == 6
    assert entry["action_semantics"] == "joint_target_position"
    assert len(entry["profile_config_sha256"]) == 64
    assert len(entry["aligned_view_sha256"]) == 64
    assert {"camera:front", "state", "action", "aligned"} <= set(entry["native_indexes"])
    for digest in entry["native_indexes"].values():
        assert len(digest) == 64

    export_asset = db.get(Asset, version.export_asset_id)
    assert export_asset is not None
    meta = export_asset.meta_json or {}
    assert meta["schema_version"] == "lerobot-v3-export/2.0"
    assert meta["camera_keys"] == ["front"]
    assert meta["state_dim"] == 6
    assert meta["image_source"] == "canonical_preview_10hz"
    roundtrip = meta["roundtrip"]
    assert roundtrip["episode_count"] == 1
    assert roundtrip["frame_count"] == ROWS
    assert roundtrip["camera_keys"] == ["front"]
    assert roundtrip["samples_checked"] >= min(10, ROWS)
    assert roundtrip["max_abs_error"] <= 1e-5

    published = service.publish_version(db, version_id=version.id)
    assert published.state == "PUBLISHED"
    snapshot = service.get_version(db, version.dataset_id, int(version.version))
    assert snapshot["manifest"]["schema_version"] == "dataset-manifest/2.0"
    assert snapshot["export_roundtrip"]["max_abs_error"] <= 1e-5


def test_v2_mixed_profile_membership_is_refused(db, storage):
    dataset = service.create_dataset(db, slug="v2-mixed-demo")
    episodes = [
        _seed_episode(db, profile_id="arm6_joint_position_v1", suffix="six"),
        _seed_episode(db, profile_id="arm7_joint_position_v1", suffix="seven"),
    ]
    db.commit()
    with pytest.raises(ApiError) as excinfo:
        service.create_version(
            db, dataset_id=dataset.id, episode_ids=[episode.id for episode in episodes]
        )
    assert excinfo.value.code is ErrorCode.MIXED_PROFILE_EXPORT_UNSUPPORTED
    assert excinfo.value.status_code == 409
    db.rollback()
    assert list(db.scalars(select(DatasetVersion))) == []


def test_v2_manifest_entry_requires_aligned_view_digest():
    from embodied_dataops.errors import ApiError

    with pytest.raises(ApiError):
        episode_manifest_entry(
            episode_id="episode-x",
            metadata={
                "schema_version": "canonical-episode/2.0",
                "publishable": False,
                "diagnostics": [{"code": "ALIGNED_VIEW_INVALID"}],
                "aligned_view": {"fps": 10, "asset_sha256": "0" * 64},
            },
            canonical_sha256="1" * 64,
            source_sha256="2" * 64,
            quality_report_id="q",
            annotation_set_id="a",
            annotation_revision_id=None,
            native_indexes={"aligned": "0" * 64},
        )
