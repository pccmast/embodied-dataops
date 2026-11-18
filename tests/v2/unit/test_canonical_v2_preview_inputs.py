"""Preview/detection inputs for ``canonical-episode/2.0`` (BUILD_PREVIEW, DETECT_OBJECTS).

The bug: ``resolve_canonical_inputs`` only knew the 1.0 layout
(``canonical/{id}/v{n}/metadata.json`` + ``frames/front.mp4``).  A 2.0 Episode stores
its objects under ``canonical/{id}/v2/{n}/`` and has **no** thumbnail object, so both
handlers died with ``Canonical metadata object is missing for this run`` and
``GET /episodes/{id}/preview`` answered 409.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from embodied_dataops.canonical.v2.contract import (
    AlignedViewDescriptor,
    CanonicalMetadataV2,
    ModalityDescriptor,
    canonical_v2_keys,
)
from embodied_dataops.errors import ApiError, ErrorCode
from embodied_dataops.workflow.handlers import detect_objects

pytestmark = pytest.mark.no_db

EPISODE_ID = "ep-1"


def _modality(name: str, kind: str, shape: list[int], dtype: str) -> ModalityDescriptor:
    return ModalityDescriptor(
        name=name,
        kind=kind,
        dtype=dtype,
        shape=shape,
        source_clock_domain="lerobot_episode_relative",
        timestamp_provenance="source_rows",
        native_sample_count=1200,
        native_asset_sha256="c" * 64,
        alignment_method="source_row_aligned",
        alignment_tolerance_ns=0,
    )


def _metadata(camera_names: tuple[str, ...] = ("front", "wrist")) -> CanonicalMetadataV2:
    return CanonicalMetadataV2(
        episode_id=EPISODE_ID,
        source_asset_sha256="a" * 64,
        profile_id="utokyo_xarm_pick_and_place_v1",
        profile_revision=1,
        profile_config_sha256="b" * 64,
        clock_domains={"camera": "lerobot_episode_relative", "state": "lerobot_episode_relative"},
        modalities=[
            *[_modality(name, "camera", [480, 640, 3], "uint8") for name in camera_names],
            _modality("state", "state", [8], "float64"),
            _modality("action", "action", [7], "float64"),
        ],
        camera_names=list(camera_names),
        aligned_view=AlignedViewDescriptor(
            fps=10,
            interval_ns=100_000_000,
            sample_count=120,
            valid_sample_count=118,
            invalid_sample_count=2,
            state_dim=8,
            action_dim=7,
            camera_names=list(camera_names),
            alignment_method="source_row_aligned",
        ),
        state_dim=8,
        action_dim=7,
    )


class _FakeStorage:
    def __init__(self, objects: dict[str, bytes]) -> None:
        self.objects = objects

    def get_bytes(self, key: str) -> bytes:
        if key not in self.objects:
            raise ApiError(ErrorCode.OBJECT_NOT_FOUND, "missing", details={"key": key})
        return self.objects[key]

    def exists(self, key: str) -> bool:
        return key in self.objects


def _resolve(objects: dict[str, bytes]):
    ctx = SimpleNamespace(storage=_FakeStorage(objects))
    return detect_objects._resolve_v2_inputs(
        ctx,
        episode=SimpleNamespace(id=EPISODE_ID),
        run=SimpleNamespace(id="run-1"),
        output={},
        canonical_version=1,
    )


def test_v2_keys_keep_preview_and_thumbnail_under_the_same_version_prefix():
    keys = canonical_v2_keys(EPISODE_ID, 3)
    assert keys["preview"].startswith("canonical/ep-1/v2/3/preview/")
    assert keys["thumbnail"].startswith("canonical/ep-1/v2/3/preview/")
    assert keys["preview"].format(camera="front").endswith("front_10hz_640x480.mp4")


def test_v2_inputs_use_the_preview_video_and_aligned_sample_count():
    keys = canonical_v2_keys(EPISODE_ID, 1)
    preview_key = keys["preview"].format(camera="front")

    inputs = _resolve({keys["metadata"]: _metadata().canonical_json(), preview_key: b"video"})

    assert inputs.schema_version == "canonical-episode/2.0"
    assert inputs.video_key == preview_key
    assert inputs.samples_key == keys["aligned_samples"]
    assert inputs.thumbnail_key == keys["thumbnail"]
    assert inputs.metadata.camera_name == "front"
    assert inputs.metadata.sample_count == 120
    assert inputs.metadata.width == 640
    assert inputs.metadata.height == 480
    assert inputs.metadata.clock_domain == "lerobot_episode_relative"


def test_v2_inputs_prefer_front_over_declaration_order():
    """``front`` wins even when it is not the first camera in the metadata."""
    keys = canonical_v2_keys(EPISODE_ID, 1)
    preview_key = keys["preview"].format(camera="front")

    inputs = _resolve(
        {keys["metadata"]: _metadata(("wrist", "front")).canonical_json(), preview_key: b"video"}
    )

    assert inputs.metadata.camera_name == "front"
    assert inputs.video_key == preview_key


def test_missing_preview_video_is_object_not_found():
    keys = canonical_v2_keys(EPISODE_ID, 1)
    with pytest.raises(Exception) as excinfo:
        _resolve({keys["metadata"]: _metadata().canonical_json()})
    error = excinfo.value
    assert getattr(error, "code", None) in {ErrorCode.OBJECT_NOT_FOUND, "OBJECT_NOT_FOUND"}
