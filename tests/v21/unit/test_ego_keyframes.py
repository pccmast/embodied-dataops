"""Ego videos must reach the keyframe planner without inventing robot channels (C2).

The audit found the AI branch was canonical-v2-only: ``CanonicalMetadataV2`` rejects
``episode_type != robot_episode``, while an Ego import is canonical v1 by contract.
Adding a video file alone could not fix it, and fabricating 6-D state/action to satisfy
the v2 validator would violate SPEC_V2.1 §0.1.  The resolver therefore picks the
recorded video of whichever canonical generation the Episode actually has.
"""

from __future__ import annotations

import json

import pytest

from embodied_dataops.canonical.contract import canonical_keys
from embodied_dataops.canonical.v2.contract import canonical_v2_keys
from embodied_dataops.domain.enums import EpisodeType
from embodied_dataops.errors import ErrorCode, TerminalTaskError
from embodied_dataops.workflow.handlers.extract_keyframes import _resolve_video_source

pytestmark = pytest.mark.no_db

EPISODE_ID = "ep-ego-1"
DURATION_NS = 12_000_000_000


class _Storage:
    def __init__(self, objects: dict[str, bytes]) -> None:
        self.objects = objects

    def exists(self, key: str) -> bool:
        return key in self.objects

    def get_bytes(self, key: str) -> bytes:
        return self.objects[key]


class _Episode:
    def __init__(self, *, episode_type: str, schema: str, canonical_version: int = 1) -> None:
        self.id = EPISODE_ID
        self.episode_type = episode_type
        self.canonical_schema_version = schema
        self.canonical_version = canonical_version


class _Ctx:
    def __init__(self, storage: _Storage) -> None:
        self.storage = storage


def _v1_ego_objects() -> dict[str, bytes]:
    keys = canonical_keys(EPISODE_ID, 1)
    metadata = {
        "schema_version": "canonical-episode/1.0",
        "episode_id": EPISODE_ID,
        "episode_type": "ego_video_episode",
        "source_asset_sha256": "a" * 64,
        "duration_ns": DURATION_NS,
        "target_fps": 10,
        "state_dim": 0,
        "action_dim": 0,
    }
    return {
        keys["metadata"]: json.dumps(metadata).encode("utf-8"),
        keys["video"]: b"fake-mp4-bytes",
    }


def _v2_robot_objects() -> dict[str, bytes]:
    keys = canonical_v2_keys(EPISODE_ID, 1)
    metadata = {
        "schema_version": "canonical-episode/2.0",
        "episode_id": EPISODE_ID,
        "episode_type": "robot_episode",
        "source_asset_sha256": "b" * 64,
        "duration_ns": DURATION_NS,
        "aligned_view": {"fps": 10, "sample_count": 120},
    }
    return {
        keys["metadata"]: json.dumps(metadata).encode("utf-8"),
        f"{keys['prefix']}/preview/front_10hz_640x480.mp4": b"preview-bytes",
    }


def test_an_ego_episode_resolves_its_canonical_v1_video():
    storage = _Storage(_v1_ego_objects())
    episode = _Episode(episode_type=EpisodeType.EGO_VIDEO_EPISODE.value, schema="1.0")

    source = _resolve_video_source(_Ctx(storage), episode)

    assert source.schema_version == "canonical-episode/1.0"
    assert source.video_key == canonical_keys(EPISODE_ID, 1)["video"]
    assert source.duration_ns == DURATION_NS
    assert source.fps == 10


def test_a_robot_episode_still_uses_the_aligned_v2_preview():
    storage = _Storage(_v2_robot_objects())
    episode = _Episode(episode_type=EpisodeType.ROBOT_EPISODE.value, schema="2.0")

    source = _resolve_video_source(_Ctx(storage), episode)

    assert source.schema_version == "canonical-episode/2.0"
    assert source.video_key.endswith("front_10hz_640x480.mp4")
    assert source.fps == 10


def test_ego_never_gains_state_or_action_from_this_path():
    storage = _Storage(_v1_ego_objects())
    episode = _Episode(episode_type=EpisodeType.EGO_VIDEO_EPISODE.value, schema="1.0")

    source = _resolve_video_source(_Ctx(storage), episode)

    assert "state" not in source.metadata or int(source.metadata.get("state_dim") or 0) == 0
    assert int(source.metadata.get("action_dim") or 0) == 0


def test_a_missing_canonical_video_is_an_explicit_terminal_error():
    objects = _v1_ego_objects()
    keys = canonical_keys(EPISODE_ID, 1)
    objects.pop(keys["video"])
    storage = _Storage(objects)
    episode = _Episode(episode_type=EpisodeType.EGO_VIDEO_EPISODE.value, schema="1.0")

    with pytest.raises(TerminalTaskError) as excinfo:
        _resolve_video_source(_Ctx(storage), episode)
    assert excinfo.value.code is ErrorCode.OBJECT_NOT_FOUND


def test_an_un_normalized_episode_is_refused_with_the_expected_generations():
    storage = _Storage({})
    episode = _Episode(episode_type=EpisodeType.EGO_VIDEO_EPISODE.value, schema="1.0")

    with pytest.raises(TerminalTaskError) as excinfo:
        _resolve_video_source(_Ctx(storage), episode)
    assert excinfo.value.code is ErrorCode.INCOMPATIBLE_EPISODE
    assert "canonical-episode/1.0" in str(excinfo.value.details)


def test_metadata_and_episode_type_disagreement_is_rejected():
    objects = _v1_ego_objects()
    keys = canonical_keys(EPISODE_ID, 1)
    payload = json.loads(objects[keys["metadata"]])
    payload["episode_type"] = "robot_episode"
    objects[keys["metadata"]] = json.dumps(payload).encode("utf-8")

    with pytest.raises(TerminalTaskError) as excinfo:
        _resolve_video_source(
            _Ctx(_Storage(objects)),
            _Episode(episode_type=EpisodeType.EGO_VIDEO_EPISODE.value, schema="1.0"),
        )
    assert excinfo.value.code is ErrorCode.INCOMPATIBLE_EPISODE
