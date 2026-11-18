"""Integration tests for ``MCAP_ROS2_PROFILE_V1`` (SPEC 5.4, AT-04 / AT-05).

These tests drive the real parser (``McapProfileV1Parser``) over the generated
ROS 2 CDR MCAP fixtures and assert the profile contract end to end:

* AT-04 ``F05_mcap_good``: three topics decoded, 10 Hz normalisation, action
  zero-order hold never selects a future sample, header stamps preserved as
  diagnostics, canonical samples valid, quality ``PASS``;
* AT-05 ``F06_mcap_gap``: ``SYNC_MISSING_STATE`` ERROR, quality ``FAIL``, no
  fabricated/zero-filled sample, and the episode is not silently truncated;
* unsupported profiles (unknown topic, wrong schema, ROS 1, non-CDR) fail with a
  readable ``UNSUPPORTED_TOPIC_PROFILE`` and no stack trace.

The local host has no ROS 2 installation, so the parser uses the documented
``mcap`` pip + hand-written CDR fallback backend (SPEC 5.4 dual path).  The
production ``rosbag2_py`` backend is covered by a ``ros2``-marked test that is
skipped here and executed inside the Jazzy ``worker-ros2`` image.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from fixtures.make_mcap_ros2 import generate_fixtures
from mcap.writer import Writer
from services.ros2_adapter.cdr import (
    FLOAT64_MULTI_ARRAY_TYPE,
    IMAGE_TYPE,
    JOINT_STATE_TYPE,
    ROS2_MSG_DEFINITIONS,
    encode_float64_multi_array,
    encode_image,
    encode_joint_state,
)
from services.ros2_adapter.reader import (
    TOPIC_ACTION,
    TOPIC_IMAGE,
    TOPIC_JOINT_STATES,
    read_mcap_profile,
    rosbag2_available,
)

from embodied_dataops.canonical.contract import SampleRow, rows_to_robot_table, validate_samples_table
from embodied_dataops.domain.enums import DataOrigin, EpisodeType, SourceType
from embodied_dataops.errors import ApiError, ErrorCode
from embodied_dataops.imports.base import ImportContext
from embodied_dataops.imports.mcap_profile_v1 import McapProfileV1Parser
from embodied_dataops.quality.rules import QualityInput, VideoInfo, evaluate_quality

SAMPLE_INTERVAL_NS = 100_000_000


@pytest.fixture(scope="module")
def mcap_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    directory = tmp_path_factory.mktemp("mcap-import")
    generate_fixtures(directory, update_manifest=False)
    return directory


def _context(path: Path, workdir: Path | None = None) -> ImportContext:
    return ImportContext(
        import_id="import-mcap-test",
        source_type=SourceType.MCAP_ROS2_PROFILE_V1,
        data_origin=DataOrigin.SYNTHETIC,
        source_path=path,
        workdir=workdir or path.parent,
        mapping={},
        license="CC0-1.0",
        source_sha256="0" * 64,
    )


def _quality_input(episode, *, source_type: str = "MCAP_ROS2_PROFILE_V1") -> QualityInput:
    payload = episode.payload
    samples = payload["samples"]
    alignment = payload["alignment"]
    return QualityInput(
        episode_id="episode-mcap-test",
        episode_type=episode.episode_type.value,
        source_type=source_type,
        data_origin="synthetic",
        duration_ns=episode.duration_ns,
        normalization_valid=episode.normalization_valid,
        metadata={
            "robot_profile": payload.get("robot_profile", ""),
            "clock_domains": payload.get("clock_domains", {}),
            "data_origin": "synthetic",
        },
        origin_ns=payload["origin_ns"],
        t_ns=[sample["t_ns"] for sample in samples],
        state=[sample["state"] for sample in samples],
        action=[sample["action"] for sample in samples],
        state_source_t_ns=[sample["state_source_t_ns"] for sample in samples],
        action_source_t_ns=[sample["action_source_t_ns"] for sample in samples],
        joint_names=payload["joint_names"],
        state_dim=payload["state_dim"],
        action_dim=payload["action_dim"],
        stream_times=payload["stream_times"],
        grid_t_ns=payload["grid"],
        missing_frame_indexes=alignment["missing_frame_indexes"],
        missing_state_indexes=alignment["missing_state_indexes"],
        stale_action_indexes=alignment["stale_action_indexes"],
        # parser-only tests do not produce a canonical video; BLUR/FILE_DECODE are
        # covered by the quality unit tests and the QUALITY_CHECK integration test
        video=VideoInfo(expected=False),
    )


# ---------------------------------------------------------------------------
# AT-04: F05_mcap_good
# ---------------------------------------------------------------------------


def test_at_04_validate_reports_fixed_profile_and_backend(mcap_dir: Path) -> None:
    parser = McapProfileV1Parser(backend="fallback")
    probe = parser.validate(_context(mcap_dir / "F05_mcap_good.mcap"))
    assert probe["backend"] == "mcap_fallback"
    assert probe["profile"] == "ros2"
    assert set(probe["topics"]) == {TOPIC_IMAGE, TOPIC_JOINT_STATES, TOPIC_ACTION}
    assert probe["topics"][TOPIC_IMAGE] == IMAGE_TYPE
    assert probe["topics"][TOPIC_JOINT_STATES] == JOINT_STATE_TYPE
    assert probe["topics"][TOPIC_ACTION] == FLOAT64_MULTI_ARRAY_TYPE
    assert probe["message_counts"] == {
        TOPIC_IMAGE: 300,
        TOPIC_JOINT_STATES: 1020,
        TOPIC_ACTION: 515,
    }
    # container span includes the deliberate 0.3 s action history before frame 0
    assert probe["duration_ns"] == 10_290_000_000


def test_at_04_parse_decodes_three_topics_and_builds_10hz_normalisation(mcap_dir: Path) -> None:
    parser = McapProfileV1Parser(backend="fallback")
    parsed = parser.parse(_context(mcap_dir / "F05_mcap_good.mcap"))
    assert parsed.metadata["episode_count"] == 1
    assert parsed.metadata["backend"] == "mcap_fallback"
    assert len(parsed.episodes) == 1
    episode = parsed.episodes[0]
    assert episode.episode_type == EpisodeType.ROBOT_EPISODE
    assert episode.source_episode_index == 0
    assert episode.normalization_valid is True
    assert episode.diagnostics == []
    assert episode.duration_ns == 10_000_000_000

    payload = episode.payload
    assert len(payload["streams"]["camera"]["frames"]) == 300
    assert len(payload["streams"]["state"]["values"]) == 1020
    assert len(payload["streams"]["action"]["values"]) == 515
    assert payload["joint_names"] == [f"joint_{index}" for index in range(1, 7)]
    assert payload["state_dim"] == 6
    assert payload["action_dim"] == 6
    assert payload["robot_profile"] == "arm6_joint_position_v1"
    assert payload["clock_domains"] == {
        "camera": "mcap_log_time",
        "state": "mcap_log_time",
        "action": "mcap_log_time",
    }

    samples = payload["samples"]
    assert len(samples) == 100
    assert [sample["t_ns"] for sample in samples] == [
        index * SAMPLE_INTERVAL_NS for index in range(100)
    ]
    assert [sample["frame_index"] for sample in samples] == list(range(100))
    assert all(sample["sample_index"] == index for index, sample in enumerate(samples))
    assert payload["grid"] == [index * SAMPLE_INTERVAL_NS for index in range(100)]
    assert payload["alignment"]["grid_points"] == 100
    assert payload["alignment"]["missing_sample_count"] == 0
    assert payload["alignment"]["max_abs_sync_error_ms"] == 0.0
    assert payload["alignment"]["p95_abs_sync_error_ms"] == 0.0


def test_at_04_action_zoh_never_selects_a_future_sample(mcap_dir: Path) -> None:
    parsed = McapProfileV1Parser(backend="fallback").parse(_context(mcap_dir / "F05_mcap_good.mcap"))
    samples = parsed.episodes[0].payload["samples"]
    for sample in samples:
        assert sample["action_source_t_ns"] <= sample["t_ns"]
        assert sample["t_ns"] - sample["action_source_t_ns"] <= 100_000_000
        assert sample["state_source_t_ns"] <= sample["t_ns"] or sample["state_source_t_ns"] >= -50_000_000


def test_at_04_header_stamps_are_preserved_as_diagnostics_not_used_for_alignment(mcap_dir: Path) -> None:
    payload = McapProfileV1Parser(backend="fallback").parse(
        _context(mcap_dir / "F05_mcap_good.mcap")
    ).episodes[0].payload
    camera_deltas = payload["streams"]["camera"]["header_minus_log_ns"]
    state_deltas = payload["streams"]["state"]["header_minus_log_ns"]
    assert len(camera_deltas) == 300 and set(camera_deltas) == {-5_000_000}
    assert len(state_deltas) == 1020 and set(state_deltas) == {-2_000_000}
    assert payload["streams"]["camera"]["header_stamp_ns"][0] == (
        payload["streams"]["camera"]["times_ns"][0] - 5_000_000
    )
    summary = payload["header_to_log_delta_ns"]
    assert summary["camera"]["count"] == 300
    assert summary["camera"]["min_ns"] == -5_000_000
    assert summary["state"]["max_ns"] == -2_000_000
    assert summary["action"]["count"] == 0
    assert "diagnostic" in summary["camera"]["clock_note"]
    # the origin is the first camera log_time, not any header stamp
    assert payload["origin_ns"] == payload["streams"]["camera"]["times_ns"][0]


def test_at_04_payload_is_sufficient_for_canonical_parquet(mcap_dir: Path) -> None:
    payload = McapProfileV1Parser(backend="fallback").parse(
        _context(mcap_dir / "F05_mcap_good.mcap")
    ).episodes[0].payload
    rows = [
        SampleRow(
            sample_index=sample["sample_index"],
            t_ns=sample["t_ns"],
            frame_index=sample["frame_index"],
            state=sample["state"],
            action=sample["action"],
            state_source_t_ns=sample["state_source_t_ns"],
            action_source_t_ns=sample["action_source_t_ns"],
        )
        for sample in payload["samples"]
    ]
    table = rows_to_robot_table(rows)
    assert table.num_rows == 100
    assert validate_samples_table(table, "robot_episode") == []


def test_at_04_f05_quality_passes(mcap_dir: Path) -> None:
    episode = McapProfileV1Parser(backend="fallback").parse(
        _context(mcap_dir / "F05_mcap_good.mcap")
    ).episodes[0]
    evaluation = evaluate_quality(_quality_input(episode))
    assert evaluation.verdict.value == "PASS"
    assert evaluation.findings == []
    assert evaluation.metrics["valid_sample_count"] == 100
    assert evaluation.metrics["missing_sample_count"] == 0


# ---------------------------------------------------------------------------
# AT-05: F06_mcap_gap
# ---------------------------------------------------------------------------


def test_at_05_gap_produces_sync_missing_state_and_quality_fail(mcap_dir: Path) -> None:
    parsed = McapProfileV1Parser(backend="fallback").parse(_context(mcap_dir / "F06_mcap_gap.mcap"))
    episode = parsed.episodes[0]
    assert episode.normalization_valid is False
    codes = [diagnostic["code"] for diagnostic in episode.diagnostics]
    assert "SYNC_MISSING_STATE" in codes
    alignment = episode.payload["alignment"]
    assert alignment["missing_sample_count"] == 11
    assert alignment["valid_sample_count"] == 89
    assert alignment["missing_state_indexes"] == list(range(20, 31))  # 2.0 s .. 3.0 s
    assert episode.payload["grid"] == [index * SAMPLE_INTERVAL_NS for index in range(100)]

    evaluation = evaluate_quality(_quality_input(episode))
    assert evaluation.verdict.value == "FAIL"
    assert [finding.rule_code.value for finding in evaluation.findings] == ["SYNC_MISSING_STATE"]
    assert evaluation.metrics["valid_sample_count"] == 89
    assert evaluation.metrics["missing_sample_count"] == 11
    assert evaluation.metrics["normalization_valid"] is False


def test_at_05_no_samples_are_fabricated_in_the_gap(mcap_dir: Path) -> None:
    episode = McapProfileV1Parser(backend="fallback").parse(
        _context(mcap_dir / "F06_mcap_gap.mcap")
    ).episodes[0]
    samples = episode.payload["samples"]
    assert len(samples) == 89
    produced_times = {sample["t_ns"] for sample in samples}
    for missing_index in range(20, 31):
        assert missing_index * SAMPLE_INTERVAL_NS not in produced_times
    # every produced sample still has a real state value (never zero-filled)
    for sample in samples:
        assert len(sample["state"]) == 6
        assert any(abs(value) > 0 for value in sample["state"])
    # the episode is not silently truncated: the grid still spans 0..9.9 s
    assert episode.payload["grid"][0] == 0
    assert episode.payload["grid"][-1] == 9_900_000_000


# ---------------------------------------------------------------------------
# unsupported profile handling (SPEC 5.4)
# ---------------------------------------------------------------------------


def _schema_definition(message_type: str) -> bytes:
    return ROS2_MSG_DEFINITIONS.get(message_type, "").encode("utf-8")


def _write_raw_mcap(
    path: Path,
    *,
    profile: str = "ros2",
    channels: list[dict],
    messages: list[tuple[int, str, bytes]],
) -> None:
    """Write a small MCAP with arbitrary topic/schema/encoding choices."""
    with path.open("wb") as stream:
        writer = Writer(stream)
        writer.start(profile=profile, library="dataops-test")
        schema_ids: dict[str, int] = {}
        channel_ids: dict[str, int] = {}
        for channel in channels:
            message_type = channel["type"]
            if message_type not in schema_ids:
                schema_ids[message_type] = writer.register_schema(
                    name=message_type,
                    encoding=channel.get("schema_encoding", "ros2msg"),
                    data=_schema_definition(message_type),
                )
            channel_ids[channel["topic"]] = writer.register_channel(
                topic=channel["topic"],
                message_encoding=channel.get("message_encoding", "cdr"),
                schema_id=schema_ids[message_type],
            )
        for log_time, topic, data in messages:
            writer.add_message(
                channel_id=channel_ids[topic], log_time=log_time, publish_time=log_time, data=data
            )
        writer.finish()


def _valid_messages(origin: int = 1_700_000_000_000_000_000) -> list[tuple[int, str, bytes]]:
    return [
        (
            origin,
            TOPIC_IMAGE,
            encode_image(width=4, height=2, encoding="rgb8", data=bytes(24)),
        ),
        (
            origin,
            TOPIC_JOINT_STATES,
            encode_joint_state(
                names=[f"joint_{index}" for index in range(1, 7)], position=[0.0] * 6
            ),
        ),
        (origin, TOPIC_ACTION, encode_float64_multi_array(data=[0.0] * 6)),
    ]


def _assert_unsupported(path: Path, message_fragment: str = "") -> None:
    parser = McapProfileV1Parser(backend="fallback")
    with pytest.raises(ApiError) as excinfo:
        parser.validate(_context(path))
    error = excinfo.value
    assert error.code == ErrorCode.UNSUPPORTED_TOPIC_PROFILE
    assert error.message
    assert "Traceback" not in error.message
    if message_fragment:
        assert message_fragment.lower() in error.message.lower()


def test_unknown_topic_is_rejected_readably(tmp_path: Path) -> None:
    path = tmp_path / "unknown_topic.mcap"
    channels = [
        {"topic": TOPIC_IMAGE, "type": IMAGE_TYPE},
        {"topic": TOPIC_JOINT_STATES, "type": JOINT_STATE_TYPE},
        {"topic": TOPIC_ACTION, "type": FLOAT64_MULTI_ARRAY_TYPE},
        {"topic": "/tf", "type": "tf2_msgs/msg/TFMessage"},
    ]
    _write_raw_mcap(path, channels=channels, messages=_valid_messages())
    _assert_unsupported(path, "outside the fixed")


def test_wrong_schema_type_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "wrong_schema.mcap"
    channels = [
        {"topic": TOPIC_IMAGE, "type": IMAGE_TYPE},
        {"topic": TOPIC_JOINT_STATES, "type": FLOAT64_MULTI_ARRAY_TYPE},
        {"topic": TOPIC_ACTION, "type": FLOAT64_MULTI_ARRAY_TYPE},
    ]
    _write_raw_mcap(path, channels=channels, messages=_valid_messages())
    _assert_unsupported(path, "unsupported message type")


def test_missing_required_topic_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "missing_topic.mcap"
    channels = [{"topic": TOPIC_IMAGE, "type": IMAGE_TYPE}]
    _write_raw_mcap(
        path,
        channels=channels,
        messages=[(1_700_000_000_000_000_000, TOPIC_IMAGE, encode_image(width=4, height=2, encoding="rgb8", data=bytes(24)))],
    )
    _assert_unsupported(path, "missing required profile topics")


def test_ros1_profile_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "ros1.mcap"
    channels = [
        {"topic": TOPIC_IMAGE, "type": IMAGE_TYPE, "message_encoding": "ros1"},
        {"topic": TOPIC_JOINT_STATES, "type": JOINT_STATE_TYPE, "message_encoding": "ros1"},
        {"topic": TOPIC_ACTION, "type": FLOAT64_MULTI_ARRAY_TYPE, "message_encoding": "ros1"},
    ]
    _write_raw_mcap(path, profile="ros1", channels=channels, messages=_valid_messages())
    _assert_unsupported(path, "ROS 1")


def test_non_cdr_message_encoding_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "json_encoding.mcap"
    channels = [
        {"topic": TOPIC_IMAGE, "type": IMAGE_TYPE, "message_encoding": "json"},
        {"topic": TOPIC_JOINT_STATES, "type": JOINT_STATE_TYPE, "message_encoding": "json"},
        {"topic": TOPIC_ACTION, "type": FLOAT64_MULTI_ARRAY_TYPE, "message_encoding": "json"},
    ]
    _write_raw_mcap(path, channels=channels, messages=_valid_messages())
    _assert_unsupported(path, "CDR")


def test_non_mcap_bytes_fail_with_file_decode_error(tmp_path: Path) -> None:
    path = tmp_path / "not_a_bag.mcap"
    path.write_bytes(b"this is definitely not an mcap container")
    parser = McapProfileV1Parser(backend="fallback")
    with pytest.raises(ApiError) as excinfo:
        parser.validate(_context(path))
    assert excinfo.value.code == ErrorCode.FILE_DECODE_ERROR
    assert "MCAP" in excinfo.value.message


def test_multiple_mcap_files_in_directory_are_rejected(tmp_path: Path) -> None:
    directory = tmp_path / "many"
    directory.mkdir()
    for name in ("a.mcap", "b.mcap"):
        (directory / name).write_bytes(b"not really mcap")
    parser = McapProfileV1Parser(backend="fallback")
    with pytest.raises(ApiError) as excinfo:
        parser.validate(_context(directory))
    assert excinfo.value.code == ErrorCode.UNSUPPORTED_TOPIC_PROFILE
    assert "single-Episode" in excinfo.value.message


# ---------------------------------------------------------------------------
# ROS 2 native backend (production path; skipped when rosbag2_py is absent)
# ---------------------------------------------------------------------------


@pytest.mark.ros2
def test_ros2_native_backend_reads_the_same_profile(mcap_dir: Path) -> None:
    if not rosbag2_available():
        pytest.skip("rosbag2_py is only available inside the ROS 2 Jazzy worker image")
    data = read_mcap_profile(mcap_dir / "F05_mcap_good.mcap", backend="rosbag2")
    assert data.backend == "rosbag2_py"
    assert set(data.topics) == {TOPIC_IMAGE, TOPIC_JOINT_STATES, TOPIC_ACTION}
    assert len(data.camera) == 300
    assert len(data.joint_states) == 1020
    assert len(data.actions) == 515


def test_rosbag2_backend_unavailable_is_reported_honestly() -> None:
    if rosbag2_available():  # pragma: no cover - only in the ROS 2 worker image
        pytest.skip("rosbag2_py is installed; the production path is available")
    from services.ros2_adapter.reader import resolve_backend

    with pytest.raises(ApiError) as excinfo:
        resolve_backend("rosbag2")
    assert excinfo.value.code == ErrorCode.DEPENDENCY_UNAVAILABLE
    assert "Jazzy" in excinfo.value.message
