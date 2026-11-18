"""F05/F06 MCAP fixture provenance and determinism (SPEC 13.1).

Asserts that the generated ``F05_mcap_good`` / ``F06_mcap_gap`` files are real
ROS 2 CDR MCAP containers with the fixed profile topology, that generation is
byte-deterministic (fixed seed ``20261029``), and that the manifest merge helper
is concurrency-safe and preserves every fixture entry it does not own.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from fixtures.make_mcap_ros2 import (
    ACTION_HISTORY_NS,
    FIXTURE_SPECS,
    JOINT_COUNT,
    JOINT_HISTORY_NS,
    JOINT_PERIOD_NS,
    ORIGIN_NS,
    SEED,
    generate_fixtures,
    main,
    merge_manifest_entries,
    sha256_file,
)
from mcap.reader import make_reader
from services.ros2_adapter.reader import (
    TOPIC_ACTION,
    TOPIC_IMAGE,
    TOPIC_JOINT_STATES,
    read_mcap_profile,
)

EXPECTED_TOPICS = {TOPIC_IMAGE, TOPIC_JOINT_STATES, TOPIC_ACTION}
EXPECTED_SCHEMAS = {
    "sensor_msgs/msg/Image",
    "sensor_msgs/msg/JointState",
    "std_msgs/msg/Float64MultiArray",
}


@pytest.fixture(scope="module")
def generated(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, dict[str, dict]]:
    directory = tmp_path_factory.mktemp("mcap-fixtures")
    entries = generate_fixtures(directory, update_manifest=False)
    return directory, {entry["fixture_id"]: entry for entry in entries}


def test_fixture_ids_match_spec() -> None:
    assert set(FIXTURE_SPECS) == {"F05_mcap_good", "F06_mcap_gap"}
    assert SEED == 20261029
    assert FIXTURE_SPECS["F05_mcap_good"]["expected"]["verdict"] == "PASS"
    assert FIXTURE_SPECS["F06_mcap_gap"]["expected"]["verdict"] == "FAIL"
    assert "SYNC_MISSING_STATE" in FIXTURE_SPECS["F06_mcap_gap"]["expected"]["error_rules"]


def test_generated_files_and_manifest_entries(generated) -> None:
    directory, entries = generated
    for fixture_id, expected in (
        ("F05_mcap_good", {"verdict": "PASS"}),
        ("F06_mcap_gap", {"verdict": "FAIL"}),
    ):
        path = directory / f"{fixture_id}.mcap"
        assert path.is_file()
        assert path.stat().st_size > 1_000_000
        entry = entries[fixture_id]
        assert entry["fixture_id"] == fixture_id
        assert entry["kind"] == "file"
        assert entry["content_type"] == "application/vnd.mcap"
        assert entry["data_origin"] == "synthetic"
        assert entry["seed"] == 20261029
        assert entry["sha256"] == sha256_file(path)
        assert len(entry["sha256"]) == 64
        assert entry["size_bytes"] == path.stat().st_size
        assert entry["expected"]["verdict"] == expected["verdict"]
        assert entry["expected"]["outcome"]
        assert entry["generated_by"] == "fixtures/make_mcap_ros2.py"
        assert entry["status"] == "generated"


def test_generation_is_byte_deterministic(generated, tmp_path: Path) -> None:
    _directory, entries = generated
    second = tmp_path / "second"
    regenerated = generate_fixtures(
        second, update_manifest=False, fixture_ids=["F05_mcap_good", "F06_mcap_gap"]
    )
    for entry in regenerated:
        assert entry["sha256"] == entries[entry["fixture_id"]]["sha256"]
        assert sha256_file(second / f"{entry['fixture_id']}.mcap") == entry["sha256"]
    # generating only one fixture yields the same bytes as generating both
    single = tmp_path / "single"
    only_f05 = generate_fixtures(single, update_manifest=False, fixture_ids=["F05_mcap_good"])
    assert only_f05[0]["sha256"] == entries["F05_mcap_good"]["sha256"]


def test_raw_mcap_container_is_a_ros2_cdr_bag(generated) -> None:
    directory, _entries = generated
    path = directory / "F05_mcap_good.mcap"
    with path.open("rb") as stream:
        reader = make_reader(stream)
        header = reader.get_header()
        summary = reader.get_summary()
    assert header.profile == "ros2"
    assert summary is not None
    channels = {channel.topic: channel for channel in summary.channels.values()}
    assert set(channels) == EXPECTED_TOPICS
    schemas = {schema.name for schema in summary.schemas.values()}
    assert schemas == EXPECTED_SCHEMAS
    for channel in channels.values():
        assert channel.message_encoding == "cdr"
    for schema in summary.schemas.values():
        assert schema.encoding == "ros2msg"
        assert schema.data
    statistics = summary.statistics
    assert statistics is not None
    channels_by_id = summary.channels
    counts = {
        channels_by_id[channel_id].topic: count
        for channel_id, count in statistics.channel_message_counts.items()
    }
    assert counts[TOPIC_IMAGE] == 300
    assert counts[TOPIC_JOINT_STATES] == 1020
    assert counts[TOPIC_ACTION] == 515


def test_f06_removes_only_the_joint_state_window(generated) -> None:
    directory, _entries = generated
    with (directory / "F06_mcap_gap.mcap").open("rb") as stream:
        reader = make_reader(stream)
        summary = reader.get_summary()
    assert summary is not None and summary.statistics is not None
    channels_by_id = summary.channels
    counts = {
        channels_by_id[channel_id].topic: count
        for channel_id, count in summary.statistics.channel_message_counts.items()
    }
    assert counts[TOPIC_IMAGE] == 300  # camera stream untouched
    assert counts[TOPIC_ACTION] == 515  # action stream untouched
    assert counts[TOPIC_JOINT_STATES] == 1020 - 101  # 2.00 s .. 3.00 s inclusive at 100 Hz
    # the container statistics span is unchanged: camera + action still cover the
    # whole episode, only the joint-state window is missing (no truncation)
    assert summary.statistics.message_start_time == ORIGIN_NS - ACTION_HISTORY_NS
    assert summary.statistics.message_end_time == ORIGIN_NS - JOINT_HISTORY_NS + (JOINT_COUNT - 1) * JOINT_PERIOD_NS


def test_fallback_reader_backend_reports_itself(generated) -> None:
    directory, _entries = generated
    data = read_mcap_profile(directory / "F05_mcap_good.mcap", backend="fallback")
    assert data.backend == "mcap_fallback"
    assert data.profile == "ros2"
    assert set(data.topics) == EXPECTED_TOPICS
    assert len(data.camera) == 300
    assert len(data.joint_states) == 1020
    assert len(data.actions) == 515
    assert data.camera[0].message.encoding == "rgb8"
    assert data.camera[0].message.width == 160
    assert data.camera[0].message.height == 120
    assert data.joint_states[0].message.names == tuple(f"joint_{index}" for index in range(1, 7))


def test_manifest_merge_preserves_foreign_entries(tmp_path: Path) -> None:
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "schema_version": "fixtures-manifest/1.0",
                "seed": SEED,
                "data_origin": "synthetic",
                "generated_at": "2026-10-29T00:00:00Z",
                "fixtures": [
                    {
                        "fixture_id": "F01_good_lerobot",
                        "path": "fixtures/generate/F01_good_lerobot",
                        "sha256": "a" * 64,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    entries = [
        {
            "fixture_id": fixture_id,
            "path": f"fixtures/golden/mcap/{fixture_id}.mcap",
            "sha256": "b" * 64,
            "data_origin": "synthetic",
            "expected": {"verdict": "PASS"},
        }
        for fixture_id in ("F05_mcap_good", "F06_mcap_gap")
    ]
    document = merge_manifest_entries(entries, manifest_path=manifest_path)
    assert [entry["fixture_id"] for entry in document["fixtures"]] == [
        "F01_good_lerobot",
        "F05_mcap_good",
        "F06_mcap_gap",
    ]
    assert document["fixtures"][0]["sha256"] == "a" * 64

    # idempotent re-run: no duplicate entries, foreign entry untouched, lock released
    merge_manifest_entries(entries, manifest_path=manifest_path)
    reloaded = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert [entry["fixture_id"] for entry in reloaded["fixtures"]] == [
        "F01_good_lerobot",
        "F05_mcap_good",
        "F06_mcap_gap",
    ]
    assert not (tmp_path / ".manifest.lock").exists()


def test_manifest_merge_tolerates_legacy_list_format(tmp_path: Path) -> None:
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps([{"fixture_id": "F04_ego_video"}]), encoding="utf-8")
    document = merge_manifest_entries(
        [{"fixture_id": "F05_mcap_good", "path": "p", "sha256": "c" * 64}],
        manifest_path=manifest_path,
    )
    assert document["schema_version"] == "fixtures-manifest/1.0"
    assert [entry["fixture_id"] for entry in document["fixtures"]] == [
        "F04_ego_video",
        "F05_mcap_good",
    ]


def test_cli_generates_without_touching_the_repository_manifest(tmp_path: Path) -> None:
    assert main(["--output-dir", str(tmp_path), "--no-manifest"]) == 0
    assert (tmp_path / "F05_mcap_good.mcap").is_file()
    assert (tmp_path / "F06_mcap_gap.mcap").is_file()
