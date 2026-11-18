"""``MCAP_ROS2_PROFILE_V2`` importer (V2 SPEC 2.4).

Fixed, profile-bound ROS 2 profile (single Episode per MCAP container):

======================================  ==========================================
topic                                   type / handling
======================================  ==========================================
``/camera/front/image_raw``             ``sensor_msgs/msg/Image`` -- **required**
``/camera/wrist/image_raw``             ``sensor_msgs/msg/Image`` -- optional
``/joint_states``                       ``sensor_msgs/msg/JointState`` -- 6D/7D
                                        per the resolved robot profile
``/policy/action``                      ``std_msgs/msg/Float64MultiArray`` --
                                        6D/7D target joint position
``/points``                             ``sensor_msgs/msg/PointCloud2`` --
                                        optional, ignored with a diagnostic
                                        (bounded parsing belongs to the pointcloud
                                        milestone; no summary is fabricated here)
======================================  ==========================================

Anything else on another topic is ignored with a diagnostic; a missing required
topic, a wrong schema type or a non-CDR message encoding is rejected with
``UNSUPPORTED_TOPIC_PROFILE``.  ``MCAP_ROS2_PROFILE_V1`` is untouched and remains
the canonical-episode/1.0 path.

Timebase: all streams are timed by the MCAP ``log_time`` (receive time); the
origin is the first front-camera frame's ``log_time``.  ``header.stamp`` deltas
are recorded as diagnostics only.  ``clock_verified=false``: the spec only allows
"approximate log-time synchronization" without external calibration evidence.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import numpy as np

from embodied_dataops.canonical.timebase import MAX_EPISODE_DURATION_NS
from embodied_dataops.canonical.v2.model import (
    CanonicalV2Source,
    NativeCameraStream,
    NativeRobotStream,
)
from embodied_dataops.domain.enums import EpisodeType, Severity, SourceType
from embodied_dataops.domain.profiles import resolve_profile
from embodied_dataops.errors import ApiError, ErrorCode
from embodied_dataops.imports.base import EpisodeSource, ImportContext, ParsedSource, SourceParser
from embodied_dataops.imports.common import diagnostic, has_structural, merge_diagnostics
from embodied_dataops.imports.mcap_profile_v1 import MCAP_MAGIC

IMAGE_TYPE = "sensor_msgs/msg/Image"
JOINT_STATE_TYPE = "sensor_msgs/msg/JointState"
ACTION_TYPE = "std_msgs/msg/Float64MultiArray"
POINTCLOUD_TYPE = "sensor_msgs/msg/PointCloud2"

TOPIC_FRONT = "/camera/front/image_raw"
TOPIC_WRIST = "/camera/wrist/image_raw"
TOPIC_JOINT_STATES = "/joint_states"
TOPIC_ACTION = "/policy/action"
TOPIC_POINTS = "/points"

CLOCK_DOMAIN = "mcap_log_time"
TIMESTAMP_PROVENANCE = "mcap_log_time"

#: Diagnostic codes (INFO/WARN diagnostics never block; ERROR ones do).
DIAG_UNLISTED_TOPIC = "UNLISTED_TOPIC_IGNORED"
DIAG_POINTCLOUD_IGNORED = "POINTCLOUD_IGNORED"
DIAG_WRIST_MISSING = "OPTIONAL_CAMERA_MISSING"

_MAX_MESSAGES = 2_000_000


def _ros2_cdr():
    try:
        from services.ros2_adapter import cdr  # type: ignore[import-not-found]
    except ModuleNotFoundError:
        repository_root = Path(__file__).resolve().parents[3]
        if str(repository_root) not in sys.path:
            sys.path.insert(0, str(repository_root))
        from services.ros2_adapter import cdr  # type: ignore[import-not-found]
    return cdr


def _mcap_module():
    try:
        from mcap.reader import make_reader  # type: ignore[import-not-found]
    except ModuleNotFoundError as exc:  # pragma: no cover - dependency is pinned
        raise ApiError(
            ErrorCode.LEROBOT_API_UNAVAILABLE,
            "the 'mcap' package is required for MCAP_ROS2_PROFILE_V2 imports",
        ) from exc
    return make_reader


def _normalise_type(name: str | None) -> str:
    text = str(name or "")
    # rosbag2 may report "sensor_msgs/msg/Image" or "sensor_msgs/Image".
    if text.count("/") == 1:
        package, short = text.split("/", 1)
        return f"{package}/msg/{short}"
    return text


class _Probe:
    def __init__(self) -> None:
        self.topics: dict[str, dict[str, Any]] = {}
        self.message_counts: dict[str, int] = {}
        self.encoding_ok = True

    def fail(self, message: str, details: dict[str, Any] | None = None) -> ApiError:
        return ApiError(ErrorCode.UNSUPPORTED_TOPIC_PROFILE, message, details=details)


def _probe_file(path: Path) -> _Probe:
    make_reader = _mcap_module()
    probe = _Probe()
    with path.open("rb") as stream:
        reader = make_reader(stream)
        summary = None
        try:
            summary = reader.get_summary()
        except Exception:  # pragma: no cover - defensive for truncated files
            summary = None
        if summary is not None and summary.channels:
            for channel in summary.channels.values():
                schema = summary.schemas.get(channel.schema_id)
                probe.topics[channel.topic] = {
                    "schema": _normalise_type(schema.name if schema else ""),
                    "message_encoding": channel.message_encoding,
                    "channel_id": channel.id,
                }
            statistics = getattr(summary, "statistics", None)
            if statistics is not None:
                for channel_id, count in (statistics.channel_message_counts or {}).items():
                    channel = summary.channels.get(channel_id)
                    if channel is not None:
                        probe.message_counts[channel.topic] = int(count)
            return probe
        # No summary section: scan messages (bounded) to learn the topic map.
        count = 0
        for schema, channel, message in reader.iter_messages():
            del message
            probe.topics.setdefault(
                channel.topic,
                {
                    "schema": _normalise_type(schema.name if schema else ""),
                    "message_encoding": channel.message_encoding,
                    "channel_id": channel.id,
                },
            )
            probe.message_counts[channel.topic] = probe.message_counts.get(channel.topic, 0) + 1
            count += 1
            if count >= _MAX_MESSAGES:
                break
    return probe


def _check_topics(probe: _Probe, mapping: dict[str, Any]) -> tuple[str, str, dict[str, Any]]:
    front_topic = str(mapping.get("camera_key") or TOPIC_FRONT)
    wrist_topic = str(mapping.get("wrist_camera_key") or TOPIC_WRIST)
    requirements = {
        front_topic: IMAGE_TYPE,
        TOPIC_JOINT_STATES: JOINT_STATE_TYPE,
        TOPIC_ACTION: ACTION_TYPE,
    }
    missing = [topic for topic in requirements if topic not in probe.topics]
    if missing:
        raise probe.fail(
            f"MCAP_ROS2_PROFILE_V2 is missing required topic(s): {missing}",
            details={"missing_topics": missing, "observed_topics": sorted(probe.topics)},
        )
    for topic, expected in requirements.items():
        observed = str(probe.topics[topic]["schema"])
        if observed != expected:
            raise probe.fail(
                f"topic {topic} has unsupported schema {observed!r}; expected {expected}",
                details={"topic": topic, "expected": expected, "observed": observed},
            )
        if str(probe.topics[topic].get("message_encoding")) != "cdr":
            raise probe.fail(
                f"topic {topic} is not CDR encoded",
                details={
                    "topic": topic,
                    "message_encoding": probe.topics[topic].get("message_encoding"),
                },
            )
    known = {front_topic, TOPIC_JOINT_STATES, TOPIC_ACTION, wrist_topic, TOPIC_POINTS}
    unknown = sorted(set(probe.topics) - known)
    if wrist_topic in probe.topics:
        schema = str(probe.topics[wrist_topic]["schema"])
        if schema != IMAGE_TYPE:
            raise probe.fail(
                f"optional wrist camera topic {wrist_topic} has unsupported schema {schema!r}",
                details={"topic": wrist_topic, "expected": IMAGE_TYPE, "observed": schema},
            )
    if TOPIC_POINTS in probe.topics:
        schema = str(probe.topics[TOPIC_POINTS]["schema"])
        if schema != POINTCLOUD_TYPE:
            raise probe.fail(
                f"topic {TOPIC_POINTS} has unsupported schema {schema!r}",
                details={"topic": TOPIC_POINTS, "expected": POINTCLOUD_TYPE, "observed": schema},
            )
    return front_topic, wrist_topic, {
        "unknown_topics": unknown,
        "has_wrist": wrist_topic in probe.topics,
        "has_pointcloud": TOPIC_POINTS in probe.topics,
    }


def _image_to_rgb(message: Any) -> np.ndarray:
    width = int(message.width)
    height = int(message.height)
    step = int(message.step)
    encoding = str(message.encoding)
    if encoding not in {"rgb8", "bgr8"}:
        raise ApiError(
            ErrorCode.UNSUPPORTED_TOPIC_PROFILE,
            "MCAP_ROS2_PROFILE_V2 camera encoding must be rgb8 or bgr8",
            details={"encoding": encoding, "supported": ["rgb8", "bgr8"]},
        )
    if width <= 0 or height <= 0 or step < width * 3:
        raise ApiError(
            ErrorCode.FILE_DECODE_ERROR,
            "MCAP camera frame does not match its declared geometry",
            details={"width": width, "height": height, "step": step},
        )
    raw = np.frombuffer(bytes(message.data), dtype=np.uint8)
    if raw.size < height * step:
        raise ApiError(
            ErrorCode.FILE_DECODE_ERROR,
            "MCAP camera frame payload is shorter than height*step",
            details={"bytes": int(raw.size), "required": height * step},
        )
    rows = raw[: height * step].reshape(height, step)
    pixels = rows[:, : width * 3].reshape(height, width, 3)
    if encoding == "bgr8":
        pixels = pixels[:, :, ::-1]
    return np.ascontiguousarray(pixels)


class McapProfileV2Parser(SourceParser):
    """Profile-bound multi-stream MCAP v2 adapter (single Episode)."""

    source_type = SourceType.MCAP_ROS2_PROFILE_V2

    def __init__(self, *, backend: str = "python-mcap") -> None:
        #: The v2 adapter reads the container with the pinned ``mcap`` package and
        #: the shared CDR primitives; the rosbag2 worker remains a v1 concern.
        self.backend = backend
        self._last_probe: dict[str, Any] = {}

    # -- files -------------------------------------------------------------
    def _single_mcap_path(self, ctx: ImportContext) -> Path:
        source = Path(ctx.source_path)
        if source.is_dir():
            candidates = sorted(source.glob("*.mcap"))
            if len(candidates) != 1:
                raise ApiError(
                    ErrorCode.UNSUPPORTED_TOPIC_PROFILE,
                    "MCAP_ROS2_PROFILE_V2 requires exactly one single-Episode .mcap file",
                    details={"mcap_files": [path.name for path in candidates]},
                )
            source = candidates[0]
        if not source.is_file():
            raise ApiError(ErrorCode.SOURCE_NOT_FOUND, f"MCAP source not found: {source.name}")
        with source.open("rb") as handle:
            if handle.read(8) != MCAP_MAGIC:
                raise ApiError(
                    ErrorCode.FILE_DECODE_ERROR,
                    "File is not an MCAP container (bad magic bytes)",
                    details={"filename": source.name},
                )
        return source

    # -- SourceParser ------------------------------------------------------
    def validate(self, ctx: ImportContext) -> dict[str, Any]:
        path = self._single_mcap_path(ctx)
        probe = _probe_file(path)
        front_topic, wrist_topic, topic_info = _check_topics(probe, dict(ctx.mapping or {}))
        resolve_profile(ctx.mapping)  # MAPPING_REQUIRED / NOT_FOUND before any work
        self._last_probe = {
            "backend": self.backend,
            "topics": {topic: dict(info) for topic, info in sorted(probe.topics.items())},
            "message_counts": dict(sorted(probe.message_counts.items())),
            "front_topic": front_topic,
            "wrist_topic": wrist_topic,
            **topic_info,
        }
        return dict(self._last_probe)

    def parse(self, ctx: ImportContext) -> ParsedSource:
        path = self._single_mcap_path(ctx)
        probe = _probe_file(path)
        front_topic, wrist_topic, topic_info = _check_topics(probe, dict(ctx.mapping or {}))
        resolved = resolve_profile(ctx.mapping)
        diagnostics: list[dict[str, Any]] = []
        for topic in topic_info["unknown_topics"]:
            diagnostics.append(
                diagnostic(
                    DIAG_UNLISTED_TOPIC,
                    Severity.INFO,
                    f"topic {topic} is not part of MCAP_ROS2_PROFILE_V2 and is ignored",
                    details={"topic": topic},
                    structural=False,
                )
            )
        if topic_info["has_pointcloud"]:
            diagnostics.append(
                diagnostic(
                    DIAG_POINTCLOUD_IGNORED,
                    Severity.INFO,
                    "/points is present but bounded PointCloud2 summary parsing is not produced "
                    "by the v2 importer; no pointcloud summary asset is fabricated",
                    details={"topic": TOPIC_POINTS, "message_type": POINTCLOUD_TYPE},
                    structural=False,
                )
            )
        expected_terminal = bool((ctx.mapping or {}).get("expected_terminal_marker", False))
        episode = EpisodeSource(
            source_episode_index=0,
            episode_type=EpisodeType.ROBOT_EPISODE,
            task_label=str((ctx.mapping or {}).get("task_label") or ""),
            duration_ns=0,
            payload={
                "adapter": "mcap_profile_v2",
                "front_topic": front_topic,
                "wrist_topic": wrist_topic if topic_info["has_wrist"] else None,
                "profile_id": resolved.profile_id,
                "profile_revision": resolved.profile_revision,
                "profile_config_sha256": resolved.config_sha256,
                "expected_terminal_marker": expected_terminal,
                **topic_info,
            },
            diagnostics=diagnostics,
            normalization_valid=not has_structural(diagnostics),
        )
        return ParsedSource(
            episodes=[episode],
            metadata={
                "adapter": "mcap_profile_v2",
                "backend": self.backend,
                "episode_count": 1,
                "clock_domain": CLOCK_DOMAIN,
                "topics": {topic: dict(info) for topic, info in sorted(probe.topics.items())},
                "message_counts": dict(sorted(probe.message_counts.items())),
            },
            warnings=[
                "MCAP log_time is the recording system receive time; it is not camera exposure "
                "time or a hard real-time synchronisation source (clock_verified=false)",
            ],
        )

    # -- full read ---------------------------------------------------------
    def read_v2_episode(self, ctx: ImportContext, episode_index: int = 0) -> CanonicalV2Source:
        if int(episode_index) != 0:
            raise ApiError(
                ErrorCode.EPISODE_ENUMERATION_FAILED,
                "MCAP_ROS2_PROFILE_V2 contains exactly one Episode (index 0)",
                details={"episode_index": int(episode_index)},
            )
        path = self._single_mcap_path(ctx)
        mapping = dict(ctx.mapping or {})
        probe = _probe_file(path)
        front_topic, wrist_topic, topic_info = _check_topics(probe, mapping)
        resolved = resolve_profile(mapping)
        profile = resolved.config
        raw = self._read_streams(path, front_topic, wrist_topic, topic_info)
        front_frames = raw["front"]
        if not front_frames:
            raise ApiError(
                ErrorCode.UNSUPPORTED_TOPIC_PROFILE,
                f"MCAP contains no decodable frames on {front_topic}",
            )
        diagnostics: list[dict[str, Any]] = []
        for topic in topic_info["unknown_topics"]:
            diagnostics.append(
                diagnostic(
                    DIAG_UNLISTED_TOPIC,
                    Severity.INFO,
                    f"topic {topic} is not part of MCAP_ROS2_PROFILE_V2 and is ignored",
                    details={"topic": topic},
                    structural=False,
                )
            )
        if topic_info["has_pointcloud"]:
            diagnostics.append(
                diagnostic(
                    DIAG_POINTCLOUD_IGNORED,
                    Severity.INFO,
                    "/points is present but the v2 importer does not produce a pointcloud summary",
                    details={"topic": TOPIC_POINTS, "messages": raw.get("pointcloud_count", 0)},
                    structural=False,
                )
            )
        if not resolved.semantic_authority:
            diagnostics.append(
                diagnostic(
                    "UNVERIFIED_SEMANTICS",
                    Severity.ERROR,
                    "robot profile authority is not SYNTHETIC_FIXTURE/SOURCE_DOCUMENTED; "
                    "the episode may be normalized for diagnostics but must not be published",
                    details={
                        "profile_id": resolved.profile_id,
                        "profile_revision": resolved.profile_revision,
                        "authority": profile.authority if profile is not None else None,
                    },
                    structural=False,
                )
            )
        origin_ns = int(front_frames[0][0])
        span_ns = int(front_frames[-1][0]) - origin_ns
        if span_ns > MAX_EPISODE_DURATION_NS:
            raise ApiError(
                ErrorCode.SOURCE_DURATION_LIMIT,
                "MCAP episode exceeds the 600 s limit",
                details={"duration_ns": span_ns, "limit_ns": MAX_EPISODE_DURATION_NS},
            )
        # front camera stream
        front_times = [int(item[0]) for item in front_frames]
        front_messages = [item[1] for item in front_frames]
        cameras: list[NativeCameraStream] = [
            NativeCameraStream(
                name="front",
                source_key=front_topic,
                clock_domain=CLOCK_DOMAIN,
                timestamp_provenance=TIMESTAMP_PROVENANCE,
                times_ns=front_times,
                width=int(front_messages[0].width),
                height=int(front_messages[0].height),
                encoding=str(front_messages[0].encoding),
                frame_getter=_frame_getter(front_messages),
                source_asset_id=ctx.source_sha256,
            )
        ]
        if raw["wrist"]:
            wrist_times = [int(item[0]) for item in raw["wrist"]]
            wrist_messages = [item[1] for item in raw["wrist"]]
            cameras.append(
                NativeCameraStream(
                    name="wrist",
                    source_key=wrist_topic,
                    clock_domain=CLOCK_DOMAIN,
                    timestamp_provenance=TIMESTAMP_PROVENANCE,
                    times_ns=wrist_times,
                    width=int(wrist_messages[0].width),
                    height=int(wrist_messages[0].height),
                    encoding=str(wrist_messages[0].encoding),
                    frame_getter=_frame_getter(wrist_messages),
                    source_asset_id=ctx.source_sha256,
                )
            )
        elif mapping.get("wrist_camera_key"):
            diagnostics.append(
                diagnostic(
                    DIAG_WRIST_MISSING,
                    Severity.WARN,
                    "mapping.wrist_camera_key was provided but the topic has no decodable frames",
                    details={"wrist_topic": wrist_topic},
                    structural=False,
                )
            )
        # robot streams
        state_times, state_values, state_names, state_diag = _read_joint_states(
            raw["joint_states"], profile
        )
        action_times, action_values, action_diag = _read_actions(raw["actions"], profile)
        diagnostics.extend(state_diag)
        diagnostics.extend(action_diag)
        diagnostics.extend(_monotonic_diagnostics(front_times, state_times, action_times))
        expected_terminal = bool(mapping.get("expected_terminal_marker", False))
        return CanonicalV2Source(
            episode_id="",  # filled by the handler
            source_asset_sha256=ctx.source_sha256,
            source_type=SourceType.MCAP_ROS2_PROFILE_V2.value,
            source_episode_index=0,
            data_origin=ctx.data_origin.value,
            task_label=str(mapping.get("task_label") or ""),
            profile=profile,
            profile_id=resolved.profile_id,
            profile_revision=resolved.profile_revision,
            profile_config_sha256=resolved.config_sha256,
            profile_authority=profile.authority if profile is not None else "",
            joint_names=list(profile.joint_names) if profile is not None else list(state_names),
            action_names=(
                list(profile.action_names) if profile is not None else list(state_names)
            ),
            state_semantics=profile.state_semantics if profile is not None else "",
            action_semantics=profile.action_semantics if profile is not None else "",
            state_unit=profile.state_unit if profile is not None else "rad",
            action_unit=profile.action_unit if profile is not None else "rad",
            diagnostic_only=resolved.mode == "DIAGNOSTIC_ONLY",
            semantic_authority=bool(resolved.semantic_authority),
            cameras=cameras,
            state=(
                NativeRobotStream(
                    name="state",
                    source_key=TOPIC_JOINT_STATES,
                    clock_domain=CLOCK_DOMAIN,
                    timestamp_provenance=TIMESTAMP_PROVENANCE,
                    times_ns=state_times,
                    values=state_values,
                    unit=profile.state_unit if profile is not None else "rad",
                    source_asset_id=ctx.source_sha256,
                )
                if state_times
                else None
            ),
            action=(
                NativeRobotStream(
                    name="action",
                    source_key=TOPIC_ACTION,
                    clock_domain=CLOCK_DOMAIN,
                    timestamp_provenance=TIMESTAMP_PROVENANCE,
                    times_ns=action_times,
                    values=action_values,
                    unit=profile.action_unit if profile is not None else "rad",
                    source_asset_id=ctx.source_sha256,
                )
                if action_times
                else None
            ),
            alignment_mode="native",
            origin_ns=origin_ns,
            clock_domains={
                "camera_front": CLOCK_DOMAIN,
                "state": CLOCK_DOMAIN,
                "action": CLOCK_DOMAIN,
            },
            native_rates_hz={},
            required_modalities=["front", "state", "action"],
            diagnostics=merge_diagnostics(diagnostics),
            expected_terminal_marker=expected_terminal,
            terminal_marker_found=None,
            pointcloud=None,
            mapping_report={
                "topic_profile": "MCAP_ROS2_PROFILE_V2",
                "front_topic": front_topic,
                "wrist_topic": wrist_topic if cameras and len(cameras) > 1 else None,
                "joint_state_topic": TOPIC_JOINT_STATES,
                "action_topic": TOPIC_ACTION,
                "pointcloud_topic": TOPIC_POINTS if topic_info["has_pointcloud"] else None,
                "unknown_topics": topic_info["unknown_topics"],
                "joint_names_observed": list(state_names),
                "header_stamp_note": "header.stamp deltas are diagnostics only; log_time is the base",
            },
            source={
                "topics": {topic: dict(info) for topic, info in sorted(probe.topics.items())},
                "message_counts": dict(sorted(probe.message_counts.items())),
                "origin_log_time_ns": origin_ns,
                "clock_verified": False,
                "clock_note": "MCAP log_time is receive time; no external calibration evidence",
            },
        )

    # -- reading -----------------------------------------------------------
    @staticmethod
    def _read_streams(
        path: Path,
        front_topic: str,
        wrist_topic: str,
        topic_info: dict[str, Any],
    ) -> dict[str, Any]:
        make_reader = _mcap_module()
        cdr = _ros2_cdr()
        streams: dict[str, Any] = {
            "front": [],
            "wrist": [],
            "joint_states": [],
            "actions": [],
            "pointcloud_count": 0,
        }
        wanted = {front_topic, TOPIC_JOINT_STATES, TOPIC_ACTION}
        if topic_info["has_wrist"]:
            wanted.add(wrist_topic)
        if topic_info["has_pointcloud"]:
            wanted.add(TOPIC_POINTS)
        total = 0
        with path.open("rb") as stream:
            reader = make_reader(stream)
            for schema, channel, message in reader.iter_messages():
                total += 1
                if total > _MAX_MESSAGES:
                    raise ApiError(
                        ErrorCode.FILE_DECODE_ERROR,
                        "MCAP container exceeds the message budget for one Episode",
                        details={"limit": _MAX_MESSAGES},
                    )
                topic = channel.topic
                if topic not in wanted:
                    continue
                if topic == TOPIC_POINTS:
                    streams["pointcloud_count"] += 1
                    continue
                message_type = _normalise_type(schema.name if schema else "")
                try:
                    decoded, _alignment_origin = cdr.decode_with_alignment_fallback(
                        message_type, message.data
                    )
                except Exception as exc:
                    raise ApiError(
                        ErrorCode.FILE_DECODE_ERROR,
                        f"cannot decode CDR message on {topic}",
                        details={"topic": topic, "message_type": message_type,
                                 "reason": type(exc).__name__},
                    ) from exc
                entry = (int(message.log_time), decoded)
                if topic == front_topic:
                    streams["front"].append(entry)
                elif topic == wrist_topic:
                    streams["wrist"].append(entry)
                elif topic == TOPIC_JOINT_STATES:
                    streams["joint_states"].append(entry)
                elif topic == TOPIC_ACTION:
                    streams["actions"].append(entry)
        for name in ("front", "wrist", "joint_states", "actions"):
            streams[name].sort(key=lambda item: item[0])
        return streams


def _frame_getter(messages: list[Any]):
    def getter(source_index: int) -> np.ndarray:
        return _image_to_rgb(messages[int(source_index)])

    return getter


def _read_joint_states(
    samples: list[tuple[int, Any]],
    profile: Any,
) -> tuple[list[int], list[list[float]], list[str], list[dict[str, Any]]]:
    diagnostics: list[dict[str, Any]] = []
    times: list[int] = []
    values: list[list[float]] = []
    observed: list[str] = []
    required = list(profile.joint_names) if profile is not None else []
    names_valid = True
    for log_time, message in samples:
        names = [str(name) for name in message.names]
        position = [float(value) for value in message.position]
        observed = names or observed
        if required and (len(set(names)) != len(names) or set(names) != set(required)):
            names_valid = False
            values.append(position)
        elif required and names:
            order = sorted(range(len(names)), key=lambda index: names[index])
            ordered = [position[index] for index in order]
            if [names[index] for index in order] != required:
                # profile order is authoritative; reorder by profile names
                by_name = dict(zip(names, position, strict=False))
                ordered = [float(by_name[name]) for name in required]
            values.append(ordered)
        else:
            values.append(position)
        times.append(int(log_time))
    if required and not names_valid:
        diagnostics.append(
            diagnostic(
                "DIMENSION_MISMATCH",
                Severity.ERROR,
                "/joint_states joint names do not match the resolved robot profile",
                details={"observed_joint_names": sorted(set(observed)), "required": required},
            )
        )
    if not samples:
        diagnostics.append(
            diagnostic(
                "SYNC_MISSING_STATE",
                Severity.ERROR,
                "the MCAP file contains no /joint_states messages",
                details={"topic": TOPIC_JOINT_STATES},
            )
        )
    return times, values, observed, diagnostics


def _read_actions(
    samples: list[tuple[int, Any]],
    profile: Any,
) -> tuple[list[int], list[list[float]], list[dict[str, Any]]]:
    diagnostics: list[dict[str, Any]] = []
    times = [int(log_time) for log_time, _message in samples]
    values = [[float(value) for value in message.data] for _log_time, message in samples]
    expected = int(profile.action_dim) if profile is not None else None
    bad = sorted({len(row) for row in values if expected is not None and len(row) != expected})
    if bad:
        diagnostics.append(
            diagnostic(
                "DIMENSION_MISMATCH",
                Severity.ERROR,
                "/policy/action length does not match the resolved robot profile",
                details={"observed_action_dims": bad, "required": expected},
            )
        )
    if not samples:
        diagnostics.append(
            diagnostic(
                "SYNC_STALE_ACTION",
                Severity.ERROR,
                "the MCAP file contains no /policy/action messages",
                details={"topic": TOPIC_ACTION},
            )
        )
    return times, values, diagnostics


def _monotonic_diagnostics(
    front_times: list[int], state_times: list[int], action_times: list[int]
) -> list[dict[str, Any]]:
    diagnostics: list[dict[str, Any]] = []
    for name, times in (("camera:front", front_times), ("state", state_times), ("action", action_times)):
        violations = [
            index for index in range(1, len(times)) if times[index] <= times[index - 1]
        ]
        if violations:
            diagnostics.append(
                diagnostic(
                    "TIMESTAMP_NON_MONOTONIC",
                    Severity.ERROR,
                    f"{name} log_time is not strictly increasing",
                    details={
                        "stream": name,
                        "violation_indexes": violations[:20],
                        "violation_count": len(violations),
                    },
                )
            )
    return diagnostics


__all__ = [
    "ACTION_TYPE",
    "CLOCK_DOMAIN",
    "DIAG_POINTCLOUD_IGNORED",
    "DIAG_UNLISTED_TOPIC",
    "IMAGE_TYPE",
    "JOINT_STATE_TYPE",
    "McapProfileV2Parser",
    "POINTCLOUD_TYPE",
    "TOPIC_ACTION",
    "TOPIC_FRONT",
    "TOPIC_JOINT_STATES",
    "TOPIC_POINTS",
    "TOPIC_WRIST",
]
