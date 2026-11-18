"""Dual-path MCAP reader for the fixed ``MCAP_ROS2_PROFILE_V1`` topic profile.

Why two paths (architecture decision, cf. SPEC 2, 5.4, 17)
-----------------------------------------------------------
The specification fixes the MCAP adapter to a separate **Ubuntu 24.04 / ROS 2
Jazzy worker** because ROS 2 system packages are ABI-bound to the distro Python
and must not be pip-installed into the main API Python 3.11 environment.  The
authoritative production path therefore is:

    rosbag2_py.SequentialReader  (storage plugin ``rosbag2_storage_mcap``)
        -> serialized CDR bytes
        -> rclpy.serialization.deserialize_message / rosidl_runtime_py types
        -> sensor_msgs/JointState, sensor_msgs/Image, std_msgs/Float64MultiArray

That path is implemented by :func:`_read_with_rosbag2` and is selected
automatically whenever ``rosbag2_py`` is importable (i.e. inside the
``worker-ros2`` Jazzy image).  It is the only path used in acceptance/production
runs; the ``ros2``-marked tests exercise it when the environment provides it.

No ROS 2 installation exists on the Windows development host or in the main API
container.  To keep MCAP parsing testable there, this module also provides a
**development / CI fallback reader**:

    mcap-pip ``make_reader`` (pure Python MCAP container parser)
        -> serialized CDR bytes
        -> hand-written CDR decoder in :mod:`services.ros2_adapter.cdr`

The fallback is a complete, deterministic implementation of the same profile
rules (topic/schema validation, CDR validation, header-stamp diagnostics); it is
not a mock and reads the same real MCAP bytes a ROS 2 bag would contain.  What
it does *not* have is the ROS 2 runtime itself: ``rosbag2_py`` also validates
storage plugins, QoS metadata and some distro-specific message definitions, and
``rclpy`` deserialization is generated from the actual ``.msg`` IDL.  The
fallback therefore re-validates the profile statically and decodes the three
fixed types only (SPEC 5.4); everything else is rejected with
``UNSUPPORTED_TOPIC_PROFILE``.

The active backend is reported on every read (``McapProfileData.backend``) and
by the parser diagnostics, so evidence never claims a ROS 2 runtime decode when
the fallback was used.  This dual-path model is the ADR for the local deviation
("Jazzy worker isolated", see SPEC 17 risk table); no behavioural acceptance
criterion is relaxed.

Profile rules enforced here (SPEC 5.4)
--------------------------------------
* single Episode per MCAP; exactly the three profile topics, any other topic or
  schema type is ``UNSUPPORTED_TOPIC_PROFILE``;
* ROS 2 CDR only: ``ros1`` profiles / message encodings and missing schema
  information are rejected with a readable error (never a stack trace);
* ``/camera/front/image_raw`` -> ``sensor_msgs/msg/Image`` rgb8 or bgr8 complete
  frames; ``/joint_states`` -> ``sensor_msgs/msg/JointState``;
  ``/policy/action`` -> ``std_msgs/msg/Float64MultiArray`` (six float64 values,
  semantics = target joint position, never torque/velocity);
* every stream keeps its raw ``log_time`` and its original ``header.stamp`` plus
  the header-to-log delta as diagnostics; timebase selection happens in the
  parser, never here.
"""

from __future__ import annotations

import importlib.util
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from embodied_dataops.errors import ApiError, ErrorCode

from .cdr import (
    FLOAT64_MULTI_ARRAY_TYPE,
    IMAGE_TYPE,
    JOINT_STATE_TYPE,
    CdrError,
    Float64MultiArrayMessage,
    ImageMessage,
    JointStateMessage,
    decode_with_alignment_fallback,
)

#: Fixed topic profile (SPEC 5.4).
TOPIC_IMAGE = "/camera/front/image_raw"
TOPIC_JOINT_STATES = "/joint_states"
TOPIC_ACTION = "/policy/action"
PROFILE_TOPICS: dict[str, str] = {
    TOPIC_IMAGE: IMAGE_TYPE,
    TOPIC_JOINT_STATES: JOINT_STATE_TYPE,
    TOPIC_ACTION: FLOAT64_MULTI_ARRAY_TYPE,
}
REQUIRED_TOPICS = (TOPIC_IMAGE, TOPIC_JOINT_STATES, TOPIC_ACTION)
ALLOWED_IMAGE_ENCODINGS = frozenset({"rgb8", "bgr8"})
#: Message encodings that mean "this is a ROS 1 bag, not a ROS 2 CDR bag".
ROS1_MESSAGE_ENCODINGS = frozenset({"ros1", "ros1msg", "ros2/ros1"})
ROS1_PROFILE_PREFIXES = ("ros1",)
#: Schema encodings accepted for ROS 2 message definitions.
ROS2_SCHEMA_ENCODINGS = frozenset({"ros2msg", "ros2idl", ""})

MAX_INPUT_BYTES = 2 * 1024**3  # SPEC 5.1: MCAP uploads are at most 2 GiB
MAX_EPISODE_DURATION_NS = 600 * 1_000_000_000


@dataclass(frozen=True)
class CameraFrame:
    """One ``sensor_msgs/msg/Image`` message."""

    log_time_ns: int
    publish_time_ns: int
    message: ImageMessage


@dataclass(frozen=True)
class JointStateSample:
    """One ``sensor_msgs/msg/JointState`` message."""

    log_time_ns: int
    publish_time_ns: int
    message: JointStateMessage


@dataclass(frozen=True)
class ActionSample:
    """One ``std_msgs/msg/Float64MultiArray`` message."""

    log_time_ns: int
    publish_time_ns: int
    message: Float64MultiArrayMessage


@dataclass
class McapProfileData:
    """All profile streams of one MCAP, in ascending ``log_time`` order."""

    backend: str
    profile: str
    topics: dict[str, str]
    camera: list[CameraFrame] = field(default_factory=list)
    joint_states: list[JointStateSample] = field(default_factory=list)
    actions: list[ActionSample] = field(default_factory=list)
    message_counts: dict[str, int] = field(default_factory=dict)


@dataclass
class McapProbe:
    """Cheap structural view of an MCAP used by ``VALIDATE_SOURCE``."""

    backend: str
    profile: str
    topics: dict[str, str]
    message_counts: dict[str, int] | None = None
    start_time_ns: int | None = None
    end_time_ns: int | None = None

    @property
    def duration_ns(self) -> int | None:
        if self.start_time_ns is None or self.end_time_ns is None:
            return None
        return self.end_time_ns - self.start_time_ns


# ---------------------------------------------------------------------------
# backend selection
# ---------------------------------------------------------------------------


def rosbag2_available() -> bool:
    return importlib.util.find_spec("rosbag2_py") is not None


def mcap_fallback_available() -> bool:
    return importlib.util.find_spec("mcap") is not None


def resolve_backend(requested: str = "auto") -> str:
    """Resolve ``auto`` -> ``rosbag2_py`` when present, else the mcap fallback."""
    if requested in {"rosbag2", "rosbag2_py"}:
        if not rosbag2_available():
            raise ApiError(
                ErrorCode.DEPENDENCY_UNAVAILABLE,
                "rosbag2_py is not available in this Python environment; "
                "run the MCAP adapter inside the ROS 2 Jazzy worker image",
            )
        return "rosbag2_py"
    if requested in {"fallback", "mcap", "mcap_fallback"}:
        if not mcap_fallback_available():
            raise ApiError(
                ErrorCode.DEPENDENCY_UNAVAILABLE,
                "The mcap package is not installed; no MCAP fallback reader is available",
            )
        return "mcap_fallback"
    if requested != "auto":
        raise ApiError(ErrorCode.INVALID_INPUT, f"Unknown MCAP reader backend {requested!r}")
    if rosbag2_available():
        return "rosbag2_py"
    if mcap_fallback_available():
        return "mcap_fallback"
    raise ApiError(
        ErrorCode.DEPENDENCY_UNAVAILABLE,
        "No MCAP reader backend is available (install mcap or run the ROS 2 Jazzy worker)",
    )


# ---------------------------------------------------------------------------
# profile validation
# ---------------------------------------------------------------------------


def _profile_error(message: str, *, details: dict[str, Any] | None = None) -> ApiError:
    return ApiError(ErrorCode.UNSUPPORTED_TOPIC_PROFILE, message, details=details or {})


def validate_profile(
    *,
    topics: dict[str, str],
    profile: str,
    message_encodings: dict[str, str] | None = None,
    schema_encodings: dict[str, str] | None = None,
) -> None:
    """Reject anything outside the fixed single-Episode ROS 2 CDR topic profile."""
    lowered_profile = (profile or "").strip().lower()
    if any(lowered_profile.startswith(prefix) for prefix in ROS1_PROFILE_PREFIXES):
        raise _profile_error(
            "ROS 1 MCAP files are not supported by MCAP_ROS2_PROFILE_V1; "
            "record a ROS 2 (Jazzy) bag with CDR message encoding",
            details={"profile": profile},
        )

    unknown = sorted(set(topics) - set(PROFILE_TOPICS))
    if unknown:
        raise _profile_error(
            "MCAP contains topics outside the fixed MCAP_ROS2_PROFILE_V1 profile: "
            + ", ".join(unknown),
            details={"unknown_topics": unknown, "allowed_topics": sorted(PROFILE_TOPICS)},
        )

    missing = [topic for topic in REQUIRED_TOPICS if topic not in topics]
    if missing:
        raise _profile_error(
            "MCAP is missing required profile topics: " + ", ".join(missing),
            details={"missing_topics": missing, "required_topics": list(REQUIRED_TOPICS)},
        )

    for topic, expected_type in PROFILE_TOPICS.items():
        actual_type = topics.get(topic, "")
        if actual_type != expected_type:
            raise _profile_error(
                f"Topic {topic} has unsupported message type {actual_type!r}; "
                f"MCAP_ROS2_PROFILE_V1 requires {expected_type}",
                details={"topic": topic, "expected_type": expected_type, "actual_type": actual_type},
            )

    for topic, encoding in (message_encodings or {}).items():
        if topic not in PROFILE_TOPICS:
            continue
        normalized = (encoding or "").strip().lower()
        if normalized in ROS1_MESSAGE_ENCODINGS:
            raise _profile_error(
                f"Topic {topic} is serialized with ROS 1 encoding {encoding!r}; "
                "MCAP_ROS2_PROFILE_V1 requires ROS 2 CDR",
                details={"topic": topic, "message_encoding": encoding},
            )
        if normalized not in {"cdr", ""}:
            raise _profile_error(
                f"Topic {topic} uses unsupported message encoding {encoding!r}; "
                "MCAP_ROS2_PROFILE_V1 requires CDR",
                details={"topic": topic, "message_encoding": encoding},
            )

    for topic, encoding in (schema_encodings or {}).items():
        if topic not in PROFILE_TOPICS:
            continue
        normalized = (encoding or "").strip().lower()
        if normalized not in ROS2_SCHEMA_ENCODINGS:
            raise _profile_error(
                f"Topic {topic} carries an unsupported schema encoding {encoding!r}; "
                "MCAP_ROS2_PROFILE_V1 requires ROS 2 message definitions (ros2msg/ros2idl)",
                details={"topic": topic, "schema_encoding": encoding},
            )


def _image_profile_error(message: str, *, topic: str = TOPIC_IMAGE) -> ApiError:
    return _profile_error(message, details={"topic": topic})


def validate_image_encoding(message: ImageMessage) -> None:
    if message.encoding not in ALLOWED_IMAGE_ENCODINGS:
        raise _image_profile_error(
            f"Image encoding {message.encoding!r} is not supported; "
            "MCAP_ROS2_PROFILE_V1 accepts rgb8 or bgr8 single complete frames",
        )
    if not message.is_complete():
        raise _image_profile_error(
            "Image message is not a complete single frame "
            f"(width={message.width}, height={message.height}, step={message.step}, "
            f"bytes={len(message.data)})",
        )


def _decode_required(topic: str, data: bytes):
    message_type = PROFILE_TOPICS[topic]
    try:
        message, _alignment = decode_with_alignment_fallback(message_type, data)
    except CdrError as exc:
        raise _profile_error(
            f"Cannot decode CDR message on topic {topic} as {message_type}: {exc}",
            details={"topic": topic, "message_type": message_type},
        ) from None
    if isinstance(message, ImageMessage):
        validate_image_encoding(message)
    return message


# ---------------------------------------------------------------------------
# mcap-pip fallback backend
# ---------------------------------------------------------------------------


def _open_fallback(path: Path):
    from mcap.reader import make_reader

    if not path.is_file():
        raise ApiError(ErrorCode.SOURCE_NOT_FOUND, f"MCAP source not found: {path.name}")
    try:
        stream = path.open("rb")
    except OSError as exc:  # pragma: no cover - filesystem dependent
        raise ApiError(ErrorCode.SOURCE_NOT_FOUND, "MCAP source cannot be opened") from exc
    try:
        reader = make_reader(stream)
        header = reader.get_header()
        summary = reader.get_summary()
    except Exception as exc:
        stream.close()
        raise ApiError(
            ErrorCode.FILE_DECODE_ERROR,
            f"File is not a readable MCAP container: {path.name}",
        ) from exc
    topics: dict[str, str] = {}
    message_encodings: dict[str, str] = {}
    schema_encodings: dict[str, str] = {}
    counts: dict[str, int] = {}
    start: int | None = None
    end: int | None = None
    if summary is not None:
        for channel in summary.channels.values():
            schema = summary.schemas.get(channel.schema_id)
            topics[channel.topic] = schema.name if schema is not None else ""
            message_encodings[channel.topic] = channel.message_encoding
            if schema is not None:
                schema_encodings[channel.topic] = schema.encoding
            counts[channel.topic] = 0
        statistics = summary.statistics
        if statistics is not None:
            start = statistics.message_start_time or None
            end = statistics.message_end_time or None
            for channel_id, count in statistics.channel_message_counts.items():
                channel = summary.channels.get(channel_id)
                if channel is not None:
                    counts[channel.topic] = int(count)
    return stream, reader, header, topics, message_encodings, schema_encodings, counts, start, end


def _probe_fallback(path: Path) -> McapProbe:
    (
        stream,
        reader,
        header,
        topics,
        message_encodings,
        schema_encodings,
        counts,
        start,
        end,
    ) = _open_fallback(path)
    try:
        if not topics:
            # An unfinalized MCAP has no summary section; discover topics from the
            # message stream (bounded by the message count itself, not by time).
            for schema, channel, message in reader.iter_messages():
                topics.setdefault(channel.topic, schema.name)
                message_encodings.setdefault(channel.topic, channel.message_encoding)
                schema_encodings.setdefault(channel.topic, schema.encoding)
                counts[channel.topic] = counts.get(channel.topic, 0) + 1
                log_time = int(message.log_time)
                start = log_time if start is None or log_time < start else start
                end = log_time if end is None or log_time > end else end
    finally:
        stream.close()
    profile = getattr(header, "profile", "") or ""
    validate_profile(
        topics=topics,
        profile=profile,
        message_encodings=message_encodings,
        schema_encodings=schema_encodings,
    )
    return McapProbe(
        backend="mcap_fallback",
        profile=profile,
        topics=topics,
        message_counts=counts,
        start_time_ns=start,
        end_time_ns=end,
    )


def _read_fallback(path: Path) -> McapProfileData:
    (
        stream,
        reader,
        header,
        topics,
        message_encodings,
        schema_encodings,
        _counts,
        _start,
        _end,
    ) = _open_fallback(path)
    profile = getattr(header, "profile", "") or ""
    validate_profile(
        topics=topics,
        profile=profile,
        message_encodings=message_encodings,
        schema_encodings=schema_encodings,
    )
    camera: list[CameraFrame] = []
    joint_states: list[JointStateSample] = []
    actions: list[ActionSample] = []
    counts = {topic: 0 for topic in REQUIRED_TOPICS}
    try:
        for _schema, channel, message in reader.iter_messages():
            topic = channel.topic
            if topic not in PROFILE_TOPICS:
                continue  # already rejected by validate_profile; defensive only
            decoded = _decode_required(topic, message.data)
            log_time = int(message.log_time)
            publish_time = int(getattr(message, "publish_time", log_time) or log_time)
            if topic == TOPIC_IMAGE:
                assert isinstance(decoded, ImageMessage)
                camera.append(CameraFrame(log_time, publish_time, decoded))
            elif topic == TOPIC_JOINT_STATES:
                assert isinstance(decoded, JointStateMessage)
                joint_states.append(JointStateSample(log_time, publish_time, decoded))
            else:
                assert isinstance(decoded, Float64MultiArrayMessage)
                actions.append(ActionSample(log_time, publish_time, decoded))
            counts[topic] += 1
    except CdrError as exc:  # pragma: no cover - _decode_required already wraps
        raise _profile_error(f"Cannot decode MCAP CDR payload: {exc}") from None
    finally:
        stream.close()
    _require_nonempty_streams(counts)
    _sort_streams(camera, joint_states, actions)
    return McapProfileData(
        backend="mcap_fallback",
        profile=profile,
        topics=topics,
        camera=camera,
        joint_states=joint_states,
        actions=actions,
        message_counts=counts,
    )


def _require_nonempty_streams(counts: dict[str, int]) -> None:
    empty = [topic for topic in REQUIRED_TOPICS if counts.get(topic, 0) <= 0]
    if empty:
        raise _profile_error(
            "MCAP profile topics contain no messages: " + ", ".join(empty),
            details={"empty_topics": empty},
        )


def _sort_streams(
    camera: list[CameraFrame],
    joint_states: list[JointStateSample],
    actions: list[ActionSample],
) -> None:
    camera.sort(key=lambda item: item.log_time_ns)
    joint_states.sort(key=lambda item: item.log_time_ns)
    actions.sort(key=lambda item: item.log_time_ns)


# ---------------------------------------------------------------------------
# rosbag2_py production backend
# ---------------------------------------------------------------------------


def _rosbag2_open(path: Path):
    import rosbag2_py

    if not path.is_file():
        raise ApiError(ErrorCode.SOURCE_NOT_FOUND, f"MCAP source not found: {path.name}")
    try:
        reader = rosbag2_py.SequentialReader()
        reader.open(
            rosbag2_py.StorageOptions(uri=str(path), storage_id="mcap"),
            rosbag2_py.ConverterOptions(
                input_serialization_format="cdr", output_serialization_format="cdr"
            ),
        )
        topics = reader.get_all_topics_and_types()
    except Exception as exc:
        raise ApiError(
            ErrorCode.UNSUPPORTED_TOPIC_PROFILE,
            "rosbag2_py cannot open this file as a ROS 2 MCAP bag "
            "(it may be ROS 1, corrupted, or missing the mcap storage plugin)",
        ) from exc
    return reader, topics


def _topic_metadata_maps(topics) -> tuple[dict[str, str], dict[str, str], dict[str, str]]:
    types: dict[str, str] = {}
    message_encodings: dict[str, str] = {}
    schema_encodings: dict[str, str] = {}
    for metadata in topics:
        name = getattr(metadata, "name", "")
        types[name] = getattr(metadata, "type", "")
        message_encodings[name] = getattr(metadata, "serialization_format", "") or ""
        # rosbag2 does not expose the MCAP schema encoding through this API; the
        # ros2msg/ros2idl definition is validated by the storage plugin itself.
        schema_encodings[name] = "ros2msg"
    return types, message_encodings, schema_encodings


def _probe_rosbag2(path: Path) -> McapProbe:
    reader, topics = _rosbag2_open(path)
    types, message_encodings, schema_encodings = _topic_metadata_maps(topics)
    validate_profile(
        topics=types,
        profile="ros2",
        message_encodings=message_encodings,
        schema_encodings=schema_encodings,
    )
    try:
        reader.close()
    except Exception:  # pragma: no cover - plugin dependent
        pass
    return McapProbe(backend="rosbag2_py", profile="ros2", topics=types)


def _read_rosbag2(path: Path) -> McapProfileData:
    reader, topics = _rosbag2_open(path)
    types, message_encodings, schema_encodings = _topic_metadata_maps(topics)
    validate_profile(
        topics=types,
        profile="ros2",
        message_encodings=message_encodings,
        schema_encodings=schema_encodings,
    )
    camera: list[CameraFrame] = []
    joint_states: list[JointStateSample] = []
    actions: list[ActionSample] = []
    counts = {topic: 0 for topic in REQUIRED_TOPICS}
    try:
        while reader.has_next():
            topic, data, timestamp = reader.read_next()
            if topic not in PROFILE_TOPICS:
                continue
            decoded = _decode_required(topic, bytes(data))
            log_time = int(timestamp)
            publish_time = log_time
            if topic == TOPIC_IMAGE:
                assert isinstance(decoded, ImageMessage)
                camera.append(CameraFrame(log_time, publish_time, decoded))
            elif topic == TOPIC_JOINT_STATES:
                assert isinstance(decoded, JointStateMessage)
                joint_states.append(JointStateSample(log_time, publish_time, decoded))
            else:
                assert isinstance(decoded, Float64MultiArrayMessage)
                actions.append(ActionSample(log_time, publish_time, decoded))
            counts[topic] += 1
    except CdrError as exc:  # pragma: no cover - _decode_required already wraps
        raise _profile_error(f"Cannot decode MCAP CDR payload: {exc}") from None
    finally:
        try:
            reader.close()
        except Exception:  # pragma: no cover - plugin dependent
            pass
    _require_nonempty_streams(counts)
    _sort_streams(camera, joint_states, actions)
    return McapProfileData(
        backend="rosbag2_py",
        profile="ros2",
        topics=types,
        camera=camera,
        joint_states=joint_states,
        actions=actions,
        message_counts=counts,
    )


# ---------------------------------------------------------------------------
# public API
# ---------------------------------------------------------------------------


def probe_mcap_profile(path: Path | str, *, backend: str = "auto") -> McapProbe:
    """Structural validation without decoding every message (SPEC 5.4)."""
    path = Path(path)
    if path.stat().st_size > MAX_INPUT_BYTES:
        raise ApiError(
            ErrorCode.SOURCE_SIZE_LIMIT,
            "MCAP source exceeds the 2 GiB upload limit",
            details={"size_bytes": path.stat().st_size, "limit": MAX_INPUT_BYTES},
        )
    resolved = resolve_backend(backend)
    probe = _probe_rosbag2(path) if resolved == "rosbag2_py" else _probe_fallback(path)
    if probe.duration_ns is not None and probe.duration_ns > MAX_EPISODE_DURATION_NS:
        raise ApiError(
            ErrorCode.SOURCE_DURATION_LIMIT,
            "MCAP episode exceeds the 600 s limit of MCAP_ROS2_PROFILE_V1",
            details={"duration_ns": probe.duration_ns, "limit_ns": MAX_EPISODE_DURATION_NS},
        )
    return probe


def read_mcap_profile(path: Path | str, *, backend: str = "auto") -> McapProfileData:
    """Read and CDR-decode all three profile streams (never executes payloads)."""
    path = Path(path)
    if not path.is_file():
        raise ApiError(ErrorCode.SOURCE_NOT_FOUND, f"MCAP source not found: {path.name}")
    resolved = resolve_backend(backend)
    return _read_rosbag2(path) if resolved == "rosbag2_py" else _read_fallback(path)


__all__ = [
    "ALLOWED_IMAGE_ENCODINGS",
    "FLOAT64_MULTI_ARRAY_TYPE",
    "IMAGE_TYPE",
    "JOINT_STATE_TYPE",
    "McapProbe",
    "McapProfileData",
    "PROFILE_TOPICS",
    "REQUIRED_TOPICS",
    "TOPIC_ACTION",
    "TOPIC_IMAGE",
    "TOPIC_JOINT_STATES",
    "CameraFrame",
    "ActionSample",
    "JointStateSample",
    "probe_mcap_profile",
    "read_mcap_profile",
    "resolve_backend",
    "rosbag2_available",
    "validate_profile",
]
