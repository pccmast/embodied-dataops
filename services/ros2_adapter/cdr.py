"""Hand-written OMG CDR codec for the three ``MCAP_ROS2_PROFILE_V1`` messages.

The production reader for the ``worker-ros2`` image deserializes messages with
the official ROS 2 stack (``rosbag2_py`` + ``rclpy.serialization`` /
``rosidl_runtime_py`` -- see :mod:`services.ros2_adapter.reader`).  That stack
is *not* importable from the main API Python 3.11 environment (SPEC 2, 5.4 and
the ROS 2 ABI risk entry in SPEC 17), so the repository also ships a pure
Python CDR codec:

* :class:`CdrWriter` / the ``encode_*`` helpers are used by
  ``fixtures/make_mcap_ros2.py`` to produce real ROS 2 CDR MCAP fixtures that
  the local fallback reader can consume without any ROS installation;
* :class:`CdrReader` / the ``decode_*`` helpers let ``mcap``-pip based local
  development (and CI) read the same bytes back.

Wire format (OMG CDR / DDS-XTypes, little endian by default)
------------------------------------------------------------
A serialized ROS 2 message is a 4-byte *encapsulation header* followed by the
CDR stream::

    byte 0    0x00
    byte 1    0x00 = CDR big endian, 0x01 = CDR little endian
    bytes 2-3 0x0000 options

Alignment of every primitive is computed from the **start of the serialized
payload (offset 0 = first encapsulation byte)**; after the header the stream
position is 4, so an ``float64`` member inserted at that point is padded by 4
bytes.  This matches Fast CDR (the implementation used by ROS 2 Jazzy / Fast
DDS).  Because third-party bags have historically been produced by other DDS
vendors whose serializers align relative to the byte *after* the encapsulation,
:class:`CdrReader` accepts an ``alignment_origin`` override.  The reader in
:mod:`services.ros2_adapter.reader` first decodes with the Fast CDR convention
and only then retries the alternate base; both attempts are validated strictly
(exact byte counts / image sizes), so a wrong base cannot silently succeed.

Message layouts (``ros2msg``)
-----------------------------
``sensor_msgs/msg/Image``::

    std_msgs/Header header
      builtin_interfaces/Time stamp {int32 sec, uint32 nanosec}
      string frame_id
    uint32 height
    uint32 width
    string encoding
    uint8  is_bigendian
    uint32 step
    uint8[] data

``sensor_msgs/msg/JointState``::

    std_msgs/Header header
    string[]  name
    float64[] position
    float64[] velocity
    float64[] effort

``std_msgs/msg/Float64MultiArray`` (no header) ::

    std_msgs/MultiArrayLayout layout
      std_msgs/MultiArrayDimension[] dim {string label, uint32 size, uint32 stride}
      uint32 data_offset
    float64[] data

Every helper is pure, has no ROS dependency and never executes the payload.
"""

from __future__ import annotations

import struct
from collections.abc import Sequence
from dataclasses import dataclass

#: Message type names used by the fixed topic profile (SPEC 5.4).
IMAGE_TYPE = "sensor_msgs/msg/Image"
JOINT_STATE_TYPE = "sensor_msgs/msg/JointState"
FLOAT64_MULTI_ARRAY_TYPE = "std_msgs/msg/Float64MultiArray"

#: ``ros2msg`` schema definitions registered in generated MCAP fixtures.
ROS2_MSG_DEFINITIONS: dict[str, str] = {
    IMAGE_TYPE: (
        "std_msgs/Header header\n"
        "uint32 height\n"
        "uint32 width\n"
        "string encoding\n"
        "uint8 is_bigendian\n"
        "uint32 step\n"
        "uint8[] data\n"
        "\n"
        "================================================================================\n"
        "MSG: std_msgs/msg/Header\n"
        "builtin_interfaces/Time stamp\n"
        "string frame_id\n"
        "\n"
        "================================================================================\n"
        "MSG: builtin_interfaces/msg/Time\n"
        "int32 sec\n"
        "uint32 nanosec\n"
    ),
    JOINT_STATE_TYPE: (
        "std_msgs/Header header\n"
        "string[] name\n"
        "float64[] position\n"
        "float64[] velocity\n"
        "float64[] effort\n"
        "\n"
        "================================================================================\n"
        "MSG: std_msgs/msg/Header\n"
        "builtin_interfaces/Time stamp\n"
        "string frame_id\n"
        "\n"
        "================================================================================\n"
        "MSG: builtin_interfaces/msg/Time\n"
        "int32 sec\n"
        "uint32 nanosec\n"
    ),
    FLOAT64_MULTI_ARRAY_TYPE: (
        "MultiArrayLayout layout\n"
        "float64[] data\n"
        "\n"
        "================================================================================\n"
        "MSG: std_msgs/msg/MultiArrayLayout\n"
        "MultiArrayDimension[] dim\n"
        "uint32 data_offset\n"
        "\n"
        "================================================================================\n"
        "MSG: std_msgs/msg/MultiArrayDimension\n"
        "string label\n"
        "uint32 size\n"
        "uint32 stride\n"
    ),
}

#: Defensive decoder limits: corrupt/attacker-controlled lengths must never
#: allocate unbounded memory.
MAX_SEQUENCE_ELEMENTS = 4_000_000
MAX_STRING_BYTES = 1_000_000
MAX_OCTET_SEQUENCE_BYTES = 128 * 1024 * 1024
MAX_TOTAL_PAYLOAD_BYTES = 256 * 1024 * 1024


class CdrError(ValueError):
    """Raised when a payload is not a decodable CDR message of the requested type."""


# ---------------------------------------------------------------------------
# Messages
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ImageMessage:
    """Decoded ``sensor_msgs/msg/Image``."""

    header_stamp_ns: int | None
    frame_id: str
    height: int
    width: int
    encoding: str
    is_bigendian: bool
    step: int
    data: bytes

    @property
    def channels(self) -> int:
        return {"rgb8": 3, "bgr8": 3, "mono8": 1}.get(self.encoding, 0)

    def is_complete(self) -> bool:
        return self.channels > 0 and self.step >= self.width * self.channels and len(self.data) == self.step * self.height


@dataclass(frozen=True)
class JointStateMessage:
    """Decoded ``sensor_msgs/msg/JointState``."""

    header_stamp_ns: int | None
    frame_id: str
    names: tuple[str, ...]
    position: tuple[float, ...]
    velocity: tuple[float, ...]
    effort: tuple[float, ...]


@dataclass(frozen=True)
class Float64MultiArrayMessage:
    """Decoded ``std_msgs/msg/Float64MultiArray``."""

    data: tuple[float, ...]
    layout_dimensions: tuple[tuple[str, int, int], ...]
    data_offset: int


def stamp_to_ns(sec: int, nanosec: int) -> int:
    return int(sec) * 1_000_000_000 + int(nanosec)


def ns_to_stamp(value_ns: int) -> tuple[int, int]:
    sec, nanosec = divmod(int(value_ns), 1_000_000_000)
    return sec, nanosec


# ---------------------------------------------------------------------------
# Writer
# ---------------------------------------------------------------------------


class CdrWriter:
    """Little/big endian CDR serializer with Fast CDR alignment semantics."""

    __slots__ = ("_buffer", "_offset", "_fmt")

    def __init__(self, little_endian: bool = True) -> None:
        self._fmt = "<" if little_endian else ">"
        self._buffer = bytearray(b"\x00\x01\x00\x00" if little_endian else b"\x00\x00\x00\x00")
        self._offset = 4

    # -- primitives --------------------------------------------------------
    def _align(self, alignment: int) -> None:
        pad = (-self._offset) % alignment
        if pad:
            self._buffer.extend(b"\x00" * pad)
            self._offset += pad

    def _pack(self, fmt: str, value: int | float) -> None:
        size = struct.calcsize(fmt)
        self._align(size)
        self._buffer.extend(struct.pack(self._fmt + fmt, value))
        self._offset += size

    def write_uint8(self, value: int) -> None:
        self._pack("B", int(value))

    def write_int8(self, value: int) -> None:
        self._pack("b", int(value))

    def write_uint16(self, value: int) -> None:
        self._pack("H", int(value))

    def write_int32(self, value: int) -> None:
        self._pack("i", int(value))

    def write_uint32(self, value: int) -> None:
        self._pack("I", int(value))

    def write_int64(self, value: int) -> None:
        self._pack("q", int(value))

    def write_uint64(self, value: int) -> None:
        self._pack("Q", int(value))

    def write_float32(self, value: float) -> None:
        self._pack("f", float(value))

    def write_float64(self, value: float) -> None:
        self._pack("d", float(value))

    def write_string(self, value: str) -> None:
        encoded = value.encode("utf-8")
        self.write_uint32(len(encoded) + 1)  # CDR strings include the NUL terminator
        self._buffer.extend(encoded)
        self._buffer.append(0)
        self._offset += len(encoded) + 1

    # -- sequences ---------------------------------------------------------
    def write_octet_sequence(self, data: bytes) -> None:
        self.write_uint32(len(data))
        self._buffer.extend(data)
        self._offset += len(data)

    def write_string_sequence(self, values: Sequence[str]) -> None:
        self.write_uint32(len(values))
        for value in values:
            self.write_string(value)

    def write_float64_sequence(self, values: Sequence[float]) -> None:
        # Fast CDR aligns each element as it is serialized, so an *empty*
        # sequence contributes only its uint32 length and no padding (this is
        # also what the generated ROS 2 deserializers and `rosbags` assume).
        self.write_uint32(len(values))
        for value in values:
            self._align(8)
            self._buffer.extend(struct.pack(self._fmt + "d", float(value)))
            self._offset += 8

    def write_uint32_sequence(self, values: Sequence[int]) -> None:
        self.write_uint32(len(values))
        for value in values:
            self._align(4)
            self._buffer.extend(struct.pack(self._fmt + "I", int(value)))
            self._offset += 4

    # -- result ------------------------------------------------------------
    def to_bytes(self) -> bytes:
        return bytes(self._buffer)


def write_header(writer: CdrWriter, stamp_ns: int, frame_id: str) -> None:
    sec, nanosec = ns_to_stamp(stamp_ns)
    writer.write_int32(sec)
    writer.write_uint32(nanosec)
    writer.write_string(frame_id)


def encode_image(
    *,
    width: int,
    height: int,
    encoding: str,
    data: bytes,
    step: int | None = None,
    header_stamp_ns: int = 0,
    frame_id: str = "camera_front",
    little_endian: bool = True,
) -> bytes:
    """Encode one ``sensor_msgs/msg/Image`` (rgb8/bgr8 complete frame)."""
    step = step if step is not None else width * 3
    writer = CdrWriter(little_endian=little_endian)
    write_header(writer, header_stamp_ns, frame_id)
    writer.write_uint32(height)
    writer.write_uint32(width)
    writer.write_string(encoding)
    writer.write_uint8(0)
    writer.write_uint32(step)
    writer.write_octet_sequence(data)
    return writer.to_bytes()


def encode_joint_state(
    *,
    names: Sequence[str],
    position: Sequence[float],
    velocity: Sequence[float] | None = None,
    effort: Sequence[float] | None = None,
    header_stamp_ns: int = 0,
    frame_id: str = "",
    little_endian: bool = True,
) -> bytes:
    """Encode one ``sensor_msgs/msg/JointState``."""
    writer = CdrWriter(little_endian=little_endian)
    write_header(writer, header_stamp_ns, frame_id)
    writer.write_string_sequence(list(names))
    writer.write_float64_sequence(list(position))
    writer.write_float64_sequence(list(velocity or []))
    writer.write_float64_sequence(list(effort or []))
    return writer.to_bytes()


def encode_float64_multi_array(
    *,
    data: Sequence[float],
    little_endian: bool = True,
) -> bytes:
    """Encode one ``std_msgs/msg/Float64MultiArray`` with an empty layout."""
    writer = CdrWriter(little_endian=little_endian)
    writer.write_uint32(0)  # layout.dim sequence length
    writer.write_uint32(0)  # layout.data_offset
    writer.write_float64_sequence(list(data))
    return writer.to_bytes()


# ---------------------------------------------------------------------------
# Reader
# ---------------------------------------------------------------------------


class CdrReader:
    """Strict CDR deserializer.

    ``alignment_origin`` is ``0`` for the Fast CDR convention (alignment base is
    the first encapsulation byte) and ``4`` for DDS writers that align relative
    to the first byte after the encapsulation header.
    """

    __slots__ = ("_data", "_offset", "_fmt", "_origin")

    def __init__(self, data: bytes, *, alignment_origin: int = 0) -> None:
        if len(data) < 4:
            raise CdrError("payload is shorter than the 4-byte CDR encapsulation header")
        if len(data) > MAX_TOTAL_PAYLOAD_BYTES:
            raise CdrError("payload exceeds the decoder size limit")
        if data[0] != 0x00:
            raise CdrError("invalid CDR encapsulation identifier byte")
        if data[1] == 0x01:
            fmt = "<"
        elif data[1] == 0x00:
            fmt = ">"
        else:
            raise CdrError("unsupported CDR encapsulation representation identifier")
        self._data = memoryview(data)
        self._offset = 4
        self._fmt = fmt
        self._origin = alignment_origin

    @property
    def remaining(self) -> int:
        return len(self._data) - self._offset

    def _align(self, alignment: int) -> None:
        relative = self._offset - self._origin
        pad = (-relative) % alignment
        self._offset += pad
        if self._offset > len(self._data):
            raise CdrError("alignment padding exceeds the payload")

    def _unpack(self, fmt: str) -> int | float:
        size = struct.calcsize(fmt)
        self._align(size)
        if self._offset + size > len(self._data):
            raise CdrError("unexpected end of CDR payload")
        (value,) = struct.unpack_from(self._fmt + fmt, self._data, self._offset)
        self._offset += size
        return value

    def read_uint8(self) -> int:
        return int(self._unpack("B"))

    def read_int8(self) -> int:
        return int(self._unpack("b"))

    def read_uint16(self) -> int:
        return int(self._unpack("H"))

    def read_int32(self) -> int:
        return int(self._unpack("i"))

    def read_uint32(self) -> int:
        return int(self._unpack("I"))

    def read_int64(self) -> int:
        return int(self._unpack("q"))

    def read_uint64(self) -> int:
        return int(self._unpack("Q"))

    def read_float32(self) -> float:
        return float(self._unpack("f"))

    def read_float64(self) -> float:
        return float(self._unpack("d"))

    def read_string(self) -> str:
        length = self.read_uint32()
        if length == 0:
            return ""
        if length > MAX_STRING_BYTES:
            raise CdrError("CDR string exceeds the decoder size limit")
        if self._offset + length > len(self._data):
            raise CdrError("unexpected end of CDR payload while reading a string")
        raw = bytes(self._data[self._offset : self._offset + length])
        self._offset += length
        if raw.endswith(b"\x00"):
            raw = raw[:-1]
        try:
            return raw.decode("utf-8")
        except UnicodeDecodeError as exc:  # pragma: no cover - only for corrupt payloads
            raise CdrError("CDR string is not valid UTF-8") from exc

    def _read_sequence_length(self) -> int:
        length = self.read_uint32()
        if length > MAX_SEQUENCE_ELEMENTS:
            raise CdrError("CDR sequence exceeds the decoder element limit")
        return length

    def read_octet_sequence(self) -> bytes:
        length = self._read_sequence_length()
        if length > MAX_OCTET_SEQUENCE_BYTES:
            raise CdrError("octet sequence exceeds the decoder size limit")
        if self._offset + length > len(self._data):
            raise CdrError("unexpected end of CDR payload while reading an octet sequence")
        raw = bytes(self._data[self._offset : self._offset + length])
        self._offset += length
        return raw

    def read_string_sequence(self) -> tuple[str, ...]:
        return tuple(self.read_string() for _ in range(self._read_sequence_length()))

    def read_float64_sequence(self) -> tuple[float, ...]:
        length = self._read_sequence_length()
        # Fast CDR aligns the first element to the element size.
        return tuple(self.read_float64() for _ in range(length))


def read_header(reader: CdrReader) -> tuple[int, str]:
    sec = reader.read_int32()
    nanosec = reader.read_uint32()
    frame_id = reader.read_string()
    return stamp_to_ns(sec, nanosec), frame_id


def decode_image(data: bytes, *, alignment_origin: int = 0) -> ImageMessage:
    reader = CdrReader(data, alignment_origin=alignment_origin)
    stamp_ns, frame_id = read_header(reader)
    height = reader.read_uint32()
    width = reader.read_uint32()
    encoding = reader.read_string()
    is_bigendian = reader.read_uint8()
    step = reader.read_uint32()
    payload = reader.read_octet_sequence()
    if reader.remaining != 0:
        raise CdrError(f"Image payload has {reader.remaining} trailing bytes")
    message = ImageMessage(
        header_stamp_ns=stamp_ns,
        frame_id=frame_id,
        height=int(height),
        width=int(width),
        encoding=encoding,
        is_bigendian=bool(is_bigendian),
        step=int(step),
        data=payload,
    )
    if not message.is_complete():
        raise CdrError(
            "Image payload is not a complete single frame "
            f"(width={message.width}, height={message.height}, "
            f"encoding={message.encoding!r}, step={message.step}, bytes={len(data)})"
        )
    return message


def decode_joint_state(data: bytes, *, alignment_origin: int = 0) -> JointStateMessage:
    reader = CdrReader(data, alignment_origin=alignment_origin)
    stamp_ns, frame_id = read_header(reader)
    names = reader.read_string_sequence()
    position = reader.read_float64_sequence()
    velocity = reader.read_float64_sequence()
    effort = reader.read_float64_sequence()
    if reader.remaining != 0:
        raise CdrError(f"JointState payload has {reader.remaining} trailing bytes")
    if len(position) != len(names):
        raise CdrError("JointState name/position lengths differ")
    return JointStateMessage(
        header_stamp_ns=stamp_ns,
        frame_id=frame_id,
        names=names,
        position=position,
        velocity=velocity,
        effort=effort,
    )


def decode_float64_multi_array(data: bytes, *, alignment_origin: int = 0) -> Float64MultiArrayMessage:
    reader = CdrReader(data, alignment_origin=alignment_origin)
    dim_count = reader._read_sequence_length()
    dimensions: list[tuple[str, int, int]] = []
    for _ in range(dim_count):
        label = reader.read_string()
        size = reader.read_uint32()
        stride = reader.read_uint32()
        dimensions.append((label, int(size), int(stride)))
    data_offset = reader.read_uint32()
    values = reader.read_float64_sequence()
    if reader.remaining != 0:
        raise CdrError(f"Float64MultiArray payload has {reader.remaining} trailing bytes")
    return Float64MultiArrayMessage(
        data=values,
        layout_dimensions=tuple(dimensions),
        data_offset=int(data_offset),
    )


_DECODERS = {
    IMAGE_TYPE: decode_image,
    JOINT_STATE_TYPE: decode_joint_state,
    FLOAT64_MULTI_ARRAY_TYPE: decode_float64_multi_array,
}


def decode_message(message_type: str, data: bytes, *, alignment_origin: int = 0):
    """Decode one CDR payload by ROS 2 message type name."""
    try:
        decoder = _DECODERS[message_type]
    except KeyError as exc:
        raise CdrError(f"unsupported ROS 2 message type {message_type!r}") from exc
    return decoder(data, alignment_origin=alignment_origin)


def decode_with_alignment_fallback(message_type: str, data: bytes):
    """Decode with the Fast CDR alignment base, then the alternate DDS base.

    Both attempts are strict (trailing bytes, exact image sizes and name/position
    lengths are validated), so a wrong alignment cannot decode "successfully".
    Returns ``(message, alignment_origin_used)``.
    """
    first_error: CdrError | None = None
    try:
        return decode_message(message_type, data, alignment_origin=0), 0
    except CdrError as exc:
        first_error = exc
    try:
        return decode_message(message_type, data, alignment_origin=4), 4
    except CdrError as exc:
        raise CdrError(
            f"CDR payload for {message_type} could not be decoded with either "
            f"alignment convention; first error: {first_error}"
        ) from exc


__all__ = [
    "CdrError",
    "CdrReader",
    "CdrWriter",
    "Float64MultiArrayMessage",
    "ImageMessage",
    "JointStateMessage",
    "ROS2_MSG_DEFINITIONS",
    "IMAGE_TYPE",
    "JOINT_STATE_TYPE",
    "FLOAT64_MULTI_ARRAY_TYPE",
    "decode_float64_multi_array",
    "decode_image",
    "decode_joint_state",
    "decode_message",
    "decode_with_alignment_fallback",
    "encode_float64_multi_array",
    "encode_image",
    "encode_joint_state",
    "ns_to_stamp",
    "stamp_to_ns",
]
