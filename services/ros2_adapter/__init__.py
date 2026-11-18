"""ROS 2 / MCAP adapter for the fixed ``MCAP_ROS2_PROFILE_V1`` topic profile.

Two runtime paths share one profile contract (see :mod:`services.ros2_adapter.reader`):

* **production / acceptance** -- Ubuntu 24.04 + ROS 2 Jazzy ``worker-ros2``
  container, ``rosbag2_py`` + ``rosbag2_storage_mcap`` + ``rclpy`` deserialization;
* **development / CI fallback** -- ``mcap`` pip package plus the hand-written CDR
  codec in :mod:`services.ros2_adapter.cdr`, used on hosts without a ROS 2
  installation.

Only the three profile topics are decoded.  Nothing in this package imports ROS
2 at module import time, so the main API container can import the fixture
generator and the pure-Python parser without installing ROS.
"""

from __future__ import annotations

from .cdr import (
    FLOAT64_MULTI_ARRAY_TYPE,
    IMAGE_TYPE,
    JOINT_STATE_TYPE,
    Float64MultiArrayMessage,
    ImageMessage,
    JointStateMessage,
    decode_with_alignment_fallback,
    encode_float64_multi_array,
    encode_image,
    encode_joint_state,
)
from .reader import (
    ALLOWED_IMAGE_ENCODINGS,
    PROFILE_TOPICS,
    REQUIRED_TOPICS,
    TOPIC_ACTION,
    TOPIC_IMAGE,
    TOPIC_JOINT_STATES,
    ActionSample,
    CameraFrame,
    JointStateSample,
    McapProbe,
    McapProfileData,
    probe_mcap_profile,
    read_mcap_profile,
    resolve_backend,
    rosbag2_available,
    validate_profile,
)

__all__ = [
    "ALLOWED_IMAGE_ENCODINGS",
    "FLOAT64_MULTI_ARRAY_TYPE",
    "IMAGE_TYPE",
    "JOINT_STATE_TYPE",
    "PROFILE_TOPICS",
    "REQUIRED_TOPICS",
    "TOPIC_ACTION",
    "TOPIC_IMAGE",
    "TOPIC_JOINT_STATES",
    "ActionSample",
    "CameraFrame",
    "Float64MultiArrayMessage",
    "ImageMessage",
    "JointStateMessage",
    "JointStateSample",
    "McapProbe",
    "McapProfileData",
    "decode_with_alignment_fallback",
    "encode_float64_multi_array",
    "encode_image",
    "encode_joint_state",
    "probe_mcap_profile",
    "read_mcap_profile",
    "resolve_backend",
    "rosbag2_available",
    "validate_profile",
]
