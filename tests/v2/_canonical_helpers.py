"""Shared in-memory builders for the D2-M1 canonical v2 tests (no services needed)."""

from __future__ import annotations

import numpy as np

from embodied_dataops.canonical.v2.model import (
    CanonicalV2Source,
    NativeCameraStream,
    NativeRobotStream,
)
from embodied_dataops.domain.profiles import RobotProfileConfig, load_profiles


def profile(profile_id: str = "arm6_joint_position_v1") -> RobotProfileConfig:
    for candidate in load_profiles():
        if candidate.profile_id == profile_id:
            return candidate
    raise AssertionError(f"unknown profile {profile_id}")


def times_at(rate: int, duration_ns: int) -> list[int]:
    interval = 1_000_000_000 // rate
    return [index * interval for index in range(int(duration_ns // interval) + 1)]


def prealigned_source(
    *,
    profile_id: str = "arm6_joint_position_v1",
    cameras: tuple[str, ...] = ("front",),
    rows: int = 40,
    fps: int = 10,
) -> CanonicalV2Source:
    config = profile(profile_id)
    times = [int(index * 1_000_000_000 / fps) for index in range(rows)]
    rng = np.random.default_rng(20261029)
    states = rng.uniform(-0.5, 0.5, size=(rows, config.state_dim)).astype(np.float64)
    actions = (states + 0.01).astype(np.float64)
    frame = np.zeros((64, 96, 3), dtype=np.uint8)
    return CanonicalV2Source(
        episode_id="episode-prealigned",
        source_asset_sha256="a" * 64,
        source_type="LEROBOT_V3",
        source_episode_index=0,
        data_origin="synthetic",
        task_label="synthetic row-paired episode",
        profile=config,
        profile_id=config.profile_id,
        profile_revision=config.profile_revision,
        profile_config_sha256=config.config_sha256(),
        profile_authority=config.authority,
        joint_names=list(config.joint_names),
        state_semantics=config.state_semantics,
        action_semantics=config.action_semantics,
        state_unit=config.state_unit,
        action_unit=config.action_unit,
        semantic_authority=True,
        cameras=[
            NativeCameraStream(
                name=name,
                source_key=f"observation.images.{name}",
                clock_domain="lerobot_episode_relative",
                timestamp_provenance="lerobot_row_timestamps",
                times_ns=list(times),
                width=96,
                height=64,
                frame_getter=lambda _index: frame,
                source_asset_id="a" * 64,
            )
            for name in cameras
        ],
        state=NativeRobotStream(
            name="state",
            source_key="observation.state",
            clock_domain="lerobot_episode_relative",
            timestamp_provenance="lerobot_row_timestamps",
            times_ns=list(times),
            values=states.tolist(),
            unit=config.state_unit,
        ),
        action=NativeRobotStream(
            name="action",
            source_key="action",
            clock_domain="lerobot_episode_relative",
            timestamp_provenance="lerobot_row_timestamps",
            times_ns=list(times),
            values=actions.tolist(),
            unit=config.action_unit,
        ),
        alignment_mode="prealigned",
        prealigned_row_times_ns=list(times),
        source_row_rate_hz=float(fps),
        origin_ns=0,
        required_modalities=["front", "state", "action"],
    )


def native_source(
    *,
    profile_id: str = "arm6_joint_position_v1",
    with_wrist: bool = True,
    duration_ns: int = 2_000_000_000,
    rates: dict[str, int] | None = None,
    drop_front_window: tuple[int, int] | None = None,
    drop_state_window: tuple[int, int] | None = None,
    action_start_ns: int = 0,
) -> CanonicalV2Source:
    config = profile(profile_id)
    rates = dict(rates or {"front": 30, "wrist": 15, "state": 100, "action": 50})
    origin = 1_700_000_000_000_000_000
    front = [origin + value for value in times_at(rates["front"], duration_ns)]
    wrist = [origin + value for value in times_at(rates["wrist"], duration_ns)]
    state = [origin + value for value in times_at(rates["state"], duration_ns)]
    action = [
        origin + value
        for value in times_at(rates["action"], duration_ns)
        if origin + value >= origin + action_start_ns
    ]
    if drop_front_window is not None:
        low, high = drop_front_window
        front = [value for value in front if not origin + low <= value < origin + high]
    if drop_state_window is not None:
        low, high = drop_state_window
        state = [value for value in state if not origin + low <= value < origin + high]
    rng = np.random.default_rng(20261029)
    state_values = rng.uniform(-0.5, 0.5, size=(len(state), config.state_dim)).astype(np.float64)
    action_values = rng.uniform(-0.5, 0.5, size=(len(action), config.action_dim)).astype(np.float64)
    frame = np.zeros((64, 96, 3), dtype=np.uint8)
    cameras = [
        NativeCameraStream(
            name="front",
            source_key="/camera/front/image_raw",
            clock_domain="mcap_log_time",
            timestamp_provenance="mcap_log_time",
            times_ns=front,
            width=96,
            height=64,
            frame_getter=lambda _index: frame,
            source_asset_id="b" * 64,
        )
    ]
    if with_wrist:
        cameras.append(
            NativeCameraStream(
                name="wrist",
                source_key="/camera/wrist/image_raw",
                clock_domain="mcap_log_time",
                timestamp_provenance="mcap_log_time",
                times_ns=wrist,
                width=96,
                height=64,
                frame_getter=lambda _index: frame,
                source_asset_id="b" * 64,
            )
        )
    return CanonicalV2Source(
        episode_id="episode-native",
        source_asset_sha256="b" * 64,
        source_type="MCAP_ROS2_PROFILE_V2",
        source_episode_index=0,
        data_origin="synthetic",
        task_label="synthetic native multi-stream episode",
        profile=config,
        profile_id=config.profile_id,
        profile_revision=config.profile_revision,
        profile_config_sha256=config.config_sha256(),
        profile_authority=config.authority,
        joint_names=list(config.joint_names),
        state_semantics=config.state_semantics,
        action_semantics=config.action_semantics,
        state_unit=config.state_unit,
        action_unit=config.action_unit,
        semantic_authority=True,
        cameras=cameras,
        state=NativeRobotStream(
            name="state",
            source_key="/joint_states",
            clock_domain="mcap_log_time",
            timestamp_provenance="mcap_log_time",
            times_ns=state,
            values=state_values.tolist(),
            unit=config.state_unit,
        ),
        action=NativeRobotStream(
            name="action",
            source_key="/policy/action",
            clock_domain="mcap_log_time",
            timestamp_provenance="mcap_log_time",
            times_ns=action,
            values=action_values.tolist(),
            unit=config.action_unit,
        ),
        alignment_mode="native",
        origin_ns=origin,
        required_modalities=["front", "state", "action"],
    )
