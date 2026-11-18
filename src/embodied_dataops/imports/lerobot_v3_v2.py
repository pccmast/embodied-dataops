"""``LEROBOT_V3`` v2 adapter: profile-bound, 1-2 cameras, 6D/7D (V2 SPEC 2.4).

This module *extends* the V1 official-API reader
(:class:`embodied_dataops.imports.lerobot_v3.LerobotV3Parser`); the V1 importer is
untouched and remains the canonical-episode/1.0 path.

Differences from V1 (all required by the V2 spec):

* robot semantics come from the versioned Profile Registry
  (:func:`embodied_dataops.domain.profiles.resolve_profile`): the import mapping
  must name ``profile_id`` + ``profile_revision`` (or opt into
  ``diagnostic_only``); the profile is **never** inferred from ``shape``/file name;
* 1-2 cameras require an explicit mapping (``camera_key`` for front,
  ``camera_keys`` mapping/list for front+wrist); an unmapped source camera is
  reported, never silently dropped;
* LeRobot rows are already aligned: the adapter keeps the row pairing and records
  ``timestamp_provenance=lerobot_row_timestamps`` with **no fabricated native
  frequency** (``native_rate_hz`` stays NULL for these modalities);
* the output is a :class:`CanonicalV2Source` consumed by
  :func:`embodied_dataops.canonical.v2.build.build_canonical_v2`.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from embodied_dataops.canonical.v2.model import (
    CanonicalV2Source,
    NativeCameraStream,
    NativeRobotStream,
)
from embodied_dataops.domain.enums import EpisodeType, Severity, SourceType
from embodied_dataops.domain.profiles import ResolvedProfile, resolve_profile
from embodied_dataops.errors import ApiError, ErrorCode
from embodied_dataops.imports.base import EpisodeSource, ImportContext, ParsedSource
from embodied_dataops.imports.common import diagnostic, has_structural, merge_diagnostics
from embodied_dataops.imports.lerobot_v3 import (
    ACTION_KEY_DEFAULT,
    STATE_KEY_DEFAULT,
    LerobotEpisodeArrays,
    LerobotInfo,
    LerobotV3Parser,
    _nonfinite_count,
    _safe_sidecar_summary,
    to_float_list,
    to_rgb_uint8,
)

TIMESTAMP_PROVENANCE = "lerobot_row_timestamps"
CLOCK_DOMAIN = "lerobot_episode_relative"
DIAG_UNMAPPED_CAMERA = "CAMERA_UNMAPPED_IN_SOURCE"


@dataclass
class V2LerobotMapping:
    """Explicit, profile-bound mapping of one LeRobot v2 import."""

    profile: ResolvedProfile
    state_key: str
    action_key: str
    camera_keys: dict[str, str]
    available_cameras: list[str]
    state_dim: int | None
    action_dim: int | None
    diagnostics: list[dict[str, Any]] = field(default_factory=list)
    semantics_evidence: str = ""

    @property
    def profile_id(self) -> str | None:
        return self.profile.profile_id

    @property
    def profile_revision(self) -> int | None:
        return self.profile.profile_revision

    @property
    def profile_config_sha256(self) -> str | None:
        return self.profile.config_sha256

    def as_dict(self) -> dict[str, Any]:
        return {
            "profile_id": self.profile_id,
            "profile_revision": self.profile_revision,
            "profile_config_sha256": self.profile_config_sha256,
            "mode": self.profile.mode,
            "semantic_authority": self.profile.semantic_authority,
            "camera_keys": dict(self.camera_keys),
            "available_cameras": list(self.available_cameras),
            "state_key": self.state_key,
            "action_key": self.action_key,
            "state_dim": self.state_dim,
            "action_dim": self.action_dim,
            "semantics_evidence": self.semantics_evidence,
        }


def _camera_mapping(
    mapping: dict[str, Any], info: LerobotInfo
) -> tuple[dict[str, str], list[str]]:
    available = info.camera_keys
    cameras: dict[str, str] = {}
    raw_keys = mapping.get("camera_keys")
    if raw_keys is not None:
        if isinstance(raw_keys, dict):
            for name in ("front", "wrist"):
                if raw_keys.get(name):
                    cameras[name] = str(raw_keys[name])
            unknown = sorted(set(raw_keys) - {"front", "wrist"})
            if unknown:
                raise ApiError(
                    ErrorCode.INVALID_MAPPING,
                    "mapping.camera_keys may only contain the canonical names 'front' and 'wrist'",
                    details={"unknown_keys": unknown, "available_cameras": available},
                )
        elif isinstance(raw_keys, (list, tuple)):
            if len(raw_keys) > 2:
                raise ApiError(
                    ErrorCode.INVALID_MAPPING,
                    "mapping.camera_keys accepts at most two ordered camera features (front, wrist)",
                    details={"camera_keys": list(raw_keys)},
                )
            names = ["front", "wrist"]
            for index, value in enumerate(raw_keys):
                cameras[names[index]] = str(value)
        else:
            raise ApiError(
                ErrorCode.INVALID_MAPPING,
                "mapping.camera_keys must be a {'front'|'wrist': feature} object or an ordered list",
            )
    front_key = mapping.get("camera_key")
    if front_key and "front" not in cameras:
        cameras["front"] = str(front_key)
    if not cameras:
        raise ApiError(
            ErrorCode.MAPPING_REQUIRED,
            "LeRobot v2 imports require an explicit camera mapping: mapping.camera_key (front) "
            "and optionally mapping.camera_keys {'front': key, 'wrist': key}",
            details={"available_cameras": available},
        )
    for name, key in cameras.items():
        if key not in info.features:
            raise ApiError(
                ErrorCode.INVALID_MAPPING,
                f"mapping camera {name!r} is not a feature of this dataset",
                details={"camera_key": key, "available_cameras": available},
            )
        if key not in available:
            raise ApiError(
                ErrorCode.INVALID_MAPPING,
                f"mapping camera {name!r} must be an observation.images.* feature",
                details={"camera_key": key, "available_cameras": available},
            )
    if "front" not in cameras:
        raise ApiError(
            ErrorCode.MAPPING_REQUIRED,
            "the front camera mapping is required (profile-required camera)",
            details={"available_cameras": available, "mapped": cameras},
        )
    return cameras, available


def resolve_lerobot_v2_mapping(
    info: LerobotInfo,
    mapping: dict[str, Any] | None,
    *,
    semantics_sidecar: Any = None,
) -> V2LerobotMapping:
    """Resolve the explicit profile + camera/state/action mapping (never guesses)."""
    raw = dict(mapping or {})
    profile = resolve_profile(raw)
    state_key = str(raw.get("state_key") or STATE_KEY_DEFAULT)
    if state_key not in info.features:
        raise ApiError(
            ErrorCode.INVALID_MAPPING,
            "mapping.state_key is not a feature of this dataset",
            details={"state_key": state_key, "features": sorted(info.features)},
        )
    action_key = str(raw.get("action_key") or ACTION_KEY_DEFAULT)
    if action_key not in info.features:
        raise ApiError(
            ErrorCode.INVALID_MAPPING,
            "mapping.action_key is not a feature of this dataset",
            details={"action_key": action_key, "features": sorted(info.features)},
        )
    camera_keys, available = _camera_mapping(raw, info)
    diagnostics: list[dict[str, Any]] = []
    unmapped = [key for key in available if key not in set(camera_keys.values())]
    if unmapped:
        diagnostics.append(
            diagnostic(
                DIAG_UNMAPPED_CAMERA,
                Severity.WARN,
                "dataset declares camera features that the explicit mapping does not import",
                details={"unmapped_camera_features": unmapped, "mapped_camera_keys": camera_keys},
                structural=False,
            )
        )
    if not profile.semantic_authority:
        diagnostics.append(
            diagnostic(
                "UNVERIFIED_SEMANTICS",
                Severity.ERROR,
                "robot profile authority is not SYNTHETIC_FIXTURE/SOURCE_DOCUMENTED; "
                "the episode may be normalized for diagnostics but must not be published",
                details={
                    "profile_id": profile.profile_id,
                    "profile_revision": profile.profile_revision,
                    "authority": profile.config.authority if profile.config else None,
                },
                structural=False,
            )
        )
    evidence = ""
    if isinstance(semantics_sidecar, dict):
        evidence = str(
            semantics_sidecar.get("evidence") or semantics_sidecar.get("description") or ""
        ).strip()
    if raw.get("semantics_evidence"):
        extra = str(raw["semantics_evidence"]).strip()
        evidence = f"{evidence}; upload mapping evidence: {extra}" if evidence else extra
    return V2LerobotMapping(
        profile=profile,
        state_key=state_key,
        action_key=action_key,
        camera_keys=camera_keys,
        available_cameras=available,
        state_dim=info.feature_dim(state_key),
        action_dim=info.feature_dim(action_key),
        diagnostics=diagnostics,
        semantics_evidence=evidence,
    )


def _feature_size(info: LerobotInfo, key: str) -> tuple[int, int]:
    feature = info.feature(key) or {}
    shape = list(feature.get("shape") or [])
    if len(shape) == 3:
        return int(shape[1]), int(shape[2])
    return 0, 0


class LerobotV3V2Parser(LerobotV3Parser):
    """Profile-bound LeRobot v3 reader producing :class:`CanonicalV2Source`."""

    source_type = SourceType.LEROBOT_V3

    def validate(self, ctx: ImportContext) -> dict[str, Any]:
        source = Path(ctx.source_path)
        if source.is_dir():
            from embodied_dataops.imports.lerobot_v3 import resolve_dataset_root

            root = resolve_dataset_root(source)
            info = self._read_info(root)
            semantics, _sidecar = self._read_sidecars(root)
            archive_meta: dict[str, Any] = {"kind": "directory"}
        else:
            from embodied_dataops.imports.lerobot_v3 import inspect_lerobot_archive, parse_info_payload

            inspection = inspect_lerobot_archive(source)
            info = parse_info_payload(inspection.info_payload)
            semantics = inspection.semantics_payload
            archive_meta = {
                "kind": "tar.gz",
                "entries": inspection.entries,
                "unpacked_bytes": inspection.total_bytes,
            }
        resolved = resolve_lerobot_v2_mapping(info, ctx.mapping, semantics_sidecar=semantics)
        return {
            "format": "lerobot-v3",
            "adapter": "lerobot_v3_v2",
            "archive": archive_meta,
            "info": info.as_summary(),
            "resolved_mapping": resolved.as_dict(),
            "episode_count_declared": info.total_episodes,
            "total_frames_declared": info.total_frames,
            "required_api": "lerobot==0.4.4",
            "api_available": True,
        }

    def parse(self, ctx: ImportContext) -> ParsedSource:
        root = self._materialize_root(ctx)
        info = self._read_info(root)
        semantics, source_sidecar = self._read_sidecars(root)
        resolved = resolve_lerobot_v2_mapping(info, ctx.mapping, semantics_sidecar=semantics)
        dataset = self._open_dataset(ctx, root, source_sidecar)
        episodes = self._enumerate_episodes(dataset, info)
        if not episodes:
            raise ApiError(
                ErrorCode.EPISODE_ENUMERATION_FAILED,
                "the official LeRobot API enumerated no Episodes",
            )
        sources: list[EpisodeSource] = []
        for meta in episodes:
            source = self._episode_source(ctx, info, resolved, meta)
            sources.append(source)
        return ParsedSource(
            episodes=sources,
            metadata={
                "adapter": "lerobot_v3_v2",
                "codebase_version": info.codebase_version,
                "fps": info.fps,
                "robot_type": info.robot_type,
                "episode_count": len(sources),
                "resolved_mapping": resolved.as_dict(),
                "enumerated_by": "official_lerobot_api",
                "source_sidecar": _safe_sidecar_summary(source_sidecar),
            },
            warnings=[],
        )

    # -- per-episode -------------------------------------------------------
    def _episode_source(
        self,
        ctx: ImportContext,
        info: LerobotInfo,
        resolved: V2LerobotMapping,
        meta: Any,
    ) -> EpisodeSource:
        diagnostics = list(resolved.diagnostics)
        profile = resolved.profile.config
        if profile is not None:
            if resolved.state_dim != profile.state_dim:
                diagnostics.append(
                    diagnostic(
                        "DIMENSION_MISMATCH",
                        Severity.ERROR,
                        f"observation.state is {resolved.state_dim}D but profile "
                        f"{profile.profile_id}@{profile.profile_revision} requires {profile.state_dim}D",
                        details={"state_dim": resolved.state_dim, "required": profile.state_dim},
                    )
                )
            if resolved.action_dim != profile.action_dim:
                diagnostics.append(
                    diagnostic(
                        "DIMENSION_MISMATCH",
                        Severity.ERROR,
                        f"action is {resolved.action_dim}D but profile requires {profile.action_dim}D",
                        details={"action_dim": resolved.action_dim, "required": profile.action_dim},
                    )
                )
        expected_terminal = bool(ctx.mapping.get("expected_terminal_marker", False))
        return EpisodeSource(
            source_episode_index=int(meta.episode_index),
            episode_type=EpisodeType.ROBOT_EPISODE,
            task_label=meta.task_label,
            duration_ns=int(round(meta.length / info.fps * 1_000_000_000)) if info.fps else 0,
            payload={
                "adapter": "lerobot_v3_v2",
                "episode_index": int(meta.episode_index),
                "profile_id": resolved.profile_id,
                "profile_revision": resolved.profile_revision,
                "profile_config_sha256": resolved.profile_config_sha256,
                "camera_keys": dict(resolved.camera_keys),
                "state_key": resolved.state_key,
                "action_key": resolved.action_key,
                "source_fps": info.fps,
                "length": int(meta.length),
                "expected_terminal_marker": expected_terminal,
            },
            diagnostics=diagnostics,
            normalization_valid=not has_structural(diagnostics),
        )

    def read_v2_episode(self, ctx: ImportContext, episode_index: int) -> CanonicalV2Source:
        """Read one Episode and keep the already-aligned LeRobot row pairing."""
        root = self._materialize_root(ctx)
        info = self._read_info(root)
        semantics, source_sidecar = self._read_sidecars(root)
        resolved = resolve_lerobot_v2_mapping(info, ctx.mapping, semantics_sidecar=semantics)
        dataset = self._open_dataset(ctx, root, source_sidecar)
        meta = self._episode_meta(ctx, dataset, info, episode_index)
        arrays = self._episode_arrays(
            dataset, meta, info, resolved.state_key, resolved.action_key
        )
        profile = resolved.profile.config
        diagnostics = list(resolved.diagnostics)
        diagnostics.extend(
            _array_diagnostics(arrays, profile_id=resolved.profile_id, profile=profile)
        )
        row_times = list(arrays.timestamps_ns)
        cameras: list[NativeCameraStream] = []
        for canonical_name, feature_key in resolved.camera_keys.items():
            height, width = _feature_size(info, feature_key)
            cameras.append(
                NativeCameraStream(
                    name=canonical_name,
                    source_key=feature_key,
                    clock_domain=CLOCK_DOMAIN,
                    timestamp_provenance=TIMESTAMP_PROVENANCE,
                    times_ns=list(row_times),
                    width=width,
                    height=height,
                    encoding="rgb8",
                    frame_getter=_camera_frame_getter(
                        dataset, arrays, feature_key, len(row_times)
                    ),
                    source_asset_id=ctx.source_sha256,
                )
            )
        state = NativeRobotStream(
            name="state",
            source_key=resolved.state_key,
            clock_domain=CLOCK_DOMAIN,
            timestamp_provenance=TIMESTAMP_PROVENANCE,
            times_ns=list(row_times),
            values=[to_float_list(row) for row in arrays.states],
            unit=profile.state_unit if profile is not None else "rad",
            source_asset_id=ctx.source_sha256,
        )
        action = NativeRobotStream(
            name="action",
            source_key=resolved.action_key,
            clock_domain=CLOCK_DOMAIN,
            timestamp_provenance=TIMESTAMP_PROVENANCE,
            times_ns=list(row_times),
            values=[to_float_list(row) for row in arrays.actions],
            unit=profile.action_unit if profile is not None else "rad",
            source_asset_id=ctx.source_sha256,
        )
        mapping_payload = resolved.as_dict()
        mapping_payload["semantics_sidecar_summary"] = _safe_sidecar_summary(source_sidecar)
        mapping_payload["dataset_fps"] = info.fps
        mapping_payload["row_pairing"] = "already_aligned_rows_preserved"
        return CanonicalV2Source(
            episode_id="",  # filled by the handler (it owns the Episode id)
            source_asset_sha256=ctx.source_sha256,
            source_type=SourceType.LEROBOT_V3.value,
            source_episode_index=int(episode_index),
            data_origin=ctx.data_origin.value,
            task_label=meta.task_label,
            profile=profile,
            profile_id=resolved.profile_id,
            profile_revision=resolved.profile_revision,
            profile_config_sha256=resolved.profile_config_sha256,
            profile_authority=profile.authority if profile is not None else "",
            joint_names=list(profile.joint_names) if profile is not None else [],
            action_names=list(profile.action_names) if profile is not None else [],
            state_semantics=profile.state_semantics if profile is not None else "",
            action_semantics=profile.action_semantics if profile is not None else "",
            state_unit=profile.state_unit if profile is not None else "rad",
            action_unit=profile.action_unit if profile is not None else "rad",
            diagnostic_only=resolved.profile.mode == "DIAGNOSTIC_ONLY",
            semantic_authority=bool(resolved.profile.semantic_authority),
            cameras=cameras,
            state=state,
            action=action,
            alignment_mode="prealigned",
            prealigned_row_times_ns=list(row_times),
            source_row_rate_hz=float(info.fps),
            origin_ns=0,
            clock_domains={
                "camera_front": CLOCK_DOMAIN,
                "state": CLOCK_DOMAIN,
                "action": CLOCK_DOMAIN,
            },
            native_rates_hz={},
            required_modalities=["front", "state", "action"],
            diagnostics=merge_diagnostics(diagnostics),
            expected_terminal_marker=bool(ctx.mapping.get("expected_terminal_marker", False)),
            terminal_marker_found=None,
            pointcloud=None,
            mapping_report=dict(mapping_payload),
            source={
                "repo_id": self._resolve_repo_id(ctx, root),
                "codebase_version": info.codebase_version,
                "dataset_fps": info.fps,
                "robot_type": info.robot_type,
                "state_key": resolved.state_key,
                "action_key": resolved.action_key,
                "camera_keys": dict(resolved.camera_keys),
                "semantics_evidence": resolved.semantics_evidence,
                "source_sidecar": _safe_sidecar_summary(source_sidecar),
                "row_pairing": "already_aligned_rows_preserved",
            },
        )


def _array_diagnostics(
    arrays: LerobotEpisodeArrays,
    *,
    profile_id: str | None,
    profile: Any,
) -> list[dict[str, Any]]:
    diagnostics: list[dict[str, Any]] = []
    state_dim = len(arrays.states[0]) if arrays.states else 0
    action_dim = len(arrays.actions[0]) if arrays.actions else 0
    if profile is not None:
        if state_dim != profile.state_dim:
            diagnostics.append(
                diagnostic(
                    "DIMENSION_MISMATCH",
                    Severity.ERROR,
                    f"observation.state is {state_dim}D; profile requires {profile.state_dim}D",
                    details={"state_dim": state_dim, "required": profile.state_dim},
                )
            )
        if action_dim != profile.action_dim:
            diagnostics.append(
                diagnostic(
                    "DIMENSION_MISMATCH",
                    Severity.ERROR,
                    f"action is {action_dim}D; profile requires {profile.action_dim}D",
                    details={"action_dim": action_dim, "required": profile.action_dim},
                )
            )
    monotonic = [
        index
        for index in range(1, len(arrays.timestamps_ns))
        if arrays.timestamps_ns[index] <= arrays.timestamps_ns[index - 1]
    ]
    if monotonic:
        diagnostics.append(
            diagnostic(
                "TIMESTAMP_NON_MONOTONIC",
                Severity.ERROR,
                "LeRobot episode timestamps are not strictly increasing",
                details={"violation_indexes": monotonic[:20], "violation_count": len(monotonic)},
            )
        )
    nonfinite = _nonfinite_count(arrays.states) + _nonfinite_count(arrays.actions)
    if nonfinite:
        diagnostics.append(
            diagnostic(
                "NONFINITE_VALUE",
                Severity.ERROR,
                "state/action contains NaN or infinite values",
                details={"count": nonfinite, "profile_id": profile_id},
            )
        )
    return diagnostics


def _camera_frame_getter(
    dataset: Any,
    arrays: LerobotEpisodeArrays,
    feature_key: str,
    row_count: int,
) -> Callable[[int], np.ndarray]:
    from_from = arrays.from_index

    def getter(source_index: int) -> np.ndarray:
        index = int(source_index)
        if not 0 <= index < row_count:
            raise ApiError(
                ErrorCode.INVALID_INPUT,
                "camera frame index is outside the episode",
                details={"source_index": index, "row_count": row_count},
            )
        row = dataset[from_from + index]
        value = row.get(feature_key)
        if value is None:
            raise ApiError(
                ErrorCode.FILE_DECODE_ERROR,
                "LeRobot row is missing the mapped camera feature",
                details={"camera_key": feature_key},
            )
        return to_rgb_uint8(value)

    return getter


__all__ = [
    "CLOCK_DOMAIN",
    "DIAG_UNMAPPED_CAMERA",
    "LerobotV3V2Parser",
    "TIMESTAMP_PROVENANCE",
    "V2LerobotMapping",
    "resolve_lerobot_v2_mapping",
]
