"""Robot Profile Registry (V2 SPEC 2.1, ADR-0007).

A profile is an immutable, versioned description of the robot joint layout,
units/semantics, limits and camera requirements.  Profile selection is explicit
(`profile_id` + `profile_revision` in the import mapping) or explicitly
diagnostic-only; it is **never** inferred from tensor shape or file names.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from embodied_dataops.canonical.contract import canonical_json_bytes
from embodied_dataops.domain.profiles_v21 import (
    PROFILE_SCHEMA_VERSION_21,
    PUBLISH_LEVEL_FORMAT,
    PUBLISH_LEVEL_SEMANTIC,
    RobotProfileV21,
)
from embodied_dataops.errors import ApiError, ErrorCode

PROFILE_SCHEMA_VERSION = "robot-profile/2.0"
AUTHORITY_SYNTHETIC = "SYNTHETIC_FIXTURE"
AUTHORITY_DOCUMENTED = "SOURCE_DOCUMENTED"
AUTHORITY_UNVERIFIED = "UNVERIFIED"
AUTHORITIES = (AUTHORITY_SYNTHETIC, AUTHORITY_DOCUMENTED, AUTHORITY_UNVERIFIED)
SEMANTIC_AUTHORITIES = (AUTHORITY_SYNTHETIC, AUTHORITY_DOCUMENTED)

#: Either schema generation; V2.1 code must treat both through the shared property names
#: (``state_dim`` / ``action_dim`` / ``joint_names`` / ``limits_available`` / ...).
#: The alias itself is declared after ``RobotProfileConfig`` is defined.

REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_PROFILE_FILE = REPOSITORY_ROOT / "contracts" / "robot_profiles_v2.json"
DEFAULT_PROFILE_FILE_V21 = REPOSITORY_ROOT / "contracts" / "robot_profiles_v21.json"
DIAGNOSTIC_ONLY = "diagnostic_only"


class RobotProfileConfig(BaseModel):
    """One immutable robot profile revision (``robot-profile/2.0``)."""

    model_config = ConfigDict(extra="forbid")

    schema_version: str = PROFILE_SCHEMA_VERSION
    profile_id: str = Field(min_length=1, max_length=80)
    profile_revision: int = Field(ge=1)
    authority: Literal["SYNTHETIC_FIXTURE", "SOURCE_DOCUMENTED", "UNVERIFIED"]
    source_evidence_asset_id: str | None = None

    state_semantics: str = Field(min_length=1)
    action_semantics: str = Field(min_length=1)
    state_unit: str = Field(min_length=1)
    action_unit: str = Field(min_length=1)

    joint_names: list[str] = Field(min_length=1)
    joint_lower_rad: list[float]
    joint_upper_rad: list[float]
    max_abs_velocity_rad_s: list[float]
    continuous_joints: list[bool]

    required_camera_names: list[str] = Field(min_length=1)
    optional_camera_names: list[str] = Field(default_factory=list)
    aligned_fps: int = Field(ge=1, le=240)
    preview_fps: int = Field(ge=1, le=240)
    max_clock_offset_ns: int = Field(ge=0)
    notes: str = ""

    @model_validator(mode="after")
    def _validate_lengths(self) -> RobotProfileConfig:
        dim = len(self.joint_names)
        for field_name in ("joint_lower_rad", "joint_upper_rad", "max_abs_velocity_rad_s", "continuous_joints"):
            values = getattr(self, field_name)
            if len(values) != dim:
                raise ValueError(f"{field_name} must have {dim} entries (joint_names length)")
        for low, high in zip(self.joint_lower_rad, self.joint_upper_rad, strict=True):
            if low >= high:
                raise ValueError("joint_lower_rad must be smaller than joint_upper_rad")
        for velocity in self.max_abs_velocity_rad_s:
            if velocity <= 0:
                raise ValueError("max_abs_velocity_rad_s entries must be positive")
        if not self.state_semantics or not self.action_semantics:
            raise ValueError("state/action semantics must be documented")
        return self

    @property
    def state_dim(self) -> int:
        return len(self.joint_names)

    @property
    def action_dim(self) -> int:
        return len(self.joint_names)

    @property
    def action_names(self) -> list[str]:
        """A 2.0 profile derives both dimensions from one list, so names are shared."""
        return list(self.joint_names)

    @property
    def semantic_authority(self) -> bool:
        return self.authority in SEMANTIC_AUTHORITIES

    def canonical_json(self) -> bytes:
        return canonical_json_bytes(self.model_dump(mode="json"))

    def config_sha256(self) -> str:
        return hashlib.sha256(self.canonical_json()).hexdigest()

    @property
    def limits_available(self) -> bool:
        """A 2.0 profile always declares its limits block."""
        return True

    @property
    def publish_level(self) -> str:
        return PUBLISH_LEVEL_SEMANTIC if self.semantic_authority else PUBLISH_LEVEL_FORMAT


#: Either schema generation.  V2.1 consumers read both through the shared property names
#: (``state_dim`` / ``action_dim`` / ``joint_names`` / ``limits_available`` / ``publish_level``).
RobotProfile = RobotProfileConfig | RobotProfileV21


class ResolvedProfile(BaseModel):
    """Result of resolving an import mapping against the registry."""

    model_config = ConfigDict(extra="forbid")

    profile_id: str | None
    profile_revision: int | None
    config: RobotProfile | None
    mode: Literal["SEMANTIC", "DIAGNOSTIC_ONLY"]
    semantic_authority: bool = False

    @property
    def config_sha256(self) -> str | None:
        return self.config.config_sha256() if self.config else None

    @property
    def state_dim(self) -> int | None:
        return self.config.state_dim if self.config else None

    @property
    def action_dim(self) -> int | None:
        return self.config.action_dim if self.config else None


def parse_profile(entry: dict[str, Any]) -> RobotProfile:
    """Dispatch on ``schema_version``; unknown generations are rejected, never coerced.

    Public because every entry point that reads a registry row (normalizer, QC,
    export) must parse by the row's own generation instead of assuming 2.0.
    """
    version = str(entry.get("schema_version") or PROFILE_SCHEMA_VERSION)
    if version == PROFILE_SCHEMA_VERSION_21:
        return RobotProfileV21.model_validate(entry)
    if version == PROFILE_SCHEMA_VERSION:
        return RobotProfileConfig.model_validate(entry)
    raise ApiError(
        ErrorCode.INVALID_INPUT,
        f"unsupported robot profile schema_version {version!r}",
        details={"supported": [PROFILE_SCHEMA_VERSION, PROFILE_SCHEMA_VERSION_21]},
    )


#: Backwards-compatible private alias (existing module-internal call sites).
_parse_profile = parse_profile


def _read_profile_file(profile_path: Path) -> list[RobotProfile]:
    if not profile_path.is_file():
        raise ApiError(
            ErrorCode.INVALID_INPUT,
            "robot profile config file not found",
            details={"path": str(profile_path)},
        )
    try:
        payload = json.loads(profile_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ApiError(ErrorCode.INVALID_INPUT, "robot profile file is not valid JSON") from exc
    entries = payload.get("profiles") if isinstance(payload, dict) else None
    if not isinstance(entries, list) or not entries:
        raise ApiError(ErrorCode.INVALID_INPUT, "robot profile file must contain a non-empty profiles list")
    profiles: list[RobotProfile] = []
    for entry in entries:
        try:
            profiles.append(_parse_profile(entry))
        except ApiError:
            raise
        except Exception as exc:
            raise ApiError(
                ErrorCode.INVALID_INPUT,
                f"invalid robot profile: {type(exc).__name__}: {exc}",
                details={"entry": str(entry)[:200]},
            ) from exc
    return profiles


def load_profiles(path: Path | str | None = None) -> list[RobotProfile]:
    """Load the V2 registry (``robot-profile/2.0``), or one explicit file.

    Kept exactly as V2 behaved so every existing caller and test is unaffected.
    V2.1 callers that need both generations use :func:`load_all_profiles`.
    """
    profile_path = Path(path) if path is not None else DEFAULT_PROFILE_FILE
    return _read_profile_file(profile_path)


def load_all_profiles() -> list[RobotProfile]:
    """Load ``robot-profile/2.0`` **and** ``robot-profile/2.1`` as one registry.

    Duplicate ``(profile_id, revision)`` pairs across the two files are a configuration
    error, not a silent override: a revision number must identify exactly one contract.
    """
    profiles: list[RobotProfile] = []
    seen: set[tuple[str, int]] = set()
    candidates = [DEFAULT_PROFILE_FILE]
    if DEFAULT_PROFILE_FILE_V21.is_file():
        candidates.append(DEFAULT_PROFILE_FILE_V21)
    for candidate in candidates:
        for profile in _read_profile_file(candidate):
            key = (profile.profile_id, profile.profile_revision)
            if key in seen:
                raise ApiError(
                    ErrorCode.INVALID_INPUT,
                    "duplicate robot profile revision across profile files",
                    details={
                        "profile_id": profile.profile_id,
                        "profile_revision": profile.profile_revision,
                    },
                )
            seen.add(key)
            profiles.append(profile)
    return profiles


def profile_index(profiles: list[RobotProfile]) -> dict[tuple[str, int], RobotProfile]:
    return {(profile.profile_id, profile.profile_revision): profile for profile in profiles}


def combined_registry(
    registered: list[RobotProfile],
    *,
    builtin: list[RobotProfile] | None = None,
) -> list[RobotProfile]:
    """Overlay registered revisions on the built-in registry.

    Used by the normalizer: the DB registry is authoritative for the revisions it
    actually holds, while the shipped ``robot_profiles*.json`` files guarantee that a
    newly added V2.1 profile resolves on a host whose registry has not been re-seeded.
    A revision number must still identify exactly one contract: if a registered row
    disagrees with the file for the same ``(profile_id, profile_revision)``, the
    registry is inconsistent and the caller gets ``CONFLICT`` instead of a silent
    override.
    """
    entries = profile_index(builtin if builtin is not None else load_all_profiles())
    for profile in registered:
        key = (profile.profile_id, profile.profile_revision)
        existing = entries.get(key)
        if existing is not None and existing.config_sha256() != profile.config_sha256():
            raise ApiError(
                ErrorCode.CONFLICT,
                "registered robot profile differs from the shipped profile file",
                details={
                    "profile_id": profile.profile_id,
                    "profile_revision": profile.profile_revision,
                    "registered_sha256": profile.config_sha256(),
                    "file_sha256": existing.config_sha256(),
                },
            )
        entries[key] = profile
    return [entries[key] for key in sorted(entries)]


def resolve_profile(
    mapping: dict[str, Any] | None,
    *,
    profiles: list[RobotProfile] | None = None,
) -> ResolvedProfile:
    """Resolve the import mapping to a profile binding.

    Rules (SPEC 2.1 / 2.4):
    * explicit ``profile_id`` + ``profile_revision`` is required for semantic use;
    * ``diagnostic_only: true`` is the only alternative and can never be published;
    * shape / camera count / file names are never used to guess a profile;
    * ``UNVERIFIED`` authority is returned but flagged as non-semantic.
    """
    mapping = dict(mapping or {})
    if mapping.get(DIAGNOSTIC_ONLY) is True:
        return ResolvedProfile(
            profile_id=None,
            profile_revision=None,
            config=None,
            mode="DIAGNOSTIC_ONLY",
            semantic_authority=False,
        )
    if mapping.get(DIAGNOSTIC_ONLY) not in (None, False):
        raise ApiError(
            ErrorCode.INVALID_MAPPING,
            "mapping.diagnostic_only must be a boolean",
            details={"diagnostic_only": mapping.get(DIAGNOSTIC_ONLY)},
        )

    profile_id = str(mapping.get("profile_id") or "").strip()
    revision_raw = mapping.get("profile_revision")
    if not profile_id or revision_raw is None:
        raise ApiError(
            ErrorCode.MAPPING_REQUIRED,
            "LeRobot v3 / MCAP v2 imports require mapping.profile_id and profile_revision, "
            "or explicit mapping.diagnostic_only=true",
            details={
                "expected": ["profile_id", "profile_revision"],
                "alternative": "diagnostic_only=true (preview only, never publishable)",
            },
        )
    try:
        revision = int(revision_raw)
    except (TypeError, ValueError) as exc:
        raise ApiError(
            ErrorCode.INVALID_MAPPING,
            "mapping.profile_revision must be an integer",
            details={"profile_revision": revision_raw},
        ) from exc

    registry = profile_index(profiles if profiles is not None else load_all_profiles())
    config = registry.get((profile_id, revision))
    if config is None:
        raise ApiError(
            ErrorCode.NOT_FOUND,
            "robot profile revision is not registered",
            details={
                "profile_id": profile_id,
                "profile_revision": revision,
                "available": [
                    {"profile_id": key[0], "profile_revision": key[1]} for key in sorted(registry)
                ],
            },
        )
    if not config.semantic_authority:
        return ResolvedProfile(
            profile_id=config.profile_id,
            profile_revision=config.profile_revision,
            config=config,
            mode="SEMANTIC",
            semantic_authority=False,
        )
    return ResolvedProfile(
        profile_id=config.profile_id,
        profile_revision=config.profile_revision,
        config=config,
        mode="SEMANTIC",
        semantic_authority=True,
    )


def config_sha256_for(payload: dict[str, Any]) -> str:
    return hashlib.sha256(canonical_json_bytes(payload)).hexdigest()


def upsert_builtin_profiles(
    session,
    *,
    path: Path | str | None = None,
    profiles: list[RobotProfile] | None = None,
    include_v21: bool = False,
) -> list[Any]:
    """Idempotently register the built-in profiles in ``robot_profiles``.

    ``include_v21=True`` also registers the ``robot-profile/2.1`` revisions (AgiBot,
    LIBERO, the real utokyo snapshot, and the restated 2.1 forms of the synthetic
    arms).  The default stays at the V2 set so V2 seeding behaviour is unchanged.
    """
    from sqlalchemy import select

    from embodied_dataops.db.models import RobotProfile
    from embodied_dataops.ids import new_id

    if profiles is not None:
        entries = profiles
    elif include_v21 and path is None:
        entries = load_all_profiles()
    else:
        entries = load_profiles(path)
    registered = []
    for config in entries:
        existing = session.scalars(
            select(RobotProfile).where(
                RobotProfile.profile_id == config.profile_id,
                RobotProfile.profile_revision == config.profile_revision,
            )
        ).first()
        if existing is not None:
            if existing.config_sha256 != config.config_sha256():
                raise ApiError(
                    ErrorCode.CONFLICT,
                    "registered robot profile differs from the profile file",
                    details={
                        "profile_id": config.profile_id,
                        "profile_revision": config.profile_revision,
                        "registered_sha256": existing.config_sha256,
                        "file_sha256": config.config_sha256(),
                    },
                )
            registered.append(existing)
            continue
        row = RobotProfile(
            id=new_id(),
            profile_id=config.profile_id,
            profile_revision=config.profile_revision,
            authority=config.authority,
            source_evidence_asset_id=config.source_evidence_asset_id,
            config_json=config.model_dump(mode="json"),
            config_sha256=config.config_sha256(),
        )
        session.add(row)
        session.flush()
        registered.append(row)
    return registered
