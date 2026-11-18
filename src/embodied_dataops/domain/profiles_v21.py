"""Robot Profile Registry — ``robot-profile/2.1`` (SPEC_V2.1 §4.2, OV-03/OV-04).

V2.1 has to describe robots whose **state and action are different widths** (LIBERO
8/7, AgiBot 16/16 with two gripper channels) and whose physical limits are simply
not documented.  The V2 schema (``robot-profile/2.0``) could not express either: it
derived both dimensions from one ``joint_names`` list and always demanded
``joint_lower_rad`` / ``joint_upper_rad`` / ``max_abs_velocity_rad_s``.

Design rules implemented here:

* **Additive, never destructive.** A 2.0 file keeps loading unchanged, so every
  published dataset keeps the same ``config_sha256``.  2.1 profiles are new
  ``profile_id`` + ``profile_revision`` pairs.
* **Dimensions are per modality.** ``state.dimension`` and ``action.dimension`` are
  independent; ``features`` must have exactly that length.
* **Units are per dimension.** ``unknown`` is a first-class value; it is *not* a
  placeholder for "assume radians".
* **A gripper channel may not claim radians.** ``role="gripper"`` with
  ``unit="rad"`` is rejected at load time (SPEC_V2.1 §4.1: "特别不能把两个夹爪值当 rad").
* **No limits means no physics QC, not "limits are zero".**  A profile without a
  ``limits`` block reports ``limits_available=False``; the QC layer must then record
  ``NOT_APPLICABLE`` instead of ``PASS``.
* **Publish level is derived, never asserted.**  Only ``SYNTHETIC_FIXTURE`` and
  ``SOURCE_DOCUMENTED`` reach ``SEMANTIC_VERIFIED``; everything else is
  ``FORMAT_VERIFIED`` and may not be labelled ``TRAINING_READY``.
"""

from __future__ import annotations

import hashlib
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from embodied_dataops.canonical.contract import canonical_json_bytes

PROFILE_SCHEMA_VERSION_21 = "robot-profile/2.1"

AUTHORITY_SYNTHETIC = "SYNTHETIC_FIXTURE"
AUTHORITY_DOCUMENTED = "SOURCE_DOCUMENTED"
AUTHORITY_DOCUMENTED_PARTIAL = "SOURCE_DOCUMENTED_PARTIAL"
AUTHORITY_UNVERIFIED = "UNVERIFIED"
AUTHORITIES_V21 = (
    AUTHORITY_SYNTHETIC,
    AUTHORITY_DOCUMENTED,
    AUTHORITY_DOCUMENTED_PARTIAL,
    AUTHORITY_UNVERIFIED,
)

#: Authorities that may support ``TRAINING_READY`` exports (SPEC_V2.1 §4.2).
SEMANTIC_AUTHORITIES_V21 = (AUTHORITY_SYNTHETIC, AUTHORITY_DOCUMENTED)

PUBLISH_LEVEL_SEMANTIC = "SEMANTIC_VERIFIED"
PUBLISH_LEVEL_FORMAT = "FORMAT_VERIFIED"
PUBLISH_LEVELS = (PUBLISH_LEVEL_SEMANTIC, PUBLISH_LEVEL_FORMAT)

Role = Literal["joint", "gripper", "unknown"]
ROLE_JOINT: Role = "joint"
ROLE_GRIPPER: Role = "gripper"
ROLE_UNKNOWN: Role = "unknown"
ROLES: tuple[Role, ...] = (ROLE_JOINT, ROLE_GRIPPER, ROLE_UNKNOWN)

UNIT_UNKNOWN = "unknown"

Alignment = Literal["NATIVE_STREAMS", "SOURCE_ROW_ALIGNED"]
ALIGNMENT_NATIVE: Alignment = "NATIVE_STREAMS"
ALIGNMENT_SOURCE_ROW: Alignment = "SOURCE_ROW_ALIGNED"
ALIGNMENTS: tuple[Alignment, ...] = (ALIGNMENT_NATIVE, ALIGNMENT_SOURCE_ROW)


class FeatureSpec(BaseModel):
    """One declared dimension of a modality (``state`` or ``action``)."""

    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=1, max_length=120)
    unit: str = Field(min_length=1, max_length=32)
    role: Role = ROLE_UNKNOWN

    @model_validator(mode="after")
    def _gripper_is_not_radians(self) -> FeatureSpec:
        # A gripper channel is a normalised opening or a linear displacement; claiming
        # radians is the exact mistake SPEC_V2.1 §4.1 calls out for the AgiBot profile.
        if self.role == ROLE_GRIPPER and self.unit in {"rad", "radian", "radians"}:
            raise ValueError(
                f"feature {self.name!r} is role=gripper but declares unit={self.unit!r}; "
                "a gripper channel is not a joint angle"
            )
        return self


class ModalitySpec(BaseModel):
    """``state`` or ``action``: an explicit dimension plus per-dimension evidence."""

    model_config = ConfigDict(extra="forbid")

    dimension: int = Field(ge=1, le=256)
    features: list[FeatureSpec] = Field(min_length=1)
    semantics: str = Field(min_length=1, max_length=80)
    unit: str = Field(default=UNIT_UNKNOWN, max_length=32)

    @model_validator(mode="after")
    def _features_match_dimension(self) -> ModalitySpec:
        if len(self.features) != self.dimension:
            raise ValueError(
                f"features has {len(self.features)} entries but dimension is {self.dimension}"
            )
        names = [feature.name for feature in self.features]
        if len(set(names)) != len(names):
            raise ValueError("feature names must be unique inside one modality")
        return self

    @property
    def names(self) -> list[str]:
        return [feature.name for feature in self.features]

    @property
    def roles(self) -> list[str]:
        return [feature.role for feature in self.features]


class LimitsSpec(BaseModel):
    """Optional physical limits.  Absent means "not documented", never "unlimited"."""

    model_config = ConfigDict(extra="forbid")

    lower: list[float]
    upper: list[float]
    max_abs_velocity: list[float]
    continuous: list[bool]

    @model_validator(mode="after")
    def _validate(self) -> LimitsSpec:
        sizes = {len(self.lower), len(self.upper), len(self.max_abs_velocity), len(self.continuous)}
        if len(sizes) != 1:
            raise ValueError("limits arrays must all have the same length")
        for low, high in zip(self.lower, self.upper, strict=True):
            if low >= high:
                raise ValueError("limits.lower must be smaller than limits.upper")
        for velocity in self.max_abs_velocity:
            if velocity <= 0:
                raise ValueError("limits.max_abs_velocity entries must be positive")
        return self

    @property
    def dimension(self) -> int:
        return len(self.lower)


class RobotProfileV21(BaseModel):
    """One immutable robot profile revision in ``robot-profile/2.1`` form."""

    model_config = ConfigDict(extra="forbid")

    schema_version: str = PROFILE_SCHEMA_VERSION_21
    profile_id: str = Field(min_length=1, max_length=80)
    profile_revision: int = Field(ge=1)
    authority: Literal[
        "SYNTHETIC_FIXTURE",
        "SOURCE_DOCUMENTED",
        "SOURCE_DOCUMENTED_PARTIAL",
        "UNVERIFIED",
    ]
    source_dataset: str = ""
    source_evidence_asset_id: str | None = None

    state: ModalitySpec
    action: ModalitySpec
    limits: LimitsSpec | None = None

    cameras: list[str] = Field(min_length=1)
    optional_cameras: list[str] = Field(default_factory=list)
    #: How many camera features the *source* dataset declares.  ``cameras`` above is the
    #: subset the canonical view keeps (canonical-episode/2.0 has ``front`` + optional
    #: ``wrist``); the rest are preview-only or dropped with a diagnostic.
    expected_source_cameras: int = Field(default=0, ge=0)
    aligned_fps: int = Field(ge=1, le=240)
    preview_fps: int = Field(ge=1, le=240)
    max_clock_offset_ns: int = Field(ge=0)
    timestamp_alignment: Alignment = ALIGNMENT_NATIVE
    semantic_qc_enabled: bool = True
    notes: str = ""

    @model_validator(mode="after")
    def _cross_field(self) -> RobotProfileV21:
        if self.limits is not None and self.limits.dimension != self.state.dimension:
            raise ValueError(
                f"limits describe {self.limits.dimension} dims but state.dimension is "
                f"{self.state.dimension}"
            )
        overlap = set(self.cameras) & set(self.optional_cameras)
        if overlap:
            raise ValueError(f"cameras listed as both required and optional: {sorted(overlap)}")
        if self.expected_source_cameras and self.expected_source_cameras < len(self.cameras):
            raise ValueError(
                f"expected_source_cameras={self.expected_source_cameras} is smaller than the "
                f"{len(self.cameras)} canonical cameras"
            )
        if self.semantic_qc_enabled and self.limits is None:
            raise ValueError(
                "semantic_qc_enabled=true requires a limits block; without physical limits the "
                "physics rules must be NOT_APPLICABLE, not silently enabled"
            )
        return self

    # -- dimensions ---------------------------------------------------------
    @property
    def state_dim(self) -> int:
        return self.state.dimension

    @property
    def action_dim(self) -> int:
        return self.action.dimension

    @property
    def unequal_dimensions(self) -> bool:
        return self.state.dimension != self.action.dimension

    # -- namespace used by the V2 canonical/QC/export code ------------------
    @property
    def joint_names(self) -> list[str]:
        """State dimension names (the V2 code calls any state channel a 'joint')."""
        return self.state.names

    @property
    def action_names(self) -> list[str]:
        return self.action.names

    @property
    def state_semantics(self) -> str:
        return self.state.semantics

    @property
    def action_semantics(self) -> str:
        return self.action.semantics

    @property
    def state_unit(self) -> str:
        return self.state.unit

    @property
    def action_unit(self) -> str:
        return self.action.unit

    @property
    def required_camera_names(self) -> list[str]:
        return list(self.cameras)

    @property
    def optional_camera_names(self) -> list[str]:
        return list(self.optional_cameras)

    @property
    def limits_available(self) -> bool:
        return self.limits is not None

    @property
    def joint_lower_rad(self) -> list[float]:
        return list(self.limits.lower) if self.limits is not None else []

    @property
    def joint_upper_rad(self) -> list[float]:
        return list(self.limits.upper) if self.limits is not None else []

    @property
    def max_abs_velocity_rad_s(self) -> list[float]:
        return list(self.limits.max_abs_velocity) if self.limits is not None else []

    @property
    def continuous_joints(self) -> list[bool]:
        return list(self.limits.continuous) if self.limits is not None else []

    # -- publication --------------------------------------------------------
    @property
    def semantic_authority(self) -> bool:
        return self.authority in SEMANTIC_AUTHORITIES_V21

    @property
    def publish_level(self) -> str:
        return PUBLISH_LEVEL_SEMANTIC if self.semantic_authority else PUBLISH_LEVEL_FORMAT

    @property
    def training_ready_allowed(self) -> bool:
        return self.semantic_authority and self.semantic_qc_enabled

    def canonical_json(self) -> bytes:
        return canonical_json_bytes(self.model_dump(mode="json"))

    def config_sha256(self) -> str:
        return hashlib.sha256(self.canonical_json()).hexdigest()

    def as_legacy_dict(self) -> dict[str, Any]:
        """Flatten to the 2.0 vocabulary for consumers that only know the old shape.

        ``joint_names`` keeps the state names because that is what the V2 canonical
        metadata records; the action width is carried separately by
        :attr:`action_dim`, which every V2.1-aware caller must read.
        """
        return {
            "profile_id": self.profile_id,
            "profile_revision": self.profile_revision,
            "authority": self.authority,
            "state_dim": self.state.dimension,
            "action_dim": self.action.dimension,
            "joint_names": self.state.names,
            "action_names": self.action.names,
            "required_camera_names": self.required_camera_names,
            "optional_camera_names": self.optional_camera_names,
            "publish_level": self.publish_level,
            "semantic_qc_enabled": self.semantic_qc_enabled,
            "limits_available": self.limits_available,
        }
