"""Layered publication admission (SPEC_V2.1 §4.2, §14).

A profile's ``authority`` decides what a dataset built from it may claim:

* ``SEMANTIC_VERIFIED`` -- units, action definition, clock mapping and the applicable
  QC rules are evidenced, so the version may be exported as ``ROBOT_LEROBOT_V3`` and
  labelled ``TRAINING_READY``.
* ``FORMAT_VERIFIED`` -- structure, lengths, timing and readability are verified but
  the physics is not documented.  Such data may still be frozen and published, but
  only as ``RESEARCH_SNAPSHOT``; calling it ``TRAINING_READY`` would assert a
  semantic guarantee the source cannot support.

The level is **derived from the profile revision**, never passed in by a caller, and a
profile that cannot be found resolves to ``FORMAT_VERIFIED``: an unknown provenance is
not evidence of semantic correctness.
"""

from __future__ import annotations

from typing import Any

from embodied_dataops.domain.profiles import load_all_profiles, parse_profile, profile_index
from embodied_dataops.domain.profiles_v21 import PUBLISH_LEVEL_FORMAT, PUBLISH_LEVEL_SEMANTIC
from embodied_dataops.errors import ApiError, ErrorCode

EXPORT_KIND_ROBOT = "ROBOT_LEROBOT_V3"
EXPORT_KIND_EGO_BUNDLE = "EGO_ANNOTATION_BUNDLE"
EXPORT_KIND_RESEARCH = "RESEARCH_SNAPSHOT"
EXPORT_KINDS = (EXPORT_KIND_ROBOT, EXPORT_KIND_EGO_BUNDLE, EXPORT_KIND_RESEARCH)

#: The manifest key that freezes the level next to the profile binding.
MANIFEST_PUBLISH_LEVEL_KEY = "publish_level"


def publish_level_for_profile(
    session: Any,
    *,
    profile_id: str | None,
    profile_revision: int | None,
) -> str:
    """Resolve one profile revision's level from the registry, then the shipped files."""
    if not profile_id or profile_revision is None:
        return PUBLISH_LEVEL_FORMAT
    from sqlalchemy import select  # noqa: PLC0415 - keeps this module import-light

    from embodied_dataops.db.models import RobotProfile as RobotProfileRow  # noqa: PLC0415

    row = session.scalars(
        select(RobotProfileRow).where(
            RobotProfileRow.profile_id == str(profile_id),
            RobotProfileRow.profile_revision == int(profile_revision),
        )
    ).first()
    candidates: list[Any] = []
    if row is not None:
        try:
            candidates.append(parse_profile(dict(row.config_json)))
        except Exception:  # noqa: BLE001 - a corrupt row must not grant a level
            pass
    try:
        shipped = profile_index(load_all_profiles()).get((str(profile_id), int(profile_revision)))
    except ApiError:
        shipped = None
    if shipped is not None:
        candidates.append(shipped)
    for profile in candidates:
        return str(profile.publish_level)
    return PUBLISH_LEVEL_FORMAT


def export_kind_for_publish_level(publish_level: str) -> str:
    return EXPORT_KIND_ROBOT if publish_level == PUBLISH_LEVEL_SEMANTIC else EXPORT_KIND_RESEARCH


def training_ready_allowed(publish_level: str) -> bool:
    return publish_level == PUBLISH_LEVEL_SEMANTIC


def assert_publishable(*, publish_level: str, export_kind: str) -> None:
    """Refuse a version whose claimed kind outruns what its profile evidences."""
    if training_ready_allowed(publish_level):
        return
    if export_kind == EXPORT_KIND_RESEARCH:
        return
    raise ApiError(
        ErrorCode.PUBLISH_LEVEL_CONFLICT,
        "a FORMAT_VERIFIED profile can only be published as a RESEARCH_SNAPSHOT; "
        "TRAINING_READY requires evidenced units, action semantics and clock mapping",
        details={
            "publish_level": publish_level,
            "export_kind": export_kind,
            "required_export_kind": EXPORT_KIND_RESEARCH,
        },
    )


__all__ = [
    "EXPORT_KINDS",
    "EXPORT_KIND_EGO_BUNDLE",
    "EXPORT_KIND_RESEARCH",
    "EXPORT_KIND_ROBOT",
    "MANIFEST_PUBLISH_LEVEL_KEY",
    "assert_publishable",
    "export_kind_for_publish_level",
    "publish_level_for_profile",
    "training_ready_allowed",
]
