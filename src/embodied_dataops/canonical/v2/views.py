"""``canonical_views`` registration for canonical-episode/2.0 (V2 SPEC 2.2).

Three views are registered per canonical version:

* ``NATIVE``      -- the source-order indices (no resampling, no frame loss);
* ``ALIGNED_TRAINING`` -- the derived 10 Hz training grid;
* ``PREVIEW``     -- the regenerable display videos.

Each row is keyed by ``(episode_id, canonical_version, view_kind, config_sha256)``
where ``config_sha256`` is the SHA-256 of the canonical JSON of the exact view
generation parameters (profile config, fps, cameras, tolerances).  Registration is
idempotent, so a retried NORMALIZE_EPISODE task cannot create duplicates.
"""

from __future__ import annotations

import hashlib
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from embodied_dataops.canonical.contract import canonical_json_bytes
from embodied_dataops.canonical.v2.contract import (
    ALIGNED_FPS,
    ALIGNED_INTERVAL_NS,
    VIEW_ALIGNED_TRAINING,
    VIEW_KINDS,
    VIEW_NATIVE,
    VIEW_PREVIEW,
)
from embodied_dataops.canonical.v2.model import CanonicalV2Source
from embodied_dataops.db.models import CanonicalView
from embodied_dataops.ids import new_id


def view_config(
    view_kind: str,
    *,
    source: CanonicalV2Source,
    aligned_fps: int = ALIGNED_FPS,
    interval_ns: int = ALIGNED_INTERVAL_NS,
    image_tolerance_ns: int = 50_000_000,
    state_tolerance_ns: int = 50_000_000,
    action_tolerance_ns: int = 100_000_000,
    preview_fps: int = 10,
    preview_width: int = 640,
    preview_height: int = 480,
) -> dict[str, Any]:
    """Canonical view generation parameters (the hash target of the view row)."""
    if view_kind not in VIEW_KINDS:
        raise ValueError(f"unknown canonical view kind {view_kind!r}")
    base: dict[str, Any] = {
        "view_kind": view_kind,
        "profile_id": source.profile_id,
        "profile_revision": source.profile_revision,
        "profile_config_sha256": source.profile_config_sha256,
        "camera_names": list(source.camera_names),
        "alignment_mode": source.alignment_mode,
    }
    if view_kind == VIEW_NATIVE:
        base["native"] = True
        base["source_clock_domains"] = {
            **{f"camera_{camera.name}": camera.clock_domain for camera in source.cameras},
            **({"state": source.state.clock_domain} if source.state is not None else {}),
            **({"action": source.action.clock_domain} if source.action is not None else {}),
        }
    elif view_kind == VIEW_ALIGNED_TRAINING:
        base.update(
            {
                "fps": int(aligned_fps),
                "interval_ns": int(interval_ns),
                "image_tolerance_ns": int(image_tolerance_ns),
                "state_tolerance_ns": int(state_tolerance_ns),
                "action_tolerance_ns": int(action_tolerance_ns),
            }
        )
    else:
        base.update(
            {
                "fps": int(preview_fps),
                "width": int(preview_width),
                "height": int(preview_height),
                "lossy": True,
                "note": "regenerable preview, not training truth",
            }
        )
    return base


def config_sha256(config: dict[str, Any]) -> str:
    return hashlib.sha256(canonical_json_bytes(config)).hexdigest()


def register_canonical_views(
    session: Session,
    *,
    episode_id: str,
    canonical_version: int,
    metadata_asset_id: str | None,
    configs: dict[str, dict[str, Any]],
) -> list[CanonicalView]:
    """Idempotently register NATIVE / ALIGNED_TRAINING / PREVIEW rows."""
    view_kind_by_config = {
        str(config["view_kind"]): (view_kind, config) for view_kind, config in configs.items()
    }
    rows: list[CanonicalView] = []
    for view_kind, (declared_kind, config) in view_kind_by_config.items():
        if declared_kind != view_kind:
            raise ValueError("view config kind does not match its registration key")
        digest = config_sha256(config)
        existing = session.scalars(
            select(CanonicalView).where(
                CanonicalView.episode_id == episode_id,
                CanonicalView.canonical_version == int(canonical_version),
                CanonicalView.view_kind == view_kind,
                CanonicalView.config_sha256 == digest,
            )
        ).first()
        if existing is not None:
            rows.append(existing)
            continue
        row = CanonicalView(
            id=new_id(),
            episode_id=episode_id,
            canonical_version=int(canonical_version),
            view_kind=view_kind,
            config_sha256=digest,
            metadata_asset_id=metadata_asset_id,
        )
        session.add(row)
        session.flush()
        rows.append(row)
    return rows


def build_all_view_configs(
    source: CanonicalV2Source,
    **overrides: Any,
) -> dict[str, dict[str, Any]]:
    return {
        view_kind: view_config(view_kind, source=source, **overrides)
        for view_kind in (VIEW_NATIVE, VIEW_ALIGNED_TRAINING, VIEW_PREVIEW)
    }


__all__ = [
    "build_all_view_configs",
    "config_sha256",
    "register_canonical_views",
    "view_config",
]
