"""Dataset version service layer (SPEC 10.1, AT-15/16/17/18).

Design note (working-first v1): the HTTP route calls ``create_version`` which
validates eligibility, freezes the manifest, runs the official LeRobot export and
its independent-reader round-trip **synchronously**, then only returns.  The
operation is idempotent in the sense that a retry with the same membership cannot
produce duplicate assets: a failed version stays FAILED and the caller creates a
new version number.  A production deployment can move the export into a worker
without changing the persisted contract (BUILDING -> READY/FAILED,
PUBLISHED immutable).

Everything is scoped to the caller's transaction; these functions only flush.
"""

from __future__ import annotations

import hashlib
import tempfile
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from embodied_dataops import TARGET_FPS
from embodied_dataops.api.auth import Principal
from embodied_dataops.canonical.contract import (
    ROBOT_PROFILE,
    CanonicalEpisode,
    CanonicalMetadata,
    canonical_keys,
    read_samples_parquet,
    validate_samples_table,
)
from embodied_dataops.datasets import lerobot_reader
from embodied_dataops.datasets.lerobot_writer import (
    cleanup,
    export_canonical_episodes,
    tar_directory,
)
from embodied_dataops.datasets.manifest import (
    build_manifest,
    manifest_bytes,
    manifest_sha256,
    parse_manifest,
    utc_timestamp,
)
from embodied_dataops.db.models import (
    AnnotationReview,
    AnnotationSet,
    Asset,
    Dataset,
    DatasetVersion,
    DatasetVersionEpisode,
    Episode,
    QualityReport,
    SourceImport,
)
from embodied_dataops.domain.enums import (
    AnnotationStatus,
    DatasetVersionState,
    EpisodeType,
    QualityStatus,
)
from embodied_dataops.domain.state_machines import DATASET_VERSION_TRANSITIONS
from embodied_dataops.errors import ApiError, ErrorCode
from embodied_dataops.ids import new_id
from embodied_dataops.observability.audit import record_audit
from embodied_dataops.storage.base import StorageBackend, get_storage, sha256_file
from embodied_dataops.workflow.outbox import utcnow

MANIFEST_KEY_TEMPLATE = "datasets/{dataset_id}/v{version}/manifest.json"
EXPORT_KEY_TEMPLATE = "datasets/{dataset_id}/v{version}/export/lerobot-v3.tar.gz"

#: A license must be explicitly known before an episode may enter a publishable
#: dataset version (SPEC 10.1 last bullet).
UNKNOWN_LICENSES = frozenset({"", "unknown", "none", "n/a", "na", "unlicensed", "proprietary-unknown"})

#: WARN releases require an explicit reviewer justification of at least 10 chars
#: (SPEC 7.2).
WARN_RELEASE_NOTE_MIN_CHARS = 10


# ---------------------------------------------------------------------------
# small helpers
# ---------------------------------------------------------------------------
@dataclass
class EpisodeEligibility:
    episode: Episode
    quality_report_id: str
    annotation_set_id: str
    canonical_digest: str
    source_sha256: str
    task_label: str


@dataclass
class ExportBuild:
    asset_id: str
    archive_path: Path
    repo_id: str
    expected: list[lerobot_reader.ExpectedEpisode]
    episode_count: int
    total_frames: int
    workdir: Path
    package_version: str | None = None
    # --- V2 (canonical-episode/2.0) export contract -------------------------
    schema_version: str = "lerobot-v3-export/1.0"
    camera_keys: list[str] = field(default_factory=list)
    state_dim: int = 6
    action_dim: int = 6
    aligned_fps: int = TARGET_FPS
    #: Feature labels frozen into the export; a 2.1 profile can have 8/7.
    state_names: list[str] = field(default_factory=list)
    action_names: list[str] = field(default_factory=list)


@dataclass
class FreezeResult:
    manifest: dict[str, Any]
    manifest_sha256: str
    manifest_asset_id: str
    roundtrip: lerobot_reader.RoundTripResult
    episode_count: int
    total_frames: int
    extra: dict[str, Any] = field(default_factory=dict)


def _not_found(entity: str, entity_id: str) -> ApiError:
    return ApiError(ErrorCode.NOT_FOUND, f"{entity} not found: {entity_id}")


def _store_bytes(
    session: Session,
    *,
    key: str,
    data: bytes,
    media_type: str,
    origin: str,
    logical_name: str,
    meta: dict[str, Any] | None = None,
) -> Asset:
    stored = get_storage().put_bytes(key, data, content_type=media_type)
    asset = Asset(
        id=new_id(),
        sha256=stored.sha256 or hashlib.sha256(data).hexdigest(),
        byte_size=len(data),
        media_type=media_type,
        object_key=key,
        origin=origin,
        logical_name=logical_name,
        meta_json=meta or {},
    )
    session.add(asset)
    session.flush()
    return asset


def _store_file(
    session: Session,
    *,
    key: str,
    path: Path,
    media_type: str,
    origin: str,
    logical_name: str,
    meta: dict[str, Any] | None = None,
) -> Asset:
    stored = get_storage().put_file(key, path, content_type=media_type)
    asset = Asset(
        id=new_id(),
        sha256=stored.sha256 or sha256_file(path),
        byte_size=path.stat().st_size,
        media_type=media_type,
        object_key=key,
        origin=origin,
        logical_name=logical_name,
        meta_json=meta or {},
    )
    session.add(asset)
    session.flush()
    return asset


def _sha256_stored_object(storage: StorageBackend, asset: Asset) -> str:
    """SHA-256 of the *stored bytes* (not the metadata), for publish re-verification."""
    with tempfile.TemporaryDirectory(prefix="dataops-sha-") as directory:
        target = Path(directory) / "object.bin"
        storage.download_to(asset.object_key, target)
        return sha256_file(target)


def _ensure_coverage_snapshot(session: Session, version: DatasetVersion) -> None:
    """V2 D2-6 hook: freeze scenario coverage when a version is PUBLISHED.

    The module lands with the coverage milestone; while it is absent the publish
    path stays unchanged (guarded import, no fabricated snapshot).
    """
    try:
        from embodied_dataops.feedback.coverage import ensure_coverage_snapshot
    except ImportError:  # pragma: no cover - module lands with D2-M6
        return
    ensure_coverage_snapshot(session, version)


def _set_state(version: DatasetVersion, target: DatasetVersionState) -> None:
    current = DatasetVersionState(version.state)
    if target not in DATASET_VERSION_TRANSITIONS.get(current, frozenset()):
        if current is DatasetVersionState.PUBLISHED:
            raise ApiError(
                ErrorCode.VERSION_IMMUTABLE,
                "Published dataset versions are immutable",
                details={"version_id": version.id, "state": current.value},
            )
        raise ApiError(
            ErrorCode.VERSION_NOT_READY,
            f"Illegal dataset version transition {current.value} -> {target.value}",
            details={"version_id": version.id, "from": current.value, "to": target.value},
        )
    version.state = target.value


# ---------------------------------------------------------------------------
# dataset
# ---------------------------------------------------------------------------
SLUG_MAX_LEN = 120


def _validate_slug(slug: str) -> str:
    import re

    if not isinstance(slug, str):
        raise ApiError(ErrorCode.INVALID_INPUT, "slug must be a string")
    value = slug.strip()
    if not value or len(value) > SLUG_MAX_LEN:
        raise ApiError(
            ErrorCode.INVALID_INPUT,
            f"slug must be 1..{SLUG_MAX_LEN} characters",
            details={"slug": value[:200]},
        )
    if not re.match(r"^[a-z0-9][a-z0-9._-]*$", value):
        raise ApiError(
            ErrorCode.INVALID_INPUT,
            "slug may only contain lowercase letters, digits, '.', '_' and '-'",
            details={"slug": value},
        )
    return value


def create_dataset(
    session: Session,
    *,
    slug: str,
    description: str = "",
    actor: Principal | None = None,
    request_id: str | None = None,
) -> Dataset:
    value = _validate_slug(slug)
    existing = session.scalars(select(Dataset).where(Dataset.slug == value)).first()
    if existing is not None:
        raise ApiError(
            ErrorCode.CONFLICT,
            f"Dataset slug already exists: {value}",
            details={"slug": value, "dataset_id": existing.id},
        )
    dataset = Dataset(
        id=new_id(),
        slug=value,
        description=(description or "").strip(),
        created_at=utcnow(),
    )
    session.add(dataset)
    session.flush()
    record_audit(
        session,
        actor=actor,
        action="dataset.create",
        entity_type="dataset",
        entity_id=dataset.id,
        after={"slug": dataset.slug},
        request_id=request_id,
    )
    return dataset


def get_dataset(session: Session, dataset_id: str) -> Dataset:
    dataset = session.get(Dataset, dataset_id)
    if dataset is None:
        raise _not_found("Dataset", dataset_id)
    return dataset


# ---------------------------------------------------------------------------
# eligibility (AT-15, AT-16)
# ---------------------------------------------------------------------------
def _latest_quality_report(session: Session, episode_id: str) -> QualityReport | None:
    statement = (
        select(QualityReport)
        .where(QualityReport.episode_id == episode_id)
        .order_by(QualityReport.created_at.desc(), QualityReport.id.desc())
    )
    return session.scalars(statement).first()


def _approved_annotation_set(session: Session, episode_id: str) -> AnnotationSet | None:
    statement = (
        select(AnnotationSet)
        .where(
            AnnotationSet.episode_id == episode_id,
            AnnotationSet.status == AnnotationStatus.APPROVED.value,
        )
        .order_by(AnnotationSet.created_at.desc(), AnnotationSet.id.desc())
    )
    return session.scalars(statement).first()


def _source_sha256(episode: Episode, source: SourceImport | None, metadata: CanonicalMetadata) -> str:
    for candidate in (
        source.sha256 if source is not None else None,
        metadata.source_asset_sha256,
    ):
        if candidate and len(candidate) == 64:
            return candidate
    raise ApiError(
        ErrorCode.INCOMPATIBLE_EPISODE,
        "Episode has no traceable source SHA-256 for the dataset manifest",
        details={"episode_id": episode.id},
    )


def _load_canonical_metadata(episode: Episode) -> CanonicalMetadata:
    keys = canonical_keys(episode.id, episode.canonical_version or 1)
    storage = get_storage()
    if not storage.exists(keys["metadata"]):
        raise ApiError(
            ErrorCode.INCOMPATIBLE_EPISODE,
            "Episode has no canonical metadata object; run NORMALIZE_EPISODE first",
            details={"episode_id": episode.id},
        )
    try:
        return CanonicalMetadata.model_validate_json(storage.get_bytes(keys["metadata"]))
    except ApiError:
        raise
    except Exception as exc:
        raise ApiError(
            ErrorCode.INCOMPATIBLE_EPISODE,
            "Episode canonical metadata is not a valid canonical-episode/1.0 document",
            details={"episode_id": episode.id, "reason": f"{type(exc).__name__}: {exc}"},
        ) from exc


def validate_episode_eligible(session: Session, episode_id: str) -> EpisodeEligibility:
    """Enforce every SPEC 10.1 pre-export condition or raise the 409 error code."""
    episode = session.get(Episode, episode_id)
    if episode is None:
        raise _not_found("Episode", episode_id)

    # V2 (canonical-episode/2.0) episodes have their own profile-bound eligibility
    # rules; the V1 rules below reject 7D/multi-camera metadata and are unchanged.
    if str(episode.canonical_schema_version or "1.0") == "2.0":
        from embodied_dataops.datasets.export_v2 import validate_v2_episode_eligible

        return validate_v2_episode_eligible(session, episode)

    if episode.episode_type != EpisodeType.ROBOT_EPISODE.value:
        raise ApiError(
            ErrorCode.INCOMPATIBLE_EPISODE,
            "Only robot_episode data can be exported as LeRobot robot trajectories",
            details={"episode_id": episode_id, "episode_type": episode.episode_type},
        )

    if episode.quality_status not in {QualityStatus.PASS.value, QualityStatus.WARN.value}:
        raise ApiError(
            ErrorCode.QC_STATUS_NOT_ELIGIBLE,
            "Episode quality verdict is not PASS or WARN",
            details={"episode_id": episode_id, "quality_status": episode.quality_status},
        )
    quality = _latest_quality_report(session, episode_id)
    if quality is None:
        raise ApiError(
            ErrorCode.QC_STATUS_NOT_ELIGIBLE,
            "Episode has no stored QualityReport",
            details={"episode_id": episode_id},
        )

    annotation = _approved_annotation_set(session, episode_id)
    if annotation is None:
        raise ApiError(
            ErrorCode.ANNOTATION_NOT_APPROVED,
            "Episode has no APPROVED AnnotationSet",
            details={"episode_id": episode_id},
        )
    if episode.quality_status == QualityStatus.WARN.value:
        review = session.scalars(
            select(AnnotationReview).where(AnnotationReview.annotation_set_id == annotation.id)
        ).first()
        note = (review.note or "").strip() if review is not None else ""
        if len(note) < WARN_RELEASE_NOTE_MIN_CHARS:
            raise ApiError(
                ErrorCode.ANNOTATION_NOT_APPROVED,
                f"WARN release requires an approval note of at least {WARN_RELEASE_NOTE_MIN_CHARS} characters",
                details={"episode_id": episode_id, "annotation_set_id": annotation.id},
            )

    source = session.get(SourceImport, episode.source_import_id)
    license_value = (source.license or "").strip().lower() if source is not None else ""
    if license_value in UNKNOWN_LICENSES:
        raise ApiError(
            ErrorCode.LICENSE_UNKNOWN,
            "Episode source licence is unknown; unknown-licence data cannot be published in a dataset version",
            details={"episode_id": episode_id, "license": source.license if source else None},
        )

    metadata = _load_canonical_metadata(episode)
    if metadata.episode_type != EpisodeType.ROBOT_EPISODE.value:
        raise ApiError(
            ErrorCode.INCOMPATIBLE_EPISODE,
            "Canonical metadata episode_type is not robot_episode",
            details={"episode_id": episode_id, "canonical_episode_type": metadata.episode_type},
        )
    if (
        int(metadata.target_fps) != TARGET_FPS
        or int(metadata.state_dim) != 6
        or int(metadata.action_dim) != 6
        or metadata.camera_name != "front"
    ):
        raise ApiError(
            ErrorCode.INCOMPATIBLE_EPISODE,
            "Episode is not the 10 Hz / 640x480 / 6D / single-front-camera canonical profile",
            details={
                "episode_id": episode_id,
                "target_fps": metadata.target_fps,
                "state_dim": metadata.state_dim,
                "action_dim": metadata.action_dim,
                "camera_name": metadata.camera_name,
            },
        )
    if metadata.robot_profile and metadata.robot_profile != ROBOT_PROFILE:
        raise ApiError(
            ErrorCode.INCOMPATIBLE_EPISODE,
            f"Episode robot profile is not {ROBOT_PROFILE}",
            details={"episode_id": episode_id, "robot_profile": metadata.robot_profile},
        )
    if not (metadata.task_label or "").strip():
        raise ApiError(
            ErrorCode.INCOMPATIBLE_EPISODE,
            "Episode has no task label; a LeRobot export requires one task per episode",
            details={"episode_id": episode_id},
        )
    canonical_digest = episode.canonical_sha256 or ""
    if len(canonical_digest) != 64:
        raise ApiError(
            ErrorCode.INCOMPATIBLE_EPISODE,
            "Episode canonical_sha256 is missing; normalization must complete first",
            details={"episode_id": episode_id},
        )

    keys = canonical_keys(episode.id, episode.canonical_version or 1)
    storage = get_storage()
    if not storage.exists(keys["samples"]):
        raise ApiError(
            ErrorCode.INCOMPATIBLE_EPISODE,
            "Episode has no canonical samples.parquet object",
            details={"episode_id": episode_id},
        )
    if not storage.exists(keys["video"]):
        raise ApiError(
            ErrorCode.INCOMPATIBLE_EPISODE,
            "Episode has no canonical frames/front.mp4 object",
            details={"episode_id": episode_id},
        )

    return EpisodeEligibility(
        episode=episode,
        quality_report_id=quality.id,
        annotation_set_id=annotation.id,
        canonical_digest=canonical_digest,
        source_sha256=_source_sha256(episode, source, metadata),
        task_label=(metadata.task_label or "").strip(),
    )


# ---------------------------------------------------------------------------
# version creation
# ---------------------------------------------------------------------------
def _normalise_episode_ids(episode_ids: Iterable[str]) -> list[str]:
    values = [str(value) for value in episode_ids]
    if not values:
        raise ApiError(ErrorCode.EMPTY_VERSION, "A dataset version needs at least one episode")
    if len(set(values)) != len(values):
        raise ApiError(
            ErrorCode.INVALID_INPUT,
            "episode_ids contains duplicate values",
            details={"duplicate_count": len(values) - len(set(values))},
        )
    return sorted(values)


def _next_version_number(session: Session, dataset_id: str) -> int:
    current = session.scalars(
        select(func.max(DatasetVersion.version)).where(DatasetVersion.dataset_id == dataset_id)
    ).one()
    return int(current or 0) + 1


def _resolve_annotation_revision_id(session: Session, annotation_set_id: str | None) -> str | None:
    """V2 D2-3 hook: bind the approved immutable annotation revision (None for v1)."""
    if not annotation_set_id:
        return None
    try:
        from embodied_dataops.annotations.revisions import resolve_approved_revision_id
    except ImportError:  # pragma: no cover - module lands with D2-M3
        return None
    try:
        return resolve_approved_revision_id(session, annotation_set_id)
    except Exception:  # pragma: no cover - legacy v1 sets have no revision rows
        return None


def _snapshot_episodes(
    session: Session,
    version: DatasetVersion,
    eligibility: Sequence[EpisodeEligibility],
) -> None:
    for position, entry in enumerate(sorted(eligibility, key=lambda item: item.episode.id)):
        session.add(
            DatasetVersionEpisode(
                version_id=version.id,
                episode_id=entry.episode.id,
                canonical_digest=entry.canonical_digest,
                quality_report_id=entry.quality_report_id,
                annotation_set_id=entry.annotation_set_id,
                annotation_revision_id=_resolve_annotation_revision_id(
                    session, entry.annotation_set_id
                ),
                position=position,
            )
        )
    session.flush()


def _rows_are_canonical_v2(session: Session, rows: Sequence[DatasetVersionEpisode]) -> bool:
    """True when every member Episode is a canonical-episode/2.0 episode."""
    if not rows:
        return False
    for row in rows:
        episode = session.get(Episode, row.episode_id)
        if episode is None or str(episode.canonical_schema_version or "1.0") != "2.0":
            return False
    return True


def _contains_canonical_v2_episode(session: Session, episode_ids: Sequence[str]) -> bool:
    """True when at least one selected Episode is canonical-episode/2.0 (mixed -> V2 gate)."""
    for episode_id in episode_ids:
        episode = session.get(Episode, episode_id)
        if episode is not None and str(episode.canonical_schema_version or "1.0") == "2.0":
            return True
    return False


def _manifest_digest_of(stored: bytes) -> str:
    """Recompute the canonical manifest digest for either manifest schema."""
    import json

    payload = json.loads(stored.decode("utf-8"))
    if payload.get("schema_version") == "dataset-manifest/2.0":
        from embodied_dataops.datasets.manifest_v2 import manifest_v2_sha256, parse_manifest_v2

        return manifest_v2_sha256(parse_manifest_v2(stored))
    return manifest_sha256(parse_manifest(stored))


def _parse_stored_manifest(stored: bytes) -> dict[str, Any]:
    import json

    payload = json.loads(stored.decode("utf-8"))
    if payload.get("schema_version") == "dataset-manifest/2.0":
        from embodied_dataops.datasets.manifest_v2 import parse_manifest_v2

        return parse_manifest_v2(stored)
    return parse_manifest(stored)


def _episode_profile_id(episode: Episode) -> str | None:
    """The profile an episode was normalised against.

    V2.1 episodes carry ``profile_id`` on the row; V1 episodes only record the
    binding inside their canonical metadata, which is where the numbers came from.
    """
    if episode.profile_id:
        return str(episode.profile_id)
    try:
        metadata = _load_canonical_metadata(episode)
    except ApiError:
        return None
    profile = getattr(metadata, "robot_profile", None)
    return str(profile) if profile else None


def _shipped_revision(profile_id: str, profile_revision: int | None) -> int:
    """Resolve the revision a V1 binding refers to (the newest shipped revision)."""
    if profile_revision:
        return int(profile_revision)
    from embodied_dataops.domain.profiles import load_all_profiles, profile_index

    revisions = [revision for (pid, revision) in profile_index(load_all_profiles()) if pid == profile_id]
    return max(revisions) if revisions else 0


def _manifest_profile_block(
    session: Session, bindings: Sequence[tuple[str | None, int | None]]
) -> dict[str, Any]:
    """Freeze the publish level that this version's episodes actually evidence.

    A version may only claim the level of its **weakest** profile revision: mixing a
    ``SEMANTIC_VERIFIED`` episode with a ``FORMAT_VERIFIED`` one cannot yield a
    training-ready claim.  Returning an empty ``publish_level`` is deliberate -- the
    publish gate treats a manifest without a level as unpublishable instead of
    skipping the check (fail-closed).
    """
    from embodied_dataops.datasets.publish_policy import (  # noqa: PLC0415
        MANIFEST_PUBLISH_LEVEL_KEY,
        publish_level_for_profile,
    )
    from embodied_dataops.domain.profiles_v21 import (  # noqa: PLC0415
        PUBLISH_LEVEL_FORMAT,
        PUBLISH_LEVEL_SEMANTIC,
    )

    levels: dict[tuple[str, int], str] = {}
    for profile_id, profile_revision in bindings:
        if not profile_id:
            # No profile binding means no evidence at all: leave the level empty so
            # publish refuses instead of assuming the weakest *named* level.
            continue
        key = (str(profile_id), _shipped_revision(str(profile_id), profile_revision))
        if key in levels:
            continue
        levels[key] = publish_level_for_profile(
            session, profile_id=key[0], profile_revision=key[1]
        )
    if not levels:
        return {MANIFEST_PUBLISH_LEVEL_KEY: ""}
    rank = {PUBLISH_LEVEL_SEMANTIC: 2, PUBLISH_LEVEL_FORMAT: 1}
    weakest = min(levels.values(), key=lambda level: rank.get(level, 0))
    return {
        MANIFEST_PUBLISH_LEVEL_KEY: weakest,
        "profiles": [
            {
                "profile_id": profile_id,
                "profile_revision": revision,
                MANIFEST_PUBLISH_LEVEL_KEY: level,
            }
            for (profile_id, revision), level in sorted(levels.items())
        ],
    }


def _freeze_manifest(session: Session, version: DatasetVersion, dataset: Dataset) -> tuple[dict, str, Asset]:
    rows = list(
        session.scalars(
            select(DatasetVersionEpisode)
            .where(DatasetVersionEpisode.version_id == version.id)
            .order_by(DatasetVersionEpisode.position)
        )
    )
    if not rows:
        raise ApiError(ErrorCode.EMPTY_VERSION, "Dataset version has no episode snapshots")
    if _rows_are_canonical_v2(session, rows):
        from embodied_dataops.datasets.export_v2 import freeze_manifest_v2

        return freeze_manifest_v2(session, version, dataset, rows)
    episodes: list[dict[str, Any]] = []
    profile_bindings: list[tuple[str | None, int | None]] = []
    for row in rows:
        episode = session.get(Episode, row.episode_id)
        if episode is None:  # pragma: no cover - FK protects this
            raise _not_found("Episode", row.episode_id)
        profile_bindings.append((_episode_profile_id(episode), episode.profile_revision))
        source = session.get(SourceImport, episode.source_import_id)
        source_sha = ""
        if source is not None and source.sha256 and len(source.sha256) == 64:
            source_sha = source.sha256
        if not source_sha:
            metadata = _load_canonical_metadata(episode)
            source_sha = metadata.source_asset_sha256
        episodes.append(
            {
                "episode_id": row.episode_id,
                "canonical_sha256": row.canonical_digest,
                "source_sha256": source_sha,
                "quality_report_id": row.quality_report_id,
                "annotation_set_id": row.annotation_set_id,
            }
        )
    manifest = build_manifest(
        dataset_slug=dataset.slug,
        version=int(version.version),
        episodes=episodes,
        profile=_manifest_profile_block(session, profile_bindings),
        created_at=utc_timestamp(),
    )
    body = manifest_bytes(manifest)
    digest = manifest_sha256(manifest)
    key = MANIFEST_KEY_TEMPLATE.format(dataset_id=version.dataset_id, version=version.version)
    asset = _store_bytes(
        session,
        key=key,
        data=body,
        media_type="application/json",
        origin="export",
        logical_name=f"{dataset.slug}-v{version.version}-manifest.json",
        meta={
            "kind": "dataset_manifest",
            "dataset_id": version.dataset_id,
            "version": version.version,
            "episode_count": len(episodes),
        },
    )
    if asset.sha256 != digest:  # pragma: no cover - defensive
        raise ApiError(ErrorCode.INTERNAL_ERROR, "Manifest asset digest does not match canonical manifest")
    version.manifest_asset_id = asset.id
    version.manifest_sha256 = digest
    session.flush()
    return manifest, digest, asset


def build_export(session: Session, version: DatasetVersion) -> ExportBuild:
    """Materialise the canonical episodes and export them with the official Writer.

    The caller owns the returned ``workdir`` and must call ``cleanup`` on it.
    """
    dataset = get_dataset(session, version.dataset_id)
    snapshots = list(
        session.scalars(
            select(DatasetVersionEpisode)
            .where(DatasetVersionEpisode.version_id == version.id)
            .order_by(DatasetVersionEpisode.position)
        )
    )
    if not snapshots:
        raise ApiError(ErrorCode.EMPTY_VERSION, "Dataset version has no episode snapshots")
    snapshots.sort(key=lambda row: row.episode_id)
    if _rows_are_canonical_v2(session, snapshots):
        from embodied_dataops.datasets.export_v2 import build_export_v2

        return build_export_v2(session, version, snapshots=snapshots)

    storage = get_storage()
    workdir = Path(tempfile.mkdtemp(prefix=f"dataops-export-{version.id[:8]}-"))
    export_root = workdir / "lerobot-v3"
    episodes: list[CanonicalEpisode] = []
    expected: list[lerobot_reader.ExpectedEpisode] = []
    try:
        for snapshot in snapshots:
            episode = session.get(Episode, snapshot.episode_id)
            if episode is None:
                raise _not_found("Episode", snapshot.episode_id)
            keys = canonical_keys(episode.id, episode.canonical_version or 1)
            local_dir = workdir / "canonical" / episode.id
            local_dir.mkdir(parents=True, exist_ok=True)
            metadata_path = local_dir / "metadata.json"
            storage.download_to(keys["metadata"], metadata_path)
            metadata = CanonicalMetadata.model_validate_json(metadata_path.read_bytes())
            samples_path = local_dir / "samples.parquet"
            storage.download_to(keys["samples"], samples_path)
            video_path = local_dir / "front.mp4"
            storage.download_to(keys["video"], video_path)

            samples = read_samples_parquet(samples_path)
            contract_errors = validate_samples_table(samples, "robot_episode")
            if contract_errors:
                raise ApiError(
                    ErrorCode.INCOMPATIBLE_EPISODE,
                    "Canonical samples violate canonical-episode/1.0 and cannot be exported",
                    details={"episode_id": episode.id, "errors": contract_errors[:10]},
                )
            episodes.append(
                CanonicalEpisode(metadata=metadata, samples=samples, video_path=video_path)
            )
            rows = samples.to_pylist()
            expected.append(
                lerobot_reader.ExpectedEpisode(
                    episode_id=episode.id,
                    task_label=(metadata.task_label or "").strip(),
                    samples=[
                        lerobot_reader.ExpectedSample(
                            frame_index=int(row["frame_index"]),
                            t_ns=int(row["t_ns"]),
                            state=[float(value) for value in row["state"]],
                            action=[float(value) for value in row["action"]],
                        )
                        for row in rows
                    ],
                )
            )

        if not episodes:
            raise ApiError(ErrorCode.EMPTY_VERSION, "Dataset version has no canonical episodes")
        repo_id = f"embodied-dataops/{dataset.slug}-v{version.version}"
        result = export_canonical_episodes(export_root, episodes, repo_id=repo_id)
        archive = tar_directory(export_root, workdir / "lerobot-v3.tar.gz")
        key = EXPORT_KEY_TEMPLATE.format(dataset_id=version.dataset_id, version=version.version)
        asset = _store_file(
            session,
            key=key,
            path=archive,
            media_type="application/gzip",
            origin="export",
            logical_name=f"{dataset.slug}-v{version.version}-lerobot-v3.tar.gz",
            meta={
                "kind": "lerobot_v3_export",
                "dataset_id": version.dataset_id,
                "version": version.version,
                "repo_id": repo_id,
                "episode_count": result.episode_count,
                "total_frames": result.total_frames,
                "writer_package_version": result.package_version,
            },
        )
        version.export_asset_id = asset.id
        session.flush()
        return ExportBuild(
            asset_id=asset.id,
            archive_path=archive,
            repo_id=repo_id,
            expected=expected,
            episode_count=result.episode_count,
            total_frames=result.total_frames,
            workdir=workdir,
            package_version=result.package_version,
        )
    except Exception:
        cleanup(workdir)
        raise


def _freeze_and_verify(session: Session, version: DatasetVersion) -> FreezeResult:
    dataset = get_dataset(session, version.dataset_id)
    manifest, digest, _manifest_asset = _freeze_manifest(session, version, dataset)
    build = build_export(session, version)
    try:
        if build.schema_version == "lerobot-v3-export/2.0":
            from embodied_dataops.datasets.export_v2 import verify_build_v2

            roundtrip = verify_build_v2(build, workdir=build.workdir / "verify")
        else:
            roundtrip = lerobot_reader.verify_export(
                build.archive_path,
                repo_id=build.repo_id,
                expected=build.expected,
                workdir=build.workdir / "verify",
            )
    finally:
        cleanup(build.workdir)
    export_asset = session.get(Asset, build.asset_id)
    if export_asset is not None:
        export_asset.meta_json = {
            **(export_asset.meta_json or {}),
            "roundtrip": roundtrip.as_dict(),
        }
    return FreezeResult(
        manifest=manifest,
        manifest_sha256=digest,
        manifest_asset_id=version.manifest_asset_id or "",
        roundtrip=roundtrip,
        episode_count=build.episode_count,
        total_frames=build.total_frames,
    )


def create_version(
    session: Session,
    *,
    dataset_id: str,
    episode_ids: Sequence[str],
    description: str = "",
    actor: Principal | None = None,
    request_id: str | None = None,
) -> DatasetVersion:
    """Validate membership, freeze the manifest, export and verify synchronously."""
    dataset = session.get(Dataset, dataset_id, with_for_update=True)
    if dataset is None:
        raise _not_found("Dataset", dataset_id)

    ordered_ids = _normalise_episode_ids(episode_ids)
    if _contains_canonical_v2_episode(session, ordered_ids):
        # V2 branch: same transaction/state machine, profile-bound canonical v2
        # metadata, dataset-manifest/2.0 and the official Reader round-trip.
        # A mixed v1+v2 membership is rejected with 409
        # MIXED_PROFILE_EXPORT_UNSUPPORTED before any version row is created.
        from embodied_dataops.datasets.export_v2 import create_v2_version

        return create_v2_version(
            session,
            dataset_id=dataset.id,
            episode_ids=ordered_ids,
            description=description,
            actor=actor,
            request_id=request_id,
        )

    eligibility = [validate_episode_eligible(session, episode_id) for episode_id in ordered_ids]

    version = DatasetVersion(
        id=new_id(),
        dataset_id=dataset.id,
        version=_next_version_number(session, dataset.id),
        state=DatasetVersionState.BUILDING.value,
        description=(description or "").strip(),
        created_by=actor.key_id if actor is not None else None,
        created_at=utcnow(),
    )
    session.add(version)
    session.flush()
    _snapshot_episodes(session, version, eligibility)

    record_audit(
        session,
        actor=actor,
        action="dataset_version.create",
        entity_type="dataset_version",
        entity_id=version.id,
        after={
            "dataset_id": dataset.id,
            "version": version.version,
            "episode_ids": ordered_ids,
        },
        request_id=request_id,
    )

    try:
        frozen = _freeze_and_verify(session, version)
    except ApiError as exc:
        version.state = DatasetVersionState.FAILED.value
        version.error_code = exc.code.value
        version.error_detail = exc.message[:2000]
        session.flush()
        record_audit(
            session,
            actor=actor,
            action="dataset_version.failed",
            entity_type="dataset_version",
            entity_id=version.id,
            after={"error_code": exc.code.value},
            request_id=request_id,
        )
        return version

    _set_state(version, DatasetVersionState.READY)
    session.flush()
    record_audit(
        session,
        actor=actor,
        action="dataset_version.ready",
        entity_type="dataset_version",
        entity_id=version.id,
        after={
            "manifest_sha256": frozen.manifest_sha256,
            "episode_count": frozen.episode_count,
            "total_frames": frozen.total_frames,
            "roundtrip": frozen.roundtrip.as_dict(),
        },
        request_id=request_id,
    )
    return version


# ---------------------------------------------------------------------------
# membership mutation (AT-18)
# ---------------------------------------------------------------------------
def _assert_membership_mutable(version: DatasetVersion) -> None:
    state = DatasetVersionState(version.state)
    if state is DatasetVersionState.PUBLISHED:
        raise ApiError(
            ErrorCode.VERSION_IMMUTABLE,
            "Published dataset versions are immutable; create a new version instead",
            details={"version_id": version.id, "manifest_sha256": version.manifest_sha256},
        )
    if state is not DatasetVersionState.BUILDING:
        raise ApiError(
            ErrorCode.VERSION_NOT_READY,
            f"A {state.value} dataset version cannot change membership; only BUILDING versions can",
            details={"version_id": version.id, "state": state.value},
        )


def modify_version_membership(
    session: Session,
    *,
    version_id: str,
    add_episode_ids: Sequence[str] = (),
    remove_episode_ids: Sequence[str] = (),
    actor: Principal | None = None,
    request_id: str | None = None,
) -> DatasetVersion:
    """Add/remove episodes on a BUILDING version; PUBLISHED is refused (AT-18)."""
    version = session.get(DatasetVersion, version_id, with_for_update=True)
    if version is None:
        raise _not_found("Dataset version", version_id)
    _assert_membership_mutable(version)
    removed = set(remove_episode_ids)
    added = _normalise_episode_ids(add_episode_ids) if add_episode_ids else []
    snapshot_rows = list(
        session.scalars(
            select(DatasetVersionEpisode).where(DatasetVersionEpisode.version_id == version.id)
        )
    )
    current = {row.episode_id: row for row in snapshot_rows}
    for episode_id in removed:
        row = current.pop(episode_id, None)
        if row is not None:
            session.delete(row)
    for episode_id in added:
        if episode_id in current:
            continue
        entry = validate_episode_eligible(session, episode_id)
        current[episode_id] = DatasetVersionEpisode(
            version_id=version.id,
            episode_id=episode_id,
            canonical_digest=entry.canonical_digest,
            quality_report_id=entry.quality_report_id,
            annotation_set_id=entry.annotation_set_id,
            position=0,
        )
        session.add(current[episode_id])
    if not current:
        raise ApiError(ErrorCode.EMPTY_VERSION, "A dataset version cannot become empty")
    for position, episode_id in enumerate(sorted(current)):
        current[episode_id].position = position
    session.flush()
    try:
        _freeze_and_verify(session, version)
    except ApiError as exc:
        version.state = DatasetVersionState.FAILED.value
        version.error_code = exc.code.value
        version.error_detail = exc.message[:2000]
        session.flush()
        return version
    _set_state(version, DatasetVersionState.READY)
    session.flush()
    record_audit(
        session,
        actor=actor,
        action="dataset_version.membership_changed",
        entity_type="dataset_version",
        entity_id=version.id,
        after={"episode_ids": sorted(current)},
        request_id=request_id,
    )
    return version


# ---------------------------------------------------------------------------
# publish (AT-18)
# ---------------------------------------------------------------------------
def publish_version(
    session: Session,
    *,
    version_id: str,
    actor: Principal | None = None,
    request_id: str | None = None,
) -> DatasetVersion:
    """READY -> PUBLISHED with a second SHA-256 verification of stored objects."""
    version = session.get(DatasetVersion, version_id, with_for_update=True)
    if version is None:
        raise _not_found("Dataset version", version_id)
    state = DatasetVersionState(version.state)
    if state is DatasetVersionState.PUBLISHED:
        raise ApiError(
            ErrorCode.VERSION_IMMUTABLE,
            "Dataset version is already PUBLISHED and immutable",
            details={"version_id": version.id},
        )
    if state is not DatasetVersionState.READY:
        raise ApiError(
            ErrorCode.VERSION_NOT_READY,
            "Only a READY dataset version can be published",
            details={"version_id": version.id, "state": state.value},
        )

    manifest_asset = session.get(Asset, version.manifest_asset_id) if version.manifest_asset_id else None
    export_asset = session.get(Asset, version.export_asset_id) if version.export_asset_id else None
    if manifest_asset is None or export_asset is None:
        raise ApiError(
            ErrorCode.VERSION_NOT_READY,
            "Dataset version has no frozen manifest or export object",
            details={"version_id": version.id},
        )

    storage = get_storage()
    manifest_bytes_stored = storage.get_bytes(manifest_asset.object_key)
    manifest_hash = hashlib.sha256(manifest_bytes_stored).hexdigest()
    if (
        manifest_hash != manifest_asset.sha256
        or version.manifest_sha256 != manifest_hash
        or manifest_hash != _manifest_digest_of(manifest_bytes_stored)
    ):
        raise ApiError(
            ErrorCode.VERSION_NOT_READY,
            "Manifest SHA-256 re-verification failed; refusing to publish",
            details={
                "version_id": version.id,
                "expected": version.manifest_sha256,
                "observed": manifest_hash,
            },
        )
    export_hash = _sha256_stored_object(storage, export_asset)
    if export_hash != export_asset.sha256:
        raise ApiError(
            ErrorCode.VERSION_NOT_READY,
            "Export package SHA-256 re-verification failed; refusing to publish",
            details={"version_id": version.id, "expected": export_asset.sha256, "observed": export_hash},
        )
    # Layered publication (SPEC_V2.1 §4.2): the frozen manifest carries the profile's
    # publish level, so a FORMAT_VERIFIED revision cannot be published as a training
    # dataset even if a caller asks for it.
    from embodied_dataops.datasets.publish_policy import (  # noqa: PLC0415
        MANIFEST_PUBLISH_LEVEL_KEY,
        assert_publishable,
    )

    manifest_payload = _parse_stored_manifest(manifest_bytes_stored)
    publish_level = str(
        (manifest_payload.get("profile") or {}).get(MANIFEST_PUBLISH_LEVEL_KEY) or ""
    )
    if not publish_level:
        raise ApiError(
            ErrorCode.PUBLISH_LEVEL_CONFLICT,
            "manifest does not freeze a publish level; refusing to publish",
            details={
                "version_id": version.id,
                "manifest_schema_version": manifest_payload.get("schema_version"),
            },
        )
    assert_publishable(publish_level=publish_level, export_kind=version.export_kind)

    previous_state = version.state
    _set_state(version, DatasetVersionState.PUBLISHED)
    version.published_at = utcnow()
    _ensure_coverage_snapshot(session, version)
    session.flush()
    record_audit(
        session,
        actor=actor,
        action="dataset_version.publish",
        entity_type="dataset_version",
        entity_id=version.id,
        before={"state": previous_state, "manifest_sha256": version.manifest_sha256},
        after={
            "state": version.state,
            "manifest_sha256": version.manifest_sha256,
            "export_sha256": export_asset.sha256,
            "export_kind": version.export_kind,
            "publish_level": publish_level or None,
        },
        request_id=request_id,
    )
    return version


# ---------------------------------------------------------------------------
# read API
# ---------------------------------------------------------------------------
def get_version(session: Session, dataset_id: str, version: int) -> dict[str, Any]:
    dataset = session.get(Dataset, dataset_id)
    if dataset is None:
        raise _not_found("Dataset", dataset_id)
    row = session.scalars(
        select(DatasetVersion).where(
            DatasetVersion.dataset_id == dataset_id, DatasetVersion.version == int(version)
        )
    ).first()
    if row is None:
        raise _not_found("Dataset version", f"{dataset_id}/v{version}")

    manifest: dict[str, Any] | None = None
    manifest_url: str | None = None
    export_url: str | None = None
    export_sha: str | None = None
    export_roundtrip: dict[str, Any] | None = None
    storage = get_storage()
    if row.manifest_asset_id:
        asset = session.get(Asset, row.manifest_asset_id)
        if asset is not None:
            try:
                manifest = _parse_stored_manifest(storage.get_bytes(asset.object_key))
                manifest_url = storage.presign_get(asset.object_key)
            except ApiError:
                manifest = None
    if row.export_asset_id:
        asset = session.get(Asset, row.export_asset_id)
        if asset is not None:
            export_url = storage.presign_get(asset.object_key)
            export_sha = asset.sha256
            export_roundtrip = (asset.meta_json or {}).get("roundtrip")

    snapshots = list(
        session.scalars(
            select(DatasetVersionEpisode)
            .where(DatasetVersionEpisode.version_id == row.id)
            .order_by(DatasetVersionEpisode.position)
        )
    )
    episodes = [
        {
            "episode_id": snapshot.episode_id,
            "canonical_digest": snapshot.canonical_digest,
            "quality_report_id": snapshot.quality_report_id,
            "annotation_set_id": snapshot.annotation_set_id,
            "annotation_revision_id": snapshot.annotation_revision_id,
            "position": snapshot.position,
        }
        for snapshot in snapshots
    ]
    return {
        "id": row.id,
        "dataset_id": row.dataset_id,
        "dataset_slug": dataset.slug,
        "version": int(row.version),
        "state": row.state,
        "description": row.description,
        "export_kind": row.export_kind,
        "publish_level": ((manifest or {}).get("profile") or {}).get("publish_level"),
        "training_ready": str(
            ((manifest or {}).get("profile") or {}).get("publish_level") or ""
        )
        == "SEMANTIC_VERIFIED",
        "manifest": manifest,
        "manifest_sha256": row.manifest_sha256,
        "manifest_url": manifest_url,
        "export_url": export_url,
        "export_sha256": export_sha,
        "export_roundtrip": export_roundtrip,
        "episodes": episodes,
        "created_at": row.created_at.isoformat() if row.created_at else None,
        "published_at": row.published_at.isoformat() if row.published_at else None,
        "error_code": row.error_code,
        "error_detail": row.error_detail,
    }


def list_versions(session: Session, dataset_id: str) -> list[DatasetVersion]:
    get_dataset(session, dataset_id)
    statement = (
        select(DatasetVersion)
        .where(DatasetVersion.dataset_id == dataset_id)
        .order_by(DatasetVersion.version.desc())
    )
    return list(session.scalars(statement))
