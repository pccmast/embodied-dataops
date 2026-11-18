"""``EGO_ANNOTATION_BUNDLE`` export (SPEC_V2.1 §14, V21-15).

An ego video produces **action intervals**, not a robot trajectory.  Publishing it as a
LeRobot dataset would be a factual lie, so it gets its own export kind with a fixed
four-file layout:

``manifest.json``     dataset identity, counts, licence pointers
``clips.jsonl``       one line per episode: duration, source hash, access policy
``annotation.jsonl``  one line per reviewed action interval
``provenance.json``   sources, model/prompt versions, egress decisions

The rule that matters most: **video bytes are only included when redistribution is
permitted**.  Otherwise the bundle carries a content hash and a controlled-access
pointer, which is enough to reproduce the experiment but not enough to leak the clip.

This module builds the payloads as pure functions so the layout can be asserted without
a database; :func:`assemble_bundle` adds the tar.gz wrapper.
"""

from __future__ import annotations

import hashlib
import io
import json
import tarfile
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from typing import Any

BUNDLE_SCHEMA = "ego-annotation-bundle/2.1"
BUNDLE_FILES = ("manifest.json", "clips.jsonl", "annotation.jsonl", "provenance.json")
EGO_EPISODE_TYPE = "ego_video_episode"
ROBOT_EPISODE_TYPE = "robot_episode"


class BundleError(ValueError):
    """The requested episodes cannot form a legal ego bundle."""


@dataclass
class BundleInput:
    """Everything the bundle needs, already resolved by the caller."""

    dataset_slug: str
    version: int
    episodes: Sequence[dict[str, Any]]
    actions: Sequence[dict[str, Any]]
    provenance: dict[str, Any]
    #: True when the licence allows shipping the mp4 inside the archive.
    video_redistributable: bool = False


@dataclass
class BundleResult:
    files: dict[str, bytes]
    manifest: dict[str, Any] = field(default_factory=dict)
    archive: bytes = b""

    @property
    def archive_sha256(self) -> str:
        return hashlib.sha256(self.archive).hexdigest()

    @property
    def action_count(self) -> int:
        return int(self.manifest.get("counts", {}).get("actions", 0))


def _jsonl(rows: Iterable[dict[str, Any]]) -> bytes:
    return ("".join(json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n" for row in rows)).encode(
        "utf-8"
    )


def _json(payload: dict[str, Any]) -> bytes:
    return json.dumps(payload, sort_keys=True, ensure_ascii=False, indent=2).encode("utf-8")


def build_bundle_payloads(bundle: BundleInput) -> BundleResult:
    """Assemble the four files.  Raises :class:`BundleError` on an illegal mix."""
    if not bundle.dataset_slug:
        raise BundleError("dataset_slug is required")
    if int(bundle.version) < 1:
        raise BundleError("dataset version must be >= 1")
    if not bundle.episodes:
        raise BundleError("an ego bundle must contain at least one episode")

    episode_ids: set[str] = set()
    clips: list[dict[str, Any]] = []
    for episode in bundle.episodes:
        episode_id = str(episode.get("episode_id") or "")
        if not episode_id:
            raise BundleError("every episode needs an episode_id")
        if episode_id in episode_ids:
            raise BundleError(f"duplicate episode_id {episode_id!r}")
        episode_ids.add(episode_id)
        episode_type = str(episode.get("episode_type") or "")
        if episode_type == ROBOT_EPISODE_TYPE:
            # The whole point of a separate export kind: no robot trajectory in here.
            raise BundleError(
                f"episode {episode_id} is a robot_episode; a robot episode cannot be exported as "
                "an ego annotation bundle"
            )
        if episode_type and episode_type != EGO_EPISODE_TYPE:
            raise BundleError(f"unsupported episode_type {episode_type!r} for an ego bundle")
        clip = {
            "episode_id": episode_id,
            "duration_ns": int(episode.get("duration_ns") or 0),
            "source_asset_sha256": str(episode.get("source_asset_sha256") or ""),
            "data_origin": str(episode.get("data_origin") or ""),
            "license_id": str(episode.get("license_id") or ""),
            # Never a URL: a bundle that leaks an internal presigned link is a leak.
            "video_included": bool(bundle.video_redistributable),
            "controlled_access_pointer": (
                None
                if bundle.video_redistributable
                else f"sha256:{episode.get('source_asset_sha256')}"
            ),
        }
        clips.append(clip)

    annotations: list[dict[str, Any]] = []
    for action in bundle.actions:
        episode_id = str(action.get("episode_id") or "")
        if episode_id not in episode_ids:
            raise BundleError(f"action refers to an episode outside this version: {episode_id!r}")
        start = int(action.get("start_ns") or 0)
        end = int(action.get("end_ns") or 0)
        if end <= start:
            raise BundleError(f"action interval for {episode_id} is not positive: {start}..{end}")
        annotations.append(
            {
                "episode_id": episode_id,
                "verb": str(action.get("verb") or ""),
                "noun": str(action.get("noun") or ""),
                "start_ns": start,
                "end_ns": end,
                "evidence_frame_indices": [int(v) for v in action.get("evidence_frame_indices") or []],
                "origin": str(action.get("origin") or "MODEL"),
                "review_state": str(action.get("review_state") or "PENDING"),
                "annotation_revision_id": action.get("annotation_revision_id"),
            }
        )

    manifest = {
        "schema_version": BUNDLE_SCHEMA,
        "export_kind": "EGO_ANNOTATION_BUNDLE",
        "dataset_slug": bundle.dataset_slug,
        "version": int(bundle.version),
        "counts": {"episodes": len(clips), "actions": len(annotations)},
        "files": list(BUNDLE_FILES),
        "video_included": bool(bundle.video_redistributable),
        "robot_actions_present": False,
        "licence_pointers": [
            {"episode_id": clip["episode_id"], "license_id": clip["license_id"]} for clip in clips
        ],
        "notes": (
            "Ego action intervals. This bundle is not a robot control trajectory and must not be "
            "converted into LeRobot action arrays."
        ),
    }
    provenance = {
        "schema_version": BUNDLE_SCHEMA,
        "export_kind": "EGO_ANNOTATION_BUNDLE",
        **{str(key): value for key, value in bundle.provenance.items()},
    }

    files = {
        "manifest.json": _json(manifest),
        "clips.jsonl": _jsonl(clips),
        "annotation.jsonl": _jsonl(annotations),
        "provenance.json": _json(provenance),
    }
    return BundleResult(files=files, manifest=manifest)


def assemble_bundle(bundle: BundleInput, *, video_bytes: bytes | None = None) -> BundleResult:
    """Build the four files and pack them into a deterministic tar.gz."""
    result = build_bundle_payloads(bundle)
    buffer = io.BytesIO()
    # ``mtime=0`` and a fixed order make the archive byte-identical for the same input.
    with tarfile.open(fileobj=buffer, mode="w:gz") as archive:
        for name in BUNDLE_FILES:
            data = result.files[name]
            info = tarfile.TarInfo(name=name)
            info.size = len(data)
            info.mtime = 0
            info.mode = 0o644
            archive.addfile(info, io.BytesIO(data))
        if bundle.video_redistributable and video_bytes:
            info = tarfile.TarInfo(name="video.mp4")
            info.size = len(video_bytes)
            info.mtime = 0
            info.mode = 0o644
            archive.addfile(info, io.BytesIO(video_bytes))
    result.archive = buffer.getvalue()
    return result


__all__ = [
    "BUNDLE_FILES",
    "BUNDLE_SCHEMA",
    "BundleError",
    "BundleInput",
    "BundleResult",
    "assemble_bundle",
    "build_bundle_payloads",
]


# ---------------------------------------------------------------------------
# service wiring
# ---------------------------------------------------------------------------


def create_ego_bundle_version(
    session,
    *,
    dataset_id: str,
    episode_ids: Sequence[str],
    storage,
    description: str = "",
    licence_allows_video: bool = False,
    actor=None,
) -> dict[str, Any]:
    """Create a ``READY`` ``EGO_ANNOTATION_BUNDLE`` version and store its archive.

    This is the only place an ego episode becomes a published artifact, and it produces a
    bundle -- never a LeRobot dataset.  Robot versions keep going through
    ``datasets.service.create_version``, which rejects ``ego_video_episode`` outright.
    """
    from embodied_dataops.db.models import (
        AnnotationRevision,
        AnnotationSet,
        Asset,
        Dataset,
        DatasetVersion,
        DatasetVersionEpisode,
        Episode,
        SourceImport,
    )
    from embodied_dataops.domain.enums import AssetOrigin, DatasetVersionState
    from embodied_dataops.ids import new_id

    dataset = session.get(Dataset, dataset_id, with_for_update=True)
    if dataset is None:
        raise BundleError(f"dataset {dataset_id!r} does not exist")
    ordered = list(dict.fromkeys(episode_ids))
    if not ordered:
        raise BundleError("episode_ids must not be empty")

    episodes: list[dict[str, Any]] = []
    actions: list[dict[str, Any]] = []
    provenance_sources: list[dict[str, Any]] = []

    for episode_id in ordered:
        episode = session.get(Episode, episode_id)
        if episode is None:
            raise BundleError(f"episode {episode_id!r} does not exist")
        if episode.episode_type == ROBOT_EPISODE_TYPE:
            raise BundleError(
                f"episode {episode_id} is a robot_episode; it belongs in a ROBOT_LEROBOT_V3 version"
            )
        source = session.get(SourceImport, episode.source_import_id)
        if source is None:
            raise BundleError(f"episode {episode_id} has no SourceImport")
        if episode.quality_status not in (None, "PASS", "WARN"):
            raise BundleError(
                f"episode {episode_id} failed QC ({episode.quality_status}); it cannot be published"
            )
        annotation_set = session.execute(
            __import__("sqlalchemy").select(AnnotationSet).where(AnnotationSet.episode_id == episode.id)
        ).scalars().first()
        approved_revision_id = getattr(annotation_set, "approved_revision_id", None)
        if annotation_set is None or not approved_revision_id:
            raise BundleError(
                f"episode {episode_id} has no approved annotation revision; an ego bundle is a "
                "reviewed deliverable, not a raw model dump"
            )
        revision = session.get(AnnotationRevision, approved_revision_id)
        document = {}
        if revision is not None and revision.artifact_asset_id:
            artifact = session.get(Asset, revision.artifact_asset_id)
            if artifact is not None:
                import json as _json

                document = _json.loads(storage.get_bytes(artifact.object_key))
        for item in document.get("actions") or []:
            actions.append(
                {
                    "episode_id": episode.id,
                    "verb": item.get("verb"),
                    "noun": item.get("noun"),
                    "start_ns": item.get("start_ns"),
                    "end_ns": item.get("end_ns"),
                    "evidence_frame_indices": item.get("evidence_frame_indices") or [],
                    "origin": item.get("origin") or "MODEL",
                    "review_state": "APPROVED",
                    "annotation_revision_id": approved_revision_id,
                }
            )
        episodes.append(
            {
                "episode_id": episode.id,
                "episode_type": episode.episode_type,
                "duration_ns": int(episode.duration_ns or 0),
                "source_asset_sha256": str(episode.canonical_sha256 or ""),
                "data_origin": episode.data_origin,
                "license_id": str(source.license or ""),
            }
        )
        provenance_sources.append(
            {
                "episode_id": episode.id,
                "import_id": source.id,
                "source_type": source.source_type,
                "data_origin": source.data_origin,
                "license": source.license,
                "source_revision": source.source_revision,
                "egress_allowed": bool(source.egress_allowed),
            }
        )

    bundle = BundleInput(
        dataset_slug=dataset.slug,
        version=_next_ego_version(session, DatasetVersion, dataset.id),
        episodes=episodes,
        actions=actions,
        provenance={"sources": provenance_sources, "pipeline_version": "2.1.0"},
        video_redistributable=bool(licence_allows_video),
    )
    result = assemble_bundle(bundle)

    version = DatasetVersion(
        id=new_id(),
        dataset_id=dataset.id,
        version=bundle.version,
        state=DatasetVersionState.READY.value,
        export_kind="EGO_ANNOTATION_BUNDLE",
        description=description.strip(),
        created_by=getattr(actor, "key_id", None),
        manifest_sha256=hashlib.sha256(result.files["manifest.json"]).hexdigest(),
    )
    session.add(version)
    session.flush()

    key = f"datasets/{dataset.id}/v{version.version}/export/ego-annotation-bundle.tar.gz"
    stored = storage.put_bytes(key, result.archive, content_type="application/gzip")
    asset = Asset(
        id=new_id(),
        sha256=stored.sha256 or result.archive_sha256,
        byte_size=len(result.archive),
        media_type="application/gzip",
        object_key=key,
        origin=AssetOrigin.EXPORT.value,
        logical_name=f"{dataset.slug}-v{version.version}-ego-bundle.tar.gz",
        meta_json={
            "kind": "ego_annotation_bundle",
            "schema_version": BUNDLE_SCHEMA,
            "episodes": len(episodes),
            "actions": len(actions),
            "video_included": bool(licence_allows_video),
        },
    )
    session.add(asset)
    session.flush()
    version.export_asset_id = asset.id

    for position, episode_id in enumerate(ordered):
        session.add(
            DatasetVersionEpisode(
                version_id=version.id,
                episode_id=episode_id,
                canonical_digest=str(episodes[position]["source_asset_sha256"]),
                quality_report_id="",
                position=position,
            )
        )
    session.flush()

    return {
        "dataset_id": dataset.id,
        "version": version.version,
        "version_id": version.id,
        "export_kind": "EGO_ANNOTATION_BUNDLE",
        "state": version.state,
        "episodes": len(episodes),
        "actions": len(actions),
        "archive_sha256": asset.sha256,
        "archive_bytes": asset.byte_size,
        "manifest_sha256": version.manifest_sha256,
    }


def _next_ego_version(session, DatasetVersion, dataset_id: str) -> int:
    from sqlalchemy import func, select

    current = session.scalar(
        select(func.coalesce(func.max(DatasetVersion.version), 0)).where(
            DatasetVersion.dataset_id == dataset_id
        )
    )
    return int(current or 0) + 1
