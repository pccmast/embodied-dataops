"""Scenario tags and evidence-backed dataset-version coverage (V2 D2-6, SPEC §7).

Coverage is defined over **Episodes, never frames**, and only over
``DatasetVersion`` rows in the ``PUBLISHED`` state.  ``UNKNOWN`` tags are stored
for audit but never contribute to coverage; a per-version read-only snapshot is
computed when the version is published and every later lookup reads that frozen
snapshot (subsequent collection must go into a new version).

Hook for the publish path::

    from embodied_dataops.feedback.coverage import ensure_coverage_snapshot
    ...
    ensure_coverage_snapshot(session, version)   # after state=PUBLISHED, before flush

The function is idempotent, so it is safe to call from both publish and a lazy
read fallback.
"""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from datetime import UTC
from typing import Any

from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from embodied_dataops.api.auth import Principal
from embodied_dataops.canonical.contract import canonical_json_bytes
from embodied_dataops.db.models import (
    DatasetVersion,
    DatasetVersionEpisode,
    Episode,
    ScenarioCoverageSnapshot,
    ScenarioTag,
)
from embodied_dataops.errors import ApiError, ErrorCode
from embodied_dataops.feedback.evaluation import validate_scenario_key
from embodied_dataops.ids import new_id
from embodied_dataops.observability.audit import record_audit
from embodied_dataops.workflow.outbox import utcnow

#: Allowed provenance of a scenario label (SPEC §7.1).
SCENARIO_TAG_BASES: tuple[str, ...] = ("SOURCE_DECLARED", "HUMAN_VERIFIED", "UNKNOWN")

#: Only these bases count as evidence; stronger basis wins per episode/scenario.
CREDIBLE_BASE_PRIORITY: tuple[str, ...] = ("HUMAN_VERIFIED", "SOURCE_DECLARED")

#: Returned when a (version, scenario_key) pair has no credible tag.
UNKNOWN_BASIS = "UNKNOWN"


# ---------------------------------------------------------------------------
# scenario tags
# ---------------------------------------------------------------------------
def _serialize_tag(tag: ScenarioTag) -> dict[str, Any]:
    return {
        "id": tag.id,
        "episode_id": tag.episode_id,
        "scenario_key": tag.scenario_key,
        "basis": tag.basis,
        "source_evidence_asset_id": tag.source_evidence_asset_id,
        "verified_by": tag.verified_by,
        "created_at": tag.created_at.isoformat() if tag.created_at else None,
    }


def serialize_tag(tag: ScenarioTag) -> dict[str, Any]:
    """Public serializer used by the HTTP route."""
    return _serialize_tag(tag)


def list_scenario_tags(session: Session, episode_id: str) -> list[dict[str, Any]]:
    if session.get(Episode, episode_id) is None:
        raise ApiError(ErrorCode.NOT_FOUND, f"Episode not found: {episode_id}")
    statement = (
        select(ScenarioTag)
        .where(ScenarioTag.episode_id == episode_id)
        .order_by(ScenarioTag.scenario_key, ScenarioTag.basis)
    )
    return [_serialize_tag(tag) for tag in session.scalars(statement)]


def add_scenario_tag(
    session: Session,
    *,
    episode_id: str,
    scenario_key: str,
    basis: str,
    actor: Principal | None = None,
    note: str = "",
    source_evidence_asset_id: str | None = None,
    request_id: str | None = None,
) -> ScenarioTag:
    """Attach one evidence label to an Episode (idempotent per episode/key/basis)."""
    if session.get(Episode, episode_id) is None:
        raise ApiError(ErrorCode.NOT_FOUND, f"Episode not found: {episode_id}")
    key = validate_scenario_key(scenario_key)
    if basis not in SCENARIO_TAG_BASES:
        raise ApiError(
            ErrorCode.VALIDATION_FAILED,
            "basis must be one of SOURCE_DECLARED, HUMAN_VERIFIED, UNKNOWN",
            details={"basis": basis, "allowed": list(SCENARIO_TAG_BASES)},
        )
    if basis == "HUMAN_VERIFIED" and actor is None:
        raise ApiError(
            ErrorCode.FORBIDDEN,
            "HUMAN_VERIFIED scenario tags require an authenticated reviewer",
        )
    existing = session.scalars(
        select(ScenarioTag).where(
            ScenarioTag.episode_id == episode_id,
            ScenarioTag.scenario_key == key,
            ScenarioTag.basis == basis,
        )
    ).first()
    if existing is not None:
        return existing

    tag = ScenarioTag(
        id=new_id(),
        episode_id=episode_id,
        scenario_key=key,
        basis=basis,
        source_evidence_asset_id=source_evidence_asset_id,
        verified_by=actor.key_id if basis == "HUMAN_VERIFIED" and actor is not None else None,
        created_at=utcnow(),
    )
    session.add(tag)
    session.flush()
    if basis != "UNKNOWN":
        record_audit(
            session,
            actor=actor,
            action="scenario_tag.add",
            entity_type="scenario_tag",
            entity_id=tag.id,
            after={
                "episode_id": episode_id,
                "scenario_key": key,
                "basis": basis,
                "note": note[:500],
            },
            request_id=request_id,
        )
    return tag


def credible_scenario_basis(
    session: Session, episode_id: str, scenario_key: str
) -> str | None:
    """Strongest credible basis for one episode/scenario pair, or ``None``."""
    basis_values = set(
        session.scalars(
            select(ScenarioTag.basis).where(
                ScenarioTag.episode_id == episode_id,
                ScenarioTag.scenario_key == scenario_key,
                ScenarioTag.basis.in_(CREDIBLE_BASE_PRIORITY),
            )
        )
    )
    for basis in CREDIBLE_BASE_PRIORITY:
        if basis in basis_values:
            return basis
    return None


def _best_basis(bases: Sequence[str]) -> str | None:
    for basis in CREDIBLE_BASE_PRIORITY:
        if basis in bases:
            return basis
    return None


# ---------------------------------------------------------------------------
# coverage snapshots
# ---------------------------------------------------------------------------
def _snapshot_digest(version: DatasetVersion, tags: list[tuple[str, str, str]]) -> str:
    payload = {
        "dataset_version_id": version.id,
        "manifest_sha256": version.manifest_sha256 or "",
        "tags": sorted([list(item) for item in tags]),
    }
    return hashlib.sha256(canonical_json_bytes(payload)).hexdigest()


def _existing_snapshots(
    session: Session, version_id: str
) -> list[ScenarioCoverageSnapshot]:
    statement = (
        select(ScenarioCoverageSnapshot)
        .where(ScenarioCoverageSnapshot.dataset_version_id == version_id)
        .order_by(ScenarioCoverageSnapshot.scenario_key, ScenarioCoverageSnapshot.tagging_basis)
    )
    return list(session.scalars(statement))


def ensure_coverage_snapshot(
    session: Session, version: DatasetVersion
) -> list[ScenarioCoverageSnapshot]:
    """Compute the frozen per-PUBLISHED-version coverage snapshot (idempotent).

    Called from the publish path; the lazy read fallback in :func:`get_coverage`
    uses the same function so a missing hook cannot produce a wrong number.
    """
    existing = _existing_snapshots(session, version.id)
    if existing:
        return existing

    episode_ids = list(
        session.scalars(
            select(DatasetVersionEpisode.episode_id).where(
                DatasetVersionEpisode.version_id == version.id
            )
        )
    )
    if not episode_ids:
        return []

    tag_rows = list(
        session.scalars(
            select(ScenarioTag).where(
                ScenarioTag.episode_id.in_(episode_ids),
                ScenarioTag.basis.in_(CREDIBLE_BASE_PRIORITY),
            )
        )
    )
    bases_by_pair: dict[tuple[str, str], list[str]] = {}
    for tag in tag_rows:
        bases_by_pair.setdefault((tag.episode_id, tag.scenario_key), []).append(tag.basis)

    counts: dict[tuple[str, str], int] = {}
    digest_tags: list[tuple[str, str, str]] = []
    for (episode_id, scenario_key), bases in bases_by_pair.items():
        basis = _best_basis(bases)
        if basis is None:
            continue
        counts[(scenario_key, basis)] = counts.get((scenario_key, basis), 0) + 1
        digest_tags.append((episode_id, scenario_key, basis))

    digest = _snapshot_digest(version, digest_tags)
    computed_at = utcnow()
    created: list[ScenarioCoverageSnapshot] = []
    for (scenario_key, basis), episode_count in sorted(counts.items()):
        created.append(
            ScenarioCoverageSnapshot(
                id=new_id(),
                dataset_version_id=version.id,
                scenario_key=scenario_key,
                episode_count=int(episode_count),
                tagging_basis=basis,
                source_digest=digest,
                computed_at=computed_at,
            )
        )
    if not created:
        return []
    try:
        with session.begin_nested():
            session.add_all(created)
            session.flush()
    except IntegrityError:
        # Concurrent publish/lookup computed the same snapshot first.
        return _existing_snapshots(session, version.id)
    return created


def bump_episode_revisions(
    session: Session, episode_ids: Sequence[str]
) -> dict[str, int]:
    """Bump ``episodes.business_revision`` for affected episodes (DO-A02 C2).

    Uses the shared ``operations.revisions.bump`` helper when available and falls
    back to an inline increment so a missing optional module cannot silently skip
    the revision bump.  One call site: :func:`record_version_publication`.
    """
    revisions: dict[str, int] = {}
    try:
        from embodied_dataops.operations.revisions import bump  # noqa: PLC0415
    except Exception:  # pragma: no cover - the helper is part of this repository
        bump = None  # type: ignore[assignment]
    for episode_id in sorted(set(episode_ids)):
        episode = session.get(Episode, episode_id)
        if episode is None:
            continue
        if bump is not None:
            revisions[episode_id] = int(bump(session, episode))
        else:  # pragma: no cover - fallback branch
            episode.business_revision = int(episode.business_revision or 1) + 1
            revisions[episode_id] = int(episode.business_revision)
    session.flush()
    return revisions


def record_version_publication(
    session: Session, version: DatasetVersion
) -> dict[str, Any]:
    """Publish-time side effects: frozen coverage snapshot + episode revision bumps.

    Call once from ``publish_version`` after the version is set to ``PUBLISHED``
    and its ``published_at`` is recorded, before the transaction commits::

        from embodied_dataops.feedback.coverage import record_version_publication
        ...
        record_version_publication(session, version)
    """
    snapshots = ensure_coverage_snapshot(session, version)
    episode_ids = list(
        session.scalars(
            select(DatasetVersionEpisode.episode_id).where(
                DatasetVersionEpisode.version_id == version.id
            )
        )
    )
    revisions = bump_episode_revisions(session, episode_ids)
    return {
        "dataset_version_id": version.id,
        "snapshot_rows": len(snapshots),
        "episode_revisions": revisions,
    }


def get_coverage(
    session: Session, *, dataset_version_id: str, scenario_key: str
) -> dict[str, Any]:
    """Episode-count coverage for one published version + scenario key."""
    version = session.get(DatasetVersion, dataset_version_id)
    if version is None:
        raise ApiError(
            ErrorCode.NOT_FOUND, f"Dataset version not found: {dataset_version_id}"
        )
    if version.state != "PUBLISHED":
        raise ApiError(
            ErrorCode.CONFLICT,
            "Coverage is only defined for PUBLISHED dataset versions",
            details={"dataset_version_id": version.id, "state": version.state},
        )
    key = validate_scenario_key(scenario_key)
    snapshots = [
        row
        for row in ensure_coverage_snapshot(session, version)
        if row.scenario_key == key
    ]
    by_basis = {row.tagging_basis: int(row.episode_count) for row in snapshots}
    human = by_basis.get("HUMAN_VERIFIED", 0)
    declared = by_basis.get("SOURCE_DECLARED", 0)
    episode_count = human + declared
    if human > 0:
        tagging_basis = "HUMAN_VERIFIED"
    elif declared > 0:
        tagging_basis = "SOURCE_DECLARED"
    else:
        tagging_basis = UNKNOWN_BASIS
    computed_at = max((row.computed_at for row in snapshots), default=None)
    source_digest = snapshots[0].source_digest if snapshots else None
    return {
        "dataset_version_id": version.id,
        "dataset_id": version.dataset_id,
        "version": int(version.version),
        "scenario_key": key,
        "episode_count": int(episode_count),
        "tagging_basis": tagging_basis,
        "breakdown": {
            "HUMAN_VERIFIED": human,
            "SOURCE_DECLARED": declared,
            "UNKNOWN": 0,
        },
        "denominator": "episode_count",
        # A frozen snapshot must read back byte-identically.  ``computed_at`` is
        # timestamptz, and the driver renders it in the *session* time zone, so on a
        # non-UTC host the same instant serialised as ``+08:00`` instead of ``+00:00``
        # and the "frozen" comparison drifted for no data reason.  Normalise to UTC.
        "computed_at": _utc_isoformat(computed_at),
        "source_digest": source_digest,
        "snapshot_frozen": bool(snapshots),
    }


def _utc_isoformat(value: Any) -> str | None:
    """Serialise a timestamp in UTC regardless of the connection's time zone."""
    if value is None:
        return None
    if getattr(value, "tzinfo", None) is None:
        return value.isoformat()
    return value.astimezone(UTC).isoformat()
