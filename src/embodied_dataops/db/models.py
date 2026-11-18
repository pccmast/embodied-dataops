"""Database models.

Field contracts follow SPEC 4.3 literally.  Additive (non-destructive) columns
and tables are allowed; required columns and unique constraints are never
removed.  Every primary key is a UUIDv7 generated in Python (SPEC 4.2), stored
as a canonical 36-character string so the schema remains portable.
"""

from __future__ import annotations

from datetime import datetime
from decimal import Decimal

from sqlalchemy import (
    BigInteger,
    Boolean,
    CheckConstraint,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    Numeric,
    String,
    Text,
    UniqueConstraint,
    func,
    text,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column, relationship

from embodied_dataops.db.base import Base

ID = String(36)
SHA = String(64)
TS = DateTime(timezone=True)


def _created_at() -> Mapped[datetime]:
    return mapped_column(TS, nullable=False, server_default=func.now())


class RobotProfile(Base):
    """Versioned robot profile registry (V2 SPEC 2.1).

    A profile is an immutable, versioned description of joint names/dimensions,
    units/semantics, limits and camera requirements.  ``authority`` states how the
    semantics were established; only ``SYNTHETIC_FIXTURE`` (built-in demo
    fixtures) and ``SOURCE_DOCUMENTED`` may pass semantic QC.
    """

    __tablename__ = "robot_profiles"

    id: Mapped[str] = mapped_column(ID, primary_key=True)
    profile_id: Mapped[str] = mapped_column(String(80), nullable=False)
    profile_revision: Mapped[int] = mapped_column(Integer, nullable=False)
    authority: Mapped[str] = mapped_column(String(32), nullable=False)
    source_evidence_asset_id: Mapped[str | None] = mapped_column(
        ID, ForeignKey("assets.id"), nullable=True
    )
    config_json: Mapped[dict] = mapped_column(JSONB, nullable=False)
    config_sha256: Mapped[str] = mapped_column(SHA, nullable=False)
    created_at: Mapped[datetime] = mapped_column(TS, nullable=False, server_default=func.now())

    __table_args__ = (
        UniqueConstraint("profile_id", "profile_revision", name="profile_revision_unique"),
        CheckConstraint(
            "authority IN ('SYNTHETIC_FIXTURE','SOURCE_DOCUMENTED',"
            "'SOURCE_DOCUMENTED_PARTIAL','UNVERIFIED')",
            name="authority_valid",
        ),
        CheckConstraint("profile_revision >= 1", name="revision_positive"),
    )


class ApiKey(Base):
    __tablename__ = "api_keys"

    id: Mapped[str] = mapped_column(ID, primary_key=True)
    name: Mapped[str] = mapped_column(String(120), nullable=False, default="")
    key_hash: Mapped[str] = mapped_column(SHA, nullable=False, unique=True)
    role: Mapped[str] = mapped_column(String(20), nullable=False)
    enabled: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)
    created_at: Mapped[datetime] = mapped_column(TS, nullable=False, server_default=func.now())
    revoked_at: Mapped[datetime | None] = mapped_column(TS, nullable=True)

    __table_args__ = (
        CheckConstraint("role IN ('viewer','operator','reviewer','admin')", name="role_valid"),
    )


class Asset(Base):
    __tablename__ = "assets"

    id: Mapped[str] = mapped_column(ID, primary_key=True)
    sha256: Mapped[str] = mapped_column(SHA, nullable=False)
    byte_size: Mapped[int] = mapped_column(BigInteger, nullable=False)
    media_type: Mapped[str] = mapped_column(String(120), nullable=False)
    object_key: Mapped[str] = mapped_column(String(1024), nullable=False, unique=True)
    origin: Mapped[str] = mapped_column(String(20), nullable=False)
    logical_name: Mapped[str] = mapped_column(String(200), nullable=False, default="")
    meta_json: Mapped[dict] = mapped_column(JSONB, nullable=False, default=dict)
    created_at: Mapped[datetime] = mapped_column(TS, nullable=False, server_default=func.now())

    __table_args__ = (
        Index("ix_assets_sha256_byte_size", "sha256", "byte_size"),
        CheckConstraint(
            "origin IN ('raw','staging','canonical','derived','export','report')", name="origin_valid"
        ),
    )


class SourceImport(Base):
    __tablename__ = "source_imports"

    id: Mapped[str] = mapped_column(ID, primary_key=True)
    source_type: Mapped[str] = mapped_column(String(40), nullable=False)
    upload_state: Mapped[str] = mapped_column(String(20), nullable=False)
    asset_id: Mapped[str | None] = mapped_column(ID, ForeignKey("assets.id"), nullable=True)
    data_origin: Mapped[str] = mapped_column(String(30), nullable=False)
    license: Mapped[str] = mapped_column(String(200), nullable=False, default="")
    source_uri: Mapped[str | None] = mapped_column(String(1024), nullable=True)
    source_revision: Mapped[str | None] = mapped_column(String(200), nullable=True)
    filename: Mapped[str] = mapped_column(String(500), nullable=False, default="")
    size_bytes: Mapped[int] = mapped_column(BigInteger, nullable=False, default=0)
    sha256: Mapped[str | None] = mapped_column(SHA, nullable=True)
    mapping_json: Mapped[dict] = mapped_column(JSONB, nullable=False, default=dict)
    #: V2.1 §8: whether the bytes may leave the deployment for a remote model call.
    #: Defaults to false; nothing in the AI pipeline can flip it.
    egress_allowed: Mapped[bool] = mapped_column(
        Boolean, nullable=False, default=False, server_default="false"
    )
    staging_key: Mapped[str | None] = mapped_column(String(1024), nullable=True)
    created_by: Mapped[str | None] = mapped_column(ID, ForeignKey("api_keys.id"), nullable=True)
    created_at: Mapped[datetime] = mapped_column(TS, nullable=False, server_default=func.now())
    completed_at: Mapped[datetime | None] = mapped_column(TS, nullable=True)
    error_code: Mapped[str | None] = mapped_column(String(60), nullable=True)
    error_detail: Mapped[str | None] = mapped_column(Text, nullable=True)
    # V2 D2-6: optional collection-order provenance (only APPROVED/IN_PROGRESS orders)
    collection_order_id: Mapped[str | None] = mapped_column(
        ID, ForeignKey("collection_orders.id", use_alter=True, name="fk_source_imports_collection_order_id"), nullable=True
    )

    asset: Mapped[Asset | None] = relationship(lazy="joined")

    __table_args__ = (
        CheckConstraint(
            "source_type IN ('LEROBOT_V3','EGO_MP4','MCAP_ROS2_PROFILE_V1','MCAP_ROS2_PROFILE_V2')", name="source_type_valid"
        ),
        CheckConstraint(
            "upload_state IN ('INITIATED','UPLOADED','VALIDATING','READY','FAILED')",
            name="upload_state_valid",
        ),
        CheckConstraint(
            "data_origin IN ('synthetic','public_dataset','robot_capture','ego_capture')",
            name="data_origin_valid",
        ),
    )


class ImportJob(Base):
    __tablename__ = "import_jobs"

    id: Mapped[str] = mapped_column(ID, primary_key=True)
    import_id: Mapped[str] = mapped_column(ID, ForeignKey("source_imports.id"), nullable=False)
    state: Mapped[str] = mapped_column(String(20), nullable=False)
    error_code: Mapped[str | None] = mapped_column(String(60), nullable=True)
    error_detail: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(TS, nullable=False, server_default=func.now())
    started_at: Mapped[datetime | None] = mapped_column(TS, nullable=True)
    finished_at: Mapped[datetime | None] = mapped_column(TS, nullable=True)

    __table_args__ = (
        CheckConstraint("state IN ('QUEUED','RUNNING','SUCCEEDED','FAILED')", name="state_valid"),
        Index("ix_import_jobs_import_id", "import_id"),
    )


class Episode(Base):
    __tablename__ = "episodes"

    id: Mapped[str] = mapped_column(ID, primary_key=True)
    source_import_id: Mapped[str] = mapped_column(ID, ForeignKey("source_imports.id"), nullable=False)
    source_episode_index: Mapped[int] = mapped_column(Integer, nullable=False)
    episode_type: Mapped[str] = mapped_column(String(30), nullable=False)
    task_label: Mapped[str] = mapped_column(String(500), nullable=False, default="")
    duration_ns: Mapped[int] = mapped_column(BigInteger, nullable=False, default=0)
    status: Mapped[str] = mapped_column(String(30), nullable=False, default="IMPORTED")
    canonical_version: Mapped[int] = mapped_column(Integer, nullable=False, default=1)
    canonical_sha256: Mapped[str | None] = mapped_column(SHA, nullable=True)
    quality_status: Mapped[str | None] = mapped_column(String(10), nullable=True)
    sample_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    data_origin: Mapped[str] = mapped_column(String(30), nullable=False, default="synthetic")
    # --- V2 (SPEC 2.2): profile binding, business revision and schema version -----
    profile_id: Mapped[str | None] = mapped_column(String(80), nullable=True)
    profile_revision: Mapped[int | None] = mapped_column(Integer, nullable=True)
    business_revision: Mapped[int] = mapped_column(
        BigInteger, nullable=False, default=1, server_default="1"
    )
    canonical_schema_version: Mapped[str] = mapped_column(
        String(32), nullable=False, default="1.0", server_default="1.0"
    )
    created_at: Mapped[datetime] = mapped_column(TS, nullable=False, server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(
        TS, nullable=False, server_default=func.now(), onupdate=func.now()
    )

    __table_args__ = (
        UniqueConstraint("source_import_id", "source_episode_index", name="source_episode_unique"),
        CheckConstraint(
            "episode_type IN ('robot_episode','ego_video_episode')", name="episode_type_valid"
        ),
        CheckConstraint(
            "status IN ('IMPORTED','PROCESSING','READY_FOR_REVIEW','QUALITY_FAILED')", name="status_valid"
        ),
        CheckConstraint("quality_status IS NULL OR quality_status IN ('PASS','WARN','FAIL')", name="quality_valid"),
        CheckConstraint("business_revision >= 1", name="business_revision_positive"),
        Index("ix_episodes_created_id", "created_at", "id"),
        Index("ix_episodes_profile", "profile_id", "profile_revision"),
    )


class Modality(Base):
    __tablename__ = "modalities"

    id: Mapped[str] = mapped_column(ID, primary_key=True)
    episode_id: Mapped[str] = mapped_column(ID, ForeignKey("episodes.id"), nullable=False)
    name: Mapped[str] = mapped_column(String(120), nullable=False)
    kind: Mapped[str] = mapped_column(String(20), nullable=False)
    dtype: Mapped[str] = mapped_column(String(60), nullable=False, default="")
    shape_json: Mapped[list | dict] = mapped_column(JSONB, nullable=False, default=list)
    clock_domain: Mapped[str] = mapped_column(String(80), nullable=False)
    sample_count: Mapped[int] = mapped_column(BigInteger, nullable=False, default=0)
    min_t_ns: Mapped[int | None] = mapped_column(BigInteger, nullable=True)
    max_t_ns: Mapped[int | None] = mapped_column(BigInteger, nullable=True)
    source_key: Mapped[str] = mapped_column(String(300), nullable=False, default="")
    object_key: Mapped[str | None] = mapped_column(String(1024), nullable=True)
    # --- V2 (SPEC 2.2): native/aligned indices, stream index and clock evidence ---
    stream_index: Mapped[int] = mapped_column(Integer, nullable=False, default=0, server_default="0")
    native_rate_hz: Mapped[float | None] = mapped_column(Float, nullable=True)
    native_index_asset_id: Mapped[str | None] = mapped_column(
        ID, ForeignKey("assets.id"), nullable=True
    )
    aligned_index_asset_id: Mapped[str | None] = mapped_column(
        ID, ForeignKey("assets.id"), nullable=True
    )
    timestamp_provenance: Mapped[str | None] = mapped_column(String(32), nullable=True)
    clock_verified: Mapped[bool] = mapped_column(
        Boolean, nullable=False, default=False, server_default="false"
    )

    __table_args__ = (
        CheckConstraint("kind IN ('camera','state','action','depth','pointcloud')", name="kind_valid"),
        UniqueConstraint("episode_id", "name", "stream_index", name="stream_unique"),
        CheckConstraint("stream_index >= 0", name="stream_index_nonnegative"),
    )


class CanonicalView(Base):
    """A generated canonical view (NATIVE / ALIGNED_TRAINING / PREVIEW) per episode
    version and generation config (V2 SPEC 2.2).
    """

    __tablename__ = "canonical_views"

    id: Mapped[str] = mapped_column(ID, primary_key=True)
    episode_id: Mapped[str] = mapped_column(ID, ForeignKey("episodes.id"), nullable=False)
    canonical_version: Mapped[int] = mapped_column(Integer, nullable=False)
    view_kind: Mapped[str] = mapped_column(String(32), nullable=False)
    config_sha256: Mapped[str] = mapped_column(SHA, nullable=False)
    metadata_asset_id: Mapped[str | None] = mapped_column(ID, ForeignKey("assets.id"), nullable=True)
    created_at: Mapped[datetime] = mapped_column(TS, nullable=False, server_default=func.now())

    __table_args__ = (
        UniqueConstraint(
            "episode_id", "canonical_version", "view_kind", "config_sha256", name="view_config_unique"
        ),
        CheckConstraint(
            "view_kind IN ('NATIVE','ALIGNED_TRAINING','PREVIEW')", name="view_kind_valid"
        ),
    )


class PipelineRun(Base):
    __tablename__ = "pipeline_runs"

    id: Mapped[str] = mapped_column(ID, primary_key=True)
    episode_id: Mapped[str] = mapped_column(ID, ForeignKey("episodes.id"), nullable=False)
    pipeline_version: Mapped[str] = mapped_column(String(20), nullable=False, default="1.0.0")
    state: Mapped[str] = mapped_column(String(20), nullable=False, default="QUEUED")
    requested_by: Mapped[str | None] = mapped_column(ID, ForeignKey("api_keys.id"), nullable=True)
    cancel_requested: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    error_code: Mapped[str | None] = mapped_column(String(60), nullable=True)
    created_at: Mapped[datetime] = mapped_column(TS, nullable=False, server_default=func.now())
    started_at: Mapped[datetime | None] = mapped_column(TS, nullable=True)
    finished_at: Mapped[datetime | None] = mapped_column(TS, nullable=True)

    __table_args__ = (
        CheckConstraint(
            "state IN ('QUEUED','RUNNING','SUCCEEDED','FAILED','CANCELLED')", name="state_valid"
        ),
        Index("ix_pipeline_runs_episode_id", "episode_id"),
    )


class Task(Base):
    __tablename__ = "tasks"

    id: Mapped[str] = mapped_column(ID, primary_key=True)
    # Episode-level pipeline tasks reference a PipelineRun; import-level tasks
    # reference an ImportJob.  Exactly one of the two is set (SPEC 6.1).
    run_id: Mapped[str | None] = mapped_column(ID, ForeignKey("pipeline_runs.id"), nullable=True)
    import_job_id: Mapped[str | None] = mapped_column(ID, ForeignKey("import_jobs.id"), nullable=True)
    step: Mapped[str] = mapped_column(String(40), nullable=False)
    # V2.1: "" for singleton steps, the clip_id for per-clip fan-out.  Part of the
    # uniqueness key so one run can hold many VLM_PREANNOTATE_CLIP tasks.
    unit_key: Mapped[str] = mapped_column(String(64), nullable=False, default="", server_default="")
    pool: Mapped[str] = mapped_column(String(20), nullable=False, default="cpu")
    state: Mapped[str] = mapped_column(String(20), nullable=False, default="PENDING")
    attempt: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    max_attempts: Mapped[int] = mapped_column(Integer, nullable=False, default=3)
    business_revision: Mapped[int] = mapped_column(
        BigInteger, nullable=False, default=1, server_default="1"
    )
    next_run_at: Mapped[datetime] = mapped_column(TS, nullable=False, server_default=func.now())
    worker_id: Mapped[str | None] = mapped_column(String(120), nullable=True)
    lease_token: Mapped[str | None] = mapped_column(String(36), nullable=True)
    lease_until: Mapped[datetime | None] = mapped_column(TS, nullable=True)
    cancel_requested: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    input_digest: Mapped[str | None] = mapped_column(SHA, nullable=True)
    output_asset_id: Mapped[str | None] = mapped_column(ID, ForeignKey("assets.id"), nullable=True)
    output_json: Mapped[dict] = mapped_column(JSONB, nullable=False, default=dict)
    error_code: Mapped[str | None] = mapped_column(String(60), nullable=True)
    error_detail: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(TS, nullable=False, server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(
        TS, nullable=False, server_default=func.now(), onupdate=func.now()
    )

    __table_args__ = (
        CheckConstraint(
            "state IN ('PENDING','RUNNING','RETRY_WAIT','SUCCEEDED','FAILED','CANCELLED')",
            name="state_valid",
        ),
        CheckConstraint(
            "(run_id IS NOT NULL AND import_job_id IS NULL) OR (run_id IS NULL AND import_job_id IS NOT NULL)",
            name="one_parent",
        ),
        CheckConstraint(
            "step IN ('VALIDATE_SOURCE','IMPORT_EPISODES','NORMALIZE_EPISODE','QUALITY_CHECK',"
            "'DETECT_OBJECTS','BUILD_PREVIEW','READY_FOR_REVIEW',"
            "'EXTRACT_KEYFRAMES','VLM_PREANNOTATE_CLIP','MERGE_CANDIDATE_ANNOTATIONS',"
            "'VALIDATE_AI_OUTPUT')",
            name="step_valid",
        ),
        Index(
            "ix_tasks_run_id_step_unit",
            "run_id",
            "step",
            "unit_key",
            unique=True,
            postgresql_where=text("run_id IS NOT NULL"),
        ),
        Index(
            "ix_tasks_import_job_id_step",
            "import_job_id",
            "step",
            unique=True,
            postgresql_where=text("import_job_id IS NOT NULL"),
        ),
        CheckConstraint("business_revision >= 1", name="business_revision_positive"),
        Index("ix_tasks_claim", "state", "pool", "next_run_at"),
        Index("ix_tasks_lease", "state", "lease_until"),
    )


class TaskDependency(Base):
    __tablename__ = "task_dependencies"

    task_id: Mapped[str] = mapped_column(ID, ForeignKey("tasks.id"), primary_key=True)
    depends_on_task_id: Mapped[str] = mapped_column(ID, ForeignKey("tasks.id"), primary_key=True)

    __table_args__ = (
        CheckConstraint("task_id <> depends_on_task_id", name="no_self_dependency"),
    )


class IdempotencyRequest(Base):
    __tablename__ = "idempotency_requests"

    id: Mapped[str] = mapped_column(ID, primary_key=True)
    actor_key_id: Mapped[str] = mapped_column(ID, ForeignKey("api_keys.id"), nullable=False)
    route: Mapped[str] = mapped_column(String(200), nullable=False)
    idempotency_key: Mapped[str] = mapped_column(String(200), nullable=False)
    request_sha256: Mapped[str] = mapped_column(SHA, nullable=False)
    response_status: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    response_json: Mapped[dict] = mapped_column(JSONB, nullable=False, default=dict)
    created_at: Mapped[datetime] = mapped_column(TS, nullable=False, server_default=func.now())
    expires_at: Mapped[datetime] = mapped_column(TS, nullable=False)

    __table_args__ = (
        UniqueConstraint("actor_key_id", "route", "idempotency_key", name="actor_route_key_unique"),
    )


class OutboxEvent(Base):
    __tablename__ = "outbox_events"

    id: Mapped[str] = mapped_column(ID, primary_key=True)
    topic: Mapped[str] = mapped_column(String(80), nullable=False)
    payload_json: Mapped[dict] = mapped_column(JSONB, nullable=False, default=dict)
    published_at: Mapped[datetime | None] = mapped_column(TS, nullable=True)
    created_at: Mapped[datetime] = mapped_column(TS, nullable=False, server_default=func.now())
    delivery_attempt: Mapped[int] = mapped_column(Integer, nullable=False, default=0)

    __table_args__ = (
        Index("ix_outbox_unpublished", "published_at", "created_at"),
    )


class QualityReport(Base):
    __tablename__ = "quality_reports"

    id: Mapped[str] = mapped_column(ID, primary_key=True)
    episode_id: Mapped[str] = mapped_column(ID, ForeignKey("episodes.id"), nullable=False)
    run_id: Mapped[str] = mapped_column(ID, ForeignKey("pipeline_runs.id"), nullable=False)
    policy_version: Mapped[str] = mapped_column(String(40), nullable=False)
    status: Mapped[str] = mapped_column(String(10), nullable=False)
    summary_json: Mapped[dict] = mapped_column(JSONB, nullable=False, default=dict)
    report_asset_id: Mapped[str | None] = mapped_column(ID, ForeignKey("assets.id"), nullable=True)
    # V2 D2-2: profile binding and report schema version
    profile_id: Mapped[str | None] = mapped_column(String(80), nullable=True)
    profile_revision: Mapped[int | None] = mapped_column(Integer, nullable=True)
    config_sha256: Mapped[str | None] = mapped_column(SHA, nullable=True)
    schema_version: Mapped[str] = mapped_column(
        String(40), nullable=False, default="quality-report/1.0", server_default="quality-report/1.0"
    )
    created_at: Mapped[datetime] = mapped_column(TS, nullable=False, server_default=func.now())

    __table_args__ = (
        UniqueConstraint("episode_id", "run_id", name="episode_run_unique"),
        CheckConstraint("status IN ('PASS','WARN','FAIL')", name="status_valid"),
    )


class QualityFinding(Base):
    __tablename__ = "quality_findings"

    id: Mapped[str] = mapped_column(ID, primary_key=True)
    report_id: Mapped[str] = mapped_column(ID, ForeignKey("quality_reports.id"), nullable=False)
    rule_code: Mapped[str] = mapped_column(String(60), nullable=False)
    severity: Mapped[str] = mapped_column(String(10), nullable=False)
    observed_json: Mapped[dict] = mapped_column(JSONB, nullable=False, default=dict)
    limit_json: Mapped[dict] = mapped_column(JSONB, nullable=False, default=dict)
    evidence_json: Mapped[dict] = mapped_column(JSONB, nullable=False, default=dict)
    # V2 D2-2: explicit provenance of the evidence behind the finding
    source_stream: Mapped[str | None] = mapped_column(String(120), nullable=True)
    source_clock_domain: Mapped[str | None] = mapped_column(String(80), nullable=True)
    method: Mapped[str | None] = mapped_column(String(80), nullable=True)
    limits_json: Mapped[dict] = mapped_column(JSONB, nullable=False, default=dict)
    uncertainty_note: Mapped[str | None] = mapped_column(Text, nullable=True)

    __table_args__ = (
        CheckConstraint("severity IN ('INFO','WARN','ERROR')", name="severity_valid"),
        Index("ix_quality_findings_report_id", "report_id"),
    )


class AnnotationSet(Base):
    __tablename__ = "annotation_sets"

    id: Mapped[str] = mapped_column(ID, primary_key=True)
    episode_id: Mapped[str] = mapped_column(ID, ForeignKey("episodes.id"), nullable=False)
    run_id: Mapped[str] = mapped_column(ID, ForeignKey("pipeline_runs.id"), nullable=False)
    model_id: Mapped[str] = mapped_column(String(120), nullable=False)
    model_sha256: Mapped[str] = mapped_column(SHA, nullable=False)
    status: Mapped[str] = mapped_column(String(20), nullable=False, default="PENDING_REVIEW")
    artifact_asset_id: Mapped[str | None] = mapped_column(ID, ForeignKey("assets.id"), nullable=True)
    summary_json: Mapped[dict] = mapped_column(JSONB, nullable=False, default=dict)
    created_at: Mapped[datetime] = mapped_column(TS, nullable=False, server_default=func.now())
    # V2 D2-3: revision pointer; the latest explicitly approved immutable revision
    current_revision_no: Mapped[int] = mapped_column(
        Integer, nullable=False, default=1, server_default="1"
    )
    approved_revision_id: Mapped[str | None] = mapped_column(
        ID,
        ForeignKey("annotation_revisions.id", use_alter=True, name="fk_annotation_sets_approved_revision_id"),
        nullable=True,
    )

    __table_args__ = (
        CheckConstraint("status IN ('PENDING_REVIEW','APPROVED','REJECTED')", name="status_valid"),
        CheckConstraint("current_revision_no >= 1", name="revision_positive"),
        Index("ix_annotation_sets_episode_id", "episode_id"),
    )


class AnnotationReview(Base):
    __tablename__ = "annotation_reviews"

    id: Mapped[str] = mapped_column(ID, primary_key=True)
    annotation_set_id: Mapped[str] = mapped_column(ID, ForeignKey("annotation_sets.id"), nullable=False)
    reviewer_key_id: Mapped[str] = mapped_column(ID, ForeignKey("api_keys.id"), nullable=False)
    decision: Mapped[str] = mapped_column(String(20), nullable=False)
    note: Mapped[str] = mapped_column(Text, nullable=False, default="")
    created_at: Mapped[datetime] = mapped_column(TS, nullable=False, server_default=func.now())

    __table_args__ = (
        CheckConstraint("decision IN ('APPROVED','REJECTED')", name="decision_valid"),
        UniqueConstraint("annotation_set_id", name="one_review_per_set"),
    )


class AnnotationRevision(Base):
    """Immutable annotation content revision (V2 D2-3, annotation-revision/2.0)."""

    __tablename__ = "annotation_revisions"

    id: Mapped[str] = mapped_column(ID, primary_key=True)
    annotation_set_id: Mapped[str] = mapped_column(ID, ForeignKey("annotation_sets.id"), nullable=False)
    revision_no: Mapped[int] = mapped_column(Integer, nullable=False)
    base_revision_id: Mapped[str | None] = mapped_column(ID, ForeignKey("annotation_revisions.id"), nullable=True)
    artifact_asset_id: Mapped[str | None] = mapped_column(ID, ForeignKey("assets.id"), nullable=True)
    content_sha256: Mapped[str] = mapped_column(SHA, nullable=False)
    state: Mapped[str] = mapped_column(String(20), nullable=False, default="DRAFT")
    created_by: Mapped[str | None] = mapped_column(ID, ForeignKey("api_keys.id"), nullable=True)
    created_at: Mapped[datetime] = mapped_column(TS, nullable=False, server_default=func.now())
    approved_by: Mapped[str | None] = mapped_column(ID, ForeignKey("api_keys.id"), nullable=True)
    approved_at: Mapped[datetime | None] = mapped_column(TS, nullable=True)

    __table_args__ = (
        UniqueConstraint("annotation_set_id", "revision_no", name="set_revision_unique"),
        CheckConstraint("revision_no >= 1", name="revision_positive"),
        CheckConstraint("state IN ('DRAFT','IN_REVIEW','APPROVED','REJECTED')", name="state_valid"),
        Index("ix_annotation_revisions_set", "annotation_set_id", "revision_no"),
    )


class AnnotationRevisionEdit(Base):
    """Audit row for one annotation-revision edit (before/after digests + summary)."""

    __tablename__ = "annotation_revision_edits"

    id: Mapped[str] = mapped_column(ID, primary_key=True)
    revision_id: Mapped[str] = mapped_column(ID, ForeignKey("annotation_revisions.id"), nullable=False)
    actor_key_id: Mapped[str] = mapped_column(ID, ForeignKey("api_keys.id"), nullable=False)
    change_summary_json: Mapped[dict] = mapped_column(JSONB, nullable=False, default=dict)
    before_sha256: Mapped[str] = mapped_column(SHA, nullable=False)
    after_sha256: Mapped[str] = mapped_column(SHA, nullable=False)
    created_at: Mapped[datetime] = mapped_column(TS, nullable=False, server_default=func.now())

    __table_args__ = (Index("ix_annotation_revision_edits_revision_id", "revision_id"),)


class Dataset(Base):
    __tablename__ = "datasets"

    id: Mapped[str] = mapped_column(ID, primary_key=True)
    slug: Mapped[str] = mapped_column(String(120), nullable=False, unique=True)
    description: Mapped[str] = mapped_column(Text, nullable=False, default="")
    created_at: Mapped[datetime] = mapped_column(TS, nullable=False, server_default=func.now())


class DatasetVersion(Base):
    __tablename__ = "dataset_versions"

    id: Mapped[str] = mapped_column(ID, primary_key=True)
    dataset_id: Mapped[str] = mapped_column(ID, ForeignKey("datasets.id"), nullable=False)
    version: Mapped[int] = mapped_column(Integer, nullable=False)
    state: Mapped[str] = mapped_column(String(20), nullable=False, default="BUILDING")
    manifest_asset_id: Mapped[str | None] = mapped_column(ID, ForeignKey("assets.id"), nullable=True)
    export_asset_id: Mapped[str | None] = mapped_column(ID, ForeignKey("assets.id"), nullable=True)
    manifest_sha256: Mapped[str | None] = mapped_column(SHA, nullable=True)
    #: ``ROBOT_LEROBOT_V3`` | ``EGO_ANNOTATION_BUNDLE`` | ``RESEARCH_SNAPSHOT``.
    #: An ego bundle can never be published as a robot control trajectory.
    export_kind: Mapped[str] = mapped_column(
        String(32), nullable=False, default="ROBOT_LEROBOT_V3", server_default="ROBOT_LEROBOT_V3"
    )
    description: Mapped[str] = mapped_column(Text, nullable=False, default="")
    created_by: Mapped[str | None] = mapped_column(ID, ForeignKey("api_keys.id"), nullable=True)
    error_code: Mapped[str | None] = mapped_column(String(60), nullable=True)
    error_detail: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(TS, nullable=False, server_default=func.now())
    published_at: Mapped[datetime | None] = mapped_column(TS, nullable=True)

    __table_args__ = (
        UniqueConstraint("dataset_id", "version", name="dataset_version_unique"),
        CheckConstraint("state IN ('BUILDING','READY','PUBLISHED','FAILED')", name="state_valid"),
        CheckConstraint(
            "export_kind IN ('ROBOT_LEROBOT_V3','EGO_ANNOTATION_BUNDLE','RESEARCH_SNAPSHOT')",
            name="export_kind_valid",
        ),
    )


class DatasetVersionEpisode(Base):
    __tablename__ = "dataset_version_episodes"

    version_id: Mapped[str] = mapped_column(ID, ForeignKey("dataset_versions.id"), primary_key=True)
    episode_id: Mapped[str] = mapped_column(ID, ForeignKey("episodes.id"), primary_key=True)
    canonical_digest: Mapped[str] = mapped_column(SHA, nullable=False)
    quality_report_id: Mapped[str] = mapped_column(ID, ForeignKey("quality_reports.id"), nullable=False)
    annotation_set_id: Mapped[str | None] = mapped_column(
        ID, ForeignKey("annotation_sets.id"), nullable=True
    )
    annotation_revision_id: Mapped[str | None] = mapped_column(
        ID, ForeignKey("annotation_revisions.id"), nullable=True
    )
    position: Mapped[int] = mapped_column(Integer, nullable=False, default=0)


class EvaluationReport(Base):
    __tablename__ = "evaluation_reports"

    id: Mapped[str] = mapped_column(ID, primary_key=True)
    dataset_version_id: Mapped[str] = mapped_column(
        ID, ForeignKey("dataset_versions.id"), nullable=False
    )
    source: Mapped[str] = mapped_column(String(40), nullable=False, default="imported")
    schema_version: Mapped[str] = mapped_column(String(40), nullable=False, default="evaluation/1.0")
    report_asset_id: Mapped[str | None] = mapped_column(ID, ForeignKey("assets.id"), nullable=True)
    summary_json: Mapped[dict] = mapped_column(JSONB, nullable=False, default=dict)
    created_at: Mapped[datetime] = mapped_column(TS, nullable=False, server_default=func.now())
    business_revision: Mapped[int] = mapped_column(
        BigInteger, nullable=False, default=1, server_default="1"
    )

    __table_args__ = (CheckConstraint("business_revision >= 1", name="business_revision_positive"),)


class GapAnalysis(Base):
    __tablename__ = "gap_analyses"

    id: Mapped[str] = mapped_column(ID, primary_key=True)
    evaluation_report_id: Mapped[str] = mapped_column(
        ID, ForeignKey("evaluation_reports.id"), nullable=False, unique=True
    )
    state: Mapped[str] = mapped_column(String(20), nullable=False, default="QUEUED")
    threshold: Mapped[float] = mapped_column(Float, nullable=False, default=0.8)
    min_trials: Mapped[int] = mapped_column(Integer, nullable=False, default=20)
    summary_json: Mapped[dict] = mapped_column(JSONB, nullable=False, default=dict)
    results_asset_id: Mapped[str | None] = mapped_column(ID, ForeignKey("assets.id"), nullable=True)
    created_at: Mapped[datetime] = mapped_column(TS, nullable=False, server_default=func.now())
    finished_at: Mapped[datetime | None] = mapped_column(TS, nullable=True)

    __table_args__ = (
        CheckConstraint("state IN ('QUEUED','RUNNING','SUCCEEDED','FAILED')", name="state_valid"),
    )


class CollectionOrder(Base):
    __tablename__ = "collection_orders"

    id: Mapped[str] = mapped_column(ID, primary_key=True)
    evaluation_report_id: Mapped[str] = mapped_column(
        ID, ForeignKey("evaluation_reports.id"), nullable=False
    )
    scenario_key: Mapped[str] = mapped_column(String(64), nullable=False)
    reason: Mapped[str] = mapped_column(Text, nullable=False)
    requested_count: Mapped[int] = mapped_column(Integer, nullable=False)
    state: Mapped[str] = mapped_column(String(20), nullable=False, default="PROPOSED", server_default="PROPOSED")
    created_by: Mapped[str | None] = mapped_column(ID, ForeignKey("api_keys.id"), nullable=True)
    approved_by: Mapped[str | None] = mapped_column(ID, ForeignKey("api_keys.id"), nullable=True)
    approved_at: Mapped[datetime | None] = mapped_column(TS, nullable=True)
    approval_note: Mapped[str] = mapped_column(Text, nullable=False, default="")
    is_simulated_source: Mapped[bool] = mapped_column(
        Boolean, nullable=False, default=True, server_default="true"
    )
    created_at: Mapped[datetime] = mapped_column(TS, nullable=False, server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(
        TS, nullable=False, server_default=func.now(), onupdate=func.now()
    )

    __table_args__ = (
        UniqueConstraint("evaluation_report_id", "scenario_key", name="report_scenario_unique"),
        CheckConstraint(
            "state IN ('PROPOSED','APPROVED','IN_PROGRESS','FULFILLED','CANCELLED')", name="state_valid"
        ),
        CheckConstraint("scenario_key ~ '^[a-z0-9_]{3,64}$'", name="scenario_key_format"),
        Index("ix_collection_orders_created_id", "created_at", "id"),
    )


class CollectionOrderEpisode(Base):
    """Episode linked to a collection order as fulfillment data (V2 D2-6)."""

    __tablename__ = "collection_order_episodes"

    order_id: Mapped[str] = mapped_column(ID, ForeignKey("collection_orders.id"), primary_key=True)
    episode_id: Mapped[str] = mapped_column(ID, ForeignKey("episodes.id"), primary_key=True)
    linked_by: Mapped[str | None] = mapped_column(ID, ForeignKey("api_keys.id"), nullable=True)
    linked_at: Mapped[datetime] = mapped_column(TS, nullable=False, server_default=func.now())

    __table_args__ = (
        UniqueConstraint("episode_id", name="episode_single_order_unique"),
        Index("ix_collection_order_episodes_order_id", "order_id"),
    )


class ScenarioTag(Base):
    """Evidence-backed scenario label (V2 D2-6). UNKNOWN never counts as coverage."""

    __tablename__ = "scenario_tags"

    id: Mapped[str] = mapped_column(ID, primary_key=True)
    episode_id: Mapped[str] = mapped_column(ID, ForeignKey("episodes.id"), nullable=False)
    scenario_key: Mapped[str] = mapped_column(String(64), nullable=False)
    basis: Mapped[str] = mapped_column(String(20), nullable=False)
    source_evidence_asset_id: Mapped[str | None] = mapped_column(ID, ForeignKey("assets.id"), nullable=True)
    verified_by: Mapped[str | None] = mapped_column(ID, ForeignKey("api_keys.id"), nullable=True)
    created_at: Mapped[datetime] = mapped_column(TS, nullable=False, server_default=func.now())

    __table_args__ = (
        UniqueConstraint("episode_id", "scenario_key", "basis", name="episode_scenario_basis_unique"),
        CheckConstraint("basis IN ('SOURCE_DECLARED','HUMAN_VERIFIED','UNKNOWN')", name="basis_valid"),
        CheckConstraint("scenario_key ~ '^[a-z0-9_]{3,64}$'", name="scenario_key_format"),
    )


class ScenarioCoverageSnapshot(Base):
    """Immutable per-PUBLISHED-version scenario coverage snapshot (V2 D2-6)."""

    __tablename__ = "scenario_coverage_snapshots"

    id: Mapped[str] = mapped_column(ID, primary_key=True)
    dataset_version_id: Mapped[str] = mapped_column(ID, ForeignKey("dataset_versions.id"), nullable=False)
    scenario_key: Mapped[str] = mapped_column(String(64), nullable=False)
    episode_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    tagging_basis: Mapped[str] = mapped_column(String(20), nullable=False, default="UNKNOWN")
    source_digest: Mapped[str] = mapped_column(SHA, nullable=False)
    computed_at: Mapped[datetime] = mapped_column(TS, nullable=False, server_default=func.now())

    __table_args__ = (
        UniqueConstraint("dataset_version_id", "scenario_key", "tagging_basis", name="version_scenario_basis_unique"),
        CheckConstraint("tagging_basis IN ('HUMAN_VERIFIED','SOURCE_DECLARED','UNKNOWN')", name="basis_valid"),
    )


class OperationReceipt(Base):
    """Idempotent receipt of one controlled business operation (DO-A02 C2/C4)."""

    __tablename__ = "operation_receipts"

    id: Mapped[str] = mapped_column(ID, primary_key=True)
    tenant_id: Mapped[str] = mapped_column(String(32), nullable=False, default="default")
    action_kind: Mapped[str] = mapped_column(String(40), nullable=False)
    target_id: Mapped[str] = mapped_column(ID, nullable=False)
    source_business_revision: Mapped[int] = mapped_column(BigInteger, nullable=False)
    idempotency_key: Mapped[str] = mapped_column(String(200), nullable=False)
    request_sha256: Mapped[str] = mapped_column(SHA, nullable=False)
    effect_state: Mapped[str] = mapped_column(String(20), nullable=False, default="COMMITTED")
    downstream_kind: Mapped[str | None] = mapped_column(String(40), nullable=True)
    downstream_id: Mapped[str | None] = mapped_column(ID, nullable=True)
    error_code: Mapped[str | None] = mapped_column(String(60), nullable=True)
    created_at: Mapped[datetime] = mapped_column(TS, nullable=False, server_default=func.now())

    __table_args__ = (
        UniqueConstraint("tenant_id", "idempotency_key", name="tenant_idempotency_unique"),
        CheckConstraint(
            "action_kind IN ('REPROCESS_EPISODE','RETRY_FAILED_TASK','ANALYZE_EVALUATION_GAPS')",
            name="action_kind_valid",
        ),
        CheckConstraint(
            "effect_state IN ('COMMITTED','IN_PROGRESS','FAILED')", name="effect_state_valid"
        ),
    )


class ActionEffectFence(Base):
    """One valid effect per business revision/action; never deleted by TTL."""

    __tablename__ = "action_effect_fences"

    id: Mapped[str] = mapped_column(ID, primary_key=True)
    tenant_id: Mapped[str] = mapped_column(String(32), nullable=False, default="default")
    action_kind: Mapped[str] = mapped_column(String(40), nullable=False)
    target_id: Mapped[str] = mapped_column(ID, nullable=False)
    source_business_revision: Mapped[int] = mapped_column(BigInteger, nullable=False)
    operation_id: Mapped[str] = mapped_column(ID, ForeignKey("operation_receipts.id"), nullable=False)
    created_at: Mapped[datetime] = mapped_column(TS, nullable=False, server_default=func.now())

    __table_args__ = (
        UniqueConstraint(
            "tenant_id", "action_kind", "target_id", "source_business_revision", name="effect_fence_unique"
        ),
        CheckConstraint(
            "action_kind IN ('REPROCESS_EPISODE','RETRY_FAILED_TASK','ANALYZE_EVALUATION_GAPS')",
            name="action_kind_valid",
        ),
    )


class AuditEvent(Base):
    __tablename__ = "audit_events"

    id: Mapped[str] = mapped_column(ID, primary_key=True)
    actor_id: Mapped[str | None] = mapped_column(ID, ForeignKey("api_keys.id"), nullable=True)
    action: Mapped[str] = mapped_column(String(80), nullable=False)
    entity_type: Mapped[str] = mapped_column(String(60), nullable=False)
    entity_id: Mapped[str] = mapped_column(String(60), nullable=False)
    before_json: Mapped[dict | None] = mapped_column(JSONB, nullable=True)
    after_json: Mapped[dict | None] = mapped_column(JSONB, nullable=True)
    request_id: Mapped[str | None] = mapped_column(String(80), nullable=True)
    created_at: Mapped[datetime] = mapped_column(TS, nullable=False, server_default=func.now())

    __table_args__ = (Index("ix_audit_events_entity", "entity_type", "entity_id"),)


# ---------------------------------------------------------------------------
# V2.1 AI pre-annotation (SPEC_V2.1 §5.2, §13)
# ---------------------------------------------------------------------------


class AIClipInput(Base):
    """One clip that was actually sent to a vision-language model.

    The frame manifest asset keeps the position -> grid-index mapping, so an
    ``evidence_frame_indices`` value from the model (clip-local position) can always be
    resolved back to a real frame.
    """

    __tablename__ = "ai_clip_inputs"

    id: Mapped[str] = mapped_column(ID, primary_key=True)
    episode_id: Mapped[str] = mapped_column(ID, ForeignKey("episodes.id"), nullable=False)
    clip_id: Mapped[str] = mapped_column(SHA, nullable=False, unique=True)
    start_ns: Mapped[int] = mapped_column(BigInteger, nullable=False)
    end_ns: Mapped[int] = mapped_column(BigInteger, nullable=False)
    source_asset_sha256: Mapped[str] = mapped_column(SHA, nullable=False)
    preprocessing_sha256: Mapped[str] = mapped_column(SHA, nullable=False)
    frame_manifest_asset_id: Mapped[str] = mapped_column(ID, ForeignKey("assets.id"), nullable=False)
    frame_count: Mapped[int] = mapped_column(Integer, nullable=False)
    created_at: Mapped[datetime] = mapped_column(TS, nullable=False, server_default=func.now())

    __table_args__ = (
        CheckConstraint(
            "start_ns >= 0 AND end_ns > start_ns", name="clip_interval_valid"
        ),
        CheckConstraint("frame_count BETWEEN 1 AND 8", name="frame_count_valid"),
        Index("ix_ai_clip_inputs_episode", "episode_id", "start_ns"),
    )


class ModelInference(Base):
    """One logical model task for one clip (may contain several billed attempts)."""

    __tablename__ = "model_inferences"

    id: Mapped[str] = mapped_column(ID, primary_key=True)
    run_id: Mapped[str] = mapped_column(ID, ForeignKey("pipeline_runs.id"), nullable=False)
    task_id: Mapped[str] = mapped_column(ID, ForeignKey("tasks.id"), nullable=False)
    clip_input_id: Mapped[str] = mapped_column(ID, ForeignKey("ai_clip_inputs.id"), nullable=False)
    provider: Mapped[str] = mapped_column(String(48), nullable=False)
    model_name: Mapped[str] = mapped_column(String(96), nullable=False)
    mode: Mapped[str] = mapped_column(String(12), nullable=False)
    prompt_version: Mapped[str] = mapped_column(String(64), nullable=False)
    prompt_sha256: Mapped[str] = mapped_column(SHA, nullable=False)
    request_sha256: Mapped[str] = mapped_column(SHA, nullable=False)
    output_asset_id: Mapped[str | None] = mapped_column(ID, ForeignKey("assets.id"), nullable=True)
    output_sha256: Mapped[str | None] = mapped_column(SHA, nullable=True)
    status: Mapped[str] = mapped_column(String(16), nullable=False, default="QUEUED")
    error_code: Mapped[str | None] = mapped_column(String(64), nullable=True)
    estimated_cost_usd: Mapped[Decimal | None] = mapped_column(Numeric(14, 6), nullable=True)
    latency_ms: Mapped[int | None] = mapped_column(Integer, nullable=True)
    created_at: Mapped[datetime] = mapped_column(TS, nullable=False, server_default=func.now())
    finished_at: Mapped[datetime | None] = mapped_column(TS, nullable=True)

    __table_args__ = (
        CheckConstraint("mode IN ('LIVE','REPLAY')", name="mode_valid"),
        CheckConstraint(
            "status IN ('QUEUED','RUNNING','SUCCEEDED','FAILED','BLOCKED')", name="status_valid"
        ),
        # One committed product per (run, clip, provider, model, prompt).  A retried
        # remote call inserts another attempt row, never another inference.
        UniqueConstraint(
            "run_id",
            "clip_input_id",
            "provider",
            "model_name",
            "prompt_sha256",
            name="inference_identity_unique",
        ),
        Index("ix_model_inferences_run", "run_id"),
    )


class ModelCallAttempt(Base):
    """One real remote HTTP attempt; this is where duplicate billing becomes visible."""

    __tablename__ = "model_call_attempts"

    id: Mapped[str] = mapped_column(ID, primary_key=True)
    inference_id: Mapped[str] = mapped_column(ID, ForeignKey("model_inferences.id"), nullable=False)
    attempt_no: Mapped[int] = mapped_column(Integer, nullable=False)
    provider_request_id: Mapped[str | None] = mapped_column(String(128), nullable=True)
    response_sha256: Mapped[str | None] = mapped_column(SHA, nullable=True)
    usage_prompt_tokens: Mapped[int | None] = mapped_column(Integer, nullable=True)
    usage_completion_tokens: Mapped[int | None] = mapped_column(Integer, nullable=True)
    estimated_cost_usd: Mapped[Decimal | None] = mapped_column(Numeric(14, 6), nullable=True)
    response_status: Mapped[int | None] = mapped_column(Integer, nullable=True)
    error_code: Mapped[str | None] = mapped_column(String(64), nullable=True)
    #: ``KNOWN`` | ``UNKNOWN_UPSTREAM_CHARGE`` | ``NOT_BILLED``.  A transport failure
    #: can happen after the provider accepted the request, so "no recorded cost" must
    #: not be read as "not billed" (SPEC_V2.1 §5.4).
    cost_state: Mapped[str] = mapped_column(
        String(24), nullable=False, default="NOT_BILLED", server_default="NOT_BILLED"
    )
    started_at: Mapped[datetime] = mapped_column(TS, nullable=False)
    finished_at: Mapped[datetime | None] = mapped_column(TS, nullable=True)

    __table_args__ = (
        CheckConstraint("attempt_no BETWEEN 1 AND 3", name="attempt_no_valid"),
        CheckConstraint(
            "cost_state IN ('KNOWN','UNKNOWN_UPSTREAM_CHARGE','NOT_BILLED')",
            name="cost_state_valid",
        ),
        UniqueConstraint("inference_id", "attempt_no", name="inference_attempt_unique"),
    )


class CandidateActionAnnotation(Base):
    """One candidate action interval produced by a model, awaiting human review."""

    __tablename__ = "candidate_action_annotations"

    id: Mapped[str] = mapped_column(ID, primary_key=True)
    annotation_set_id: Mapped[str] = mapped_column(
        ID, ForeignKey("annotation_sets.id"), nullable=False
    )
    model_inference_id: Mapped[str] = mapped_column(
        ID, ForeignKey("model_inferences.id"), nullable=False
    )
    clip_input_id: Mapped[str] = mapped_column(ID, ForeignKey("ai_clip_inputs.id"), nullable=False)
    revision_id: Mapped[str | None] = mapped_column(
        ID, ForeignKey("annotation_revisions.id"), nullable=True
    )
    action_index: Mapped[int] = mapped_column(Integer, nullable=False)
    verb: Mapped[str] = mapped_column(String(32), nullable=False)
    noun: Mapped[str] = mapped_column(String(64), nullable=False)
    start_ns: Mapped[int] = mapped_column(BigInteger, nullable=False)
    end_ns: Mapped[int] = mapped_column(BigInteger, nullable=False)
    evidence_frame_indexes: Mapped[list] = mapped_column(JSONB, nullable=False, default=list)
    uncertainty: Mapped[str] = mapped_column(String(8), nullable=False)
    review_state: Mapped[str] = mapped_column(String(16), nullable=False, default="PENDING")

    __table_args__ = (
        CheckConstraint("action_index BETWEEN 0 AND 11", name="action_index_valid"),
        CheckConstraint("end_ns > start_ns", name="action_interval_valid"),
        CheckConstraint("uncertainty IN ('low','medium','high')", name="uncertainty_valid"),
        CheckConstraint(
            "review_state IN ('PENDING','IN_REVIEW','APPROVED','REJECTED')", name="review_state_valid"
        ),
        UniqueConstraint("model_inference_id", "action_index", name="inference_action_unique"),
        Index("ix_candidate_actions_annotation_set", "annotation_set_id"),
    )
