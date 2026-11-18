"""Annotation API schemas (SPEC 8.1, 9.4).

These models are the OpenAPI 3.1 surface of

* ``GET  /api/v1/episodes/{episode_id}/annotations``
* ``POST /api/v1/annotation-sets/{annotation_set_id}/reviews``

and validate model output as untrusted input (SPEC 12.1): coordinates are
bounds-checked against the final 640x480 canvas and review notes are length
limited.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from embodied_dataops import TARGET_HEIGHT, TARGET_WIDTH

ReviewDecisionLiteral = Literal["APPROVED", "REJECTED"]
AnnotationStatusLiteral = Literal["PENDING_REVIEW", "APPROVED", "REJECTED"]


class DetectionOut(BaseModel):
    """One detected object in final 640x480 pixel coordinates."""

    model_config = ConfigDict(extra="forbid")

    class_id: int = Field(ge=0, le=79, description="COCO class id")
    class_name: str = Field(min_length=1, max_length=80)
    score: float = Field(ge=0.0, le=1.0, description="Confidence rounded to 6 decimals")
    xyxy: list[float] = Field(
        min_length=4,
        max_length=4,
        description=f"Box corners [x1,y1,x2,y2] within 0..{TARGET_WIDTH} x 0..{TARGET_HEIGHT}",
    )

    @field_validator("xyxy")
    @classmethod
    def _bounds(cls, value: list[float]) -> list[float]:
        x1, y1, x2, y2 = value
        if not (0.0 <= x1 <= x2 <= TARGET_WIDTH):
            raise ValueError(f"xyxy x-range must satisfy 0 <= x1 <= x2 <= {TARGET_WIDTH}")
        if not (0.0 <= y1 <= y2 <= TARGET_HEIGHT):
            raise ValueError(f"xyxy y-range must satisfy 0 <= y1 <= y2 <= {TARGET_HEIGHT}")
        return value


class AnnotationFrameOut(BaseModel):
    """One sampled (1 fps) frame and its detections."""

    model_config = ConfigDict(extra="forbid")

    frame_index: int = Field(ge=0, description="Canonical 10 Hz frame index (0, 10, 20, ...)")
    t_ns: int = Field(ge=0)
    preview_url: str = Field(
        min_length=1,
        description="Short-lived presigned URL of the canonical preview video (seek to frame_index)",
    )
    detections: list[DetectionOut] = Field(default_factory=list)


class AnnotationSetOut(BaseModel):
    """Metadata of one immutable AnnotationSet."""

    model_config = ConfigDict(extra="forbid")

    annotation_set_id: str
    episode_id: str
    run_id: str
    model_id: str
    model_sha256: str
    status: AnnotationStatusLiteral
    created_at: datetime
    schema_version: str | None = None
    tool: str | None = Field(default=None, description="'yolo11n' for real inference, 'fake' for test doubles")
    is_synthetic: bool = Field(
        default=False, description="True when produced by FakeDetector (never real inference)"
    )
    model: dict[str, Any] = Field(default_factory=dict)
    stats: dict[str, Any] = Field(default_factory=dict)
    summary: dict[str, Any] = Field(default_factory=dict)
    artifact_url: str
    artifact_sha256: str
    artifact_size: int = 0
    preview_video_url: str
    thumbnail_url: str | None = None
    preview_expires_at: datetime
    # V2 D2-M3: revision pointers (v1 sets keep revision_count=0).
    current_revision_no: int = 1
    approved_revision_id: str | None = None
    revision_count: int = 0


class EpisodeAnnotationsOut(BaseModel):
    """Payload of ``GET /episodes/{id}/annotations``."""

    model_config = ConfigDict(extra="forbid")

    annotation_set: AnnotationSetOut
    frames: list[AnnotationFrameOut] = Field(default_factory=list)


class EpisodeAnnotationsResponse(BaseModel):
    """Success envelope (SPEC 9.1)."""

    model_config = ConfigDict(extra="forbid")

    data: EpisodeAnnotationsOut
    request_id: str


class AnnotationReviewRequest(BaseModel):
    """Request body of ``POST /annotation-sets/{id}/reviews``."""

    model_config = ConfigDict(extra="forbid")

    decision: ReviewDecisionLiteral
    note: str = Field(default="", max_length=4000)

    @field_validator("note")
    @classmethod
    def _strip(cls, value: str) -> str:
        return value.strip()


class AnnotationReviewCreated(BaseModel):
    """New review decision returned to the caller."""

    model_config = ConfigDict(extra="forbid")

    review_id: str
    new_status: AnnotationStatusLiteral


class AnnotationReviewResponse(BaseModel):
    """Success envelope (SPEC 9.1) for ``POST /annotation-sets/{id}/reviews``."""

    model_config = ConfigDict(extra="forbid")

    data: AnnotationReviewCreated
    request_id: str


# ---------------------------------------------------------------------------
# V2 D2-M3: annotation-revision/2.0 (SPEC §4.2, DO2-14..17)
# ---------------------------------------------------------------------------
RevisionOriginLiteral = Literal["MODEL", "HUMAN"]
RevisionStateLiteral = Literal["DRAFT", "IN_REVIEW", "APPROVED", "REJECTED"]
RevisionDecisionLiteral = Literal["APPROVED", "REJECTED"]
REVISION_SCHEMA_VERSION = "annotation-revision/2.0"


class AnnotationRevisionDetection(BaseModel):
    """One detection inside a ``annotation-revision/2.0`` frame.

    Coordinates must satisfy SPEC §4.2: ``0 <= x1 < x2 <= 640`` and
    ``0 <= y1 < y2 <= 480``.  ``score`` is ``null`` for a purely manual box.
    """

    model_config = ConfigDict(extra="forbid")

    detection_id: str = Field(min_length=1, max_length=100)
    class_id: int = Field(ge=0, le=79, description="Detector/COCO class id (0..79)")
    class_name: str = Field(
        min_length=1,
        max_length=80,
        description="Fixed detector taxonomy name, or 'other' for a manual class",
    )
    score: float | None = Field(default=None, ge=0.0, le=1.0)
    xyxy: list[float] = Field(
        min_length=4,
        max_length=4,
        description=f"Box corners [x1,y1,x2,y2] with 0<=x1<x2<={TARGET_WIDTH}, 0<=y1<y2<={TARGET_HEIGHT}",
    )
    origin: RevisionOriginLiteral = Field(description="MODEL output or HUMAN edit")

    @field_validator("xyxy")
    @classmethod
    def _bounds(cls, value: list[float]) -> list[float]:
        x1, y1, x2, y2 = value
        if not (0.0 <= x1 < x2 <= float(TARGET_WIDTH)):
            raise ValueError(f"xyxy x-range must satisfy 0 <= x1 < x2 <= {TARGET_WIDTH}")
        if not (0.0 <= y1 < y2 <= float(TARGET_HEIGHT)):
            raise ValueError(f"xyxy y-range must satisfy 0 <= y1 < y2 <= {TARGET_HEIGHT}")
        return value


class AnnotationRevisionFrame(BaseModel):
    """One video frame of an ``annotation-revision/2.0`` document."""

    model_config = ConfigDict(extra="forbid")

    frame_index: int = Field(ge=0)
    t_ns: int = Field(ge=0)
    detections: list[AnnotationRevisionDetection] = Field(default_factory=list, max_length=100)


class AnnotationRevisionDocument(BaseModel):
    """Full ``annotation-revision/2.0`` content returned by GET."""

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["annotation-revision/2.0"]
    annotation_set_id: str
    revision_no: int = Field(ge=1)
    camera_name: str = Field(min_length=1, max_length=120)
    frame_width: int
    frame_height: int
    frames: list[AnnotationRevisionFrame]


class AnnotationRevisionCreateRequest(BaseModel):
    """Request body of ``POST /annotation-sets/{id}/revisions`` (SPEC §4.2)."""

    model_config = ConfigDict(extra="forbid")

    base_revision_no: int = Field(ge=1)
    frames: list[AnnotationRevisionFrame] = Field(
        default_factory=list,
        max_length=600,
        description="Only the frames to replace; every other frame is copied from the base revision",
    )
    note: str = Field(
        min_length=10,
        max_length=500,
        description="Change note, 10..500 characters (trimmed before validation)",
    )

    @field_validator("note")
    @classmethod
    def _strip_note(cls, value: str) -> str:
        cleaned = value.strip()
        if len(cleaned) < 10 or len(cleaned) > 500:
            raise ValueError("note must be 10..500 characters after trimming whitespace")
        return cleaned


class AnnotationRevisionDecisionRequest(BaseModel):
    """Request body of ``POST /annotation-revisions/{id}/decision``."""

    model_config = ConfigDict(extra="forbid")

    decision: RevisionDecisionLiteral
    note: str = Field(default="", max_length=4000)

    @field_validator("note")
    @classmethod
    def _strip(cls, value: str) -> str:
        return (value or "").strip()


class AnnotationRevisionOut(BaseModel):
    """Metadata of one immutable annotation revision."""

    model_config = ConfigDict(extra="forbid")

    revision_id: str | None = None
    annotation_set_id: str
    revision_no: int
    state: RevisionStateLiteral
    base_revision_id: str | None = None
    content_sha256: str
    artifact_sha256: str | None = None
    created_by: str | None = None
    created_at: datetime | None = None
    approved_by: str | None = None
    approved_at: datetime | None = None
    change_summary: dict[str, Any] | None = None
    note: str | None = None


class AnnotationRevisionListOut(BaseModel):
    """Payload of ``GET /annotation-sets/{id}/revisions``."""

    model_config = ConfigDict(extra="forbid")

    annotation_set_id: str
    status: str
    current_revision_no: int
    approved_revision_id: str | None = None
    revisions: list[AnnotationRevisionOut] = Field(default_factory=list)


class AnnotationRevisionContentOut(BaseModel):
    """Payload of ``GET /annotation-sets/{id}/revisions/{no}``."""

    model_config = ConfigDict(extra="forbid")

    annotation_set_id: str
    revision_no: int
    revision_id: str | None = None
    state: RevisionStateLiteral
    content_sha256: str
    etag: str
    annotation: AnnotationRevisionDocument


class AnnotationRevisionCreatedOut(BaseModel):
    """New revision returned by ``POST /annotation-sets/{id}/revisions``."""

    model_config = ConfigDict(extra="forbid")

    annotation_set_id: str
    current_revision_no: int
    etag: str
    revision: AnnotationRevisionOut


class AnnotationRevisionDecisionOut(BaseModel):
    """Result of ``POST /annotation-revisions/{id}/decision``."""

    model_config = ConfigDict(extra="forbid")

    annotation_set_id: str
    status: str
    current_revision_no: int
    approved_revision_id: str | None = None
    revision: AnnotationRevisionOut


class AnnotationRevisionListResponse(BaseModel):
    """Success envelope for the revision list."""

    model_config = ConfigDict(extra="forbid")

    data: AnnotationRevisionListOut
    request_id: str


class AnnotationRevisionResponse(BaseModel):
    """Success envelope for one revision content."""

    model_config = ConfigDict(extra="forbid")

    data: AnnotationRevisionContentOut
    request_id: str


class AnnotationRevisionCreatedResponse(BaseModel):
    """Success envelope for a new revision."""

    model_config = ConfigDict(extra="forbid")

    data: AnnotationRevisionCreatedOut
    request_id: str


class AnnotationRevisionDecisionResponse(BaseModel):
    """Success envelope for a revision decision."""

    model_config = ConfigDict(extra="forbid")

    data: AnnotationRevisionDecisionOut
    request_id: str


__all__ = [
    "AnnotationFrameOut",
    "AnnotationRevisionContentOut",
    "AnnotationRevisionCreateRequest",
    "AnnotationRevisionCreatedOut",
    "AnnotationRevisionCreatedResponse",
    "AnnotationRevisionDecisionOut",
    "AnnotationRevisionDecisionRequest",
    "AnnotationRevisionDecisionResponse",
    "AnnotationRevisionDetection",
    "AnnotationRevisionDocument",
    "AnnotationRevisionFrame",
    "AnnotationRevisionListOut",
    "AnnotationRevisionListResponse",
    "AnnotationRevisionOut",
    "AnnotationRevisionResponse",
    "AnnotationReviewCreated",
    "AnnotationReviewRequest",
    "AnnotationReviewResponse",
    "AnnotationSetOut",
    "DetectionOut",
    "EpisodeAnnotationsOut",
    "EpisodeAnnotationsResponse",
    "REVISION_SCHEMA_VERSION",
]
