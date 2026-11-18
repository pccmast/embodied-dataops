"""Domain enums shared by the database layer, API and workers (SPEC 4.3, 6.2)."""

from __future__ import annotations

from enum import StrEnum


class Role(StrEnum):
    VIEWER = "viewer"
    OPERATOR = "operator"
    REVIEWER = "reviewer"
    ADMIN = "admin"


class SourceType(StrEnum):
    LEROBOT_V3 = "LEROBOT_V3"
    EGO_MP4 = "EGO_MP4"
    MCAP_ROS2_PROFILE_V1 = "MCAP_ROS2_PROFILE_V1"
    MCAP_ROS2_PROFILE_V2 = "MCAP_ROS2_PROFILE_V2"


class DataOrigin(StrEnum):
    SYNTHETIC = "synthetic"
    PUBLIC_DATASET = "public_dataset"
    ROBOT_CAPTURE = "robot_capture"
    EGO_CAPTURE = "ego_capture"


class UploadState(StrEnum):
    INITIATED = "INITIATED"
    UPLOADED = "UPLOADED"
    VALIDATING = "VALIDATING"
    READY = "READY"
    FAILED = "FAILED"


class ImportJobState(StrEnum):
    QUEUED = "QUEUED"
    RUNNING = "RUNNING"
    SUCCEEDED = "SUCCEEDED"
    FAILED = "FAILED"


class EpisodeType(StrEnum):
    ROBOT_EPISODE = "robot_episode"
    EGO_VIDEO_EPISODE = "ego_video_episode"


class EpisodeStatus(StrEnum):
    IMPORTED = "IMPORTED"
    PROCESSING = "PROCESSING"
    READY_FOR_REVIEW = "READY_FOR_REVIEW"
    QUALITY_FAILED = "QUALITY_FAILED"


class RunState(StrEnum):
    QUEUED = "QUEUED"
    RUNNING = "RUNNING"
    SUCCEEDED = "SUCCEEDED"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"


class Step(StrEnum):
    VALIDATE_SOURCE = "VALIDATE_SOURCE"
    IMPORT_EPISODES = "IMPORT_EPISODES"
    NORMALIZE_EPISODE = "NORMALIZE_EPISODE"
    QUALITY_CHECK = "QUALITY_CHECK"
    DETECT_OBJECTS = "DETECT_OBJECTS"
    BUILD_PREVIEW = "BUILD_PREVIEW"
    READY_FOR_REVIEW = "READY_FOR_REVIEW"
    # --- V2.1 AI pre-annotation branch (SPEC_V2.1 §5.1, §5.5) -----------------
    EXTRACT_KEYFRAMES = "EXTRACT_KEYFRAMES"
    VLM_PREANNOTATE_CLIP = "VLM_PREANNOTATE_CLIP"
    MERGE_CANDIDATE_ANNOTATIONS = "MERGE_CANDIDATE_ANNOTATIONS"
    VALIDATE_AI_OUTPUT = "VALIDATE_AI_OUTPUT"


class TaskState(StrEnum):
    PENDING = "PENDING"
    RUNNING = "RUNNING"
    RETRY_WAIT = "RETRY_WAIT"
    SUCCEEDED = "SUCCEEDED"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"


class QualityStatus(StrEnum):
    PASS = "PASS"
    WARN = "WARN"
    FAIL = "FAIL"


class Severity(StrEnum):
    INFO = "INFO"
    WARN = "WARN"
    ERROR = "ERROR"


class AnnotationStatus(StrEnum):
    PENDING_REVIEW = "PENDING_REVIEW"
    APPROVED = "APPROVED"
    REJECTED = "REJECTED"


class ReviewDecision(StrEnum):
    APPROVED = "APPROVED"
    REJECTED = "REJECTED"


class DatasetVersionState(StrEnum):
    BUILDING = "BUILDING"
    READY = "READY"
    PUBLISHED = "PUBLISHED"
    FAILED = "FAILED"


class AssetOrigin(StrEnum):
    RAW = "raw"
    STAGING = "staging"
    CANONICAL = "canonical"
    DERIVED = "derived"
    EXPORT = "export"
    REPORT = "report"


class AnalysisState(StrEnum):
    QUEUED = "QUEUED"
    RUNNING = "RUNNING"
    SUCCEEDED = "SUCCEEDED"
    FAILED = "FAILED"


class ModalityKind(StrEnum):
    CAMERA = "camera"
    STATE = "state"
    ACTION = "action"
    DEPTH = "depth"
    POINTCLOUD = "pointcloud"


#: Task steps that execute on each worker pool (SPEC 3.1, 5.4).
STEP_POOL: dict[Step, str] = {
    Step.VALIDATE_SOURCE: "cpu",
    Step.IMPORT_EPISODES: "cpu",
    Step.NORMALIZE_EPISODE: "cpu",
    Step.QUALITY_CHECK: "cpu",
    Step.DETECT_OBJECTS: "vision",
    Step.BUILD_PREVIEW: "cpu",
    Step.READY_FOR_REVIEW: "cpu",
    Step.EXTRACT_KEYFRAMES: "cpu",
    Step.VLM_PREANNOTATE_CLIP: "vision",
    Step.MERGE_CANDIDATE_ANNOTATIONS: "cpu",
    Step.VALIDATE_AI_OUTPUT: "cpu",
}

#: Per-step wall-clock timeout in seconds (SPEC 6.1).
STEP_TIMEOUT_SECONDS: dict[Step, int] = {
    Step.VALIDATE_SOURCE: 120,
    Step.IMPORT_EPISODES: 600,
    Step.NORMALIZE_EPISODE: 600,
    Step.QUALITY_CHECK: 120,
    Step.DETECT_OBJECTS: 600,
    Step.BUILD_PREVIEW: 120,
    Step.READY_FOR_REVIEW: 30,
    Step.EXTRACT_KEYFRAMES: 300,
    # One clip can retry twice with 5 s + 20 s backoff inside a 60 s timeout each.
    Step.VLM_PREANNOTATE_CLIP: 900,
    Step.MERGE_CANDIDATE_ANNOTATIONS: 120,
    Step.VALIDATE_AI_OUTPUT: 120,
}

#: Episode-level pipeline DAG (SPEC 6.1).  Dependencies are explicit.
PIPELINE_DAG: dict[Step, tuple[Step, ...]] = {
    Step.NORMALIZE_EPISODE: (),
    Step.QUALITY_CHECK: (Step.NORMALIZE_EPISODE,),
    Step.DETECT_OBJECTS: (Step.QUALITY_CHECK,),
    Step.BUILD_PREVIEW: (Step.QUALITY_CHECK,),
    Step.READY_FOR_REVIEW: (Step.DETECT_OBJECTS, Step.BUILD_PREVIEW),
}

#: V2.1 DAG (SPEC_V2.1 §1.1, OV-02).  The V2 DAG above is untouched so V1/V2 runs keep
#: their exact shape; a 2.1 run opts in through ``PIPELINE_VERSION_DAG``.
#:
#: ``VLM_PREANNOTATE_CLIP`` fans out once per clip: the dependency below is the
#: *template* edge from ``EXTRACT_KEYFRAMES``, and the builder materialises one task per
#: clip plus one ``MERGE`` dependency on every instance.
PIPELINE_DAG_V21: dict[Step, tuple[Step, ...]] = {
    Step.NORMALIZE_EPISODE: (),
    Step.QUALITY_CHECK: (Step.NORMALIZE_EPISODE,),
    Step.DETECT_OBJECTS: (Step.QUALITY_CHECK,),
    Step.EXTRACT_KEYFRAMES: (Step.QUALITY_CHECK,),
    Step.VLM_PREANNOTATE_CLIP: (Step.EXTRACT_KEYFRAMES,),
    Step.BUILD_PREVIEW: (Step.QUALITY_CHECK,),
    Step.MERGE_CANDIDATE_ANNOTATIONS: (Step.VLM_PREANNOTATE_CLIP, Step.DETECT_OBJECTS),
    Step.VALIDATE_AI_OUTPUT: (Step.MERGE_CANDIDATE_ANNOTATIONS,),
    Step.READY_FOR_REVIEW: (Step.VALIDATE_AI_OUTPUT, Step.BUILD_PREVIEW),
}

#: DAG selected per ``pipeline_version`` (OV-10: new runs are 2.1.0, old objects frozen).
PIPELINE_DAG_BY_VERSION: dict[str, dict[Step, tuple[Step, ...]]] = {
    "1.0.0": PIPELINE_DAG,
    "2.0.0": PIPELINE_DAG,
    "2.1.0": PIPELINE_DAG_V21,
}

#: Steps whose downstream failure must not silently mark the episode ready for review.
AI_STEPS_V21: tuple[Step, ...] = (
    Step.EXTRACT_KEYFRAMES,
    Step.VLM_PREANNOTATE_CLIP,
    Step.MERGE_CANDIDATE_ANNOTATIONS,
    Step.VALIDATE_AI_OUTPUT,
)
