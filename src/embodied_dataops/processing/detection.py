"""Object detection for AI pre-annotation (SPEC 8.1).

Two implementations exist and they are deliberately distinguishable:

``Yolo11nDetector``
    The real pre-annotation path: Ultralytics YOLO11n public weights, CPU
    inference (``imgsz=640, conf=0.35, iou=0.5, max_det=100``) over COCO
    classes.  The weight SHA-256 is verified against
    ``settings.yolo_weights_sha256`` when configured and recorded in the
    annotation artifact.  This platform only demonstrates pre-annotation; it
    never claims robot hand-eye / manipulation accuracy.  Ultralytics code and
    weights are AGPL-3.0 (or commercial) -- see ``docs/licenses.md``.

``FakeDetector``
    A deterministic stand-in used *only* for contract and scale tests, e.g. the
    100-Episode load test where SPEC 12.3 explicitly requires FakeDetector.
    Every artifact it produces is labelled ``tool="fake"`` and
    ``is_synthetic=true``; it must never be presented as real inference and its
    scores/boxes carry no visual meaning.

Sampling contract (SPEC 8.1): the canonical episode is a constant 10 Hz video,
so 1 fps sampling is ``frame_index = 0, 10, 20, ...`` and the sample timestamp
is ``t_ns = frame_index * 100 ms``.  Output coordinates are floats in the final
640x480 pixel space and the JSON is sorted by ``frame_index, class_id,
-confidence`` with scores rounded to 6 decimals.

Low-confidence detections: ``conf=0.35`` is passed to the model, so official
candidates can never contain ``score < 0.35``.  ``filter_official_candidates``
is applied anyway (defence in depth and a single place to change the threshold);
``build_annotation_document`` reports how many candidates were excluded so the
quality policy can emit ``LOW_CONF_DETECTION`` INFO findings.
"""

from __future__ import annotations

import abc
import hashlib
import importlib.util
import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from embodied_dataops import (
    ANNOTATION_SCHEMA_VERSION,
    SAMPLE_INTERVAL_NS,
    TARGET_FPS,
    TARGET_HEIGHT,
    TARGET_WIDTH,
)
from embodied_dataops.config import Settings, get_settings
from embodied_dataops.errors import ErrorCode, TerminalTaskError
from embodied_dataops.storage.base import sha256_file

#: Inference parameters fixed by SPEC 8.1.
DETECTION_CONF = 0.35
DETECTION_IOU = 0.5
DETECTION_MAX_DET = 100
DETECTION_IMGSZ = 640
DETECTION_DEVICE = "cpu"
#: 10 Hz canonical video -> 1 fps sampling.
SAMPLE_STRIDE = TARGET_FPS

#: COCO class names of the public YOLO11n weights (Ultralytics).
COCO_CLASS_NAMES: tuple[str, ...] = (
    "person",
    "bicycle",
    "car",
    "motorcycle",
    "airplane",
    "bus",
    "train",
    "truck",
    "boat",
    "traffic light",
    "fire hydrant",
    "stop sign",
    "parking meter",
    "bench",
    "bird",
    "cat",
    "dog",
    "horse",
    "sheep",
    "cow",
    "elephant",
    "bear",
    "zebra",
    "giraffe",
    "backpack",
    "umbrella",
    "handbag",
    "tie",
    "suitcase",
    "frisbee",
    "skis",
    "snowboard",
    "sports ball",
    "kite",
    "baseball bat",
    "baseball glove",
    "skateboard",
    "surfboard",
    "tennis racket",
    "bottle",
    "wine glass",
    "cup",
    "fork",
    "knife",
    "spoon",
    "bowl",
    "banana",
    "apple",
    "sandwich",
    "orange",
    "broccoli",
    "carrot",
    "hot dog",
    "pizza",
    "donut",
    "cake",
    "chair",
    "couch",
    "potted plant",
    "bed",
    "dining table",
    "toilet",
    "tv",
    "laptop",
    "mouse",
    "remote",
    "keyboard",
    "cell phone",
    "microwave",
    "oven",
    "toaster",
    "sink",
    "refrigerator",
    "book",
    "clock",
    "vase",
    "scissors",
    "teddy bear",
    "hair drier",
    "toothbrush",
)

#: Honest marker for the fake model: it ships no weights, so its "weights
#: sha256" commits to a constant string instead of pretending to hash a model.
FAKE_DETECTOR_MARKER = b"dataops-fake-detector:no-model-weights:v1"
FAKE_WEIGHTS_SHA256 = hashlib.sha256(FAKE_DETECTOR_MARKER).hexdigest()
COCO_CLASS_COUNT = len(COCO_CLASS_NAMES)


def frame_t_ns(frame_index: int) -> int:
    """Episode-relative timestamp of one 10 Hz canonical frame."""
    return int(frame_index) * SAMPLE_INTERVAL_NS


def sampled_frame_indices(frame_count: int, *, stride: int = SAMPLE_STRIDE) -> list[int]:
    """1 fps sample indices (``0, 10, 20, ...``) of a 10 Hz canonical video."""
    if stride < 1:
        raise ValueError("stride must be >= 1")
    return list(range(0, max(0, int(frame_count)), stride))


def round_score(score: float) -> float:
    """SPEC 8.1: confidence rounded to 6 decimals."""
    return round(float(score), 6)


def clamp_xyxy(
    xyxy: Sequence[float], *, width: int = TARGET_WIDTH, height: int = TARGET_HEIGHT
) -> tuple[float, float, float, float]:
    """Clamp a box into the final 640x480 canvas and normalise corner order."""
    x1, y1, x2, y2 = (float(value) for value in xyxy[:4])
    if x1 > x2:
        x1, x2 = x2, x1
    if y1 > y2:
        y1, y2 = y2, y1
    x1 = min(max(x1, 0.0), float(width))
    x2 = min(max(x2, 0.0), float(width))
    y1 = min(max(y1, 0.0), float(height))
    y2 = min(max(y2, 0.0), float(height))
    return (x1, y1, x2, y2)


@dataclass(frozen=True)
class Detection:
    """One detected object in final 640x480 pixel coordinates."""

    class_id: int
    class_name: str
    score: float
    xyxy: tuple[float, float, float, float]

    def to_json(self) -> dict[str, Any]:
        x1, y1, x2, y2 = clamp_xyxy(self.xyxy)
        return {
            "class_id": int(self.class_id),
            "class_name": str(self.class_name),
            "score": round_score(self.score),
            "xyxy": [x1, y1, x2, y2],
        }


@dataclass(frozen=True)
class FrameDetections:
    """Detections produced for one sampled frame."""

    frame_index: int
    t_ns: int
    detections: list[Detection]


@dataclass(frozen=True)
class ModelInfo:
    """Model identity recorded in the annotation artifact (SPEC 8.1)."""

    name: str
    weights_sha256: str
    conf: float = DETECTION_CONF
    iou: float = DETECTION_IOU

    def to_json(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "weights_sha256": self.weights_sha256,
            "conf": round(float(self.conf), 6),
            "iou": round(float(self.iou), 6),
        }


class Detector(abc.ABC):
    """Detector interface used by the DETECT_OBJECTS handler."""

    name: str
    tool: str
    is_synthetic: bool
    conf: float
    iou: float
    weights_sha256: str

    @property
    def model_info(self) -> ModelInfo:
        return ModelInfo(
            name=self.name, weights_sha256=self.weights_sha256, conf=self.conf, iou=self.iou
        )

    @abc.abstractmethod
    def prepare(self) -> ModelInfo:
        """Validate weights/availability and return the recorded model identity."""

    @abc.abstractmethod
    def detect(self, frames: Iterable[tuple[int, np.ndarray]]) -> list[FrameDetections]:
        """Run detection over ``(frame_index, rgb_frame)`` pairs."""


# ---------------------------------------------------------------------------
# deterministic fake detector (contract/scale tests only)
# ---------------------------------------------------------------------------


class FakeDetector(Detector):
    """Deterministic detector for contract and load tests -- NOT real inference.

    Boxes/scores are derived from the SHA-256 of each frame's pixels, so the
    same input always yields byte-identical annotation JSON.  Artifacts are
    stamped ``tool="fake"`` / ``is_synthetic=true`` so downstream code, the API
    and reviewers can never mistake them for model output.
    """

    name = "fake"
    tool = "fake"
    is_synthetic = True
    #: class ids chosen from COCO so the artifact still satisfies its schema
    _CLASS_IDS = (0, 41, 47, 56, 73)

    def __init__(self, *, conf: float = DETECTION_CONF, iou: float = DETECTION_IOU) -> None:
        self.conf = float(conf)
        self.iou = float(iou)
        self.weights_sha256 = FAKE_WEIGHTS_SHA256

    def prepare(self) -> ModelInfo:
        return self.model_info

    def detect(self, frames: Iterable[tuple[int, np.ndarray]]) -> list[FrameDetections]:
        results: list[FrameDetections] = []
        for frame_index, frame in frames:
            array = np.asarray(frame, dtype=np.uint8)
            digest = hashlib.sha256(array.tobytes()).digest()
            height, width = int(array.shape[0]), int(array.shape[1])
            count = digest[0] % 3  # deterministic 0..2 boxes per frame
            detections: list[Detection] = []
            for slot in range(count):
                seed = int.from_bytes(digest[1 + 2 * slot : 3 + 2 * slot], "big")
                class_id = self._CLASS_IDS[seed % len(self._CLASS_IDS)]
                score = self.conf + (digest[8 + slot] / 255.0) * (1.0 - self.conf)
                box_w = 24 + (digest[12 + slot] % max(1, width // 4))
                box_h = 24 + (digest[14 + slot] % max(1, height // 4))
                x1 = (seed * 7 + slot * 13) % max(1, width - 8)
                y1 = (digest[16 + slot] * 5 + slot * 17) % max(1, height - 8)
                x2 = min(width, x1 + box_w)
                y2 = min(height, y1 + box_h)
                detections.append(
                    Detection(
                        class_id=class_id,
                        class_name=COCO_CLASS_NAMES[class_id],
                        score=score,
                        xyxy=(float(x1), float(y1), float(x2), float(y2)),
                    )
                )
            results.append(
                FrameDetections(
                    frame_index=int(frame_index),
                    t_ns=frame_t_ns(int(frame_index)),
                    detections=detections,
                )
            )
        return results


# ---------------------------------------------------------------------------
# real YOLO11n detector
# ---------------------------------------------------------------------------


class Yolo11nDetector(Detector):
    """Ultralytics YOLO11n CPU detector (SPEC 8.1).

    Weights are validated before ``ultralytics`` is imported: a missing file is
    ``MODEL_UNAVAILABLE`` and a SHA-256 mismatch against
    ``settings.yolo_weights_sha256`` is ``MODEL_CHECKSUM_MISMATCH``.
    """

    name = "yolo11n"
    tool = "yolo11n"
    is_synthetic = False

    def __init__(
        self,
        weights_path: str | Path,
        *,
        expected_sha256: str = "",
        conf: float = DETECTION_CONF,
        iou: float = DETECTION_IOU,
        max_det: int = DETECTION_MAX_DET,
        imgsz: int = DETECTION_IMGSZ,
        device: str = DETECTION_DEVICE,
        batch_size: int = 8,
    ) -> None:
        self.weights_path = Path(weights_path)
        self.expected_sha256 = (expected_sha256 or "").strip().lower()
        self.conf = float(conf)
        self.iou = float(iou)
        self.max_det = int(max_det)
        self.imgsz = int(imgsz)
        self.device = device
        self.batch_size = max(1, int(batch_size))
        self.weights_sha256 = ""
        self._model: Any = None
        self._names: dict[int, str] = {}

    # -- weights / model ----------------------------------------------------
    def verify_weights(self) -> str:
        if not self.weights_path.is_file():
            raise TerminalTaskError(
                ErrorCode.MODEL_UNAVAILABLE,
                "YOLO11n weights are not available; real model inference cannot run",
                details={
                    "detector": self.name,
                    "weights_path": str(self.weights_path),
                    "hint": "place yolo11n.pt locally and set YOLO_WEIGHTS_PATH",
                },
            )
        digest = sha256_file(self.weights_path)
        if self.expected_sha256 and digest != self.expected_sha256:
            raise TerminalTaskError(
                ErrorCode.MODEL_CHECKSUM_MISMATCH,
                "YOLO11n weights SHA-256 does not match settings.yolo_weights_sha256",
                details={
                    "expected_sha256": self.expected_sha256,
                    "observed_sha256": digest,
                    "weights_path": str(self.weights_path),
                },
            )
        return digest

    def prepare(self) -> ModelInfo:
        if self._model is not None:
            return self.model_info
        self.weights_sha256 = self.verify_weights()
        if importlib.util.find_spec("ultralytics") is None:
            raise TerminalTaskError(
                ErrorCode.MODEL_UNAVAILABLE,
                "ultralytics is not installed; real YOLO11n inference is unavailable",
                details={
                    "detector": self.name,
                    "weights_sha256": self.weights_sha256,
                    "hint": "install the optional `vision` extra (ultralytics)",
                },
            )
        try:
            from ultralytics import YOLO
        except Exception as exc:  # pragma: no cover - broken installation
            raise TerminalTaskError(
                ErrorCode.MODEL_UNAVAILABLE,
                "ultralytics could not be imported",
                details={"detector": self.name, "error_type": type(exc).__name__},
            ) from exc
        try:
            model = YOLO(str(self.weights_path))
        except Exception as exc:
            raise TerminalTaskError(
                ErrorCode.MODEL_UNAVAILABLE,
                "YOLO11n weights could not be loaded",
                details={
                    "detector": self.name,
                    "weights_sha256": self.weights_sha256,
                    "error_type": type(exc).__name__,
                },
            ) from exc
        raw_names = getattr(model, "names", None)
        if isinstance(raw_names, Mapping):
            self._names = {int(key): str(value) for key, value in raw_names.items()}
        else:  # pragma: no cover - defensive for future ultralytics versions
            self._names = dict(enumerate(COCO_CLASS_NAMES))
        self._model = model
        return self.model_info

    def _class_name(self, class_id: int) -> str:
        name = self._names.get(int(class_id))
        if name is not None:
            return name
        if 0 <= int(class_id) < COCO_CLASS_COUNT:
            return COCO_CLASS_NAMES[int(class_id)]
        return f"class_{int(class_id)}"

    # -- inference ----------------------------------------------------------
    def detect(self, frames: Iterable[tuple[int, np.ndarray]]) -> list[FrameDetections]:
        self.prepare()
        results: list[FrameDetections] = []
        batch: list[tuple[int, np.ndarray]] = []

        def flush() -> None:
            if not batch:
                return
            images = [np.ascontiguousarray(frame) for _, frame in batch]
            predictions = self._predict(images)
            for (frame_index, _), prediction in zip(batch, predictions, strict=True):
                detections: list[Detection] = []
                boxes = getattr(prediction, "boxes", None)
                if boxes is not None and len(boxes) > 0:
                    xyxy = boxes.xyxy.cpu().numpy()
                    scores = boxes.conf.cpu().numpy()
                    classes = boxes.cls.cpu().numpy()
                    for coords, score, class_id in zip(xyxy, scores, classes, strict=True):
                        class_index = int(class_id)
                        detections.append(
                            Detection(
                                class_id=class_index,
                                class_name=self._class_name(class_index),
                                score=float(score),
                                xyxy=tuple(float(value) for value in coords[:4]),  # type: ignore[arg-type]
                            )
                        )
                results.append(
                    FrameDetections(
                        frame_index=int(frame_index),
                        t_ns=frame_t_ns(int(frame_index)),
                        detections=filter_official_candidates(detections, self.conf),
                    )
                )
            batch.clear()

        for item in frames:
            batch.append(item)
            if len(batch) >= self.batch_size:
                flush()
        flush()
        return results

    def _predict(self, images: list[np.ndarray]) -> list[Any]:
        try:
            return list(
                self._model.predict(
                    source=images,
                    imgsz=self.imgsz,
                    conf=self.conf,
                    iou=self.iou,
                    max_det=self.max_det,
                    device=self.device,
                    verbose=False,
                )
            )
        except Exception as exc:
            raise TerminalTaskError(
                ErrorCode.MODEL_UNAVAILABLE,
                "YOLO11n inference failed",
                details={"detector": self.name, "error_type": type(exc).__name__},
            ) from exc


# ---------------------------------------------------------------------------
# detector factory / artifact building / validation
# ---------------------------------------------------------------------------


def build_detector(settings: Settings | None = None) -> Detector:
    """Pick the detector configured by ``settings.detector`` (fake | yolo11n)."""
    settings = settings or get_settings()
    if settings.detector == "yolo11n":
        return Yolo11nDetector(
            settings.yolo_weights_path,
            expected_sha256=settings.yolo_weights_sha256,
            conf=settings.yolo_conf,
            iou=settings.yolo_iou,
            max_det=DETECTION_MAX_DET,
            imgsz=DETECTION_IMGSZ,
            device=settings.yolo_device,
        )
    return FakeDetector(conf=settings.yolo_conf, iou=settings.yolo_iou)


def filter_official_candidates(
    detections: Iterable[Detection], conf: float = DETECTION_CONF
) -> list[Detection]:
    """Keep official candidates only: ``score >= conf`` (SPEC 7.1)."""
    return [detection for detection in detections if float(detection.score) >= float(conf)]


def _stable_detection_key(detection: Detection) -> tuple[float, ...]:
    x1, y1, x2, y2 = clamp_xyxy(detection.xyxy)
    return (
        float(int(detection.class_id)),
        -round_score(detection.score),
        x1,
        y1,
        x2,
        y2,
    )


def build_annotation_document(
    *,
    episode_id: str,
    model: ModelInfo,
    frames: Iterable[FrameDetections],
    tool: str,
    is_synthetic: bool,
    extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build the immutable ``annotation/1.0`` artifact for one Episode.

    Ordering: frames ascending by ``frame_index``; within a frame sorted by
    ``class_id, -confidence`` with coordinates as a deterministic tie-breaker.
    Coordinates are clamped into the final 640x480 canvas, scores rounded to
    6 decimals, and candidates below ``model.conf`` are never emitted.
    """
    ordered_frames = sorted(frames, key=lambda item: int(item.frame_index))
    out_frames: list[dict[str, Any]] = []
    total = 0
    empty_frames = 0
    low_confidence_excluded = 0
    previous_index = -1
    for frame in ordered_frames:
        index = int(frame.frame_index)
        if index <= previous_index:
            raise ValueError(f"duplicate or unordered frame_index {index} in detection output")
        previous_index = index
        official = filter_official_candidates(frame.detections, model.conf)
        low_confidence_excluded += len(frame.detections) - len(official)
        cleaned = [
            Detection(
                class_id=int(detection.class_id),
                class_name=str(detection.class_name),
                score=detection.score,
                xyxy=clamp_xyxy(detection.xyxy),
            )
            for detection in official
        ]
        cleaned.sort(key=_stable_detection_key)
        if not cleaned:
            empty_frames += 1
        total += len(cleaned)
        out_frames.append(
            {
                "frame_index": index,
                "t_ns": frame_t_ns(index),
                "detections": [detection.to_json() for detection in cleaned],
            }
        )
    document: dict[str, Any] = {
        "schema_version": ANNOTATION_SCHEMA_VERSION,
        "episode_id": str(episode_id),
        "model": model.to_json(),
        "tool": str(tool),
        "is_synthetic": bool(is_synthetic),
        "params": {
            "imgsz": DETECTION_IMGSZ,
            "conf": round(float(model.conf), 6),
            "iou": round(float(model.iou), 6),
            "max_det": DETECTION_MAX_DET,
            "device": DETECTION_DEVICE,
            "sample_fps": 1,
            "source_fps": TARGET_FPS,
        },
        "frames": out_frames,
        "stats": {
            "frames": len(out_frames),
            "detections": total,
            "empty_frames": empty_frames,
            "low_confidence_excluded": low_confidence_excluded,
        },
    }
    if extra:
        document.update(dict(extra))
    return document


_SHA256_HEX = set("0123456789abcdef")


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in _SHA256_HEX for character in value.lower())
    )


def _is_finite_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value))


def validate_annotation_document(document: Any) -> list[str]:
    """Validate an ``annotation/1.0`` artifact (schema + coordinate bounds).

    Model output is untrusted input (SPEC 12.1), so this is used both to check
    our own output before it becomes an immutable artifact and by the review
    service when serving stored JSON.  Returns a list of violation strings.
    """
    errors: list[str] = []
    if not isinstance(document, Mapping):
        return ["annotation document must be a JSON object"]
    if document.get("schema_version") != ANNOTATION_SCHEMA_VERSION:
        errors.append(
            f"schema_version must be {ANNOTATION_SCHEMA_VERSION!r}, got {document.get('schema_version')!r}"
        )
    episode_id = document.get("episode_id")
    if not isinstance(episode_id, str) or not episode_id:
        errors.append("episode_id must be a non-empty string")
    model = document.get("model")
    if not isinstance(model, Mapping):
        errors.append("model must be an object")
    else:
        if not isinstance(model.get("name"), str) or not model.get("name"):
            errors.append("model.name must be a non-empty string")
        if not _is_sha256(model.get("weights_sha256")):
            errors.append("model.weights_sha256 must be a 64-character SHA-256 hex string")
        for key in ("conf", "iou"):
            value = model.get(key)
            if not _is_finite_number(value) or not 0.0 < float(value) <= 1.0:
                errors.append(f"model.{key} must be a number in (0, 1]")
    tool = document.get("tool")
    if not isinstance(tool, str) or not tool:
        errors.append("tool must be a non-empty string (e.g. 'yolo11n' or 'fake')")
    frames = document.get("frames")
    if not isinstance(frames, list):
        return errors + ["frames must be a list"]
    previous_index = -1
    for position, frame in enumerate(frames):
        if not isinstance(frame, Mapping):
            errors.append(f"frames[{position}] must be an object")
            continue
        frame_index = frame.get("frame_index")
        if not isinstance(frame_index, int) or isinstance(frame_index, bool) or frame_index < 0:
            errors.append(f"frames[{position}].frame_index must be a non-negative integer")
            continue
        if frame_index <= previous_index:
            errors.append(f"frames[{position}].frame_index is not strictly increasing")
        previous_index = frame_index
        t_ns = frame.get("t_ns")
        if not isinstance(t_ns, int) or isinstance(t_ns, bool) or t_ns < 0:
            errors.append(f"frames[{position}].t_ns must be a non-negative integer")
        elif t_ns != frame_t_ns(frame_index):
            errors.append(
                f"frames[{position}].t_ns={t_ns} does not match the 10 Hz grid ({frame_t_ns(frame_index)})"
            )
        detections = frame.get("detections")
        if not isinstance(detections, list):
            errors.append(f"frames[{position}].detections must be a list")
            continue
        previous_key: tuple[float, ...] | None = None
        for offset, detection in enumerate(detections):
            prefix = f"frames[{position}].detections[{offset}]"
            if not isinstance(detection, Mapping):
                errors.append(f"{prefix} must be an object")
                continue
            class_id = detection.get("class_id")
            if not isinstance(class_id, int) or isinstance(class_id, bool) or not 0 <= class_id < COCO_CLASS_COUNT:
                errors.append(f"{prefix}.class_id must be an integer in [0, {COCO_CLASS_COUNT - 1}]")
            else:
                class_name = detection.get("class_name")
                if class_name != COCO_CLASS_NAMES[class_id]:
                    errors.append(f"{prefix}.class_name does not match COCO class {class_id}")
            score = detection.get("score")
            if not _is_finite_number(score) or not 0.0 <= float(score) <= 1.0:
                errors.append(f"{prefix}.score must be a number in [0, 1]")
            else:
                if abs(float(score) - round_score(float(score))) > 1e-9:
                    errors.append(f"{prefix}.score must be rounded to 6 decimals")
            xyxy = detection.get("xyxy")
            if not isinstance(xyxy, (list, tuple)) or len(xyxy) != 4 or not all(
                _is_finite_number(value) for value in xyxy
            ):
                errors.append(f"{prefix}.xyxy must be four finite floats")
            else:
                x1, y1, x2, y2 = (float(value) for value in xyxy)
                if not (0.0 <= x1 <= x2 <= TARGET_WIDTH):
                    errors.append(f"{prefix}.xyxy x-range must satisfy 0 <= x1 <= x2 <= {TARGET_WIDTH}")
                if not (0.0 <= y1 <= y2 <= TARGET_HEIGHT):
                    errors.append(f"{prefix}.xyxy y-range must satisfy 0 <= y1 <= y2 <= {TARGET_HEIGHT}")
                if _is_finite_number(score):
                    key = (float(int(class_id)) if isinstance(class_id, int) else 0.0, -round_score(float(score)), x1, y1, x2, y2)
                    if previous_key is not None and key < previous_key:
                        errors.append(f"{prefix} violates the (class_id, -confidence) ordering")
                    previous_key = key
            if not isinstance(detection.get("class_name"), str) or not detection.get("class_name"):
                errors.append(f"{prefix}.class_name must be a non-empty string")
    return errors
