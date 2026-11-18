"""AT-21: real YOLO11n inference evidence (SPEC 8.1, SPEC 14 AT-21).

This test runs the *real* ``ultralytics`` YOLO11n CPU detector over a small
synthetic MP4 and writes machine-readable evidence:

* ``artifacts/test-results/at21_yolo_evidence.json`` -- weights SHA-256, input
  video SHA-256, inference parameters, model identity, frame/detection counts;
* ``artifacts/test-results/at21_annotation.json`` -- the generated
  ``annotation/1.0`` JSON with ``tool="yolo11n"`` and ``is_synthetic=false``.

If ``ultralytics`` or the YOLO11n weights are unavailable, the test SKIPS with
an explicit reason.  A skip means AT-21 is BLOCKED in the hand-off report --
``FakeDetector`` output is never substituted or presented as real inference.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
from datetime import UTC, datetime
from pathlib import Path

import cv2
import numpy as np
import pytest

from embodied_dataops import TARGET_HEIGHT, TARGET_WIDTH
from embodied_dataops.canonical.contract import canonical_json_bytes
from embodied_dataops.config import get_settings
from embodied_dataops.processing import detection, preview
from embodied_dataops.storage.base import sha256_file

pytestmark = [pytest.mark.e2e, pytest.mark.vision]

REPO_ROOT = Path(__file__).resolve().parents[2]
EVIDENCE_DIR = REPO_ROOT / "artifacts" / "test-results"


def _resolve_weights() -> Path | None:
    settings = get_settings()
    candidates: list[Path] = []
    if settings.yolo_weights_path:
        candidates.append(Path(settings.yolo_weights_path))
    candidates.extend(
        [REPO_ROOT / "var" / "models" / "yolo11n.pt", REPO_ROOT / "yolo11n.pt"]
    )
    for candidate in candidates:
        if candidate.is_file() and candidate.stat().st_size > 0:
            return candidate
    return None


def _ultralytics_sample_images() -> list[Path]:
    """Ultralytics ships sample photos (bus.jpg / zidane.jpg) with the package."""
    spec = importlib.util.find_spec("ultralytics")
    if spec is None or not spec.origin:
        return []
    assets = Path(spec.origin).parent / "assets"
    return [path for path in (assets / "bus.jpg", assets / "zidane.jpg") if path.is_file()]


def _build_input_video(path: Path, *, frame_count: int = 20) -> tuple[Path, bool]:
    """Assemble a tiny 10 fps 640x480 clip from real sample photos (or drawings)."""
    sample_frames: list[np.ndarray] = []
    for image_path in _ultralytics_sample_images():
        image = cv2.imread(str(image_path))
        if image is None:
            continue
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        sample_frames.append(preview.letterbox_to_canvas(rgb).image)
    used_package_images = bool(sample_frames)
    if not sample_frames:
        canvas = np.zeros((TARGET_HEIGHT, TARGET_WIDTH, 3), dtype=np.uint8)
        cv2.rectangle(canvas, (80, 90), (340, 390), (40, 160, 230), -1)
        cv2.circle(canvas, (520, 240), 90, (230, 200, 40), -1)
        sample_frames = [canvas]

    writer = cv2.VideoWriter(
        str(path), cv2.VideoWriter_fourcc(*"mp4v"), 10.0, (TARGET_WIDTH, TARGET_HEIGHT)
    )
    if not writer.isOpened():
        pytest.skip("OpenCV mp4v encoder is unavailable: cannot build the AT-21 input clip")
    try:
        for index in range(frame_count):
            frame = sample_frames[index % len(sample_frames)].copy()
            cv2.putText(
                frame,
                f"AT-21 sample {index}",
                (16, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 255),
                2,
            )
            writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    finally:
        writer.release()
    return path, used_package_images


def _write_evidence(evidence: dict, document: dict) -> None:
    EVIDENCE_DIR.mkdir(parents=True, exist_ok=True)
    (EVIDENCE_DIR / "at21_yolo_evidence.json").write_text(
        json.dumps(evidence, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    (EVIDENCE_DIR / "at21_annotation.json").write_text(
        json.dumps(document, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def test_at_21_real_yolo11n_inference(tmp_path: Path) -> None:
    if importlib.util.find_spec("ultralytics") is None:
        pytest.skip(
            "AT-21 BLOCKED: ultralytics is not installed, so real YOLO11n inference "
            "was not executed (FakeDetector is not substituted)"
        )
    weights = _resolve_weights()
    if weights is None:
        pytest.skip(
            "AT-21 BLOCKED: YOLO11n weights (yolo11n.pt) are not available offline; "
            "set YOLO_WEIGHTS_PATH to a local weights file"
        )

    settings = get_settings()
    video_path, used_package_images = _build_input_video(tmp_path / "at21_input.mp4")
    input_sha256 = sha256_file(video_path)
    weights_sha256 = sha256_file(weights)

    detector = detection.Yolo11nDetector(
        weights,
        expected_sha256=settings.yolo_weights_sha256,
        conf=detection.DETECTION_CONF,
        iou=detection.DETECTION_IOU,
        max_det=detection.DETECTION_MAX_DET,
        imgsz=detection.DETECTION_IMGSZ,
        device=detection.DETECTION_DEVICE,
        batch_size=4,
    )
    model_info = detector.prepare()
    assert model_info.weights_sha256 == weights_sha256

    sampled = list(
        preview.iter_frames(
            video_path, stride=detection.SAMPLE_STRIDE, require_640x480=True
        )
    )
    assert len(sampled) == 2  # 20 frames at 10 Hz -> 1 fps samples 0 and 10
    frame_detections = detector.detect(
        (frame.frame_index, frame.rgb) for frame in sampled
    )
    document = detection.build_annotation_document(
        episode_id="at-21-demo-episode",
        model=model_info,
        frames=frame_detections,
        tool=detector.tool,
        is_synthetic=detector.is_synthetic,
        extra={
            "at_id": "AT-21",
            "input_video_sha256": input_sha256,
            "input_note": (
                "synthetic clip assembled from Ultralytics package sample photos "
                "(not robot capture)"
                if used_package_images
                else "synthetic drawn frames (not robot capture)"
            ),
        },
    )

    # --- honest assertions: this is real model output, schema-valid ---------
    assert detection.validate_annotation_document(document) == []
    assert document["tool"] == "yolo11n"
    assert document["is_synthetic"] is False
    assert document["model"]["name"] == "yolo11n"
    assert document["model"]["weights_sha256"] == weights_sha256
    assert document["model"]["conf"] == pytest.approx(0.35)
    assert document["model"]["iou"] == pytest.approx(0.5)
    assert document["params"] == {
        "imgsz": 640,
        "conf": 0.35,
        "iou": 0.5,
        "max_det": 100,
        "device": "cpu",
        "sample_fps": 1,
        "source_fps": 10,
    }
    assert document["stats"]["frames"] == len(sampled)
    assert [frame["frame_index"] for frame in document["frames"]] == [0, 10]
    for frame in document["frames"]:
        for item in frame["detections"]:
            x1, y1, x2, y2 = item["xyxy"]
            assert 0.0 <= x1 <= x2 <= TARGET_WIDTH
            assert 0.0 <= y1 <= y2 <= TARGET_HEIGHT
            assert item["score"] >= 0.35
    if used_package_images:
        # bus.jpg / zidane.jpg are real COCO-style photos: the model must find
        # something.  This guards against a silently empty/incorrect inference.
        assert document["stats"]["detections"] > 0

    ultralytics_version = "unknown"
    try:  # pragma: no cover - real environment only
        import ultralytics

        ultralytics_version = getattr(ultralytics, "__version__", "unknown")
    except Exception:
        pass

    evidence = {
        "at_id": "AT-21",
        "status": "PASS",
        "generated_at": datetime.now(UTC).isoformat(),
        "model": {
            "name": model_info.name,
            "weights_path": str(weights),
            "weights_sha256": weights_sha256,
            "conf": model_info.conf,
            "iou": model_info.iou,
            "imgsz": detection.DETECTION_IMGSZ,
            "max_det": detection.DETECTION_MAX_DET,
            "device": detection.DETECTION_DEVICE,
            "provider": "Ultralytics",
            "license_note": "Ultralytics YOLO code/weights are AGPL-3.0 or commercial",
        },
        "input": {
            "video_path": str(video_path),
            "video_sha256": input_sha256,
            "video_bytes": video_path.stat().st_size,
            "width": TARGET_WIDTH,
            "height": TARGET_HEIGHT,
            "fps": 10,
            "source": (
                "ultralytics package sample photos, synthetic 10 fps clip"
                if used_package_images
                else "drawn synthetic frames, synthetic 10 fps clip"
            ),
            "data_origin": "synthetic",
        },
        "sampling": {"stride": detection.SAMPLE_STRIDE, "sampled_frame_indices": [frame.frame_index for frame in sampled]},
        "output": {
            "schema_version": document["schema_version"],
            "tool": document["tool"],
            "is_synthetic": document["is_synthetic"],
            "frames": document["stats"]["frames"],
            "detections": document["stats"]["detections"],
            "empty_frames": document["stats"]["empty_frames"],
            "annotation_sha256": hashlib.sha256(
                canonical_json_bytes(document)
            ).hexdigest(),
        },
        "ultralytics_version": ultralytics_version,
        "annotation_document": document,
    }
    _write_evidence(evidence, document)

    # Evidence files are the machine-readable AT-21 record.
    written = json.loads(
        (EVIDENCE_DIR / "at21_yolo_evidence.json").read_text(encoding="utf-8")
    )
    assert written["model"]["weights_sha256"] == weights_sha256
    assert written["output"]["tool"] == "yolo11n"
    assert written["output"]["is_synthetic"] is False
