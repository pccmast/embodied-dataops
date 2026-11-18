"""Unit tests for the detection contract (SPEC 8.1).

No network and no model download are needed: the YOLO11n path is covered for
weight validation plus an optional real-inference test that skips with an
explicit reason when ``ultralytics``/weights are unavailable.
"""

from __future__ import annotations

import copy
import hashlib
import importlib.util
from pathlib import Path

import numpy as np
import pytest

from embodied_dataops import ANNOTATION_SCHEMA_VERSION, SAMPLE_INTERVAL_NS, TARGET_HEIGHT, TARGET_WIDTH
from embodied_dataops.canonical.contract import canonical_json_bytes
from embodied_dataops.config import Settings
from embodied_dataops.errors import ErrorCode, TerminalTaskError
from embodied_dataops.processing import detection
from embodied_dataops.processing.detection import (
    COCO_CLASS_NAMES,
    FAKE_WEIGHTS_SHA256,
    Detection,
    FakeDetector,
    FrameDetections,
    ModelInfo,
    Yolo11nDetector,
    build_annotation_document,
    build_detector,
    frame_t_ns,
    sampled_frame_indices,
    validate_annotation_document,
)
from embodied_dataops.storage.base import sha256_file

pytestmark = pytest.mark.unit

EPISODE_ID = "018f0000-0000-7000-8000-000000000001"


def _frames(count: int = 6) -> list[tuple[int, np.ndarray]]:
    rng = np.random.default_rng(20261029)
    return [
        (index * 10, rng.integers(0, 255, size=(TARGET_HEIGHT, TARGET_WIDTH, 3), dtype=np.uint8))
        for index in range(count)
    ]


def _fake_document(detector: FakeDetector | None = None) -> tuple[dict, ModelInfo]:
    detector = detector or FakeDetector()
    model_info = detector.prepare()
    frames = detector.detect(_frames())
    document = build_annotation_document(
        episode_id=EPISODE_ID,
        model=model_info,
        frames=frames,
        tool=detector.tool,
        is_synthetic=detector.is_synthetic,
    )
    return document, model_info


def _manual_document() -> dict:
    frames = [
        FrameDetections(
            frame_index=0,
            t_ns=0,
            detections=[
                Detection(class_id=56, class_name=COCO_CLASS_NAMES[56], score=0.9, xyxy=(10, 10, 100, 100)),
                Detection(class_id=41, class_name=COCO_CLASS_NAMES[41], score=0.5, xyxy=(5, 5, 50, 50)),
                Detection(class_id=0, class_name=COCO_CLASS_NAMES[0], score=0.7, xyxy=(1, 2, 30, 40)),
            ],
        )
    ]
    return build_annotation_document(
        episode_id=EPISODE_ID,
        model=ModelInfo(name="yolo11n", weights_sha256="a" * 64),
        frames=frames,
        tool="yolo11n",
        is_synthetic=False,
    )


# ---------------------------------------------------------------------------
# sampling contract
# ---------------------------------------------------------------------------


def test_one_fps_sampling_of_ten_hz_canonical_video() -> None:
    assert sampled_frame_indices(100) == list(range(0, 100, 10))
    assert sampled_frame_indices(0) == []
    assert frame_t_ns(0) == 0
    assert frame_t_ns(1) == SAMPLE_INTERVAL_NS
    assert frame_t_ns(10) == 1_000_000_000  # 1 s


# ---------------------------------------------------------------------------
# deterministic fake detector (test double, never real inference)
# ---------------------------------------------------------------------------


def test_fake_detector_is_deterministic() -> None:
    first, _ = _fake_document(FakeDetector())
    second, _ = _fake_document(FakeDetector())
    assert canonical_json_bytes(first) == canonical_json_bytes(second)


def test_fake_artifact_is_clearly_labelled_as_fake() -> None:
    document, model_info = _fake_document()
    assert document["schema_version"] == ANNOTATION_SCHEMA_VERSION
    assert document["tool"] == "fake"
    assert document["is_synthetic"] is True
    assert document["model"]["name"] == "fake"
    assert document["model"]["weights_sha256"] == FAKE_WEIGHTS_SHA256
    assert model_info.weights_sha256 == FAKE_WEIGHTS_SHA256
    assert document["model"]["conf"] == pytest.approx(0.35)
    assert document["model"]["iou"] == pytest.approx(0.5)


def test_fake_detector_output_satisfies_annotation_schema() -> None:
    document, _ = _fake_document()
    assert validate_annotation_document(document) == []
    assert document["stats"]["frames"] == len(_frames())
    assert document["stats"]["frames"] == len(document["frames"])
    assert document["params"]["sample_fps"] == 1


def test_detections_are_sorted_and_coordinates_are_bounded() -> None:
    document = _manual_document()
    assert validate_annotation_document(document) == []
    detections = document["frames"][0]["detections"]
    assert [item["class_id"] for item in detections] == [0, 41, 56]
    for frame in document["frames"]:
        for item in frame["detections"]:
            x1, y1, x2, y2 = item["xyxy"]
            assert 0.0 <= x1 <= x2 <= TARGET_WIDTH
            assert 0.0 <= y1 <= y2 <= TARGET_HEIGHT
            assert item["score"] == round(item["score"], 6)


def test_low_confidence_candidates_never_become_official() -> None:
    below = Detection(class_id=41, class_name=COCO_CLASS_NAMES[41], score=0.349999, xyxy=(1, 1, 5, 5))
    boundary = Detection(class_id=41, class_name=COCO_CLASS_NAMES[41], score=0.35, xyxy=(2, 2, 6, 6))
    official = detection.filter_official_candidates([below, boundary], 0.35)
    assert [item.score for item in official] == [0.35]
    document = build_annotation_document(
        episode_id=EPISODE_ID,
        model=ModelInfo(name="yolo11n", weights_sha256="b" * 64),
        frames=[FrameDetections(frame_index=0, t_ns=0, detections=[below, boundary])],
        tool="yolo11n",
        is_synthetic=False,
    )
    assert document["stats"]["low_confidence_excluded"] == 1
    assert [item["score"] for item in document["frames"][0]["detections"]] == [0.35]


# ---------------------------------------------------------------------------
# schema validation rejects bad model output (SPEC 12.1)
# ---------------------------------------------------------------------------


def _first_detection(document: dict) -> dict:
    return document["frames"][0]["detections"][0]


def test_validation_rejects_wrong_schema_version() -> None:
    document = _manual_document()
    document["schema_version"] = "annotation/2.0"
    assert any("schema_version" in error for error in validate_annotation_document(document))


def test_validation_rejects_out_of_canvas_coordinates() -> None:
    document = _manual_document()
    _first_detection(document)["xyxy"] = [10.0, 10.0, float(TARGET_WIDTH + 1), 100.0]
    assert any("x-range" in error for error in validate_annotation_document(document))


def test_validation_rejects_non_rounded_scores() -> None:
    document = _manual_document()
    _first_detection(document)["score"] = 0.1234567
    assert any("rounded" in error for error in validate_annotation_document(document))


def test_validation_rejects_unsorted_detections() -> None:
    document = _manual_document()
    detections = document["frames"][0]["detections"]
    detections[0], detections[-1] = detections[-1], detections[0]
    assert any("ordering" in error for error in validate_annotation_document(document))


def test_validation_rejects_bad_model_weights_hash() -> None:
    document = _manual_document()
    document["model"]["weights_sha256"] = "not-a-sha256"
    assert any("weights_sha256" in error for error in validate_annotation_document(document))


def test_validation_accepts_unknown_extra_keys_only_additively() -> None:
    document = _manual_document()
    document["canonical_version"] = 1
    assert validate_annotation_document(document) == []


# ---------------------------------------------------------------------------
# detector selection and YOLO11n weight integrity
# ---------------------------------------------------------------------------


def test_build_detector_uses_settings() -> None:
    fake = build_detector(Settings(detector="fake"))
    assert isinstance(fake, FakeDetector)
    assert fake.tool == "fake"
    yolo = build_detector(
        Settings(detector="yolo11n", yolo_weights_path=Path("var/models/yolo11n.pt"))
    )
    assert isinstance(yolo, Yolo11nDetector)
    assert yolo.tool == "yolo11n"
    assert yolo.is_synthetic is False


def test_yolo_missing_weights_is_terminal_model_unavailable(tmp_path: Path) -> None:
    detector = Yolo11nDetector(tmp_path / "missing-yolo11n.pt")
    with pytest.raises(TerminalTaskError) as excinfo:
        detector.prepare()
    assert excinfo.value.code == ErrorCode.MODEL_UNAVAILABLE


def test_yolo_checksum_mismatch_is_terminal(tmp_path: Path) -> None:
    weights = tmp_path / "yolo11n.pt"
    weights.write_bytes(b"definitely-not-a-real-checkpoint")
    detector = Yolo11nDetector(weights, expected_sha256="0" * 64)
    with pytest.raises(TerminalTaskError) as excinfo:
        detector.prepare()
    assert excinfo.value.code == ErrorCode.MODEL_CHECKSUM_MISMATCH
    assert excinfo.value.details["observed_sha256"] == sha256_file(weights)
    # The declared checksum and the observed hash are both recorded; no inference
    # was attempted or claimed.
    assert detector._model is None


def test_yolo_checksum_match_but_ultralytics_missing(tmp_path: Path) -> None:
    weights = tmp_path / "yolo11n.pt"
    payload = b"dummy-weights-for-checksum-validation"
    weights.write_bytes(payload)
    if importlib.util.find_spec("ultralytics") is not None:
        pytest.skip("ultralytics is installed; real loading is covered by the vision test")
    detector = Yolo11nDetector(weights, expected_sha256=hashlib.sha256(payload).hexdigest())
    with pytest.raises(TerminalTaskError) as excinfo:
        detector.prepare()
    assert excinfo.value.code == ErrorCode.MODEL_UNAVAILABLE
    assert "ultralytics" in excinfo.value.message
    assert detector.weights_sha256 == hashlib.sha256(payload).hexdigest()


@pytest.mark.vision
def test_real_yolo11n_inference_if_available(tmp_path: Path) -> None:
    """Real YOLO11n inference on one synthetic frame; skips with an explicit reason."""
    if importlib.util.find_spec("ultralytics") is None:
        pytest.skip("ultralytics is not installed: no real YOLO11n inference executed")
    settings = Settings()
    weights = Path(settings.yolo_weights_path)
    if not weights.is_file():
        pytest.skip(f"YOLO11n weights not found at {weights}: no real YOLO11n inference executed")
    detector = Yolo11nDetector(
        weights, expected_sha256=settings.yolo_weights_sha256, batch_size=1
    )
    model_info = detector.prepare()
    assert model_info.name == "yolo11n"
    assert model_info.weights_sha256 == sha256_file(weights)
    rgb = np.zeros((TARGET_HEIGHT, TARGET_WIDTH, 3), dtype=np.uint8)
    rgb[100:300, 150:400] = (180, 40, 40)
    results = detector.detect([(0, rgb), (10, rgb)])
    assert [item.frame_index for item in results] == [0, 10]
    document = build_annotation_document(
        episode_id=EPISODE_ID,
        model=model_info,
        frames=results,
        tool=detector.tool,
        is_synthetic=detector.is_synthetic,
    )
    assert validate_annotation_document(document) == []
    assert document["tool"] == "yolo11n"
    assert document["is_synthetic"] is False
    assert document["model"]["weights_sha256"] == sha256_file(weights)
    for frame in document["frames"]:
        for item in frame["detections"]:
            assert 0.0 <= item["score"] <= 1.0
            assert item["score"] >= 0.35


def test_duplicate_frame_indices_are_rejected() -> None:
    frames = [
        FrameDetections(frame_index=0, t_ns=0, detections=[]),
        FrameDetections(frame_index=0, t_ns=0, detections=[]),
    ]
    with pytest.raises(ValueError):
        build_annotation_document(
            episode_id=EPISODE_ID,
            model=ModelInfo(name="fake", weights_sha256=FAKE_WEIGHTS_SHA256),
            frames=frames,
            tool="fake",
            is_synthetic=True,
        )


def test_manual_document_is_immutable_copy_source() -> None:
    # Guards the test helper itself: mutation tests always start from a fresh doc.
    first = _manual_document()
    second = copy.deepcopy(first)
    second["frames"][0]["detections"][0]["xyxy"] = [0, 0, 1, 1]
    assert first["frames"][0]["detections"][0]["xyxy"] != second["frames"][0]["detections"][0]["xyxy"]
