# Model card — YOLO11n pre-annotation

## What this model is used for

Pre-annotation only: 1 fps object detection over the canonical 640×480 video of
a robot or ego episode, producing candidate boxes that a human reviewer must
approve or reject. It is **not** a policy, does not infer robot actions, and its
output is never used as `action`/`state` ground truth or as a training label
without human review.

## Model identity and evidence

| Item | Value |
|---|---|
| Model | Ultralytics YOLO11n (COCO pretrained) |
| Weights file | `./var/models/yolo11n.pt` (`YOLO_WEIGHTS_PATH`) |
| Weights SHA-256 | recorded in `.env` as `YOLO_WEIGHTS_SHA256`; the detector refuses to run when a configured digest does not match (`MODEL_CHECKSUM_MISMATCH`) |
| Inference parameters | `imgsz=640`, `conf=0.35`, `iou=0.5`, `max_det=100`, `device=cpu` |
| Sampling | canonical 10 Hz video, frames `0,10,20,…` (1 fps) |
| Output contract | `annotation/1.0`, `xyxy` floats in 640×480 pixels, sorted by `(frame_index, class_id, -confidence)`, score rounded to 6 decimals |
| Licence | AGPL-3.0 (Ultralytics); see `docs/licenses.md` |
| FakeDetector | deterministic test double, clearly labelled `"tool": "fake"`; only for contract/scale tests, never reported as real inference |

## Intended use

Assistive visual labelling of tabletop objects in demo videos; UI review of
candidate boxes; measuring the annotation pipeline.

## Out of scope / limitations

* No robot-hand-eye accuracy claim, no cross-camera/cross-temporal tracking.
* COCO classes only; no private fine-tuned weights in v1.
* CPU-only measurement is environment-specific; no throughput SLA is claimed.
* Not validated on real production robot footage; synthetic fixtures are marked
  `data_origin=synthetic`.
* Scoring, boundary and sorting behaviour are unit-tested, but detection quality
  is reported as measured, not as a benchmark result.

## Reproducibility

`DETECTOR=yolo11n` enables real inference. The AT-21 test records the weights
SHA-256, input video SHA-256, parameters and the resulting annotation JSON digest
under `artifacts/` so the run can be independently re-executed. When the model or
network is unavailable the run is reported as BLOCKED — it is never replaced by
FakeDetector evidence.
