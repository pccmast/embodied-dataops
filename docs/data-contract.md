# Data contract

This document freezes the wire and storage contracts implemented by the code.
It mirrors SPEC §4, §5, §7, §10 and is the reference for tests.

## 1. Identifiers and time

| Item | Contract |
|---|---|
| Primary keys | UUIDv7 generated in Python (`embodied_dataops.ids.uuid7`), stored as 36-char strings |
| Database time | `TIMESTAMPTZ` in UTC |
| Series time | `int64` nanoseconds, `t_ns` is episode-relative (`k × 100 000 000` for 10 Hz) |
| Source clock | every stream persists `source_t_ns` + `source_clock_domain`; clocks are never mixed |
| Boundary comparisons | tolerance equality uses `<=` (50 ms, 100 ms); quality thresholds use strict `>` (`>200 ms`, `>30%`) |

## 2. Canonical Episode Contract `canonical-episode/1.0`

Object layout (immutable once produced):

```text
canonical/{episode_id}/v1/
  metadata.json               # required fields below
  samples.parquet             # one row per 10 Hz sample
  frames/front.mp4            # 10 fps, H.264 yuv420p, no audio, 640x480 letterboxed
  frames/thumbnail.jpg        # built by BUILD_PREVIEW
annotations/{episode_id}/{annotation_set_id}/annotation_candidates.json
quality/{episode_id}/{run_id}/quality.json
datasets/{slug}/v{n}/manifest.json + export.tar.gz
```

`metadata.json` required keys: `schema_version`, `episode_id`, `episode_type`,
`data_origin`, `source_type`, `source_asset_sha256`, `source_episode_index`,
`duration_ns`, `target_fps`, `camera_name`, `clock_domain`, `joint_names`,
`state_dim`, `action_dim`, `samples_parquet`, `video_path`, `pipeline_version`.
Additive keys used by this implementation: `task_label`, `robot_profile`,
`source_license`, `mapping`, `normalization_valid`, `sample_count`,
`frame_transforms`, `created_at`.

### robot_episode samples.parquet

| column | type | rule |
|---|---|---|
| `sample_index` | int32 | 0..N-1 contiguous |
| `t_ns` | int64 | exactly `k * 100_000_000` |
| `frame_index` | int32 | equals `sample_index` (10 fps video) |
| `state` | fixed_size_list<float32>[6] | joint positions (rad), order `joint_1..joint_6` |
| `action` | fixed_size_list<float32>[6] | target joint positions (rad), same order |
| `state_source_t_ns` | int64 | source time used for the interpolated state (episode-relative) |
| `action_source_t_ns` | int64 | source time of the selected action; **never > `t_ns`** |

### ego_video_episode samples.parquet

Only `sample_index`, `t_ns`, `frame_index`. `state_dim=0`, `action_dim=0`,
`joint_names=[]`. **No synthetic action/state is ever created** and such an
episode can never be exported as a LeRobot robot episode.

Validation is implemented in `embodied_dataops.canonical.contract.validate_samples_table`
and runs before publication/export.

## 3. Alignment rules

| Source | Rule |
|---|---|
| LeRobot v3 | rows are already aligned; pick the nearest complete row within **≤ 50 ms** and copy state+action together; reject duplicate/non-monotonic timestamps |
| MCAP image | nearest frame within **≤ 50 ms** (ties pick the earlier frame) |
| MCAP joint state | linear interpolation, **both** bracket endpoints within **≤ 50 ms** |
| MCAP action | zero-order hold strictly into the past, age **≤ 100 ms**; never a future action |
| Ego MP4 | source presentation timestamps mapped onto the 10 Hz grid, nearest within **≤ 50 ms** |

Missing points are recorded as `SYNC_*` findings; they are never zero-filled
and never silently dropped. A grid point missing a required modality makes the
robot episode unpublishable (`FAIL`).

## 4. Quality policy `qc-policy/1.0`

| Rule | Severity | Threshold |
|---|---|---|
| `FILE_DECODE_ERROR` | ERROR | ffmpeg full-decode != 0 or 0 frames |
| `DURATION_EXCEEDED` | ERROR | duration > 600 s |
| `TIMESTAMP_NON_MONOTONIC` | ERROR | source timestamps not strictly increasing (duplicates count) |
| `FRAME_GAP_EXCESS` | WARN | source camera gap > 200 ms; ERROR if a 10 Hz grid point has no image |
| `SYNC_MISSING_FRAME` | ERROR | no frame within ≤50 ms of a grid point |
| `SYNC_MISSING_STATE` | ERROR | no bracketing state pair with both ends ≤50 ms |
| `SYNC_STALE_ACTION` | ERROR | no action at t' ≤ t with t − t' ≤ 100 ms |
| `DIMENSION_MISMATCH` | ERROR | state/action != 6D, or joint names missing/duplicated |
| `NONFINITE_VALUE` | ERROR | NaN/Inf in state/action |
| `UNVERIFIED_SEMANTICS` | ERROR | 6-D joint-position semantics not evidenced for a LeRobot robot episode |
| `EMPTY_ANNOTATION` | INFO | model produced zero detections |
| `LOW_CONF_DETECTION` | INFO | detection score < 0.35 (excluded from official candidates) |
| `BLUR_FRAME_RATE` | WARN | >30% of frames have 640×480 grayscale Laplacian variance < 40 (heuristic) |

Verdict: any `ERROR` → `FAIL`; else any `WARN` → `WARN`; else `PASS`; `INFO` never
changes the verdict. Every finding carries `rule_code`, `severity`, `observed`,
`threshold`, `affected_time_ranges`, `evidence_asset_ids`. Reports also include
`valid_sample_count`, `missing_sample_count`, `max_abs_sync_error_ms`, `p95_abs_sync_error_ms`.

## 5. Annotation `annotation/1.0`

Coordinates are floats in final 640×480 pixels, `xyxy ∈ [0,640]×[0,480]`,
sorted by `(frame_index, class_id, -confidence)`, score rounded to 6 decimals,
`conf=0.35`, `iou=0.5`, `max_det=100`, `imgsz=640`, `device=cpu`, sampled at
1 fps (`frame_index = 0, 10, 20, …`). The artifact records
`model.weights_sha256`. `FakeDetector` artifacts are labelled `"tool": "fake"`
and are only valid for contract/scale tests.

Annotation sets are immutable: `PENDING_REVIEW → APPROVED | REJECTED` once, by a
`reviewer`; a rejected set requires a new detection run. Approving a `WARN`
episode requires a review note of at least 10 characters.

## 6. Dataset manifest `dataset-manifest/1.0`

Canonical JSON (UTF-8, sorted keys, no extra whitespace) with:

```json
{
  "schema_version": "dataset-manifest/1.0",
  "dataset_slug": "robot-grab",
  "version": 1,
  "export_format": "lerobot-v3",
  "target_fps": 10,
  "episodes": [{"episode_id": "...", "canonical_sha256": "...", "source_sha256": "...",
                "quality_report_id": "...", "annotation_set_id": "..."}],
  "pipeline_version": "1.0.0",
  "qc_policy_version": "qc-policy/1.0",
  "created_at": "2026-10-29T00:00:00Z"
}
```

Episodes are ordered by `episode_id`. `manifest_sha256` is the SHA-256 of those
canonical bytes and is re-verified again on publish. `BUILDING → READY →
PUBLISHED`; `PUBLISHED` versions are permanently immutable (membership, reviews
and object references cannot change).

Export is only allowed for `robot_episode`s that are 10 Hz/6-D, QC `PASS|WARN`,
with an `APPROVED` annotation set and a known licence. Export goes through the
official LeRobot Writer and is verified by a separate official Reader process
(episode count, frame count, dims, task labels, random samples ≤1e-5).

## 7. Evaluation `evaluation/1.0`

`results[].scenario_key` matches `^[a-z0-9_]{3,64}$`. A gap exists when
`trials >= 20` and `successes/trials < 0.80`; the created `CollectionOrder` has
`requested_count = max(20, 2*(trials-successes))` and is unique per
`(evaluation_report_id, scenario_key)`. Repeated analysis is idempotent.
`is_simulated=true` is surfaced in API/UI output as simulated, never as a real
model-training result.
