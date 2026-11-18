# Public real-data provenance (AT-24 / DO2-31)

Status: **real public data verified (download + official Reader), importer read-only
path still PARTIAL**. Updated 2026-10-29.

## Verified dataset (executed on this host)

| Field | Value |
|---|---|
| Hub repo id | `lerobot/utokyo_xarm_pick_and_place` |
| Pinned revision (commit SHA) | `da56c23e1b6d32c101c300b85dc54288b53ea63c` |
| Licence | `apache-2.0` (from the dataset card / HfApi `cardData.license`) |
| Download endpoint | `https://hf-mirror.com` (`HF_ENDPOINT` mirror; upstream huggingface.co was not reachable for large files) |
| `codebase_version` | `v3.0` |
| Episodes / frames / fps | 102 / 7490 / 10 |
| Cameras | `observation.images.hand_image`, `observation.images.image`, `observation.images.image2` |
| State / action shape | `observation.state [8]` float32, `action [7]` float32 (3 cameras) |
| Local size | 21 files, 57,157,525 bytes |
| Archive SHA-256 (deterministic tar.gz) | recorded in `artifacts/public-dataset/import-report.json` after the import run |

## Evidence produced

1. **Download** via `huggingface_hub.snapshot_download` with `HF_ENDPOINT=https://hf-mirror.com`
   (`public-data/utokyo_xarm_pick_and_place`, 74.5 s, 10 files) — command is explicit
   opt-in, never run by tests/build.
2. **Official Reader read-back** (`lerobot==0.4.4`, `LeRobotDataset(root=...)`, new
   process): opened successfully, `num_episodes=102`, `num_frames=7490`, fps=10,
   camera keys as above, frame 0 state `(8,)`, action `(7,)`, task text
   "Pick up a white plate, and then place it on the red plate." →
   `PUBLIC_READER_OK` (transcript in `artifacts/public-dataset/`).
3. **Platform import attempt** through the production import service with
   `data_origin=public_dataset`, licence `apache-2.0`, `source_uri`, `source_revision`:
   the V1 importer rejects it with `MAPPING_REQUIRED` because V1 requires an explicit
   `robot_profile=arm6_joint_position_v1` and this real dataset is 8-D state / 7-D
   action with 3 cameras. This is the intended V1 safety behaviour (no silent
   dimension adaptation, no guessing); `robot_profile="diagnostic_only"` is a V2
   feature (SPEC_ADDENDUM_V2 §2.1). The run and reason are recorded in
   `artifacts/public-dataset/import-report.json`.

## Honest status

- **AT-24 (V1)**: real public LeRobot v3 data is downloaded, provenance is pinned and
  the official Reader parsed it; a *read-only inspection/preview demonstration* is
  possible. It is **not** yet imported through the DataOps import pipeline because V1
  only supports the 6-D arm profile. Treat AT-24 as **PARTIAL** until the V2
  diagnostic-only mapping exists (DO2-03/DO2-05) — do not claim a full import.
- **DO2-31 (V2)**: now **not network-blocked**; the remaining work is the V2
  diagnostic-only mapping + 6-D/7-D profile rules. After that, this dataset can be
  imported for read-only preview (never truncated to 6-D and never published as a
  6-D training set).

## Reproducible commands

```bash
# explicit opt-in only; nothing here runs in CI/build
export HF_ENDPOINT=https://hf-mirror.com
uv run python -m scripts.import_public_example \
    --repo lerobot/utokyo_xarm_pick_and_place \
    --revision da56c23e1b6d32c101c300b85dc54288b53ea63c \
    --target public-data --storage s3
```

`scripts/import_public_example.py` records repo/revision/licence/shape/archive SHA and
the import outcome under `artifacts/public-dataset/import-report.json`.

## Why the earlier "blocked" note changed

The prior note said HuggingFace large files were unreachable. With `HF_ENDPOINT` set
to the `hf-mirror.com` mirror, the 57 MB dataset downloaded in 74.5 s, so the network
blocker is resolved for datasets of this size. Larger datasets were not attempted and
must be evaluated case by case; any failure is still reported as `BLOCKED`.
