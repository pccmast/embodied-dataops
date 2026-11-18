# Third-party licences and model weights

This project is Apache-2.0. Every third-party component keeps its own licence.
This file records what is used, where, and the obligations that follow. It is not
legal advice.

## Runtime services (Docker images, pinned in `infra/versions.lock`)

| Component | Version | Licence | Notes |
|---|---|---|---|
| PostgreSQL | 16.10 | PostgreSQL License | separate process |
| Redis | 7.4.5 | RSALv2 / SSPLv1 (dual, since 7.4) | separate process; no Redis source is redistributed |
| PGSTY Silo (S3-compatible object store) | RELEASE.2026-10-26T00-00-00Z | AGPL-3.0-or-later | drop-in MinIO-compatible community fork used as the object store; separate process, no AGPL code linked into this project. MinIO upstream Docker Hub images are discontinued (ADR-0008) |
| Prometheus | v3.5.0 | Apache-2.0 | |
| Grafana | 12.2.0 | AGPL-3.0 | separate service |
| ROS 2 Jazzy base image | jazzy-ros-base | Apache-2.0 (ROS 2), image includes Ubuntu packages | worker-ros2 only |

## Python dependencies (resolved in `uv.lock`)

| Package | Licence | Role |
|---|---|---|
| FastAPI, Starlette, Pydantic, SQLAlchemy, Alembic, uvicorn | MIT / BSD-3 | API, ORM, migrations |
| psycopg (v3) | LGPL-3.0 (with linking exception) | PostgreSQL driver |
| redis-py | MIT | Streams client |
| boto3 / botocore | Apache-2.0 | S3/MinIO client |
| pyarrow / parquet | Apache-2.0 | columnar samples |
| numpy | BSD-3 | numerics |
| opencv-python-headless | Apache-2.0 (OpenCV), MIT wrapper | video frames |
| Pillow | HPND | thumbnails |
| prometheus-client | Apache-2.0 | metrics |
| streamlit | Apache-2.0 | UI |
| pytest / pytest-asyncio / pytest-timeout | MIT | tests |
| mcap | Apache-2.0 (Foxglove) | MCAP fallback reader and fixtures |
| PyAV (dev/test fallback) | BSD-3 | probing/decoding fallback |
| imageio-ffmpeg (dev/test fallback) | BSD-2 (wrapper); bundled FFmpeg build under its own GPL/LGPL terms | local ffmpeg binary |
| `lerobot==0.4.4` | Apache-2.0 | official LeRobot v3 dataset API |
| torch / torchvision | BSD-3 | required by lerobot / YOLO |
| ultralytics | **AGPL-3.0** (or commercial licence) | YOLO11n inference |

## Model weights

| Model | Source | Licence | SHA-256 |
|---|---|---|---|
| YOLO11n (`yolo11n.pt`) | Ultralytics YOLO11 release (Ultralytics/YOLO11 HF mirror used for the download in this environment) | **AGPL-3.0** (Ultralytics) | recorded at first download in `docs/model-card.md` and `var/models/yolo11n.pt`; `YOLO_WEIGHTS_SHA256` must be set to enable real inference |

## Obligations recorded

1. **AGPL (ultralytics)**: used in-process for pre-annotation. This repository is
   not distributed as a closed-source commercial product. Any commercial
   redistribution must either comply with AGPL-3.0 (source offer for the whole
   combined work) or obtain an Ultralytics commercial licence. The detector is
   optional (`DETECTOR=fake` runs the platform without it).
2. **AGPL (MinIO/Grafana)**: used as separate network services in compose; no
   AGPL code is linked into this application's binaries.
3. **FFmpeg**: the delivery image installs the distribution package; codec
   licensing depends on the build (LGPL-2.1+ with optional GPL parts). This
   project only encodes H.264/AAC-free MP4 and does not redistribute a custom
   FFmpeg build.
4. **Data licences**: imported datasets must carry a licence that allows the
   intended use. Unknown licences block `PUBLISHED` (`LICENSE_UNKNOWN`), and
   `docs/public-dataset-provenance.md` records the licence of any real public
   dataset used for demonstration.

## V2.1 additions (2026-10-30)

| Component | Source | Licence | Notes |
|---|---|---|---|
| Qwen2.5-VL-7B-Instruct (`qwen2.5-vl-7b-instruct`) | Alibaba Cloud Model Studio (DashScope, OpenAI-compatible endpoint) | Model/provider terms of the Model Studio service | Used as a **remote data-production tool**, not as an Agent. The key `DASHSCOPE_API_KEY` lives only in the process environment / secret manager, never in the database, logs or screenshots. No model weights are redistributed with this repository. |
| DashScope HTTP API | `https://dashscope.aliyuncs.com/compatible-mode/v1` | Provider service agreement | Outbound calls are gated by `source_imports.egress_allowed` (default **false**). Only JPEG keyframes are sent; joint values, raw video and logs are never sent. |
| Ultralytics YOLO11n (re-checked for V2.1) | `var/models/yolo11n.pt` | **AGPL-3.0** or commercial | Still the only detector for the V2.1 live path. The real-inference evidence records the exact weights SHA-256 in `reports/v21/yolo-real.json`. |

V2.1 obligations:

1. **No restricted bytes in git.** EPIC-KITCHENS video, AgiBot/LIBERO snapshots and
   model weights are fetched by explicit opt-in scripts; `data/DATA_SOURCES_V21.yaml`
   records the licence claim and `egress_allowed=false` for each.
2. **Provider billing is not exactly-once.** A lost response can be billed while the
   local result is discarded. Reports must state `UNKNOWN_UPSTREAM_CHARGE` instead of
   claiming a precise total (SPEC_V2.1 §5.4).
3. **Replay is not a live result.** `VLM_PROVIDER=replay` runs the same code path with a
   recorded response and is labelled `mode=REPLAY`; it can never satisfy a real-model
   acceptance item.
4. **AgiBot / LIBERO licences** must be re-verified against the pinned commit when the
   snapshots are actually fetched (`revision: PIN_AT_FIRST_FETCH` in the source manifest).
