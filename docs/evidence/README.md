# Shipped evidence

Small, sanitised extracts of measurements taken on real hosts. Absolute paths, hostnames,
instance UUIDs and database connection strings were replaced with placeholders; no measured
value was changed.

| File | What it shows | Machine |
|---|---|---|
| [`acceptance-v21.json`](acceptance-v21.json) | V2.1 acceptance matrix: 43 items with PASS / PARTIAL / BLOCKED / NOT_RUN and the reason for each non-PASS | Linux, Python 3.12 |
| [`yolo-real.json`](yolo-real.json) | Real YOLO11n inference: 10 decoded frames, weights and source SHA-256, p50/p95 latency | NVIDIA A10, CUDA 12.8 |
| [`api-latency.json`](api-latency.json) | `GET /episodes` and pipeline-run creation P95 against 10 000 metadata rows | 32 vCPU, PostgreSQL 16 local |

## Reading rules

* A number here is **what one host measured once**, with the command recorded in
  [`../benchmark.md`](../benchmark.md). It is not a service-level agreement.
* `BLOCKED` / `NOT_RUN` entries are part of the evidence. They are never silently dropped or
  re-labelled, and the corresponding acceptance item stays unmet.
* Synthetic fixtures and the fake detector are labelled as such in the report body. Platform
  throughput measured with `FakeDetector` says nothing about real YOLO inference cost.
* `yolo-real.json` references `reports/v21/vlm-live.json`, which is **not** shipped: the live
  VLM run required a provider key that was not available, so those items are `BLOCKED`.
